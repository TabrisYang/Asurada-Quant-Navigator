"""阿斯拉量化系統 — 預測追蹤器

從 LLM 回答中提取結構化預測，存入 SQLite，
定期驗證預測結果，並生成績效統計回饋給 LLM。

預測生命週期：active → hit_target / hit_stop / expired / invalidated（v108）
- invalidated: watcher 偵測到失效條件成立（價格穿越邊界），不計入 hit-rate 分母
"""

import json
import math
import re
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from loguru import logger

from app.core.config.settings import settings
from app.utils.timezone import taipei_now


# ═══════════════════════════════════════════════════════
# v108 Phase 3：失效條件文字 → 結構化 JSON parser
# ═══════════════════════════════════════════════════════

# 失效條件常見模式（中文 LLM 輸出為主）：
#   「跌破 $0.2442」「跌破 0.2442」「低於 $X」「< $X」
#   「突破 $0.2552」「高於 $X」「上方 $X」「> $X」
#   「+ 放量 > 1.5×」「+ 成交量 > 2 倍」「(放量)」
#   「FOMC 公布前 12h 未觸邊界」（事件條件，本版不處理）
#
# 設計：
# - 需要有方向關鍵字（突破/跌破/高於/低於/上方/下方/超過/⬆/⬇）或「< $」「> $」
# - 純 `>` 或 `<` 不匹配（會被誤抓量過濾條件 "> 1.5×"）
_INV_PRICE_BELOW = re.compile(
    r"(?:跌破|低於|下方|⬇|<\s*=?\s*\$)\s*\$?(\d+(?:[.,]\d+)?)",
    re.IGNORECASE,
)
_INV_PRICE_ABOVE = re.compile(
    r"(?:突破|高於|上方|超過|⬆|>\s*=?\s*\$)\s*\$?(\d+(?:[.,]\d+)?)",
    re.IGNORECASE,
)
_INV_VOL_MULT = re.compile(
    r"(?:放量|成交量|量\s*>\s*).*?(\d+(?:\.\d+)?)\s*[×x倍]",
    re.IGNORECASE,
)


def parse_invalidation_to_json(text: str) -> Optional[dict]:
    """v108 Phase 3：把 LLM 寫的失效條件文字解析為結構化條件。

    Args:
        text: LLM 輸出的失效條件文字，例如
              「突破 $0.2552（Donchian 上緣）+ 放量 > 1.5× 均量 → 切換趨勢做多；跌破 $0.2442 → 切換趨勢做空」

    Returns:
        dict 或 None。dict 結構：
        {
            "conditions": [
                {"type": "price_above", "threshold": 0.2552, "volume_mult": 1.5},
                {"type": "price_below", "threshold": 0.2442}
            ],
            "raw_text": "...",
            "parser_version": "v108_phase3"
        }
        若沒抓到任何價格條件 → 回 None（caller 應 fallback 到純文字儲存）。

    限制：
        - 僅支援價格條件（突破 / 跌破 / 高於 / 低於）+ 量過濾
        - 事件條件（FOMC 前 12h 等）由 event_calendar_sync 預先處理，不在此 parser 範疇
        - 「方向切換」（→ 切換做多/空）不解析動作層，只記條件
    """
    if not text or not text.strip():
        return None

    text_norm = text.replace(",", "")  # 統一去千位逗號（避免 0.255,2 解析錯）
    conditions: list[dict] = []
    seen_thresholds: set[tuple[str, float]] = set()  # 去重

    # 抽段：用 ；、；、→ 等切（同一條件的量過濾在同段內較易判斷）
    segments = re.split(r"[；;]|→|->", text_norm)

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        # 段內是否有量過濾
        vol_m = _INV_VOL_MULT.search(seg)
        vol_mult = float(vol_m.group(1)) if vol_m else None

        # 取所有價格條件（一個段可能多個，少見但允許）
        for pattern, cond_type in (
            (_INV_PRICE_ABOVE, "price_above"),
            (_INV_PRICE_BELOW, "price_below"),
        ):
            for m in pattern.finditer(seg):
                try:
                    threshold = float(m.group(1))
                except ValueError:
                    continue
                key = (cond_type, threshold)
                if key in seen_thresholds:
                    continue
                seen_thresholds.add(key)
                cond = {"type": cond_type, "threshold": threshold}
                if vol_mult is not None:
                    cond["volume_mult"] = vol_mult
                conditions.append(cond)

    if not conditions:
        return None

    return {
        "conditions": conditions,
        "raw_text": text.strip()[:500],  # 截斷避免 JSON 太長
        "parser_version": "v108_phase3",
    }

_PREDICTIONS_PATTERN = re.compile(
    r"---PREDICTIONS---\s*\n(.*?)(?:\n---END_PREDICTIONS---|$)",
    re.DOTALL,
)

_PREDICTION_LINE = re.compile(
    r"-\s*\[direction:(long|short)\]\s+"
    r"entry=([\d.]+)\s+"
    r"target=([\d.]+)\s+"
    r"stop=([\d.]+)\s+"
    r"timeframe=(\S+)\s+"
    r"confidence=(high|medium|low)\s+"
    r"regime=(\S+)\s+"
    r"indicators=([\w,]+)"
    r"(?:\s+invalidation=(.+))?"
)

# ═══════════════════════════════════════════════════════════
# v100：可見結論卡 regex（使用者看得到的格式 — 取代舊隱藏 block）
# ═══════════════════════════════════════════════════════════
_VISIBLE_CARD_PATTERN = re.compile(
    r"📊\s*本次分析總結.*?(?=═══|\Z)",
    re.DOTALL,
)
# v104 Fix C：偏多/偏空單向計劃（lean_long / lean_short）— 信心 medium、倉位 ×0.7
_LEAN_CARD_PATTERN = re.compile(
    r"(?:🟢\s*偏多單向計劃|🔴\s*偏空單向計劃).*?(?=═══|\Z)",
    re.DOTALL,
)
# 「建議觀望」格式 — 不產生 prediction
_OBSERVE_CARD_PATTERN = re.compile(
    r"⚠️\s*本次分析無法產生具體預測",
)
# v103 1A：雙向計劃格式 — 產生 2 筆 predictions（多空各 1）
_BILATERAL_CARD_PATTERN = re.compile(
    r"🔀\s*雙向計劃.*?(?=═══|\Z)",
    re.DOTALL,
)
# 雙向計劃內的子段（多空兩段）
_BILATERAL_LONG_BLOCK = re.compile(
    r"🟢\s*做多計劃.*?(?=🔴|⏱|═══|$)",
    re.DOTALL,
)
_BILATERAL_SHORT_BLOCK = re.compile(
    r"🔴\s*做空計劃.*?(?=🟢|⏱|═══|$)",
    re.DOTALL,
)
# 雙向子段內的欄位 regex（趨向比較鬆，可能寫成「進場：$X1/$X2 分批」）
_BILAT_ENTRIES = re.compile(r"進場[：:].*?\$?([\d.,]+)(?:[/、]\$?([\d.,]+))?")
_BILAT_SL = re.compile(r"SL[：:]?\s*\$?([\d.,]+)")
_BILAT_TP = re.compile(r"TP[：:]?\s*\$?([\d.,]+)")
_BILAT_RR = re.compile(r"RR[\s：:]*([\d.]+)")

_CARD_DIRECTION = re.compile(r"🎯\s*方向：\s*(做多|做空|偏多|偏空)\s*([\S]+)?")
_CARD_ENTRY = re.compile(r"📍\s*進場：\s*\$?([\d.,]+)")
_CARD_TARGET = re.compile(r"🎯\s*目標：\s*\$?([\d.,]+)")
_CARD_STOP = re.compile(r"🛑\s*止損：\s*\$?([\d.,]+)")
_CARD_TIMEFRAME = re.compile(r"⏱\s*時間框：\s*(\S+)")
_CARD_CONFIDENCE = re.compile(r"📊\s*信心：\s*(高|中|低|high|medium|low)")
_CARD_INDICATORS = re.compile(r"🔍\s*主要指標：\s*([^\n]+)")
_CARD_REGIME = re.compile(r"🌐\s*市場\s*regime：\s*(\S+)")
_CARD_INVALIDATION = re.compile(r"❌\s*失效條件：\s*(.+?)(?=\n|───|$)", re.DOTALL)

_CONFIDENCE_MAP = {"高": "high", "中": "medium", "低": "low"}


def _parse_visible_card(text: str) -> Optional[dict]:
    """v100：從可見結論卡解析 prediction（單筆，因為新版每回應只有 1 張卡）。

    Returns:
        dict（同 _PREDICTION_LINE 解析結果結構），失敗回 None。
    """
    card = _VISIBLE_CARD_PATTERN.search(text)
    if not card:
        return None
    block = card.group(0)

    # 數字含逗號去除
    norm = re.sub(r"(\d),(\d)", r"\1\2", block)

    direction_m = _CARD_DIRECTION.search(norm)
    entry_m = _CARD_ENTRY.search(norm)
    target_m = _CARD_TARGET.search(norm)
    stop_m = _CARD_STOP.search(norm)
    tf_m = _CARD_TIMEFRAME.search(norm)
    conf_m = _CARD_CONFIDENCE.search(norm)
    ind_m = _CARD_INDICATORS.search(norm)
    regime_m = _CARD_REGIME.search(norm)
    inv_m = _CARD_INVALIDATION.search(norm)

    # 必要欄位缺失 → 棄
    if not all([direction_m, entry_m, target_m, stop_m, tf_m, conf_m]):
        return None

    raw_dir = direction_m.group(1)
    direction = "long" if raw_dir in ("做多", "偏多") else "short"
    is_lean = raw_dir in ("偏多", "偏空")  # v104 Fix C：偏多/偏空 = lean 卡
    confidence_raw = conf_m.group(1).lower()
    confidence = _CONFIDENCE_MAP.get(conf_m.group(1), confidence_raw)
    # 偏多/偏空強制 confidence=medium（即使 LLM 寫 high 也降）
    if is_lean:
        confidence = "medium"

    try:
        entry = float(entry_m.group(1))
        target = float(target_m.group(1))
        stop = float(stop_m.group(1))
    except (ValueError, IndexError):
        return None

    tf_str = tf_m.group(1)
    return {
        "direction": direction,
        "entry_price": entry,
        "target_price": target,
        "stop_price": stop,
        "timeframe_str": tf_str,
        "timeframe_hours": _parse_timeframe_hours(tf_str),
        "confidence": confidence,
        "regime": regime_m.group(1) if regime_m else "unknown",
        "indicators": ind_m.group(1).strip().replace(" ", "") if ind_m else "",
        "invalidation": inv_m.group(1).strip() if inv_m else "",
        # v104 Fix C：lean 卡標記（偏多/偏空 = 倉位 ×0.7）
        "is_lean": is_lean,
        "position_size_multiplier": 0.7 if is_lean else 1.0,
    }


def _parse_bilateral_card(text: str) -> list[dict]:
    """v103 1A：從雙向計劃卡解析 2 筆 predictions（多 + 空）。

    Returns:
        list[dict]：[long_pred, short_pred] / 失敗回 []
    """
    card = _BILATERAL_CARD_PATTERN.search(text)
    if not card:
        return []
    block = re.sub(r"(\d),(\d)", r"\1\2", card.group(0))

    # 共用欄位（兩個子預測共用）
    tf_m = _CARD_TIMEFRAME.search(block)
    conf_m = _CARD_CONFIDENCE.search(block)
    ind_m = _CARD_INDICATORS.search(block)
    regime_m = _CARD_REGIME.search(block)
    inv_m = _CARD_INVALIDATION.search(block)

    if not tf_m or not conf_m:
        return []

    tf_str = tf_m.group(1)
    confidence = _CONFIDENCE_MAP.get(conf_m.group(1), conf_m.group(1).lower())
    common = {
        "timeframe_str": tf_str,
        "timeframe_hours": _parse_timeframe_hours(tf_str),
        "confidence": confidence,
        "regime": regime_m.group(1) if regime_m else "ranging",
        "indicators": ind_m.group(1).strip().replace(" ", "") if ind_m else "",
        "invalidation": inv_m.group(1).strip() if inv_m else "",
        "is_bilateral": 1,
    }

    # 共用 pair_id 將 2 筆預測關聯起來
    import time
    pair_id = int(time.time() * 1000)  # ms timestamp 當 pair_id

    results = []

    # 解 long 子段
    long_block = _BILATERAL_LONG_BLOCK.search(block)
    if long_block:
        long_text = long_block.group(0)
        entries_m = _BILAT_ENTRIES.search(long_text)
        sl_m = _BILAT_SL.search(long_text)
        tp_m = _BILAT_TP.search(long_text)
        if entries_m and sl_m and tp_m:
            try:
                entry = float(entries_m.group(1))
                stop = float(sl_m.group(1))
                target = float(tp_m.group(1))
                results.append({
                    **common,
                    "direction": "long",
                    "entry_price": entry,
                    "target_price": target,
                    "stop_price": stop,
                    "bilateral_pair_id": pair_id,
                })
            except (ValueError, IndexError):
                pass

    # 解 short 子段
    short_block = _BILATERAL_SHORT_BLOCK.search(block)
    if short_block:
        short_text = short_block.group(0)
        entries_m = _BILAT_ENTRIES.search(short_text)
        sl_m = _BILAT_SL.search(short_text)
        tp_m = _BILAT_TP.search(short_text)
        if entries_m and sl_m and tp_m:
            try:
                entry = float(entries_m.group(1))
                stop = float(sl_m.group(1))
                target = float(tp_m.group(1))
                results.append({
                    **common,
                    "direction": "short",
                    "entry_price": entry,
                    "target_price": target,
                    "stop_price": stop,
                    "bilateral_pair_id": pair_id,
                })
            except (ValueError, IndexError):
                pass

    return results  # 0 / 1 / 2 筆都可能


def _sanitize_prediction_block(block: str) -> str:
    """預處理預測區塊，提高正則匹配容錯性。"""
    # 移除數字中的逗號：83,500 → 83500
    block = re.sub(r"(\d),(\d)", r"\1\2", block)
    # direction / confidence 統一小寫
    block = re.sub(r"direction:(\w+)", lambda m: f"direction:{m.group(1).lower()}", block)
    block = re.sub(r"confidence=(\w+)", lambda m: f"confidence={m.group(1).lower()}", block)
    # 指標列表去空格：RSI, MACD → RSI,MACD
    block = re.sub(
        r"indicators=([\w,\s]+?)(\s+invalidation=|\s*$)",
        lambda m: f"indicators={m.group(1).replace(' ', '')}{m.group(2)}",
        block,
        flags=re.MULTILINE,
    )
    return block


def _parse_lean_card(text: str) -> Optional[dict]:
    """v104 Fix C：偏多/偏空單向計劃 — 跟 visible card 解析邏輯相同，但前綴是 🟢/🔴。

    技巧：找到 lean card 區塊，把開頭加上「📊 本次分析總結」假標頭，
    然後丟給 _parse_visible_card 解析（重用所有欄位 regex），最後標 is_lean。
    """
    card = _LEAN_CARD_PATTERN.search(text)
    if not card:
        return None
    # 包成 visible card 格式讓既有解析重用
    pseudo = "📊 本次分析總結\n" + card.group(0)
    pred = _parse_visible_card(pseudo)
    if pred and not pred.get("is_lean"):
        # _parse_visible_card 已根據「偏多/偏空」標 is_lean，但保險起見再標一次
        pred["is_lean"] = True
        pred["position_size_multiplier"] = 0.7
        pred["confidence"] = "medium"
    return pred


def _strip_markdown_emphasis(text: str) -> str:
    r"""v104.4：移除 markdown 強調符號（** * __ _），避免 parser regex 因此 fail。

    LLM 常常會把標題寫成 `🔀 **雙向計劃**` 或 `📊 ***本次分析總結***`，
    導致 `🔀\s*雙向計劃` 這種 regex 不 match。
    這個函式只移除強調標記，保留所有實際文字 + emoji + 結構符號。
    """
    if not text:
        return text
    # ** 跟 __ bold（先處理較長的）
    text = re.sub(r"\*\*([^\n*]+?)\*\*", r"\1", text)
    text = re.sub(r"__([^\n_]+?)__", r"\1", text)
    # * 跟 _ italic（保留單獨 * 不誤殺，例如算式 "5 * 3"）
    text = re.sub(r"(?<![\w*])\*([^\n*]+?)\*(?![\w*])", r"\1", text)
    text = re.sub(r"(?<![\w_])_([^\n_]+?)_(?![\w_])", r"\1", text)
    return text


def parse_predictions(llm_response: str) -> list[dict]:
    """從 LLM 回答中解析預測（v104：雙向 → 2 筆 / 單向（含 lean）→ 1 筆 / 觀望 → 0 筆）。

    v104.4：先剝 markdown bold/italic 標記，避免 LLM 寫成 `**雙向計劃**` 時 parser 失敗。
    """
    # 先剝 markdown 標記（純加法、不破壞 emoji 跟結構符號）
    llm_response = _strip_markdown_emphasis(llm_response)

    # 觀望卡：不產生 prediction
    if _OBSERVE_CARD_PATTERN.search(llm_response):
        return []

    # v103 1A：雙向計劃優先（產出 2 筆 predictions）
    bilateral_preds = _parse_bilateral_card(llm_response)
    if bilateral_preds:
        return bilateral_preds

    # v104 Fix C：偏多/偏空單向計劃（lean）— 在標準 visible card 之前嘗試
    lean_pred = _parse_lean_card(llm_response)
    if lean_pred:
        return [lean_pred]

    # 單向結論卡（v100 既有）
    visible_pred = _parse_visible_card(llm_response)
    if visible_pred:
        return [visible_pred]

    # Fallback：舊隱藏 ---PREDICTIONS--- block（向下相容歷史對話）
    match = _PREDICTIONS_PATTERN.search(llm_response)
    if not match:
        return []

    block = _sanitize_prediction_block(match.group(1))
    results = []
    for m in _PREDICTION_LINE.finditer(block):
        try:
            tf_str = m.group(5)
            hours = _parse_timeframe_hours(tf_str)
            results.append({
                "direction": m.group(1),
                "entry_price": float(m.group(2)),
                "target_price": float(m.group(3)),
                "stop_price": float(m.group(4)),
                "timeframe_str": tf_str,
                "timeframe_hours": hours,
                "confidence": m.group(6),
                "regime": m.group(7),
                "indicators": m.group(8),
                "invalidation": m.group(9) or "",
            })
        except (ValueError, IndexError) as e:
            logger.warning(f"解析預測行失敗: {e}")
    return results


def strip_predictions(text: str) -> str:
    """移除 PREDICTIONS 區塊，不顯示給使用者。"""
    return re.sub(
        r"\n?---PREDICTIONS---.*?---END_PREDICTIONS---\n?",
        "",
        text,
        flags=re.DOTALL,
    ).strip()


def _parse_timeframe_hours(tf: str) -> int:
    """將 '48h', '7d' 等轉為小時數。"""
    tf = tf.lower().strip()
    try:
        if tf.endswith("h"):
            return int(tf[:-1])
        elif tf.endswith("d"):
            return int(tf[:-1]) * 24
        elif tf.endswith("w"):
            return int(tf[:-1]) * 168
    except (ValueError, IndexError):
        logger.warning(f"無法解析 timeframe '{tf}'，使用預設 72h")
    return 72  # default 3 days


class PredictionTracker:
    """預測存儲與查詢。"""

    def __init__(self):
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

    def _ensure_db(self):
        if self._conn:
            return
        db_file = settings.db_path / "predictions.db"
        # WAL + busy_timeout：消除多元件並發 predictions.db 時的讀寫互鎖（避免持鎖卡死事件迴圈）
        from app.core.db_utils import open_sqlite
        self._conn = open_sqlite(db_file, check_same_thread=False, row_factory=True)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                target_price REAL NOT NULL,
                stop_price REAL NOT NULL,
                timeframe_hours INTEGER NOT NULL,
                confidence TEXT NOT NULL,
                regime TEXT NOT NULL,
                indicators TEXT NOT NULL,
                source_question TEXT,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                validated_at TEXT,
                actual_outcome_pct REAL,
                max_favorable_pct REAL,
                max_adverse_pct REAL,
                validation_note TEXT
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_pred_symbol_status
            ON predictions(symbol, status)
        """)
        # 遷移：加入 ml_enhanced 欄位（若不存在）
        try:
            self._conn.execute(
                "ALTER TABLE predictions ADD COLUMN ml_enhanced INTEGER DEFAULT 0"
            )
        except sqlite3.OperationalError:
            pass  # 欄位已存在

        # 遷移：加入 notes 欄位（若不存在）
        try:
            self._conn.execute(
                "ALTER TABLE predictions ADD COLUMN notes TEXT DEFAULT ''"
            )
        except sqlite3.OperationalError:
            pass  # 欄位已存在

        # Beta-Binomial 貝氏參數表
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS bayesian_params (
                symbol TEXT NOT NULL,
                alpha REAL NOT NULL DEFAULT 2.0,
                beta REAL NOT NULL DEFAULT 2.0,
                updated_at TEXT,
                PRIMARY KEY (symbol)
            )
        """)

        # 遷移：加入 invalidation 欄位（若不存在）
        try:
            self._conn.execute(
                "ALTER TABLE predictions ADD COLUMN invalidation TEXT DEFAULT ''"
            )
        except sqlite3.OperationalError:
            pass  # 欄位已存在

        # ★ v103 1A：雙向計劃支援欄位（純加法）
        try:
            self._conn.execute(
                "ALTER TABLE predictions ADD COLUMN is_bilateral INTEGER DEFAULT 0"
            )
        except sqlite3.OperationalError:
            pass
        try:
            self._conn.execute(
                "ALTER TABLE predictions ADD COLUMN bilateral_pair_id INTEGER DEFAULT NULL"
            )
        except sqlite3.OperationalError:
            pass

        # 遷移：加入 milestones 欄位（若不存在）
        try:
            self._conn.execute(
                "ALTER TABLE predictions ADD COLUMN milestones TEXT DEFAULT ''"
            )
        except sqlite3.OperationalError:
            pass  # 欄位已存在

        # v105 A1：加入 horizon_class 欄位（short/medium/long 持倉分類，給 ML 訓練分桶用）
        try:
            self._conn.execute(
                "ALTER TABLE predictions ADD COLUMN horizon_class TEXT DEFAULT 'medium'"
            )
        except sqlite3.OperationalError:
            pass

        # v105 C1：加入 regime_std 欄位（標準化的 6 種 regime label）
        try:
            self._conn.execute(
                "ALTER TABLE predictions ADD COLUMN regime_std TEXT DEFAULT NULL"
            )
        except sqlite3.OperationalError:
            pass

        # v105.3 Bug 7：加入 position_size_multiplier 欄位（lean 卡 0.7、其他 1.0）
        try:
            self._conn.execute(
                "ALTER TABLE predictions ADD COLUMN position_size_multiplier REAL DEFAULT 1.0"
            )
        except sqlite3.OperationalError:
            pass

        # v108 Phase 3：失效條件結構化儲存 + watcher 自動標記
        for _col_def in (
            "invalidation_json TEXT DEFAULT NULL",      # 結構化失效條件（JSON）
            "invalidated_at TEXT DEFAULT NULL",          # watcher 標記失效時間
            "invalidation_trigger TEXT DEFAULT NULL",    # 觸發描述（如「價格 > $0.2552」）
        ):
            try:
                self._conn.execute(f"ALTER TABLE predictions ADD COLUMN {_col_def}")
            except sqlite3.OperationalError:
                pass  # 欄位已存在

        # v120：訊號層回測 — 每筆 prediction 記錄當下的衍生品 / 機構訊號值
        # 這些欄位讓我們能算「funding 為負 + 看多」的歷史命中率，配合 v118
        # regime_warning 一起警告 LLM「該訊號組合過去無 alpha」。
        # signals_json / buckets_json 是彈性容器，未來新訊號不用再 migration。
        for _col_def in (
            "funding_at_entry REAL DEFAULT NULL",            # funding rate %
            "oi_24h_change_at_entry REAL DEFAULT NULL",      # OI 24h 變化 %
            "premium_at_entry REAL DEFAULT NULL",            # Coinbase Premium %
            "long_short_at_entry REAL DEFAULT NULL",         # global long/short ratio
            "etf_flow_7d_at_entry REAL DEFAULT NULL",        # ETF 7 天累積流入 USD（v119.5 跳過，欄位先留）
            "ob_imbalance_at_entry REAL DEFAULT NULL",       # 訂單簿 bid/ask depth 比
            "fear_greed_at_entry INTEGER DEFAULT NULL",      # Fear & Greed 0-100
            "signals_json TEXT DEFAULT NULL",                # 完整 raw 訊號值 JSON
            "buckets_json TEXT DEFAULT NULL",                # bucket classification JSON
        ):
            try:
                self._conn.execute(f"ALTER TABLE predictions ADD COLUMN {_col_def}")
            except sqlite3.OperationalError:
                pass  # 欄位已存在

        # 覆盤報告紀錄表
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS review_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                symbol TEXT,
                report TEXT NOT NULL,
                summary_json TEXT,
                is_auto INTEGER DEFAULT 0
            )
        """)
        self._conn.commit()

        # ★ v101 schema：純加法（鐵律：不動既有 predictions 表結構）
        self._ensure_schema_v101()

    def _ensure_schema_v101(self):
        """v101 模仿學習 schema — 純加法保護。

        鐵律：
        - ✅ CREATE TABLE IF NOT EXISTS / ALTER TABLE ADD COLUMN
        - ❌ 嚴禁 DROP / RENAME / MODIFY 既有結構
        - 任何時候關掉 v101 flag，舊 v100 query 仍能正常運作
        """
        if not self._conn:
            return

        # prediction_features：每筆預測當時的 39 特徵快照（給模仿學習訓練用）
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_features (
                prediction_id INTEGER PRIMARY KEY,
                -- 動量超買超賣 (4)
                rsi_14 REAL, stoch_rsi REAL, macd_hist REAL, macd_signal_diff REAL,
                -- 趨勢 (4)
                adx_14 REAL, ema_20_60_diff_pct REAL, supertrend_dir INTEGER, plus_minus_di REAL,
                -- 波動率 (4)
                atr_pct REAL, bb_width REAL, bb_position REAL, hv_30 REAL,
                -- 量能 (3)
                volume_ratio REAL, obv_slope_10 REAL, cvd_change REAL,
                -- 結構 (4)
                smc_bias INTEGER, decomp_trend_slope REAL, decomp_residual_vol REAL, fvg_count INTEGER,
                -- regime one-hot (7)
                regime_trending_up INTEGER, regime_trending_down INTEGER, regime_ranging INTEGER,
                regime_high_vol INTEGER, regime_low_vol INTEGER, regime_unknown INTEGER, regime_confidence REAL,
                -- 跨股廣度 (4)
                breadth_pct REAL, btc_dominance_pct REAL, rs_vs_basket REAL, leadership_score REAL,
                -- 預測 metadata (6)
                direction_long INTEGER, confidence_score INTEGER, timeframe_hours INTEGER,
                target_distance_pct REAL, stop_distance_pct REAL, rr_ratio REAL,
                -- 歷史校準 (3)
                recent_30d_winrate REAL, recent_n INTEGER, calibration_brier REAL,
                -- meta
                snapshot_at TEXT, source TEXT,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id)
            )
        """)

        # v104 Q4：prediction_features 加 8 個 lag column（純加法）
        for col_name in [
            "rsi_14_lag5", "rsi_14_change_5",
            "macd_hist_lag5", "macd_hist_change_5",
            "close_return_5", "close_return_20",
            "volatility_change_30", "trend_persistence",
        ]:
            try:
                self._conn.execute(
                    f"ALTER TABLE prediction_features ADD COLUMN {col_name} REAL"
                )
            except sqlite3.OperationalError:
                pass  # 欄位已存在

        # shadow_predictions：v101 在 SHADOW MODE 期間偷跑的預測（不給使用者看，用來驗證）
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS shadow_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER,
                v100_p_hit REAL,
                v101_p_hit REAL,
                v101_p_ml REAL,
                v101_p_rule REAL,
                v101_blend_alpha REAL,
                v101_mode TEXT,
                model_version TEXT,
                created_at TEXT,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id)
            )
        """)

        # imitation_model_metrics：每次訓練的 metrics + Champion-Challenger 狀態
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS imitation_model_metrics (
                version INTEGER PRIMARY KEY AUTOINCREMENT,
                trained_at TEXT NOT NULL,
                trainset_n INTEGER,
                auc REAL,
                train_auc REAL,
                lockbox_auc REAL,
                brier REAL,
                overfit_gap REAL,
                feature_importance TEXT,
                status TEXT,
                is_champion INTEGER DEFAULT 0,
                is_stable_fallback INTEGER DEFAULT 0,
                quality_gate_passed INTEGER DEFAULT 0,
                quality_gate_failed_reasons TEXT
            )
        """)

        # v103 3A：per-regime 模型支援 — regime 欄位 NULL = all-in-one champion
        try:
            self._conn.execute(
                "ALTER TABLE imitation_model_metrics ADD COLUMN regime TEXT DEFAULT NULL"
            )
        except sqlite3.OperationalError:
            pass

        # v103 4B：SHAP 統計用，每次推論記錄 top features
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS shap_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                logged_at TEXT NOT NULL,
                model_version INTEGER,
                regime TEXT,
                top_features_json TEXT
            )
        """)

        # quality_gate_log：每週評估的紀錄（Phase 0.9 用）
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS quality_gate_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluated_at TEXT NOT NULL,
                ready INTEGER,
                gates_json TEXT,
                metrics_json TEXT,
                action_taken TEXT
            )
        """)
        self._conn.commit()

    @staticmethod
    def classify_horizon(timeframe_hours: int) -> str:
        """v105 A1：依持倉時數分類為 short/medium/long（給 ML 訓練分桶用）。

        - short:  < 24h（15m / 1h / 4h 持倉）
        - medium: 24-168h（4h-1w）
        - long:   > 168h（> 1w）
        """
        if timeframe_hours is None:
            return "medium"  # 未提供當作中等持倉
        try:
            h = int(timeframe_hours)
        except (TypeError, ValueError):
            return "medium"
        if h < 24:
            return "short"
        if h > 168:
            return "long"
        return "medium"

    def _capture_signals(self, pid: int, chart_state: dict) -> None:
        """v120.3：從 chart_state.external_signals capture 當下訊號值 + bucket 分類，
        UPDATE 進剛 INSERT 的 prediction 行。

        所有失敗都 silent（不影響主流程）。
        """
        if not chart_state or not isinstance(chart_state, dict):
            return
        ext = chart_state.get("external_signals") or {}
        derivatives = ext.get("derivatives") or {}
        sentiment = ext.get("sentiment") or {}
        try:
            from app.core.signal_buckets import classify_all_signals
            buckets = classify_all_signals(derivatives, sentiment)
            # Q2：SMC setup 類型（若該次分析有跑 SMC 並產出進場價）併入 buckets，
            # 供 get_single_signal_stats(signal_name="smc_setup", ...) 累積各 setup 歷史命中率
            _smc_setup = chart_state.get("smc_setup")
            if _smc_setup:
                buckets["smc_setup"] = _smc_setup
            # 完整 raw 訊號值存 JSON（彈性容器）
            signals_json = json.dumps(
                {"derivatives": derivatives, "sentiment": sentiment},
                ensure_ascii=False,
            )
            buckets_json = json.dumps(buckets, ensure_ascii=False)
            self._conn.execute(
                """UPDATE predictions
                   SET funding_at_entry = ?,
                       oi_24h_change_at_entry = ?,
                       premium_at_entry = ?,
                       long_short_at_entry = ?,
                       etf_flow_7d_at_entry = ?,
                       ob_imbalance_at_entry = ?,
                       fear_greed_at_entry = ?,
                       signals_json = ?,
                       buckets_json = ?
                   WHERE id = ?""",
                (
                    derivatives.get("funding_rate_pct"),
                    derivatives.get("open_interest_24h_change_pct"),
                    derivatives.get("coinbase_premium_pct"),
                    derivatives.get("global_long_short_ratio"),
                    derivatives.get("etf_flow_7d_usd"),
                    derivatives.get("ob_imbalance_ratio"),
                    sentiment.get("fear_greed_value"),
                    signals_json,
                    buckets_json,
                    pid,
                ),
            )
        except Exception as e:
            logger.debug(f"[v120] _capture_signals 失敗（不影響主流程）pid={pid}: {e}")

    def store(
        self,
        symbol: str,
        timeframe: str,
        prediction: dict,
        source_question: str = "",
        chart_state: Optional[dict] = None,  # v120：給 _capture_signals 用
    ) -> int:
        """存入一筆預測，返回 id。

        v120：可選傳入 chart_state — 會自動 capture external_signals 當下值
        + bucket 分類進 signal_at_entry 欄位（修「看漲說漲」用）。
        """
        with self._lock:
            self._ensure_db()
            now = taipei_now()
            hours = prediction["timeframe_hours"]
            expires = now + timedelta(hours=hours)
            horizon = self.classify_horizon(hours)
            # v105 C1：標準化 regime label（給 per-regime 模型訓練用）
            try:
                from app.core.regime_mapping import standardize_regime
                regime_std = standardize_regime(prediction.get("regime", ""))
            except Exception:
                regime_std = "unknown"

            # 多時間框架矛盾檢測 + 限制
            rows = self._conn.execute(
                "SELECT timeframe, direction FROM predictions WHERE status='active' AND symbol=?",
                (symbol,),
            ).fetchall()
            conflicts = [r for r in rows if r["direction"] != prediction["direction"]]
            same_dir = [r for r in rows if r["direction"] == prediction["direction"]]

            # 限制：同一 symbol 最多 3 筆活躍預測
            if len(rows) >= 3:
                logger.warning(
                    f"預測數量已達上限: {symbol} 已有 {len(rows)} 筆活躍預測，拒絕新增"
                )
                return -1

            # 限制：矛盾方向預測不超過 1 筆（允許一筆對沖，但不允許混亂）
            if len(conflicts) >= 1:
                conflict_info = ", ".join(
                    f"{r['timeframe']} {r['direction']}" for r in conflicts
                )
                logger.warning(
                    f"預測方向衝突過多: 新={prediction['direction']} {timeframe} "
                    f"vs 現有={conflict_info}（{symbol}），已有矛盾預測，拒絕新增"
                )
                return -1

            if conflicts:
                conflict_info = ", ".join(
                    f"{r['timeframe']} {r['direction']}" for r in conflicts
                )
                logger.warning(
                    f"預測方向衝突: 新={prediction['direction']} {timeframe} "
                    f"vs 現有={conflict_info}（{symbol}）"
                )

            # v108 Phase 3：嘗試把 invalidation 文字解析為結構化 JSON（給 watcher 用）
            _inv_text = prediction.get("invalidation", "")
            _inv_json = None
            if _inv_text:
                try:
                    _parsed = parse_invalidation_to_json(_inv_text)
                    if _parsed:
                        _inv_json = json.dumps(_parsed, ensure_ascii=False)
                except Exception as _ip_err:
                    logger.debug(f"invalidation parse 失敗（純文字 fallback）：{_ip_err}")

            cursor = self._conn.execute(
                """INSERT INTO predictions
                   (symbol, timeframe, direction, entry_price, target_price, stop_price,
                    timeframe_hours, confidence, regime, indicators, invalidation,
                    invalidation_json,
                    source_question, created_at, expires_at, status,
                    is_bilateral, bilateral_pair_id, horizon_class, regime_std,
                    position_size_multiplier)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?)""",
                (
                    symbol, timeframe,
                    prediction["direction"],
                    prediction["entry_price"],
                    prediction["target_price"],
                    prediction["stop_price"],
                    hours,
                    prediction["confidence"],
                    prediction["regime"],
                    prediction["indicators"],
                    _inv_text,
                    _inv_json,
                    source_question,
                    now.isoformat(),
                    expires.isoformat(),
                    prediction.get("is_bilateral", 0),
                    prediction.get("bilateral_pair_id"),
                    horizon,
                    regime_std,
                    prediction.get("position_size_multiplier", 1.0),
                ),
            )
            pid = cursor.lastrowid
            # v120.3：capture external_signals 進新欄位（必須在 commit 之前同 transaction 內）
            if chart_state and pid:
                self._capture_signals(pid, chart_state)
            self._conn.commit()
            logger.info(
                f"預測已儲存 #{pid}: {symbol} {prediction['direction']} "
                f"entry={prediction['entry_price']} target={prediction['target_price']} "
                f"stop={prediction['stop_price']} expires={expires.isoformat()}"
            )
            return pid

    def get_active(self, symbol: Optional[str] = None) -> list[dict]:
        """取得尚未驗證的預測。"""
        with self._lock:
            self._ensure_db()
            if symbol:
                rows = self._conn.execute(
                    "SELECT * FROM predictions WHERE status='active' AND symbol=? ORDER BY created_at DESC",
                    (symbol,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM predictions WHERE status='active' ORDER BY created_at DESC",
                ).fetchall()
            return [dict(r) for r in rows]

    def get_validated(
        self,
        symbol: Optional[str] = None,
        regime: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """取得已驗證的預測（支援按 symbol 和 regime 過濾）。"""
        with self._lock:
            self._ensure_db()
            query = "SELECT * FROM predictions WHERE status NOT IN ('active', 'invalidated')"
            params = []
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            if regime:
                query += " AND regime = ?"
                params.append(regime)
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            rows = self._conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def update_validation(
        self,
        pred_id: int,
        status: str,
        actual_outcome_pct: float,
        max_favorable_pct: float,
        max_adverse_pct: float,
        note: str = "",
        hit_at: str | None = None,
    ):
        """更新預測的驗證結果。

        hit_at: 實際觸及目標/止損的 K 線時間。若未提供則用當前時間。
        """
        with self._lock:
            self._ensure_db()
            validated_time = hit_at if hit_at else taipei_now().isoformat()
            # 查詢 symbol 用於貝氏更新
            row = self._conn.execute("SELECT symbol FROM predictions WHERE id=?", (pred_id,)).fetchone()
            pred_symbol = row["symbol"] if row else None

            self._conn.execute(
                """UPDATE predictions
                   SET status=?, validated_at=?, actual_outcome_pct=?,
                       max_favorable_pct=?, max_adverse_pct=?, validation_note=?
                   WHERE id=?""",
                (
                    status,
                    validated_time,
                    actual_outcome_pct,
                    max_favorable_pct,
                    max_adverse_pct,
                    note,
                    pred_id,
                ),
            )
            self._conn.commit()

            # 自動更新 Beta-Binomial 參數
            if pred_symbol and status in ("hit_target", "hit_stop"):
                try:
                    self.update_bayesian(pred_symbol, status == "hit_target")
                except Exception:
                    pass  # 貝氏更新失敗不影響主流程

    def clear_all(self, symbol: Optional[str] = None) -> int:
        """清除所有預測紀錄。若提供 symbol 則只清除該幣對。"""
        with self._lock:
            self._ensure_db()
            if symbol:
                cursor = self._conn.execute(
                    "DELETE FROM predictions WHERE symbol = ?", (symbol,),
                )
            else:
                cursor = self._conn.execute("DELETE FROM predictions")
            self._conn.commit()
            count = cursor.rowcount
            logger.info(f"已清除 {count} 筆預測紀錄" + (f"（{symbol}）" if symbol else ""))
            return count

    def get_stats(
        self,
        symbol: Optional[str] = None,
        regime: Optional[str] = None,
        days: int = 90,
    ) -> dict:
        """計算預測績效統計（帶時間衰減權重）。"""
        with self._lock:
            return self._get_stats_unlocked(symbol, regime, days)

    def _get_stats_unlocked(
        self,
        symbol: Optional[str] = None,
        regime: Optional[str] = None,
        days: int = 90,
    ) -> dict:
        self._ensure_db()
        cutoff = (taipei_now() - timedelta(days=days)).isoformat()
        query = "SELECT * FROM predictions WHERE status NOT IN ('active', 'invalidated') AND created_at > ?"
        params: list = [cutoff]
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if regime:
            query += " AND regime = ?"
            params.append(regime)

        rows = self._conn.execute(query, params).fetchall()
        if not rows:
            return {"total": 0, "message": "尚無已驗證的預測記錄"}

        preds = [dict(r) for r in rows]
        now = taipei_now()

        # 自適應時間衰減：近期連續錯誤時加速衰減（更重視近期表現）
        # 基礎 λ=0.02（半衰期 ~35 天），近 5 筆全錯時 λ=0.05（半衰期 ~14 天）
        recent_preds = sorted(preds, key=lambda p: p["created_at"], reverse=True)[:5]
        recent_wrong = sum(1 for p in recent_preds if p["status"] == "hit_stop")
        if recent_wrong >= 4:
            decay_lambda = 0.05  # 近期表現差 → 加速衰減，更重視最新數據
        elif recent_wrong >= 3:
            decay_lambda = 0.035
        else:
            decay_lambda = 0.02  # 正常半衰期 ~35 天

        hit_target_w = 0.0
        hit_stop_w = 0.0
        expired_w = 0.0
        total_w = 0.0
        outcomes = []
        indicator_wins = {}
        indicator_total = {}
        confidence_hits = {"high": [0, 0], "medium": [0, 0], "low": [0, 0]}

        for p in preds:
            created = datetime.fromisoformat(p["created_at"])
            if created.tzinfo is None:
                created = created.replace(tzinfo=now.tzinfo)
            days_ago = (now - created).days
            weight = math.exp(-decay_lambda * days_ago)

            total_w += weight
            is_win = p["status"] == "hit_target"

            if p["status"] == "hit_target":
                hit_target_w += weight
            elif p["status"] == "hit_stop":
                hit_stop_w += weight
            else:
                expired_w += weight

            if p["actual_outcome_pct"] is not None:
                outcomes.append(p["actual_outcome_pct"])

            # per-indicator stats
            for ind in (p.get("indicators") or "").split(","):
                ind = ind.strip()
                if not ind:
                    continue
                indicator_total[ind] = indicator_total.get(ind, 0) + 1
                if is_win:
                    indicator_wins[ind] = indicator_wins.get(ind, 0) + 1

            conf = p.get("confidence", "medium")
            if conf in confidence_hits:
                confidence_hits[conf][1] += 1
                if is_win:
                    confidence_hits[conf][0] += 1

        win_rate = hit_target_w / total_w if total_w > 0 else 0
        has_outcomes = len(outcomes) > 0
        outcomes_arr = np.array(outcomes) if has_outcomes else np.array([0.0])

        # indicator win rates (≥1 sample, low-sample marked)
        indicator_stats = {}
        for ind, total in indicator_total.items():
            if total >= 1:
                wins = indicator_wins.get(ind, 0)
                indicator_stats[ind] = {
                    "win_rate": round(wins / total * 100, 1),
                    "samples": total,
                }

        # confidence calibration
        conf_calibration = {}
        for conf, (wins, total) in confidence_hits.items():
            if total >= 2:
                conf_calibration[conf] = {
                    "win_rate": round(wins / total * 100, 1),
                    "samples": total,
                }

        # Brier Score + ECE 校準指標
        calibration = _compute_calibration_metrics(preds)

        # Beta-Binomial 貝氏後驗
        bayesian = self._get_bayesian_posterior(symbol)

        return {
            "total": len(preds),
            "win_rate_weighted": round(win_rate * 100, 1),
            "hit_target": sum(1 for p in preds if p["status"] == "hit_target"),
            "hit_stop": sum(1 for p in preds if p["status"] == "hit_stop"),
            "expired": sum(1 for p in preds if p["status"] == "expired"),
            "avg_outcome_pct": round(float(np.mean(outcomes_arr)), 2) if has_outcomes else None,
            "median_outcome_pct": round(float(np.median(outcomes_arr)), 2) if has_outcomes else None,
            "best_outcome_pct": round(float(np.max(outcomes_arr)), 2) if has_outcomes else None,
            "worst_outcome_pct": round(float(np.min(outcomes_arr)), 2) if has_outcomes else None,
            "indicator_performance": indicator_stats,
            "confidence_calibration": conf_calibration,
            "calibration_metrics": calibration,
            "bayesian_posterior": bayesian,
            "decay_lambda": decay_lambda,
            "decay_halflife_days": round(0.693 / decay_lambda, 1),
            "decay_adaptive": decay_lambda != 0.02,
            "sample_sufficient": len(preds) >= 8,
        }

    def get_winrate_with_ci(
        self,
        symbol: Optional[str] = None,
        days: int = 90,
    ) -> dict:
        """v124：回傳含 Wilson CI 的 winrate（給 P1 機率三聯顯示用）。

        - 點估值 win_rate_weighted_pct：時間衰減加權（與系統現有 win_rate 一致）
        - CI：用 raw hit_target / (hit_target + hit_stop) Wilson CI（unweighted）
        - 分母排除 expired：expired = 時間到沒到 TP/SL，不算 hit/miss
        - n vs n_decided：n 為總已驗證樣本（含 expired），n_decided 為 CI 分母
        """
        from app.core.stats_utils import wilson_ci

        stats = self.get_stats(symbol=symbol, days=days)
        if stats.get("total", 0) == 0:
            return {
                "status": "no_history",
                "reason": "n=0",
                "n": 0,
                "n_decided": 0,
            }

        hit_target_raw = int(stats.get("hit_target", 0))
        hit_stop_raw = int(stats.get("hit_stop", 0))
        n_decided = hit_target_raw + hit_stop_raw
        n_total = int(stats.get("total", 0))

        if n_decided == 0:
            return {
                "status": "no_decided",
                "reason": f"n_total={n_total} but n_decided=0 (all expired/active)",
                "n": n_total,
                "n_decided": 0,
                "win_rate_weighted_pct": stats.get("win_rate_weighted"),
            }

        ci_lo, ci_hi = wilson_ci(hit_target_raw, n_decided)
        win_rate_raw_pct = round(hit_target_raw / n_decided * 100, 1)

        return {
            "status": "ok",
            "win_rate_weighted_pct": stats.get("win_rate_weighted"),
            "win_rate_raw_pct": win_rate_raw_pct,
            "n": n_total,
            "n_decided": n_decided,
            "hit_target": hit_target_raw,
            "hit_stop": hit_stop_raw,
            "expired": int(stats.get("expired", 0)),
            "ci_pct": [ci_lo, ci_hi],
            "ci_width_pp": round(ci_hi - ci_lo, 1),
            "decay_halflife_days": stats.get("decay_halflife_days"),
        }


    @staticmethod
    def classify_regime(regime: str) -> str:
        """v118：把自由文字 regime 字串 normalize 成 4 類，供命中率聚合用。

        系統歷史 regime 字串高度發散（94 種），LLM 會給「trending_up」、
        「趨勢上行」、「強趨勢上行+時框衝突」等。直接 group by raw string 樣本
        散得太細，無法建立統計顯著的 per-regime 命中率。改 normalize 成
        BULLISH / BEARISH / RANGE / OTHER 4 類聚合。
        """
        if not regime:
            return "OTHER"
        s = regime.lower()
        # bearish 比 bullish 先判（「趨勢下行」要先 match down 不能 match up）
        if any(k in s for k in ["down", "下行", "空頭", "下降", "偏空", "bearish"]):
            return "BEARISH"
        if any(k in s for k in ["up", "上行", "多頭", "上升", "bullish"]):
            return "BULLISH"
        if any(k in s for k in ["rang", "盤整", "均值", "震盪", "波動壓縮"]):
            return "RANGE"
        return "OTHER"

    def get_regime_class_stats(
        self, symbol: Optional[str] = None, days: int = 30,
    ) -> dict[str, dict]:
        """v118：聚合過去 N 天該 symbol 在 4 個 regime class（BULLISH/BEARISH/RANGE/OTHER）
        的命中率 + 多空分布。

        Returns:
            {
              "BULLISH": {"samples": N, "win_rate": X, "long": L, "short": S, "long_pct": P},
              "BEARISH": {...},
              "RANGE": {...},
              "OTHER": {...},
            }
            （只回傳 samples > 0 的 class）
        """
        with self._lock:
            self._ensure_db()
            cutoff = (taipei_now() - timedelta(days=days)).isoformat()
            query = (
                "SELECT regime, direction, status FROM predictions "
                "WHERE status NOT IN ('active', 'invalidated') AND created_at > ?"
            )
            params: list = [cutoff]
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            rows = self._conn.execute(query, params).fetchall()

            class_buckets: dict[str, dict] = {
                "BULLISH": {"long": 0, "short": 0, "wins": 0, "total": 0},
                "BEARISH": {"long": 0, "short": 0, "wins": 0, "total": 0},
                "RANGE": {"long": 0, "short": 0, "wins": 0, "total": 0},
                "OTHER": {"long": 0, "short": 0, "wins": 0, "total": 0},
            }
            for r in rows:
                cls = self.classify_regime(r["regime"])
                bucket = class_buckets[cls]
                bucket["total"] += 1
                if r["direction"] == "long":
                    bucket["long"] += 1
                elif r["direction"] == "short":
                    bucket["short"] += 1
                if r["status"] == "hit_target":
                    bucket["wins"] += 1

            result: dict[str, dict] = {}
            for cls, d in class_buckets.items():
                if d["total"] == 0:
                    continue
                result[cls] = {
                    "samples": d["total"],
                    "win_rate": round(d["wins"] / d["total"] * 100, 1),
                    "long": d["long"],
                    "short": d["short"],
                    "long_pct": round(d["long"] / d["total"] * 100, 1),
                }
            return result

    def get_single_signal_stats(
        self,
        symbol: Optional[str],
        signal_name: str,
        bucket: str,
        direction: Optional[str] = None,
        days: int = 90,
    ) -> dict:
        """v120.5：查單一訊號 bucket 的歷史命中率（可選帶方向過濾）。

        用 SQLite json_extract 從 buckets_json 抓 bucket 值精確 match。

        Args:
            signal_name: 'funding' / 'fear_greed' / 'premium' / ... 對應 buckets_json key
            bucket: 'POSITIVE' / 'NEGATIVE' / 'EXTREME_FEAR' / ... 對應 bucket value
            direction: 'long' / 'short' / None（None = 不限）

        Returns:
            {"win_rate": float, "samples": int, "wins": int, "losses": int}
        """
        with self._lock:
            self._ensure_db()
            cutoff = (taipei_now() - timedelta(days=days)).isoformat()
            params: list = [signal_name, bucket, cutoff]
            sql = (
                "SELECT status, direction FROM predictions "
                "WHERE buckets_json IS NOT NULL "
                "  AND json_extract(buckets_json, '$.' || ?) = ? "
                "  AND created_at > ? "
                "  AND status NOT IN ('active', 'invalidated')"
            )
            if symbol:
                sql += " AND symbol = ?"
                params.append(symbol)
            if direction:
                sql += " AND direction = ?"
                params.append(direction)

            rows = self._conn.execute(sql, params).fetchall()
            total = len(rows)
            if total == 0:
                return {"win_rate": 0, "samples": 0, "wins": 0, "losses": 0}
            wins = sum(1 for r in rows if r["status"] == "hit_target")
            losses = sum(1 for r in rows if r["status"] == "hit_stop")
            return {
                "win_rate": round(wins / total * 100, 1),
                "samples": total,
                "wins": wins,
                "losses": losses,
            }

    def get_signal_combo_stats(
        self,
        symbol: Optional[str],
        current_buckets: dict[str, str],
        direction: Optional[str] = None,
        days: int = 90,
    ) -> dict:
        """v120.5：查訊號組合（多個 buckets 同時 match）的歷史命中率。

        Args:
            current_buckets: 例 {"funding": "POSITIVE", "fear_greed": "FEAR"}
                忽略 'UNKNOWN' bucket（沒 fetch 到的訊號不參與 match）
            direction: 'long' / 'short' / None（None = 不限）

        Returns:
            {"win_rate": float, "samples": int, "matched_signals": [...]}
            樣本不足（< 5）會 graceful 回 0 命中率 + samples=N
        """
        # 過濾 UNKNOWN
        filtered = {k: v for k, v in current_buckets.items() if v and v != "UNKNOWN"}
        if not filtered:
            return {"win_rate": 0, "samples": 0, "matched_signals": []}

        with self._lock:
            self._ensure_db()
            cutoff = (taipei_now() - timedelta(days=days)).isoformat()
            sql = (
                "SELECT status FROM predictions "
                "WHERE buckets_json IS NOT NULL "
                "  AND created_at > ? "
                "  AND status NOT IN ('active', 'invalidated')"
            )
            params: list = [cutoff]
            for sig_name, sig_bucket in filtered.items():
                sql += " AND json_extract(buckets_json, '$.' || ?) = ?"
                params.extend([sig_name, sig_bucket])
            if symbol:
                sql += " AND symbol = ?"
                params.append(symbol)
            if direction:
                sql += " AND direction = ?"
                params.append(direction)

            rows = self._conn.execute(sql, params).fetchall()
            total = len(rows)
            if total == 0:
                return {"win_rate": 0, "samples": 0, "matched_signals": list(filtered.keys())}
            wins = sum(1 for r in rows if r["status"] == "hit_target")
            return {
                "win_rate": round(wins / total * 100, 1),
                "samples": total,
                "wins": wins,
                "matched_signals": list(filtered.keys()),
            }

    def get_direction_stats(
        self, symbol: Optional[str] = None, days: int = 90,
    ) -> dict[str, dict]:
        """取得多/空方向各自的勝率與樣本數。"""
        with self._lock:
            self._ensure_db()
            cutoff = (taipei_now() - timedelta(days=days)).isoformat()
            query = "SELECT direction, status FROM predictions WHERE status NOT IN ('active', 'invalidated') AND created_at > ?"
            params: list = [cutoff]
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)

            rows = self._conn.execute(query, params).fetchall()
            result: dict[str, dict] = {}
            for d in ("long", "short"):
                dir_rows = [r for r in rows if r["direction"] == d]
                total = len(dir_rows)
                if total == 0:
                    continue
                wins = sum(1 for r in dir_rows if r["status"] == "hit_target")
                result[d] = {
                    "win_rate": round(wins / total * 100, 1),
                    "samples": total,
                }
            return result

    def get_regime_stats(
        self, symbol: Optional[str] = None, days: int = 90,
    ) -> dict:
        """取得各 regime 的預測準確率比較。"""
        with self._lock:
            self._ensure_db()
            cutoff = (taipei_now() - timedelta(days=days)).isoformat()
            query = "SELECT regime, status FROM predictions WHERE status NOT IN ('active', 'invalidated') AND created_at > ?"
            params: list = [cutoff]
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)

            rows = self._conn.execute(query, params).fetchall()
            if not rows:
                return {"total": 0, "regimes": {}}

            regime_data: dict[str, dict] = {}
            for r in rows:
                regime = r["regime"] or "unknown"
                if regime not in regime_data:
                    regime_data[regime] = {"wins": 0, "losses": 0, "expired": 0, "total": 0}
                regime_data[regime]["total"] += 1
                if r["status"] == "hit_target":
                    regime_data[regime]["wins"] += 1
                elif r["status"] == "hit_stop":
                    regime_data[regime]["losses"] += 1
                else:
                    regime_data[regime]["expired"] += 1

            result = {}
            for regime, data in regime_data.items():
                decided = data["wins"] + data["losses"]
                win_rate = round(data["wins"] / decided * 100, 1) if decided > 0 else 0
                result[regime] = {
                    "win_rate": win_rate,
                    "wins": data["wins"],
                    "losses": data["losses"],
                    "expired": data["expired"],
                    "total": data["total"],
                    "reliable": decided >= 5,
                }

            # 找出最強和最弱的 regime
            reliable = {k: v for k, v in result.items() if v["reliable"]}
            best = max(reliable, key=lambda k: reliable[k]["win_rate"]) if reliable else None
            worst = min(reliable, key=lambda k: reliable[k]["win_rate"]) if reliable else None

            return {
                "total": len(rows),
                "regimes": result,
                "best_regime": best,
                "worst_regime": worst,
                "recommendation": (
                    f"模型在 {best} 市場表現最好（{result[best]['win_rate']}%），"
                    f"在 {worst} 市場表現最差（{result[worst]['win_rate']}%）"
                    if best and worst and best != worst
                    else "樣本不足，無法區分 regime 表現差異"
                ),
            }

    def get_recent_streak(
        self, symbol: Optional[str] = None, n: int = 10,
    ) -> dict:
        """取得最近 N 筆預測的連勝/連敗資訊（expired 不算勝敗，跳過）。"""
        with self._lock:
            self._ensure_db()
            query = (
                "SELECT status FROM predictions "
                "WHERE status IN ('hit_target', 'hit_stop')"
            )
            params: list = []
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(n)

            rows = self._conn.execute(query, params).fetchall()
            if not rows:
                return {"total": 0, "current_streak": 0, "streak_type": None}

            statuses = [r["status"] for r in rows]
            recent_wins = sum(1 for s in statuses if s == "hit_target")
            recent_losses = sum(1 for s in statuses if s == "hit_stop")

            current_streak = 0
            streak_type = statuses[0] if statuses else None
            for s in statuses:
                if s == streak_type:
                    current_streak += 1
                else:
                    break

            return {
                "total": len(statuses),
                "recent_wins": recent_wins,
                "recent_losses": recent_losses,
                "current_streak": current_streak,
                "streak_type": streak_type,
            }

    # ─── 策略日誌 / 覆盤 ──────────────────

    def update_note(self, pred_id: int, note: str):
        """更新預測的使用者筆記"""
        with self._lock:
            self._ensure_db()
            self._conn.execute(
                "UPDATE predictions SET notes = ? WHERE id = ?",
                (note, pred_id),
            )
            self._conn.commit()

    def get_review_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> dict:
        """取得覆盤所需的資料（某時間範圍內的所有已驗證預測）"""
        with self._lock:
            self._ensure_db()
            query = "SELECT * FROM predictions WHERE status NOT IN ('active', 'invalidated')"
            params: list = []
            if start_date:
                query += " AND created_at >= ?"
                params.append(start_date)
            if end_date:
                query += " AND created_at <= ?"
                params.append(end_date)
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            query += " ORDER BY created_at DESC"
            rows = self._conn.execute(query, params).fetchall()
            predictions = [dict(r) for r in rows]

            total = len(predictions)
            wins = sum(1 for p in predictions if p["status"] == "hit_target")
            losses = sum(1 for p in predictions if p["status"] == "hit_stop")

            return {
                "predictions": predictions,
                "summary": {
                    "total": total,
                    "wins": wins,
                    "losses": losses,
                    "expired": total - wins - losses,
                    "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
                },
            }

    # ─── 覆盤報告紀錄 ──────────────────

    def save_review(
        self,
        report: str,
        symbol: Optional[str] = None,
        summary: Optional[dict] = None,
        is_auto: bool = False,
    ) -> int:
        """儲存覆盤報告，返回 id。"""
        import json as _json
        with self._lock:
            self._ensure_db()
            now = taipei_now().isoformat()
            cursor = self._conn.execute(
                "INSERT INTO review_log (created_at, symbol, report, summary_json, is_auto) VALUES (?, ?, ?, ?, ?)",
                (now, symbol or "", report, _json.dumps(summary or {}, ensure_ascii=False), 1 if is_auto else 0),
            )
            self._conn.commit()
            rid = cursor.lastrowid
            logger.info(f"覆盤報告已儲存 #{rid} (auto={is_auto})")
            return rid

    def get_last_review_time(self) -> Optional[datetime]:
        """取得最近一次自動覆盤的時間。"""
        with self._lock:
            self._ensure_db()
            row = self._conn.execute(
                "SELECT created_at FROM review_log WHERE is_auto=1 ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if not row:
                return None
            dt = datetime.fromisoformat(row["created_at"])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=taipei_now().tzinfo)
            return dt

    def get_reviews(self, limit: int = 10) -> list[dict]:
        """取得歷史覆盤報告列表。"""
        with self._lock:
            self._ensure_db()
            rows = self._conn.execute(
                "SELECT id, created_at, symbol, report, summary_json, is_auto FROM review_log ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]


    # ═══════════════════════════════════════════════════
    #  Beta-Binomial 貝氏更新
    # ═══════════════════════════════════════════════════

    def _get_bayesian_posterior(self, symbol: Optional[str] = None) -> dict:
        """取得 Beta-Binomial 後驗分佈參數。"""
        key = symbol or "__global__"
        try:
            row = self._conn.execute(
                "SELECT alpha, beta FROM bayesian_params WHERE symbol=?", (key,)
            ).fetchone()
            if row:
                a, b = row["alpha"], row["beta"]
            else:
                a, b = 2.0, 2.0  # 弱先驗
        except Exception:
            a, b = 2.0, 2.0

        mean = a / (a + b)
        # 95% credible interval（Beta 分佈的近似）
        from scipy.stats import beta as beta_dist
        try:
            ci_low, ci_high = beta_dist.ppf([0.025, 0.975], a, b)
        except Exception:
            ci_low, ci_high = 0.0, 1.0

        return {
            "alpha": round(a, 2),
            "beta": round(b, 2),
            "posterior_mean": round(mean * 100, 1),
            "credible_interval_95": [round(ci_low * 100, 1), round(ci_high * 100, 1)],
            "total_observations": int(a + b - 4),  # 減去先驗的 alpha=2, beta=2
        }

    def update_bayesian(self, symbol: str, is_win: bool):
        """每次預測驗證後更新 Beta 參數。"""
        with self._lock:
            self._ensure_db()
            key = symbol or "__global__"
            row = self._conn.execute(
                "SELECT alpha, beta FROM bayesian_params WHERE symbol=?", (key,)
            ).fetchone()

            if row:
                a, b = row["alpha"], row["beta"]
            else:
                a, b = 2.0, 2.0

            if is_win:
                a += 1
            else:
                b += 1

            self._conn.execute(
                "INSERT OR REPLACE INTO bayesian_params (symbol, alpha, beta, updated_at) VALUES (?, ?, ?, ?)",
                (key, a, b, taipei_now().isoformat()),
            )
            # 同時更新全域
            if key != "__global__":
                self.update_bayesian("__global__", is_win)

            self._conn.commit()


# ═══════════════════════════════════════════════════
#  校準指標函式（模組層級）
# ═══════════════════════════════════════════════════

def _compute_calibration_metrics(preds: list[dict]) -> dict:
    """計算 Brier Score、ECE、Reliability Curve。"""
    if not preds:
        return {"brier_score": None, "ece": None, "reliability_curve": []}

    # 信心度映射：high=0.8, medium=0.5, low=0.3
    conf_to_prob = {"high": 0.8, "medium": 0.5, "low": 0.3}
    predicted_probs = []
    actual_outcomes = []

    for p in preds:
        conf = p.get("confidence", "medium")
        prob = conf_to_prob.get(conf, 0.5)
        predicted_probs.append(prob)
        actual_outcomes.append(1.0 if p["status"] == "hit_target" else 0.0)

    predicted = np.array(predicted_probs)
    actual = np.array(actual_outcomes)

    # Brier Score: mean((predicted - actual)^2)，越低越好，隨機 = 0.25
    brier = float(np.mean((predicted - actual) ** 2))

    # ECE: Expected Calibration Error
    bins = [0.0, 0.35, 0.55, 0.75, 1.01]
    reliability = []
    ece = 0.0
    for i in range(len(bins) - 1):
        mask = (predicted >= bins[i]) & (predicted < bins[i + 1])
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        avg_pred = float(predicted[mask].mean())
        avg_actual = float(actual[mask].mean())
        ece += abs(avg_actual - avg_pred) * (n_bin / len(preds))
        reliability.append({
            "bin": f"{bins[i]:.0%}-{bins[i+1]:.0%}",
            "avg_predicted": round(avg_pred * 100, 1),
            "avg_actual": round(avg_actual * 100, 1),
            "count": int(n_bin),
            "gap": round((avg_actual - avg_pred) * 100, 1),
        })

    return {
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "reliability_curve": reliability,
        "interpretation": (
            "校準良好" if ece < 0.05
            else "校準中等" if ece < 0.10
            else "校準偏差大，預測機率與實際不符"
        ),
    }


prediction_tracker = PredictionTracker()
