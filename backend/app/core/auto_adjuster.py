"""阿斯拉量化系統 — 自動調整引擎

基於預測績效數據，自動計算並存儲調整規則，
以「強制約束」形式注入 LLM prompt，確保 AI 遵循歷史績效教訓。

調整類型：
- indicator_weight: 指標權重（1.5x 加權 / 0.3x 抑制）
- confidence_scale: 信心等級校準縮放
- direction_bias: 多/空方向偏好限制
- risk_multiplier: 倉位風險乘數
"""

import sqlite3
from typing import Optional

from loguru import logger

from app.core.config.settings import settings
from app.core.prediction_tracker import prediction_tracker
from app.utils.timezone import taipei_now


# ─── 閾值常數 ──────────────────────────────

_MIN_INDICATOR_SAMPLES = 5
_BOOST_WIN_RATE = 55       # ≥55% → 加權
_SUPPRESS_WIN_RATE = 40    # <40% → 抑制
_CONFIDENCE_GAP = 15       # 實際與預期偏差 >15pp 時校準
_DIRECTION_MIN_SAMPLES = 8
_DIRECTION_SUPPRESS_RATE = 35  # <35% → 抑制該方向
_STREAK_MILD = 3           # 連敗 ≥3 → 0.5x
_STREAK_SEVERE = 5         # 連敗 ≥5 → 0.25x

_EXPECTED_CONFIDENCE = {"high": 70, "medium": 50, "low": 30}


class AutoAdjuster:
    """自動調整引擎：計算、存儲、生成強制規則。"""

    def __init__(self):
        self._conn: Optional[sqlite3.Connection] = None

    def _ensure_db(self):
        if self._conn:
            return
        db_file = settings.db_path / "predictions.db"
        self._conn = sqlite3.connect(str(db_file), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_adjustments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                adjustment_type TEXT NOT NULL,
                key TEXT NOT NULL,
                value REAL NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL,
                user_override INTEGER DEFAULT 0
            )
        """)
        self._conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_adj_type_key
            ON prediction_adjustments(symbol, adjustment_type, key)
        """)
        self._conn.commit()

    # ─── 計算 ──────────────────────────────

    def compute_adjustments(self, symbol: Optional[str] = None) -> list[dict]:
        """從預測統計數據計算所有調整規則。"""
        stats = prediction_tracker.get_stats(symbol=symbol, days=90)
        if stats["total"] < 3:
            return []

        adjustments: list[dict] = []

        # 1. 指標權重
        ind_perf = stats.get("indicator_performance", {})
        recent_stats = prediction_tracker.get_stats(symbol=symbol, days=30)
        recent_ind = recent_stats.get("indicator_performance", {})

        for ind, data in ind_perf.items():
            if data["samples"] < _MIN_INDICATOR_SAMPLES:
                continue
            wr = data["win_rate"]

            if wr >= _BOOST_WIN_RATE:
                weight = 1.5
                reason = f"勝率{wr}%/{data['samples']}筆 → 優先採用"
            elif wr < _SUPPRESS_WIN_RATE:
                weight = 0.3
                reason = f"勝率{wr}%/{data['samples']}筆 → 禁止作為主要訊號"
            else:
                weight = 1.0
                reason = f"勝率{wr}%/{data['samples']}筆 → 正常使用"

            # Alpha 衰退加罰
            r30 = recent_ind.get(ind)
            if r30 and r30["samples"] >= 2:
                drop = data["win_rate"] - r30["win_rate"]
                if drop >= 15:
                    weight *= 0.5
                    reason += f"（⚠️衰退: 90天{wr}%→30天{r30['win_rate']}%）"

            adjustments.append({
                "symbol": symbol,
                "adjustment_type": "indicator_weight",
                "key": ind,
                "value": round(weight, 2),
                "reason": reason,
            })

        # 2. 信心校準
        conf_cal = stats.get("confidence_calibration", {})
        for level, data in conf_cal.items():
            expected = _EXPECTED_CONFIDENCE.get(level, 50)
            actual = data["win_rate"]
            gap = actual - expected

            if abs(gap) > _CONFIDENCE_GAP:
                scale = round(actual / expected, 2) if expected > 0 else 1.0
                label = {"high": "高", "medium": "中", "low": "低"}.get(level, level)
                if gap < 0:
                    reason = f"{label}信心實際{actual}%（預期{expected}%）→ 下調至{scale}x"
                else:
                    reason = f"{label}信心實際{actual}%（預期{expected}%）→ 上調至{scale}x"
                adjustments.append({
                    "symbol": symbol,
                    "adjustment_type": "confidence_scale",
                    "key": level,
                    "value": scale,
                    "reason": reason,
                })

        # 3. 方向偏好
        dir_stats = prediction_tracker.get_direction_stats(symbol=symbol, days=90)
        for direction, data in dir_stats.items():
            if data["samples"] >= _DIRECTION_MIN_SAMPLES and data["win_rate"] < _DIRECTION_SUPPRESS_RATE:
                label = "做多" if direction == "long" else "做空"
                adjustments.append({
                    "symbol": symbol,
                    "adjustment_type": "direction_bias",
                    "key": direction,
                    "value": -1,
                    "reason": f"{label}勝率{data['win_rate']}%（{data['samples']}筆）→ 抑制",
                })

        # 4. 風險乘數（基於連敗）
        streak = prediction_tracker.get_recent_streak(symbol=symbol, n=10)
        if streak.get("total", 0) > 0 and streak.get("streak_type") == "hit_stop":
            cs = streak["current_streak"]
            if cs >= _STREAK_SEVERE:
                multiplier = 0.25
                reason = f"連續止損{cs}次 → 倉位縮減至 25%"
            elif cs >= _STREAK_MILD:
                multiplier = 0.5
                reason = f"連續止損{cs}次 → 倉位縮減至 50%"
            else:
                multiplier = 1.0
                reason = ""
            if multiplier < 1.0:
                adjustments.append({
                    "symbol": symbol,
                    "adjustment_type": "risk_multiplier",
                    "key": "position_size",
                    "value": multiplier,
                    "reason": reason,
                })

        return adjustments

    # ─── 存儲 ──────────────────────────────

    def store_adjustments(self, adjustments: list[dict]):
        """Upsert 調整規則（跳過 user_override=1 的項目）。"""
        self._ensure_db()
        now = taipei_now().isoformat()

        for adj in adjustments:
            sym = adj["symbol"]
            atype = adj["adjustment_type"]
            key = adj["key"]

            # 檢查是否被使用者手動覆蓋
            existing = self._conn.execute(
                "SELECT user_override FROM prediction_adjustments "
                "WHERE symbol IS ? AND adjustment_type=? AND key=?",
                (sym, atype, key),
            ).fetchone()

            if existing and existing["user_override"] == 1:
                continue

            self._conn.execute(
                """INSERT INTO prediction_adjustments
                   (symbol, adjustment_type, key, value, reason, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(symbol, adjustment_type, key) DO UPDATE SET
                   value=excluded.value, reason=excluded.reason, created_at=excluded.created_at""",
                (sym, atype, key, adj["value"], adj["reason"], now),
            )
        self._conn.commit()

    # ─── 查詢 ──────────────────────────────

    def get_active_adjustments(self, symbol: Optional[str] = None) -> list[dict]:
        """取得有效的調整規則。"""
        self._ensure_db()
        if symbol:
            rows = self._conn.execute(
                "SELECT * FROM prediction_adjustments "
                "WHERE symbol IS NULL OR symbol=? "
                "ORDER BY adjustment_type, key",
                (symbol,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM prediction_adjustments ORDER BY adjustment_type, key",
            ).fetchall()
        return [dict(r) for r in rows]

    def set_override(self, adj_id: int, value: Optional[float] = None) -> dict:
        """使用者手動覆蓋調整值。"""
        self._ensure_db()
        if value is not None:
            self._conn.execute(
                "UPDATE prediction_adjustments SET value=?, user_override=1 WHERE id=?",
                (value, adj_id),
            )
        else:
            self._conn.execute(
                "UPDATE prediction_adjustments SET user_override=1 WHERE id=?",
                (adj_id,),
            )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT * FROM prediction_adjustments WHERE id=?", (adj_id,),
        ).fetchone()
        return dict(row) if row else {}

    # ─── Prompt 生成 ──────────────────────

    def generate_hard_constraints(self, symbol: Optional[str] = None) -> str:
        """生成強制規則 prompt，注入 LLM context。"""
        adjustments = self.get_active_adjustments(symbol)
        if not adjustments:
            return ""

        lines = ["【★★★ 自動調整規則 — 強制執行，不可忽略 ★★★】"]

        # 分類
        ind_weights = [a for a in adjustments if a["adjustment_type"] == "indicator_weight"]
        conf_scales = [a for a in adjustments if a["adjustment_type"] == "confidence_scale"]
        dir_biases = [a for a in adjustments if a["adjustment_type"] == "direction_bias"]
        risk_mults = [a for a in adjustments if a["adjustment_type"] == "risk_multiplier"]

        if ind_weights:
            lines.append("\n【指標權重】")
            for a in ind_weights:
                tag = "⬆" if a["value"] > 1 else ("⬇" if a["value"] < 1 else "─")
                lines.append(f"- {a['key']}: {a['value']}x {tag} {a['reason']}")
            boosted = [a["key"] for a in ind_weights if a["value"] > 1]
            suppressed = [a["key"] for a in ind_weights if a["value"] < 1]
            if boosted:
                lines.append(f"→ 優先使用: {', '.join(boosted)}")
            if suppressed:
                lines.append(f"→ 禁止主導: {', '.join(suppressed)}（不可作為主要進出場依據）")

        if conf_scales:
            lines.append("\n【信心校準】")
            for a in conf_scales:
                lines.append(f"- {a['reason']}")
            # 檢查高信心是否需要額外條件
            high_scale = next((a for a in conf_scales if a["key"] == "high" and a["value"] < 1), None)
            if high_scale:
                lines.append("→ 規則：除非 ≥3 個加權指標同時確認，否則不得標註「高信心」")

        if dir_biases:
            lines.append("\n【方向限制】")
            for a in dir_biases:
                lines.append(f"- {a['reason']}")
                if a["key"] == "short":
                    lines.append("→ 規則：除非有極強反轉證據（≥3 個獨立訊號），否則不建議做空")
                elif a["key"] == "long":
                    lines.append("→ 規則：除非有極強突破證據（≥3 個獨立訊號），否則不建議做多")

        if risk_mults:
            lines.append("\n【風險控制】")
            for a in risk_mults:
                lines.append(f"- {a['reason']}")
            lines.append("→ 所有倉位建議都必須乘以上述倍數")

        return "\n".join(lines)

    # ─── 完整流程 ──────────────────────────

    def run_adjustment_cycle(self, symbol: Optional[str] = None) -> dict:
        """執行完整調整流程：計算 → 存儲 → 返回摘要。"""
        self._ensure_db()
        adjustments = self.compute_adjustments(symbol)

        if adjustments:
            self.store_adjustments(adjustments)
            logger.info(
                f"[自動調整] 計算 {len(adjustments)} 條規則"
                + (f"（{symbol}）" if symbol else "")
            )
        return {
            "adjustments_count": len(adjustments),
            "types": list({a["adjustment_type"] for a in adjustments}),
            "symbol": symbol,
        }


auto_adjuster = AutoAdjuster()


# ─── 模組級便利函式 ──────────────────────

def run_adjustment_cycle(symbol: Optional[str] = None) -> dict:
    return auto_adjuster.run_adjustment_cycle(symbol)


def generate_hard_constraints(symbol: Optional[str] = None) -> str:
    return auto_adjuster.generate_hard_constraints(symbol)


def get_active_adjustments(symbol: Optional[str] = None) -> list[dict]:
    return auto_adjuster.get_active_adjustments(symbol)
