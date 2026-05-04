"""阿斯拉量化系統 — 歷史分析結果快取（含數據指紋）

快取 LLM 的分析結果，避免重複計算。
核心設計：
- 每筆快取都綁定「數據指紋」（symbol + timeframe + 數據範圍 hash）
- 數據更新時，指紋改變 → 快取自動失效
- 歷史固定區間（結束日 < 今天 - 7 天）→ 快取長期有效
- 近期數據（包含最近 7 天）→ 快取 6 小時後失效
- 快取回應明確標註來源和分析時間
"""

import hashlib
import shutil
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Optional

from loguru import logger

from app.core.config.settings import settings
from app.core.symbol_extractor import should_bypass_cache

# 近期數據快取有效期（秒）
_RECENT_TTL = 1 * 3600  # 1 小時（投資分析時效性高）

# 歷史數據快取有效期（秒）
_HISTORY_TTL = 7 * 24 * 3600  # 7 天（從 30 天縮短）

# 判斷「近期」的天數閾值
_RECENT_DAYS = 7


def _compute_query_hash(question: str, symbol: str, timeframe: str) -> str:
    """計算問題的語意 hash（用於快取 key）"""
    # 移除空格、標點，統一小寫，取核心語意
    normalized = question.strip().lower()
    raw = f"{normalized}|{symbol}|{timeframe}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _compute_data_fingerprint(
    symbol: str, timeframe: str, data_points: int, last_timestamp: str,
    last_close: float = 0.0,
    regime: str = "",
    confidence_bucket: int = 0,
) -> str:
    """計算數據指紋：數據、價格、regime 或信心改變時指紋也變。

    v106 D2 強化：把 regime 和 confidence bucket 納入指紋
    → 同 symbol 但 regime 從 trending_up 變 ranging 也會 invalidate cache。
    """
    close_bucket = round(last_close, 2) if last_close else 0
    raw = f"{symbol}|{timeframe}|{data_points}|{last_timestamp}|{close_bucket}|{regime}|{confidence_bucket}"
    return hashlib.md5(raw.encode()).hexdigest()[:10]


def _extract_regime_signature(chart_state: Optional[dict]) -> tuple[str, int]:
    """從 chart_state 取出 (regime, confidence_bucket) — 給指紋用。

    confidence 用 0.1 寬度 bucket，避免微小波動導致 cache 失效。
    """
    if not chart_state:
        return "", 0
    ri = chart_state.get("currentRegime") or {}
    regime = (ri.get("regime") or "unknown").strip()
    try:
        conf = float(ri.get("confidence") or 0)
        bucket = int(round(conf * 10))  # 0.0-1.0 → 0-10
    except (TypeError, ValueError):
        bucket = 0
    return regime, bucket


def _is_volatility_significant(
    chart_state: Optional[dict], cached_close: float, threshold_pct: float = 1.5,
) -> bool:
    """v106 D2：價格相對快取時刻變化超過 threshold% → 視為「重大波動」需重跑。

    1.5% 是日內較大的單一移動；4h ATR 通常 1-2%。
    """
    if not chart_state or not cached_close:
        return False
    cur_close = (chart_state.get("priceOverview") or {}).get("lastClose") or 0
    try:
        cur = float(cur_close)
        prev = float(cached_close)
    except (TypeError, ValueError):
        return False
    if prev <= 0:
        return False
    delta_pct = abs(cur - prev) / prev * 100
    return delta_pct >= threshold_pct


def _is_recent_query(chart_state: Optional[dict]) -> bool:
    """判斷查詢是否涉及近期數據"""
    if not chart_state:
        return True  # 無 chart_state 視為近期

    end_date = chart_state.get("endDate")
    if not end_date:
        return True  # 無結束日期視為到「現在」

    try:
        end = datetime.strptime(end_date[:10], "%Y-%m-%d")
        cutoff = datetime.utcnow() - timedelta(days=_RECENT_DAYS)
        return end >= cutoff
    except (ValueError, TypeError):
        return True


class AnalysisCache:
    """分析結果快取（SQLite）"""

    def __init__(self):
        self._db_path = settings.db_path / "analysis_cache.db"
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        try:
            # 自動遷移：舊版扁平結構 → 新版 db/ 子目錄
            old_path = settings.data_path / "analysis_cache.db"
            if old_path.exists() and old_path.resolve() != self._db_path.resolve():
                if not self._db_path.exists():
                    shutil.move(str(old_path), str(self._db_path))
                    logger.info(f"遷移 DB: analysis_cache.db → {self._db_path}")
                    for ext in ("-wal", "-shm"):
                        aux = old_path.parent / f"analysis_cache.db{ext}"
                        if aux.exists():
                            shutil.move(str(aux), str(self._db_path.parent / f"analysis_cache.db{ext}"))

            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=3000")

            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    query_hash TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    symbol TEXT,
                    timeframe TEXT,
                    data_fingerprint TEXT,
                    answer TEXT NOT NULL,
                    is_recent INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 0
                )
            """)
            self._conn.commit()
            logger.info(f"分析快取已初始化：{self._db_path}")
        except Exception as e:
            logger.error(f"分析快取初始化失敗: {e}")
            self._conn = None

    def try_get(self, question: str, chart_state: Optional[dict] = None) -> Optional[str]:
        """
        嘗試從快取取得分析結果。

        Returns:
            快取的回答（帶標記），或 None
        """
        if not self._conn or not chart_state:
            return None

        symbol = chart_state.get("symbol", "")
        timeframe = chart_state.get("timeframe", "")
        if not symbol:
            return None

        if should_bypass_cache(question, symbol):
            logger.debug(f"分析快取跳過：問題中的幣種 ≠ chart_state 幣種 ({symbol})")
            return None

        query_hash = _compute_query_hash(question, symbol, timeframe)

        try:
            row = self._conn.execute(
                "SELECT answer, data_fingerprint, is_recent, created_at FROM cache WHERE query_hash = ?",
                (query_hash,),
            ).fetchone()

            if not row:
                return None

            answer, cached_fp, is_recent, created_at = row
            now = time.time()

            # 檢查 TTL
            ttl = _RECENT_TTL if is_recent else _HISTORY_TTL
            if now - created_at > ttl:
                # 已過期，刪除
                self._conn.execute("DELETE FROM cache WHERE query_hash = ?", (query_hash,))
                self._conn.commit()
                return None

            # 檢查數據指紋（含最新收盤價 + regime + confidence bucket）
            regime, conf_bucket = _extract_regime_signature(chart_state)
            current_fp = _compute_data_fingerprint(
                symbol, timeframe,
                chart_state.get("dataPoints", 0),
                chart_state.get("priceOverview", {}).get("lastTimestamp", ""),
                chart_state.get("priceOverview", {}).get("lastClose", 0.0),
                regime=regime,
                confidence_bucket=conf_bucket,
            )
            if cached_fp and cached_fp != current_fp:
                # 數據已更新（價格/regime/信心變了），快取失效
                self._conn.execute("DELETE FROM cache WHERE query_hash = ?", (query_hash,))
                self._conn.commit()
                logger.info(f"快取失效（指紋不同 — 可能是 regime/價格/信心變化）: {query_hash}")
                return None

            # 命中！更新 hit count
            self._conn.execute(
                "UPDATE cache SET hit_count = hit_count + 1 WHERE query_hash = ?",
                (query_hash,),
            )
            self._conn.commit()

            # 加上標記
            cache_time = datetime.utcfromtimestamp(created_at).strftime("%Y-%m-%d %H:%M")
            return (
                f"{answer}\n\n"
                f"_⚡ 來自分析快取（分析時間：{cache_time} UTC，未消耗 token）\n"
                f"數據如已更新，快取將自動失效。_"
            )

        except Exception as e:
            logger.warning(f"查詢分析快取失敗: {e}")
            return None

    def store(
        self,
        question: str,
        answer: str,
        chart_state: Optional[dict] = None,
    ):
        """存入分析結果"""
        if not self._conn or not chart_state or not answer.strip():
            return
        if len(answer) < 50:
            return  # 太短的回答不值得快取

        symbol = chart_state.get("symbol", "")
        timeframe = chart_state.get("timeframe", "")
        if not symbol:
            return

        query_hash = _compute_query_hash(question, symbol, timeframe)
        is_recent = 1 if _is_recent_query(chart_state) else 0
        regime, conf_bucket = _extract_regime_signature(chart_state)
        data_fp = _compute_data_fingerprint(
            symbol, timeframe,
            chart_state.get("dataPoints", 0),
            chart_state.get("priceOverview", {}).get("lastTimestamp", ""),
            chart_state.get("priceOverview", {}).get("lastClose", 0.0),
            regime=regime,
            confidence_bucket=conf_bucket,
        )

        try:
            self._conn.execute(
                """INSERT OR REPLACE INTO cache
                   (query_hash, question, symbol, timeframe, data_fingerprint,
                    answer, is_recent, created_at, hit_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)""",
                (query_hash, question[:200], symbol, timeframe, data_fp,
                 answer, is_recent, time.time()),
            )
            self._conn.commit()
        except Exception as e:
            logger.warning(f"存入分析快取失敗: {e}")

    def cleanup(self):
        """清理過期快取"""
        if not self._conn:
            return
        now = time.time()
        try:
            # 刪除過期的近期快取
            self._conn.execute(
                "DELETE FROM cache WHERE is_recent = 1 AND ? - created_at > ?",
                (now, _RECENT_TTL),
            )
            # 刪除過期的歷史快取
            self._conn.execute(
                "DELETE FROM cache WHERE is_recent = 0 AND ? - created_at > ?",
                (now, _HISTORY_TTL),
            )
            self._conn.commit()
        except Exception as e:
            logger.warning(f"清理分析快取失敗: {e}")

    def get_stats(self) -> dict:
        """取得快取統計"""
        if not self._conn:
            return {"total_entries": 0, "total_hits": 0}
        try:
            row = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(hit_count), 0) FROM cache"
            ).fetchone()
            return {"total_entries": row[0], "total_hits": row[1]}
        except Exception:
            return {"total_entries": 0, "total_hits": 0}


# 全域單例
analysis_cache = AnalysisCache()
