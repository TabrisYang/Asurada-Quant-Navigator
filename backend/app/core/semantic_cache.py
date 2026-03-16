"""阿斯拉量化系統 — 語意快取（向量相似度匹配）

核心設計：
- 每筆快取存儲問題文字 + 嵌入向量 + LLM 回答 + 數據指紋
- 查詢時先按 symbol + timeframe 過濾，再做向量餘弦相似度比對
- 相似度超過閾值 → 命中，不消耗 token
- 越用越好：每次 LLM 新回答都自動存入，擴大語意覆蓋範圍

快取層級：
  L1 知識快取（關鍵字匹配，14 筆固定知識）
  L2 分析快取（精確 hash 匹配）
  L3 語意快取（向量相似度匹配）← 本模組
  L4 LLM 呼叫
"""

import hashlib
import shutil
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from loguru import logger

from app.core.config.settings import settings
from app.core import embedding_service
from app.core.symbol_extractor import should_bypass_cache

_HIGH_SIMILARITY_THRESHOLD = 0.92
_SIMILARITY_THRESHOLD = 0.75

_RECENT_TTL = 1 * 3600      # 近期數據快取 1 小時（投資分析時效性高）
_HISTORY_TTL = 7 * 24 * 3600  # 歷史數據快取 7 天（從 30 天縮短）
_RECENT_DAYS = 7

_MAX_ENTRIES_PER_SCOPE = 500


def _is_recent_query(chart_state: Optional[dict]) -> bool:
    if not chart_state:
        return True
    end_date = chart_state.get("endDate")
    if not end_date:
        return True
    try:
        end = datetime.strptime(end_date[:10], "%Y-%m-%d")
        cutoff = datetime.utcnow() - timedelta(days=_RECENT_DAYS)
        return end >= cutoff
    except (ValueError, TypeError):
        return True


def _compute_data_fingerprint(
    symbol: str, timeframe: str, data_points: int, last_ts: str,
    last_close: float = 0.0,
) -> str:
    close_bucket = round(last_close, 2) if last_close else 0
    raw = f"{symbol}|{timeframe}|{data_points}|{last_ts}|{close_bucket}"
    return hashlib.md5(raw.encode()).hexdigest()[:10]


class SemanticCache:
    """基於向量相似度的語意快取"""

    def __init__(self):
        self._db_path = settings.db_path / "semantic_cache.db"
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        try:
            old_path = settings.data_path / "semantic_cache.db"
            if old_path.exists() and old_path.resolve() != self._db_path.resolve():
                if not self._db_path.exists():
                    shutil.move(str(old_path), str(self._db_path))
                    logger.info(f"遷移 DB: semantic_cache.db → {self._db_path}")
                    for ext in ("-wal", "-shm"):
                        aux = old_path.parent / f"semantic_cache.db{ext}"
                        if aux.exists():
                            shutil.move(str(aux), str(self._db_path.parent / f"semantic_cache.db{ext}"))

            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=3000")

            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    answer TEXT NOT NULL,
                    symbol TEXT NOT NULL DEFAULT '',
                    timeframe TEXT NOT NULL DEFAULT '',
                    data_fingerprint TEXT,
                    is_recent INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 0,
                    last_hit_at REAL
                )
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_vectors_scope
                ON vectors (symbol, timeframe)
            """)
            self._conn.commit()

            count = self._conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
            logger.info(f"語意快取已初始化：{self._db_path}（{count} 筆向量）")
        except Exception as e:
            logger.error(f"語意快取初始化失敗: {e}")
            self._conn = None

    def try_get(self, question: str, chart_state: Optional[dict] = None) -> Optional[str]:
        """
        語意匹配查詢：找到與 question 語意最相似的快取回答。
        只返回高置信度命中（≥ HIGH_SIMILARITY_THRESHOLD）。

        Returns:
            匹配的回答（帶來源標記），或 None
        """
        result = self._find_best_match(question, chart_state)
        if not result or result["similarity"] < _HIGH_SIMILARITY_THRESHOLD:
            return None

        cache_time = datetime.utcfromtimestamp(result["created_at"]).strftime("%Y-%m-%d %H:%M")
        return (
            f"{result['answer']}\n\n"
            f"_⚡ 來自語意快取（相似度 {result['similarity']:.0%}，分析時間：{cache_time} UTC，未消耗 token）\n"
            f"越多人問，匹配越精準。_"
        )

    def try_get_with_score(self, question: str, chart_state: Optional[dict] = None) -> Optional[dict]:
        """
        帶分數的語意匹配查詢。

        Returns:
            {"answer": str, "similarity": float, "created_at": float} 或 None
            返回所有超過 _SIMILARITY_THRESHOLD 的匹配（包含中等相似度）
        """
        return self._find_best_match(question, chart_state)

    def _find_best_match(self, question: str, chart_state: Optional[dict] = None) -> Optional[dict]:
        """內部核心匹配邏輯"""
        if not self._conn or not embedding_service.is_available():
            return None

        symbol = (chart_state or {}).get("symbol", "")
        timeframe = (chart_state or {}).get("timeframe", "")
        if not symbol:
            return None

        if should_bypass_cache(question, symbol):
            logger.debug(f"語意快取跳過：問題中的幣種 ≠ chart_state 幣種 ({symbol})")
            return None

        query_vec = embedding_service.encode(question)
        if query_vec is None:
            return None

        try:
            now = time.time()
            min_ts = now - _HISTORY_TTL
            rows = self._conn.execute(
                "SELECT id, embedding, answer, data_fingerprint, is_recent, created_at "
                "FROM vectors WHERE symbol = ? AND timeframe = ? AND created_at > ? "
                "ORDER BY created_at DESC LIMIT 200",
                (symbol, timeframe, min_ts),
            ).fetchall()

            if not rows:
                return None

            best_id = None
            best_sim = 0.0
            best_answer = ""
            best_created = 0.0

            for row_id, emb_blob, answer, cached_fp, is_recent, created_at in rows:
                if is_recent and (now - created_at > _RECENT_TTL):
                    continue

                if cached_fp and chart_state:
                    current_fp = _compute_data_fingerprint(
                        symbol, timeframe,
                        chart_state.get("dataPoints", 0),
                        chart_state.get("priceOverview", {}).get("lastTimestamp", ""),
                        chart_state.get("priceOverview", {}).get("lastClose", 0.0),
                    )
                    if current_fp != cached_fp:
                        continue

                cached_vec = np.frombuffer(emb_blob, dtype=np.float32)
                if cached_vec.shape[0] != embedding_service.get_vector_dim():
                    continue

                sim = embedding_service.cosine_similarity(query_vec, cached_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_id = row_id
                    best_answer = answer
                    best_created = created_at

            if best_sim < _SIMILARITY_THRESHOLD:
                return None

            self._conn.execute(
                "UPDATE vectors SET hit_count = hit_count + 1, last_hit_at = ? WHERE id = ?",
                (now, best_id),
            )
            self._conn.commit()

            return {
                "answer": best_answer,
                "similarity": best_sim,
                "created_at": best_created,
            }

        except Exception as e:
            logger.warning(f"語意快取查詢失敗: {e}")
            return None

    def store(
        self,
        question: str,
        answer: str,
        chart_state: Optional[dict] = None,
    ):
        """
        存入新的問答向量。每次 LLM 回答後自動呼叫。
        """
        if not self._conn or not embedding_service.is_available():
            return
        if not answer.strip() or len(answer) < 50:
            return

        symbol = (chart_state or {}).get("symbol", "")
        timeframe = (chart_state or {}).get("timeframe", "")
        if not symbol:
            return

        vec = embedding_service.encode(question)
        if vec is None:
            return

        existing = self._check_duplicate(vec, symbol, timeframe)
        if existing:
            return

        is_recent = 1 if _is_recent_query(chart_state) else 0
        data_fp = ""
        if chart_state:
            data_fp = _compute_data_fingerprint(
                symbol, timeframe,
                chart_state.get("dataPoints", 0),
                chart_state.get("priceOverview", {}).get("lastTimestamp", ""),
                chart_state.get("priceOverview", {}).get("lastClose", 0.0),
            )

        try:
            self._conn.execute(
                """INSERT INTO vectors
                   (question, embedding, answer, symbol, timeframe,
                    data_fingerprint, is_recent, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    question[:200],
                    vec.tobytes(),
                    answer,
                    symbol,
                    timeframe,
                    data_fp,
                    is_recent,
                    time.time(),
                ),
            )
            self._conn.commit()

            self._trim_if_needed(symbol, timeframe)

        except Exception as e:
            logger.warning(f"語意快取存入失敗: {e}")

    def _check_duplicate(self, vec: np.ndarray, symbol: str, timeframe: str) -> bool:
        """檢查是否已有高度相似的向量（避免重複存入）"""
        try:
            rows = self._conn.execute(
                "SELECT embedding FROM vectors WHERE symbol = ? AND timeframe = ? "
                "ORDER BY created_at DESC LIMIT 100",
                (symbol, timeframe),
            ).fetchall()

            for (emb_blob,) in rows:
                cached_vec = np.frombuffer(emb_blob, dtype=np.float32)
                if cached_vec.shape[0] != embedding_service.get_vector_dim():
                    continue
                sim = embedding_service.cosine_similarity(vec, cached_vec)
                if sim > 0.95:
                    return True
            return False
        except Exception:
            return False

    def _trim_if_needed(self, symbol: str, timeframe: str):
        """限制每個 scope 的向量數量，刪除最舊且命中最少的"""
        try:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM vectors WHERE symbol = ? AND timeframe = ?",
                (symbol, timeframe),
            ).fetchone()[0]

            if count > _MAX_ENTRIES_PER_SCOPE:
                excess = count - _MAX_ENTRIES_PER_SCOPE
                self._conn.execute(
                    """DELETE FROM vectors WHERE id IN (
                        SELECT id FROM vectors
                        WHERE symbol = ? AND timeframe = ?
                        ORDER BY hit_count ASC, created_at ASC
                        LIMIT ?
                    )""",
                    (symbol, timeframe, excess),
                )
                self._conn.commit()
                logger.debug(f"語意快取修剪：{symbol}/{timeframe} 刪除 {excess} 筆舊向量")
        except Exception as e:
            logger.warning(f"語意快取修剪失敗: {e}")

    def cleanup(self):
        """清理過期快取"""
        if not self._conn:
            return
        now = time.time()
        try:
            self._conn.execute(
                "DELETE FROM vectors WHERE is_recent = 1 AND ? - created_at > ?",
                (now, _RECENT_TTL),
            )
            self._conn.execute(
                "DELETE FROM vectors WHERE is_recent = 0 AND ? - created_at > ?",
                (now, _HISTORY_TTL),
            )
            self._conn.commit()
        except Exception as e:
            logger.warning(f"語意快取清理失敗: {e}")

    def get_stats(self) -> dict:
        """取得快取統計"""
        if not self._conn:
            return {"total_vectors": 0, "total_hits": 0, "model_available": False}
        try:
            row = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(hit_count), 0) FROM vectors"
            ).fetchone()
            return {
                "total_vectors": row[0],
                "total_hits": row[1],
                "model_available": embedding_service.is_available(),
                "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
                "similarity_threshold": _SIMILARITY_THRESHOLD,
            }
        except Exception:
            return {"total_vectors": 0, "total_hits": 0, "model_available": False}


semantic_cache = SemanticCache()
