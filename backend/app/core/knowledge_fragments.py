"""阿斯拉量化系統 — 知識碎片儲存庫

自動從 LLM 回答中提取結構化知識碎片，存入向量資料庫，
在新問題到來時以 RAG 方式注入 LLM prompt，使 AI 越用越聰明。

快取層級定位：
  L1 知識快取（關鍵字精確命中）
  L2 分析快取（hash 精確命中）
  L3 語意快取（高相似度直接返回，≥0.92 → 0 token）
  L3.5 知識融合（中度相似 ≥0.65 → RAG 注入碎片，低 token）
  L4 LLM 呼叫
"""

import re
import shutil
import sqlite3
import time
from datetime import datetime
from typing import Optional

import numpy as np
from loguru import logger

from app.core.config.settings import settings
from app.core import embedding_service

_MAX_FRAGMENTS_PER_SYMBOL = 200
_FRAGMENT_TTL_DAYS = 90
_MIN_FRAGMENT_LENGTH = 30

# 碎片類型品質權重（可操作性越高 → 權重越高）
_TYPE_QUALITY: dict[str, float] = {
    "strategy": 1.0,
    "support_resistance": 0.95,
    "lesson": 1.0,
    "factor_update": 0.95,
    "invalidation": 0.90,
    "trend": 0.85,
    "next_validation": 0.85,
    "pattern": 0.80,
    "regime_tag": 0.80,
    "scores": 0.75,
    "indicator": 0.70,
    "volume": 0.70,
    "sentiment": 0.60,
    "general": 0.50,
}


def _compute_fragment_quality(fragment_type: str, hit_count: int) -> float:
    """計算碎片品質分數 [0, 1]。

    根據碎片類型的分析可操作性和歷史命中次數。
    """
    type_score = _TYPE_QUALITY.get(fragment_type, 0.50)
    hit_bonus = min(hit_count * 0.02, 0.20)  # max +0.20 from hits
    return min(type_score + hit_bonus, 1.0)


# ─── 從 LLM 回答中提取知識碎片 ────────────────────

_KEY_INSIGHTS_PATTERN = re.compile(
    r"---KEY_INSIGHTS---\s*\n(.*?)(?:\n---END_INSIGHTS---|$)",
    re.DOTALL,
)


def parse_key_insights(llm_response: str) -> list[dict]:
    """從 LLM 回答中解析 KEY_INSIGHTS 區塊。

    LLM 被要求在回答末尾附加：
    ---KEY_INSIGHTS---
    - [type:support_resistance] BTC 在 95000 附近有強支撐
    - [type:trend] ETH 4H 出現看漲背離
    ---END_INSIGHTS---

    Returns:
        [{"type": "support_resistance", "content": "BTC 在 95000 附近有強支撐"}, ...]
    """
    m = _KEY_INSIGHTS_PATTERN.search(llm_response)
    if not m:
        return []

    raw_block = m.group(1).strip()
    fragments = []

    for line in raw_block.split("\n"):
        line = line.strip().lstrip("-").strip()
        if not line or len(line) < _MIN_FRAGMENT_LENGTH:
            continue

        ftype = "general"
        type_match = re.match(r"\[type:(\w+)\]\s*(.*)", line)
        if type_match:
            ftype = type_match.group(1)
            line = type_match.group(2)

        if len(line) >= _MIN_FRAGMENT_LENGTH:
            fragments.append({"type": ftype, "content": line})

    return fragments


def strip_key_insights(llm_response: str) -> str:
    """移除回答中的 KEY_INSIGHTS 區塊（不顯示給使用者）"""
    return _KEY_INSIGHTS_PATTERN.sub("", llm_response).rstrip()


# ─── SYSTEM_DISTILL 解析（v2 新增）──────────────────

_SYSTEM_DISTILL_PATTERN = re.compile(
    r"---SYSTEM_DISTILL---\s*\n(.*?)(?:\n---END_DISTILL---|$)",
    re.DOTALL,
)

_DISTILL_LINE_RE = re.compile(r"\[(\w+)\]\s*(.*)")

# SYSTEM_DISTILL 碎片的類型品質權重
_DISTILL_TYPE_QUALITY: dict[str, float] = {
    "regime_tag": 0.80,
    "factor_update": 0.95,
    "scores": 0.75,
    "lesson": 1.0,
    "invalidation": 0.90,
    "next_validation": 0.85,
}


def parse_system_distill(llm_response: str) -> list[dict]:
    """從 LLM 回答中解析 SYSTEM_DISTILL 區塊。

    LLM 被要求在深度分析結束後附加：
    ---SYSTEM_DISTILL---
    - [regime_tag] 趨勢上行_高波動_時框一致
    - [factor_update] RSI_4H:weakening, ADX_4H:stable
    - [scores] bull=7 bear=2 neutral=1 confidence=high consistency=5/7
    - [lesson] 盤整突破後的 ADX 上升是有效的趨勢確認信號
    - [invalidation] 若跌破 0.245 則多頭結構失效
    - [next_validation] 觀察 4H 收盤是否站穩 EMA20
    ---END_DISTILL---

    Returns:
        [{"type": "regime_tag", "content": "趨勢上行_高波動_時框一致"}, ...]
    """
    m = _SYSTEM_DISTILL_PATTERN.search(llm_response)
    if not m:
        return []

    raw_block = m.group(1).strip()
    fragments: list[dict] = []

    for line in raw_block.split("\n"):
        line = line.strip().lstrip("-").strip()
        if not line:
            continue

        dm = _DISTILL_LINE_RE.match(line)
        if dm:
            dtype = dm.group(1)
            content = dm.group(2).strip()
            if content:
                fragments.append({"type": dtype, "content": content})

    return fragments


def strip_system_distill(llm_response: str) -> str:
    """移除回答中的 SYSTEM_DISTILL 區塊（不顯示給使用者）"""
    return _SYSTEM_DISTILL_PATTERN.sub("", llm_response).rstrip()


# ─── 知識碎片資料庫 ────────────────────────────────

class KnowledgeFragmentStore:
    """向量化知識碎片儲存庫（SQLite + embedding）"""

    def __init__(self):
        self._db_path = settings.db_path / "knowledge_fragments.db"
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        try:
            old_path = settings.data_path / "knowledge_fragments.db"
            if old_path.exists() and old_path.resolve() != self._db_path.resolve():
                if not self._db_path.exists():
                    shutil.move(str(old_path), str(self._db_path))
                    logger.info(f"遷移 DB: knowledge_fragments.db → {self._db_path}")

            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=3000")

            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS fragments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    fragment_type TEXT NOT NULL DEFAULT 'general',
                    symbol TEXT NOT NULL DEFAULT '',
                    source_question TEXT,
                    created_at REAL NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 0,
                    last_hit_at REAL,
                    is_seed INTEGER NOT NULL DEFAULT 0
                )
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_frag_symbol
                ON fragments (symbol)
            """)
            # v155 migration：refuted_count — 碎片來源預測被驗證錯誤的次數
            try:
                self._conn.execute(
                    "ALTER TABLE fragments ADD COLUMN refuted_count INTEGER NOT NULL DEFAULT 0"
                )
            except sqlite3.OperationalError:
                pass  # 欄位已存在
            self._conn.commit()

            count = self._conn.execute("SELECT COUNT(*) FROM fragments").fetchone()[0]
            logger.info(f"知識碎片庫已初始化：{self._db_path}（{count} 筆碎片）")
        except Exception as e:
            logger.error(f"知識碎片庫初始化失敗: {e}")
            self._conn = None

    def _insert_fragment_no_commit(
        self,
        content: str,
        fragment_type: str = "general",
        symbol: str = "",
        source_question: str = "",
        is_seed: bool = False,
    ) -> bool:
        """INSERT 一筆 fragment 但不 commit、不 trim。

        供 store_batch 用 — caller 負責 transaction commit + 結尾統一 trim。
        回傳 True/False 表示是否實際 INSERT（最小長度 / encode 失敗 / dedup 都會跳過）。
        """
        if len(content) < _MIN_FRAGMENT_LENGTH:
            return False

        vec = embedding_service.encode(content)
        if vec is None:
            return False

        if self._is_duplicate(vec, symbol):
            return False

        try:
            self._conn.execute(
                """INSERT INTO fragments
                   (content, embedding, fragment_type, symbol,
                    source_question, created_at, is_seed)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    content[:500],
                    vec.tobytes(),
                    fragment_type,
                    symbol,
                    (source_question or "")[:200],
                    time.time(),
                    1 if is_seed else 0,
                ),
            )
            return True
        except Exception as e:
            logger.warning(f"知識碎片 INSERT 失敗: {e}")
            return False

    def store_fragment(
        self,
        content: str,
        fragment_type: str = "general",
        symbol: str = "",
        source_question: str = "",
        is_seed: bool = False,
    ) -> bool:
        """存入一筆知識碎片（自動編碼向量 + 去重）— 單筆同步 commit + trim。"""
        if not self._conn or not embedding_service.is_available():
            return False

        inserted = self._insert_fragment_no_commit(
            content, fragment_type, symbol, source_question, is_seed
        )
        if inserted:
            try:
                self._conn.commit()
                self._trim_if_needed(symbol)
            except Exception as e:
                logger.warning(f"知識碎片存入 commit/trim 失敗: {e}")
                return False
        return inserted

    def store_batch(
        self,
        fragments: list[dict],
        symbol: str = "",
        source_question: str = "",
    ) -> int:
        """批次存入知識碎片（v116 優化：單一 transaction + 結尾 trim 一次）。

        改善前：N 筆 fragments = N×2 次 commit（INSERT+trim 各一），
        SQLite WAL fsync 鎖住 connection 數百 ms，造成 frontend 切標的卡頓。
        改善後：N 筆 = 2 次 commit（batch INSERT 1 次 + 結尾 trim 1 次）。

        Args:
            fragments: [{"type": "...", "content": "..."}, ...]
        Returns:
            實際存入的筆數
        """
        if not self._conn or not embedding_service.is_available():
            return 0

        stored = 0
        try:
            # `with self._conn:` 自動 begin → commit on success / rollback on exception
            with self._conn:
                for frag in fragments:
                    if self._insert_fragment_no_commit(
                        content=frag["content"],
                        fragment_type=frag.get("type", "general"),
                        symbol=symbol,
                        source_question=source_question,
                    ):
                        stored += 1
            # 結尾 trim 一次（_trim_if_needed 內部已 commit）
            if stored > 0:
                self._trim_if_needed(symbol)
        except Exception as e:
            logger.warning(f"知識碎片批次存入失敗: {e}")
        return stored

    def retrieve_relevant(
        self,
        question: str,
        symbol: str = "",
        top_k: int = 5,
        min_similarity: float = 0.45,
    ) -> list[dict]:
        """根據問題語意檢索最相關的知識碎片（含品質分級 + 時間衰減）。

        排名公式：final_score = similarity * 0.70 + quality * 0.15 + time_freshness * 0.15
        - quality: 根據碎片類型的分析可操作性 + 命中次數
        - time_freshness: 30天半衰期指數衰減

        Returns:
            [{"content": ..., "type": ..., "similarity": ..., "score": ..., "id": ...}, ...]
        """
        if not self._conn or not embedding_service.is_available():
            return []

        query_vec = embedding_service.encode(question)
        if query_vec is None:
            return []

        try:
            where_clause = "WHERE 1=1"
            params: list = []

            if symbol:
                where_clause += " AND (symbol = ? OR symbol = '')"
                params.append(symbol)

            now = time.time()
            ttl_seconds = _FRAGMENT_TTL_DAYS * 24 * 3600
            where_clause += " AND (is_seed = 1 OR ? - created_at < ?)"
            params.extend([now, ttl_seconds])

            rows = self._conn.execute(
                f"SELECT id, content, embedding, fragment_type, created_at, hit_count, symbol, refuted_count "
                f"FROM fragments {where_clause}",
                params,
            ).fetchall()

            if not rows:
                return []

            results = []
            for row_id, content, emb_blob, ftype, created_at, hit_count, frag_symbol, refuted_count in rows:
                cached_vec = np.frombuffer(emb_blob, dtype=np.float32)
                if cached_vec.shape[0] != embedding_service.get_vector_dim():
                    continue
                sim = embedding_service.cosine_similarity(query_vec, cached_vec)
                if sim < min_similarity:
                    continue

                quality = _compute_fragment_quality(ftype, hit_count) * (0.5 ** refuted_count)
                days_old = (now - created_at) / 86400
                time_freshness = float(np.exp(-0.023 * days_old))  # ~30-day half-life

                score = sim * 0.70 + quality * 0.15 + time_freshness * 0.15

                results.append({
                    "id": row_id,
                    "content": content,
                    "type": ftype,
                    "similarity": round(sim, 4),
                    "score": round(score, 4),
                    "created_at": created_at,
                    # v145：回傳碎片來源標的（'' = 通用碎片），供 prompt 標註，
                    # 避免 LLM 把別標的的勝率當成當前標的的歷史勝率（Q8 前視偏差）
                    "symbol": frag_symbol or "通用",
                })

            results.sort(key=lambda x: x["score"], reverse=True)
            top_results = results[:top_k]

            if top_results:
                ids = [r["id"] for r in top_results]
                placeholders = ",".join("?" * len(ids))
                self._conn.execute(
                    f"UPDATE fragments SET hit_count = hit_count + 1, last_hit_at = ? "
                    f"WHERE id IN ({placeholders})",
                    [now] + ids,
                )
                self._conn.commit()

            return top_results

        except Exception as e:
            logger.warning(f"知識碎片檢索失敗: {e}")
            return []

    def _is_duplicate(self, vec: np.ndarray, symbol: str) -> bool:
        """檢查是否已有高度相似的碎片"""
        try:
            rows = self._conn.execute(
                "SELECT embedding FROM fragments WHERE symbol = ? OR symbol = '' "
                "ORDER BY created_at DESC LIMIT 200",
                (symbol,),
            ).fetchall()
            for (emb_blob,) in rows:
                cached_vec = np.frombuffer(emb_blob, dtype=np.float32)
                if cached_vec.shape[0] != embedding_service.get_vector_dim():
                    continue
                sim = embedding_service.cosine_similarity(vec, cached_vec)
                if sim > 0.92:
                    return True
            return False
        except Exception:
            return False

    def _trim_if_needed(self, symbol: str):
        """限制每個 symbol 的碎片數量"""
        try:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM fragments WHERE symbol = ? AND is_seed = 0",
                (symbol,),
            ).fetchone()[0]
            if count > _MAX_FRAGMENTS_PER_SYMBOL:
                excess = count - _MAX_FRAGMENTS_PER_SYMBOL
                self._conn.execute(
                    """DELETE FROM fragments WHERE id IN (
                        SELECT id FROM fragments
                        WHERE symbol = ? AND is_seed = 0
                        ORDER BY hit_count ASC, created_at ASC
                        LIMIT ?
                    )""",
                    (symbol, excess),
                )
                self._conn.commit()
                logger.debug(f"知識碎片修剪：{symbol} 刪除 {excess} 筆")
        except Exception as e:
            logger.warning(f"知識碎片修剪失敗: {e}")

    def penalize_by_source(self, source_question: str, symbol: str) -> int:
        """v155 證偽淘汰：來源預測被驗證錯誤 → 同源碎片 refuted_count+1，
        累計 2 次直接刪除（不等 90 天 TTL）。種子碎片不受影響。

        join key = source_question（predictions 存全文、fragments 截 200 字，
        呼叫端需先截斷）。回傳受影響筆數。
        """
        if not self._conn or not source_question:
            return 0
        try:
            cur = self._conn.execute(
                "UPDATE fragments SET refuted_count = refuted_count + 1 "
                "WHERE source_question = ? AND symbol = ? AND is_seed = 0",
                (source_question, symbol),
            )
            n_penalized = cur.rowcount
            cur2 = self._conn.execute(
                "DELETE FROM fragments WHERE refuted_count >= 2 AND is_seed = 0",
            )
            n_deleted = cur2.rowcount
            self._conn.commit()
            if n_penalized:
                logger.info(
                    f"[fragment_refute] 降權 {n_penalized} 筆（symbol={symbol}）"
                    + (f"，刪除 {n_deleted} 筆（refuted>=2）" if n_deleted else "")
                )
            return n_penalized
        except Exception as e:
            logger.warning(f"碎片證偽降權失敗: {e}")
            return 0

    def cleanup_expired(self):
        """清理過期碎片（保留種子碎片）"""
        if not self._conn:
            return
        now = time.time()
        ttl_seconds = _FRAGMENT_TTL_DAYS * 24 * 3600
        try:
            deleted = self._conn.execute(
                "DELETE FROM fragments WHERE is_seed = 0 AND ? - created_at > ?",
                (now, ttl_seconds),
            ).rowcount
            self._conn.commit()
            if deleted:
                logger.info(f"知識碎片清理：刪除 {deleted} 筆過期碎片")
        except Exception as e:
            logger.warning(f"知識碎片清理失敗: {e}")

    def get_stats(self) -> dict:
        if not self._conn:
            return {"total_fragments": 0, "total_hits": 0, "seed_count": 0}
        try:
            row = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(hit_count), 0) FROM fragments"
            ).fetchone()
            seed = self._conn.execute(
                "SELECT COUNT(*) FROM fragments WHERE is_seed = 1"
            ).fetchone()[0]
            return {
                "total_fragments": row[0],
                "total_hits": row[1],
                "seed_count": seed,
            }
        except Exception:
            return {"total_fragments": 0, "total_hits": 0, "seed_count": 0}

    def list_all(
        self,
        symbol: Optional[str] = None,
        fragment_type: Optional[str] = None,
        sort_by: str = "hit_count",
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        """列出所有知識碎片（供前端瀏覽器使用）。"""
        if not self._conn:
            return {"fragments": [], "total": 0, "symbols": [], "types": []}

        try:
            where_parts = ["1=1"]
            params: list = []
            if symbol:
                where_parts.append("symbol = ?")
                params.append(symbol)
            if fragment_type:
                where_parts.append("fragment_type = ?")
                params.append(fragment_type)
            where_clause = " AND ".join(where_parts)

            sort_map = {
                "hit_count": "hit_count DESC, created_at DESC",
                "created_at": "created_at DESC",
                "type": "fragment_type ASC, hit_count DESC",
            }
            order = sort_map.get(sort_by, sort_map["hit_count"])

            total = self._conn.execute(
                f"SELECT COUNT(*) FROM fragments WHERE {where_clause}", params,
            ).fetchone()[0]

            rows = self._conn.execute(
                f"""SELECT id, content, fragment_type, symbol, source_question,
                           created_at, hit_count, is_seed
                    FROM fragments
                    WHERE {where_clause}
                    ORDER BY {order}
                    LIMIT ? OFFSET ?""",
                params + [limit, offset],
            ).fetchall()

            now = time.time()
            fragments = []
            for r in rows:
                fid, content, ftype, sym, src_q, created_at, hits, is_seed = r
                age_days = (now - created_at) / 86400
                quality = _compute_fragment_quality(ftype, hits)
                fragments.append({
                    "id": fid,
                    "content": content,
                    "type": ftype,
                    "symbol": sym or "通用",
                    "source_question": src_q or "",
                    "hit_count": hits,
                    "quality_score": round(quality, 2),
                    "age_days": round(age_days, 1),
                    "created_at": datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M"),
                    "is_seed": bool(is_seed),
                })

            all_symbols = [
                r[0] or "通用" for r in
                self._conn.execute("SELECT DISTINCT symbol FROM fragments ORDER BY symbol").fetchall()
            ]
            all_types = [
                r[0] for r in
                self._conn.execute("SELECT DISTINCT fragment_type FROM fragments ORDER BY fragment_type").fetchall()
            ]

            return {
                "fragments": fragments,
                "total": total,
                "symbols": sorted(set(all_symbols)),
                "types": all_types,
            }
        except Exception as e:
            logger.warning(f"列出知識碎片失敗: {e}")
            return {"fragments": [], "total": 0, "symbols": [], "types": []}

    def delete_by_id(self, fragment_id: int) -> bool:
        """刪除指定 ID 的知識碎片。"""
        if not self._conn:
            return False
        try:
            deleted = self._conn.execute(
                "DELETE FROM fragments WHERE id = ?", (fragment_id,),
            ).rowcount
            self._conn.commit()
            return deleted > 0
        except Exception as e:
            logger.warning(f"刪除知識碎片失敗: {e}")
            return False


fragment_store = KnowledgeFragmentStore()
