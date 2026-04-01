"""阿斯拉量化系統 — Token 用量追蹤器

使用 SQLite 持久化記錄每次 LLM 呼叫的 token 用量與估算費用。
設計考量：
- API Key 只存 hash（SHA-256 加鹽），不存明文
- 非同步寫入（fire-and-forget），不阻塞主流程
- 自動清理超過 180 天的記錄（可設定）
- 費用顯示標註「估算值，僅供參考」
"""

import asyncio
import hashlib
import shutil
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

from loguru import logger

from app.core.config.settings import settings

# 固定鹽值（不需密碼級安全，只需穩定 hash 來關聯同一個 Key）
_HASH_SALT = "asura_quant_2026"

# 記錄保留天數
_RETENTION_DAYS = 180


def _hash_api_key(api_key: str) -> str:
    """對 API Key 做加鹽 SHA-256 hash（不可逆）"""
    salted = f"{_HASH_SALT}:{api_key}"
    return hashlib.sha256(salted.encode()).hexdigest()[:16]  # 只取前 16 碼，夠用且更緊湊


class UsageTracker:
    """Token 用量追蹤器（SQLite 後端）"""

    def __init__(self):
        self._db_path = settings.db_path / "usage.db"
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        """初始化資料庫和表結構"""
        try:
            # 自動遷移：舊版扁平結構 → 新版 db/ 子目錄
            old_path = settings.data_path / "usage.db"
            if old_path.exists() and old_path.resolve() != self._db_path.resolve():
                if not self._db_path.exists():
                    shutil.move(str(old_path), str(self._db_path))
                    logger.info(f"遷移 DB: usage.db → {self._db_path}")
                    # 同時搬移 WAL/SHM 附加檔
                    for ext in ("-wal", "-shm"):
                        aux = old_path.parent / f"usage.db{ext}"
                        if aux.exists():
                            shutil.move(str(aux), str(self._db_path.parent / f"usage.db{ext}"))

            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")  # 提升並發性能
            self._conn.execute("PRAGMA busy_timeout=3000")

            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS token_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    api_key_hash TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    estimated_cost_usd REAL NOT NULL DEFAULT 0.0,
                    conversation_id TEXT,
                    request_type TEXT DEFAULT 'chat'
                )
            """)

            # 索引加速查詢
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_usage_key_time
                ON token_usage (api_key_hash, timestamp)
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_usage_timestamp
                ON token_usage (timestamp)
            """)

            self._conn.commit()
            logger.info(f"Token 用量追蹤器已初始化：{self._db_path}")
        except Exception as e:
            logger.error(f"用量追蹤器初始化失敗: {e}")
            self._conn = None

    def record_usage(
        self,
        api_key: str,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        estimated_cost_usd: float,
        conversation_id: Optional[str] = None,
        request_type: str = "chat",
    ):
        """
        記錄一次 LLM 呼叫的 token 用量。

        設計為 fire-and-forget：失敗時只 log 不拋例外，不影響主流程。
        """
        if not self._conn:
            return

        try:
            key_hash = _hash_api_key(api_key)
            now = datetime.utcnow().isoformat()

            self._conn.execute(
                """INSERT INTO token_usage
                   (timestamp, api_key_hash, provider, model,
                    prompt_tokens, completion_tokens, total_tokens,
                    estimated_cost_usd, conversation_id, request_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (now, key_hash, provider, model,
                 prompt_tokens, completion_tokens, total_tokens,
                 estimated_cost_usd, conversation_id, request_type),
            )
            self._conn.commit()
        except Exception as e:
            logger.warning(f"記錄 token 用量失敗（不影響主流程）: {e}")

    def record_usage_async(self, **kwargs):
        """非同步包裝：在背景執行記錄，完全不阻塞"""
        try:
            loop = asyncio.get_event_loop()
            loop.call_soon(lambda: self.record_usage(**kwargs))
        except RuntimeError:
            # 如果沒有 event loop，直接同步執行
            self.record_usage(**kwargs)

    def get_summary(self, api_key: str) -> dict:
        """
        取得某個 API Key 的用量摘要。

        Returns:
            {
                "session": {...},     # 本次 session（今日）
                "today": {...},       # 今日累計
                "month": {...},       # 本月累計
                "all_time": {...},    # 歷史總計
                "note": "估算值，僅供參考"
            }
        """
        if not self._conn:
            return self._empty_summary()

        key_hash = _hash_api_key(api_key)
        now = datetime.utcnow()
        today_start = now.strftime("%Y-%m-%dT00:00:00")
        month_start = now.strftime("%Y-%m-01T00:00:00")

        return {
            "today": self._query_range(key_hash, today_start, None),
            "month": self._query_range(key_hash, month_start, None),
            "all_time": self._query_range(key_hash, None, None),
            "recent_models": self._query_recent_models(key_hash),
            "note": "費用為估算值，實際帳單請查閱供應商後台",
        }

    def get_daily_breakdown(self, api_key: str, days: int = 30) -> list[dict]:
        """取得最近 N 天的每日用量明細"""
        if not self._conn:
            return []

        key_hash = _hash_api_key(api_key)
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()

        try:
            cursor = self._conn.execute(
                """SELECT
                       date(timestamp) as day,
                       SUM(prompt_tokens) as prompt_tokens,
                       SUM(completion_tokens) as completion_tokens,
                       SUM(total_tokens) as total_tokens,
                       SUM(estimated_cost_usd) as cost,
                       COUNT(*) as request_count
                   FROM token_usage
                   WHERE api_key_hash = ? AND timestamp >= ?
                   GROUP BY date(timestamp)
                   ORDER BY day DESC""",
                (key_hash, since),
            )
            return [
                {
                    "date": row[0],
                    "prompt_tokens": row[1],
                    "completion_tokens": row[2],
                    "total_tokens": row[3],
                    "estimated_cost_usd": round(row[4], 6),
                    "request_count": row[5],
                }
                for row in cursor.fetchall()
            ]
        except Exception as e:
            logger.warning(f"查詢每日用量失敗: {e}")
            return []

    def cleanup_old_records(self):
        """清理超過保留天數的記錄"""
        if not self._conn:
            return

        cutoff = (datetime.utcnow() - timedelta(days=_RETENTION_DAYS)).isoformat()
        try:
            result = self._conn.execute(
                "DELETE FROM token_usage WHERE timestamp < ?", (cutoff,)
            )
            self._conn.commit()
            deleted = result.rowcount
            if deleted > 0:
                logger.info(f"已清理 {deleted} 筆超過 {_RETENTION_DAYS} 天的用量記錄")
        except Exception as e:
            logger.warning(f"清理舊記錄失敗: {e}")

    # ─── 內部方法 ──────────────────────────────

    def _query_range(
        self, key_hash: str, start: Optional[str], end: Optional[str]
    ) -> dict:
        """查詢特定時間範圍的累計用量"""
        try:
            conditions = ["api_key_hash = ?"]
            params: list = [key_hash]

            if start:
                conditions.append("timestamp >= ?")
                params.append(start)
            if end:
                conditions.append("timestamp <= ?")
                params.append(end)

            where = " AND ".join(conditions)
            cursor = self._conn.execute(  # type: ignore
                f"""SELECT
                       COALESCE(SUM(prompt_tokens), 0),
                       COALESCE(SUM(completion_tokens), 0),
                       COALESCE(SUM(total_tokens), 0),
                       COALESCE(SUM(estimated_cost_usd), 0.0),
                       COUNT(*)
                   FROM token_usage WHERE {where}""",
                params,
            )
            row = cursor.fetchone()
            return {
                "prompt_tokens": row[0],
                "completion_tokens": row[1],
                "total_tokens": row[2],
                "estimated_cost_usd": round(row[3], 6),
                "request_count": row[4],
            }
        except Exception as e:
            logger.warning(f"查詢用量範圍失敗: {e}")
            return self._empty_range()

    def _query_recent_models(self, key_hash: str, limit: int = 5) -> list[str]:
        """查詢最近使用的模型"""
        try:
            cursor = self._conn.execute(  # type: ignore
                """SELECT DISTINCT model FROM token_usage
                   WHERE api_key_hash = ?
                   ORDER BY id DESC LIMIT ?""",
                (key_hash, limit),
            )
            return [row[0] for row in cursor.fetchall()]
        except Exception:
            return []

    @staticmethod
    def _empty_summary() -> dict:
        return {
            "today": UsageTracker._empty_range(),
            "month": UsageTracker._empty_range(),
            "all_time": UsageTracker._empty_range(),
            "recent_models": [],
            "note": "用量追蹤尚未初始化",
        }

    @staticmethod
    def _empty_range() -> dict:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "request_count": 0,
        }


# 全域單例
usage_tracker = UsageTracker()
