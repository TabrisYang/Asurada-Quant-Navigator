"""阿斯拉量化系統 — 對話歷史持久化

使用 SQLite 保存對話記錄，支援：
- 建立/繼續對話
- 列出歷史對話
- 載入特定對話（重開瀏覽器可回顧）
- 自動清理超過保留天數的記錄
- 對話帶時間戳，方便日後識別數據時效性
"""

import json
import shutil
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger

from app.core.config.settings import settings

# 對話保留天數（改為 60 天 — 蒸餾後原始資料仍保留一段時間供參考）
_RETENTION_DAYS = 60

# 每個對話最多保留的訊息數
_MAX_MESSAGES_PER_CONVERSATION = 200


class ChatHistoryStore:
    """對話歷史持久化存儲（SQLite）"""

    def __init__(self):
        self._db_path = settings.db_path / "chat_history.db"
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        """初始化資料庫和表結構"""
        try:
            # 自動遷移：舊版扁平結構 → 新版 db/ 子目錄
            old_path = settings.data_path / "chat_history.db"
            if old_path.exists() and old_path.resolve() != self._db_path.resolve():
                if not self._db_path.exists():
                    shutil.move(str(old_path), str(self._db_path))
                    logger.info(f"遷移 DB: chat_history.db → {self._db_path}")
                    for ext in ("-wal", "-shm"):
                        aux = old_path.parent / f"chat_history.db{ext}"
                        if aux.exists():
                            shutil.move(str(aux), str(self._db_path.parent / f"chat_history.db{ext}"))

            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=3000")

            # 對話主表
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT '新對話',
                    symbol TEXT,
                    timeframe TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    message_count INTEGER NOT NULL DEFAULT 0,
                    api_key_hash TEXT
                )
            """)

            # 訊息表
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    token_usage TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            """)

            # 索引
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conv
                ON messages (conversation_id, id)
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conv_updated
                ON conversations (updated_at DESC)
            """)

            self._conn.commit()
            logger.info(f"對話歷史存儲已初始化：{self._db_path}")
        except Exception as e:
            logger.error(f"對話歷史存儲初始化失敗: {e}")
            self._conn = None

    def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        token_usage: Optional[dict] = None,
    ) -> str:
        """
        保存一條訊息。如果對話不存在，自動建立。

        Returns:
            conversation_id
        """
        if not self._conn or not content.strip():
            return conversation_id

        try:
            now = datetime.utcnow().isoformat()

            # 確保對話存在
            existing = self._conn.execute(
                "SELECT id FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()

            if not existing:
                # 自動生成標題：取使用者第一句話的前 30 字
                title = content[:30] + ("..." if len(content) > 30 else "") if role == "user" else "新對話"
                self._conn.execute(
                    """INSERT INTO conversations (id, title, symbol, timeframe, created_at, updated_at, message_count)
                       VALUES (?, ?, ?, ?, ?, ?, 0)""",
                    (conversation_id, title, symbol, timeframe, now, now),
                )

            # 保存訊息
            usage_json = json.dumps(token_usage, ensure_ascii=False) if token_usage else None
            self._conn.execute(
                """INSERT INTO messages (conversation_id, role, content, timestamp, token_usage)
                   VALUES (?, ?, ?, ?, ?)""",
                (conversation_id, role, content, now, usage_json),
            )

            # 更新對話的 updated_at 和 message_count
            self._conn.execute(
                """UPDATE conversations
                   SET updated_at = ?, message_count = message_count + 1,
                       symbol = COALESCE(?, symbol),
                       timeframe = COALESCE(?, timeframe)
                   WHERE id = ?""",
                (now, symbol, timeframe, conversation_id),
            )

            # 如果是使用者的第一句話，更新標題
            if role == "user":
                msg_count = self._conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE conversation_id = ? AND role = 'user'",
                    (conversation_id,),
                ).fetchone()[0]
                if msg_count == 1:
                    title = content[:30] + ("..." if len(content) > 30 else "")
                    self._conn.execute(
                        "UPDATE conversations SET title = ? WHERE id = ?",
                        (title, conversation_id),
                    )

            # 清理超出上限的舊訊息
            total = self._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()[0]
            if total > _MAX_MESSAGES_PER_CONVERSATION:
                self._conn.execute(
                    """DELETE FROM messages WHERE id IN (
                        SELECT id FROM messages
                        WHERE conversation_id = ?
                        ORDER BY id ASC
                        LIMIT ?
                    )""",
                    (conversation_id, total - _MAX_MESSAGES_PER_CONVERSATION),
                )

            self._conn.commit()
            return conversation_id

        except Exception as e:
            logger.warning(f"保存對話訊息失敗（不影響主流程）: {e}")
            return conversation_id

    def list_conversations(self, limit: int = 20, offset: int = 0) -> list[dict]:
        """列出最近的對話（按更新時間倒序）"""
        if not self._conn:
            return []

        try:
            cursor = self._conn.execute(
                """SELECT id, title, symbol, timeframe, created_at, updated_at, message_count
                   FROM conversations
                   ORDER BY updated_at DESC
                   LIMIT ? OFFSET ?""",
                (limit, offset),
            )
            return [
                {
                    "id": row[0],
                    "title": row[1],
                    "symbol": row[2],
                    "timeframe": row[3],
                    "created_at": row[4],
                    "updated_at": row[5],
                    "message_count": row[6],
                }
                for row in cursor.fetchall()
            ]
        except Exception as e:
            logger.warning(f"列出對話失敗: {e}")
            return []

    def get_conversation_messages(
        self, conversation_id: str, limit: int = 100
    ) -> list[dict]:
        """取得特定對話的訊息歷史"""
        if not self._conn:
            return []

        try:
            cursor = self._conn.execute(
                """SELECT role, content, timestamp, token_usage
                   FROM messages
                   WHERE conversation_id = ?
                   ORDER BY id ASC
                   LIMIT ?""",
                (conversation_id, limit),
            )
            messages = []
            for row in cursor.fetchall():
                msg = {
                    "role": row[0],
                    "content": row[1],
                    "timestamp": row[2],
                }
                if row[3]:
                    try:
                        msg["usage"] = json.loads(row[3])
                    except json.JSONDecodeError:
                        pass
                messages.append(msg)
            return messages
        except Exception as e:
            logger.warning(f"取得對話訊息失敗: {e}")
            return []

    def delete_conversation(self, conversation_id: str) -> bool:
        """刪除一個對話及其所有訊息"""
        if not self._conn:
            return False

        try:
            self._conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            self._conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            self._conn.commit()
            return True
        except Exception as e:
            logger.warning(f"刪除對話失敗: {e}")
            return False

    def get_stats(self) -> dict:
        """取得對話歷史統計（供蒸餾狀態判斷）"""
        if not self._conn:
            return {"total_conversations": 0, "total_messages": 0, "oldest_date": None}
        try:
            conv_count = self._conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            msg_count = self._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            oldest = self._conn.execute("SELECT MIN(created_at) FROM conversations").fetchone()[0]
            return {
                "total_conversations": conv_count,
                "total_messages": msg_count,
                "oldest_date": oldest,
            }
        except Exception:
            return {"total_conversations": 0, "total_messages": 0, "oldest_date": None}

    def cleanup_old_records(self):
        """清理超過保留天數的對話"""
        if not self._conn:
            return

        cutoff = (datetime.utcnow() - timedelta(days=_RETENTION_DAYS)).isoformat()
        try:
            # 先找出要刪除的對話 ID
            old_ids = self._conn.execute(
                "SELECT id FROM conversations WHERE updated_at < ?", (cutoff,)
            ).fetchall()

            if old_ids:
                ids = [row[0] for row in old_ids]
                placeholders = ",".join("?" * len(ids))
                self._conn.execute(f"DELETE FROM messages WHERE conversation_id IN ({placeholders})", ids)
                self._conn.execute(f"DELETE FROM conversations WHERE id IN ({placeholders})", ids)
                self._conn.commit()
                logger.info(f"已清理 {len(ids)} 筆超過 {_RETENTION_DAYS} 天的對話記錄")
        except Exception as e:
            logger.warning(f"清理舊對話失敗: {e}")


# 全域單例
chat_history = ChatHistoryStore()
