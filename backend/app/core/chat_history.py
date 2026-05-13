"""阿斯拉量化系統 — 對話歷史持久化

使用 SQLite 保存對話記錄，支援：
- 建立/繼續對話
- 列出歷史對話
- 載入特定對話（重開瀏覽器可回顧）
- 自動清理超過保留天數的記錄
- 對話帶時間戳，方便日後識別數據時效性
"""

import json
from datetime import datetime, timedelta
from typing import Optional

from loguru import logger

from app.core.base_db import BaseDBStore

# 對話保留天數（改為 60 天 — 蒸餾後原始資料仍保留一段時間供參考）
_RETENTION_DAYS = 60

# 每個對話最多保留的訊息數
_MAX_MESSAGES_PER_CONVERSATION = 200


class ChatHistoryStore(BaseDBStore):
    """對話歷史持久化存儲（SQLite）"""

    def __init__(self):
        super().__init__("chat_history.db")
        self._init_db()

    def _init_db(self):
        """初始化資料庫和表結構"""
        try:
            self._connect()
            if not self._conn:
                return

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
                    status TEXT NOT NULL DEFAULT 'completed',
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            """)

            # v115 schema migration: 既有 DB 加 status column（既有訊息預設 completed）
            try:
                self._conn.execute(
                    "ALTER TABLE messages ADD COLUMN status TEXT NOT NULL DEFAULT 'completed'"
                )
                logger.info("[v115] messages 表加 status column（既有訊息預設 completed）")
            except Exception:
                pass

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
            # v125-C: 用 `with self._conn:` 包成 transaction，保證 partial write 不發生
            # SQLite context manager 語意：成功 → COMMIT、exception → ROLLBACK
            # 之前的版本：多個 execute 之間若 exception、commit() 沒呼叫，下次 commit
            # 可能把半截 transaction 跟新 statements 一起 commit 進去（partial save）
            with self._conn:
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
                # with self._conn: 結束時自動 COMMIT；若上面任一 execute 拋例外則自動 ROLLBACK
            return conversation_id

        except Exception as e:
            logger.warning(f"保存對話訊息失敗（不影響主流程）: {e}")
            return conversation_id

    # ─── v115：Producer/Consumer 解耦支援 ──────────────────

    def create_streaming_message(
        self,
        conversation_id: str,
        role: str = "assistant",
        initial_content: str = "",
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> Optional[int]:
        """建立 status='streaming' 的空訊息，回傳 message_id。

        Producer 用此 message_id 增量 append 內容，
        Consumer 從同一 message_id tail 讀並 yield SSE。
        """
        if not self._conn:
            return None

        try:
            now = datetime.utcnow().isoformat()

            # 確保對話存在（複製 save_message 的 upsert 邏輯）
            existing = self._conn.execute(
                "SELECT id FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if not existing:
                self._conn.execute(
                    """INSERT INTO conversations (id, title, symbol, timeframe, created_at, updated_at, message_count)
                       VALUES (?, ?, ?, ?, ?, ?, 0)""",
                    (conversation_id, "新對話", symbol, timeframe, now, now),
                )

            cursor = self._conn.execute(
                """INSERT INTO messages (conversation_id, role, content, timestamp, status)
                   VALUES (?, ?, ?, ?, 'streaming')""",
                (conversation_id, role, initial_content, now),
            )
            message_id = cursor.lastrowid

            self._conn.execute(
                """UPDATE conversations
                   SET updated_at = ?, message_count = message_count + 1,
                       symbol = COALESCE(?, symbol),
                       timeframe = COALESCE(?, timeframe)
                   WHERE id = ?""",
                (now, symbol, timeframe, conversation_id),
            )

            self._conn.commit()
            return message_id
        except Exception as e:
            logger.warning(f"建立 streaming message 失敗: {e}")
            return None

    def append_message_content(self, message_id: int, additional_content: str) -> bool:
        """增量 append 文字到指定訊息（producer 用）。"""
        if not self._conn or not additional_content:
            return False
        try:
            self._conn.execute(
                "UPDATE messages SET content = content || ? WHERE id = ?",
                (additional_content, message_id),
            )
            self._conn.commit()
            return True
        except Exception as e:
            logger.warning(f"append message content 失敗: {e}")
            return False

    def update_message_status(
        self,
        message_id: int,
        status: str,
        final_content: Optional[str] = None,
        token_usage: Optional[dict] = None,
    ) -> bool:
        """更新 message status（streaming → completed / error）。

        若同時提供 final_content 會 overwrite content（producer 在結束時可一次性
        用 strip 後的 clean text 覆寫，避免 partial token 殘留中段 markup）。
        """
        if not self._conn:
            return False
        try:
            if final_content is not None:
                usage_json = json.dumps(token_usage, ensure_ascii=False) if token_usage else None
                self._conn.execute(
                    "UPDATE messages SET content = ?, status = ?, token_usage = ? WHERE id = ?",
                    (final_content, status, usage_json, message_id),
                )
            else:
                self._conn.execute(
                    "UPDATE messages SET status = ? WHERE id = ?",
                    (status, message_id),
                )
            self._conn.commit()
            return True
        except Exception as e:
            logger.warning(f"update message status 失敗: {e}")
            return False

    def get_message_content_after(
        self, message_id: int, offset: int = 0
    ) -> tuple[str, str]:
        """從 message_id 的 content 讀取 offset 之後的部分 + 當前 status。

        Consumer 用此 method tail DB：
            new_content, status = chat_history.get_message_content_after(mid, offset)
            if new_content:
                yield SSE token
                offset += len(new_content)
            if status in ('completed', 'error'):
                break

        Returns:
            (new_content_after_offset, current_status)
            找不到 message 時回傳 ('', 'error')
        """
        if not self._conn:
            return ("", "error")
        try:
            row = self._conn.execute(
                "SELECT content, status FROM messages WHERE id = ?",
                (message_id,),
            ).fetchone()
            if not row:
                return ("", "error")
            content, status = row[0] or "", row[1] or "completed"
            if offset >= len(content):
                return ("", status)
            return (content[offset:], status)
        except Exception as e:
            logger.warning(f"get message content after 失敗: {e}")
            return ("", "error")

    # ─── 既有 method ──────────────────────────────

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
                """SELECT id, role, content, timestamp, token_usage, status
                   FROM messages
                   WHERE conversation_id = ?
                   ORDER BY id ASC
                   LIMIT ?""",
                (conversation_id, limit),
            )
            messages = []
            for row in cursor.fetchall():
                msg = {
                    "id": row[0],
                    "role": row[1],
                    "content": row[2],
                    "timestamp": row[3],
                    "status": row[5] or "completed",
                }
                if row[4]:
                    try:
                        msg["usage"] = json.loads(row[4])
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
