"""阿斯拉量化系統 — 知識蒸餾模組（漸進式知識壓縮）

核心概念：不刪除舊對話，而是透過 LLM 將其「蒸餾」成精華知識。
設計：
- knowledge_distilled 表：存放蒸餾後的知識摘要（按幣種/時間維度歸類）
- user_profile 表：存放使用者分析風格檔案（從歷史對話中提煉）
- 蒸餾流程：預覽 → 確認 → 壓縮 → 刪除原始訊息
- 蒸餾後的知識自動注入每次 LLM 對話 context
- 版本控制：每次蒸餾都有版本號，可回滾
"""

import json
import shutil
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Optional

from loguru import logger

from app.core.config.settings import settings

# 蒸餾觸發的最小天數（累積多少天才建議蒸餾）
DISTILL_MIN_DAYS = 30

# 蒸餾知識注入 context 的最大字數限制
MAX_CONTEXT_CHARS = 2000

# 使用者風格檔案最大字數
MAX_PROFILE_CHARS = 500


class KnowledgeDistiller:
    """知識蒸餾器（SQLite 後端）"""

    def __init__(self):
        self._db_path = settings.db_path / "knowledge.db"
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        """初始化資料庫"""
        try:
            # 自動遷移：舊版扁平結構 → 新版 db/ 子目錄
            old_path = settings.data_path / "knowledge.db"
            if old_path.exists() and old_path.resolve() != self._db_path.resolve():
                if not self._db_path.exists():
                    shutil.move(str(old_path), str(self._db_path))
                    logger.info(f"遷移 DB: knowledge.db → {self._db_path}")
                    for ext in ("-wal", "-shm"):
                        aux = old_path.parent / f"knowledge.db{ext}"
                        if aux.exists():
                            shutil.move(str(aux), str(self._db_path.parent / f"knowledge.db{ext}"))

            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=3000")

            # 蒸餾知識表：按幣種分類的分析摘要
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS distilled_knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    period_start TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    key_numbers TEXT,
                    source_message_count INTEGER NOT NULL DEFAULT 0,
                    original_chars INTEGER NOT NULL DEFAULT 0,
                    distilled_chars INTEGER NOT NULL DEFAULT 0,
                    version INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    distilled_by TEXT DEFAULT 'llm'
                )
            """)

            # 使用者分析風格檔案
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profile (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # 蒸餾歷史（供回滾）
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS distill_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    distill_type TEXT NOT NULL,
                    original_data TEXT NOT NULL,
                    distilled_result TEXT NOT NULL,
                    token_used INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
            """)

            # 索引
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_dk_symbol
                ON distilled_knowledge (symbol)
            """)

            self._conn.commit()
            logger.info(f"知識蒸餾器已初始化：{self._db_path}")
        except Exception as e:
            logger.error(f"知識蒸餾器初始化失敗: {e}")
            self._conn = None

    # ─── 蒸餾狀態查詢 ─────────────────────────

    def get_distill_status(self, chat_history_conn: sqlite3.Connection) -> dict:
        """
        查詢目前的蒸餾狀態（供前端判斷是否該觸發蒸餾）。

        Returns:
            {
                "should_distill": bool,      # 是否建議蒸餾
                "undistilled_days": int,      # 未蒸餾的天數
                "undistilled_messages": int,  # 未蒸餾的訊息數
                "undistilled_chars": int,     # 未蒸餾的總字數
                "estimated_tokens": int,      # 預估蒸餾消耗的 token
                "existing_knowledge": int,    # 已有的知識條數
                "last_distill_time": str|None # 上次蒸餾時間
            }
        """
        result = {
            "should_distill": False,
            "undistilled_days": 0,
            "undistilled_messages": 0,
            "undistilled_chars": 0,
            "estimated_tokens": 0,
            "existing_knowledge": 0,
            "last_distill_time": None,
        }

        try:
            # 查已有的知識數量
            if self._conn:
                row = self._conn.execute("SELECT COUNT(*) FROM distilled_knowledge").fetchone()
                result["existing_knowledge"] = row[0] if row else 0

                row = self._conn.execute(
                    "SELECT MAX(created_at) FROM distill_history"
                ).fetchone()
                result["last_distill_time"] = row[0] if row and row[0] else None

            # 從 chat_history 查未蒸餾的資料
            last_distill = result["last_distill_time"]
            if last_distill:
                cursor = chat_history_conn.execute(
                    """SELECT COUNT(*), COALESCE(SUM(LENGTH(content)), 0),
                              MIN(timestamp), MAX(timestamp)
                       FROM messages WHERE timestamp > ?""",
                    (last_distill,),
                )
            else:
                cursor = chat_history_conn.execute(
                    """SELECT COUNT(*), COALESCE(SUM(LENGTH(content)), 0),
                              MIN(timestamp), MAX(timestamp)
                       FROM messages"""
                )

            row = cursor.fetchone()
            if row and row[0] > 0:
                result["undistilled_messages"] = row[0]
                result["undistilled_chars"] = row[1]

                try:
                    earliest = datetime.fromisoformat(row[2])
                    latest = datetime.fromisoformat(row[3])
                    result["undistilled_days"] = max(1, (latest - earliest).days)
                except (ValueError, TypeError):
                    result["undistilled_days"] = 0

                # 預估 token：輸入(原始文字的 token) + 輸出(摘要的 token)
                # 粗估：1 個中文字 ≈ 1.5 token
                est_input_tokens = int(result["undistilled_chars"] * 1.5)
                est_output_tokens = min(3000, est_input_tokens // 3)
                result["estimated_tokens"] = est_input_tokens + est_output_tokens

                result["should_distill"] = result["undistilled_days"] >= DISTILL_MIN_DAYS

        except Exception as e:
            logger.warning(f"查詢蒸餾狀態失敗: {e}")

        return result

    # ─── 準備蒸餾材料 ─────────────────────────

    def prepare_distill_material(
        self, chat_history_conn: sqlite3.Connection
    ) -> dict:
        """
        從 chat_history 提取待蒸餾的對話材料，按幣種分組。

        Returns:
            {
                "groups": {
                    "BTC/USDT": [{"q": "...", "a": "...", "time": "..."}],
                    "ETH/USDT": [...],
                    "_general": [...]  # 無特定幣種的通用對話
                },
                "total_messages": int,
                "total_chars": int,
            }
        """
        # 確定起始時間
        last_distill = None
        if self._conn:
            row = self._conn.execute(
                "SELECT MAX(created_at) FROM distill_history"
            ).fetchone()
            last_distill = row[0] if row and row[0] else None

        # 取出對話和訊息
        try:
            if last_distill:
                conversations = chat_history_conn.execute(
                    """SELECT c.id, c.symbol, c.timeframe
                       FROM conversations c
                       WHERE c.updated_at > ?
                       ORDER BY c.updated_at ASC""",
                    (last_distill,),
                ).fetchall()
            else:
                conversations = chat_history_conn.execute(
                    """SELECT c.id, c.symbol, c.timeframe
                       FROM conversations c
                       ORDER BY c.updated_at ASC"""
                ).fetchall()

            groups: dict[str, list] = {}
            total_messages = 0
            total_chars = 0

            for conv_id, symbol, timeframe in conversations:
                group_key = symbol or "_general"

                msgs = chat_history_conn.execute(
                    """SELECT role, content, timestamp
                       FROM messages
                       WHERE conversation_id = ?
                       ORDER BY id ASC""",
                    (conv_id,),
                ).fetchall()

                # 將 Q&A 配對
                current_q = None
                for role, content, ts in msgs:
                    total_messages += 1
                    total_chars += len(content)

                    if role == "user":
                        current_q = {"q": content, "time": ts}
                    elif role == "assistant" and current_q:
                        qa = {**current_q, "a": content, "timeframe": timeframe or ""}
                        if group_key not in groups:
                            groups[group_key] = []
                        groups[group_key].append(qa)
                        current_q = None

            return {
                "groups": groups,
                "total_messages": total_messages,
                "total_chars": total_chars,
            }
        except Exception as e:
            logger.error(f"準備蒸餾材料失敗: {e}")
            return {"groups": {}, "total_messages": 0, "total_chars": 0}

    # ─── 建構 LLM 蒸餾 Prompt ────────────────

    def build_distill_prompt(self, symbol: str, qa_pairs: list[dict]) -> str:
        """
        建構送給 LLM 做蒸餾的 prompt。

        要求 LLM 輸出：
        1. 分析時間線（關鍵日期 + 指標數值 + 結論）
        2. 使用者分析風格觀察
        """
        # 整理 Q&A 為文字
        qa_text = ""
        for i, qa in enumerate(qa_pairs[:50], 1):  # 最多 50 組
            ts = qa.get("time", "")[:10]
            q = qa["q"][:200]
            a = qa["a"][:300]
            qa_text += f"\n[{i}] 日期:{ts}\nQ: {q}\nA: {a}\n"

        symbol_label = symbol if symbol != "_general" else "一般性問題"

        return f"""你是一個量化分析知識整理助手。以下是使用者與 AI 的歷史對話記錄（關於 {symbol_label}）。

請將這些對話整理成精簡的知識摘要，格式如下：

## 分析時間線
按日期整理關鍵分析結果，每條包含：日期、觀察到的指標數值、得出的結論。
只保留有具體數字或明確結論的內容，刪除閒聊和重複內容。

## 關鍵數字
列出所有重要的價格、指標閾值、百分比等數字。

## 使用者偏好
觀察使用者最常使用的指標、偏好的時間級別、關注的價格區間等。

要求：
- 嚴格控制在 800 字以內
- 保留所有具體數字（價格、RSI 值、日期等）
- 如果有不確定的推論，用「?」標記
- 用繁體中文回答

---
以下是原始對話記錄：
{qa_text}"""

    def build_profile_prompt(self, all_qa_pairs: list[dict]) -> str:
        """建構使用者風格分析的 prompt"""
        sample = all_qa_pairs[:30]
        questions = "\n".join(
            f"- {qa['q'][:100]}" for qa in sample
        )

        return f"""根據以下使用者的歷史提問，分析其投資分析風格：

{questions}

請用 200 字以內，以條列方式歸納：
1. 常用的技術指標
2. 偏好的時間級別
3. 關注的價格行為模式
4. 風險偏好傾向
5. 其他明顯偏好

用繁體中文回答。"""

    # ─── 保存蒸餾結果 ─────────────────────────

    def save_distilled_knowledge(
        self,
        symbol: str,
        period_start: str,
        period_end: str,
        summary: str,
        key_numbers: str,
        source_count: int,
        original_chars: int,
    ) -> int:
        """保存一條蒸餾知識"""
        if not self._conn:
            return -1

        try:
            now = datetime.utcnow().isoformat()
            # 取得當前版本號
            row = self._conn.execute(
                "SELECT MAX(version) FROM distilled_knowledge WHERE symbol = ?",
                (symbol,),
            ).fetchone()
            version = (row[0] or 0) + 1

            cursor = self._conn.execute(
                """INSERT INTO distilled_knowledge
                   (symbol, period_start, period_end, summary, key_numbers,
                    source_message_count, original_chars, distilled_chars,
                    version, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (symbol, period_start, period_end, summary, key_numbers,
                 source_count, original_chars, len(summary),
                 version, now),
            )
            self._conn.commit()
            return cursor.lastrowid or -1
        except Exception as e:
            logger.error(f"保存蒸餾知識失敗: {e}")
            return -1

    def save_user_profile(self, profile_content: str) -> int:
        """保存/更新使用者分析風格"""
        if not self._conn:
            return -1

        try:
            now = datetime.utcnow().isoformat()
            # 檢查是否已有
            existing = self._conn.execute(
                "SELECT id, version FROM user_profile WHERE profile_type = 'analysis_style'"
            ).fetchone()

            if existing:
                new_version = existing[1] + 1
                self._conn.execute(
                    """UPDATE user_profile
                       SET content = ?, version = ?, updated_at = ?
                       WHERE id = ?""",
                    (profile_content[:MAX_PROFILE_CHARS], new_version, now, existing[0]),
                )
                self._conn.commit()
                return existing[0]
            else:
                cursor = self._conn.execute(
                    """INSERT INTO user_profile (profile_type, content, version, created_at, updated_at)
                       VALUES ('analysis_style', ?, 1, ?, ?)""",
                    (profile_content[:MAX_PROFILE_CHARS], now, now),
                )
                self._conn.commit()
                return cursor.lastrowid or -1
        except Exception as e:
            logger.error(f"保存使用者風格失敗: {e}")
            return -1

    def save_distill_history(
        self, distill_type: str, original_data: str, result: str, tokens: int
    ):
        """保存蒸餾歷史（供回滾）"""
        if not self._conn:
            return
        try:
            now = datetime.utcnow().isoformat()
            self._conn.execute(
                """INSERT INTO distill_history (distill_type, original_data, distilled_result, token_used, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (distill_type, original_data[:50000], result[:10000], tokens, now),
            )
            self._conn.commit()
        except Exception as e:
            logger.warning(f"保存蒸餾歷史失敗: {e}")

    # ─── 讀取蒸餾知識（供注入 LLM context）───────

    def get_context_for_symbol(self, symbol: Optional[str] = None) -> str:
        """
        取得某個幣種的蒸餾知識（用於注入 LLM context）。
        如果超過 MAX_CONTEXT_CHARS，取最新版本。
        """
        if not self._conn:
            return ""

        try:
            parts = []

            # 1. 使用者風格
            profile = self._conn.execute(
                "SELECT content FROM user_profile WHERE profile_type = 'analysis_style' ORDER BY version DESC LIMIT 1"
            ).fetchone()
            if profile:
                parts.append(f"【使用者分析風格】\n{profile[0]}")

            # 2. 幣種知識
            if symbol:
                rows = self._conn.execute(
                    """SELECT summary, period_start, period_end
                       FROM distilled_knowledge
                       WHERE symbol = ?
                       ORDER BY version DESC
                       LIMIT 3""",
                    (symbol,),
                ).fetchall()
                if rows:
                    for summary, ps, pe in rows:
                        parts.append(f"【{symbol} 分析記錄 {ps[:10]}~{pe[:10]}】\n{summary}")

            # 3. 通用知識
            rows = self._conn.execute(
                """SELECT summary, period_start, period_end
                   FROM distilled_knowledge
                   WHERE symbol = '_general'
                   ORDER BY version DESC
                   LIMIT 2"""
            ).fetchall()
            if rows:
                for summary, ps, pe in rows:
                    parts.append(f"【通用分析記錄 {ps[:10]}~{pe[:10]}】\n{summary}")

            if not parts:
                return ""

            context = "\n\n".join(parts)

            # 截斷超長內容
            if len(context) > MAX_CONTEXT_CHARS:
                context = context[:MAX_CONTEXT_CHARS] + "\n... (知識摘要已截斷)"

            return context

        except Exception as e:
            logger.warning(f"讀取蒸餾知識失敗: {e}")
            return ""

    def get_all_knowledge(self) -> list[dict]:
        """取得所有蒸餾知識（供前端展示）"""
        if not self._conn:
            return []
        try:
            rows = self._conn.execute(
                """SELECT id, symbol, period_start, period_end, summary,
                          source_message_count, original_chars, distilled_chars,
                          version, created_at
                   FROM distilled_knowledge
                   ORDER BY created_at DESC"""
            ).fetchall()
            return [
                {
                    "id": r[0], "symbol": r[1],
                    "period_start": r[2], "period_end": r[3],
                    "summary": r[4],
                    "source_messages": r[5],
                    "original_chars": r[6], "distilled_chars": r[7],
                    "compression_ratio": round(r[7] / max(r[6], 1) * 100, 1),
                    "version": r[8], "created_at": r[9],
                }
                for r in rows
            ]
        except Exception as e:
            logger.warning(f"取得蒸餾知識失敗: {e}")
            return []

    def get_user_profile(self) -> Optional[str]:
        """取得使用者風格檔案"""
        if not self._conn:
            return None
        try:
            row = self._conn.execute(
                "SELECT content FROM user_profile WHERE profile_type = 'analysis_style' ORDER BY version DESC LIMIT 1"
            ).fetchone()
            return row[0] if row else None
        except Exception:
            return None


# 全域單例
knowledge_distiller = KnowledgeDistiller()
