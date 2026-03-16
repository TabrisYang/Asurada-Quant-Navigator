"""阿斯拉量化系統 — API Key 安全管理器

使用 Fernet 對稱加密儲存 API Key，搭配 Session Token 機制：
1. 使用者提交 API Key → 加密後存入記憶體 → 回傳 session_id
2. 後續請求只需帶 session_id，不再傳輸明文 Key
3. Session 有過期機制（預設 24 小時）
4. Key 永遠不會寫入磁碟或出現在 log 中
"""

import uuid
import time
from typing import Optional

from cryptography.fernet import Fernet
from loguru import logger


class KeyManager:
    """API Key 安全管理器（記憶體內加密儲存）"""

    def __init__(self, session_ttl: int = 86400):
        """
        Args:
            session_ttl: Session 存活時間（秒），預設 24 小時
        """
        # 每次啟動生成新的加密金鑰（不持久化）
        self._fernet = Fernet(Fernet.generate_key())
        self._sessions: dict[str, dict] = {}  # session_id -> {encrypted_key, provider, model, created_at, ...}
        self._session_ttl = session_ttl

    def store_key(
        self,
        api_key: str,
        provider: str,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> str:
        """
        加密儲存 API Key，回傳 session_id

        Args:
            api_key: 明文 API Key（處理完立即丟棄）
            provider: LLM 供應商 ID
            model_name: 模型名稱
            base_url: Ollama base URL

        Returns:
            session_id (UUID string)
        """
        session_id = str(uuid.uuid4())

        # 加密 API Key
        encrypted = self._fernet.encrypt(api_key.encode("utf-8"))

        self._sessions[session_id] = {
            "encrypted_key": encrypted,
            "provider": provider,
            "model_name": model_name,
            "base_url": base_url,
            "created_at": time.time(),
        }

        # 清理過期 session
        self._cleanup_expired()

        logger.info(f"API Key 已加密儲存，session: {session_id[:8]}...")
        return session_id

    def get_key(self, session_id: str) -> Optional[str]:
        """
        透過 session_id 取回解密後的 API Key

        Returns:
            解密後的 API Key，如果 session 無效或過期則回傳 None
        """
        session = self._sessions.get(session_id)
        if not session:
            return None

        # 檢查是否過期
        if time.time() - session["created_at"] > self._session_ttl:
            del self._sessions[session_id]
            logger.info(f"Session 已過期: {session_id[:8]}...")
            return None

        try:
            decrypted = self._fernet.decrypt(session["encrypted_key"])
            return decrypted.decode("utf-8")
        except Exception:
            logger.error(f"API Key 解密失敗: {session_id[:8]}...")
            return None

    def get_session_info(self, session_id: str) -> Optional[dict]:
        """取得 session 的非敏感資訊（不含 Key）"""
        session = self._sessions.get(session_id)
        if not session:
            return None

        if time.time() - session["created_at"] > self._session_ttl:
            del self._sessions[session_id]
            return None

        return {
            "provider": session["provider"],
            "model_name": session.get("model_name"),
            "base_url": session.get("base_url"),
            "created_at": session["created_at"],
            "expires_in": int(self._session_ttl - (time.time() - session["created_at"])),
        }

    def revoke_session(self, session_id: str) -> bool:
        """撤銷（刪除）一個 session"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Session 已撤銷: {session_id[:8]}...")
            return True
        return False

    def _cleanup_expired(self):
        """清除所有過期的 session"""
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s["created_at"] > self._session_ttl
        ]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            logger.info(f"已清除 {len(expired)} 個過期 session")


# 全域單例
key_manager = KeyManager()
