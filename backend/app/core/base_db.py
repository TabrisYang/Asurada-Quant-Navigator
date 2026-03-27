"""阿斯拉量化系統 — DB 基類

統一 DB 連線初始化、WAL 設定、舊路徑遷移邏輯，
避免 chat_history / semantic_cache / prediction_tracker 重複代碼。

支援 SQLite（預設）與 PostgreSQL（透過 DATABASE_URL 環境變數切換）。
"""

import shutil
import sqlite3
from pathlib import Path
from typing import Optional, Union

from loguru import logger

from app.core.config.settings import settings

# PostgreSQL 支援（可選依賴）
try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False


class BaseDBStore:
    """DB 存儲基類，支援 SQLite 和 PostgreSQL"""

    def __init__(self, db_name: str, old_dir: Optional[Path] = None):
        """
        Args:
            db_name: DB 檔名（如 'chat_history.db'，SQLite 模式使用）
            old_dir: 舊版 DB 所在目錄（用於自動遷移），預設為 settings.data_path
        """
        self._db_path = settings.db_path / db_name
        self._conn: Optional[Union[sqlite3.Connection, "psycopg2.extensions.connection"]] = None
        self._old_dir = old_dir or settings.data_path
        self._db_name = db_name
        self._use_postgres = settings.use_postgres

    @property
    def placeholder(self) -> str:
        """SQL 參數佔位符：SQLite 用 ?，PostgreSQL 用 %s"""
        return "%s" if self._use_postgres else "?"

    def _connect(self) -> Optional[Union[sqlite3.Connection, "psycopg2.extensions.connection"]]:
        """建立 DB 連線（自動選擇 SQLite 或 PostgreSQL）"""
        if self._use_postgres:
            return self._connect_postgres()
        return self._connect_sqlite()

    def _connect_sqlite(self) -> Optional[sqlite3.Connection]:
        """建立 SQLite 連線（含遷移 + WAL 設定）"""
        try:
            self._migrate_if_needed()
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=3000")
            return self._conn
        except Exception as e:
            logger.error(f"{self._db_name} SQLite 連線失敗: {e}")
            self._conn = None
            return None

    def _connect_postgres(self) -> Optional["psycopg2.extensions.connection"]:
        """建立 PostgreSQL 連線"""
        if not HAS_PSYCOPG2:
            logger.error("DATABASE_URL 指定 PostgreSQL 但未安裝 psycopg2，回退至 SQLite")
            self._use_postgres = False
            return self._connect_sqlite()
        try:
            self._conn = psycopg2.connect(settings.database_url)
            self._conn.autocommit = False
            logger.info(f"{self._db_name} PostgreSQL 連線成功")
            return self._conn
        except Exception as e:
            logger.error(f"{self._db_name} PostgreSQL 連線失敗: {e}，回退至 SQLite")
            self._use_postgres = False
            return self._connect_sqlite()

    def _migrate_if_needed(self):
        """從舊路徑自動遷移 DB 檔案（僅 SQLite）"""
        old_path = self._old_dir / self._db_name
        if old_path.exists() and old_path.resolve() != self._db_path.resolve():
            if not self._db_path.exists():
                shutil.move(str(old_path), str(self._db_path))
                logger.info(f"遷移 DB: {self._db_name} → {self._db_path}")
                for ext in ("-wal", "-shm"):
                    aux = old_path.parent / f"{self._db_name}{ext}"
                    if aux.exists():
                        shutil.move(str(aux), str(self._db_path.parent / f"{self._db_name}{ext}"))
