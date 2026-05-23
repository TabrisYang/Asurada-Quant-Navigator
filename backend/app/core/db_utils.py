"""阿斯拉量化系統 — SQLite 連線工具。

統一以 WAL + busy_timeout 開啟共享 DB（特別是 predictions.db）。

背景（根因修復）：predictions.db 被 9+ 元件並發存取，原本用預設 rollback journal
（寫鎖擋所有讀/寫）且無 busy_timeout → 某執行緒持鎖時 DB 操作被卡 → 串流路徑在事件迴圈上
搶同一鎖無限等待 → 整個後端凍死。WAL 讓讀者不擋寫者、寫者不擋讀者；busy_timeout 確保
競爭時最多等 N 毫秒後拋錯（釋放鎖），永不無限卡。
"""

import sqlite3
from pathlib import Path
from typing import Union


def open_sqlite(
    db_path: Union[str, Path],
    *,
    busy_timeout_ms: int = 30000,
    check_same_thread: bool = True,
    row_factory: bool = False,
) -> sqlite3.Connection:
    """以 WAL + busy_timeout 開啟 SQLite 連線。

    - journal_mode=WAL：per-DB 持久設定，第一個用此 helper 開的連線即把整個 DB 轉 WAL，
      之後所有連線（含其他程序）讀寫不再互鎖。
    - busy_timeout：per-connection，競爭時最多等 N 毫秒後拋 'database is locked'，不無限卡。
    """
    conn = sqlite3.connect(
        str(db_path),
        check_same_thread=check_same_thread,
        timeout=busy_timeout_ms / 1000.0,
    )
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
    except Exception:
        pass
    if row_factory:
        conn.row_factory = sqlite3.Row
    return conn
