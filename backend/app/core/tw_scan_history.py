"""阿斯拉量化系統 — 台股掃描歷史儲存

SQLite 儲存每次掃描的參數 + 結果，支援：
- 列出歷次掃描
- 取特定一次的結果
- 「回看」：取當時結果 + 現在價，計算後續漲跌幅
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from app.core.config.settings import settings


_MAX_HISTORY_ROWS = 100  # 自動保留最新 N 筆，超過會刪最舊的


class TwScanHistory:
    """台股掃描歷史存儲（SQLite）。"""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = settings.db_path / "tw_scan_history.db"
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tw_bb_scans (
                scan_id        TEXT PRIMARY KEY,
                scanned_at     TEXT NOT NULL,
                timeframe      TEXT NOT NULL,
                params_json    TEXT NOT NULL,
                results_json   TEXT NOT NULL,
                total_scanned  INTEGER NOT NULL,
                total_found    INTEGER NOT NULL,
                total_fail     INTEGER NOT NULL,
                duration_sec   REAL NOT NULL,
                failures_json  TEXT NOT NULL DEFAULT '[]'
            )
        """)
        # 對舊 DB 做相容 migration（新建 DB 會 no-op）
        try:
            conn.execute(
                "ALTER TABLE tw_bb_scans ADD COLUMN failures_json TEXT NOT NULL DEFAULT '[]'"
            )
        except sqlite3.OperationalError:
            pass  # 欄位已存在
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tw_bb_scans_at "
            "ON tw_bb_scans(scanned_at DESC)"
        )
        conn.commit()
        conn.close()
        logger.info(f"TwScanHistory: 已初始化 {self._db_path}")

    def save(
        self,
        *,
        timeframe: str,
        params: dict,
        results: list[dict],
        total_scanned: int,
        total_found: int,
        total_fail: int,
        duration_sec: float,
        failures: Optional[list[dict]] = None,
    ) -> str:
        """保存一次掃描結果，回傳 scan_id。"""
        scan_id = uuid.uuid4().hex[:12]
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            "INSERT INTO tw_bb_scans "
            "(scan_id, scanned_at, timeframe, params_json, results_json, "
            " total_scanned, total_found, total_fail, duration_sec, failures_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                scan_id,
                datetime.now().isoformat(timespec="seconds"),
                timeframe,
                json.dumps(params, ensure_ascii=False),
                json.dumps(results, ensure_ascii=False),
                total_scanned,
                total_found,
                total_fail,
                duration_sec,
                json.dumps(failures or [], ensure_ascii=False),
            ),
        )
        # 自動清理：只保留最新 _MAX_HISTORY_ROWS 筆
        conn.execute(
            "DELETE FROM tw_bb_scans WHERE scan_id NOT IN ("
            "  SELECT scan_id FROM tw_bb_scans "
            "  ORDER BY scanned_at DESC LIMIT ?"
            ")",
            (_MAX_HISTORY_ROWS,),
        )
        conn.commit()
        conn.close()
        return scan_id

    def list_recent(self, limit: int = 20) -> list[dict]:
        """列出最近 N 次掃描摘要（不含 results）。"""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.execute(
            "SELECT scan_id, scanned_at, timeframe, params_json, "
            "       total_scanned, total_found, total_fail, duration_sec "
            "FROM tw_bb_scans ORDER BY scanned_at DESC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "scan_id": r[0],
                "scanned_at": r[1],
                "timeframe": r[2],
                "params": json.loads(r[3]),
                "total_scanned": r[4],
                "total_found": r[5],
                "total_fail": r[6],
                "duration_sec": r[7],
            }
            for r in rows
        ]

    def get(self, scan_id: str) -> Optional[dict]:
        """取特定一次完整結果。"""
        conn = sqlite3.connect(self._db_path)
        row = conn.execute(
            "SELECT scan_id, scanned_at, timeframe, params_json, results_json, "
            "       total_scanned, total_found, total_fail, duration_sec, failures_json "
            "FROM tw_bb_scans WHERE scan_id = ?",
            (scan_id,),
        ).fetchone()
        conn.close()
        if not row:
            return None
        failures_raw = row[9] if len(row) > 9 else None
        return {
            "scan_id": row[0],
            "scanned_at": row[1],
            "timeframe": row[2],
            "params": json.loads(row[3]),
            "results": json.loads(row[4]),
            "total_scanned": row[5],
            "total_found": row[6],
            "total_fail": row[7],
            "duration_sec": row[8],
            "failures": json.loads(failures_raw) if failures_raw else [],
        }

    def delete(self, scan_id: str) -> bool:
        conn = sqlite3.connect(self._db_path)
        cursor = conn.execute("DELETE FROM tw_bb_scans WHERE scan_id = ?", (scan_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        conn.close()
        return deleted


# 單例
tw_scan_history = TwScanHistory()
