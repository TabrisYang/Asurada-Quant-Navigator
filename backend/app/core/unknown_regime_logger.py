"""自動記錄系統判定為 unknown 的 regime 樣本。

用途：
- 累積資料供未來分析「是否有系統性新型態 regime」
- 給 audit_system_health.py 統計近期 unknown 頻率
- 累積 6+ 個月後可考慮 Level 2 auto-classify

存儲：backend/data/db/unknown_regime_log.db
Schema:
  - id INTEGER PRIMARY KEY
  - timestamp TEXT      ← 系統時間
  - symbol TEXT
  - timeframe TEXT
  - confidence REAL
  - tentative_regime TEXT  ← 系統猜測（最接近的 regime label）
  - features_json TEXT     ← 特徵向量（ADX, ATR%, RSI, BB position 等）
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from app.core.config.settings import settings


def _db_path() -> Path:
    p = settings.db_path / "unknown_regime_log.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _ensure_table(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS unknown_regimes (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            symbol          TEXT NOT NULL,
            timeframe       TEXT NOT NULL,
            confidence      REAL,
            tentative_regime TEXT,
            features_json   TEXT
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_unknown_ts ON unknown_regimes(timestamp DESC)"
    )
    conn.commit()


def log_unknown_regime(
    symbol: str,
    timeframe: str,
    regime_info: dict,
    features: Optional[dict] = None,
) -> bool:
    """記錄一筆 unknown regime 樣本。

    只在 confidence < 0.5 時呼叫（由 caller 自行判斷）。
    """
    try:
        conn = sqlite3.connect(_db_path())
        _ensure_table(conn)
        conn.execute(
            "INSERT INTO unknown_regimes "
            "(timestamp, symbol, timeframe, confidence, tentative_regime, features_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                datetime.now().isoformat(),
                symbol,
                timeframe,
                float(regime_info.get("confidence", 0.0)),
                str(regime_info.get("regime", "unknown")),
                json.dumps(features or regime_info.get("details", {}), ensure_ascii=False),
            ),
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.debug(f"unknown_regime_logger 寫入失敗（不影響主流程）: {e}")
        return False


def get_recent_count(days: int = 30) -> int:
    """過去 N 天 unknown regime 累積筆數。"""
    from datetime import timedelta
    db = _db_path()
    if not db.exists():
        return 0
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    try:
        conn = sqlite3.connect(db)
        n = conn.execute(
            "SELECT COUNT(*) FROM unknown_regimes WHERE timestamp >= ?",
            (cutoff,),
        ).fetchone()[0]
        conn.close()
        return int(n)
    except sqlite3.OperationalError:
        return 0
