"""阿斯拉量化系統 — v106 A2：持倉追蹤器（純本地，不接交易所 API）。

讓使用者輸入實際持倉，系統把分析跟個人組合連動：
- 「ADA 多單佔總部位 30%，目前 lean_short → 建議減至 15%」
- 「BTC 跌破止損 $90k，你的持倉風險上升」
- 「整體組合 60% 偏多，應分散方向」

純本地 SQLite，**不上雲、不接交易所**（規避法規 + 金鑰風險）。
使用者手動輸入，定期更新。

Schema:
    positions:
        id, symbol, direction, size_usd, avg_entry_price,
        stop_loss, take_profit, opened_at, updated_at, notes
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger

from app.utils.timezone import taipei_now

_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "db" / "positions.db"


class PositionTracker:
    """持倉追蹤單例。使用者透過 API 增刪改查。"""

    def __init__(self):
        self._conn: Optional[sqlite3.Connection] = None
        import threading
        self._lock = threading.Lock()

    def _ensure_db(self):
        if self._conn is not None:
            return
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                size_usd REAL NOT NULL,
                avg_entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                leverage REAL DEFAULT 1.0,
                opened_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                notes TEXT DEFAULT '',
                status TEXT DEFAULT 'open'
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_pos_symbol ON positions(symbol, status)")
        self._conn.commit()
        logger.info(f"PositionTracker 已初始化：{_DB_PATH}")

    # ─── 增刪改查 ─────────────────────────

    def add(
        self,
        symbol: str,
        direction: str,
        size_usd: float,
        avg_entry_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        leverage: float = 1.0,
        notes: str = "",
    ) -> int:
        """新增一筆持倉，回傳 id。"""
        with self._lock:
            self._ensure_db()
            assert self._conn is not None
            now = taipei_now().isoformat()
            cursor = self._conn.execute(
                """INSERT INTO positions
                   (symbol, direction, size_usd, avg_entry_price, stop_loss, take_profit,
                    leverage, opened_at, updated_at, notes, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')""",
                (symbol, direction, size_usd, avg_entry_price, stop_loss, take_profit,
                 leverage, now, now, notes),
            )
            self._conn.commit()
            pid = cursor.lastrowid or -1
            logger.info(f"[Position] 新增 #{pid} {symbol} {direction} ${size_usd:.0f}")
            return pid

    def update(self, pid: int, **fields) -> bool:
        """更新欄位。允許的 key：size_usd / avg_entry_price / stop_loss / take_profit /
        leverage / notes / status。"""
        with self._lock:
            self._ensure_db()
            assert self._conn is not None
            allowed = {
                "size_usd", "avg_entry_price", "stop_loss", "take_profit",
                "leverage", "notes", "status",
            }
            updates = {k: v for k, v in fields.items() if k in allowed}
            if not updates:
                return False
            updates["updated_at"] = taipei_now().isoformat()
            sets = ", ".join(f"{k}=?" for k in updates)
            params = list(updates.values()) + [pid]
            cur = self._conn.execute(f"UPDATE positions SET {sets} WHERE id=?", params)
            self._conn.commit()
            return cur.rowcount > 0

    def close(self, pid: int) -> bool:
        """標 status=closed（不真正刪除，保留歷史）。"""
        return self.update(pid, status="closed")

    def delete(self, pid: int) -> bool:
        """真刪除（非典型操作；通常用 close）。"""
        with self._lock:
            self._ensure_db()
            assert self._conn is not None
            cur = self._conn.execute("DELETE FROM positions WHERE id=?", (pid,))
            self._conn.commit()
            return cur.rowcount > 0

    def list_open(self, symbol: Optional[str] = None) -> list[dict]:
        """列出所有 open positions（可篩 symbol）。"""
        with self._lock:
            self._ensure_db()
            assert self._conn is not None
            if symbol:
                rows = self._conn.execute(
                    "SELECT * FROM positions WHERE status='open' AND symbol=? ORDER BY opened_at DESC",
                    (symbol,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM positions WHERE status='open' ORDER BY opened_at DESC"
                ).fetchall()
            return [dict(r) for r in rows]

    def list_all(self, limit: int = 100) -> list[dict]:
        """列出全部（含已關）。"""
        with self._lock:
            self._ensure_db()
            assert self._conn is not None
            rows = self._conn.execute(
                "SELECT * FROM positions ORDER BY opened_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ─── 摘要 / 統計 ─────────────────────────

    def get_summary(self) -> dict:
        """回傳整體組合摘要：總曝險、各 symbol 比重、淨多空比、平均更新天數。"""
        with self._lock:
            self._ensure_db()
            assert self._conn is not None
            rows = self._conn.execute(
                "SELECT * FROM positions WHERE status='open'"
            ).fetchall()
            positions = [dict(r) for r in rows]

        if not positions:
            return {
                "total_positions": 0,
                "total_exposure_usd": 0.0,
                "by_symbol": {},
                "long_short_ratio": None,
                "freshness_warning": None,
            }

        total = sum(p["size_usd"] for p in positions)
        long_total = sum(p["size_usd"] for p in positions if p["direction"] == "long")
        short_total = sum(p["size_usd"] for p in positions if p["direction"] == "short")

        by_symbol: dict[str, dict] = {}
        for p in positions:
            sym = p["symbol"]
            if sym not in by_symbol:
                by_symbol[sym] = {
                    "long_usd": 0.0, "short_usd": 0.0,
                    "net_usd": 0.0, "pct_of_total": 0.0,
                    "positions": [],
                }
            if p["direction"] == "long":
                by_symbol[sym]["long_usd"] += p["size_usd"]
                by_symbol[sym]["net_usd"] += p["size_usd"]
            else:
                by_symbol[sym]["short_usd"] += p["size_usd"]
                by_symbol[sym]["net_usd"] -= p["size_usd"]
            by_symbol[sym]["positions"].append({
                "id": p["id"], "direction": p["direction"], "size_usd": p["size_usd"],
                "entry": p["avg_entry_price"], "sl": p["stop_loss"], "tp": p["take_profit"],
            })

        for sym in by_symbol:
            total_for_sym = by_symbol[sym]["long_usd"] + by_symbol[sym]["short_usd"]
            by_symbol[sym]["pct_of_total"] = round(total_for_sym / total * 100, 1) if total > 0 else 0

        # 過期警示：最久沒更新的 position
        try:
            now = taipei_now()
            oldest_update = min(
                datetime.fromisoformat(p["updated_at"]) for p in positions
            )
            days_old = (now - oldest_update).days
            freshness_warning = (
                f"最舊一筆持倉 {days_old} 天沒更新，建議重新核對"
                if days_old > 7 else None
            )
        except Exception:
            freshness_warning = None

        return {
            "total_positions": len(positions),
            "total_exposure_usd": round(total, 2),
            "long_total_usd": round(long_total, 2),
            "short_total_usd": round(short_total, 2),
            "long_short_ratio": (
                round(long_total / short_total, 2) if short_total > 0 else None
            ),
            "by_symbol": by_symbol,
            "freshness_warning": freshness_warning,
        }

    def get_positions_for_symbol(self, symbol: str) -> dict:
        """供 chat_state 注入用 — 給 LLM 看當前 symbol 的持倉狀態。"""
        positions = self.list_open(symbol=symbol)
        if not positions:
            return {"has_position": False}

        long_usd = sum(p["size_usd"] for p in positions if p["direction"] == "long")
        short_usd = sum(p["size_usd"] for p in positions if p["direction"] == "short")
        net_usd = long_usd - short_usd

        # 整體組合裡的佔比
        summary = self.get_summary()
        total = summary.get("total_exposure_usd", 0)
        pct_of_portfolio = round((long_usd + short_usd) / total * 100, 1) if total > 0 else 0

        return {
            "has_position": True,
            "symbol": symbol,
            "long_usd": round(long_usd, 2),
            "short_usd": round(short_usd, 2),
            "net_usd": round(net_usd, 2),
            "net_direction": "long" if net_usd > 0 else "short" if net_usd < 0 else "neutral",
            "pct_of_portfolio": pct_of_portfolio,
            "positions": [
                {
                    "id": p["id"],
                    "direction": p["direction"],
                    "size_usd": p["size_usd"],
                    "avg_entry": p["avg_entry_price"],
                    "stop_loss": p["stop_loss"],
                    "take_profit": p["take_profit"],
                    "leverage": p["leverage"],
                    "opened_at": p["opened_at"],
                }
                for p in positions
            ],
        }


# Module-level singleton
position_tracker = PositionTracker()
