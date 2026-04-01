"""阿斯拉量化系統 — 預測追蹤器

從 LLM 回答中提取結構化預測，存入 SQLite，
定期驗證預測結果，並生成績效統計回饋給 LLM。

預測生命週期：active → hit_target / hit_stop / expired
"""

import math
import re
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from loguru import logger

from app.core.config.settings import settings
from app.utils.timezone import taipei_now

_PREDICTIONS_PATTERN = re.compile(
    r"---PREDICTIONS---\s*\n(.*?)(?:\n---END_PREDICTIONS---|$)",
    re.DOTALL,
)

_PREDICTION_LINE = re.compile(
    r"-\s*\[direction:(long|short)\]\s+"
    r"entry=([\d.]+)\s+"
    r"target=([\d.]+)\s+"
    r"stop=([\d.]+)\s+"
    r"timeframe=(\S+)\s+"
    r"confidence=(high|medium|low)\s+"
    r"regime=(\S+)\s+"
    r"indicators=([\w,]+)"
    r"(?:\s+invalidation=(.+))?"
)


def _sanitize_prediction_block(block: str) -> str:
    """預處理預測區塊，提高正則匹配容錯性。"""
    # 移除數字中的逗號：83,500 → 83500
    block = re.sub(r"(\d),(\d)", r"\1\2", block)
    # direction / confidence 統一小寫
    block = re.sub(r"direction:(\w+)", lambda m: f"direction:{m.group(1).lower()}", block)
    block = re.sub(r"confidence=(\w+)", lambda m: f"confidence={m.group(1).lower()}", block)
    # 指標列表去空格：RSI, MACD → RSI,MACD
    block = re.sub(
        r"indicators=([\w,\s]+?)(\s+invalidation=|\s*$)",
        lambda m: f"indicators={m.group(1).replace(' ', '')}{m.group(2)}",
        block,
        flags=re.MULTILINE,
    )
    return block


def parse_predictions(llm_response: str) -> list[dict]:
    """從 LLM 回答中解析 PREDICTIONS 區塊。"""
    match = _PREDICTIONS_PATTERN.search(llm_response)
    if not match:
        return []

    block = _sanitize_prediction_block(match.group(1))
    results = []
    for m in _PREDICTION_LINE.finditer(block):
        try:
            tf_str = m.group(5)
            hours = _parse_timeframe_hours(tf_str)
            results.append({
                "direction": m.group(1),
                "entry_price": float(m.group(2)),
                "target_price": float(m.group(3)),
                "stop_price": float(m.group(4)),
                "timeframe_str": tf_str,
                "timeframe_hours": hours,
                "confidence": m.group(6),
                "regime": m.group(7),
                "indicators": m.group(8),
                "invalidation": m.group(9) or "",
            })
        except (ValueError, IndexError) as e:
            logger.warning(f"解析預測行失敗: {e}")
    return results


def strip_predictions(text: str) -> str:
    """移除 PREDICTIONS 區塊，不顯示給使用者。"""
    return re.sub(
        r"\n?---PREDICTIONS---.*?---END_PREDICTIONS---\n?",
        "",
        text,
        flags=re.DOTALL,
    ).strip()


def _parse_timeframe_hours(tf: str) -> int:
    """將 '48h', '7d' 等轉為小時數。"""
    tf = tf.lower().strip()
    if tf.endswith("h"):
        return int(tf[:-1])
    elif tf.endswith("d"):
        return int(tf[:-1]) * 24
    elif tf.endswith("w"):
        return int(tf[:-1]) * 168
    return 72  # default 3 days


class PredictionTracker:
    """預測存儲與查詢。"""

    def __init__(self):
        self._conn: Optional[sqlite3.Connection] = None

    def _ensure_db(self):
        if self._conn:
            return
        db_file = settings.db_path / "predictions.db"
        self._conn = sqlite3.connect(str(db_file), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                target_price REAL NOT NULL,
                stop_price REAL NOT NULL,
                timeframe_hours INTEGER NOT NULL,
                confidence TEXT NOT NULL,
                regime TEXT NOT NULL,
                indicators TEXT NOT NULL,
                source_question TEXT,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                validated_at TEXT,
                actual_outcome_pct REAL,
                max_favorable_pct REAL,
                max_adverse_pct REAL,
                validation_note TEXT
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_pred_symbol_status
            ON predictions(symbol, status)
        """)
        # 遷移：加入 ml_enhanced 欄位（若不存在）
        try:
            self._conn.execute(
                "ALTER TABLE predictions ADD COLUMN ml_enhanced INTEGER DEFAULT 0"
            )
        except sqlite3.OperationalError:
            pass  # 欄位已存在

        # 遷移：加入 notes 欄位（若不存在）
        try:
            self._conn.execute(
                "ALTER TABLE predictions ADD COLUMN notes TEXT DEFAULT ''"
            )
        except sqlite3.OperationalError:
            pass  # 欄位已存在

        # 遷移：加入 invalidation 欄位（若不存在）
        try:
            self._conn.execute(
                "ALTER TABLE predictions ADD COLUMN invalidation TEXT DEFAULT ''"
            )
        except sqlite3.OperationalError:
            pass  # 欄位已存在

        # 遷移：加入 milestones 欄位（若不存在）
        try:
            self._conn.execute(
                "ALTER TABLE predictions ADD COLUMN milestones TEXT DEFAULT ''"
            )
        except sqlite3.OperationalError:
            pass  # 欄位已存在
        self._conn.commit()

    def store(
        self,
        symbol: str,
        timeframe: str,
        prediction: dict,
        source_question: str = "",
    ) -> int:
        """存入一筆預測，返回 id。"""
        self._ensure_db()
        now = taipei_now()
        hours = prediction["timeframe_hours"]
        expires = now + timedelta(hours=hours)

        # 多時間框架矛盾檢測
        existing = self.get_active(symbol)
        conflicts = [
            p for p in existing
            if p["direction"] != prediction["direction"]
        ]
        if conflicts:
            conflict_info = ", ".join(
                f"{p['timeframe']} {p['direction']}" for p in conflicts
            )
            logger.warning(
                f"預測方向衝突: 新={prediction['direction']} {timeframe} "
                f"vs 現有={conflict_info}（{symbol}）"
            )

        cursor = self._conn.execute(
            """INSERT INTO predictions
               (symbol, timeframe, direction, entry_price, target_price, stop_price,
                timeframe_hours, confidence, regime, indicators, invalidation,
                source_question, created_at, expires_at, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')""",
            (
                symbol, timeframe,
                prediction["direction"],
                prediction["entry_price"],
                prediction["target_price"],
                prediction["stop_price"],
                hours,
                prediction["confidence"],
                prediction["regime"],
                prediction["indicators"],
                prediction.get("invalidation", ""),
                source_question,
                now.isoformat(),
                expires.isoformat(),
            ),
        )
        self._conn.commit()
        pid = cursor.lastrowid
        logger.info(
            f"預測已儲存 #{pid}: {symbol} {prediction['direction']} "
            f"entry={prediction['entry_price']} target={prediction['target_price']} "
            f"stop={prediction['stop_price']} expires={expires.isoformat()}"
        )
        return pid

    def get_active(self, symbol: Optional[str] = None) -> list[dict]:
        """取得尚未驗證的預測。"""
        self._ensure_db()
        if symbol:
            rows = self._conn.execute(
                "SELECT * FROM predictions WHERE status='active' AND symbol=? ORDER BY created_at DESC",
                (symbol,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM predictions WHERE status='active' ORDER BY created_at DESC",
            ).fetchall()
        return [dict(r) for r in rows]

    def get_validated(
        self,
        symbol: Optional[str] = None,
        regime: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """取得已驗證的預測（支援按 symbol 和 regime 過濾）。"""
        self._ensure_db()
        query = "SELECT * FROM predictions WHERE status != 'active'"
        params = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if regime:
            query += " AND regime = ?"
            params.append(regime)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def update_validation(
        self,
        pred_id: int,
        status: str,
        actual_outcome_pct: float,
        max_favorable_pct: float,
        max_adverse_pct: float,
        note: str = "",
        hit_at: str | None = None,
    ):
        """更新預測的驗證結果。

        hit_at: 實際觸及目標/止損的 K 線時間。若未提供則用當前時間。
        """
        self._ensure_db()
        validated_time = hit_at if hit_at else taipei_now().isoformat()
        self._conn.execute(
            """UPDATE predictions
               SET status=?, validated_at=?, actual_outcome_pct=?,
                   max_favorable_pct=?, max_adverse_pct=?, validation_note=?
               WHERE id=?""",
            (
                status,
                validated_time,
                actual_outcome_pct,
                max_favorable_pct,
                max_adverse_pct,
                note,
                pred_id,
            ),
        )
        self._conn.commit()

    def clear_all(self, symbol: Optional[str] = None) -> int:
        """清除所有預測紀錄。若提供 symbol 則只清除該幣對。"""
        self._ensure_db()
        if symbol:
            cursor = self._conn.execute(
                "DELETE FROM predictions WHERE symbol = ?", (symbol,),
            )
        else:
            cursor = self._conn.execute("DELETE FROM predictions")
        self._conn.commit()
        count = cursor.rowcount
        logger.info(f"已清除 {count} 筆預測紀錄" + (f"（{symbol}）" if symbol else ""))
        return count

    def get_stats(
        self,
        symbol: Optional[str] = None,
        regime: Optional[str] = None,
        days: int = 90,
    ) -> dict:
        """計算預測績效統計（帶時間衰減權重）。"""
        self._ensure_db()
        cutoff = (taipei_now() - timedelta(days=days)).isoformat()
        query = "SELECT * FROM predictions WHERE status != 'active' AND created_at > ?"
        params: list = [cutoff]
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if regime:
            query += " AND regime = ?"
            params.append(regime)

        rows = self._conn.execute(query, params).fetchall()
        if not rows:
            return {"total": 0, "message": "尚無已驗證的預測記錄"}

        preds = [dict(r) for r in rows]
        now = taipei_now()
        decay_lambda = 0.02  # half-life ~35 days

        hit_target_w = 0.0
        hit_stop_w = 0.0
        expired_w = 0.0
        total_w = 0.0
        outcomes = []
        indicator_wins = {}
        indicator_total = {}
        confidence_hits = {"high": [0, 0], "medium": [0, 0], "low": [0, 0]}

        for p in preds:
            days_ago = (now - datetime.fromisoformat(p["created_at"])).days
            weight = math.exp(-decay_lambda * days_ago)

            total_w += weight
            is_win = p["status"] == "hit_target"

            if p["status"] == "hit_target":
                hit_target_w += weight
            elif p["status"] == "hit_stop":
                hit_stop_w += weight
            else:
                expired_w += weight

            if p["actual_outcome_pct"] is not None:
                outcomes.append(p["actual_outcome_pct"])

            # per-indicator stats
            for ind in (p.get("indicators") or "").split(","):
                ind = ind.strip()
                if not ind:
                    continue
                indicator_total[ind] = indicator_total.get(ind, 0) + 1
                if is_win:
                    indicator_wins[ind] = indicator_wins.get(ind, 0) + 1

            conf = p.get("confidence", "medium")
            if conf in confidence_hits:
                confidence_hits[conf][1] += 1
                if is_win:
                    confidence_hits[conf][0] += 1

        win_rate = hit_target_w / total_w if total_w > 0 else 0
        outcomes_arr = np.array(outcomes) if outcomes else np.array([0])

        # indicator win rates (≥1 sample, low-sample marked)
        indicator_stats = {}
        for ind, total in indicator_total.items():
            if total >= 1:
                wins = indicator_wins.get(ind, 0)
                indicator_stats[ind] = {
                    "win_rate": round(wins / total * 100, 1),
                    "samples": total,
                }

        # confidence calibration
        conf_calibration = {}
        for conf, (wins, total) in confidence_hits.items():
            if total >= 2:
                conf_calibration[conf] = {
                    "win_rate": round(wins / total * 100, 1),
                    "samples": total,
                }

        return {
            "total": len(preds),
            "win_rate_weighted": round(win_rate * 100, 1),
            "hit_target": sum(1 for p in preds if p["status"] == "hit_target"),
            "hit_stop": sum(1 for p in preds if p["status"] == "hit_stop"),
            "expired": sum(1 for p in preds if p["status"] == "expired"),
            "avg_outcome_pct": round(float(np.mean(outcomes_arr)), 2),
            "median_outcome_pct": round(float(np.median(outcomes_arr)), 2),
            "best_outcome_pct": round(float(np.max(outcomes_arr)), 2),
            "worst_outcome_pct": round(float(np.min(outcomes_arr)), 2),
            "indicator_performance": indicator_stats,
            "confidence_calibration": conf_calibration,
            "decay_halflife_days": 35,
            "sample_sufficient": len(preds) >= 8,
        }


    def get_direction_stats(
        self, symbol: Optional[str] = None, days: int = 90,
    ) -> dict[str, dict]:
        """取得多/空方向各自的勝率與樣本數。"""
        self._ensure_db()
        cutoff = (taipei_now() - timedelta(days=days)).isoformat()
        query = "SELECT direction, status FROM predictions WHERE status != 'active' AND created_at > ?"
        params: list = [cutoff]
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        rows = self._conn.execute(query, params).fetchall()
        result: dict[str, dict] = {}
        for d in ("long", "short"):
            dir_rows = [r for r in rows if r["direction"] == d]
            total = len(dir_rows)
            if total == 0:
                continue
            wins = sum(1 for r in dir_rows if r["status"] == "hit_target")
            result[d] = {
                "win_rate": round(wins / total * 100, 1),
                "samples": total,
            }
        return result

    def get_recent_streak(
        self, symbol: Optional[str] = None, n: int = 10,
    ) -> dict:
        """取得最近 N 筆預測的連勝/連敗資訊。"""
        self._ensure_db()
        query = (
            "SELECT status FROM predictions "
            "WHERE status != 'active'"
        )
        params: list = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(n)

        rows = self._conn.execute(query, params).fetchall()
        if not rows:
            return {"total": 0}

        statuses = [r["status"] for r in rows]
        recent_wins = sum(1 for s in statuses if s == "hit_target")
        recent_losses = sum(1 for s in statuses if s == "hit_stop")

        current_streak = 0
        streak_type = statuses[0] if statuses else None
        for s in statuses:
            if s == streak_type:
                current_streak += 1
            else:
                break

        return {
            "total": len(statuses),
            "recent_wins": recent_wins,
            "recent_losses": recent_losses,
            "current_streak": current_streak,
            "streak_type": streak_type,
        }

    # ─── 策略日誌 / 覆盤 ──────────────────

    def update_note(self, pred_id: int, note: str):
        """更新預測的使用者筆記"""
        self._ensure_db()
        self._conn.execute(
            "UPDATE predictions SET notes = ? WHERE id = ?",
            (note, pred_id),
        )
        self._conn.commit()

    def get_review_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> dict:
        """取得覆盤所需的資料（某時間範圍內的所有已驗證預測）"""
        self._ensure_db()
        query = "SELECT * FROM predictions WHERE status != 'active'"
        params: list = []
        if start_date:
            query += " AND created_at >= ?"
            params.append(start_date)
        if end_date:
            query += " AND created_at <= ?"
            params.append(end_date)
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        query += " ORDER BY created_at DESC"
        rows = self._conn.execute(query, params).fetchall()
        predictions = [dict(r) for r in rows]

        total = len(predictions)
        wins = sum(1 for p in predictions if p["status"] == "hit_target")
        losses = sum(1 for p in predictions if p["status"] == "hit_stop")

        return {
            "predictions": predictions,
            "summary": {
                "total": total,
                "wins": wins,
                "losses": losses,
                "expired": total - wins - losses,
                "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
            },
        }


prediction_tracker = PredictionTracker()
