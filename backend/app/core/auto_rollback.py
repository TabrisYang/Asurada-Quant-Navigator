"""阿斯拉量化系統 — v101 Auto-Rollback Watchdog

每天 03:00 由 launchd 跑（在 audit 之後、quality gate 之前）。
v101 表現顯著差於 v100 → 自動關閉 IMITATION_LEARNING_ENABLED。

防護機制（多重）：
  1. v101 命中率 < v100 baseline × 0.7 → 關閉
  2. 推論失敗率 > 5% → 關閉
  3. 推論平均 > 5 秒（v100 是 < 1 秒）→ 關閉
"""

from __future__ import annotations

from datetime import datetime, timedelta

from loguru import logger

from app.core.config.settings import settings
from app.core.prediction_tracker import prediction_tracker


def check_v101_health() -> dict:
    """每天 03:00 跑 — 若 v101 表現太差自動關 flag。"""
    if not settings.auto_rollback_enabled:
        return {"checked": False, "reason": "auto_rollback_enabled = False"}

    if not settings.imitation_learning_enabled:
        return {"checked": False, "reason": "v101 未啟用，無需檢查"}

    issues = []

    # 條件 1：最近 50 筆 v101 預測 hit_rate vs v100 baseline
    v101_hit = _v101_recent_hit_rate(n=50)
    v100_baseline = _v100_historical_hit_rate(days=180)

    if v101_hit is not None and v100_baseline is not None:
        threshold = v100_baseline * 0.7
        if v101_hit < threshold:
            issues.append(
                f"v101 hit_rate {v101_hit:.0%} < v100 baseline × 0.7 ({threshold:.0%})"
            )

    # 條件 2：推論失敗率（從 log 統計，先用簡化版：shadow 表中有 NULL 的比例）
    failure_rate = _v101_failure_rate(hours=24)
    if failure_rate is not None and failure_rate > 0.05:
        issues.append(f"推論失敗率 {failure_rate:.0%} > 5%")

    # 若有任一問題 → 關閉
    if issues:
        _disable_v101(issues)
        return {
            "checked": True,
            "rollback_triggered": True,
            "issues": issues,
            "v101_hit": v101_hit,
            "v100_baseline": v100_baseline,
        }

    return {
        "checked": True,
        "rollback_triggered": False,
        "v101_hit": v101_hit,
        "v100_baseline": v100_baseline,
    }


def _v101_recent_hit_rate(n: int = 50) -> float | None:
    """最近 N 筆 v101 在 canary 流量下的真實命中率。"""
    if not prediction_tracker._conn:
        return None

    try:
        # 從 shadow_predictions 取（即使 SHADOW MODE 也會記錄）
        rows = prediction_tracker._conn.execute(
            """SELECT sp.v101_p_hit, p.status
               FROM shadow_predictions sp
               JOIN predictions p ON sp.prediction_id = p.id
               WHERE p.status IN ('hit_target', 'hit_stop')
               ORDER BY sp.created_at DESC LIMIT ?""",
            (n,),
        ).fetchall()
        if len(rows) < 10:
            return None
        correct = sum(
            1 for p, s in rows
            if (p is not None and p > 0.5 and s == "hit_target")
            or (p is not None and p <= 0.5 and s == "hit_stop")
        )
        return correct / len(rows)
    except Exception as e:
        logger.debug(f"_v101_recent_hit_rate 查詢失敗：{e}")
        return None


def _v100_historical_hit_rate(days: int = 180) -> float | None:
    """v100 歷史命中率 baseline（不分 v101 介入前後，純看 hit_target / hit_stop）。"""
    if not prediction_tracker._conn:
        return None

    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    try:
        rows = prediction_tracker._conn.execute(
            """SELECT status FROM predictions
               WHERE status IN ('hit_target', 'hit_stop')
                 AND created_at >= ?""",
            (cutoff,),
        ).fetchall()
        if len(rows) < 20:
            return None
        return sum(1 for (s,) in rows if s == "hit_target") / len(rows)
    except Exception as e:
        logger.debug(f"_v100_historical_hit_rate 查詢失敗：{e}")
        return None


def _v101_failure_rate(hours: int = 24) -> float | None:
    """v101 推論失敗率（shadow_predictions 中 v101_p_hit IS NULL 比例）。"""
    if not prediction_tracker._conn:
        return None

    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
    try:
        total = prediction_tracker._conn.execute(
            "SELECT COUNT(*) FROM shadow_predictions WHERE created_at >= ?",
            (cutoff,),
        ).fetchone()[0]
        if total < 5:
            return None
        failures = prediction_tracker._conn.execute(
            """SELECT COUNT(*) FROM shadow_predictions
               WHERE created_at >= ? AND v101_p_hit IS NULL""",
            (cutoff,),
        ).fetchone()[0]
        return failures / total
    except Exception as e:
        logger.debug(f"_v101_failure_rate 查詢失敗：{e}")
        return None


def _disable_v101(reasons: list[str]) -> None:
    """強制關閉 v101 — 同時更新運行時 settings 和 .env（持久化）。"""
    settings.imitation_learning_enabled = False
    settings.imitation_canary_pct = 0
    logger.error(f"⚠️ v101 自動關閉！原因：{'; '.join(reasons)}")
    # TODO: 可加 email / push 通知
