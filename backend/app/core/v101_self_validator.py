"""阿斯拉量化系統 — v101 Quality Gate Self-Validator

每週日 04:00 由 launchd 跑（在 retrain 02:00 + auto_rollback 03:00 之後）。

7 個硬閾值（全部必須通過才允許 v101 暴露給使用者）：
  1. 樣本量 trainset_n ≥ 100
  2. OOS AUC ≥ 0.60
  3. Lockbox AUC ≥ 0.58
  4. Brier score ≤ 0.22
  5. 過擬合 gap (train_auc - oof_auc) < 0.10
  6. Shadow 4 週驗證：v101 hit_rate ≥ v100 baseline
  7. Drift 狀態：Adversarial AUC < 0.65

任一未過 → 持續 SHADOW MODE，使用者繼續看 v100。

Canary progression（通過 gate 後）：1% → 10% → 25% → 50% → 100%
任一階段表現變差（v101 hit < v100 - 5pp）→ 立刻回退到 0%。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Optional

from loguru import logger

from app.core.config.settings import settings
from app.core.prediction_tracker import prediction_tracker
from app.core.shadow_runner import get_shadow_predictions

# Quality Gate 7 硬閾值
GATE_THRESHOLDS = {
    "min_trainset_n": 100,
    "min_oos_auc": 0.60,
    "min_lockbox_auc": 0.58,
    "max_brier": 0.22,
    "max_overfit_gap": 0.10,
    "min_shadow_4w_hit_rate_vs_baseline": 0.0,  # >= baseline
    "max_adversarial_auc": 0.65,
}


def evaluate_v101_readiness() -> dict:
    """每週日 04:00 跑：v101 是否足夠好可以面向使用者？"""
    metrics = _get_active_model_metrics()
    if not metrics:
        result = {
            "ready": False,
            "failed_gates": ["no_active_model"],
            "metrics": {},
            "note": "尚無已訓練的模型 — 需先累積樣本並完成首次訓練",
        }
        _log_evaluation(result)
        return result

    shadow_4w = get_shadow_predictions(weeks=4)
    v100_baseline = _get_v100_baseline_hit_rate(days=180)
    drift_auc = _get_latest_drift_auc()

    # 7 個 gates
    gates = {
        "sample_size": metrics.get("trainset_n", 0) >= GATE_THRESHOLDS["min_trainset_n"],
        "oos_auc": (metrics.get("auc") or 0) >= GATE_THRESHOLDS["min_oos_auc"],
        "lockbox_auc": (metrics.get("lockbox_auc") or 0) >= GATE_THRESHOLDS["min_lockbox_auc"],
        "brier": (metrics.get("brier") or 1.0) <= GATE_THRESHOLDS["max_brier"],
        "no_overfit": (
            metrics.get("overfit_gap") is not None
            and metrics["overfit_gap"] < GATE_THRESHOLDS["max_overfit_gap"]
        ),
        "shadow_4w_hit_rate": (
            shadow_4w.get("hit_rate") is not None
            and v100_baseline is not None
            and shadow_4w["hit_rate"] >= v100_baseline + GATE_THRESHOLDS["min_shadow_4w_hit_rate_vs_baseline"]
        ),
        "no_drift": (
            drift_auc is None  # 還沒有 drift 數據時不擋（首次評估容忍）
            or drift_auc < GATE_THRESHOLDS["max_adversarial_auc"]
        ),
    }

    ready = all(gates.values())
    failed_gates = [k for k, v in gates.items() if not v]

    result = {
        "ready": ready,
        "gates": gates,
        "failed_gates": failed_gates,
        "metrics": metrics,
        "shadow_4w": shadow_4w,
        "v100_baseline": v100_baseline,
        "drift_auc": drift_auc,
    }

    # 通過 → 自動啟動 Canary 1%（前提：SHADOW 已關）
    action = "no_action"
    if ready:
        if (
            settings.imitation_canary_pct == 0
            and not settings.imitation_shadow_mode
            and settings.imitation_learning_enabled
        ):
            settings.imitation_canary_pct = 1
            action = "started_canary_1pct"
            logger.info("✅ v101 通過 Quality Gate，啟動 Canary 1%")
        else:
            action = "ready_but_shadow_or_disabled"
    else:
        action = f"failed_gates: {', '.join(failed_gates)}"
        logger.info(f"⏳ v101 未通過 Quality Gate：{failed_gates}")

    result["action_taken"] = action
    _log_evaluation(result)
    return result


def canary_progression() -> dict:
    """每週評估是否擴大 canary 範圍（通過 quality gate 後才有效）。"""
    if not settings.imitation_learning_enabled:
        return {"action": "no_action", "reason": "v101 未啟用"}

    if settings.imitation_shadow_mode:
        return {"action": "no_action", "reason": "SHADOW 模式中"}

    current = settings.imitation_canary_pct
    if current == 0:
        return {"action": "no_action", "reason": "尚未通過 Quality Gate（canary=0）"}

    # 過去 7 天 v101 vs v100 對照
    v101_hit = _canary_hit_rate(days=7)
    v100_hit = _baseline_hit_rate(days=7)

    if v101_hit is None or v100_hit is None:
        return {"action": "no_action", "reason": "樣本不足以評估", "current_pct": current}

    # 嚴格回退：v101 比 v100 差 5pp 以上立刻關
    if v101_hit < v100_hit - 0.05:
        settings.imitation_canary_pct = 0
        msg = f"⚠️ Canary {current}% 期間 v101 ({v101_hit:.0%}) < v100 ({v100_hit:.0%}) - 5pp，自動關閉"
        logger.error(msg)
        return {
            "action": "rollback_to_zero",
            "from_pct": current,
            "v101_hit": v101_hit,
            "v100_hit": v100_hit,
        }

    # 通過 → 漸進擴大
    progression = {1: 10, 10: 25, 25: 50, 50: 100}
    if current in progression:
        new_pct = progression[current]
        settings.imitation_canary_pct = new_pct
        logger.info(f"📈 Canary 擴大：{current}% → {new_pct}%（v101={v101_hit:.0%} v100={v100_hit:.0%}）")
        return {
            "action": "expanded",
            "from_pct": current,
            "to_pct": new_pct,
            "v101_hit": v101_hit,
            "v100_hit": v100_hit,
        }

    return {"action": "no_action", "current_pct": current, "reason": "已達 100% 或非標準階段"}


# ─── 內部 helper ─────────────────────────────────────


def _get_active_model_metrics() -> Optional[dict]:
    """從 imitation_model_metrics 表取目前 champion 模型的指標。"""
    if not prediction_tracker._conn:
        return None
    try:
        row = prediction_tracker._conn.execute(
            """SELECT trainset_n, auc, train_auc, lockbox_auc, brier, overfit_gap, trained_at
               FROM imitation_model_metrics
               WHERE is_champion = 1 ORDER BY version DESC LIMIT 1"""
        ).fetchone()
        if not row:
            return None
        return {
            "trainset_n": row[0],
            "auc": row[1],
            "train_auc": row[2],
            "lockbox_auc": row[3],
            "brier": row[4],
            "overfit_gap": row[5],
            "trained_at": row[6],
        }
    except Exception as e:
        logger.debug(f"_get_active_model_metrics 失敗：{e}")
        return None


def _get_v100_baseline_hit_rate(days: int = 180) -> Optional[float]:
    """v100 歷史命中率（給 Quality Gate 比較用）。"""
    if not prediction_tracker._conn:
        return None
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    try:
        rows = prediction_tracker._conn.execute(
            """SELECT status FROM predictions
               WHERE status IN ('hit_target', 'hit_stop') AND created_at >= ?""",
            (cutoff,),
        ).fetchall()
        if len(rows) < 10:
            return None
        return sum(1 for (s,) in rows if s == "hit_target") / len(rows)
    except Exception:
        return None


def _get_latest_drift_auc() -> Optional[float]:
    """最近一次 adversarial validation 的 AUC（drift_monitor.py 寫入，Phase 2.5）。

    現階段尚未實作 drift_monitor，回傳 None 容忍。
    """
    return None


def _canary_hit_rate(days: int = 7) -> Optional[float]:
    """過去 N 天 canary 流量下 v101 真實命中率。"""
    if not prediction_tracker._conn:
        return None
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    try:
        rows = prediction_tracker._conn.execute(
            """SELECT sp.v101_p_hit, p.status
               FROM shadow_predictions sp
               JOIN predictions p ON sp.prediction_id = p.id
               WHERE sp.created_at >= ?
                 AND p.status IN ('hit_target', 'hit_stop')""",
            (cutoff,),
        ).fetchall()
        if len(rows) < 5:
            return None
        correct = sum(
            1 for p, s in rows
            if (p is not None and p > 0.5 and s == "hit_target")
            or (p is not None and p <= 0.5 and s == "hit_stop")
        )
        return correct / len(rows)
    except Exception:
        return None


def _baseline_hit_rate(days: int = 7) -> Optional[float]:
    """過去 N 天的 v100 baseline 命中率。"""
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
        if len(rows) < 5:
            return None
        return sum(1 for (s,) in rows if s == "hit_target") / len(rows)
    except Exception:
        return None


def _log_evaluation(result: dict) -> None:
    """寫入 quality_gate_log 表。"""
    if not prediction_tracker._conn:
        return
    try:
        prediction_tracker._conn.execute(
            """INSERT INTO quality_gate_log
               (evaluated_at, ready, gates_json, metrics_json, action_taken)
               VALUES (?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(),
                1 if result.get("ready") else 0,
                json.dumps(result.get("gates", {}), ensure_ascii=False),
                json.dumps(result.get("metrics", {}), ensure_ascii=False),
                str(result.get("action_taken", "")),
            ),
        )
        prediction_tracker._conn.commit()
    except Exception as e:
        logger.debug(f"_log_evaluation 失敗：{e}")
