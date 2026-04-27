"""阿斯拉量化系統 — v101 Champion-Challenger 模型管理（Phase 2.4）

維護三個模型版本：
  - Champion：當前正在 production 跑的版本
  - Challenger：本週新訓練、待 promote 的版本
  - Stable Fallback：3 個月前驗證 robust 的版本（緊急退路）

自動切換規則：
  - 新模型 oof/lockbox AUC > champion + 0.02 → promote
  - champion 4 週 AUC 低於 stable_fallback - 0.05 → rollback to stable_fallback
  - champion 8 週連續退化 → 強制停用 ML（auto_rollback）
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from app.core.config.settings import settings
from app.core.prediction_tracker import prediction_tracker


def get_status() -> dict:
    """三個模型版本 + canary + quality gate 完整狀態。"""
    prediction_tracker._ensure_db()
    if not prediction_tracker._conn:
        return {"error": "DB not initialized"}

    rows = prediction_tracker._conn.execute(
        """SELECT version, trained_at, trainset_n, auc, lockbox_auc, brier,
                  status, is_champion, is_stable_fallback, feature_importance
           FROM imitation_model_metrics ORDER BY version DESC LIMIT 20"""
    ).fetchall()

    history = []
    champion = None
    stable_fallback = None
    challenger = None  # 最近一個非 champion / 非 fallback 的成功訓練版本

    for r in rows:
        d = dict(r)
        # parse feature_importance
        if d.get("feature_importance"):
            try:
                import json
                d["feature_importance"] = json.loads(d["feature_importance"])
            except Exception:
                d["feature_importance"] = None
        history.append(d)
        if d["is_champion"]:
            champion = d
        elif d["is_stable_fallback"]:
            stable_fallback = d
        elif d["status"] == "activated" and challenger is None:
            challenger = d

    # 最新 quality gate 評估
    qg_row = prediction_tracker._conn.execute(
        "SELECT * FROM quality_gate_log ORDER BY id DESC LIMIT 1"
    ).fetchone()
    last_gate = dict(qg_row) if qg_row else None

    from app.core.canary import get_canary_status
    canary = get_canary_status()

    return {
        "champion": champion,
        "challenger": challenger,
        "stable_fallback": stable_fallback,
        "history": history,
        "canary": canary,
        "last_quality_gate": last_gate,
    }


def manual_rollback_to_stable() -> dict:
    """1-click 回到 stable_fallback。緊急時用。"""
    prediction_tracker._ensure_db()
    if not prediction_tracker._conn:
        return {"status": "error", "message": "DB 未初始化"}

    fallback = prediction_tracker._conn.execute(
        "SELECT version FROM imitation_model_metrics WHERE is_stable_fallback = 1 ORDER BY version DESC LIMIT 1"
    ).fetchone()
    if not fallback:
        # 沒有 stable_fallback → 直接停用 ML
        settings.imitation_learning_enabled = False
        settings.imitation_canary_pct = 0
        return {
            "status": "no_stable_fallback",
            "action": "disabled_v101_completely",
            "message": "無 stable_fallback 可退回，已停用 v101",
        }

    target_version = fallback[0]
    # 把當前 champion 標記為 deprecated，stable_fallback 變 champion
    prediction_tracker._conn.execute("UPDATE imitation_model_metrics SET is_champion = 0")
    prediction_tracker._conn.execute(
        "UPDATE imitation_model_metrics SET is_champion = 1, is_stable_fallback = 0 WHERE version = ?",
        (target_version,),
    )
    prediction_tracker._conn.commit()

    # 強制重新載入 imitation_predictor
    try:
        from app.core.imitation_predictor import imitation_predictor
        imitation_predictor._reload()
    except Exception:
        pass

    logger.warning(f"⚠️ 手動回退至 stable_fallback v{target_version}")
    return {
        "status": "ok",
        "action": "rolled_back",
        "to_version": target_version,
    }


def disable_v101() -> dict:
    """1-click 停用 v101，所有 user 看 v100。"""
    settings.imitation_learning_enabled = False
    settings.imitation_canary_pct = 0
    logger.warning("⚠️ v101 已手動停用")
    return {
        "status": "ok",
        "action": "disabled",
        "message": "v101 已停用，所有使用者看到 v100 體驗",
    }
