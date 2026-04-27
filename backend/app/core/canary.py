"""阿斯拉量化系統 — v101 Canary 分流（含 Quality Gate 守衛）

決定每次推論時要不要把 v101 結果暴露給使用者。

四層守衛（必須全通過才會 use_v101 = True）：
  1. settings.imitation_learning_enabled = True
  2. settings.imitation_shadow_mode = False
  3. v101_passes_quality_gate() = True（7 個硬閾值）
  4. random < settings.imitation_canary_pct%

任一未過 → 使用者看到 100% v100 體驗。
"""

from __future__ import annotations

import random
from typing import Optional

from loguru import logger

from app.core.config.settings import settings


def use_v101(symbol: Optional[str] = None) -> bool:
    """判斷本次推論是否該把 v101 結果暴露給使用者。

    符合「永不變壞」鐵律：任一條件失敗 → 直接 False，使用者用 v100。
    """
    # 1. 主開關必須開
    if not settings.imitation_learning_enabled:
        return False

    # 2. SHADOW 模式期間永遠 False（v101 偷跑但不暴露）
    if settings.imitation_shadow_mode:
        return False

    # 3. Quality Gate 7 硬閾值
    if settings.quality_gate_enabled and not v101_passes_quality_gate():
        return False

    # 4. Canary 分流（漸進啟用 1% → 100%）
    canary_pct = max(0, min(100, settings.imitation_canary_pct))
    if canary_pct == 0:
        return False
    return random.random() * 100 < canary_pct


def v101_passes_quality_gate() -> bool:
    """快取版的 Quality Gate 結果查詢 — 直接讀 quality_gate_log 表。

    不在這裡重算 gates（每週 04:00 由 v101_self_validator 評估），
    避免每次推論都重跑 adversarial validation 等重操作。
    """
    from app.core.prediction_tracker import prediction_tracker

    if not prediction_tracker._conn:
        return False

    try:
        row = prediction_tracker._conn.execute(
            "SELECT ready FROM quality_gate_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return False
        return bool(row[0])
    except Exception as e:
        logger.debug(f"v101_passes_quality_gate 查詢失敗：{e}")
        return False


def get_canary_status() -> dict:
    """給 PredictionDashboard / API 看的 canary 狀態摘要。"""
    return {
        "imitation_learning_enabled": settings.imitation_learning_enabled,
        "shadow_mode": settings.imitation_shadow_mode,
        "canary_pct": settings.imitation_canary_pct,
        "quality_gate_passed": v101_passes_quality_gate(),
        "active_for_users": (
            settings.imitation_learning_enabled
            and not settings.imitation_shadow_mode
            and v101_passes_quality_gate()
            and settings.imitation_canary_pct > 0
        ),
    }
