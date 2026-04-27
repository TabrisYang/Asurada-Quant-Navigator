"""阿斯拉量化系統 — v101 Shadow Mode 執行器

Shadow 偷偷跑 v101 預測但**不影響使用者輸出**。失敗永不冒泡。

設計原則：
- 主線：v100 既有邏輯產生使用者看到的輸出
- Shadow：v101 在背景跑，結果寫入 shadow_predictions 表給驗證用
- 任何 shadow 錯誤 → 只 log，使用者完全無感
- Quality Gate 看的就是 shadow 累積的 4 週數據
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Optional

from loguru import logger

from app.core.config.settings import settings
from app.core.prediction_tracker import prediction_tracker


def maybe_run_shadow(
    name: str,
    fn: Callable,
    *args,
    record_kwargs: Optional[dict] = None,
    **kwargs,
) -> Optional[Any]:
    """在 SHADOW MODE 啟用時偷跑 fn(*args, **kwargs) 並記錄結果。

    Returns: fn 的回傳值（給呼叫端要的話可參考），失敗 None。

    主線**不應該依賴**這個回傳值 — shadow 失敗使用者不能感受到。
    """
    if not settings.imitation_shadow_mode:
        return None

    try:
        result = fn(*args, **kwargs)
        if record_kwargs is not None:
            _record_shadow_result(name, result, record_kwargs)
        return result
    except Exception as e:
        logger.warning(f"[shadow:{name}] 失敗（不影響使用者）: {type(e).__name__}: {e}")
        return None


def _record_shadow_result(name: str, result: Any, record_kwargs: dict) -> None:
    """寫入 shadow_predictions 表給 quality gate 驗證用。"""
    if not isinstance(result, dict):
        return
    if not prediction_tracker._conn:
        return
    try:
        prediction_tracker._conn.execute(
            """INSERT INTO shadow_predictions
               (prediction_id, v100_p_hit, v101_p_hit, v101_p_ml, v101_p_rule,
                v101_blend_alpha, v101_mode, model_version, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record_kwargs.get("prediction_id"),
                record_kwargs.get("v100_p_hit"),
                result.get("p_hit_target"),
                result.get("p_ml"),
                result.get("p_rule"),
                result.get("blend_alpha"),
                result.get("mode"),
                str(result.get("model_version", "")),
                datetime.now().isoformat(),
            ),
        )
        prediction_tracker._conn.commit()
    except Exception as e:
        logger.debug(f"[shadow:{name}] 寫入失敗（容忍）: {e}")


def get_shadow_predictions(weeks: int = 4) -> dict:
    """取最近 N 週 shadow 預測的統計（給 Quality Gate 用）。"""
    from datetime import timedelta

    if not prediction_tracker._conn:
        return {"n": 0, "hit_rate": None, "note": "DB 未初始化"}

    cutoff = (datetime.now() - timedelta(weeks=weeks)).isoformat()
    try:
        rows = prediction_tracker._conn.execute(
            """SELECT sp.v101_p_hit, p.status
               FROM shadow_predictions sp
               JOIN predictions p ON sp.prediction_id = p.id
               WHERE sp.created_at >= ?
                 AND p.status IN ('hit_target', 'hit_stop')""",
            (cutoff,),
        ).fetchall()
    except Exception as e:
        logger.warning(f"get_shadow_predictions 失敗：{e}")
        return {"n": 0, "hit_rate": None}

    if not rows:
        return {"n": 0, "hit_rate": None, "note": f"近 {weeks} 週尚無已驗證的 shadow 預測"}

    # v101 預測 P > 0.5 視為「預測會 hit_target」
    correct = sum(
        1 for p, status in rows
        if (p is not None and p > 0.5 and status == "hit_target")
        or (p is not None and p <= 0.5 and status == "hit_stop")
    )
    return {
        "n": len(rows),
        "hit_rate": round(correct / len(rows), 3),
    }
