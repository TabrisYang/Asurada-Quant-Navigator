"""阿斯拉量化系統 — 預測驗證引擎

對比預測價格與實際 K 線數據，判定預測結果。
計算 MFE (Max Favorable Excursion) 和 MAE (Max Adverse Excursion)。
"""

from datetime import datetime
from typing import Optional

from loguru import logger

from app.core.prediction_tracker import prediction_tracker
from app.data.fetchers.crypto_engine import crypto_engine


def _infer_candle_tf(timeframe_hours: int) -> str:
    """根據持倉時長推斷驗證用的 K 線級別。"""
    if timeframe_hours <= 12:
        return "15m"
    elif timeframe_hours <= 72:
        return "1h"
    elif timeframe_hours <= 168:
        return "4h"
    return "1d"


def validate_all_active() -> dict:
    """驗證所有 active 預測，返回統計摘要。"""
    active = prediction_tracker.get_active()
    if not active:
        return {"validated": 0, "message": "沒有待驗證的預測"}

    now = datetime.now()
    validated_count = 0
    results = {"hit_target": 0, "hit_stop": 0, "expired": 0, "still_active": 0, "no_data": 0}

    for pred in active:
        try:
            result = _validate_one(pred, now)
            if result == "still_active":
                results["still_active"] += 1
            elif result == "no_data":
                results["no_data"] += 1
            else:
                results[result] = results.get(result, 0) + 1
                validated_count += 1
        except Exception as e:
            logger.error(f"驗證預測 #{pred['id']} 失敗: {e}")

    logger.info(
        f"預測驗證完成: {validated_count} 筆已驗證 "
        f"(命中={results['hit_target']}, 止損={results['hit_stop']}, "
        f"過期={results['expired']}, 仍活躍={results['still_active']})"
    )
    return {"validated": validated_count, **results}


def validate_for_symbol(symbol: str) -> dict:
    """驗證特定幣種的所有 active 預測。"""
    active = prediction_tracker.get_active(symbol)
    if not active:
        return {"validated": 0}

    now = datetime.now()
    count = 0
    for pred in active:
        try:
            result = _validate_one(pred, now)
            if result not in ("still_active", "no_data"):
                count += 1
        except Exception as e:
            logger.error(f"驗證預測 #{pred['id']} 失敗: {e}")
    return {"validated": count}


def _validate_one(pred: dict, now: datetime) -> str:
    """驗證單一預測。返回 status 或 'still_active'/'no_data'。"""
    expires_at = datetime.fromisoformat(pred["expires_at"])
    created_at = datetime.fromisoformat(pred["created_at"])
    is_expired = now > expires_at

    candle_tf = _infer_candle_tf(pred["timeframe_hours"])
    df = crypto_engine.load_local_data(
        pred["symbol"],
        candle_tf,
        created_at.strftime("%Y-%m-%d"),
        now.strftime("%Y-%m-%d"),
    )

    if df.empty or len(df) < 2:
        if is_expired:
            prediction_tracker.update_validation(
                pred["id"], "expired", 0.0, 0.0, 0.0,
                "數據不足，無法驗證，標記為過期"
            )
            return "expired"
        return "no_data"

    entry = pred["entry_price"]
    target = pred["target_price"]
    stop = pred["stop_price"]
    direction = pred["direction"]

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    if direction == "long":
        mfe_pct = float((max(highs) - entry) / entry * 100)
        mae_pct = float((min(lows) - entry) / entry * 100)
        final_pct = float((closes[-1] - entry) / entry * 100)

        hit_target_idx = None
        hit_stop_idx = None
        for i in range(len(df)):
            if hit_target_idx is None and highs[i] >= target:
                hit_target_idx = i
            if hit_stop_idx is None and lows[i] <= stop:
                hit_stop_idx = i
    else:  # short
        mfe_pct = float((entry - min(lows)) / entry * 100)
        mae_pct = float((entry - max(highs)) / entry * 100)
        final_pct = float((entry - closes[-1]) / entry * 100)

        hit_target_idx = None
        hit_stop_idx = None
        for i in range(len(df)):
            if hit_target_idx is None and lows[i] <= target:
                hit_target_idx = i
            if hit_stop_idx is None and highs[i] >= stop:
                hit_stop_idx = i

    # 判定結果：先碰到哪個就算哪個
    if hit_target_idx is not None and hit_stop_idx is not None:
        if hit_target_idx <= hit_stop_idx:
            status = "hit_target"
            note = f"先觸及目標價（第 {hit_target_idx+1} 根K線），之後也觸及止損"
        else:
            status = "hit_stop"
            note = f"先觸及止損（第 {hit_stop_idx+1} 根K線），之後也觸及目標"
    elif hit_target_idx is not None:
        status = "hit_target"
        note = f"觸及目標價（第 {hit_target_idx+1} 根K線）"
    elif hit_stop_idx is not None:
        status = "hit_stop"
        note = f"觸及止損（第 {hit_stop_idx+1} 根K線）"
    elif is_expired:
        status = "expired"
        note = f"持倉期結束，最終盈虧 {final_pct:+.2f}%"
    else:
        return "still_active"

    prediction_tracker.update_validation(
        pred["id"],
        status=status,
        actual_outcome_pct=round(final_pct, 2),
        max_favorable_pct=round(mfe_pct, 2),
        max_adverse_pct=round(mae_pct, 2),
        note=note,
    )
    logger.info(f"預測 #{pred['id']} {pred['symbol']} {direction}: {status} (MFE={mfe_pct:+.1f}% MAE={mae_pct:+.1f}%)")
    return status
