"""阿斯拉量化系統 — 預測驗證引擎

對比預測價格與實際 K 線數據，判定預測結果。
計算 MFE (Max Favorable Excursion) 和 MAE (Max Adverse Excursion)。
"""

from datetime import datetime

from loguru import logger

from app.utils.timezone import taipei_now

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

    now = taipei_now()
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

    # 驗證後觸發自動調整
    if validated_count > 0:
        try:
            from app.core.auto_adjuster import run_adjustment_cycle
            adj_result = run_adjustment_cycle()
            results["adjustments"] = adj_result
        except Exception as e:
            logger.warning(f"[自動調整] 失敗: {e}")

    return {"validated": validated_count, **results}


def validate_for_symbol(symbol: str) -> dict:
    """驗證特定幣種的所有 active 預測。"""
    active = prediction_tracker.get_active(symbol)
    if not active:
        return {"validated": 0}

    now = taipei_now()
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
    hit_idx = None
    if hit_target_idx is not None and hit_stop_idx is not None:
        if hit_target_idx <= hit_stop_idx:
            status = "hit_target"
            hit_idx = hit_target_idx
            note = f"先觸及目標價（第 {hit_target_idx+1} 根K線），之後也觸及止損"
        else:
            status = "hit_stop"
            hit_idx = hit_stop_idx
            note = f"先觸及止損（第 {hit_stop_idx+1} 根K線），之後也觸及目標"
    elif hit_target_idx is not None:
        status = "hit_target"
        hit_idx = hit_target_idx
        note = f"觸及目標價（第 {hit_target_idx+1} 根K線）"
    elif hit_stop_idx is not None:
        status = "hit_stop"
        hit_idx = hit_stop_idx
        note = f"觸及止損（第 {hit_stop_idx+1} 根K線）"
    elif is_expired:
        status = "expired"
        note = f"持倉期結束，最終盈虧 {final_pct:+.2f}%"
    else:
        return "still_active"

    # 取得實際觸及的 K 線時間（非驗證程式執行時間）
    hit_at = None
    if hit_idx is not None and "timestamp" in df.columns:
        ts = df.iloc[hit_idx]["timestamp"]
        hit_at = str(ts)

    # 里程碑追蹤：25%/50%/75% 目標達成情況
    milestones_str = ""
    target_distance = abs(target - entry)
    if target_distance > 0 and entry > 0:
        target_pct = target_distance / entry * 100
        m25 = mfe_pct >= target_pct * 0.25
        m50 = mfe_pct >= target_pct * 0.50
        m75 = mfe_pct >= target_pct * 0.75
        milestones_str = f"25%={'Y' if m25 else 'N'} 50%={'Y' if m50 else 'N'} 75%={'Y' if m75 else 'N'}"

    prediction_tracker.update_validation(
        pred["id"],
        status=status,
        actual_outcome_pct=round(final_pct, 2),
        max_favorable_pct=round(mfe_pct, 2),
        max_adverse_pct=round(mae_pct, 2),
        note=note,
        hit_at=hit_at,
    )

    # 存儲里程碑數據
    if milestones_str:
        try:
            prediction_tracker._ensure_db()
            prediction_tracker._conn.execute(
                "UPDATE predictions SET milestones=? WHERE id=?",
                (milestones_str, pred["id"]),
            )
            prediction_tracker._conn.commit()
        except Exception:
            pass

    # 自動教訓提取：高信心預測觸及止損 → 存入知識碎片
    if status == "hit_stop" and pred.get("confidence") == "high":
        try:
            from app.core.knowledge_fragments import fragment_store
            lesson = (
                f"高信心{direction}預測觸及止損 | "
                f"regime={pred.get('regime', '?')} | "
                f"indicators={pred.get('indicators', '?')} | "
                f"MFE={mfe_pct:+.1f}% MAE={mae_pct:+.1f}% | "
                f"milestones={milestones_str}"
            )
            fragment_store.store_fragment(
                content=lesson,
                fragment_type="lesson",
                symbol=pred["symbol"],
                source_question="auto_validation",
            )
            logger.info(f"自動教訓提取: 預測 #{pred['id']} {lesson[:60]}...")
        except Exception as le:
            logger.debug(f"自動教訓提取失敗: {le}")

    logger.info(f"預測 #{pred['id']} {pred['symbol']} {direction}: {status} (MFE={mfe_pct:+.1f}% MAE={mae_pct:+.1f}% {milestones_str})")
    return status
