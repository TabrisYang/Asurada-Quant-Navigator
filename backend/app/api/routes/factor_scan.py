"""阿斯拉量化系統 — 因子掃描 API

提供一鍵因子掃描功能，不經 LLM，直接計算並返回結果。
"""

from fastapi import APIRouter, Query
from loguru import logger
from pydantic import BaseModel
from typing import Optional

import numpy as np

router = APIRouter()


class FactorScanRequest(BaseModel):
    symbol: str
    timeframe: str = "4h"
    forward_period: int = 5
    top_n: int = 5


class ConditionPreview(BaseModel):
    factor: str
    operator: str = ">"
    value: float = 0
    value2: Optional[float] = None
    enabled: bool = True


class TriggerPreviewRequest(BaseModel):
    symbol: str
    timeframe: str = "4h"
    conditions: list[ConditionPreview]


@router.post("/scan")
async def factor_scan(request: FactorScanRequest):
    """執行因子掃描：計算所有因子的近期 IC、Alpha Decay、組合 IC 等。"""
    from app.data.fetchers.crypto_engine import crypto_engine
    from app.core.backtest.factor_analysis import run_factor_scan

    try:
        df = crypto_engine.load_local_data(request.symbol, request.timeframe)
        if df.empty or len(df) < 70:
            return {
                "status": "error",
                "message": f"數據不足（{len(df)} 根 K 線），至少需要 70 根。請先同步數據。",
            }

        logger.info(
            f"因子掃描 [{request.symbol} {request.timeframe}]: "
            f"{len(df)} 根 K 線, forward={request.forward_period}"
        )

        result = run_factor_scan(
            df=df,
            timeframe=request.timeframe,
            top_n=request.top_n,
            forward_period=request.forward_period,
        )
        result["symbol"] = request.symbol

        logger.info(
            f"因子掃描完成 [{request.symbol}]: "
            f"掃描 {result.get('total_factors_scanned', 0)} 個因子, "
            f"有效 {result.get('effective_count', 0)} 個"
        )

        return result

    except Exception as e:
        logger.error(f"因子掃描失敗: {e}")
        return {"status": "error", "message": str(e)}


@router.post("/trigger-preview")
async def trigger_preview(request: TriggerPreviewRequest):
    """快速計算自訂條件組合的歷史觸發次數（不做完整回測）。"""
    from app.data.fetchers.crypto_engine import crypto_engine
    from app.core.indicators import registry

    try:
        df = crypto_engine.load_local_data(request.symbol, request.timeframe)
        if df.empty or len(df) < 30:
            return {"status": "error", "message": "數據不足"}

        enabled = [c for c in request.conditions if c.enabled]
        if not enabled:
            return {"status": "success", "trigger_count": len(df), "total_bars": len(df),
                    "trigger_pct": 100.0, "conditions_used": 0}

        masks: list[np.ndarray] = []
        calc_cache: dict[str, dict] = {}

        for cond in enabled:
            factor = cond.factor
            # 因子名稱格式可能是 "indicator_series" 如 "bb_BB_Width"
            parts = factor.split("_", 1)
            indicator_id = parts[0].lower()
            series_hint = parts[1] if len(parts) > 1 else None

            if indicator_id not in calc_cache:
                calc_result = registry.calculate(indicator_id, df)
                calc_cache[indicator_id] = calc_result or {}
            calc_result = calc_cache[indicator_id]

            if not calc_result:
                continue

            # 找匹配的 series
            if series_hint and series_hint in calc_result:
                values = np.array(calc_result[series_hint], dtype=float)
            elif factor in calc_result:
                values = np.array(calc_result[factor], dtype=float)
            else:
                # 取第一個 series
                values = np.array(list(calc_result.values())[0], dtype=float)

            op = cond.operator
            val = cond.value
            if op == ">":
                mask = values > val
            elif op == "<":
                mask = values < val
            elif op == ">=":
                mask = values >= val
            elif op == "<=":
                mask = values <= val
            elif op == "between" and cond.value2 is not None:
                mask = (values >= val) & (values <= cond.value2)
            else:
                mask = values > val

            mask = np.where(np.isnan(values), False, mask)
            masks.append(mask)

        if not masks:
            return {"status": "success", "trigger_count": len(df), "total_bars": len(df),
                    "trigger_pct": 100.0, "conditions_used": 0}

        combined = masks[0]
        for m in masks[1:]:
            combined = combined & m

        trigger_count = int(np.sum(combined))
        total = len(df)

        # 最近一次觸發的時間
        last_trigger_idx = None
        if trigger_count > 0:
            indices = np.where(combined)[0]
            last_trigger_idx = int(indices[-1])

        last_trigger_time = None
        if last_trigger_idx is not None and "timestamp" in df.columns:
            last_trigger_time = str(df["timestamp"].iloc[last_trigger_idx])

        return {
            "status": "success",
            "trigger_count": trigger_count,
            "total_bars": total,
            "trigger_pct": round(trigger_count / total * 100, 2) if total > 0 else 0,
            "conditions_used": len(enabled),
            "last_trigger_time": last_trigger_time,
        }

    except Exception as e:
        logger.error(f"觸發預覽失敗: {e}")
        return {"status": "error", "message": str(e)}
