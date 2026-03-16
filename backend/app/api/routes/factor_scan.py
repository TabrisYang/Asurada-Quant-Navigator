"""阿斯拉量化系統 — 因子掃描 API

提供一鍵因子掃描功能，不經 LLM，直接計算並返回結果。
"""

from fastapi import APIRouter, Query
from loguru import logger
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


class FactorScanRequest(BaseModel):
    symbol: str
    timeframe: str = "4h"
    forward_period: int = 5
    top_n: int = 5


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
