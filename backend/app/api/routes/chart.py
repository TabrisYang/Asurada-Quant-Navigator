"""阿斯拉量化系統 — K 線圖表數據路由"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.data.fetchers.crypto_engine import crypto_engine
from app.models.schemas import (
    ChartDataRequest,
    ChartDataResponse,
    OHLCVData,
    Timeframe,
)

router = APIRouter()


def _df_to_ohlcv_list(df) -> list[OHLCVData]:
    """將 DataFrame 轉換為 OHLCVData 列表"""
    if df.empty:
        return []
    result = []
    for _, row in df.iterrows():
        item = OHLCVData(
            timestamp=str(row["timestamp"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
            final_price=float(row["final_price"]) if "final_price" in row and row.get("final_price") is not None else None,
            anomaly_detected=bool(row["anomaly_detected"]) if "anomaly_detected" in row else None,
        )
        result.append(item)
    return result


@router.get("/available/list")
async def list_available_data():
    """列出所有本地可用的數據（必須在 {symbol} 路由之前）"""
    return {"data": crypto_engine.list_available_data()}


@router.get("/data")
async def get_chart_data(
    symbol: str = Query(default="BTC/USDT"),
    timeframe: Timeframe = Query(default=Timeframe.D1),
    start_date: Optional[str] = Query(default=None),
    end_date: Optional[str] = Query(default=None),
):
    """取得指定幣種的 K 線數據（優先讀本地）。symbol 用 query param 避免 URL 斜線問題"""
    df = crypto_engine.load_local_data(
        symbol=symbol,
        timeframe=timeframe.value,
        start_date=start_date,
        end_date=end_date,
    )

    return ChartDataResponse(
        symbol=symbol,
        timeframe=timeframe.value,
        ohlcv=_df_to_ohlcv_list(df),
    )


@router.post("/")
async def query_chart_data(request: ChartDataRequest):
    """POST 方式取得 K 線數據"""
    df = crypto_engine.load_local_data(
        symbol=request.symbol,
        timeframe=request.timeframe.value,
        start_date=request.start_date,
        end_date=request.end_date,
    )

    return ChartDataResponse(
        symbol=request.symbol,
        timeframe=request.timeframe.value,
        ohlcv=_df_to_ohlcv_list(df),
    )
