"""阿斯拉量化系統 — K 線圖表數據路由"""

from fastapi import APIRouter, Query
from typing import Optional

from app.data.fetchers.crypto_engine import crypto_engine
from app.data.fetchers.tw_stock_engine import tw_stock_engine
from app.utils.symbol import is_tw_stock
from app.models.schemas import (
    ChartDataRequest,
    ChartDataResponse,
    OHLCVData,
    Timeframe,
)

router = APIRouter()


def _df_to_ohlcv_list(df) -> list[OHLCVData]:
    """將 DataFrame 轉換為 OHLCVData 列表（向量化，避免 iterrows）"""
    if df.empty:
        return []
    import numpy as np

    timestamps = df["timestamp"].astype(str).tolist()
    opens = df["open"].tolist()
    highs = df["high"].tolist()
    lows = df["low"].tolist()
    closes = df["close"].tolist()
    volumes = df["volume"].tolist()

    has_fp = "final_price" in df.columns
    has_ad = "anomaly_detected" in df.columns
    fps = df["final_price"].tolist() if has_fp else None
    ads = df["anomaly_detected"].tolist() if has_ad else None

    result = []
    for i in range(len(timestamps)):
        fp = None
        if fps is not None:
            v = fps[i]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                fp = float(v)
        ad = None
        if ads is not None:
            v = ads[i]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                ad = bool(v)
        result.append(OHLCVData(
            timestamp=timestamps[i],
            open=float(opens[i]),
            high=float(highs[i]),
            low=float(lows[i]),
            close=float(closes[i]),
            volume=float(volumes[i]),
            final_price=fp,
            anomaly_detected=ad,
        ))
    return result


@router.get("/available/list")
async def list_available_data():
    """列出所有本地可用的數據（必須在 {symbol} 路由之前）"""
    return {"data": crypto_engine.list_available_data() + tw_stock_engine.list_available_data()}


@router.get("/tw-stock-search")
async def search_tw_stock(q: str = Query(default="")):
    """搜尋台股：支援代碼或中文名稱（TWSE/TPEx 官方 + 靜態表）"""
    if not q or len(q) < 1:
        return {"results": []}
    q = q.strip()
    results = []

    # 先從 TWSE 名稱庫搜尋（最完整）
    from app.api.routes.data_sync import _tw_name_cache, _load_twse_name_db
    if len(_tw_name_cache) < 100:
        await _load_twse_name_db()

    for code, name in _tw_name_cache.items():
        if q in code or q in name:
            results.append({"code": code, "name": name, "symbol": f"{code}/TWD"})
        if len(results) >= 20:
            break

    # 補充靜態表（可能有 TWSE 沒有的）
    if len(results) < 20:
        from app.data.tw_sectors import TW_STOCK_NAMES
        for code, name in TW_STOCK_NAMES.items():
            if q in code or q in name:
                if not any(r["code"] == code for r in results):
                    results.append({"code": code, "name": name, "symbol": f"{code}/TWD"})

    return {"results": results[:20]}


@router.get("/data")
async def get_chart_data(
    symbol: str = Query(default="BTC/USDT"),
    timeframe: Timeframe = Query(default=Timeframe.D1),
    start_date: Optional[str] = Query(default=None),
    end_date: Optional[str] = Query(default=None),
):
    """取得指定幣種的 K 線數據（優先讀本地）。symbol 用 query param 避免 URL 斜線問題"""
    engine = tw_stock_engine if is_tw_stock(symbol) else crypto_engine
    df = engine.load_local_data(
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
    engine = tw_stock_engine if is_tw_stock(request.symbol) else crypto_engine
    df = engine.load_local_data(
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
