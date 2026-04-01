"""阿斯拉量化系統 — 情境預測路由"""

from fastapi import APIRouter, HTTPException, Query

from app.data.fetchers.crypto_engine import crypto_engine
from app.data.fetchers.tw_stock_engine import tw_stock_engine
from app.utils.symbol import is_tw_stock
from app.models.schemas import ScenarioRequest
from app.core.ml.scenario_predictor import scenario_predictor

router = APIRouter()

# 簡易快取（記憶體內）
_scenario_cache: dict[str, dict] = {}


def _load_data(symbol: str, timeframe: str):
    """載入本地數據"""
    engine = tw_stock_engine if is_tw_stock(symbol) else crypto_engine
    return engine.load_local_data(symbol=symbol, timeframe=timeframe)


@router.post("/predict")
async def predict_scenarios(request: ScenarioRequest):
    """產出三大情境預測"""
    df = _load_data(request.symbol, request.timeframe.value)

    if df is None or df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"找不到 {request.symbol} {request.timeframe.value} 的本地數據，請先同步數據",
        )

    if len(df) < 30:
        raise HTTPException(
            status_code=400,
            detail=f"數據不足：需要至少 30 根 K 線，目前僅有 {len(df)} 根",
        )

    try:
        result = scenario_predictor.predict_scenarios(
            df=df,
            symbol=request.symbol,
            timeframe=request.timeframe.value,
            forward_bars=request.forward_bars,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"情境預測失敗: {str(e)}")

    # 快取結果
    cache_key = f"{request.symbol}_{request.timeframe.value}"
    _scenario_cache[cache_key] = result.to_dict()

    return result.to_dict()


@router.get("/latest/{symbol}")
async def get_latest_scenarios(
    symbol: str,
    timeframe: str = Query(default="1d"),
):
    """取得最新快取的情境預測（不重新計算）"""
    cache_key = f"{symbol}_{timeframe}"
    if cache_key in _scenario_cache:
        return _scenario_cache[cache_key]

    # 快取不存在，即時計算
    df = _load_data(symbol, timeframe)
    if df is None or df.empty or len(df) < 30:
        raise HTTPException(
            status_code=404,
            detail=f"找不到 {symbol} {timeframe} 的足夠數據",
        )

    result = scenario_predictor.predict_scenarios(
        df=df, symbol=symbol, timeframe=timeframe,
    )
    _scenario_cache[cache_key] = result.to_dict()
    return result.to_dict()
