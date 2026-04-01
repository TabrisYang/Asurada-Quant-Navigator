"""阿斯拉量化系統 — SMC 訂單流結構分析路由"""

import csv
import io
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.data.fetchers.crypto_engine import crypto_engine
from app.data.fetchers.tw_stock_engine import tw_stock_engine
from app.utils.symbol import is_tw_stock
from app.models.schemas import SMCRequest
from app.core.ml.smc_detector import smc_detector

router = APIRouter()

# 簡易快取（記憶體內）
_smc_cache: dict[str, dict] = {}


def _load_data(symbol: str, timeframe: str):
    """載入本地數據"""
    engine = tw_stock_engine if is_tw_stock(symbol) else crypto_engine
    return engine.load_local_data(symbol=symbol, timeframe=timeframe)


@router.post("/analyze")
async def analyze_smc(request: SMCRequest):
    """SMC 訂單流結構分析"""
    tf_val = request.timeframe.value
    df = _load_data(request.symbol, tf_val)

    if df is None or df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"找不到 {request.symbol} {tf_val} 的本地數據，請先同步數據",
        )

    if len(df) < 30:
        raise HTTPException(
            status_code=400,
            detail=f"數據不足：需要至少 30 根 K 線，目前僅有 {len(df)} 根",
        )

    # 載入 HTF 數據
    df_htf = None
    htf_tf = request.htf
    if not htf_tf:
        _htf_map = {"15m": "1h", "1h": "4h", "4h": "1d", "1d": "1w"}
        htf_tf = _htf_map.get(tf_val)
    if htf_tf and htf_tf != tf_val:
        df_htf = _load_data(request.symbol, htf_tf)

    try:
        result = smc_detector.detect(
            df=df,
            symbol=request.symbol,
            timeframe=tf_val,
            df_htf=df_htf,
            lookback=request.lookback,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SMC 分析失敗: {str(e)}")

    result_dict = result.to_dict()
    _smc_cache[f"{request.symbol}_{tf_val}"] = result_dict
    return result_dict


@router.get("/latest/{symbol}")
async def get_latest_smc(
    symbol: str,
    timeframe: str = Query(default="1d"),
):
    """取得最新快取的 SMC 分析（不重新計算）"""
    cache_key = f"{symbol}_{timeframe}"
    if cache_key in _smc_cache:
        return _smc_cache[cache_key]

    # 快取不存在，即時計算
    df = _load_data(symbol, timeframe)
    if df is None or df.empty or len(df) < 30:
        raise HTTPException(
            status_code=404,
            detail=f"找不到 {symbol} {timeframe} 的足夠數據",
        )

    _htf_map = {"15m": "1h", "1h": "4h", "4h": "1d", "1d": "1w"}
    htf_tf = _htf_map.get(timeframe)
    df_htf = _load_data(symbol, htf_tf) if htf_tf and htf_tf != timeframe else None

    result = smc_detector.detect(
        df=df, symbol=symbol, timeframe=timeframe, df_htf=df_htf,
    )
    result_dict = result.to_dict()
    _smc_cache[cache_key] = result_dict
    return result_dict


@router.get("/export-csv/{symbol}")
async def export_smc_csv(
    symbol: str,
    timeframe: str = Query(default="1d"),
):
    """匯出 ML-Ready CSV"""
    cache_key = f"{symbol}_{timeframe}"

    # 使用快取或即時計算
    if cache_key not in _smc_cache:
        df = _load_data(symbol, timeframe)
        if df is None or df.empty or len(df) < 30:
            raise HTTPException(status_code=404, detail=f"找不到 {symbol} {timeframe} 的足夠數據")

        _htf_map = {"15m": "1h", "1h": "4h", "4h": "1d", "1d": "1w"}
        htf_tf = _htf_map.get(timeframe)
        df_htf = _load_data(symbol, htf_tf) if htf_tf and htf_tf != timeframe else None

        result = smc_detector.detect(df=df, symbol=symbol, timeframe=timeframe, df_htf=df_htf)
        _smc_cache[cache_key] = result.to_dict()

    data = _smc_cache[cache_key]
    csv_content = _build_ml_csv(data)

    safe_symbol = symbol.replace("/", "_")
    filename = f"smc_{safe_symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"

    return StreamingResponse(
        io.StringIO(csv_content),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def _build_ml_csv(data: dict) -> str:
    """將 SMC 分析結果轉換為 ML-Ready CSV"""
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    headers = [
        "Timestamp", "Ticker", "TF", "Trend_HTF", "Trend_LTF",
        "Vol_Ratio", "Body_Wick_Ratio", "FVG_Width", "ATR_14",
        "RSI_Relative", "Premium_Discount_Pos", "Session",
        "Confidence", "Win_Rate_Est", "Entry", "SL", "TP",
        "RR_Ratio", "Logic_Label",
    ]
    writer.writerow(headers)

    # Data row
    ml = data.get("ml_features", {})
    row = [
        data.get("generated_at", ""),
        data.get("symbol", ""),
        data.get("timeframe", ""),
        data.get("trend_htf", ""),
        data.get("trend_ltf", ""),
        ml.get("vol_ratio", ""),
        ml.get("body_wick_ratio", ""),
        ml.get("fvg_width", ""),
        ml.get("atr_14", ""),
        ml.get("rsi_relative", ""),
        ml.get("premium_discount_pos", ""),
        "",  # Session — 需要更多數據來判斷
        data.get("confidence", ""),
        "",  # Win_Rate_Est — 需要歷史回測數據
        data.get("entry", ""),
        data.get("stop_loss", ""),
        data.get("take_profit", ""),
        data.get("rr_ratio", ""),
        data.get("bias", ""),
    ]
    writer.writerow(row)

    return output.getvalue()
