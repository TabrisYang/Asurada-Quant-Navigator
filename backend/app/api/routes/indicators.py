"""阿斯拉量化系統 — 技術指標路由"""

import hashlib
import json
import time
from collections import OrderedDict

import pandas as pd
from fastapi import APIRouter, HTTPException
from loguru import logger

from app.core.indicators import registry
from app.data.fetchers.crypto_engine import crypto_engine
from app.data.fetchers.external import external_fetcher
from app.models.schemas import (
    IndicatorRequest,
    IndicatorData,
    ConditionSearchRequest,
    ConditionSearchResponse,
    MatchedPeriod,
    DisplayMode,
)

# ─── 指標計算快取（LRU，最多 128 筆，TTL 5 分鐘）───────
_CACHE_MAX = 128
_CACHE_TTL = 300
_indicator_cache: OrderedDict[str, tuple[float, dict]] = OrderedDict()


def _cache_key(indicator_type: str, symbol: str, timeframe: str,
               start_date: str | None, end_date: str | None, params: dict) -> str:
    raw = json.dumps({
        "t": indicator_type, "s": symbol, "tf": timeframe,
        "sd": start_date, "ed": end_date, "p": params,
    }, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_get(key: str) -> dict | None:
    item = _indicator_cache.get(key)
    if item is None:
        return None
    ts, data = item
    if time.time() - ts > _CACHE_TTL:
        _indicator_cache.pop(key, None)
        return None
    _indicator_cache.move_to_end(key)
    return data


def _cache_set(key: str, data: dict):
    _indicator_cache[key] = (time.time(), data)
    if len(_indicator_cache) > _CACHE_MAX:
        _indicator_cache.popitem(last=False)

router = APIRouter()


@router.get("/list")
async def list_indicators():
    """取得所有可用指標清單"""
    info_list = registry.to_info_list()
    return {"indicators": info_list, "total": len(info_list)}


@router.post("/calculate")
async def calculate_indicator(request: IndicatorRequest):
    """計算指定指標"""
    indicator = registry.get(request.indicator_type.lower())
    if not indicator:
        raise HTTPException(status_code=404, detail=f"找不到指標: {request.indicator_type}")

    symbol = request.parameters.get("symbol", "BTC/USDT")
    timeframe = request.parameters.get("timeframe", "1d")
    start_date = request.parameters.get("start_date")
    end_date = request.parameters.get("end_date")

    calc_params = {k: v for k, v in request.parameters.items()
                   if k not in ("symbol", "timeframe", "start_date", "end_date")}

    # ─── 外部數據指標（不依賴 OHLCV）────────────────
    if indicator.data_source == "external":
        result = await _calculate_external_indicator(
            indicator_id=indicator.id,
            symbol=str(symbol),
            start_date=str(start_date) if start_date else None,
            end_date=str(end_date) if end_date else None,
            timeframe=str(timeframe),
            params=calc_params,
        )
        if result is None:
            raise HTTPException(status_code=500, detail=f"外部指標 {indicator.name} 數據抓取失敗")

        return IndicatorData(
            name=indicator.name,
            display_mode=DisplayMode(indicator.display_mode),
            data=result,
            parameters=calc_params or {k: v["default"] for k, v in indicator.parameters.items()},
        )

    # ─── OHLCV 指標（本地數據計算，帶快取）────────────────
    ck = _cache_key(
        request.indicator_type.lower(), str(symbol), str(timeframe),
        str(start_date) if start_date else None,
        str(end_date) if end_date else None,
        calc_params,
    )
    cached = _cache_get(ck)
    if cached is not None:
        return IndicatorData(
            name=indicator.name,
            display_mode=DisplayMode(indicator.display_mode),
            data=cached,
            parameters=calc_params or {k: v["default"] for k, v in indicator.parameters.items()},
        )

    df = crypto_engine.load_local_data(
        symbol=str(symbol),
        timeframe=str(timeframe),
        start_date=str(start_date) if start_date else None,
        end_date=str(end_date) if end_date else None,
    )

    if df.empty:
        raise HTTPException(status_code=404, detail=f"找不到 {symbol} {timeframe} 的數據")

    _calc_params_with_ctx = dict(calc_params or {})
    _calc_params_with_ctx.setdefault("_symbol", str(symbol))
    _calc_params_with_ctx.setdefault("_timeframe", str(timeframe))
    result = registry.calculate(request.indicator_type.lower(), df, _calc_params_with_ctx)

    if result is None:
        raise HTTPException(status_code=500, detail="指標計算失敗")

    _cache_set(ck, result)

    return IndicatorData(
        name=indicator.name,
        display_mode=DisplayMode(indicator.display_mode),
        data=result,
        parameters=calc_params or {k: v["default"] for k, v in indicator.parameters.items()},
    )


async def _calculate_external_indicator(
    indicator_id: str,
    symbol: str,
    start_date: str | None,
    end_date: str | None,
    timeframe: str,
    params: dict,
) -> dict[str, list] | None:
    """處理外部數據來源的指標計算"""

    try:
        if indicator_id == "fear_greed":
            return await _calc_fear_greed_aligned(start_date, end_date, timeframe, symbol)
        elif indicator_id == "funding":
            return await _calc_funding_aligned(symbol, start_date, end_date, timeframe, params)
        else:
            logger.warning(f"不支援的外部指標: {indicator_id}")
            return None
    except Exception as e:
        logger.error(f"外部指標 {indicator_id} 計算失敗: {e}")
        return None


async def _calc_fear_greed_aligned(
    start_date: str | None,
    end_date: str | None,
    timeframe: str,
    symbol: str,
) -> dict[str, list]:
    """
    抓取 Fear & Greed 數據，並對齊到 OHLCV 的時間軸。
    Fear & Greed 是每日指數，對於更小的時間級別使用前向填充。
    """
    fg_df = await external_fetcher.fetch_fear_greed(
        start_date=start_date,
        end_date=end_date,
    )

    if fg_df.empty:
        logger.warning("Fear & Greed 數據為空")
        return {"Fear_Greed": []}

    # 載入 OHLCV 時間軸作為對齊基準
    ohlcv_df = crypto_engine.load_local_data(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    )

    if ohlcv_df.empty:
        # 沒有 OHLCV 數據，直接回傳 FG 原始數據
        return {
            "Fear_Greed": fg_df["value"].tolist(),
            "Classification": fg_df["classification"].tolist(),
        }

    # 對齊到 OHLCV 時間軸（使用日期做前向填充合併）
    ohlcv_df["timestamp"] = pd.to_datetime(ohlcv_df["timestamp"])
    fg_df["timestamp"] = pd.to_datetime(fg_df["timestamp"])

    # 建立日期索引方便 merge
    ohlcv_df["_date"] = ohlcv_df["timestamp"].dt.date
    fg_df["_date"] = fg_df["timestamp"].dt.date

    merged = ohlcv_df[["_date"]].merge(
        fg_df[["_date", "value", "classification"]],
        on="_date",
        how="left",
    )

    # 前向填充（週末等無數據的日子用前一天的值）
    merged["value"] = merged["value"].ffill()

    def _safe(v):
        return None if pd.isna(v) else round(float(v), 1)

    # 同時回傳 0-100 的參考線（25=極端恐懼, 50=中性, 75=極端貪婪）
    n = len(merged)
    return {
        "Fear_Greed": [_safe(v) for v in merged["value"]],
        "Extreme_Fear": [25.0] * n,
        "Extreme_Greed": [75.0] * n,
    }


async def _calc_funding_aligned(
    symbol: str,
    start_date: str | None,
    end_date: str | None,
    timeframe: str,
    params: dict,
) -> dict[str, list]:
    """
    抓取 Funding Rate 數據，並對齊到 OHLCV 的時間軸。
    Funding Rate 每 8 小時一筆，用 merge_asof 對齊。
    """
    fr_df = await external_fetcher.fetch_funding_rate(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )

    if fr_df.empty:
        logger.warning(f"Funding Rate 數據為空: {symbol}")
        return {"Funding_Rate": []}

    # 載入 OHLCV 時間軸
    ohlcv_df = crypto_engine.load_local_data(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    )

    if ohlcv_df.empty:
        return {
            "Funding_Rate": [round(float(v), 6) for v in fr_df["funding_rate"]],
        }

    # 對齊：使用 merge_asof（向後取最近的 funding rate）
    ohlcv_df["timestamp"] = pd.to_datetime(ohlcv_df["timestamp"])
    fr_df["timestamp"] = pd.to_datetime(fr_df["timestamp"])

    ohlcv_sorted = ohlcv_df[["timestamp"]].sort_values("timestamp")
    fr_sorted = fr_df[["timestamp", "funding_rate"]].sort_values("timestamp")

    merged = pd.merge_asof(
        ohlcv_sorted,
        fr_sorted,
        on="timestamp",
        direction="backward",  # 用最近過去的 funding rate
    )

    threshold = float(params.get("alert_threshold", 0.05))

    def _safe(v):
        return None if pd.isna(v) else round(float(v), 6)

    return {
        "Funding_Rate": [_safe(v) for v in merged["funding_rate"]],
        "Alert_High": [threshold] * len(merged),
        "Alert_Low": [-threshold] * len(merged),
    }


@router.post("/search", response_model=ConditionSearchResponse)
async def search_conditions(request: ConditionSearchRequest):
    """條件搜尋 — 找出滿足條件的時間段"""
    import pandas as pd

    df = crypto_engine.load_local_data(
        symbol=request.symbol,
        timeframe=request.timeframe.value,
        start_date=request.start_date,
        end_date=request.end_date,
    )

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"找不到 {request.symbol} {request.timeframe.value} 的數據"
        )

    # 計算所有需要的指標
    condition_masks = []
    for cond in request.conditions:
        indicator_id = cond.indicator.lower()
        calc_result = registry.calculate(indicator_id, df, cond.parameters or {})
        if not calc_result:
            raise HTTPException(status_code=400, detail=f"無法計算指標: {cond.indicator}")

        # 取第一個 series 作為比較對象
        series_name = list(calc_result.keys())[0]
        values = pd.Series(calc_result[series_name])

        # 建立條件遮罩
        op = cond.operator
        if op == ">":
            mask = values > cond.value
        elif op == "<":
            mask = values < cond.value
        elif op == ">=":
            mask = values >= cond.value
        elif op == "<=":
            mask = values <= cond.value
        elif op == "==":
            mask = values == cond.value
        elif op == "cross_above":
            mask = (values > cond.value) & (values.shift(1) <= cond.value)
        elif op == "cross_below":
            mask = (values < cond.value) & (values.shift(1) >= cond.value)
        elif op == "between":
            mask = (values >= cond.value) & (values <= cond.value2)
        else:
            raise HTTPException(status_code=400, detail=f"不支援的運算子: {op}")

        condition_masks.append(mask)

    # 組合條件
    if request.logical_operator == "AND":
        combined = condition_masks[0]
        for m in condition_masks[1:]:
            combined = combined & m
    else:
        combined = condition_masks[0]
        for m in condition_masks[1:]:
            combined = combined | m

    # 找出連續匹配的時間段
    matched = df[combined].copy()
    periods = []

    if not matched.empty:
        timestamps = matched["timestamp"].tolist()
        start = timestamps[0]
        prev = start

        for i in range(1, len(timestamps)):
            current = timestamps[i]
            # 如果不連續，結束這個區段
            gap = (pd.Timestamp(current) - pd.Timestamp(prev)).total_seconds()
            expected_gap = {
                "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800
            }.get(request.timeframe.value, 86400) * 1.5

            if gap > expected_gap:
                periods.append(MatchedPeriod(start=str(start), end=str(prev)))
                start = current
            prev = current

        periods.append(MatchedPeriod(start=str(start), end=str(prev)))

    summary_parts = []
    for cond in request.conditions:
        summary_parts.append(f"{cond.indicator} {cond.operator} {cond.value}")
    condition_text = f" {request.logical_operator} ".join(summary_parts)

    return ConditionSearchResponse(
        symbol=request.symbol,
        timeframe=request.timeframe.value,
        matched_periods=periods,
        total_matches=len(matched),
        summary=f"找到 {len(periods)} 個時間段（共 {len(matched)} 根 K 線）滿足條件: {condition_text}",
    )
