"""阿斯拉量化系統 — 台股族群分析引擎

批次載入族群內個股數據 → 合成族群指數 → 技術分析 → Breadth → 相對強弱排名。
"""

import numpy as np
import pandas as pd
from loguru import logger
from typing import Optional

from app.data.tw_sectors import (
    get_sector_symbols, get_sector_name, get_stock_name, list_sectors,
)
from app.data.fetchers.tw_stock_engine import tw_stock_engine
from app.data.fetchers.crypto_engine import CryptoDataEngine


async def analyze_sector(
    sector_name: str,
    timeframe: str = "1d",
    lookback_days: int = 120,
) -> dict:
    """分析台股特定產業族群的整體趨勢。

    Args:
        sector_name: 族群名稱（支援別名，如「ai」「半導體」）
        timeframe: 時間週期（1d / 1w）
        lookback_days: 回看天數

    Returns:
        dict: 族群指數、技術分析、breadth、個股相對強弱排名
    """
    # 1. 查表
    resolved_name = get_sector_name(sector_name)
    if resolved_name is None:
        all_sectors = list_sectors()
        names = [s["name"] for s in all_sectors]
        return {
            "status": "error",
            "message": f"找不到族群「{sector_name}」",
            "available_sectors": names,
        }

    symbols = get_sector_symbols(resolved_name)
    if not symbols:
        return {"status": "error", "message": f"族群「{resolved_name}」沒有成分股"}

    logger.info(f"族群分析 [{resolved_name}]: {len(symbols)} 檔成分股，timeframe={timeframe}")

    # 2. 批次載入數據
    stock_data: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for code in symbols:
        symbol = f"{code}/TWD"
        df = tw_stock_engine.load_local_data(symbol, timeframe)

        if df is None or df.empty:
            # 嘗試即時抓取
            try:
                logger.info(f"族群分析: {code} 本地無數據，嘗試抓取...")
                df = await tw_stock_engine.fetch_ohlcv(symbol, timeframe)
            except Exception as e:
                logger.warning(f"族群分析: {code} 抓取失敗: {e}")

        if df is not None and not df.empty and len(df) >= 20:
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            # 只保留近 N 天
            if lookback_days and "timestamp" in df.columns:
                cutoff = df["timestamp"].max() - pd.Timedelta(days=lookback_days)
                df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
            if len(df) >= 20:
                stock_data[code] = df
            else:
                missing.append(code)
        else:
            missing.append(code)

    if len(stock_data) < 2:
        return {
            "status": "error",
            "message": f"族群「{resolved_name}」有效數據不足（僅 {len(stock_data)} 檔），至少需要 2 檔",
            "symbols_missing": missing,
        }

    logger.info(f"族群分析 [{resolved_name}]: {len(stock_data)} 檔有效，{len(missing)} 檔缺失")

    # 3. 合成等權族群指數
    sector_index_df = _build_sector_index(stock_data)

    # 4. 技術分析（複用現有指標引擎）
    sector_index_df = CryptoDataEngine._calculate_indicators(sector_index_df)
    technical = _extract_technical(sector_index_df)

    # 5. Breadth 指標
    breadth = _compute_breadth(stock_data)

    # 6. 個股相對強弱
    rs_ranking = _compute_relative_strength(stock_data, sector_index_df)

    # 7. 族群指數變動
    index_stats = _compute_index_stats(sector_index_df)

    # 8. 綜合解讀
    interpretation = _generate_interpretation(
        resolved_name, technical, breadth, rs_ranking, index_stats,
    )

    return {
        "status": "success",
        "sector_name": resolved_name,
        "symbols_analyzed": list(stock_data.keys()),
        "symbols_missing": missing,
        "sector_index": index_stats,
        "technical": technical,
        "breadth": breadth,
        "relative_strength": rs_ranking[:10],  # 前 10 名
        "interpretation": interpretation,
    }


# ═══════════════════════════════════════════════════════
#  內部函式
# ═══════════════════════════════════════════════════════

def _build_sector_index(stock_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """用等權日報酬平均合成族群指數，回傳虛擬 OHLCV DataFrame。"""
    # 收集所有個股的日報酬率（以 close 計算）
    returns_list = []
    for code, df in stock_data.items():
        close = df.set_index("timestamp")["close"]
        ret = close.pct_change().dropna()
        ret.name = code
        returns_list.append(ret)

    # 合併對齊日期
    all_returns = pd.concat(returns_list, axis=1)
    all_returns = all_returns.dropna(how="all")

    # 等權平均
    avg_returns = all_returns.mean(axis=1)

    # 累積成指數（基期 = 100）
    cumulative = (1 + avg_returns).cumprod() * 100

    # 建構虛擬 OHLCV（open ≈ 前一天 close，high/low 用當天各股極端估算）
    index_df = pd.DataFrame()
    index_df["timestamp"] = cumulative.index
    index_df["close"] = cumulative.values

    # 簡化：open = 前日 close, high/low 用 ±daily_std 估算
    index_df["open"] = index_df["close"].shift(1).fillna(index_df["close"].iloc[0])

    # 用各股 high/low 的平均報酬推算
    high_returns_list = []
    low_returns_list = []
    for code, df in stock_data.items():
        aligned = df.set_index("timestamp").reindex(cumulative.index)
        if "high" in aligned.columns and "close" in aligned.columns:
            hr = (aligned["high"] / aligned["close"]).dropna()
            hr.name = code
            high_returns_list.append(hr)
        if "low" in aligned.columns and "close" in aligned.columns:
            lr = (aligned["low"] / aligned["close"]).dropna()
            lr.name = code
            low_returns_list.append(lr)

    if high_returns_list:
        avg_high_ratio = pd.concat(high_returns_list, axis=1).mean(axis=1)
        index_df["high"] = (index_df["close"].values * avg_high_ratio.reindex(cumulative.index).fillna(1.0).values)
    else:
        index_df["high"] = index_df["close"]

    if low_returns_list:
        avg_low_ratio = pd.concat(low_returns_list, axis=1).mean(axis=1)
        index_df["low"] = (index_df["close"].values * avg_low_ratio.reindex(cumulative.index).fillna(1.0).values)
    else:
        index_df["low"] = index_df["close"]

    # volume: 各股 volume 加總
    vol_list = []
    for code, df in stock_data.items():
        vol = df.set_index("timestamp")["volume"]
        vol.name = code
        vol_list.append(vol)
    if vol_list:
        total_vol = pd.concat(vol_list, axis=1).sum(axis=1)
        index_df["volume"] = total_vol.reindex(cumulative.index).fillna(0).values
    else:
        index_df["volume"] = 0

    index_df = index_df.reset_index(drop=True)
    return index_df


def _extract_technical(df: pd.DataFrame) -> dict:
    """從計算完指標的 DataFrame 提取最新技術面摘要。"""
    if df.empty:
        return {}

    last = df.iloc[-1]
    result = {}

    for col in ["rsi", "atr", "bb_width", "macd", "macd_signal", "macd_hist"]:
        if col in df.columns and pd.notna(last.get(col)):
            result[col] = round(float(last[col]), 4)

    # SMA
    close = df["close"]
    if len(close) >= 20:
        result["sma_20"] = round(float(close.rolling(20).mean().iloc[-1]), 2)
        result["above_sma20"] = bool(close.iloc[-1] > result["sma_20"])
    if len(close) >= 60:
        result["sma_60"] = round(float(close.rolling(60).mean().iloc[-1]), 2)
        result["above_sma60"] = bool(close.iloc[-1] > result["sma_60"])

    # 趨勢判定
    signals = []
    rsi = result.get("rsi")
    if rsi is not None:
        if rsi > 50:
            signals.append("RSI > 50")
        else:
            signals.append("RSI < 50")

    macd_hist = result.get("macd_hist")
    if macd_hist is not None:
        if macd_hist > 0:
            signals.append("MACD 柱正值")
        else:
            signals.append("MACD 柱負值")

    if result.get("above_sma20"):
        signals.append("站上 MA20")
    if result.get("above_sma60"):
        signals.append("站上 MA60")

    bullish_count = sum(1 for s in signals if any(
        kw in s for kw in ["RSI > 50", "正值", "站上"]
    ))
    total = len(signals)
    if total > 0:
        ratio = bullish_count / total
        if ratio >= 0.7:
            result["trend"] = f"偏多（{', '.join(signals)}）"
        elif ratio <= 0.3:
            result["trend"] = f"偏空（{', '.join(signals)}）"
        else:
            result["trend"] = f"中性（{', '.join(signals)}）"

    return result


def _compute_breadth(stock_data: dict[str, pd.DataFrame]) -> dict:
    """計算族群 breadth 指標。"""
    n = len(stock_data)
    if n == 0:
        return {}

    above_ma20 = 0
    above_ma60 = 0
    rsi_above_50 = 0
    advancing = 0

    for code, df in stock_data.items():
        close = df["close"]
        if len(close) < 2:
            continue

        # 今日上漲
        if close.iloc[-1] > close.iloc[-2]:
            advancing += 1

        # MA20
        if len(close) >= 20:
            ma20 = close.rolling(20).mean().iloc[-1]
            if close.iloc[-1] > ma20:
                above_ma20 += 1

        # MA60
        if len(close) >= 60:
            ma60 = close.rolling(60).mean().iloc[-1]
            if close.iloc[-1] > ma60:
                above_ma60 += 1

        # RSI > 50
        if "rsi" in df.columns and pd.notna(df["rsi"].iloc[-1]):
            if df["rsi"].iloc[-1] > 50:
                rsi_above_50 += 1

    pct_ma20 = round(above_ma20 / n * 100, 1)
    pct_ma60 = round(above_ma60 / n * 100, 1)
    pct_rsi50 = round(rsi_above_50 / n * 100, 1)
    adv_ratio = round(advancing / n * 100, 1)

    # 解讀
    if pct_ma20 >= 70:
        interp = "族群內部一致性高，多數個股同步走強"
    elif pct_ma20 >= 50:
        interp = "族群分化，約一半個股偏強"
    elif pct_ma20 >= 30:
        interp = "族群偏弱，多數個股低於短期均線"
    else:
        interp = "族群全面弱勢，幾乎所有個股低於短期均線"

    return {
        "pct_above_ma20": pct_ma20,
        "pct_above_ma60": pct_ma60,
        "pct_rsi_above_50": pct_rsi50,
        "advancing_ratio": adv_ratio,
        "total_stocks": n,
        "interpretation": interp,
    }


def _compute_relative_strength(
    stock_data: dict[str, pd.DataFrame],
    sector_df: pd.DataFrame,
    period: int = 20,
) -> list[dict]:
    """計算各股 vs 族群的相對強弱，回傳排名。"""
    if sector_df.empty or len(sector_df) < period:
        return []

    sector_close = sector_df["close"]
    sector_ret = (sector_close.iloc[-1] / sector_close.iloc[-period] - 1) * 100

    results = []
    for code, df in stock_data.items():
        close = df["close"]
        if len(close) < period:
            continue
        stock_ret = (close.iloc[-1] / close.iloc[-period] - 1) * 100
        rs_ratio = round(float(stock_ret / sector_ret), 2) if sector_ret != 0 else 1.0

        results.append({
            "symbol": code,
            "name": get_stock_name(code),
            "return_pct": round(float(stock_ret), 2),
            "rs_ratio": rs_ratio,
        })

    results.sort(key=lambda x: x["rs_ratio"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results


def _compute_index_stats(df: pd.DataFrame) -> dict:
    """計算族群指數的變動統計。"""
    if df.empty:
        return {}

    close = df["close"]
    current = float(close.iloc[-1])

    stats = {"current_value": round(current, 2)}

    for label, period in [("1d", 1), ("5d", 5), ("20d", 20), ("60d", 60)]:
        if len(close) > period:
            prev = float(close.iloc[-1 - period])
            pct = (current / prev - 1) * 100 if prev != 0 else 0
            stats[f"change_pct_{label}"] = round(pct, 2)

    return stats


def _generate_interpretation(
    name: str,
    technical: dict,
    breadth: dict,
    rs_ranking: list[dict],
    index_stats: dict,
) -> str:
    """生成族群趨勢綜合解讀。"""
    parts = []

    # 趨勢
    trend = technical.get("trend", "")
    if trend:
        parts.append(f"{name}族群{trend}")

    # Breadth
    pct_ma20 = breadth.get("pct_above_ma20", 0)
    parts.append(f"{pct_ma20}% 個股站上 MA20")

    # 近期漲跌
    chg_20d = index_stats.get("change_pct_20d")
    if chg_20d is not None:
        if chg_20d > 0:
            parts.append(f"近 20 日上漲 {chg_20d}%")
        else:
            parts.append(f"近 20 日下跌 {chg_20d}%")

    # 領漲/領跌
    if rs_ranking:
        leader = rs_ranking[0]
        parts.append(f"龍頭 {leader['name']}（RS {leader['rs_ratio']}）")
        if len(rs_ranking) >= 2:
            laggard = rs_ranking[-1]
            parts.append(f"最弱 {laggard['name']}（RS {laggard['rs_ratio']}）")

    return "；".join(parts)
