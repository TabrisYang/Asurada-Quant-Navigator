"""STL 時序分解：把 OHLCV 拆成 趨勢 / 季節 / 殘差。

用途：給 LLM 一個比「RSI 70 偏高」更立體的 price action 描述：
  - 趨勢方向（最近 N 根的趨勢線斜率）
  - 季節性週期強度（是否有規律性循環）
  - 殘差波動率（去除可預測成分後剩多少雜訊）

整合方式：在 chat.py 的 chart_state 組裝點自動跑 decompose_price，
結果塞進 chart_state.decomposition 給 LLM。
"""

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    from statsmodels.tsa.seasonal import STL
    _STL_AVAILABLE = True
except ImportError:
    _STL_AVAILABLE = False
    STL = None


def decompose_price(
    df: pd.DataFrame,
    period: int = 20,
    seasonal: int = 7,
) -> dict:
    """STL 分解價格序列，回傳給 LLM 看的摘要。

    Args:
        df: OHLCV DataFrame（需要 close 欄位）
        period: STL 週期（日線可用 20 = 約一個月、60 = 約一季）
        seasonal: 季節性平滑窗口（建議 7-13 的奇數）

    Returns:
        dict: 摘要欄位
          - trend_slope_pct_recent_10bars: 最近 10 根的趨勢斜率（%）
          - trend_direction: "上升" / "下降" / "橫盤"
          - seasonal_strength: 季節性強度（0-N，越大越明顯）
          - seasonal_interpretation: "明顯週期" / "微弱週期" / "無明顯週期"
          - residual_volatility_pct: 殘差波動率（占當前價格 %）
          - data_points_used: 用了幾根資料
        若資料不足或 STL 不可用，回傳 {"error": "..."} 結構
    """
    if not _STL_AVAILABLE:
        return {"error": "statsmodels 未安裝，無法執行 STL 分解"}

    if df is None or "close" not in df.columns:
        return {"error": "缺少 close 欄位"}

    close = df["close"].dropna()
    min_required = period * 3  # STL 至少需要 3 個完整週期

    if len(close) < min_required:
        return {"error": f"資料不足（需至少 {min_required} 根，目前 {len(close)} 根）"}

    try:
        # robust=True 對極端值更穩定（金融資料常有跳空）
        stl = STL(close.values, period=period, seasonal=seasonal, robust=True).fit()
    except Exception as e:
        logger.warning(f"STL 分解失敗: {e}")
        return {"error": f"STL 計算錯誤：{str(e)[:80]}"}

    trend = pd.Series(stl.trend)
    seasonal_arr = pd.Series(stl.seasonal)
    resid = pd.Series(stl.resid)

    # 最近 10 根的趨勢斜率（%）
    n = min(10, len(trend))
    recent_trend = trend.tail(n)
    if recent_trend.iloc[0] != 0 and not pd.isna(recent_trend.iloc[0]):
        trend_slope_pct = (recent_trend.iloc[-1] - recent_trend.iloc[0]) / abs(recent_trend.iloc[0]) * 100
    else:
        trend_slope_pct = 0.0
    trend_slope_pct = round(float(trend_slope_pct), 2)

    if trend_slope_pct > 0.5:
        trend_direction = "上升"
    elif trend_slope_pct < -0.5:
        trend_direction = "下降"
    else:
        trend_direction = "橫盤"

    # 季節性強度（季節成分 std / 殘差 std）
    seasonal_std = float(seasonal_arr.std())
    resid_std = float(resid.std())
    seasonal_strength = round(seasonal_std / (resid_std + 1e-9), 2)

    if seasonal_strength > 1.0:
        seasonal_interpretation = "明顯週期"
    elif seasonal_strength > 0.5:
        seasonal_interpretation = "微弱週期"
    else:
        seasonal_interpretation = "無明顯週期"

    # 殘差波動率（占當前價格 %）
    current_price = float(close.iloc[-1])
    residual_volatility_pct = round(resid_std / current_price * 100, 2) if current_price > 0 else 0.0

    return {
        "trend_slope_pct_recent_10bars": trend_slope_pct,
        "trend_direction": trend_direction,
        "seasonal_strength": seasonal_strength,
        "seasonal_interpretation": seasonal_interpretation,
        "residual_volatility_pct": residual_volatility_pct,
        "period_used": period,
        "data_points_used": len(close),
    }


def is_available() -> bool:
    """STL 模組是否可用（statsmodels 是否已安裝）。"""
    return _STL_AVAILABLE
