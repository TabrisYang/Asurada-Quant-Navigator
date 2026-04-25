"""STL 時序分解 — 把 close 序列拆 trend / seasonal / residual。

跟主系統 backend/app/core/timeseries_decomposition.py 共用相同邏輯，
但這裡回傳完整序列（給 ARIMA / GARCH / LSTM 訓練用），不只摘要。
"""

from typing import NamedTuple

import pandas as pd
from statsmodels.tsa.seasonal import STL


class DecompositionResult(NamedTuple):
    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series


def decompose(close: pd.Series, period: int = 20, seasonal: int = 7) -> DecompositionResult:
    """STL 分解，回傳三個完整序列（保留 index）。"""
    stl = STL(close.values, period=period, seasonal=seasonal, robust=True).fit()
    return DecompositionResult(
        trend=pd.Series(stl.trend, index=close.index, name="trend"),
        seasonal=pd.Series(stl.seasonal, index=close.index, name="seasonal"),
        residual=pd.Series(stl.resid, index=close.index, name="residual"),
    )
