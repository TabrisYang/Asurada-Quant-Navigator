"""ARIMA 處理 STL 分解後的 trend 成分。

預測未來 N 步的 trend 走向。
"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def fit_predict(
    trend: pd.Series,
    n_forecast: int = 5,
    order: tuple[int, int, int] = (1, 1, 1),
) -> pd.Series:
    """擬合 ARIMA 並預測未來 n_forecast 步。

    Args:
        trend: STL 分解出的 trend 序列
        n_forecast: 預測步數
        order: (p, d, q) — 預設 (1,1,1) 為通用穩健設定

    Returns:
        pd.Series: 預測值（length = n_forecast）
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(trend.values, order=order).fit()
    forecast = model.forecast(steps=n_forecast)
    return pd.Series(forecast, name="trend_forecast")


def fit_predict_walk_forward(
    trend: pd.Series,
    train_size: int,
    n_forecast: int = 1,
    order: tuple[int, int, int] = (1, 1, 1),
) -> pd.Series:
    """Walk-forward 驗證：每次用 [0:i] 訓練、預測下一步。"""
    predictions = []
    for i in range(train_size, len(trend)):
        train = trend.iloc[:i]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(train.values, order=order).fit()
            pred = float(model.forecast(steps=n_forecast)[0])
        except Exception:
            pred = float(train.iloc[-1])  # fallback: persistence
        predictions.append(pred)
    return pd.Series(predictions, index=trend.index[train_size:], name="arima_walk_forward")
