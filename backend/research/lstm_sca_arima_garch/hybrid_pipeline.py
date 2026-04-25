"""混合預測 pipeline：STL + ARIMA + GARCH + LSTM 串接。

完整流程：
  1. STL 分解 close → trend + seasonal + residual
  2. ARIMA 預測 trend
  3. GARCH 預測 volatility（returns 變異數）
  4. LSTM 預測 residual 非線性
  5. 重組：final_close = trend_forecast + seasonal_forecast + residual_forecast
"""

from typing import Optional

import pandas as pd

from arima_trend import fit_predict as arima_fit_predict
from data_loader import load, get_returns
from decomposer import decompose
from garch_volatility import fit_predict as garch_fit_predict, is_available as garch_available
from lstm_residual import LSTMResidual, is_available as lstm_available


def run_hybrid_forecast(
    symbol: str,
    timeframe: str = "1d",
    n_forecast: int = 5,
    stl_period: int = 20,
    arima_order: tuple[int, int, int] = (1, 1, 1),
    lstm_seq_length: int = 10,
    lstm_hidden: int = 32,
    lstm_epochs: int = 50,
) -> dict:
    """跑完整混合預測，回傳 dict 含所有中間結果。

    PoC 階段：固定超參、不跑 SCA 優化（SCA 完整跑要 5-15 天）。
    """
    df = load(symbol, timeframe)
    if df is None or len(df) < stl_period * 4:
        return {"error": f"資料不足或不存在 ({symbol} {timeframe})"}

    close = df["close"].reset_index(drop=True)

    # 1. STL 分解
    decomp = decompose(close, period=stl_period)

    # 2. ARIMA on trend
    trend_forecast = arima_fit_predict(decomp.trend, n_forecast=n_forecast, order=arima_order)

    # 3. GARCH on returns
    returns = get_returns(df).reset_index(drop=True)
    garch_forecast = None
    if garch_available():
        try:
            garch_forecast = garch_fit_predict(returns, n_forecast=n_forecast)
        except Exception as e:
            print(f"⚠ GARCH 失敗: {e}")

    # 4. LSTM on residual
    residual_forecast = None
    if lstm_available():
        try:
            lstm = LSTMResidual(
                seq_length=lstm_seq_length,
                hidden_size=lstm_hidden,
                epochs=lstm_epochs,
            )
            lstm.fit(decomp.residual)
            residual_forecast = lstm.predict(decomp.residual, n_forecast=n_forecast)
        except Exception as e:
            print(f"⚠ LSTM 失敗: {e}")

    # 5. 季節性外推（簡單法：重複最後一個週期）
    seasonal_recent = decomp.seasonal.tail(stl_period).reset_index(drop=True)
    seasonal_forecast = pd.Series(
        [seasonal_recent.iloc[i % stl_period] for i in range(n_forecast)],
        name="seasonal_forecast",
    )

    # 6. 重組
    final_forecast = trend_forecast.values + seasonal_forecast.values
    if residual_forecast is not None:
        final_forecast = final_forecast + residual_forecast.values

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "last_close": float(close.iloc[-1]),
        "trend_forecast": trend_forecast.tolist(),
        "seasonal_forecast": seasonal_forecast.tolist(),
        "residual_forecast": residual_forecast.tolist() if residual_forecast is not None else None,
        "garch_cond_std_forecast": garch_forecast.tolist() if garch_forecast is not None else None,
        "final_close_forecast": final_forecast.tolist(),
        "config": {
            "stl_period": stl_period,
            "arima_order": list(arima_order),
            "lstm_seq_length": lstm_seq_length,
            "lstm_hidden": lstm_hidden,
            "lstm_epochs": lstm_epochs,
            "n_forecast": n_forecast,
        },
    }
