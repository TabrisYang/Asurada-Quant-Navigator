"""讀取主系統 backend/data/ohlcv/*.csv 為 PoC 提供資料。

只讀不寫，不影響 production。
"""

from pathlib import Path
from typing import Optional

import pandas as pd

_OHLCV_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ohlcv"


def _symbol_to_filename(symbol: str, timeframe: str) -> str:
    """BTC/USDT + 1d → BTC_USDT_1d.csv；2330/TWD + 1d → 2330_TWD_1d.csv"""
    return f"{symbol.replace('/', '_')}_{timeframe}.csv"


def load(symbol: str, timeframe: str = "1d") -> Optional[pd.DataFrame]:
    """從本地 CSV 讀取 OHLCV。"""
    fname = _symbol_to_filename(symbol, timeframe)
    fpath = _OHLCV_DIR / fname
    if not fpath.exists():
        print(f"⚠ 找不到 {fpath}，請先到主系統的「同步」面板下載該資料")
        return None

    df = pd.read_csv(fpath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def get_returns(df: pd.DataFrame) -> pd.Series:
    """log returns（GARCH 用）。"""
    import numpy as np
    return np.log(df["close"] / df["close"].shift(1)).dropna()
