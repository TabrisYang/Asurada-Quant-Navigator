"""本地資料存取 helpers — 依標的類型選擇資料引擎。

從 executor.py 抽出（repo health 行數護欄）。executor / learn 路由共用。
"""

from typing import Optional

import pandas as pd

from app.data.fetchers.crypto_engine import crypto_engine
from app.data.fetchers.tw_stock_engine import tw_stock_engine
from app.utils.symbol import is_tw_stock


def _load_local_data(symbol: str, timeframe: str, start: str = None, end: str = None):
    """根據 symbol 類型選擇正確的數據引擎載入本地資料"""
    engine = tw_stock_engine if is_tw_stock(symbol) else crypto_engine
    return engine.load_local_data(symbol, timeframe, start, end)


def _describe_range(df, timeframe: str) -> dict:
    """回報 df 實際涵蓋的時間範圍與根數（供回測範圍確認 / 結果透明化）。"""
    if df is None or df.empty or "timestamp" not in df.columns:
        return {"start": None, "end": None, "bars": 0, "timeframe": timeframe}
    ts = pd.to_datetime(df["timestamp"])
    return {
        "start": ts.min().strftime("%Y-%m-%d"),
        "end": ts.max().strftime("%Y-%m-%d"),
        "bars": int(len(df)),
        "timeframe": timeframe,
    }


def _needs_range_confirmation(
    symbol: str, timeframe: str, tool_label: str, loader=None,
) -> Optional[dict]:
    """回測家族工具的兩段式確認：未確認前先回報本地可用範圍，不執行。

    回傳 needs_confirmation dict（呼叫端應直接 return）；若找不到資料回傳 error dict；
    資料存在且應繼續執行則回傳 None。

    loader：資料載入函式，預設本模組 _load_local_data。executor 會傳入自己
    命名空間的綁定，讓測試 monkeypatch ex._load_local_data 的縫隙維持有效。
    """
    df_all = (loader or _load_local_data)(symbol, timeframe)
    if df_all is None or df_all.empty:
        return {"status": "error", "message": f"找不到 {symbol} {timeframe} 的本地數據，請先同步。"}
    rng = _describe_range(df_all, timeframe)
    return {
        "status": "needs_confirmation",
        "available_range": rng,
        "message": (
            f"本地 {symbol} {timeframe} 資料範圍 {rng['start']} ~ {rng['end']}，共 {rng['bars']} 根。"
            f"要用「全部歷史」{tool_label}，還是指定區間？請確認後我再執行。"
        ),
    }
