"""阿斯拉量化系統 — 數據獲取基礎類"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd


class BaseDataFetcher(ABC):
    """數據獲取器基礎介面"""

    @abstractmethod
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        抓取 OHLCV 數據

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        pass

    @abstractmethod
    async def get_supported_symbols(self) -> list[str]:
        """取得支援的交易對"""
        pass
