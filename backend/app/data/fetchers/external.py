"""阿斯拉量化系統 — 外部數據抓取器

提供無法從 OHLCV 計算的市場數據：
- 恐懼貪婪指數 (Fear & Greed Index)
- 資金費率 (Funding Rate)
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional

import httpx
import pandas as pd
from loguru import logger


class ExternalDataFetcher:
    """外部市場數據抓取器"""

    def __init__(self):
        self._cache: dict[str, tuple[float, pd.DataFrame]] = {}  # key -> (expiry_ts, df)
        self._cache_ttl = 300  # 5 分鐘快取

    def _get_cache(self, key: str) -> Optional[pd.DataFrame]:
        """取得快取（如果未過期）"""
        if key in self._cache:
            expiry, df = self._cache[key]
            if datetime.now().timestamp() < expiry:
                return df.copy()
        return None

    def _set_cache(self, key: str, df: pd.DataFrame):
        """設定快取"""
        self._cache[key] = (datetime.now().timestamp() + self._cache_ttl, df)

    # ──────────────────────────────────────────────
    # 恐懼貪婪指數（免費 API: alternative.me）
    # ──────────────────────────────────────────────
    async def fetch_fear_greed(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 0,
    ) -> pd.DataFrame:
        """
        抓取 Fear & Greed Index

        來源: https://api.alternative.me/fng/
        免費，無需 API Key，每日一筆。

        Returns:
            DataFrame with columns: timestamp, value, classification
        """
        cache_key = f"fear_greed_{start_date}_{end_date}_{limit}"
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached

        try:
            # 計算需要的天數（API 從「今天」往回推 limit 天）
            now = datetime.now()
            if start_date:
                sd = pd.Timestamp(start_date)
                days_from_today = max(int((now - sd).days) + 2, 30)
            elif limit > 0:
                days_from_today = limit
            else:
                days_from_today = 365

            url = f"https://api.alternative.me/fng/?limit={days_from_today}&format=json"

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()

            if "data" not in data:
                logger.warning(f"Fear & Greed API 回傳格式異常: {list(data.keys())}")
                return pd.DataFrame()

            rows = []
            for item in data["data"]:
                ts = datetime.fromtimestamp(int(item["timestamp"]))
                rows.append({
                    "timestamp": ts.strftime("%Y-%m-%d"),
                    "value": float(item["value"]),
                    "classification": item.get("value_classification", ""),
                })

            df = pd.DataFrame(rows)
            if df.empty:
                return df

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)

            # 依日期範圍篩選
            if start_date:
                df = df[df["timestamp"] >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df["timestamp"] <= pd.Timestamp(end_date)]

            df = df.reset_index(drop=True)
            self._set_cache(cache_key, df)
            logger.info(f"Fear & Greed 抓取完成: {len(df)} 筆")
            return df

        except Exception as e:
            logger.error(f"Fear & Greed 抓取失敗: {e}")
            return pd.DataFrame()

    # ──────────────────────────────────────────────
    # 資金費率（Binance 永續合約 API, 免費）
    # ──────────────────────────────────────────────
    async def fetch_funding_rate(
        self,
        symbol: str = "BTC/USDT",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        抓取 Binance 永續合約資金費率

        來源: https://fapi.binance.com/fapi/v1/fundingRate
        免費，無需 API Key。

        Returns:
            DataFrame with columns: timestamp, funding_rate
        """
        # 轉換 symbol 格式：BTC/USDT -> BTCUSDT
        binance_symbol = symbol.replace("/", "")

        cache_key = f"funding_{binance_symbol}_{start_date}_{end_date}"
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached

        try:
            all_rows = []
            base_url = "https://fapi.binance.com/fapi/v1/fundingRate"

            params: dict = {
                "symbol": binance_symbol,
                "limit": 1000,  # Binance 最大值
            }

            if start_date:
                sd = pd.Timestamp(start_date)
                params["startTime"] = int(sd.timestamp() * 1000)
            if end_date:
                ed = pd.Timestamp(end_date)
                params["endTime"] = int(ed.timestamp() * 1000)

            async with httpx.AsyncClient(timeout=30.0) as client:
                # 分批抓取（資金費率每 8 小時一筆）
                max_iterations = 20  # 防止無限迴圈
                iteration = 0

                while iteration < max_iterations:
                    resp = await client.get(base_url, params=params)
                    resp.raise_for_status()
                    data = resp.json()

                    if not data:
                        break

                    for item in data:
                        all_rows.append({
                            "timestamp": datetime.fromtimestamp(
                                int(item["fundingTime"]) / 1000
                            ),
                            "funding_rate": float(item["fundingRate"]) * 100,  # 轉為百分比
                        })

                    # 如果回傳不足 1000 筆，代表已抓完
                    if len(data) < 1000:
                        break

                    # 繼續從最後一筆之後抓取
                    last_ts = int(data[-1]["fundingTime"]) + 1
                    if end_date:
                        ed_ms = int(pd.Timestamp(end_date).timestamp() * 1000)
                        if last_ts >= ed_ms:
                            break
                    params["startTime"] = last_ts
                    iteration += 1

                    await asyncio.sleep(0.2)  # Rate limit

            df = pd.DataFrame(all_rows)
            if df.empty:
                return df

            df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

            self._set_cache(cache_key, df)
            logger.info(f"Funding Rate 抓取完成: {binance_symbol}, {len(df)} 筆")
            return df

        except Exception as e:
            logger.error(f"Funding Rate 抓取失敗: {e}")
            return pd.DataFrame()


# 全域單例
external_fetcher = ExternalDataFetcher()
