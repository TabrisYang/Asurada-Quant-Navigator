"""阿斯拉量化系統 — 台股基本面數據引擎

免費數據源：
- 月營收：TWSE 公開資訊觀測站
- 三大法人買賣超：TWSE
- 外資持股：TWSE
- 財報摘要（EPS/本益比/殖利率）：yfinance
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import aiohttp
from loguru import logger

from app.core.config.settings import settings


class TwFundamentalEngine:
    """台股基本面數據引擎"""

    def __init__(self):
        self.data_dir = settings.ohlcv_path.parent / "fundamental"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════
    #  1. 月營收
    # ═══════════════════════════════════════════════════

    async def fetch_revenue(self, code: str, months: int = 24) -> pd.DataFrame:
        """抓取月營收數據（TWSE 公開資訊觀測站）。"""
        filepath = self.data_dir / f"{code}_revenue.csv"

        # 嘗試讀取本地快取（< 7 天就用快取）
        if filepath.exists():
            try:
                df = pd.read_csv(filepath)
                if not df.empty:
                    return df
            except Exception:
                pass

        logger.info(f"基本面: 抓取 {code} 月營收...")
        rows = []
        now = datetime.now()

        async with aiohttp.ClientSession() as session:
            for i in range(months):
                target = now - timedelta(days=30 * i)
                year = target.year - 1911  # 民國年
                month = target.month

                try:
                    url = "https://mops.twse.com.tw/mops/web/ajax_t21sc04_ifrs"
                    data = {
                        "encodeURIComponent": 1,
                        "step": 1,
                        "firstin": 1,
                        "off": 1,
                        "TYPEK": "sii",
                        "year": str(year),
                        "month": str(month).zfill(2),
                        "co_id": code,
                    }
                    async with session.post(url, data=data, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            text = await resp.text()
                            # 解析 HTML 表格
                            revenue = self._parse_revenue_html(text, code)
                            if revenue is not None:
                                rows.append({
                                    "year_month": f"{target.year}-{month:02d}",
                                    "revenue": revenue,
                                })
                except Exception as e:
                    logger.debug(f"營收抓取 {code} {year}/{month} 失敗: {e}")
                    continue

                await asyncio.sleep(0.3)  # rate limit

        if not rows:
            # Fallback: 用 yfinance 取營收
            return await self._fetch_revenue_yfinance(code)

        df = pd.DataFrame(rows)
        df = df.sort_values("year_month").reset_index(drop=True)

        # 計算 MoM / YoY
        if len(df) >= 2:
            df["mom_pct"] = df["revenue"].pct_change() * 100
        if len(df) >= 13:
            df["yoy_pct"] = (df["revenue"] / df["revenue"].shift(12) - 1) * 100

        df.to_csv(filepath, index=False)
        logger.info(f"基本面: {code} 月營收 {len(df)} 筆已儲存")
        return df

    async def _fetch_revenue_yfinance(self, code: str) -> pd.DataFrame:
        """Fallback: 用 yfinance 取營收。"""
        try:
            import yfinance as yf
            from app.utils.symbol import symbol_to_yf_ticker
            ticker = symbol_to_yf_ticker(f"{code}/TWD")
            info = await asyncio.to_thread(lambda: yf.Ticker(ticker).quarterly_financials)
            if info is not None and not info.empty:
                if "Total Revenue" in info.index:
                    rev = info.loc["Total Revenue"].sort_index()
                    df = pd.DataFrame({
                        "year_month": [d.strftime("%Y-%m") for d in rev.index],
                        "revenue": rev.values,
                    })
                    filepath = self.data_dir / f"{code}_revenue.csv"
                    df.to_csv(filepath, index=False)
                    return df
        except Exception as e:
            logger.debug(f"yfinance 營收 fallback 失敗: {e}")
        return pd.DataFrame()

    def _parse_revenue_html(self, html: str, code: str) -> Optional[float]:
        """從 TWSE HTML 解析營收數字。"""
        try:
            # 簡易解析：找到公司代碼所在行，取營收欄位
            if code not in html:
                return None
            tables = pd.read_html(html, encoding="utf-8")
            for table in tables:
                for _, row in table.iterrows():
                    row_str = str(row.values)
                    if code in row_str:
                        # 營收通常在第 3-4 欄
                        for val in row.values:
                            try:
                                v = float(str(val).replace(",", ""))
                                if v > 1000:  # 營收至少千元
                                    return v
                            except (ValueError, TypeError):
                                continue
        except Exception:
            pass
        return None

    # ═══════════════════════════════════════════════════
    #  2. 三大法人買賣超
    # ═══════════════════════════════════════════════════

    async def fetch_institutional(self, code: str, days: int = 60) -> pd.DataFrame:
        """抓取三大法人買賣超（TWSE）。"""
        filepath = self.data_dir / f"{code}_institutional.csv"

        if filepath.exists():
            try:
                df = pd.read_csv(filepath)
                if not df.empty and len(df) >= 10:
                    return df
            except Exception:
                pass

        logger.info(f"基本面: 抓取 {code} 法人買賣超...")
        rows = []

        async with aiohttp.ClientSession() as session:
            for i in range(days):
                date = datetime.now() - timedelta(days=i)
                date_str = date.strftime("%Y%m%d")

                try:
                    url = f"https://www.twse.com.tw/rwd/zh/fund/T86?date={date_str}&selectType=ALLBUT0999&response=json"
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get("stat") == "OK" and data.get("data"):
                                for row in data["data"]:
                                    if str(row[0]).strip() == code:
                                        rows.append({
                                            "date": date.strftime("%Y-%m-%d"),
                                            "foreign_buy_sell": self._parse_num(row[4]),
                                            "trust_buy_sell": self._parse_num(row[10]),
                                            "dealer_buy_sell": self._parse_num(row[11]),
                                        })
                                        break
                except Exception:
                    continue

                await asyncio.sleep(0.5)  # TWSE rate limit 較嚴

                if len(rows) >= 30:
                    break

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.sort_values("date").reset_index(drop=True)
        df.to_csv(filepath, index=False)
        logger.info(f"基本面: {code} 法人買賣超 {len(df)} 筆已儲存")
        return df

    # ═══════════════════════════════════════════════════
    #  3. 外資持股
    # ═══════════════════════════════════════════════════

    async def fetch_foreign_holding(self, code: str) -> pd.DataFrame:
        """抓取外資持股比例（簡化版：用 yfinance）。"""
        filepath = self.data_dir / f"{code}_foreign_holding.csv"

        try:
            import yfinance as yf
            from app.utils.symbol import symbol_to_yf_ticker
            ticker = symbol_to_yf_ticker(f"{code}/TWD")
            info = await asyncio.to_thread(lambda: yf.Ticker(ticker).info)

            if info:
                holders = {
                    "institution_pct": info.get("heldPercentInstitutions", 0),
                    "insider_pct": info.get("heldPercentInsiders", 0),
                }
                df = pd.DataFrame([{
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    **holders,
                }])
                df.to_csv(filepath, index=False)
                return df
        except Exception as e:
            logger.debug(f"外資持股抓取失敗: {e}")

        if filepath.exists():
            return pd.read_csv(filepath)
        return pd.DataFrame()

    # ═══════════════════════════════════════════════════
    #  4. 財報摘要
    # ═══════════════════════════════════════════════════

    async def fetch_financials(self, code: str) -> dict:
        """抓取財報摘要（yfinance）。"""
        try:
            import yfinance as yf
            from app.utils.symbol import symbol_to_yf_ticker
            ticker = symbol_to_yf_ticker(f"{code}/TWD")
            info = await asyncio.to_thread(lambda: yf.Ticker(ticker).info)

            if not info:
                return {}

            return {
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "pb_ratio": info.get("priceToBook"),
                "dividend_yield": round(info.get("dividendYield", 0), 2) if info.get("dividendYield") else None,
                "eps_trailing": info.get("trailingEps"),
                "eps_forward": info.get("forwardEps"),
                "market_cap": info.get("marketCap"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
                "avg_volume": info.get("averageVolume"),
                "company_name": info.get("longName") or info.get("shortName"),
            }
        except Exception as e:
            logger.debug(f"財報抓取失敗: {e}")
            return {}

    # ═══════════════════════════════════════════════════
    #  工具
    # ═══════════════════════════════════════════════════

    @staticmethod
    def _parse_num(val) -> float:
        """解析 TWSE 的數字格式（含逗號）。"""
        try:
            return float(str(val).replace(",", "").strip())
        except (ValueError, TypeError):
            return 0.0


# Singleton
tw_fundamental = TwFundamentalEngine()
