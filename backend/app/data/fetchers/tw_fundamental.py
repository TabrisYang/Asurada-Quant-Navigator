"""阿斯拉量化系統 — 台股基本面數據引擎

免費數據源：
- 月營收：TWSE 公開資訊觀測站
- 三大法人買賣超：TWSE
- 外資持股：TWSE
- 財報摘要（EPS/本益比/殖利率）：yfinance
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import aiohttp
from loguru import logger

from app.core.config.settings import settings


def _cache_fresh(filepath: Path, max_age_days: int = 7) -> bool:
    """快取檔是否在 max_age_days 內（修舊 bug：原只檢查 exists 不檢查時間 → 永遠回傳舊資料）。"""
    try:
        if not filepath.exists():
            return False
        age_sec = datetime.now().timestamp() - filepath.stat().st_mtime
        return age_sec < max_age_days * 86400
    except Exception:
        return False


class _TwseEpsCircuitBreaker:
    """TWSE mops 季報失敗自動斷路：連續 5 次失敗 → 暫停 30s 走 yfinance fallback。

    避免 mops 短暫故障（404 / 503 / CAPTCHA）時整批 30 檔重複打到被封 IP。
    狀態與 tw_stock_engine._CircuitBreaker 相同：closed / open / half-open。
    """

    def __init__(self, threshold: int = 5, recovery_sec: int = 30):
        self.failures = 0
        self.opened_at = 0.0
        self.threshold = threshold
        self.recovery_sec = recovery_sec
        self.state = "closed"

    def can_call(self) -> bool:
        if self.state == "open":
            if time.time() - self.opened_at > self.recovery_sec:
                self.state = "half-open"
                logger.warning("TWSE EPS circuit breaker → half-open，嘗試恢復")
                return True
            return False
        return True

    def on_success(self):
        if self.state in ("half-open", "open"):
            logger.info("TWSE EPS circuit breaker → closed，已恢復")
        self.failures = 0
        self.state = "closed"

    def on_failure(self):
        self.failures += 1
        if self.failures >= self.threshold and self.state != "open":
            self.state = "open"
            self.opened_at = time.time()
            logger.warning(
                f"TWSE EPS circuit breaker → open，連續 {self.failures} 失敗，"
                f"暫停 {self.recovery_sec}s 改走 yfinance fallback"
            )


_twse_eps_breaker = _TwseEpsCircuitBreaker()


class _FinMindCircuitBreaker:
    """FinMind quota / rate limit 失敗斷路：連續 5 次失敗 → 30s 改走 yfinance。

    FinMind v4 free tier 每 IP / 日 600 req。超量會回 402 / 429。
    """

    def __init__(self, threshold: int = 5, recovery_sec: int = 30):
        self.failures = 0
        self.opened_at = 0.0
        self.threshold = threshold
        self.recovery_sec = recovery_sec
        self.state = "closed"

    def can_call(self) -> bool:
        if self.state == "open":
            if time.time() - self.opened_at > self.recovery_sec:
                self.state = "half-open"
                logger.warning("FinMind circuit breaker → half-open，嘗試恢復")
                return True
            return False
        return True

    def on_success(self):
        if self.state in ("half-open", "open"):
            logger.info("FinMind circuit breaker → closed，已恢復")
        self.failures = 0
        self.state = "closed"

    def on_failure(self):
        self.failures += 1
        if self.failures >= self.threshold and self.state != "open":
            self.state = "open"
            self.opened_at = time.time()
            logger.warning(
                f"FinMind circuit breaker → open，連續 {self.failures} 失敗，"
                f"暫停 {self.recovery_sec}s 走 yfinance fallback"
            )


_finmind_breaker = _FinMindCircuitBreaker()


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

        # 讀取本地快取（< 7 天才用，過期自動重抓）
        if _cache_fresh(filepath):
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

        if _cache_fresh(filepath):
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

        # 抓取失敗才退快取，但只接受 7 天內的（避免送年久資料）
        if _cache_fresh(filepath):
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
    #  5. EPS 拆解（年度 vs 季度）
    # ═══════════════════════════════════════════════════

    @staticmethod
    def _eps_sanity(value) -> Optional[float]:
        """EPS 範圍 sanity：abs > 500 或 NaN → 視為單位錯，回 None。

        台股 EPS 99% 落在 -100 ~ 500 NTD 區間（極端如美超微國外 ADR 才會超過）。
        超過視為 yfinance 把 ADR (USD) 數字誤當原股 → 過濾。
        """
        try:
            v = float(value)
        except (ValueError, TypeError):
            return None
        if v != v:  # NaN
            return None
        if abs(v) > 500:
            return None
        return v

    @staticmethod
    def _latest_published_quarter(today: datetime) -> tuple[int, int]:
        """根據今天日期推算「最近一份已公告」的 (year, season)。

        公告排程（台灣法規）：
          Q1 → 當年 5/15
          Q2 → 當年 8/14
          Q3 → 當年 11/14
          Q4 (年報) → 次年 3/31
        """
        y, m, d = today.year, today.month, today.day
        if (m, d) >= (11, 15):
            return (y, 3)
        if (m, d) >= (8, 15):
            return (y, 2)
        if (m, d) >= (5, 16):
            return (y, 1)
        if (m, d) >= (4, 1):
            return (y - 1, 4)
        return (y - 1, 3)

    # mops 目前對程式化請求回「安全性考量」擋頁（需要 JS / session cookie 流程）
    # 此 flag 預設關閉，避免每次批次浪費 5 個 request 才 trip Circuit Breaker。
    # 未來若實作 cookie/session 流程或改用其他 endpoint，可改 True 重啟。
    _TWSE_SCRAPER_ENABLED = False

    async def fetch_eps_via_finmind(self, code: str) -> dict:
        """從 FinMind v4 抓真實單季 EPS，再合成上年度年報 + 最新季。

        EPS endpoint 回傳「單季 EPS」陣列：
          [{date: '2024-03-31', type: 'EPS', value: 8.7}, ...]

        合成邏輯：
          - 最新季 EPS = 最新一筆 row
          - 上年度 EPS = 最近一個「完整 4 季」的合計
            （例如最新一筆是 2026Q1 → 找 2025Q1+Q2+Q3+Q4 都齊 → 65~66）

        Circuit Breaker：連續 5 檔失敗 → 暫停 30s 走 yfinance。
        失敗回 {}。
        """
        if not _finmind_breaker.can_call():
            return {}

        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            "dataset": "TaiwanStockFinancialStatements",
            "data_id": code,
            "start_date": "2024-01-01",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status in (402, 429):
                        _finmind_breaker.on_failure()
                        logger.debug(f"FinMind EPS {code} quota/rate-limit：{resp.status}")
                        return {}
                    if resp.status != 200:
                        _finmind_breaker.on_failure()
                        return {}
                    payload = await resp.json()
        except Exception as e:  # noqa: BLE001
            _finmind_breaker.on_failure()
            logger.debug(f"FinMind EPS {code} 連線失敗：{e}")
            return {}

        data = payload.get("data") or []
        eps_rows = [r for r in data if r.get("type") == "EPS" and r.get("value") is not None]
        if not eps_rows:
            _finmind_breaker.on_failure()
            return {}

        eps_rows.sort(key=lambda r: r["date"])  # 由舊到新

        result: dict = {}

        # 最新季 EPS
        latest = eps_rows[-1]
        latest_val = self._eps_sanity(latest["value"])
        if latest_val is not None:
            result["quarter_latest"] = latest_val
            date_str = str(latest["date"])
            try:
                y, m, _ = date_str.split("-")
                q = (int(m) - 1) // 3 + 1
                result["quarter_latest_label"] = f"{y}Q{q}"
            except (ValueError, IndexError):
                result["quarter_latest_label"] = date_str

        # 上年度 EPS：找最近「完整 4 季」的那一年加總
        by_year: dict[int, dict[int, float]] = {}
        for r in eps_rows:
            try:
                y, m, _ = str(r["date"]).split("-")
                yi, mi = int(y), int(m)
                q = (mi - 1) // 3 + 1
                val = self._eps_sanity(r["value"])
                if val is not None:
                    by_year.setdefault(yi, {})[q] = val
            except (ValueError, IndexError):
                continue

        for yr in sorted(by_year.keys(), reverse=True):
            qs = by_year[yr]
            if all(q in qs for q in (1, 2, 3, 4)):
                annual = round(qs[1] + qs[2] + qs[3] + qs[4], 2)
                clean = self._eps_sanity(annual)
                if clean is not None:
                    result["annual_prev"] = clean
                    result["annual_prev_label"] = str(yr)
                break

        if not result:
            _finmind_breaker.on_failure()
            return {}

        _finmind_breaker.on_success()
        result["data_source"] = "finmind"
        result["as_of"] = datetime.now().isoformat(timespec="seconds")
        return result

    async def fetch_pe_via_finmind(self, code: str) -> Optional[float]:
        """從 FinMind v4 抓最近一日的 PER（本益比）。失敗回 None。"""
        if not _finmind_breaker.can_call():
            return None

        start = (datetime.now() - timedelta(days=10)).date().isoformat()
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            "dataset": "TaiwanStockPER",
            "data_id": code,
            "start_date": start,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status in (402, 429):
                        _finmind_breaker.on_failure()
                        return None
                    if resp.status != 200:
                        _finmind_breaker.on_failure()
                        return None
                    payload = await resp.json()
        except Exception as e:  # noqa: BLE001
            _finmind_breaker.on_failure()
            logger.debug(f"FinMind PE {code} 連線失敗：{e}")
            return None

        rows = payload.get("data") or []
        if not rows:
            return None
        rows.sort(key=lambda r: r.get("date") or "")
        per = rows[-1].get("PER")
        try:
            pv = float(per) if per is not None else None
        except (ValueError, TypeError):
            return None
        if pv is None or pv != pv or pv <= 0 or pv > 10000:
            return None
        _finmind_breaker.on_success()
        return pv

    async def fetch_eps_via_twse(self, code: str) -> dict:
        """從 TWSE mops 季報抓 EPS（主源）。

        嘗試抓「上年度年報 (Q4)」+「最新公告季 (Q1-Q3)」共 2 期。
        回傳：成功時含 annual_prev / quarter_latest / labels，否則回 {}。

        快取：成功結果寫進 {code}_eps.csv，7 天內命中直接讀檔。
        Circuit Breaker：連續 5 檔失敗 → 暫停 30s，期間直接回 {} 讓上層 fallback。
        """
        if not self._TWSE_SCRAPER_ENABLED:
            return {}
        if not _twse_eps_breaker.can_call():
            return {}

        filepath = self.data_dir / f"{code}_eps.csv"
        if _cache_fresh(filepath):
            try:
                df = pd.read_csv(filepath)
                if not df.empty:
                    row = df.iloc[0].to_dict()
                    return {k: (None if pd.isna(v) else v) for k, v in row.items()}
            except Exception:
                pass

        today = datetime.now()
        # 上年度年報：當前已過 4/1 → 用去年 Q4；否則用前年 Q4
        annual_year = today.year - 1 if (today.month, today.day) >= (4, 1) else today.year - 2
        latest_q_year, latest_q_season = self._latest_published_quarter(today)

        # Q4 == 年報，若最新季也是 Q4 則跳過（與 annual_prev 重複）
        periods = [("annual", annual_year, 4)]
        if (latest_q_year, latest_q_season) != (annual_year, 4):
            periods.append(("quarter", latest_q_year, latest_q_season))

        result: dict = {}
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
            "Referer": "https://mops.twse.com.tw/mops/web/t164sb04",
        }

        success_any = False
        async with aiohttp.ClientSession(headers=headers) as session:
            for kind, yr, season in periods:
                try:
                    eps = await self._fetch_one_quarter_eps(session, code, yr, season)
                    await asyncio.sleep(0.5)  # mops rate limit
                    if eps is None:
                        continue
                    success_any = True
                    if kind == "annual":
                        result["annual_prev"] = eps
                        result["annual_prev_label"] = str(yr)
                    else:
                        result["quarter_latest"] = eps
                        result["quarter_latest_label"] = f"{yr}Q{season}"
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"TWSE EPS 抓取 {code} {yr}Q{season} 失敗：{e}")
                    continue

        if not success_any:
            _twse_eps_breaker.on_failure()
            return {}

        _twse_eps_breaker.on_success()
        result["data_source"] = "twse"
        result["as_of"] = datetime.now().isoformat(timespec="seconds")

        # 寫入 CSV cache
        try:
            pd.DataFrame([result]).to_csv(filepath, index=False)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"TWSE EPS cache 寫入失敗 {code}：{e}")

        return result

    async def _fetch_one_quarter_eps(
        self, session: aiohttp.ClientSession, code: str, year: int, season: int,
    ) -> Optional[float]:
        """打 mops t164sb04 拿單一期 EPS，找「基本每股盈餘」row。"""
        url = "https://mops.twse.com.tw/mops/web/ajax_t164sb04"
        data = {
            "encodeURIComponent": 1,
            "step": 1,
            "firstin": 1,
            "off": 1,
            "isQuery": "Y",
            "TYPEK": "sii",
            "co_id": code,
            "year": str(year),
            "season": str(season).zfill(2),
            "REPORT_ID": "C",
        }
        async with session.post(url, data=data, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                return None
            text = await resp.text(errors="replace")

        try:
            tables = pd.read_html(text)
        except Exception:
            return None

        # 找「基本每股盈餘（元）」或「每股盈餘」的 row，第二欄通常是數值
        for table in tables:
            for _, row in table.iterrows():
                row_str = " ".join(str(v) for v in row.values)
                if "基本每股盈餘" in row_str or ("每股盈餘" in row_str and "稀釋" not in row_str):
                    for val in row.values:
                        try:
                            v = float(str(val).replace(",", "").strip())
                            cleaned = self._eps_sanity(v)
                            if cleaned is not None:
                                return cleaned
                        except (ValueError, TypeError):
                            continue
        return None

    async def fetch_eps_breakdown(self, code: str) -> dict:
        """抓 EPS 拆解：TWSE 主源 + yfinance fallback + sanity check。

        回傳：
          {
            "annual_prev": float | None,        # 上年度（已完成會計年度）EPS
            "annual_prev_label": "2025" | None,
            "quarter_latest": float | None,     # 最新公告一季 EPS
            "quarter_latest_label": "2026Q1" | None,
            "eps_trailing": float | None,       # TTM EPS（yfinance info）
            "pe_ttm": float | None,             # 最新本益比（yfinance trailingPE）
            "data_source": "twse" | "yfinance" | "none",
            "as_of": ISO 時間戳,
            "quality": "ok" | "missing",        # 留 hook 給未來細分（filtered_*）
          }
        """
        result = {
            "annual_prev": None,
            "annual_prev_label": None,
            "quarter_latest": None,
            "quarter_latest_label": None,
            "eps_trailing": None,
            "pe_ttm": None,
            "data_source": "none",
            "as_of": datetime.now().isoformat(timespec="seconds"),
            "quality": "missing",
        }

        # 第 1 層：FinMind 主源（真實單季 EPS + 即時 PER）
        try:
            finmind = await self.fetch_eps_via_finmind(code)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"FinMind EPS 抓取整體失敗 {code}：{e}")
            finmind = {}

        if finmind:
            for k in ("annual_prev", "annual_prev_label", "quarter_latest", "quarter_latest_label"):
                if finmind.get(k) is not None:
                    result[k] = finmind[k]
            result["data_source"] = finmind.get("data_source", "finmind")
            result["as_of"] = finmind.get("as_of", result["as_of"])
            if result["annual_prev"] is not None or result["quarter_latest"] is not None:
                result["quality"] = "ok"

        # FinMind PE（與 EPS 共用 Circuit Breaker；breaker open 時 fetch_pe_via_finmind 自己回 None）
        try:
            pe_fm = await self.fetch_pe_via_finmind(code)
            if pe_fm is not None:
                result["pe_ttm"] = pe_fm
        except Exception as e:  # noqa: BLE001
            logger.debug(f"FinMind PE 抓取失敗 {code}：{e}")

        # 第 2 層：TWSE 備源（目前 _TWSE_SCRAPER_ENABLED=False，預設跳過）
        if result["annual_prev"] is None or result["quarter_latest"] is None:
            try:
                twse = await self.fetch_eps_via_twse(code)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"TWSE EPS 抓取整體失敗 {code}：{e}")
                twse = {}
            if twse:
                if result["annual_prev"] is None and twse.get("annual_prev") is not None:
                    result["annual_prev"] = twse["annual_prev"]
                    result["annual_prev_label"] = twse.get("annual_prev_label")
                if result["quarter_latest"] is None and twse.get("quarter_latest") is not None:
                    result["quarter_latest"] = twse["quarter_latest"]
                    result["quarter_latest_label"] = twse.get("quarter_latest_label")
                if result["data_source"] == "none":
                    result["data_source"] = twse.get("data_source", "twse")
                    result["as_of"] = twse.get("as_of", result["as_of"])
                if result["annual_prev"] is not None or result["quarter_latest"] is not None:
                    result["quality"] = "ok"

        # 第 3 層：yfinance fallback（FinMind/TWSE 都拿不到時才補）
        try:
            import yfinance as yf
            from app.utils.symbol import symbol_to_yf_ticker
            ticker_str = symbol_to_yf_ticker(f"{code}/TWD")

            def _grab():
                t = yf.Ticker(ticker_str)
                return {
                    "income": getattr(t, "income_stmt", None),
                    "quarterly": getattr(t, "quarterly_income_stmt", None),
                    "info": t.info,
                }

            data = await asyncio.to_thread(_grab)
            info = data.get("info") or {}
            result["eps_trailing"] = self._eps_sanity(info.get("trailingEps"))
            # 只在 FinMind 沒拿到 PE 時才用 yfinance trailingPE 補（trailingPE 對部分台股不準）
            if result["pe_ttm"] is None:
                pe_raw = info.get("trailingPE")
                try:
                    pe_val = float(pe_raw) if pe_raw is not None else None
                    if pe_val is not None and (pe_val != pe_val or pe_val <= 0 or pe_val > 10000):
                        pe_val = None
                except (ValueError, TypeError):
                    pe_val = None
                result["pe_ttm"] = pe_val

            # 只在 TWSE 兩個 EPS 都沒拿到時才用 yfinance 補
            need_annual = result["annual_prev"] is None
            need_quarter = result["quarter_latest"] is None
            eps_keys = ("Basic EPS", "Diluted EPS")  # Basic 優先（台灣公告標準）
            today_year = datetime.now().year

            if need_annual:
                annual = data.get("income")
                if annual is not None and not annual.empty:
                    for key in eps_keys:
                        if key in annual.index:
                            series = annual.loc[key].dropna().sort_index(ascending=False)
                            for ts, raw in series.items():
                                # 跳過 fiscal year 進行中的（年份 >= 今年）
                                if hasattr(ts, "year") and ts.year >= today_year:
                                    continue
                                cleaned = self._eps_sanity(raw)
                                if cleaned is None:
                                    continue
                                result["annual_prev"] = cleaned
                                result["annual_prev_label"] = (
                                    ts.strftime("%Y") if hasattr(ts, "strftime") else str(ts)
                                )
                                if result["data_source"] == "none":
                                    result["data_source"] = "yfinance"
                                break
                            if result["annual_prev"] is not None:
                                break

            if need_quarter:
                quarterly = data.get("quarterly")
                if quarterly is not None and not quarterly.empty:
                    for key in eps_keys:
                        if key in quarterly.index:
                            series = quarterly.loc[key].dropna().sort_index(ascending=False)
                            for ts, raw in series.items():
                                cleaned = self._eps_sanity(raw)
                                if cleaned is None:
                                    continue
                                result["quarter_latest"] = cleaned
                                if hasattr(ts, "year") and hasattr(ts, "month"):
                                    q = (ts.month - 1) // 3 + 1
                                    result["quarter_latest_label"] = f"{ts.year}Q{q}"
                                else:
                                    result["quarter_latest_label"] = str(ts)
                                if result["data_source"] == "none":
                                    result["data_source"] = "yfinance"
                                break
                            if result["quarter_latest"] is not None:
                                break
        except Exception as e:  # noqa: BLE001
            logger.debug(f"yfinance EPS fallback 失敗 {code}：{e}")

        if result["annual_prev"] is not None or result["quarter_latest"] is not None:
            result["quality"] = "ok"

        return result

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
