"""阿斯拉量化系統 — 台股 BB Width 壓縮掃描器

一次性掃描全部上市櫃約 1900 檔，找出目前處於布林通道壓縮（BB Width 百分位低）
的個股，並回傳最新價、日期、產業、5 日均量、近 20 日漲跌幅等資訊。

設計要點：
- 並發抓取（預設 10），各標的使用 TwStockEngine 本地快取機制
- 進度/結果以 async iterator 流式產出，方便外層 SSE 直接轉發
- 失敗率 > 5% 時中止，避免 yfinance 封 IP
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import AsyncIterator, Optional

import numpy as np
import pandas as pd
from loguru import logger

from app.core.indicators.technical import calc_vol_squeeze
from app.data.fetchers.tw_stock_engine import TwStockEngine
from app.data.fetchers.tw_universe import TwStock, tw_universe


_DEFAULT_CONCURRENCY = 10
_MIN_BARS_REQUIRED = 150  # 120 根 lookback + 20~30 根容忍
_HISTORY_DAYS = 400       # 掃描需要的歷史天數（日曆日 ~= 270 交易日，遠超 _MIN_BARS_REQUIRED）
_MAX_FAIL_RATE = 0.10     # 失敗率 > 10% 中止
_MIN_SAMPLES_FOR_FAIL_CHECK = 50


@dataclass
class ScanResult:
    code: str
    name: str
    market: str               # "listed" / "otc"
    industry: str
    price: float              # 最新收盤價
    price_date: str           # YYYY-MM-DD
    bb_width_pctile: float    # 核心排序依據
    bb_width: float           # 絕對寬度（%）
    volume_5d_avg: int        # 5 日均量（張）
    change_20d: float         # 近 20 根漲跌幅（%）

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScanProgress:
    current: int
    total: int
    found: int
    fail: int
    eta_sec: int              # 預估剩餘秒數


@dataclass
class ScanFailure:
    code: str
    name: str
    market: str
    industry: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


class TwBBWidthScanner:
    """台股 BB Width 壓縮掃描器。"""

    def __init__(self):
        self._engine = TwStockEngine()

    async def scan(
        self,
        timeframe: str = "1d",
        pctile_threshold: float = 20.0,
        markets: Optional[list[str]] = None,
        concurrency: int = _DEFAULT_CONCURRENCY,
        progress_every: int = 10,
        min_volume: int = 0,                  # 5 日均量門檻（張）；0 = 不過濾
        require_healthy_trend: bool = False,  # MA60 斜率 > 0 或 收盤 > MA20 > MA60
        max_adx: Optional[float] = None,      # ADX 上限（< 25 = 真盤整）；None = 不過濾
        persistence_bars: int = 1,            # 最近 N 根都要壓縮（>1 = 要求持續性）
        min_abs_bb_width: float = 0.0,        # 絕對 BB Width 下限 %（排除常年低波動標的）
        history_days: int = _HISTORY_DAYS,    # 抓取歷史天數（日曆日）
    ) -> AsyncIterator[dict]:
        """執行全市場掃描。

        以 async iterator 產出 dict 事件，外層 SSE 直接轉發：
          {"type": "progress", ...ScanProgress}
          {"type": "result", ...ScanResult}
          {"type": "warning", "message": str}
          {"type": "done", "total_scanned": int, "total_found": int, "duration_sec": float}
          {"type": "error", "error": str}
        """
        markets = markets or ["listed", "otc"]

        stocks = await tw_universe.fetch_all(use_cache=True)
        stocks = [s for s in stocks if s.market in markets]
        total = len(stocks)
        if total == 0:
            yield {"type": "error", "error": "標的池為空，請檢查網路連線或 TWSE ISIN 服務"}
            return

        logger.info(f"TwBBWidthScanner: 開始掃描 {total} 檔（threshold={pctile_threshold}）")
        start = time.monotonic()
        sem = asyncio.Semaphore(concurrency)
        queue: asyncio.Queue[dict] = asyncio.Queue()
        state: dict = {"current": 0, "found": 0, "fail": 0, "failures": []}

        async def worker(stock: TwStock):
            async with sem:
                try:
                    res = await self._evaluate_one(
                        stock, timeframe, pctile_threshold,
                        min_volume=min_volume,
                        require_healthy_trend=require_healthy_trend,
                        max_adx=max_adx,
                        persistence_bars=persistence_bars,
                        min_abs_bb_width=min_abs_bb_width,
                        history_days=history_days,
                    )
                    if res is not None:
                        state["found"] += 1
                        await queue.put({"type": "result", **res.to_dict()})
                except Exception as e:
                    logger.debug(f"掃描 {stock.code} 失敗: {e}")
                    failure = ScanFailure(
                        code=stock.code,
                        name=stock.name,
                        market=stock.market,
                        industry=stock.industry,
                        reason=str(e) or type(e).__name__,
                    ).to_dict()
                    state["failures"].append(failure)
                    state["fail"] += 1
                    await queue.put({"type": "failure", **failure})

                state["current"] += 1

                # 依固定節奏回報進度
                if state["current"] % progress_every == 0 or state["current"] == total:
                    elapsed = time.monotonic() - start
                    rate = state["current"] / elapsed if elapsed > 0 else 0
                    remain = (total - state["current"]) / rate if rate > 0 else 0
                    await queue.put({
                        "type": "progress",
                        "current": state["current"],
                        "total": total,
                        "found": state["found"],
                        "fail": state["fail"],
                        "eta_sec": int(remain),
                    })

        # 同時啟動所有 worker + 消費 queue
        tasks = [asyncio.create_task(worker(s)) for s in stocks]

        aborted = False
        while True:
            try:
                evt = await asyncio.wait_for(queue.get(), timeout=0.5)
                yield evt
            except asyncio.TimeoutError:
                pass

            # 失敗率過高 → 中止
            if (
                not aborted
                and state["current"] >= _MIN_SAMPLES_FOR_FAIL_CHECK
                and state["fail"] / state["current"] > _MAX_FAIL_RATE
            ):
                aborted = True
                for t in tasks:
                    t.cancel()
                yield {
                    "type": "warning",
                    "message": f"失敗率過高（{state['fail']}/{state['current']}），已中止掃描以避免 yfinance 封 IP",
                }

            if all(t.done() for t in tasks) and queue.empty():
                break

        # 清理：確保所有 task 完成（被 cancel 的也要 await）
        await asyncio.gather(*tasks, return_exceptions=True)

        duration = time.monotonic() - start
        yield {
            "type": "done",
            "total_scanned": state["current"],
            "total_found": state["found"],
            "total_fail": state["fail"],
            "duration_sec": round(duration, 1),
            "failures": state["failures"],
        }
        logger.info(
            f"TwBBWidthScanner: 完成，"
            f"掃描 {state['current']}/{total}，找到 {state['found']} 檔，"
            f"失敗 {state['fail']}，耗時 {duration:.1f}s"
        )

    # ──────────────────────────────────────────────
    # 單檔評估
    # ──────────────────────────────────────────────

    async def _evaluate_one(
        self,
        stock: TwStock,
        timeframe: str,
        pctile_threshold: float,
        *,
        min_volume: int = 0,
        require_healthy_trend: bool = False,
        max_adx: Optional[float] = None,
        persistence_bars: int = 1,
        min_abs_bb_width: float = 0.0,
        history_days: int = _HISTORY_DAYS,
    ) -> Optional[ScanResult]:
        """評估單一標的是否符合 BB 壓縮 + 進階過濾條件。"""
        symbol = f"{stock.code}/TWD"

        # 明確指定 start_date 讓 engine 啟動 backfill 邏輯：
        # 本地檔案若歷史不足，engine 會自動從 yfinance 補齊
        df = await self._engine.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            start_date=datetime.now() - timedelta(days=history_days),
        )
        # 資料抓取失敗或不足 → raise 讓 worker 計為 fail
        if df is None or len(df) == 0:
            raise ValueError("無歷史資料（ticker 不存在、下市股或資料源異常）")
        if len(df) < _MIN_BARS_REQUIRED:
            raise ValueError(
                f"歷史資料僅 {len(df)} 筆（需至少 {_MIN_BARS_REQUIRED}；新上市或資料源斷檔）"
            )

        # 套用 vol_squeeze 指標
        out = calc_vol_squeeze(df, {"bb_period": 20, "lookback": 120})
        pctile_list = out["BB_Width_Pctile"]
        if not pctile_list:
            raise ValueError("empty pctile output")

        # 壓縮持續性：最近 N 根都要 < 門檻（persistence_bars=1 相當於舊行為）
        n = max(1, persistence_bars)
        if len(pctile_list) < n:
            raise ValueError(f"pctile list too short for persistence ({len(pctile_list)} < {n})")
        recent_pctiles = pctile_list[-n:]
        for p in recent_pctiles:
            if p is None or (isinstance(p, float) and np.isnan(p)):
                raise ValueError("pctile NaN in recent bars")
        if any(p >= pctile_threshold for p in recent_pctiles):
            return None  # 正常結果：最近 N 根內有一根未壓縮

        latest_pctile = recent_pctiles[-1]

        # 計算附加指標
        last = df.iloc[-1]
        price = float(last["close"])
        price_date = pd.to_datetime(last["timestamp"]).strftime("%Y-%m-%d")

        # BB Width 絕對值（提前計算以便做絕對下限過濾）
        window20 = df["close"].tail(20)
        sma = window20.mean()
        std = window20.std()
        bb_width_abs = (2 * std / sma) * 100 if sma > 0 else 0.0

        # 進階過濾：絕對 BB Width 下限（排除常年低波動 ETF / 控股類）
        if min_abs_bb_width > 0 and bb_width_abs < min_abs_bb_width:
            return None

        volume_5d = int(df["volume"].tail(5).mean()) if "volume" in df.columns else 0
        # 台股 yfinance volume 單位是「股」，換算為張（÷1000）
        volume_5d_avg = int(volume_5d / 1000)

        # 進階過濾：成交量
        if min_volume > 0 and volume_5d_avg < min_volume:
            return None

        # 進階過濾：趨勢健康（MA60 斜率 > 0 或 收盤 > MA20 > MA60）
        if require_healthy_trend and len(df) >= 65:
            close_series = df["close"]
            ma60_series = close_series.rolling(60).mean()
            ma60_last = float(ma60_series.iloc[-1])
            ma60_prev = float(ma60_series.iloc[-6])  # 5 根前的 MA60
            ma20_last = float(close_series.tail(20).mean())

            slope_ok = not np.isnan(ma60_last) and not np.isnan(ma60_prev) and ma60_last > ma60_prev
            stack_ok = price > ma20_last > ma60_last

            if not (slope_ok or stack_ok):
                return None

        # 進階過濾：ADX 盤整（趨勢強度 < max_adx）
        if max_adx is not None and len(df) >= 30:
            try:
                adx_val = _compute_adx(df, period=14)
                if adx_val is not None and adx_val > max_adx:
                    return None
            except Exception:
                pass  # ADX 計算失敗不影響主流程

        if len(df) >= 20:
            prev_close = float(df.iloc[-20]["close"])
            change_20d = (price - prev_close) / prev_close * 100 if prev_close > 0 else 0.0
        else:
            change_20d = 0.0

        return ScanResult(
            code=stock.code,
            name=stock.name,
            market=stock.market,
            industry=stock.industry,
            price=round(price, 2),
            price_date=price_date,
            bb_width_pctile=round(float(latest_pctile), 2),
            bb_width=round(float(bb_width_abs), 2),
            volume_5d_avg=volume_5d_avg,
            change_20d=round(change_20d, 2),
        )


def _compute_adx(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """輕量 ADX：只算最後一根的值，用於盤整判定。"""
    if len(df) < period * 2:
        return None
    high = df["high"]; low = df["low"]; close = df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denom
    adx = dx.rolling(period).mean()
    if adx.empty or np.isnan(adx.iloc[-1]):
        return None
    return float(adx.iloc[-1])


# 單例
tw_bb_scanner = TwBBWidthScanner()
