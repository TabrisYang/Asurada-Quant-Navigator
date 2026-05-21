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
from app.core.bollinger_signals import classify_bollinger_signal
from app.data.fetchers.tw_stock_engine import TwStockEngine
from app.data.fetchers.tw_universe import TwStock, tw_universe


_DEFAULT_CONCURRENCY = 8  # 對齊 yfinance 專用池（_YF_EXECUTOR max_workers=8），避免下載排隊塞爆 + 對 Yahoo 較友善
_MIN_BARS_REQUIRED = 150  # 120 根 lookback + 20~30 根容忍
_HISTORY_DAYS = 400       # 掃描需要的歷史天數（日曆日 ~= 270 交易日，遠超 _MIN_BARS_REQUIRED）
_MAX_FAIL_RATE = 0.10     # 失敗率 > 10% 中止
_MIN_SAMPLES_FOR_FAIL_CHECK = 50

# Token bucket：保護 yfinance 不被高並發打爆
_RATE_LIMIT_RPS = 10      # 平均每秒最多 10 個 yfinance 請求
_RATE_BURST = 20          # 短時間爆發容量


class _TokenBucket:
    """Async-safe token bucket，平均每秒 rate 個請求，bucket 容量 capacity。

    用途：搭配高並發 worker 時，避免 yfinance 在某秒突然被 25 個請求打爆。
    並發 25 仍然有效（同時 25 個 worker 跑 pandas 計算等耗 CPU 的事），
    只是「打 yfinance」的時機會被均勻分散。
    """

    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            self._refill()
            while self.tokens < 1:
                wait = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait)
                self._refill()
            self.tokens -= 1

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now


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
    # v137：v136 Bollinger 完整訊號（若 enable_v136 為 False 則為 None）
    bollinger_signal: Optional[dict] = None

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
        enable_v136: bool = True,             # v137：是否算 v136 Bollinger 訊號（fallback 用）
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
        bucket = _TokenBucket(rate=_RATE_LIMIT_RPS, capacity=_RATE_BURST)
        queue: asyncio.Queue[dict] = asyncio.Queue()
        state: dict = {"current": 0, "found": 0, "fail": 0, "failures": []}

        async def worker(stock: TwStock):
            async with sem:
                # 限速：保護 yfinance 不被高並發瞬間打爆
                await bucket.acquire()
                try:
                    res = await self._evaluate_one(
                        stock, timeframe, pctile_threshold,
                        min_volume=min_volume,
                        require_healthy_trend=require_healthy_trend,
                        max_adx=max_adx,
                        persistence_bars=persistence_bars,
                        min_abs_bb_width=min_abs_bb_width,
                        history_days=history_days,
                        enable_v136=enable_v136,
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
        enable_v136: bool = True,
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

        # v137：算 v136 完整 Bollinger 訊號（含 8 維特徵 + classify）
        bollinger_signal_result = None
        if enable_v136:
            try:
                adx_for_bollinger = _compute_adx(df, period=14) if len(df) >= 30 else None
                bollinger_signal_result = _compute_v136_bollinger(df, adx_for_bollinger)
            except Exception as _bb_err:
                logger.debug(f"[v137] {stock.code} bollinger 計算失敗: {_bb_err}")

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
            bollinger_signal=bollinger_signal_result,
        )


def _compute_v136_bollinger(df: pd.DataFrame, adx_val: Optional[float]) -> Optional[dict]:
    """v137：依台股單支 df 算 v136 Bollinger 8 個特徵並呼叫 classify_bollinger_signal。

    與 auto_scanner.py 對應路徑等價，但用 pandas 計算（df 模式），不依賴 numpy 預計算。
    """
    if len(df) < 30:
        return None

    try:
        closes = df["close"].astype(float).values
        highs = df["high"].astype(float).values
        lows = df["low"].astype(float).values
        volumes = df["volume"].astype(float).values if "volume" in df.columns else np.zeros(len(df))

        # BB 20 + std
        sma20 = pd.Series(closes).rolling(20).mean().values
        std20 = pd.Series(closes).rolling(20).std().values
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        bb_width = (bb_upper - bb_lower) / (sma20 + 1e-10) * 100
        bb_pos = (closes - bb_lower) / (bb_upper - bb_lower + 1e-10) * 100  # 0-100 %

        # ATR(14)
        tr = np.maximum(
            highs - lows,
            np.maximum(
                np.abs(highs - np.roll(closes, 1)),
                np.abs(lows - np.roll(closes, 1)),
            ),
        )
        atr14 = pd.Series(tr).rolling(14).mean().values
        atr_50_avg = float(np.nanmean(atr14[-60:-10])) if len(atr14) >= 60 else float(atr14[-1])
        atr_relative = float(atr14[-1] / (atr_50_avg + 1e-10)) if atr_50_avg > 0 else 1.0

        # OBV
        price_change_sign = np.sign(np.diff(closes, prepend=closes[0]))
        obv = np.cumsum(price_change_sign * volumes)

        # Keltner
        keltner_upper = sma20 + 1.5 * atr14
        keltner_lower = sma20 - 1.5 * atr14

        # is_squeeze + duration
        is_squeeze = (bb_upper < keltner_upper) & (bb_lower > keltner_lower)
        squeeze_duration = np.zeros(len(is_squeeze), dtype=int)
        run = 0
        for i in range(len(is_squeeze)):
            if is_squeeze[i]:
                run += 1
            else:
                run = 0
            squeeze_duration[i] = run

        last = len(closes) - 1
        # 8 個 v136 特徵組裝成 dict（與 auto_scanner._compute_features 對齊）
        features = {
            "bb_position": float(bb_pos[last]) if not np.isnan(bb_pos[last]) else 50.0,
            "bb_position_lag1": (
                float(bb_pos[last - 1]) if last >= 1 and not np.isnan(bb_pos[last - 1]) else 50.0
            ),
            "bb_width": float(bb_width[last]) if not np.isnan(bb_width[last]) else 0.0,
            "bb_width_roc": (
                float(bb_width[last] / bb_width[last - 4] - 1) * 100
                if last >= 4 and not np.isnan(bb_width[last - 4]) and bb_width[last - 4] > 1e-10
                else 0.0
            ),
            "z_score_20": (
                float((closes[last] - sma20[last]) / std20[last])
                if not np.isnan(std20[last]) and std20[last] > 1e-10
                else 0.0
            ),
            "atr_relative": atr_relative,
            "obv_slope_10": (
                float(np.polyfit(np.arange(10), obv[-10:], 1)[0] / (np.mean(np.abs(obv[-10:])) + 1e-10))
                if last >= 10 else 0.0
            ),
            "is_squeeze": bool(is_squeeze[last]),
            "squeeze_duration": int(squeeze_duration[last]),
            "adx": float(adx_val) if adx_val is not None else 0.0,
            "pct_6bar": (
                float((closes[last] - closes[last - 6]) / closes[last - 6] * 100)
                if last >= 6 and closes[last - 6] > 0 else 0.0
            ),
        }

        prev_features = {
            "bb_position": float(bb_pos[last - 1]) if last >= 1 and not np.isnan(bb_pos[last - 1]) else 50.0,
            "bb_position_lag1": float(bb_pos[last - 2]) if last >= 2 and not np.isnan(bb_pos[last - 2]) else 50.0,
            "is_squeeze": bool(is_squeeze[last - 1]) if last >= 1 else False,
            "squeeze_duration": int(squeeze_duration[last - 1]) if last >= 1 else 0,
        }

        recent_bb_positions = [float(bb_pos[i]) for i in range(max(0, last - 4), last + 1) if not np.isnan(bb_pos[i])]

        # inline regime 推斷（與 auto_scanner._derive_regime_from_features 對齊）
        adx_v = features.get("adx", 0)
        pct_6 = features.get("pct_6bar", 0)
        bp = features.get("bb_position", 50)
        if adx_v >= 25:
            if pct_6 > 0.5 or bp > 60:
                regime = "trending_up"
            elif pct_6 < -0.5 or bp < 40:
                regime = "trending_down"
            else:
                regime = "ranging"
        else:
            regime = "ranging"

        close = float(closes[last])
        result = classify_bollinger_signal(
            features, prev_features, recent_bb_positions, regime,
            close=close,
            sma20=float(sma20[last]) if not np.isnan(sma20[last]) else close,
            bb_upper=float(bb_upper[last]) if not np.isnan(bb_upper[last]) else close * 1.02,
            bb_lower=float(bb_lower[last]) if not np.isnan(bb_lower[last]) else close * 0.98,
            atr=float(atr14[last]) if not np.isnan(atr14[last]) else close * 0.01,
        )
        if result:
            result["regime_used"] = regime
        return result
    except Exception as e:
        logger.debug(f"[v137] _compute_v136_bollinger 失敗: {e}")
        return None


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
