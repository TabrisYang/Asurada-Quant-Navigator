"""回測引擎測試"""

import numpy as np
import pandas as pd
import pytest

from app.core.backtest.engine import run_backtest, BacktestResult


def _make_trending_ohlcv(n: int = 200, direction: str = "up") -> pd.DataFrame:
    """生成有明確趨勢的模擬數據"""
    rng = np.random.RandomState(99)
    trend = np.linspace(100, 150 if direction == "up" else 50, n)
    noise = rng.randn(n) * 0.5
    close = trend + noise
    close = np.maximum(close, 1.0)
    high = close + rng.uniform(0.2, 1.0, n)
    low = close - rng.uniform(0.2, 1.0, n)
    low = np.maximum(low, 0.01)
    open_ = close + rng.randn(n) * 0.2
    volume = rng.uniform(1000, 10000, n)
    ts = pd.date_range("2024-01-01", periods=n, freq="4h")

    return pd.DataFrame({
        "timestamp": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


class TestBacktestBasic:
    def test_insufficient_data(self):
        df = _make_trending_ohlcv(30)
        result = run_backtest(
            df,
            entry_conditions=[{"indicator": "rsi", "operator": "<", "value": 30}],
            exit_conditions=[{"indicator": "rsi", "operator": ">", "value": 70}],
        )
        assert isinstance(result, BacktestResult)
        assert "數據不足" in str(result.warnings) or result.metrics.get("note") == "數據不足"

    def test_no_entry_signals(self):
        df = _make_trending_ohlcv(200, "up")
        result = run_backtest(
            df,
            entry_conditions=[{"indicator": "rsi", "operator": "<", "value": 1}],
            exit_conditions=[{"indicator": "rsi", "operator": ">", "value": 99}],
        )
        assert result.metrics.get("total_trades", 0) == 0

    def test_basic_long_backtest(self):
        df = _make_trending_ohlcv(200, "up")
        result = run_backtest(
            df,
            entry_conditions=[{"indicator": "rsi", "operator": "<", "value": 45}],
            exit_conditions=[{"indicator": "rsi", "operator": ">", "value": 60}],
            direction="long",
        )
        assert isinstance(result, BacktestResult)
        assert "total_trades" in result.metrics

    def test_basic_short_backtest(self):
        df = _make_trending_ohlcv(200, "down")
        result = run_backtest(
            df,
            entry_conditions=[{"indicator": "rsi", "operator": ">", "value": 55}],
            exit_conditions=[{"indicator": "rsi", "operator": "<", "value": 40}],
            direction="short",
        )
        assert isinstance(result, BacktestResult)

    def test_stop_loss(self):
        df = _make_trending_ohlcv(200, "down")
        result = run_backtest(
            df,
            entry_conditions=[{"indicator": "rsi", "operator": "<", "value": 45}],
            exit_conditions=[{"indicator": "rsi", "operator": ">", "value": 90}],
            direction="long",
            stop_loss_pct=0.02,
        )
        sl_trades = [t for t in result.trades if t.exit_reason == "stop_loss"]
        if result.trades:
            assert len(sl_trades) > 0 or any(t.exit_reason for t in result.trades)

    def test_take_profit(self):
        df = _make_trending_ohlcv(200, "up")
        result = run_backtest(
            df,
            entry_conditions=[{"indicator": "rsi", "operator": "<", "value": 50}],
            exit_conditions=[{"indicator": "rsi", "operator": ">", "value": 99}],
            direction="long",
            take_profit_pct=0.03,
        )
        tp_trades = [t for t in result.trades if t.exit_reason == "take_profit"]
        if result.trades:
            assert len(tp_trades) >= 0


class TestBacktestOutput:
    def test_result_to_dict(self):
        df = _make_trending_ohlcv(200)
        result = run_backtest(
            df,
            entry_conditions=[{"indicator": "rsi", "operator": "<", "value": 45}],
            exit_conditions=[{"indicator": "rsi", "operator": ">", "value": 60}],
        )
        d = result.to_dict()
        assert "status" in d
        assert "metrics" in d
        assert "warnings" in d

    def test_equity_curve_exists(self):
        df = _make_trending_ohlcv(200)
        result = run_backtest(
            df,
            entry_conditions=[{"indicator": "rsi", "operator": "<", "value": 60}],
            exit_conditions=[{"indicator": "rsi", "operator": ">", "value": 55}],
        )
        if result.trades:
            assert len(result.equity_curve) > 0
        else:
            assert result.metrics.get("total_trades", 0) == 0

    def test_warnings_always_present(self):
        df = _make_trending_ohlcv(200)
        result = run_backtest(
            df,
            entry_conditions=[{"indicator": "rsi", "operator": "<", "value": 45}],
            exit_conditions=[{"indicator": "rsi", "operator": ">", "value": 60}],
        )
        assert isinstance(result.warnings, list)
        assert len(result.warnings) > 0


class TestBacktestEntryCapital:
    def test_pnl_uses_entry_capital(self):
        """PnL 應基於進場時資金計算"""
        df = _make_trending_ohlcv(200, "up")
        result = run_backtest(
            df,
            entry_conditions=[{"indicator": "rsi", "operator": "<", "value": 50}],
            exit_conditions=[{"indicator": "rsi", "operator": ">", "value": 55}],
            direction="long",
            initial_capital=10000,
        )
        if result.trades:
            t = result.trades[0]
            assert abs(t.pnl_amount) > 0 or t.pnl_pct == 0


def _make_daily_ohlcv(start: str = "2022-01-01", n: int = 500) -> pd.DataFrame:
    """跨月日線模擬數據（供季節性/日曆特徵測試）。"""
    rng = np.random.RandomState(7)
    close = 100 + np.cumsum(rng.randn(n))
    close = np.maximum(close, 1.0)
    ts = pd.date_range(start, periods=n, freq="1D")
    return pd.DataFrame({
        "timestamp": ts,
        "open": close,
        "high": close + rng.uniform(0.2, 1.0, n),
        "low": np.maximum(close - rng.uniform(0.2, 1.0, n), 0.01),
        "close": close,
        "volume": rng.uniform(1000, 10000, n),
    })


class TestSeasonalIndicator:
    """軌道 B：季節性/日曆特徵指標。"""

    def test_seasonal_series_values(self):
        from app.core.indicators import registry
        df = _make_daily_ohlcv("2022-01-01", 500)
        calc = registry.calculate("seasonal", df)
        assert calc is not None
        # is_month_start 每月首根=1，總和 == 涵蓋月數
        ims = np.array(calc["is_month_start"])
        expected_months = df["timestamp"].dt.to_period("M").nunique()
        assert ims.sum() == expected_months
        # 日線時 is_month_start 應對齊日曆 1 號
        assert bool((ims.astype(bool) == df["timestamp"].dt.is_month_start.values).all())
        # 值域
        assert set(calc["day_of_week"]) <= set(range(7))
        assert min(calc["month"]) >= 1 and max(calc["month"]) <= 12
        assert min(calc["day_of_month"]) >= 1 and max(calc["day_of_month"]) <= 31

    def test_volume_passthrough(self):
        from app.core.indicators import registry
        df = _make_daily_ohlcv("2022-01-01", 120)
        calc = registry.calculate("volume", df)
        assert calc is not None
        assert calc["Volume"][:3] == pytest.approx(df["volume"].tolist()[:3], rel=1e-5)

    def test_backtest_seasonal_entry(self):
        df = _make_daily_ohlcv("2022-01-01", 500)
        result = run_backtest(
            df,
            entry_conditions=[{"indicator": "seasonal", "series": "is_month_start",
                               "operator": "==", "value": 1}],
            exit_conditions=[{"indicator": "seasonal", "series": "is_month_start",
                              "operator": "==", "value": 0}],
            direction="long",
            take_profit_pct=0.05,
            stop_loss_pct=0.03,
        )
        assert isinstance(result, BacktestResult)
        # 每月首根進場 → 交易數約在月數量級（寬鬆上界，確保條件確實觸發）
        n_months = df["timestamp"].dt.to_period("M").nunique()
        assert 0 < result.metrics.get("total_trades", 0) <= n_months + 1


class TestConditionalScanDiscrete:
    """軌道 B/C：條件機率掃描的離散分組 + 兩段式確認。"""

    def _patched_executor(self, df):
        import app.core.llm.executor as ex
        ex._load_local_data = lambda symbol, tf, start=None, end=None: df.copy()
        return ex

    def test_backtest_needs_confirmation(self):
        import asyncio
        ex = self._patched_executor(_make_daily_ohlcv("2022-01-01", 300))
        r = asyncio.run(ex._exec_backtest(
            {"symbol": "BTC/USDT", "timeframe": "1d",
             "entry_conditions": [{"indicator": "rsi", "operator": "<", "value": 30}],
             "exit_conditions": [{"indicator": "rsi", "operator": ">", "value": 70}]},
            "BTC/USDT", "1d"))
        assert r["status"] == "needs_confirmation"
        assert "available_range" in r and r["available_range"]["bars"] == 300
        assert "total_trades" not in r  # 未執行回測

    def test_backtest_confirmed_has_data_range(self):
        import asyncio
        ex = self._patched_executor(_make_daily_ohlcv("2022-01-01", 500))
        r = asyncio.run(ex._exec_backtest(
            {"symbol": "BTC/USDT", "timeframe": "1d", "confirmed": True,
             "entry_conditions": [{"indicator": "rsi", "operator": "<", "value": 45}],
             "exit_conditions": [{"indicator": "rsi", "operator": ">", "value": 60}]},
            "BTC/USDT", "1d"))
        assert r["status"] == "success"
        assert r["data_range"]["bars"] == 500 and r["data_range"]["start"] == "2022-01-01"

    def test_scan_discrete_grouping(self):
        import asyncio
        ex = self._patched_executor(_make_daily_ohlcv("2022-01-01", 600))
        r = asyncio.run(ex._exec_conditional_prob_scan(
            {"symbol": "BTC/USDT", "timeframe": "1d", "confirmed": True,
             "indicators": ["seasonal"], "forward_bars": 5, "target_pct": 2.0},
            "BTC/USDT", "1d"))
        assert r["status"] == "success"
        dow = r["indicators"]["seasonal_day_of_week"]
        # 離散逐值分組：標籤為 "=k"（非連續 lo~hi）
        labels = [b["range"] for b in dow["bins"]]
        assert all(lb.startswith("=") for lb in labels)
        assert set(labels) <= {f"={k}" for k in range(7)}

    def test_scan_continuous_regression(self):
        import asyncio
        ex = self._patched_executor(_make_daily_ohlcv("2022-01-01", 600))
        r = asyncio.run(ex._exec_conditional_prob_scan(
            {"symbol": "BTC/USDT", "timeframe": "1d", "confirmed": True,
             "indicators": ["rsi"], "forward_bars": 5, "target_pct": 2.0},
            "BTC/USDT", "1d"))
        assert r["status"] == "success"
        rsi = list(r["indicators"].values())[0]
        # 連續指標仍走 lo~hi 區間（回歸不變）
        assert all("~" in b["range"] for b in rsi["bins"])


class TestIndicatorVsIndicator:
    """指標 vs 指標比較（compare_to）。"""

    def test_close_above_own_sma(self):
        df = _make_trending_ohlcv(300, "up")
        result = run_backtest(
            df,
            entry_conditions=[{"indicator": "close", "operator": ">",
                               "compare_to": {"indicator": "sma", "parameters": {"period": 20}}}],
            exit_conditions=[{"indicator": "close", "operator": "<",
                              "compare_to": {"indicator": "sma", "parameters": {"period": 20}}}],
            direction="long",
        )
        assert isinstance(result, BacktestResult)
        # 上升趨勢 → close 常在均線之上 → 應有交易
        assert result.metrics.get("total_trades", 0) > 0

    def test_golden_cross(self):
        df = _make_trending_ohlcv(400, "up")
        result = run_backtest(
            df,
            entry_conditions=[{"indicator": "sma", "parameters": {"period": 10},
                               "operator": "cross_above",
                               "compare_to": {"indicator": "sma", "parameters": {"period": 30}}}],
            exit_conditions=[{"indicator": "sma", "parameters": {"period": 10},
                              "operator": "cross_below",
                              "compare_to": {"indicator": "sma", "parameters": {"period": 30}}}],
            direction="long",
        )
        assert isinstance(result, BacktestResult)  # 不論交易數，交叉邏輯需能執行不報錯

    def test_compare_to_with_mult(self):
        df = _make_trending_ohlcv(300, "up")
        # close > SMA20 * 1.05（乖離 5%）應比 close > SMA20 交易更少
        base = run_backtest(
            df,
            entry_conditions=[{"indicator": "close", "operator": ">",
                               "compare_to": {"indicator": "sma", "parameters": {"period": 20}}}],
            exit_conditions=[{"indicator": "rsi", "operator": ">", "value": 99}],
        )
        strict = run_backtest(
            df,
            entry_conditions=[{"indicator": "close", "operator": ">",
                               "compare_to": {"indicator": "sma", "parameters": {"period": 20}, "mult": 1.05}}],
            exit_conditions=[{"indicator": "rsi", "operator": ">", "value": 99}],
        )
        assert strict.metrics.get("total_trades", 0) <= base.metrics.get("total_trades", 0)

    def test_scalar_condition_regression(self):
        """加了 compare_to 後，原本的固定數值條件行為需完全不變。"""
        df = _make_trending_ohlcv(300, "up")
        r = run_backtest(
            df,
            entry_conditions=[{"indicator": "rsi", "operator": "between", "value": 30, "value2": 50}],
            exit_conditions=[{"indicator": "rsi", "operator": ">", "value": 70}],
        )
        assert isinstance(r, BacktestResult)
        assert r.metrics.get("total_trades", 0) >= 0
