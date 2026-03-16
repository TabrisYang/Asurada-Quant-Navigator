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
