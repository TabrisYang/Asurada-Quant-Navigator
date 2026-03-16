"""技術指標計算測試"""

import numpy as np
import pandas as pd
import pytest

from app.core.indicators.registry import registry


def _make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """生成模擬 OHLCV 數據"""
    rng = np.random.RandomState(seed)
    close = 100 + np.cumsum(rng.randn(n) * 0.5)
    close = np.maximum(close, 1.0)
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    low = np.maximum(low, 0.01)
    open_ = close + rng.randn(n) * 0.3
    volume = rng.uniform(1000, 10000, n)
    ts = pd.date_range("2024-01-01", periods=n, freq="1h")

    return pd.DataFrame({
        "timestamp": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


class TestIndicatorRegistry:
    """測試指標註冊中心"""

    def test_all_indicators_registered(self):
        all_inds = registry.list_all()
        assert len(all_inds) >= 20, f"應至少有 20 個指標，目前 {len(all_inds)}"

    def test_info_list_structure(self):
        info = registry.to_info_list()
        for item in info:
            assert "id" in item
            assert "name" in item
            assert "category" in item
            assert "parameters" in item
            assert "display_mode" in item

    def test_unknown_indicator_returns_none(self):
        df = _make_ohlcv(50)
        result = registry.calculate("nonexistent_indicator_xyz", df)
        assert result is None

    def test_empty_dataframe_returns_none(self):
        df = pd.DataFrame()
        result = registry.calculate("rsi", df)
        assert result is None

    def test_insufficient_data_returns_none(self):
        df = _make_ohlcv(3)
        result = registry.calculate("adx", df, {"period": 14})
        assert result is None


class TestRSI:
    def test_rsi_returns_values(self):
        df = _make_ohlcv(200)
        result = registry.calculate("rsi", df, {"period": 14})
        assert result is not None
        assert "RSI" in result
        assert len(result["RSI"]) == 200

    def test_rsi_range(self):
        df = _make_ohlcv(200)
        result = registry.calculate("rsi", df, {"period": 14})
        values = [v for v in result["RSI"] if v is not None]
        for v in values:
            assert 0 <= v <= 100, f"RSI 超出範圍: {v}"

    def test_rsi_no_divide_by_zero(self):
        """純漲行情（avg_loss=0）不應拋出異常"""
        n = 50
        close = np.linspace(100, 200, n)
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.ones(n) * 1000,
        })
        result = registry.calculate("rsi", df, {"period": 14})
        assert result is not None
        values = [v for v in result["RSI"] if v is not None]
        assert all(v >= 90 for v in values[-10:])


class TestADX:
    def test_adx_returns_values(self):
        df = _make_ohlcv(200)
        result = registry.calculate("adx", df, {"period": 14})
        assert result is not None
        assert "ADX" in result
        assert "+DI" in result
        assert "-DI" in result

    def test_adx_no_divide_by_zero(self):
        """平盤行情 (+DI + -DI 接近 0) 不應拋出異常"""
        n = 100
        base = 100.0
        close = np.full(n, base)
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "open": np.full(n, base),
            "high": np.full(n, base + 0.001),
            "low": np.full(n, base - 0.001),
            "close": close,
            "volume": np.ones(n) * 1000,
        })
        result = registry.calculate("adx", df, {"period": 14})
        assert result is not None


class TestMACD:
    def test_macd_returns_values(self):
        df = _make_ohlcv(200)
        result = registry.calculate("macd", df)
        assert result is not None
        assert "MACD" in result
        assert "Signal" in result
        assert "Histogram" in result


class TestBollingerBands:
    def test_bb_returns_values(self):
        df = _make_ohlcv(200)
        result = registry.calculate("bb", df, {"period": 20, "std_dev": 2.0})
        assert result is not None
        assert "BB_Upper" in result or "Upper" in result


class TestAllIndicators:
    """確認所有已註冊指標都能正常計算（不拋異常）"""

    def test_all_indicators_calculate_without_error(self):
        df = _make_ohlcv(300)
        for ind in registry.list_all():
            if ind.data_source != "ohlcv":
                continue
            result = registry.calculate(ind.id, df)
            assert result is None or isinstance(result, dict), (
                f"指標 {ind.id} 回傳非預期類型: {type(result)}"
            )
