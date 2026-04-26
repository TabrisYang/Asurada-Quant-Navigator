"""Ladder 進場回測測試（v99）— 驗證 ladder_config 邏輯正確 + 對照單進場可重現。"""

import numpy as np
import pandas as pd

from app.core.backtest.engine import run_backtest
from app.core.laddered_entries import compute_laddered_entries


def _choppy_ohlcv(n: int = 250, seed: int = 42) -> pd.DataFrame:
    """合成 choppy（震盪）K 線 — RSI 容易觸極值，方便回測有 trade 產生。"""
    rng = np.random.RandomState(seed)
    close = 60000 + 5000 * np.sin(np.linspace(0, 6 * np.pi, n)) + np.cumsum(rng.normal(0, 80, n))
    high = close + np.abs(rng.normal(0, 200, n))
    low = close - np.abs(rng.normal(0, 200, n))
    open_ = np.roll(close, 1); open_[0] = close[0]
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="D"),
        "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.uniform(1000, 5000, n),
    })


def _rsi_strategy() -> tuple[list, list]:
    entry = [{"indicator": "rsi", "series": "RSI", "operator": "<", "value": 40}]
    exit_ = [{"indicator": "rsi", "series": "RSI", "operator": ">", "value": 60}]
    return entry, exit_


def test_compute_laddered_entries_trending_up_returns_pyramid():
    """trending_up + confidence > 0.5 應回 50/30/20 + 三檔有效。"""
    df = _choppy_ohlcv(200)
    r = compute_laddered_entries(df, direction="long", regime="trending_up", regime_confidence=0.7, smc_long_entry=df["close"].iloc[-1])
    assert r["enabled"] is True
    assert len(r["long_entries"]) == 3
    assert r["long_entries"][0]["size_pct"] == 50  # 金字塔首檔重
    assert r["short_entries"] == []  # trending_up 不開短
    assert r["weighted_avg_entry_long"] is not None
    assert r["stop_loss_long"] < r["weighted_avg_entry_long"]
    assert r["take_profit_long"] > r["weighted_avg_entry_long"]


def test_compute_laddered_entries_low_confidence_disabled():
    """confidence < 0.5 → enabled = False + warning。"""
    df = _choppy_ohlcv(200)
    r = compute_laddered_entries(df, regime="trending_up", regime_confidence=0.3)
    assert r["enabled"] is False
    assert "confidence" in r["warning"].lower()


def test_compute_laddered_entries_ranging_inverse_pyramid():
    """ranging → 倒金字塔 25/35/40 + 多空都開。"""
    df = _choppy_ohlcv(200)
    r = compute_laddered_entries(df, direction="both", regime="ranging", regime_confidence=0.65)
    assert r["enabled"] is True
    assert len(r["long_entries"]) > 0 and len(r["short_entries"]) > 0
    assert r["long_entries"][0]["size_pct"] == 25  # 倒金字塔首檔輕


def test_run_backtest_ladder_mode_produces_trades():
    """Ladder 模式應產生 trades，且每筆 trade.entry_legs 含實際 fills。"""
    df = _choppy_ohlcv(250)
    entry, exit_ = _rsi_strategy()
    cfg = {"enabled": True, "ratios": [50, 30, 20], "price_offsets_pct": [0.0, -2.0, -4.0], "max_wait_bars": 10}
    r = run_backtest(df, entry, exit_, direction="long", stop_loss_pct=0.05, take_profit_pct=0.10, ladder_config=cfg)

    assert r.metrics["total_trades"] > 0
    assert "ladder" in r.metrics
    assert r.metrics["ladder"]["n_legs_planned"] == 3
    assert r.metrics["ladder"]["avg_fills_per_trade"] >= 1

    # 每筆 trade 都應有 entry_legs metadata
    for t in r.trades:
        assert t.entry_legs is not None
        assert t.fill_count >= 1
        assert sum(leg["ratio_pct"] for leg in t.entry_legs) > 0


def test_run_backtest_single_vs_ladder_consistency():
    """單進場 vs ladder 都跑得出 trades，且兩者 metrics 結構相容。"""
    df = _choppy_ohlcv(250)
    entry, exit_ = _rsi_strategy()

    r_single = run_backtest(df, entry, exit_, direction="long", stop_loss_pct=0.05, take_profit_pct=0.10)
    r_ladder = run_backtest(df, entry, exit_, direction="long", stop_loss_pct=0.05, take_profit_pct=0.10,
                            ladder_config={"enabled": True, "ratios": [50, 30, 20], "price_offsets_pct": [0.0, -2.0, -4.0]})

    # 兩個都應有 trades
    assert r_single.metrics["total_trades"] > 0
    assert r_ladder.metrics["total_trades"] > 0

    # 共用 metric keys 必須都存在
    for key in ["total_trades", "win_rate", "total_return_pct", "max_drawdown_pct", "sharpe_ratio"]:
        assert key in r_single.metrics
        assert key in r_ladder.metrics

    # ladder 模式專屬 metrics 只有 ladder 模式才有
    assert "ladder" not in r_single.metrics
    assert "ladder" in r_ladder.metrics


def test_run_backtest_ladder_disabled_falls_back():
    """ladder_config = None 或 enabled = False → 走單進場主迴圈。"""
    df = _choppy_ohlcv(250)
    entry, exit_ = _rsi_strategy()

    # None
    r1 = run_backtest(df, entry, exit_, direction="long", stop_loss_pct=0.05, ladder_config=None)
    assert "ladder" not in r1.metrics

    # enabled = False
    r2 = run_backtest(df, entry, exit_, direction="long", stop_loss_pct=0.05,
                     ladder_config={"enabled": False, "ratios": [50, 30, 20]})
    assert "ladder" not in r2.metrics
