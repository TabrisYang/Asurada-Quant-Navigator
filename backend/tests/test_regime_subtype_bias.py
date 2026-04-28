"""v104.1：bias_score 9 分量擴展測試。

涵蓋：
- is_btc_pair / is_alt_pair 邊界情境
- market_regime × symbol_type 6 cells 矩陣
- divergence 訊號邏輯（雙背離 / 單背離 / 不一致）
- breadth 軟加分尺度
- settings flag 一鍵回退
- top 3 截斷
- ADA 真實案例（迴歸測試）
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.utils.symbol import is_alt_pair, is_btc_pair


# ─── is_btc_pair / is_alt_pair 分類 ─────────────────────────────


@pytest.mark.parametrize(
    "symbol,expected_btc,expected_alt",
    [
        ("BTC/USDT", True, False),
        ("BTC/USD", True, False),
        ("BTC/USDC", True, False),
        ("ETH/USDT", False, True),
        ("ADA/USDT", False, True),
        ("DOGE/USD", False, True),
        ("USDC/USDT", False, False),  # 穩定幣對穩定幣 → 都不算
        ("USDT/USDC", False, False),
        ("2330/TWD", False, False),  # 台股 → 都不算
        ("AAPL/USD", False, True),  # 暫時把 AAPL 當 alt（後續可細化）
        ("", False, False),
        ("BTCUSDT", True, False),  # normalize_symbol 會自動補 /，"BTC/USDT"
    ],
)
def test_btc_alt_classification(symbol, expected_btc, expected_alt):
    assert is_btc_pair(symbol) is expected_btc, f"is_btc_pair({symbol})"
    assert is_alt_pair(symbol) is expected_alt, f"is_alt_pair({symbol})"


# ─── _compute_bias_score 矩陣 + 訊號邏輯 ─────────────────────────


def _make_synthetic_df(n: int = 100, drift: float = 0.0) -> pd.DataFrame:
    """合成 OHLCV df 給測試用（有控制的微弱 drift）。"""
    import numpy as np
    rng = np.random.default_rng(42)
    closes = 100.0 + drift * pd.Series(range(n)) + rng.normal(0, 0.5, n).cumsum()
    return pd.DataFrame({
        "open": closes - 0.1,
        "high": closes + 0.5,
        "low": closes - 0.5,
        "close": closes,
        "volume": rng.integers(1000, 5000, n),
    })


def test_market_regime_x_alt_season_alt():
    """alt_season + alt 對 → +0.15 偏多"""
    from app.core.regime_subtype import _compute_bias_score
    df = _make_synthetic_df()
    cs = {
        "crossStockSignals": {"market_regime": "alt_season"},
    }
    score, reasons, full = _compute_bias_score(df, cs, symbol="ADA/USDT")
    contribs = full["all_contributions"]
    matrix_contrib = next((c for c in contribs if "alt_season" in c["label"]), None)
    assert matrix_contrib is not None, "alt_season×alt 應產生分量"
    assert matrix_contrib["value"] == 0.15


def test_market_regime_x_alt_season_btc():
    """alt_season + BTC 對 → -0.15 偏空（資金外流）"""
    from app.core.regime_subtype import _compute_bias_score
    df = _make_synthetic_df()
    cs = {"crossStockSignals": {"market_regime": "alt_season"}}
    _, _, full = _compute_bias_score(df, cs, symbol="BTC/USDT")
    contribs = full["all_contributions"]
    matrix = next((c for c in contribs if "alt_season" in c["label"]), None)
    assert matrix is not None
    assert matrix["value"] == -0.15


def test_market_regime_x_btc_led_btc():
    """btc_led + BTC 對 → +0.15"""
    from app.core.regime_subtype import _compute_bias_score
    df = _make_synthetic_df()
    cs = {"crossStockSignals": {"market_regime": "btc_led"}}
    _, _, full = _compute_bias_score(df, cs, symbol="BTC/USDT")
    matrix = next((c for c in full["all_contributions"] if "btc_led" in c["label"]), None)
    assert matrix is not None
    assert matrix["value"] == 0.15


def test_market_regime_x_btc_led_alt():
    """btc_led + alt → -0.15"""
    from app.core.regime_subtype import _compute_bias_score
    df = _make_synthetic_df()
    cs = {"crossStockSignals": {"market_regime": "btc_led"}}
    _, _, full = _compute_bias_score(df, cs, symbol="ADA/USDT")
    matrix = next((c for c in full["all_contributions"] if "btc_led" in c["label"]), None)
    assert matrix is not None
    assert matrix["value"] == -0.15


def test_market_regime_bearish():
    """bearish 不分 symbol → -0.15"""
    from app.core.regime_subtype import _compute_bias_score
    df = _make_synthetic_df()
    cs = {"crossStockSignals": {"market_regime": "bearish"}}
    _, _, full = _compute_bias_score(df, cs, symbol="ETH/USDT")
    matrix = next((c for c in full["all_contributions"] if "bearish" in c["label"]), None)
    assert matrix is not None
    assert matrix["value"] == -0.15


def test_market_regime_mixed_no_contribution():
    """mixed → 不加分"""
    from app.core.regime_subtype import _compute_bias_score
    df = _make_synthetic_df()
    cs = {"crossStockSignals": {"market_regime": "mixed"}}
    _, _, full = _compute_bias_score(df, cs, symbol="ETH/USDT")
    matrix = next((c for c in full["all_contributions"] if "mixed" in c["label"]), None)
    assert matrix is None  # mixed 完全不加分


def test_market_regime_stablecoin_pair_no_contribution():
    """穩定幣對穩定幣（USDC/USDT）→ 矩陣完全不加分"""
    from app.core.regime_subtype import _compute_bias_score
    df = _make_synthetic_df()
    cs = {"crossStockSignals": {"market_regime": "alt_season"}}
    _, _, full = _compute_bias_score(df, cs, symbol="USDC/USDT")
    matrix = next((c for c in full["all_contributions"] if "alt_season" in c["label"]), None)
    assert matrix is None  # 穩定幣對不該被矩陣加分


# ─── divergence 訊號 ─────────────────────────


def test_divergence_double_bullish():
    """RSI + MACD 雙底背離（看漲）→ +0.20"""
    from app.core.regime_subtype import _compute_bias_score
    df = _make_synthetic_df()
    cs = {"indicatorValues": {"RSI_Div": 1.0, "MACD_Div": 1.0}}
    _, _, full = _compute_bias_score(df, cs)
    div = next((c for c in full["all_contributions"] if "雙背離↑" in c["label"]), None)
    assert div is not None
    assert div["value"] == 0.20


def test_divergence_double_bearish():
    """RSI + MACD 雙頂背離（看跌）→ -0.20"""
    from app.core.regime_subtype import _compute_bias_score
    df = _make_synthetic_df()
    cs = {"indicatorValues": {"RSI_Div": -1.0, "MACD_Div": -1.0}}
    _, _, full = _compute_bias_score(df, cs)
    div = next((c for c in full["all_contributions"] if "雙背離↓" in c["label"]), None)
    assert div is not None
    assert div["value"] == -0.20


def test_divergence_single_only():
    """只有一個背離 → ±0.10"""
    from app.core.regime_subtype import _compute_bias_score
    df = _make_synthetic_df()
    cs = {"indicatorValues": {"RSI_Div": 1.0, "MACD_Div": 0.0}}
    _, _, full = _compute_bias_score(df, cs)
    div = next((c for c in full["all_contributions"] if "RSI 背離↑" in c["label"]), None)
    assert div is not None
    assert div["value"] == 0.10


def test_divergence_inconsistent_cancels():
    """RSI+1 + MACD-1（不一致）→ 不加分"""
    from app.core.regime_subtype import _compute_bias_score
    df = _make_synthetic_df()
    cs = {"indicatorValues": {"RSI_Div": 1.0, "MACD_Div": -1.0}}
    _, _, full = _compute_bias_score(df, cs)
    # 不應該有任何 divergence 分量
    div_contribs = [c for c in full["all_contributions"] if "背離" in c["label"]]
    assert len(div_contribs) == 0


# ─── breadth 軟加分尺度 ─────────────────────────


@pytest.mark.parametrize(
    "breadth_pct,expected_value",
    [
        (70, 0.20),    # ≥ 65 → 0.20
        (65, 0.20),    # 邊界
        (60, 0.15),    # 線性中段
        (55, 0.10),    # 邊界
        (50, None),    # 中性，不加分
        (45, -0.10),   # 邊界
        (40, -0.15),   # 線性中段
        (35, -0.20),   # 邊界
        (30, -0.20),   # ≤ 35 → -0.20
    ],
)
def test_breadth_soft_scale(breadth_pct, expected_value):
    from app.core.regime_subtype import _compute_bias_score
    df = _make_synthetic_df()
    cs = {"crossStockSignals": {"breadth_pct_advancing": breadth_pct}}
    _, _, full = _compute_bias_score(df, cs)
    breadth_c = next((c for c in full["all_contributions"] if "breadth" in c["label"]), None)
    if expected_value is None:
        assert breadth_c is None, f"breadth={breadth_pct} 不該加分"
    else:
        assert breadth_c is not None
        assert abs(breadth_c["value"] - expected_value) < 0.001


# ─── settings flag 回退 ─────────────────────────


def test_settings_flag_fallback_to_5_dims(monkeypatch):
    """flag=False 時走原 5 分量行為（無 RS / 無 matrix / 無 divergence）"""
    from app.core.regime_subtype import _compute_bias_score
    df = _make_synthetic_df()
    cs = {
        "crossStockSignals": {
            "market_regime": "alt_season",
            "relative_strength_vs_basket": 1.5,
            "breadth_pct_advancing": 60,
        },
        "indicatorValues": {"RSI_Div": 1.0, "MACD_Div": 1.0},
    }
    # patch settings flag
    from app.core.config.settings import settings
    monkeypatch.setattr(settings, "bias_score_extended_dimensions", False)

    _, _, full = _compute_bias_score(df, cs, symbol="ADA/USDT")
    assert full["extended_dimensions"] is False
    # 不應該有新分量（matrix / RS / divergence）
    new_dim_labels = ["alt_season", "RS=", "雙背離", "RSI 背離", "MACD 背離"]
    for c in full["all_contributions"]:
        for label in new_dim_labels:
            assert label not in c["label"], f"flag=False 不該有 {label}: {c}"


# ─── top 3 截斷 ─────────────────────────


def test_top_3_truncation():
    """超過 3 個分量時，bias_reasons 只回傳 |contribution| 最大的 top 3"""
    from app.core.regime_subtype import _compute_bias_score
    df = _make_synthetic_df()
    # 製造 4+ 個正分量
    cs = {
        "crossStockSignals": {
            "market_regime": "alt_season",  # +0.15
            "relative_strength_vs_basket": 1.5,  # +0.10
            "breadth_pct_advancing": 60,  # +0.15 軟加
        },
        "indicatorValues": {"RSI_Div": 1.0, "MACD_Div": 1.0},  # +0.20
        "external_signals": {"derivatives": {"funding_rate_pct": -0.1}},  # +0.15
    }
    score, reasons, full = _compute_bias_score(df, cs, symbol="ETH/USDT")
    assert len(reasons) <= 3, f"top reasons 應 ≤ 3，實際 {len(reasons)}: {reasons}"
    # full metrics 應保留所有分量
    assert len(full["all_contributions"]) >= 4


# ─── 迴歸測試：ADA 案例 ─────────────────────────


def test_ada_regression_lean_long():
    """ADA/USDT ranging + 多個偏多訊號 → bias_score ≥ 0.4 → lean_long

    這是 v104 修正前被誤判為 bilateral 的真實案例。
    """
    from app.core.regime_subtype import _compute_bias_score
    df = _make_synthetic_df(n=200)  # 需 60+ 給 EMA60 用
    cs = {
        "crossStockSignals": {
            "breadth_pct_advancing": 60,
            "relative_strength_vs_basket": 1.6,
            "market_regime": "alt_season",
        },
        "external_signals": {"derivatives": {}, "sentiment": {}},
        "indicatorValues": {"RSI_Div": 1.0, "MACD_Div": 1.0},
    }
    score, reasons, full = _compute_bias_score(df, cs, symbol="ADA/USDT")
    # 預期至少 4 個正分量：雙背離 0.20 + matrix 0.15 + breadth 0.15 + RS 0.10 ≈ 0.60
    assert score >= 0.4, f"ADA 案例 bias_score 應 ≥ 0.4 觸發 lean_long，實際 {score}"
    assert len(reasons) <= 3
