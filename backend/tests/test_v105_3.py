"""v105.3 unit tests：ladder logic + RR 正確性 + parser regex 對稱性。

涵蓋：
- ladder 方向約束（long ≤ 現價、short ≥ 現價）
- ladder 倉位順序（倒金字塔接刀）
- ladder spacing（≥ 0.5×ATR）
- ladder 邊界（ATR=0、單檔）
- RR 計算正確（reward / risk 而非 max(tp_mult, MIN_RR)）
- bilateral 子段 regex 對稱
- _clamp_to_current_price / _reassign_ratios_by_position 行為
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ─── 1. ladder 方向約束 ──────────────────────────────


def _make_df(n: int = 100, base: float = 0.25, vol: float = 0.005) -> pd.DataFrame:
    """合成 OHLCV df。"""
    rng = np.random.default_rng(42)
    closes = base + rng.normal(0, vol, n).cumsum() * 0.1
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="4h"),
        "open": closes - 0.001,
        "high": closes + 0.003,
        "low": closes - 0.003,
        "close": closes,
        "volume": rng.integers(1000, 5000, n),
    })


@pytest.mark.parametrize("regime", ["ranging", "low_vol", "high_vol", "trending_up"])
def test_long_entries_at_or_below_current(regime):
    """所有 long entry 必須 ≤ 現價（v105.2 Bug A 防 regression）。"""
    from app.core.laddered_entries import compute_laddered_entries
    df = _make_df()
    current = float(df["close"].iloc[-1])
    result = compute_laddered_entries(
        df=df, direction="long", regime=regime, regime_confidence=0.7,
        timeframe_str="4h", confidence_label="medium",
    )
    if not result.get("enabled"):
        pytest.skip(f"regime={regime} ladder disabled")
    for e in result.get("long_entries", []):
        assert e["price"] <= current * 1.001, (
            f"regime={regime} long entry ${e['price']} > 現價 ${current}（容忍 0.1% 浮點）"
        )


@pytest.mark.parametrize("regime", ["ranging", "low_vol", "high_vol", "trending_down"])
def test_short_entries_at_or_above_current(regime):
    """所有 short entry 必須 ≥ 現價。"""
    from app.core.laddered_entries import compute_laddered_entries
    df = _make_df()
    current = float(df["close"].iloc[-1])
    result = compute_laddered_entries(
        df=df, direction="short", regime=regime, regime_confidence=0.7,
        timeframe_str="4h", confidence_label="medium",
    )
    if not result.get("enabled"):
        pytest.skip(f"regime={regime} ladder disabled")
    for e in result.get("short_entries", []):
        assert e["price"] >= current * 0.999, (
            f"regime={regime} short entry ${e['price']} < 現價 ${current}（容忍 0.1% 浮點）"
        )


# ─── 2. 倉位順序（倒金字塔接刀）─────────────────────────


def test_long_inverse_pyramid_largest_size_at_lowest_price():
    """v105.2 Bug B：long 最便宜的進場價配最重倉（接刀越深倉位越重）。"""
    from app.core.laddered_entries import compute_laddered_entries
    df = _make_df()
    result = compute_laddered_entries(
        df=df, direction="long", regime="ranging", regime_confidence=0.7,
        timeframe_str="4h", confidence_label="medium",
    )
    entries = result.get("long_entries", [])
    if len(entries) < 2:
        pytest.skip("ladder 不足 2 檔")
    # 找最便宜的那檔
    cheapest = min(entries, key=lambda e: e["price"])
    expensive = max(entries, key=lambda e: e["price"])
    assert cheapest["size_pct"] >= expensive["size_pct"], (
        f"長倉接刀：最便宜檔 size {cheapest['size_pct']}% 應 ≥ 最貴檔 {expensive['size_pct']}%"
    )


def test_short_inverse_pyramid_largest_size_at_highest_price():
    """short 最貴的進場價配最重倉。"""
    from app.core.laddered_entries import compute_laddered_entries
    df = _make_df()
    result = compute_laddered_entries(
        df=df, direction="short", regime="ranging", regime_confidence=0.7,
        timeframe_str="4h", confidence_label="medium",
    )
    entries = result.get("short_entries", [])
    if len(entries) < 2:
        pytest.skip("ladder 不足 2 檔")
    expensive = max(entries, key=lambda e: e["price"])
    cheap = min(entries, key=lambda e: e["price"])
    assert expensive["size_pct"] >= cheap["size_pct"], (
        f"空倉接刀：最貴檔 size {expensive['size_pct']}% 應 ≥ 最便宜檔 {cheap['size_pct']}%"
    )


# ─── 3. ladder spacing ──────────────────────────


def test_ladder_minimum_spacing():
    """v105.1：相鄰兩檔距離 ≥ 0.5×ATR（避免 BB/Donchian 重合）。"""
    from app.core.laddered_entries import compute_laddered_entries
    df = _make_df()
    result = compute_laddered_entries(
        df=df, direction="long", regime="ranging", regime_confidence=0.7,
        timeframe_str="4h", confidence_label="medium",
    )
    entries = result.get("long_entries", [])
    atr = result.get("atr_used") or 0
    if len(entries) < 2 or atr <= 0:
        pytest.skip("樣本不足驗證 spacing")
    sorted_entries = sorted(entries, key=lambda e: -e["price"])
    for i in range(1, len(sorted_entries)):
        gap = sorted_entries[i - 1]["price"] - sorted_entries[i]["price"]
        # 容忍浮點誤差 1e-6
        assert gap >= atr * 0.5 - 1e-6, (
            f"第 {i} 檔跟第 {i+1} 檔距離 {gap} < 0.5×ATR ({atr*0.5})"
        )


# ─── 4. ladder 邊界 ──────────────────────────


def test_ladder_skipped_when_low_confidence():
    """confidence < 0.5 應該回 enabled=False + 提供 SL/TP 倍數提示。"""
    from app.core.laddered_entries import compute_laddered_entries
    df = _make_df()
    result = compute_laddered_entries(
        df=df, direction="long", regime="ranging", regime_confidence=0.3,
        timeframe_str="1d", confidence_label="low",
    )
    assert result["enabled"] is False
    # v104.2：應提供 fallback hints
    assert "sl_mult_hint" in result
    assert "tp_mult_hint" in result
    # 1d × low confidence：SL 倍數應接近 2.5（基礎）×0.8（信心修正）= 2.0
    assert 1.5 <= result["sl_mult_hint"] <= 3.0


def test_ladder_handles_empty_df():
    """df 太短應該 graceful return，不爆。"""
    from app.core.laddered_entries import compute_laddered_entries
    short_df = _make_df(n=10)  # 只有 10 根
    result = compute_laddered_entries(
        df=short_df, direction="long", regime="ranging", regime_confidence=0.7,
        timeframe_str="4h",
    )
    assert result["enabled"] is False
    assert "資料不足" in result.get("warning", "")


# ─── 5. RR 計算正確性 ──────────────────────────


def test_rr_calculated_from_actual_distances():
    """v105.3 Bug C：RR 必須是 (tp - avg) / (avg - sl)，不是 max(tp_mult, MIN_RR)。"""
    from app.core.laddered_entries import compute_laddered_entries
    df = _make_df()
    result = compute_laddered_entries(
        df=df, direction="long", regime="ranging", regime_confidence=0.7,
        timeframe_str="4h", confidence_label="medium",
    )
    entries = result.get("long_entries", [])
    if not entries:
        pytest.skip("ladder 不啟用")
    avg = result["weighted_avg_entry_long"]
    sl = result["stop_loss_long"]
    tp = result["take_profit_long"]
    rr_reported = result["rr_long"]
    # 算實際 reward / risk
    risk = avg - sl
    reward = tp - avg
    if risk > 0:
        rr_actual = reward / risk
        # 容忍 round 誤差
        assert abs(rr_reported - rr_actual) < 0.05, (
            f"報告 RR={rr_reported}, 實際 reward/risk={rr_actual:.2f}（reward={reward}, risk={risk}）"
        )


# ─── 6. parser regex 對稱性 ──────────────────────────


def test_bilateral_regex_long_block_stops_at_red_circle():
    """🟢 做多計劃 block 應在 🔴 開始時結束。"""
    from app.core.prediction_tracker import _BILATERAL_LONG_BLOCK
    text = """🟢 做多計劃 內容多單
進場 100/99 SL 95 TP 110

🔴 做空計劃 內容空單
進場 105/106 SL 110 TP 95"""
    m = _BILATERAL_LONG_BLOCK.search(text)
    assert m is not None
    captured = m.group(0)
    assert "🔴" not in captured, "long block 不該吃進 short 段"
    assert "進場 100/99" in captured


def test_bilateral_regex_short_block_stops_at_green_circle():
    """v105.3 Bug 8：🔴 做空計劃 也應該在後續 🟢 出現時結束（對稱）。"""
    from app.core.prediction_tracker import _BILATERAL_SHORT_BLOCK
    text = """🔴 做空計劃 內容空單
進場 105/106 SL 110 TP 95

🟢 補充提示（不該被吃進來）"""
    m = _BILATERAL_SHORT_BLOCK.search(text)
    assert m is not None
    captured = m.group(0)
    assert "補充提示" not in captured, "short block 應在 🟢 開始時截斷"


# ─── 7. position_size_multiplier 存到 DB ───────────────


def test_lean_card_position_multiplier_in_schema():
    """v105.3 Bug 7：predictions 表必須有 position_size_multiplier 欄位。"""
    from app.core.prediction_tracker import prediction_tracker
    prediction_tracker._ensure_db()
    cols = [r[1] for r in prediction_tracker._conn.execute(
        "PRAGMA table_info(predictions)"
    ).fetchall()]
    assert "position_size_multiplier" in cols, (
        f"predictions schema 缺 position_size_multiplier 欄位。實際欄位：{cols}"
    )


def test_lean_parser_returns_correct_multiplier():
    """v104.x：偏多 lean card 解析時應該標 position_size_multiplier=0.7。"""
    from app.core.prediction_tracker import _parse_lean_card

    lean_text = """🟢 偏多單向計劃
🎯 方向：偏多 BTC/USDT
📍 進場：100
🎯 目標：110
🛑 止損：95
⏱  時間框：48h
📊 信心：medium
🔍 主要指標：RSI
🌐 市場 regime：ranging
❌ 失效條件：跌破 95"""
    pred = _parse_lean_card(lean_text)
    assert pred is not None
    assert pred.get("is_lean") is True
    assert pred.get("position_size_multiplier") == 0.7
    assert pred.get("confidence") == "medium"


def test_normal_card_position_multiplier_default_one():
    """非 lean 的 prediction 預設 position_size_multiplier=1.0。"""
    from app.core.prediction_tracker import _parse_visible_card

    text = """📊 本次分析總結
🎯 方向：做多 BTC/USDT
📍 進場：100
🎯 目標：110
🛑 止損：95
⏱  時間框：48h
📊 信心：高
🔍 主要指標：RSI
🌐 市場 regime：trending_up
❌ 失效條件：跌破 95"""
    pred = _parse_visible_card(text)
    assert pred is not None
    assert pred.get("position_size_multiplier") == 1.0
    assert pred.get("is_lean") is False
