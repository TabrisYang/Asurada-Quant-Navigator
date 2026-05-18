"""v136 完整布林通道策略系統測試。

涵蓋：
- 4 個個別 detector function（squeeze / breakout / walk / mean reversion）
- regime-aware classify_bollinger_signal 主入口
- get_entry_exit_stop 各訊號類型的進出場規則
- auto_scanner 整合：_precompute_indicators 新增的 5 個指標 array 是否完整
- _compute_features 新增的 8 個特徵欄位
- _compute_bollinger_status 在不同 regime 下的選擇
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ─── 個別 detector 測試 ────────────────────────────────────

def test_detect_squeeze_state_active():
    from app.core.bollinger_signals import detect_squeeze_state, _get_thresholds, SIGNAL_SQUEEZE_ACTIVE
    th = _get_thresholds()
    features = {"is_squeeze": True, "squeeze_duration": 7}
    assert detect_squeeze_state(features, th) == SIGNAL_SQUEEZE_ACTIVE


def test_detect_squeeze_state_not_long_enough():
    from app.core.bollinger_signals import detect_squeeze_state, _get_thresholds
    th = _get_thresholds()
    features = {"is_squeeze": True, "squeeze_duration": 3}  # < min_duration
    assert detect_squeeze_state(features, th) is None


def test_detect_squeeze_state_not_in_squeeze():
    from app.core.bollinger_signals import detect_squeeze_state, _get_thresholds
    th = _get_thresholds()
    features = {"is_squeeze": False, "squeeze_duration": 0}
    assert detect_squeeze_state(features, th) is None


def test_detect_squeeze_breakout_up():
    from app.core.bollinger_signals import detect_squeeze_breakout, _get_thresholds, SIGNAL_SQUEEZE_BREAKOUT_UP
    th = _get_thresholds()
    features = {
        "bb_position": 95, "bb_position_lag1": 70, "bb_width_roc": 12.0,
        "obv_slope_10": 0.5, "is_squeeze": False,
    }
    prev = {"is_squeeze": True}
    assert detect_squeeze_breakout(features, prev, th) == SIGNAL_SQUEEZE_BREAKOUT_UP


def test_detect_squeeze_breakout_down():
    from app.core.bollinger_signals import detect_squeeze_breakout, _get_thresholds, SIGNAL_SQUEEZE_BREAKOUT_DOWN
    th = _get_thresholds()
    features = {
        "bb_position": 5, "bb_position_lag1": 30, "bb_width_roc": 12.0,
        "obv_slope_10": -0.5, "is_squeeze": False,
    }
    prev = {"is_squeeze": True}
    assert detect_squeeze_breakout(features, prev, th) == SIGNAL_SQUEEZE_BREAKOUT_DOWN


def test_detect_squeeze_breakout_volume_does_not_confirm():
    """量價不配合（OBV slope < 0 but 突破上軌）不觸發。"""
    from app.core.bollinger_signals import detect_squeeze_breakout, _get_thresholds
    th = _get_thresholds()
    features = {
        "bb_position": 95, "bb_position_lag1": 70, "bb_width_roc": 12.0,
        "obv_slope_10": -0.3,  # OBV 下滑，量不配合
        "is_squeeze": False,
    }
    prev = {"is_squeeze": True}
    assert detect_squeeze_breakout(features, prev, th) is None


def test_detect_squeeze_breakout_still_in_squeeze():
    from app.core.bollinger_signals import detect_squeeze_breakout, _get_thresholds
    th = _get_thresholds()
    features = {"is_squeeze": True}  # 還在 squeeze，未爆發
    prev = {"is_squeeze": True}
    assert detect_squeeze_breakout(features, prev, th) is None


def test_detect_walk_the_band_upper():
    from app.core.bollinger_signals import detect_walk_the_band, _get_thresholds, SIGNAL_WALKING_UPPER
    th = _get_thresholds()
    features = {"adx": 32}
    # 近 5 根有 3 根 bb_position >= 90 → 觸上軌
    recent = [88, 91, 93, 95, 92]
    assert detect_walk_the_band(features, recent, th) == SIGNAL_WALKING_UPPER


def test_detect_walk_the_band_lower():
    from app.core.bollinger_signals import detect_walk_the_band, _get_thresholds, SIGNAL_WALKING_LOWER
    th = _get_thresholds()
    features = {"adx": 30}
    recent = [12, 9, 7, 5, 8]
    assert detect_walk_the_band(features, recent, th) == SIGNAL_WALKING_LOWER


def test_detect_walk_the_band_no_trend():
    from app.core.bollinger_signals import detect_walk_the_band, _get_thresholds
    th = _get_thresholds()
    features = {"adx": 15}  # ADX 不夠強
    recent = [88, 91, 93, 95, 92]
    assert detect_walk_the_band(features, recent, th) is None


def test_detect_mean_reversion_from_upper():
    from app.core.bollinger_signals import detect_mean_reversion, _get_thresholds, SIGNAL_REVERSION_FROM_UPPER
    th = _get_thresholds()
    features = {"bb_position": 75, "bb_position_lag1": 105}
    prev = {"bb_position": 105}
    assert detect_mean_reversion(features, prev, "ranging", th) == SIGNAL_REVERSION_FROM_UPPER


def test_detect_mean_reversion_from_lower():
    from app.core.bollinger_signals import detect_mean_reversion, _get_thresholds, SIGNAL_REVERSION_FROM_LOWER
    th = _get_thresholds()
    features = {"bb_position": 25, "bb_position_lag1": -5}
    prev = {"bb_position": -5}
    assert detect_mean_reversion(features, prev, "ranging", th) == SIGNAL_REVERSION_FROM_LOWER


def test_detect_mean_reversion_skipped_in_trending():
    """趨勢盤不該跑 mean reversion，避免 walk the band 被誤判成反轉。"""
    from app.core.bollinger_signals import detect_mean_reversion, _get_thresholds
    th = _get_thresholds()
    features = {"bb_position": 75, "bb_position_lag1": 105}
    prev = {"bb_position": 105}
    assert detect_mean_reversion(features, prev, "trending_up", th) is None


# ─── 主入口 classify_bollinger_signal 測試 ─────────────────

def _make_features(**overrides):
    """生成測試用 features dict，可覆寫部分欄位。"""
    base = {
        "bb_position": 50, "bb_position_lag1": 50, "bb_width": 3.0,
        "bb_width_roc": 0, "z_score_20": 0, "atr_relative": 1,
        "obv_slope_10": 0, "is_squeeze": False, "squeeze_duration": 0,
        "adx": 18,
    }
    base.update(overrides)
    return base


def test_classify_no_signal_returns_none():
    from app.core.bollinger_signals import classify_bollinger_signal
    features = _make_features()
    result = classify_bollinger_signal(
        features, features, [50]*5, "ranging",
        close=100, sma20=100, bb_upper=103, bb_lower=97, atr=1.0,
    )
    assert result is None


def test_classify_walking_in_trending_regime():
    """trending_up regime 應優先選 walking band 而非 squeeze active。"""
    from app.core.bollinger_signals import classify_bollinger_signal, SIGNAL_WALKING_UPPER
    features = _make_features(
        adx=30, is_squeeze=True, squeeze_duration=8,  # 故意同時滿足 squeeze & walking
    )
    recent = [88, 91, 93, 95, 92]
    result = classify_bollinger_signal(
        features, features, recent, "trending_up",
        close=110, sma20=105, bb_upper=112, bb_lower=98, atr=2.0,
    )
    # trending_up 優先順序：walking → squeeze_breakout → squeeze_active
    assert result["signal"] == SIGNAL_WALKING_UPPER


def test_classify_returns_complete_payload():
    """成功時回傳 dict 必須含所有 key（signal, label, emoji, strategy, entry_exit, features_used）。"""
    from app.core.bollinger_signals import classify_bollinger_signal
    features = _make_features(is_squeeze=True, squeeze_duration=8)
    result = classify_bollinger_signal(
        features, features, [50]*5, "ranging",
        close=100, sma20=100, bb_upper=103, bb_lower=97, atr=1.0,
    )
    assert result is not None
    for key in ["signal", "label", "emoji", "strategy", "entry_exit", "features_used"]:
        assert key in result, f"missing key: {key}"


# ─── Entry/Exit/Stop 測試 ─────────────────────────────────

def test_entry_exit_squeeze_breakout_up():
    from app.core.bollinger_signals import get_entry_exit_stop, SIGNAL_SQUEEZE_BREAKOUT_UP
    features = _make_features()
    result = get_entry_exit_stop(SIGNAL_SQUEEZE_BREAKOUT_UP, features, atr=2.0,
                                  close=100, sma20=98, bb_upper=103, bb_lower=95)
    assert result["entry"] == 100
    assert result["stop"] == 98  # 中軌
    assert result["target_1"] == 104  # close + 2*atr
    assert result["target_2"] == 108  # close + 4*atr


def test_entry_exit_mean_reversion_from_upper():
    from app.core.bollinger_signals import get_entry_exit_stop, SIGNAL_REVERSION_FROM_UPPER
    features = _make_features()
    result = get_entry_exit_stop(SIGNAL_REVERSION_FROM_UPPER, features, atr=2.0,
                                  close=102, sma20=100, bb_upper=103, bb_lower=97)
    assert result["stop"] == 103  # 上軌
    assert result["target_1"] == 100  # 中軌
    assert result["target_2"] == 97   # 下軌


# ─── auto_scanner.py 整合測試 ─────────────────────────────

def _generate_test_df(n=300, seed=42):
    np.random.seed(seed)
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    opens = closes + np.random.randn(n) * 0.1
    highs = np.maximum(closes, opens) + np.abs(np.random.randn(n)) * 0.5
    lows = np.minimum(closes, opens) - np.abs(np.random.randn(n)) * 0.5
    volumes = 1000 + np.abs(np.random.randn(n)) * 200
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes})


def test_precompute_indicators_includes_bollinger_arrays():
    """v136 _precompute_indicators 應回傳 5 個新 array。"""
    from app.core.auto_scanner import _precompute_indicators
    df = _generate_test_df()
    ind = _precompute_indicators(df)
    for key in ["bb_std", "bb_upper", "bb_lower", "obv", "keltner_upper", "keltner_lower",
                "is_squeeze", "squeeze_duration"]:
        assert key in ind, f"missing ind key: {key}"
        assert len(ind[key]) == len(df)


def test_compute_features_includes_8_new_features():
    """v136 _compute_features 應產出 8 個新欄位。"""
    from app.core.auto_scanner import _precompute_indicators, _compute_features
    df = _generate_test_df()
    ind = _precompute_indicators(df)
    n = len(df)
    idx = list(range(n - 6, n))
    features = _compute_features(
        ind["closes"], ind["opens"], ind["highs"], ind["lows"], ind["volumes"],
        ind["ma5"], ind["ma20"], ind["vol_ma20"],
        ind["rsi14"], ind["bb_pos"], ind["bb_width"],
        ind["atr14"], ind["adx"], ind["plus_di"], ind["minus_di"], idx,
        bb_std=ind.get("bb_std"), obv=ind.get("obv"),
        keltner_upper=ind.get("keltner_upper"), keltner_lower=ind.get("keltner_lower"),
        is_squeeze=ind.get("is_squeeze"), squeeze_duration=ind.get("squeeze_duration"),
    )
    for key in ["bb_position_lag1", "bb_width_roc", "z_score_20", "obv_slope_10",
                "keltner_upper", "keltner_lower", "is_squeeze", "squeeze_duration"]:
        assert key in features, f"missing feature: {key}"


def test_compute_bollinger_status_end_to_end():
    """auto_scanner._compute_bollinger_status 不應丟錯（無論有無訊號）。"""
    from app.core.auto_scanner import _precompute_indicators, _compute_features, _compute_bollinger_status
    df = _generate_test_df()
    ind = _precompute_indicators(df)
    n = len(df)
    idx = list(range(n - 6, n))
    features = _compute_features(
        ind["closes"], ind["opens"], ind["highs"], ind["lows"], ind["volumes"],
        ind["ma5"], ind["ma20"], ind["vol_ma20"],
        ind["rsi14"], ind["bb_pos"], ind["bb_width"],
        ind["atr14"], ind["adx"], ind["plus_di"], ind["minus_di"], idx,
        bb_std=ind.get("bb_std"), obv=ind.get("obv"),
        keltner_upper=ind.get("keltner_upper"), keltner_lower=ind.get("keltner_lower"),
        is_squeeze=ind.get("is_squeeze"), squeeze_duration=ind.get("squeeze_duration"),
    )
    # 不應丟錯，回 None 或 dict 都可
    result = _compute_bollinger_status(ind, features, idx)
    assert result is None or isinstance(result, dict)


def test_feature_flag_disabled_returns_none():
    """settings.bollinger_signals_enabled=False 時 _compute_bollinger_status 應回 None。"""
    from app.core.auto_scanner import _precompute_indicators, _compute_features, _compute_bollinger_status
    from app.core.config.settings import settings

    original = getattr(settings, "bollinger_signals_enabled", True)
    settings.bollinger_signals_enabled = False
    try:
        df = _generate_test_df()
        ind = _precompute_indicators(df)
        n = len(df)
        idx = list(range(n - 6, n))
        features = _compute_features(
            ind["closes"], ind["opens"], ind["highs"], ind["lows"], ind["volumes"],
            ind["ma5"], ind["ma20"], ind["vol_ma20"],
            ind["rsi14"], ind["bb_pos"], ind["bb_width"],
            ind["atr14"], ind["adx"], ind["plus_di"], ind["minus_di"], idx,
            bb_std=ind.get("bb_std"), obv=ind.get("obv"),
            keltner_upper=ind.get("keltner_upper"), keltner_lower=ind.get("keltner_lower"),
            is_squeeze=ind.get("is_squeeze"), squeeze_duration=ind.get("squeeze_duration"),
        )
        assert _compute_bollinger_status(ind, features, idx) is None
    finally:
        settings.bollinger_signals_enabled = original
