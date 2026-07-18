"""v105 單元測試：horizon_class / regime_mapping / factor_weight_learner / per_regime_calibrator。"""

import pytest


# ─── horizon_class ───────────────────────────────────


@pytest.mark.parametrize("hours,expected", [
    (4, "short"),
    (12, "short"),
    (23, "short"),
    (24, "medium"),
    (72, "medium"),
    (168, "medium"),
    (169, "long"),
    (720, "long"),
    (None, "medium"),
    (0, "short"),
])
def test_horizon_classification(hours, expected):
    from app.core.prediction_tracker import prediction_tracker
    assert prediction_tracker.classify_horizon(hours) == expected


# ─── regime_mapping ───────────────────────────────────


@pytest.mark.parametrize("raw,expected", [
    ("trending_up", "trending_up"),
    ("ranging", "ranging"),
    ("unknown", "unknown"),
    ("盤整", "ranging"),
    ("盤整偏空", "ranging"),
    ("低波盤整", "low_vol"),  # 「低波」優先匹配
    ("盤整壓縮", "low_vol"),  # 「壓縮」優先
    ("趨勢上行", "trending_up"),
    ("趨勢下行", "trending_down"),
    ("強趨勢上行", "trending_up"),
    ("高波動", "high_vol"),
    ("過渡期", "ranging"),
    ("結構轉折期", "ranging"),
    ("趨勢→盤整過渡", "ranging"),  # 「盤整」匹配
    ("", "unknown"),
    (None, "unknown"),
    ("毫無關鍵字的自由文字", "unknown"),
])
def test_regime_standardization(raw, expected):
    from app.core.regime_mapping import standardize_regime
    assert standardize_regime(raw) == expected


# ─── factor_weight_learner ───────────────────────────────────


def test_load_weights_returns_none_when_missing(tmp_path, monkeypatch):
    """沒 weights 檔時 load_learned_weights 回 None。"""
    from app.core import factor_weight_learner
    monkeypatch.setattr(factor_weight_learner, "_WEIGHTS_PATH", tmp_path / "nonexistent.json")
    assert factor_weight_learner.load_learned_weights() is None


def test_quality_gate_rejects_low_auc(tmp_path, monkeypatch):
    """低 AUC 學習結果被拒絕，weights 檔不更新。"""
    from app.core import factor_weight_learner
    weights_path = tmp_path / "bias_weights.json"
    monkeypatch.setattr(factor_weight_learner, "_WEIGHTS_PATH", weights_path)

    # 直接呼叫，預期因樣本/AUC 問題被 reject
    # 資料依賴：需要本地 predictions.db 有 verified samples；CI / 新環境沒有 → skip
    try:
        result = factor_weight_learner.fit_bias_weights(min_samples=50)
    except RuntimeError as e:
        if "verified samples" in str(e):
            pytest.skip("無 verified samples（predictions.db 空）— 資料依賴測試跳過")
        raise
    # 若 lockbox AUC < 0.55 status="rejected_low_auc"，weights 檔不存在
    if result.get("status") == "rejected_low_auc":
        assert not weights_path.exists(), "rejected 不該存到 production path"
        rejected_path = weights_path.parent / "bias_score_weights_rejected.json"
        assert rejected_path.exists(), "rejected 應該存 debug 檔"


# ─── per_regime_calibrator ───────────────────────────────────


def test_calibrator_load_missing_returns_none():
    """沒對應 regime 的 calibrator 回 None。"""
    from app.core.per_regime_calibrator import load_calibrator
    assert load_calibrator("nonexistent_regime") is None


def test_calibrate_probability_fallback_when_no_calibrator():
    """沒 calibrator 時直接回原值（fallback）。"""
    from app.core.per_regime_calibrator import calibrate_probability
    raw = 0.65
    out = calibrate_probability(raw, "nonexistent_regime")
    assert out == raw, f"沒 calibrator 應 fallback 原值，實際 {out}"


def test_per_regime_walk_forward_skips_low_samples():
    """樣本不足的 regime 應該標 skipped_low_samples 不爆。"""
    from app.core.per_regime_calibrator import per_regime_walk_forward_summary
    out = per_regime_walk_forward_summary()
    assert isinstance(out, dict)
    # 如果有任何樣本，應該至少看到部分 regime 的結果
    for regime, info in out.items():
        if isinstance(info, dict):
            assert info.get("status") in ("ok", "skipped_low_samples", "skipped_single_class", "no_data")
