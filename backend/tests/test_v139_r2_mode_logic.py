"""v139 sequence r2_mode 判定邏輯測試。

只測判定邏輯本身（純函式），不測 chat.py 整個 stream_gen 流程。
模擬 chat.py:2658-2676 _use_r2_for_sequence 計算的判定條件。
"""
from __future__ import annotations


def _use_r2_for_sequence(is_sequence_follow: bool, intents: set[str]) -> bool:
    """v139：sequence 後續訊息 + safe intent → r2_mode（與 chat.py 邏輯保持一致）"""
    R2_SAFE_INTENTS = {"fundamental_analysis", "sector_analysis", "calibrate"}
    R2_BLOCKED_INTENTS = {
        "event_analysis", "scenario", "smc", "conditional_prob",
        "deep_analysis", "deep_phase1", "deep_phase2", "deep_phase3",
        "comprehensive_analysis",
    }
    return bool(
        is_sequence_follow
        and (intents & R2_SAFE_INTENTS)
        and not (intents & R2_BLOCKED_INTENTS)
    )


def test_first_message_never_r2():
    """第 1 條訊息（is_sequence_follow=False）永遠走 full chart_state。"""
    assert _use_r2_for_sequence(False, {"comprehensive_analysis"}) is False
    assert _use_r2_for_sequence(False, {"fundamental_analysis"}) is False
    assert _use_r2_for_sequence(False, {"sector_analysis"}) is False


def test_sequence_fundamental_r2():
    """sequence + fundamental → r2_mode (省 token)"""
    assert _use_r2_for_sequence(True, {"fundamental_analysis"}) is True


def test_sequence_sector_r2():
    """sequence + sector → r2_mode"""
    assert _use_r2_for_sequence(True, {"sector_analysis"}) is True


def test_sequence_calibrate_r2():
    """sequence + calibrate → r2_mode"""
    assert _use_r2_for_sequence(True, {"calibrate"}) is True


def test_sequence_event_pattern_blocked():
    """sequence + event_analysis → blocked（保留 full，需 indicators 算 pattern 相似度）"""
    assert _use_r2_for_sequence(True, {"event_analysis"}) is False


def test_sequence_compute_laddered_no_match():
    """sequence + 一個非 safe intent → 不啟用 r2_mode"""
    # compute_laddered 不在 intent 列表（它是 mode），這裡用 'analysis' 模擬非 safe
    assert _use_r2_for_sequence(True, {"analysis"}) is False


def test_sequence_safe_with_blocked_keeps_full():
    """sequence + safe intent + blocked intent 同時存在 → 保守用 full（blocked 優先）"""
    assert _use_r2_for_sequence(True, {"fundamental_analysis", "event_analysis"}) is False
    assert _use_r2_for_sequence(True, {"sector_analysis", "comprehensive_analysis"}) is False


def test_sequence_no_intent_no_r2():
    """sequence flag 但沒任何 safe intent → 不啟用 r2_mode"""
    assert _use_r2_for_sequence(True, set()) is False
    assert _use_r2_for_sequence(True, {"general"}) is False


def test_sequence_multi_safe_intents_r2():
    """sequence + 多個 safe intents → r2_mode"""
    assert _use_r2_for_sequence(True, {"fundamental_analysis", "sector_analysis"}) is True
    assert _use_r2_for_sequence(True, {"calibrate", "fundamental_analysis"}) is True
