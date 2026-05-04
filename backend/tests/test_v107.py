"""阿斯拉量化系統 — v107 單元測試。

涵蓋：
- v107.1：mechanical_audit 數值一致性 + 必引用 + 缺失誤導
- v107.2：user_positions 已移除注入 + compress 白名單不含
- v107.3：資料缺失 prompt 規則存在 + social TTL=2h
"""

from __future__ import annotations

import importlib


# ─── v107.1：機械審查 ─────────────────────────


def test_mechanical_audit_passes_clean_report():
    from app.core.mechanical_audit import audit_final_text
    chart_state = {
        "priceOverview": {"lastClose": 80000.0},
        "indicatorValues": {"rsi": {"values": [70, 71, 72]}},
        "external_signals": {"derivatives": {"funding_rate_pct": 0.05}},
        "regimeWarning": True,
    }
    text = "目前 RSI=72，現價 $80,000，funding=+0.05%。低信心狀態下保守觀望。"
    r = audit_final_text(text, chart_state)
    assert r["passed"] is True
    assert r["n_failures"] == 0
    assert r["n_checks"] >= 4


def test_mechanical_audit_catches_rsi_mismatch():
    from app.core.mechanical_audit import audit_final_text
    chart_state = {
        "priceOverview": {"lastClose": 80000.0},
        "indicatorValues": {"rsi": {"values": [70, 71, 72]}},
        "regimeWarning": True,
    }
    # 寫 RSI=50 但實際 72，差距 > 5%
    text = "目前 RSI=50 看起來中性區間，現價 $80,000 約莫支撐位附近，建議低信心保守觀望。"
    r = audit_final_text(text, chart_state)
    assert r["passed"] is False
    assert any("RSI 數值不一致" in i for i in r["issues"])


def test_mechanical_audit_catches_missing_regime_warning():
    from app.core.mechanical_audit import audit_final_text
    chart_state = {"regimeWarning": True, "priceOverview": {"lastClose": 100}}
    text = "目前看起來方向偏多，建議做多。" * 5
    r = audit_final_text(text, chart_state)
    assert r["passed"] is False
    assert any("regimeWarning" in i for i in r["issues"])


def test_mechanical_audit_catches_missing_data_misleading():
    from app.core.mechanical_audit import audit_final_text
    # chart_state 沒有 upcoming_events / social_sentiment
    chart_state = {"priceOverview": {"lastClose": 80000}}
    text = "目前 RSI=72，現價 $80,000，整體看市場情緒中性，目前無重大事件影響，建議觀望。"
    r = audit_final_text(text, chart_state)
    assert r["passed"] is False
    issues_str = " ".join(r["issues"])
    assert "無重大事件影響" in issues_str
    assert "市場情緒中性" in issues_str


def test_mechanical_audit_skips_short_text():
    from app.core.mechanical_audit import audit_final_text
    r = audit_final_text("OK", {})
    assert r["passed"] is True
    assert r["n_checks"] == 0


# ─── v107.2：移除 user_positions 注入 ─────────────────────────


def test_user_positions_removed_from_compress_whitelist():
    from app.core.context_compressor import _ANALYSIS_FIELDS
    assert "user_positions" not in _ANALYSIS_FIELDS, "v107.2: user_positions 不該在白名單"
    assert "portfolio_summary" in _ANALYSIS_FIELDS, "portfolio_summary 應保留"


def test_compress_drops_user_positions_keeps_portfolio():
    from app.core.context_compressor import compress_chart_state
    full = {
        "symbol": "BTC/USDT", "timeframe": "4h",
        "currentRegime": {"regime": "trending_up"},
        "user_positions": {"has_position": True, "direction": "long"},
        "portfolio_summary": {"total_positions": 3, "long_short_ratio": 5.0},
    }
    out = compress_chart_state(full, {"analysis"})
    assert "user_positions" not in out
    assert "portfolio_summary" in out


def test_user_positions_prompt_rules_removed():
    from app.core.llm.function_defs import assemble_system_prompt
    prompt = assemble_system_prompt({"analysis"})
    # 個人化規則應已移除
    assert "📊 你的持倉：[direction]" not in prompt
    assert "持倉跟分析訊號反向" not in prompt
    # 公正性規則應已新增
    assert "v107.2 公正性規則" in prompt or "公正性規則" in prompt
    # 組合風控規則應仍存在
    assert "portfolio_summary.long_short_ratio" in prompt


def test_critique_endpoint_removed():
    from app.main import app
    paths = [r.path for r in app.routes if hasattr(r, "path")]
    assert "/api/chat/critique" not in paths, "v107.1: critique endpoint 應已移除"


def test_reflection_critic_module_removed():
    import importlib
    try:
        importlib.import_module("app.core.llm.reflection_critic")
        raise AssertionError("reflection_critic 模組應已刪除")
    except ImportError:
        pass  # expected


# ─── v107.3：資料缺失 prompt + social TTL ─────────────────────────


def test_missing_data_prompt_rules_present():
    from app.core.llm.function_defs import assemble_system_prompt
    prompt = assemble_system_prompt({"analysis"})
    assert "資料缺失強制明示規則" in prompt
    assert "禁用語句規則" in prompt
    assert "無可信事件資料來源" in prompt


def test_social_sentiment_ttl_is_2h():
    from app.core.social_sentiment import _CACHE_TTL
    assert _CACHE_TTL == 7200, f"v107.3: social TTL 應為 7200s（2h），實際 {_CACHE_TTL}"


# ─── 整合：SSE audit event 已介接 ─────────────────────────


def test_audit_module_importable():
    mod = importlib.import_module("app.core.mechanical_audit")
    assert hasattr(mod, "audit_final_text")
