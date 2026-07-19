"""LLM 覆核層（core/llm/verifier.py）單元測試 — 純函式，不打 LLM"""

import pytest

from app.core.llm.verifier import (
    _parse_verify_json,
    build_data_digest,
    format_verify_block,
    pick_verifier_model,
    should_verify,
)
from app.core.config.settings import settings


# ─── pick_verifier_model ─────────────────────────────────────

class TestPickVerifierModel:
    def test_override_wins(self):
        assert pick_verifier_model("claude_subscription", "claude-opus-4-8", "claude-opus-4-7") == "claude-opus-4-7"

    @pytest.mark.parametrize("main,expected", [
        ("claude-opus-4-8", "sonnet"),
        ("claude-fable-5", "sonnet"),
        ("opus", "sonnet"),
        ("claude-sonnet-4-6", "haiku"),
        ("sonnet", "haiku"),
        ("claude-haiku-4-5-20251001", "haiku"),
        ("unknown-model", "sonnet"),  # 抓不到家族 → fallback
    ])
    def test_claude_subscription_family_down(self, main, expected):
        assert pick_verifier_model("claude_subscription", main) == expected

    def test_claude_api_down(self):
        assert pick_verifier_model("claude", "claude-opus-4-20250514") == "claude-sonnet-4-20250514"
        assert pick_verifier_model("claude", "claude-3-5-sonnet-20241022") == "claude-3-5-haiku-20241022"
        assert pick_verifier_model("claude", "claude-3-haiku-20240307") == "claude-3-haiku-20240307"

    def test_openai_down(self):
        assert pick_verifier_model("openai", "gpt-4o") == "gpt-4o-mini"
        assert pick_verifier_model("openai", "gpt-4.1") == "gpt-4.1-mini"
        assert pick_verifier_model("openai", "o3") == "o4-mini"
        # 已是小模型 → 同模型
        assert pick_verifier_model("openai", "gpt-4o-mini") == "gpt-4o-mini"
        assert pick_verifier_model("openai", "gpt-5-something") == "gpt-4o-mini"

    def test_gemini_down(self):
        assert pick_verifier_model("gemini", "gemini-2.5-pro") == "gemini-2.5-flash"
        assert pick_verifier_model("gemini", "gemini-2.0-flash") == "gemini-2.0-flash-lite"
        assert pick_verifier_model("gemini", "gemini-2.0-flash-lite") == "gemini-2.0-flash-lite"

    def test_codex_subscription_down(self):
        assert pick_verifier_model("codex_subscription", "gpt-5.6-terra") == "gpt-5.6-luna"
        assert pick_verifier_model("codex_subscription", "gpt-5.6-luna") == "gpt-5.6-luna"
        assert pick_verifier_model("codex_subscription", "gpt-5.4-mini") == "gpt-5.4-mini"
        assert pick_verifier_model("codex_subscription", "gpt-5.5") == "gpt-5.6-luna"

    def test_ollama_same_model(self):
        assert pick_verifier_model("ollama", "llama3") == "llama3"


# ─── _parse_verify_json ──────────────────────────────────────

_GOOD = '{"verdict":"issues","issues":[{"type":"number","severity":"high","quote":"RSI 71","why":"實際 42.1","correction":"RSI=42.1"}]}'


class TestParseVerifyJson:
    def test_bare_json(self):
        r = _parse_verify_json(_GOOD)
        assert r["verdict"] == "issues"
        assert len(r["issues"]) == 1
        assert r["issues"][0]["type"] == "number"

    def test_markdown_fence(self):
        r = _parse_verify_json(f"```json\n{_GOOD}\n```")
        assert r is not None and r["verdict"] == "issues"

    def test_surrounding_text(self):
        r = _parse_verify_json(f"以下是覆核結果：\n{_GOOD}\n以上。")
        assert r is not None and len(r["issues"]) == 1

    def test_pass_verdict(self):
        r = _parse_verify_json('{"verdict":"pass","issues":[]}')
        assert r == {"verdict": "pass", "issues": []}

    def test_bad_type_dropped(self):
        raw = ('{"verdict":"issues","issues":['
               '{"type":"style","quote":"x","why":"y"},'
               '{"type":"logic","quote":"a","why":"b"}]}')
        r = _parse_verify_json(raw)
        assert len(r["issues"]) == 1 and r["issues"][0]["type"] == "logic"

    def test_all_bad_types_becomes_pass(self):
        r = _parse_verify_json('{"verdict":"issues","issues":[{"type":"style","quote":"x"}]}')
        assert r["verdict"] == "pass" and r["issues"] == []

    def test_max_five_issues(self):
        items = ",".join(
            f'{{"type":"number","quote":"q{i}","why":"w"}}' for i in range(8)
        )
        r = _parse_verify_json(f'{{"verdict":"issues","issues":[{items}]}}')
        assert len(r["issues"]) == 5

    @pytest.mark.parametrize("bad", ["", "not json", '{"verdict":"maybe"}', "[]", None])
    def test_invalid_returns_none(self, bad):
        assert _parse_verify_json(bad) is None


# ─── build_data_digest ───────────────────────────────────────

class TestBuildDataDigest:
    def test_basic_fields(self):
        cs = {
            "symbol": "BTCUSDT", "timeframe": "4h", "currentPrice": 65000,
            "currentRegime": {"regime": "trending"},
            "indicatorValues": {"RSI": 42.1234567, "MACD": -12.5},
            "external_signals": {
                "derivatives": {"funding_rate_pct": 0.01, "global_long_short_ratio": 1.2},
                "sentiment": {"fear_greed_value": 55},
            },
            "recent_accuracy": {"win_rate_30d": 62.0},
            "donchian_position_pct": 48.3,
        }
        d = build_data_digest(cs, None)
        assert "symbol=BTCUSDT" in d
        assert "regime=trending" in d
        assert "RSI=42.1235" in d  # round 4 位
        assert "funding_rate_pct=0.01" in d
        assert "win_rate_30d=62.0" in d
        assert "donchian_position_pct=48.3" in d

    def test_backtest_facts_included(self):
        exec_result = {"results": [{"result": {"backtest": {
            "profit_factor": 1.45, "sharpe_ratio": 1.1, "max_drawdown_pct": 12.0,
        }}}]}
        d = build_data_digest({}, exec_result)
        assert "backtest_pf=1.45" in d
        assert "backtest_mdd=12.0" in d

    def test_size_capped(self):
        cs = {"indicatorValues": {f"IND_{i}": i * 1.2345 for i in range(500)}}
        d = build_data_digest(cs, None, max_bytes=3072)
        assert len(d.encode("utf-8")) <= 3072 + 64  # 截斷註記的餘裕
        assert "digest truncated" in d

    def test_empty_state(self):
        assert build_data_digest(None, None) == ""


# ─── should_verify ───────────────────────────────────────────

_LONG_TEXT = "分析結論：RSI 42.1 偏弱，建議觀望。" * 50  # >500 字含數字


class TestShouldVerify:
    def test_analysis_intent_passes(self):
        assert should_verify(_LONG_TEXT, {"analysis"}) is True

    def test_short_text_skipped(self):
        assert should_verify("RSI 42.1", {"analysis"}) is False

    def test_no_digit_skipped(self):
        assert should_verify("純文字說明。" * 200, {"analysis"}) is False

    def test_general_intent_skipped(self):
        assert should_verify(_LONG_TEXT, {"general"}) is False
        assert should_verify(_LONG_TEXT, {"simple_query"}) is False

    def test_error_message_skipped(self):
        assert should_verify("⚠️ " + _LONG_TEXT, {"analysis"}) is False

    def test_empty_skipped(self):
        assert should_verify(None, {"analysis"}) is False
        assert should_verify(_LONG_TEXT, set()) is False

    def test_min_len_follows_settings(self):
        text = "RSI 42.1 偏弱" * 10
        assert len(text) < settings.verify_min_text_len
        assert should_verify(text, {"analysis"}) is False


# ─── format_verify_block ─────────────────────────────────────

class TestFormatVerifyBlock:
    def test_block_contains_issues(self):
        block = format_verify_block({
            "model": "sonnet",
            "issues": [{
                "type": "direction", "severity": "high",
                "quote": "建議做空", "why": "數據偏多", "correction": "應偏多解讀",
            }],
        })
        assert "AI 覆核（sonnet 交叉檢查）" in block
        assert "方向矛盾" in block
        assert "建議做空" in block
        assert "應偏多解讀" in block
