"""Codex 訂閱制 adapter 契約（v156）— 不依賴 codex CLI 安裝"""

from app.core.llm.adapter import CodexSubscriptionAdapter, create_adapter
from app.core.llm.adapters.codex_subscription import check_codex_cli_available


class TestCodexAdapter:
    def test_provider_name_and_inheritance(self):
        a = CodexSubscriptionAdapter(model="gpt-5.1-codex")
        assert a._PROVIDER_NAME == "codex_subscription"
        assert a.model == "gpt-5.1-codex"
        # 繼承 Claude 訂閱 adapter 的文字協議 function calling
        text, calls = a._parse_tool_calls('前言 <tool_call>{"name": "query_chart_data", "arguments": {}}</tool_call> 後記')
        assert calls and calls[0]["name"] == "query_chart_data"

    def test_availability_check_shape(self):
        r = check_codex_cli_available()
        assert isinstance(r["available"], bool)
        if not r["available"]:
            assert "codex" in (r["error"] or "")

    def test_create_adapter_requires_model(self):
        import pytest
        with pytest.raises(ValueError):
            create_adapter(provider="codex_subscription", model_name=None)


class TestNoKeyWhitelistConsistency:
    """回歸鎖：免 API Key 供應商白名單必須三檔一致（chat.py 曾遺漏 codex 導致
    「Session 已過期」誤報）。新增免 key 供應商時此測試會提醒同步。"""

    def test_all_whitelists_include_codex(self):
        import re
        from pathlib import Path

        backend = Path(__file__).resolve().parent.parent
        pattern = re.compile(r'provider(?:\.value)? not in \(([^)]+)\)')
        for rel in ("app/api/routes/chat.py", "app/api/routes/config.py"):
            src = (backend / rel).read_text(encoding="utf-8")
            for m in pattern.finditer(src):
                whitelist = m.group(1)
                if "claude_subscription" in whitelist:
                    assert "codex_subscription" in whitelist, (
                        f"{rel} 的免 key 白名單缺 codex_subscription：({whitelist})"
                    )
