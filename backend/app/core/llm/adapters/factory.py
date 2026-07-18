"""create_adapter 工廠 — v154 由 adapter.py 純搬家拆出（原 L2010-2057），邏輯零改動。"""

from typing import Optional

from app.core.llm.adapters.base import BaseLLMAdapter
from app.core.llm.adapters.claude_adapter import ClaudeAdapter
from app.core.llm.adapters.claude_subscription import ClaudeSubscriptionAdapter
from app.core.llm.adapters.gemini_adapter import GeminiAdapter
from app.core.llm.adapters.ollama_adapter import OllamaAdapter
from app.core.llm.adapters.openai_adapter import OpenAIAdapter


def create_adapter(
    provider: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
) -> BaseLLMAdapter:
    """工廠函式：根據供應商建立對應的適配器

    model_name 是必要的——使用者必須在設定中選擇模型。
    不再有預設模型，避免使用者不知道系統用了什麼模型。
    """
    if not model_name:
        raise ValueError("請先在系統設定中選擇要使用的模型")

    if provider == "openai":
        if not api_key:
            raise ValueError("OpenAI 需要 API Key")
        return OpenAIAdapter(api_key=api_key, model=model_name)

    elif provider == "gemini":
        if not api_key:
            raise ValueError("Gemini 需要 API Key")
        return GeminiAdapter(api_key=api_key, model=model_name)

    elif provider == "claude":
        if not api_key:
            raise ValueError("Claude 需要 API Key")
        return ClaudeAdapter(api_key=api_key, model=model_name)

    elif provider == "claude_subscription":
        # api_key 在訂閱模式下「選填」攜帶 per-user OAuth token（claude setup-token 產生）。
        # 有 token → 每位使用者各自計入自己的訂閱、不需機器登入；
        # 無 token → 沿用本機 Claude Code 登入憑證（站長本機，行為不變）。
        from app.core.auth.claude_oauth import check_claude_cli_available
        oauth_token = api_key or None
        status = check_claude_cli_available(require_login=not oauth_token)
        if not status["available"]:
            raise ValueError(status["error"] or "Claude CLI 不可用")
        return ClaudeSubscriptionAdapter(model=model_name, oauth_token=oauth_token)

    elif provider == "ollama":
        return OllamaAdapter(
            base_url=base_url or "http://localhost:11434",
            model=model_name,
        )

    else:
        raise ValueError(f"不支援的 LLM 供應商: {provider}")
