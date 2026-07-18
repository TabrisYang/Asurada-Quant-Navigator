"""LLM adapters package（v154 拆分自單檔 adapter.py）

對外 API 請一律從 app.core.llm.adapter（façade）import — 該路徑永久穩定。
"""

from app.core.llm.adapters.base import (
    BaseLLMAdapter,
    LLMResponse,
    StreamEvent,
    TokenUsage,
    _looks_rate_limited,
    _minimal_r2_chart_state,
    estimate_cost,
)
from app.core.llm.adapters.claude_adapter import ClaudeAdapter
from app.core.llm.adapters.claude_subscription import ClaudeSubscriptionAdapter
from app.core.llm.adapters.factory import create_adapter
from app.core.llm.adapters.gemini_adapter import GeminiAdapter
from app.core.llm.adapters.ollama_adapter import OllamaAdapter
from app.core.llm.adapters.openai_adapter import OpenAIAdapter

__all__ = [
    "BaseLLMAdapter", "LLMResponse", "StreamEvent", "TokenUsage",
    "OpenAIAdapter", "GeminiAdapter", "ClaudeAdapter",
    "ClaudeSubscriptionAdapter", "OllamaAdapter",
    "create_adapter", "estimate_cost",
    "_looks_rate_limited", "_minimal_r2_chart_state",
]
