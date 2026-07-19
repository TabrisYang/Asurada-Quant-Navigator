"""阿斯拉量化系統 — LLM 統一適配器（façade）

v154：原 2057 行單檔已拆為 app/core/llm/adapters/ package（純搬家，邏輯零改動）。
此檔保留為穩定 import 門面 — 全 codebase 的
`from app.core.llm.adapter import X` 一律不需改動。
新增供應商請到 adapters/ 加檔案並在此與 adapters/__init__.py 補 re-export。
"""

from app.core.llm.adapters import (
    BaseLLMAdapter,
    ClaudeAdapter,
    ClaudeSubscriptionAdapter,
    CodexSubscriptionAdapter,
    GeminiAdapter,
    LLMResponse,
    OllamaAdapter,
    OpenAIAdapter,
    StreamEvent,
    TokenUsage,
    _looks_rate_limited,
    _minimal_r2_chart_state,
    create_adapter,
    estimate_cost,
)

__all__ = [
    "BaseLLMAdapter", "LLMResponse", "StreamEvent", "TokenUsage",
    "OpenAIAdapter", "GeminiAdapter", "ClaudeAdapter",
    "ClaudeSubscriptionAdapter", "CodexSubscriptionAdapter", "OllamaAdapter",
    "create_adapter", "estimate_cost",
    "_looks_rate_limited", "_minimal_r2_chart_state",
]
