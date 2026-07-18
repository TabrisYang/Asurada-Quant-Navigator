"""ClaudeAdapter — v154 由 adapter.py 純搬家拆出（原 L981-1251），邏輯零改動。"""

import asyncio
import json
from typing import Optional

from loguru import logger

from app.core.config.settings import settings
from app.core.llm.function_defs import FUNCTION_DEFINITIONS

from app.core.llm.adapters.base import (
    BaseLLMAdapter,
    LLMResponse,
    StreamEvent,
    TokenUsage,
    _ANTHROPIC_CACHE_BETA_HEADER,
    _CACHE_CONTROL_1H,
    _LLM_TIMEOUT,
)

class ClaudeAdapter(BaseLLMAdapter):
    """Anthropic Claude 適配器"""

    def __init__(self, api_key: str = "", model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.model = model

    def _convert_functions_to_claude(self) -> list[dict]:
        """轉換為 Claude tool_use 格式"""
        tools = []
        for f in FUNCTION_DEFINITIONS:
            func = f["function"]
            tools.append({
                "name": func["name"],
                "description": func["description"],
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            })
        return tools

    async def chat(self, messages: list[dict], chart_state: Optional[dict] = None, force_text: bool = False, system_prompt: Optional[str] = None, chart_screenshot: Optional[str] = None, r2_mode: bool = False) -> LLMResponse:
        try:
            from anthropic import AsyncAnthropic
            client = AsyncAnthropic(api_key=self.api_key, timeout=_LLM_TIMEOUT)

            claude_messages = []
            for msg in messages:
                if msg["role"] in ("user", "assistant"):
                    claude_messages.append(msg)

            # 注入圖表截圖到最後一個 user 訊息（Claude Vision API）
            if chart_screenshot:
                b64_data, media_type = self._extract_base64(chart_screenshot)
                for i in range(len(claude_messages) - 1, -1, -1):
                    if claude_messages[i]["role"] == "user":
                        text_content = claude_messages[i]["content"]
                        claude_messages[i] = {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"[目前圖表截圖如下]\n{text_content}"},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": b64_data,
                                    },
                                },
                            ],
                        }
                        break

            # Prompt caching：把 system 拆成「靜態（cached）+ 動態」兩 block
            static_sys, dynamic_sys = self._build_system_blocks(chart_state, system_prompt)
            system_blocks: list[dict] = [
                {
                    "type": "text",
                    "text": static_sys,
                    "cache_control": _CACHE_CONTROL_1H,  # v139：1h TTL（beta，連跑 6 條都命中）
                }
            ]
            if dynamic_sys:
                system_blocks.append({"type": "text", "text": dynamic_sys})

            create_kwargs: dict = {
                "model": self.model,
                "max_tokens": settings.llm_max_tokens,
                "system": system_blocks,
                "messages": claude_messages,
                "temperature": settings.llm_temperature,
            }
            if not force_text:
                tools = self._convert_functions_to_claude()
                if tools:
                    # 在最後一個 tool 加 cache_control，cache 整個 tools 區塊
                    tools[-1] = {**tools[-1], "cache_control": _CACHE_CONTROL_1H}  # v139
                create_kwargs["tools"] = tools

            # v139：附加 1h cache TTL beta header（若 SDK / API 不支援會 silently 忽略）
            create_kwargs["extra_headers"] = _ANTHROPIC_CACHE_BETA_HEADER

            try:
                response = await asyncio.wait_for(
                    client.messages.create(**create_kwargs),
                    timeout=_LLM_TIMEOUT,
                )
            except Exception as _cache_err:
                # v139 fallback：若 API 不接受 1h ttl，回退到 5 min 預設
                if "ttl" in str(_cache_err).lower() or "extended-cache" in str(_cache_err).lower():
                    logger.warning(f"[v139] 1h cache TTL 失敗，fallback 5min: {_cache_err}")
                    create_kwargs.pop("extra_headers", None)
                    for blk in create_kwargs.get("system", []):
                        if isinstance(blk, dict) and blk.get("cache_control", {}).get("ttl") == "1h":
                            blk["cache_control"] = {"type": "ephemeral"}
                    for tl in create_kwargs.get("tools", []) or []:
                        if isinstance(tl, dict) and tl.get("cache_control", {}).get("ttl") == "1h":
                            tl["cache_control"] = {"type": "ephemeral"}
                    response = await asyncio.wait_for(
                        client.messages.create(**create_kwargs),
                        timeout=_LLM_TIMEOUT,
                    )
                else:
                    raise

            # 解析回應
            text = ""
            function_calls = []
            for block in response.content:
                if block.type == "text":
                    text += block.text
                elif block.type == "tool_use":
                    function_calls.append({
                        "name": block.name,
                        "arguments": block.input,
                    })

            # 提取 token 用量（含 cache 命中細項）
            usage = None
            if hasattr(response, "usage") and response.usage:
                input_tok = getattr(response.usage, "input_tokens", 0) or 0
                output_tok = getattr(response.usage, "output_tokens", 0) or 0
                cache_create = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
                cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
                usage = TokenUsage(
                    prompt_tokens=input_tok,
                    completion_tokens=output_tok,
                    total_tokens=input_tok + output_tok,
                    model=self.model,
                    provider="claude",
                    cache_creation_tokens=cache_create,
                    cache_read_tokens=cache_read,
                )
                if cache_read > 0:
                    logger.info(
                        f"Claude cache HIT: cache_read={cache_read}, "
                        f"cache_create={cache_create}, input={input_tok} tokens"
                    )

            return LLMResponse(message=text, function_calls=function_calls, raw_response=response, usage=usage)
        except Exception as e:
            logger.error(f"Claude 請求失敗: {e}")
            return LLMResponse(message=f"LLM 請求失敗: {str(e)}")

    async def chat_stream(self, messages: list[dict], chart_state: Optional[dict] = None, system_prompt: Optional[str] = None):
        response = await self.chat(messages, chart_state, system_prompt=system_prompt)
        yield response.message

    async def chat_stream_events(
        self,
        messages: list[dict],
        chart_state: Optional[dict] = None,
        force_text: bool = False,
        system_prompt: Optional[str] = None,
        chart_screenshot: Optional[str] = None,
        r2_mode: bool = False,
    ):
        """Anthropic API 真串流：使用 messages.stream 邊收 token 邊 yield。"""
        try:
            from anthropic import AsyncAnthropic
            client = AsyncAnthropic(api_key=self.api_key, timeout=_LLM_TIMEOUT)

            claude_messages = []
            for msg in messages:
                if msg["role"] in ("user", "assistant"):
                    claude_messages.append(msg)

            # 注入圖表截圖到最後一個 user 訊息（Vision API）
            if chart_screenshot:
                b64_data, media_type = self._extract_base64(chart_screenshot)
                for i in range(len(claude_messages) - 1, -1, -1):
                    if claude_messages[i]["role"] == "user":
                        text_content = claude_messages[i]["content"]
                        claude_messages[i] = {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"[目前圖表截圖如下]\n{text_content}"},
                                {"type": "image", "source": {
                                    "type": "base64", "media_type": media_type, "data": b64_data,
                                }},
                            ],
                        }
                        break

            # Prompt caching：靜態 + 動態 兩 block
            static_sys, dynamic_sys = self._build_system_blocks(chart_state, system_prompt)
            system_blocks: list[dict] = [
                {"type": "text", "text": static_sys, "cache_control": _CACHE_CONTROL_1H}  # v139
            ]
            if dynamic_sys:
                system_blocks.append({"type": "text", "text": dynamic_sys})

            create_kwargs: dict = {
                "model": self.model,
                "max_tokens": settings.llm_max_tokens,
                "system": system_blocks,
                "messages": claude_messages,
                "temperature": settings.llm_temperature,
            }
            if not force_text:
                tools = self._convert_functions_to_claude()
                if tools:
                    tools[-1] = {**tools[-1], "cache_control": _CACHE_CONTROL_1H}  # v139
                create_kwargs["tools"] = tools

            # v139：附加 1h cache TTL beta header；若不支援會 silently 忽略
            create_kwargs["extra_headers"] = _ANTHROPIC_CACHE_BETA_HEADER

            # 真串流：邊收事件邊 yield
            current_tool: Optional[dict] = None
            current_tool_args = ""
            async with client.messages.stream(**create_kwargs) as stream:
                async for event in stream:
                    etype = getattr(event, "type", "")
                    if etype == "content_block_start":
                        block = getattr(event, "content_block", None)
                        if block and getattr(block, "type", "") == "tool_use":
                            current_tool = {"name": block.name, "id": getattr(block, "id", "")}
                            current_tool_args = ""
                    elif etype == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        dtype = getattr(delta, "type", "")
                        if dtype == "text_delta":
                            yield StreamEvent(type="text_delta", text=getattr(delta, "text", ""))
                        elif dtype == "input_json_delta" and current_tool is not None:
                            current_tool_args += getattr(delta, "partial_json", "")
                    elif etype == "content_block_stop":
                        if current_tool is not None:
                            try:
                                args = json.loads(current_tool_args) if current_tool_args else {}
                            except json.JSONDecodeError:
                                args = {}
                            yield StreamEvent(type="function_call", function_call={
                                "name": current_tool["name"],
                                "arguments": args,
                            })
                            current_tool = None
                            current_tool_args = ""

                # 結束後拿 final message 取 usage + stop_reason
                final_message = await stream.get_final_message()

            # 提取 usage
            if hasattr(final_message, "usage") and final_message.usage:
                input_tok = getattr(final_message.usage, "input_tokens", 0) or 0
                output_tok = getattr(final_message.usage, "output_tokens", 0) or 0
                cache_create = getattr(final_message.usage, "cache_creation_input_tokens", 0) or 0
                cache_read = getattr(final_message.usage, "cache_read_input_tokens", 0) or 0
                usage = TokenUsage(
                    prompt_tokens=input_tok,
                    completion_tokens=output_tok,
                    total_tokens=input_tok + output_tok,
                    model=self.model,
                    provider="claude",
                    cache_creation_tokens=cache_create,
                    cache_read_tokens=cache_read,
                )
                if cache_read > 0:
                    logger.info(
                        f"Claude cache HIT (stream): cache_read={cache_read}, "
                        f"cache_create={cache_create}, input={input_tok}"
                    )
                yield StreamEvent(type="usage", usage=usage)

            stop_reason = getattr(final_message, "stop_reason", "end_turn") or "end_turn"
            yield StreamEvent(type="stop", stop_reason=stop_reason)

        except Exception as e:
            logger.error(f"Claude 串流失敗: {e}")
            yield StreamEvent(type="text_delta", text=f"\n\n[Claude 串流錯誤] {str(e)}")
            yield StreamEvent(type="stop", stop_reason="error")


