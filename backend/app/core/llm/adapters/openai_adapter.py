"""OpenAIAdapter — v154 由 adapter.py 純搬家拆出（原 L399-645），邏輯零改動。"""

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
    _LLM_STREAM_TIMEOUT,
    _LLM_TIMEOUT,
)

class OpenAIAdapter(BaseLLMAdapter):
    """OpenAI (GPT-4/4o) 適配器"""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model

    async def chat(self, messages: list[dict], chart_state: Optional[dict] = None, force_text: bool = False, system_prompt: Optional[str] = None, chart_screenshot: Optional[str] = None, r2_mode: bool = False) -> LLMResponse:
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.api_key, timeout=_LLM_TIMEOUT)

            sys_msg = {"role": "system", "content": self._build_system_message(chart_state, system_prompt)}
            all_messages = [sys_msg] + messages

            # 注入圖表截圖到最後一個 user 訊息
            if chart_screenshot:
                for i in range(len(all_messages) - 1, -1, -1):
                    if all_messages[i]["role"] == "user":
                        text_content = all_messages[i]["content"]
                        all_messages[i] = {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"[目前圖表截圖如下]\n{text_content}"},
                                {"type": "image_url", "image_url": {"url": chart_screenshot, "detail": "high"}},
                            ],
                        }
                        break

            # v106 B4：OpenAI auto-caches prompts ≥1024 tokens with stable prefix
            # 拆 system 為「靜態（前）+ 動態（後）」最大化 cache hit
            static_sys, dynamic_sys = self._build_system_blocks(chart_state, system_prompt)
            # 替換 all_messages 中第一個 system 訊息為兩段
            non_system = [m for m in all_messages if m.get("role") != "system"]
            structured_messages = [
                {"role": "system", "content": static_sys},
            ]
            if dynamic_sys:
                structured_messages.append({"role": "system", "content": dynamic_sys})
            structured_messages.extend(non_system)

            create_kwargs: dict = {
                "model": self.model,
                "messages": structured_messages,
                "temperature": settings.llm_temperature,
                "max_tokens": settings.llm_max_tokens,
            }
            if not force_text:
                create_kwargs["tools"] = FUNCTION_DEFINITIONS
                create_kwargs["tool_choice"] = "auto"

            response = await asyncio.wait_for(
                client.chat.completions.create(**create_kwargs),
                timeout=_LLM_TIMEOUT,
            )

            choice = response.choices[0]
            message = choice.message

            function_calls = []
            if not force_text and message.tool_calls:
                for tc in message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    function_calls.append({
                        "name": tc.function.name,
                        "arguments": args,
                    })

            usage = None
            if hasattr(response, "usage") and response.usage:
                usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens or 0,
                    completion_tokens=response.usage.completion_tokens or 0,
                    total_tokens=response.usage.total_tokens or 0,
                    model=self.model,
                    provider="openai",
                )

            return LLMResponse(
                message=message.content or "",
                function_calls=function_calls,
                raw_response=response,
                usage=usage,
                stop_reason=choice.finish_reason or "end_turn",
            )
        except Exception as e:
            err_str = str(e)
            logger.error(f"OpenAI 請求失敗: {e}")
            if "insufficient permissions" in err_str.lower() or "missing scopes" in err_str.lower():
                return LLMResponse(
                    message=(
                        "⚠️ **OpenAI API Key 權限不足**\n\n"
                        "你的 API Key 是受限金鑰（Restricted Key），缺少必要的權限。\n\n"
                        "**解決方法**：\n"
                        "1. 前往 [platform.openai.com/api-keys](https://platform.openai.com/api-keys)\n"
                        "2. 建立新的 API Key，選擇 **All（完整權限）**\n"
                        "3. 或編輯現有金鑰，啟用 `model.request` 權限\n"
                        "4. 將新的 Key 貼回系統設定\n\n"
                        "或者切換到 **Google Gemini**（免費額度高）作為替代方案。"
                    )
                )
            return LLMResponse(message=f"LLM 請求失敗: {err_str}")

    async def chat_stream(self, messages: list[dict], chart_state: Optional[dict] = None, system_prompt: Optional[str] = None):
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.api_key, timeout=_LLM_STREAM_TIMEOUT)

            sys_msg = {"role": "system", "content": self._build_system_message(chart_state, system_prompt)}
            all_messages = [sys_msg] + messages

            stream = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.model,
                    messages=all_messages,
                    tools=FUNCTION_DEFINITIONS,
                    tool_choice="auto",
                    temperature=settings.llm_temperature,
                    max_tokens=settings.llm_max_tokens,
                    stream=True,
                ),
                timeout=_LLM_STREAM_TIMEOUT,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"[錯誤] {str(e)}"

    async def chat_stream_events(
        self,
        messages: list[dict],
        chart_state: Optional[dict] = None,
        force_text: bool = False,
        system_prompt: Optional[str] = None,
        chart_screenshot: Optional[str] = None,
        r2_mode: bool = False,
    ):
        """OpenAI 真串流：用 chat.completions.create(stream=True)，邊收 chunk 邊 yield。

        Tool calls 在 streaming 中是 delta 形式（name 在第一個 chunk，
        arguments JSON 慢慢拼），需要按 index 累積。
        """
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.api_key, timeout=_LLM_TIMEOUT)

            # v106 B4：OpenAI auto-caches prompts ≥1024 tokens with stable prefix
            static_sys, dynamic_sys = self._build_system_blocks(chart_state, system_prompt)
            structured_messages: list[dict] = [{"role": "system", "content": static_sys}]
            if dynamic_sys:
                structured_messages.append({"role": "system", "content": dynamic_sys})
            structured_messages.extend(m for m in messages if m.get("role") != "system")
            all_messages = structured_messages

            # 注入圖表截圖到最後一個 user 訊息
            if chart_screenshot:
                for i in range(len(all_messages) - 1, -1, -1):
                    if all_messages[i]["role"] == "user":
                        text_content = all_messages[i]["content"]
                        all_messages[i] = {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"[目前圖表截圖如下]\n{text_content}"},
                                {"type": "image_url", "image_url": {"url": chart_screenshot, "detail": "high"}},
                            ],
                        }
                        break

            create_kwargs: dict = {
                "model": self.model,
                "messages": all_messages,
                "temperature": settings.llm_temperature,
                "max_tokens": settings.llm_max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if not force_text:
                create_kwargs["tools"] = FUNCTION_DEFINITIONS
                create_kwargs["tool_choice"] = "auto"

            stream = await client.chat.completions.create(**create_kwargs)

            # 累積 tool_calls（按 index）
            accumulated: dict[int, dict] = {}
            stop_reason = "end_turn"

            async for chunk in stream:
                # usage 在最後一個 chunk（無 choices）
                if hasattr(chunk, "usage") and chunk.usage:
                    yield StreamEvent(type="usage", usage=TokenUsage(
                        prompt_tokens=chunk.usage.prompt_tokens or 0,
                        completion_tokens=chunk.usage.completion_tokens or 0,
                        total_tokens=chunk.usage.total_tokens or 0,
                        model=self.model,
                        provider="openai",
                    ))
                    continue

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                if getattr(delta, "content", None):
                    yield StreamEvent(type="text_delta", text=delta.content)

                if getattr(delta, "tool_calls", None):
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in accumulated:
                            accumulated[idx] = {"name": "", "arguments": ""}
                        if tc.function:
                            if tc.function.name:
                                accumulated[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                accumulated[idx]["arguments"] += tc.function.arguments

                if choice.finish_reason:
                    stop_reason = choice.finish_reason

            # Yield 累積完成的 function calls
            for tc in accumulated.values():
                if not tc["name"]:
                    continue
                try:
                    args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                yield StreamEvent(type="function_call", function_call={
                    "name": tc["name"],
                    "arguments": args,
                })

            yield StreamEvent(type="stop", stop_reason=stop_reason)

        except Exception as e:
            logger.error(f"OpenAI 串流失敗: {e}")
            yield StreamEvent(type="text_delta", text=f"\n\n[OpenAI 串流錯誤] {str(e)}")
            yield StreamEvent(type="stop", stop_reason="error")


