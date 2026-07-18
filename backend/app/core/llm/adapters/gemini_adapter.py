"""GeminiAdapter — v154 由 adapter.py 純搬家拆出（原 L646-980），邏輯零改動。"""

import asyncio
import re
from typing import Any, Optional

from loguru import logger

from app.core.config.settings import settings
from app.core.llm.function_defs import FUNCTION_DEFINITIONS

from app.core.llm.adapters.base import (
    BaseLLMAdapter,
    LLMResponse,
    StreamEvent,
    TokenUsage,
    _LLM_TIMEOUT,
)

class GeminiAdapter(BaseLLMAdapter):
    """Google Gemini 適配器（使用新版 google-genai SDK）

    支援：
    - 429 RESOURCE_EXHAUSTED 自動重試（指數退避）
    - 友善的中文錯誤訊息
    - 只使用使用者選擇的模型，不自動降級
    """

    MAX_RETRIES = 2
    BASE_RETRY_DELAY = 3.0  # 秒

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-lite"):
        self.api_key = api_key
        self.model = model

    def _get_function_declarations(self) -> list[dict]:
        """轉換 OpenAI function 格式為 Gemini function_declarations"""
        declarations = []
        for f in FUNCTION_DEFINITIONS:
            func = f["function"]
            declarations.append({
                "name": func["name"],
                "description": func["description"],
                "parameters": func.get("parameters", {}),
            })
        return declarations

    @staticmethod
    def _is_quota_exhausted(error: Exception) -> bool:
        """判斷是否為 429 配額耗盡錯誤"""
        err_str = str(error)
        return "429" in err_str and "RESOURCE_EXHAUSTED" in err_str

    @staticmethod
    def _parse_retry_delay(error: Exception) -> float:
        """從錯誤訊息中解析 Google 建議的重試秒數"""
        err_str = str(error)
        match = re.search(r"retryDelay.*?(\d+(?:\.\d+)?)\s*s", err_str)
        if match:
            return min(float(match.group(1)), 60.0)  # 最多等 60 秒
        return 10.0

    @staticmethod
    def _friendly_quota_message(model: str, retry_delay: float) -> str:
        """生成使用者友善的配額錯誤訊息"""
        return (
            f"模型 `{model}` 的免費額度已用完。\n\n"
            f"建議方案：\n"
            f"1. 等待 {int(retry_delay)} 秒後重試\n"
            f"2. 到系統設定中選擇其他可用模型\n"
            f"3. 在 Google AI Studio 中升級為付費方案\n"
            f"4. 改用其他 LLM 供應商（如 OpenAI、Claude）"
        )

    async def _call_gemini(
        self, model: str, contents: list, config: Any
    ) -> Any:
        """對 Gemini API 發送請求，含重試邏輯"""
        from google import genai

        client = genai.Client(api_key=self.api_key)

        last_error = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        client.models.generate_content,
                        model=model,
                        contents=contents,
                        config=config,
                    ),
                    timeout=_LLM_TIMEOUT,
                )
                return response
            except Exception as e:
                last_error = e
                if self._is_quota_exhausted(e):
                    if attempt < self.MAX_RETRIES:
                        delay = self.BASE_RETRY_DELAY * (2 ** attempt)
                        logger.warning(
                            f"Gemini {model} 429 限速，{delay:.0f}s 後重試 "
                            f"(第 {attempt + 1}/{self.MAX_RETRIES} 次)"
                        )
                        await asyncio.sleep(delay)
                    else:
                        raise  # 重試耗盡，讓外層處理模型降級
                else:
                    raise  # 非 429 錯誤，直接拋出

        raise last_error  # type: ignore

    async def chat(self, messages: list[dict], chart_state: Optional[dict] = None, force_text: bool = False, system_prompt: Optional[str] = None, chart_screenshot: Optional[str] = None, r2_mode: bool = False) -> LLMResponse:
        import base64
        from google.genai import types

        config_kwargs: dict = {
            "system_instruction": self._build_system_message(chart_state, system_prompt),
            "temperature": settings.llm_temperature,
            "max_output_tokens": settings.llm_max_tokens,
        }
        if not force_text:
            tools = types.Tool(function_declarations=self._get_function_declarations())
            config_kwargs["tools"] = [tools]
        config = types.GenerateContentConfig(**config_kwargs)

        # 轉換訊息格式
        gemini_contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            gemini_contents.append(
                types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
            )

        # 注入圖表截圖到最後一個 user 訊息
        if chart_screenshot:
            b64_data, media_type = self._extract_base64(chart_screenshot)
            image_part = types.Part.from_bytes(data=base64.b64decode(b64_data), mime_type=media_type)
            for i in range(len(gemini_contents) - 1, -1, -1):
                if gemini_contents[i].role == "user":
                    gemini_contents[i].parts.insert(0, types.Part.from_text(text="[目前圖表截圖如下]"))
                    gemini_contents[i].parts.append(image_part)
                    break

        # 只使用使用者選擇的模型，不自動降級
        try:
            response = await self._call_gemini(self.model, gemini_contents, config)

            # 解析 function calls（防護 content/parts 為 None）
            function_calls = []
            text = ""
            if response.candidates:
                candidate = response.candidates[0]
                content = getattr(candidate, "content", None)
                parts = getattr(content, "parts", None) if content else None
                if parts:
                    for part in parts:
                        if hasattr(part, "function_call") and part.function_call:
                            fc = part.function_call
                            function_calls.append({
                                "name": fc.name,
                                "arguments": dict(fc.args) if fc.args else {},
                            })
                        elif hasattr(part, "text") and part.text:
                            text += part.text
                else:
                    finish_reason = getattr(candidate, "finish_reason", None)
                    logger.warning(f"Gemini 回應無內容（finish_reason={finish_reason}）")
                    if not text:
                        text = "⚠️ Gemini 回傳了空白回應，請重新提問或稍後再試。"

            # 提取 token 用量
            usage = None
            usage_meta = getattr(response, "usage_metadata", None)
            if usage_meta:
                usage = TokenUsage(
                    prompt_tokens=getattr(usage_meta, "prompt_token_count", 0) or 0,
                    completion_tokens=getattr(usage_meta, "candidates_token_count", 0) or 0,
                    total_tokens=getattr(usage_meta, "total_token_count", 0) or 0,
                    model=self.model,
                    provider="gemini",
                )

            return LLMResponse(message=text, function_calls=function_calls, raw_response=response, usage=usage)

        except Exception as e:
            if self._is_quota_exhausted(e):
                retry_delay = self._parse_retry_delay(e)
                msg = self._friendly_quota_message(self.model, retry_delay)
                logger.warning(f"Gemini 模型 {self.model} 額度已用完")
                return LLMResponse(message=msg)
            else:
                logger.error(f"Gemini 請求失敗 ({self.model}): {e}")
                return LLMResponse(message=f"Gemini 請求失敗（{self.model}）: {str(e)}")

    async def chat_stream(self, messages: list[dict], chart_state: Optional[dict] = None, system_prompt: Optional[str] = None):
        from google import genai
        from google.genai import types

        tools = types.Tool(function_declarations=self._get_function_declarations())
        config = types.GenerateContentConfig(
            system_instruction=self._build_system_message(chart_state, system_prompt),
            tools=[tools],
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_tokens,
        )

        gemini_contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            gemini_contents.append(
                types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
            )

        try:
            client = genai.Client(api_key=self.api_key)
            # 使用 asyncio.to_thread 避免同步迭代阻塞 event loop
            stream = await asyncio.to_thread(
                client.models.generate_content_stream,
                model=self.model,
                contents=gemini_contents,
                config=config,
            )

            # 將同步迭代器包裝成非阻塞：每次 next() 都放到執行緒池
            while True:
                try:
                    chunk = await asyncio.to_thread(next, stream, None)
                    if chunk is None:
                        break
                    if chunk.text:
                        yield chunk.text
                except StopIteration:
                    break

        except Exception as e:
            if self._is_quota_exhausted(e):
                retry_delay = self._parse_retry_delay(e)
                yield self._friendly_quota_message(self.model, retry_delay)
            else:
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
        """Gemini 真串流：用 generate_content_stream 邊收 chunk 邊 yield。

        Note: Gemini Python SDK 的 stream 是 sync iterator，用 asyncio.to_thread 包裝。
        function_calls 通常出現在最後一個 chunk 或某幾個 chunk。
        """
        import base64
        from google import genai
        from google.genai import types

        config_kwargs: dict = {
            "system_instruction": self._build_system_message(chart_state, system_prompt),
            "temperature": settings.llm_temperature,
            "max_output_tokens": settings.llm_max_tokens,
        }
        if not force_text:
            tools = types.Tool(function_declarations=self._get_function_declarations())
            config_kwargs["tools"] = [tools]
        config = types.GenerateContentConfig(**config_kwargs)

        gemini_contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            gemini_contents.append(
                types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
            )

        # 注入圖表截圖
        if chart_screenshot:
            b64_data, media_type = self._extract_base64(chart_screenshot)
            image_part = types.Part.from_bytes(data=base64.b64decode(b64_data), mime_type=media_type)
            for i in range(len(gemini_contents) - 1, -1, -1):
                if gemini_contents[i].role == "user":
                    gemini_contents[i].parts.insert(0, types.Part.from_text(text="[目前圖表截圖如下]"))
                    gemini_contents[i].parts.append(image_part)
                    break

        try:
            client = genai.Client(api_key=self.api_key)
            stream = await asyncio.to_thread(
                client.models.generate_content_stream,
                model=self.model,
                contents=gemini_contents,
                config=config,
            )

            collected_function_calls: list[dict] = []
            last_usage_meta = None
            stop_reason = "end_turn"

            while True:
                chunk = await asyncio.to_thread(next, stream, None)
                if chunk is None:
                    break

                # 文字內容（可能在 candidates[0].content.parts[*].text）
                if chunk.candidates:
                    cand = chunk.candidates[0]
                    content = getattr(cand, "content", None)
                    parts = getattr(content, "parts", None) if content else None
                    if parts:
                        for part in parts:
                            if hasattr(part, "function_call") and part.function_call:
                                fc = part.function_call
                                collected_function_calls.append({
                                    "name": fc.name,
                                    "arguments": dict(fc.args) if fc.args else {},
                                })
                            elif hasattr(part, "text") and part.text:
                                yield StreamEvent(type="text_delta", text=part.text)
                    fr = getattr(cand, "finish_reason", None)
                    if fr:
                        stop_reason = str(fr).lower()

                # 累積最新 usage_metadata
                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    last_usage_meta = chunk.usage_metadata

            # Yield function calls
            for fc in collected_function_calls:
                yield StreamEvent(type="function_call", function_call=fc)

            # Yield usage
            if last_usage_meta:
                yield StreamEvent(type="usage", usage=TokenUsage(
                    prompt_tokens=getattr(last_usage_meta, "prompt_token_count", 0) or 0,
                    completion_tokens=getattr(last_usage_meta, "candidates_token_count", 0) or 0,
                    total_tokens=getattr(last_usage_meta, "total_token_count", 0) or 0,
                    model=self.model,
                    provider="gemini",
                ))

            yield StreamEvent(type="stop", stop_reason=stop_reason)

        except Exception as e:
            if self._is_quota_exhausted(e):
                retry_delay = self._parse_retry_delay(e)
                yield StreamEvent(type="text_delta", text=self._friendly_quota_message(self.model, retry_delay))
            else:
                logger.error(f"Gemini 串流失敗: {e}")
                yield StreamEvent(type="text_delta", text=f"\n\n[Gemini 串流錯誤] {str(e)}")
            yield StreamEvent(type="stop", stop_reason="error")


