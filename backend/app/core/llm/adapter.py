"""阿斯拉量化系統 — LLM 統一適配器

支援 OpenAI / Gemini / Claude / Ollama。
所有供應商的 Function Calling 都轉換為統一格式。
"""

import asyncio
import json
import re
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Optional

from loguru import logger

from app.core.config.settings import settings
from app.core.llm.function_defs import FUNCTION_DEFINITIONS, SYSTEM_PROMPT, assemble_system_prompt

# 各 LLM 供應商統一超時（秒）— 從 settings 讀取，可透過 .env 覆蓋
_LLM_TIMEOUT = settings.llm_timeout
_LLM_STREAM_TIMEOUT = settings.llm_stream_timeout


class TokenUsage:
    """Token 用量統計"""

    def __init__(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        model: str = "",
        provider: str = "",
    ):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.model = model
        self.provider = provider

    def to_dict(self) -> dict:
        cost = estimate_cost(self.provider, self.model, self.prompt_tokens, self.completion_tokens)
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "provider": self.provider,
            "estimated_cost_usd": cost,
        }


# ─── 費用估算（每百萬 token，單位 USD）───────────
_PRICING: dict[str, dict[str, tuple[float, float]]] = {
    # provider -> { model_prefix: (input_per_1M, output_per_1M) }
    "openai": {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.00),
        "gpt-4-turbo": (10.00, 30.00),
        "gpt-4": (30.00, 60.00),
        "gpt-3.5-turbo": (0.50, 1.50),
        "o1": (15.00, 60.00),
        "o3": (10.00, 40.00),
    },
    "gemini": {
        "gemini-2.0-flash-lite": (0.0, 0.0),  # 免費
        "gemini-1.5-flash": (0.075, 0.30),
        "gemini-2.0-flash": (0.10, 0.40),
        "gemini-1.5-pro": (1.25, 5.00),
        "gemini-2.0-pro": (1.25, 5.00),
    },
    "claude": {
        "claude-sonnet-4": (3.00, 15.00),
        "claude-3-5-sonnet": (3.00, 15.00),
        "claude-3-5-haiku": (0.80, 4.00),
        "claude-3-opus": (15.00, 75.00),
    },
    "claude_subscription": {},  # 訂閱制，無額外費用
    "ollama": {},  # 本地免費
}


def estimate_cost(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """根據供應商和模型估算費用（USD）"""
    provider_pricing = _PRICING.get(provider, {})
    # 找最長匹配的 prefix
    matched_price = None
    best_len = 0
    for prefix, price in provider_pricing.items():
        if model.startswith(prefix) and len(prefix) > best_len:
            matched_price = price
            best_len = len(prefix)
    if not matched_price:
        return 0.0
    input_cost = (prompt_tokens / 1_000_000) * matched_price[0]
    output_cost = (completion_tokens / 1_000_000) * matched_price[1]
    return round(input_cost + output_cost, 6)


class LLMResponse:
    """統一的 LLM 回應格式"""

    def __init__(
        self,
        message: str = "",
        function_calls: Optional[list[dict[str, Any]]] = None,
        raw_response: Any = None,
        usage: Optional[TokenUsage] = None,
    ):
        self.message = message
        self.function_calls = function_calls or []
        self.raw_response = raw_response
        self.usage = usage


class BaseLLMAdapter(ABC):
    """LLM 適配器基礎類"""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        chart_state: Optional[dict] = None,
        force_text: bool = False,
        system_prompt: Optional[str] = None,
        chart_screenshot: Optional[str] = None,
    ) -> LLMResponse:
        """發送對話請求

        Args:
            force_text: 若為 True，不傳送 tools 給 LLM，強制只產生文字回應。
            system_prompt: 動態組裝的 SYSTEM_PROMPT，傳入後取代預設完整版。
            chart_screenshot: base64 JPEG 圖表截圖（data:image/jpeg;base64,...）。
        """
        pass

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[dict],
        chart_state: Optional[dict] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """串流對話

        Args:
            system_prompt: 動態組裝的 SYSTEM_PROMPT，傳入後取代預設完整版。
        """
        pass

    def _build_system_message(
        self,
        chart_state: Optional[dict] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """建立系統提示（含圖表狀態摘要）。可傳入動態組裝的 prompt 取代預設。"""
        prompt = system_prompt or SYSTEM_PROMPT
        if chart_state:
            prompt += f"\n\n目前圖表狀態：\n{json.dumps(chart_state, ensure_ascii=False, indent=2)}"
        return prompt

    @staticmethod
    def _extract_base64(data_url: str) -> tuple[str, str]:
        """從 data URL 提取 base64 資料和 media type。"""
        if data_url.startswith("data:"):
            header, b64 = data_url.split(",", 1)
            media_type = header.split(":")[1].split(";")[0]
            return b64, media_type
        return data_url, "image/jpeg"


class OpenAIAdapter(BaseLLMAdapter):
    """OpenAI (GPT-4/4o) 適配器"""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model

    async def chat(self, messages: list[dict], chart_state: Optional[dict] = None, force_text: bool = False, system_prompt: Optional[str] = None, chart_screenshot: Optional[str] = None) -> LLMResponse:
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

            create_kwargs: dict = {
                "model": self.model,
                "messages": all_messages,
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

    async def chat(self, messages: list[dict], chart_state: Optional[dict] = None, force_text: bool = False, system_prompt: Optional[str] = None, chart_screenshot: Optional[str] = None) -> LLMResponse:
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

    async def chat(self, messages: list[dict], chart_state: Optional[dict] = None, force_text: bool = False, system_prompt: Optional[str] = None, chart_screenshot: Optional[str] = None) -> LLMResponse:
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

            create_kwargs: dict = {
                "model": self.model,
                "max_tokens": settings.llm_max_tokens,
                "system": self._build_system_message(chart_state, system_prompt),
                "messages": claude_messages,
                "temperature": settings.llm_temperature,
            }
            if not force_text:
                create_kwargs["tools"] = self._convert_functions_to_claude()

            response = await asyncio.wait_for(
                client.messages.create(**create_kwargs),
                timeout=_LLM_TIMEOUT,
            )

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

            # 提取 token 用量
            usage = None
            if hasattr(response, "usage") and response.usage:
                usage = TokenUsage(
                    prompt_tokens=getattr(response.usage, "input_tokens", 0) or 0,
                    completion_tokens=getattr(response.usage, "output_tokens", 0) or 0,
                    total_tokens=(getattr(response.usage, "input_tokens", 0) or 0)
                        + (getattr(response.usage, "output_tokens", 0) or 0),
                    model=self.model,
                    provider="claude",
                )

            return LLMResponse(message=text, function_calls=function_calls, raw_response=response, usage=usage)
        except Exception as e:
            logger.error(f"Claude 請求失敗: {e}")
            return LLMResponse(message=f"LLM 請求失敗: {str(e)}")

    async def chat_stream(self, messages: list[dict], chart_state: Optional[dict] = None, system_prompt: Optional[str] = None):
        response = await self.chat(messages, chart_state, system_prompt=system_prompt)
        yield response.message


class ClaudeSubscriptionAdapter(BaseLLMAdapter):
    """透過 claude CLI 使用訂閱額度的適配器"""

    def __init__(self, model: str = "sonnet"):
        self.model = model

    # ── 共用 ──

    # Round 2 只需要這 3 個繪圖/指標函式（chat.py L1133-1136 已過濾）
    _R2_ALLOWED_FUNCTIONS = {"annotate_chart", "draw_pattern", "manage_indicator"}

    @staticmethod
    def _format_tools_for_prompt(r2_mode: bool = False) -> str:
        """將 FUNCTION_DEFINITIONS 轉成文字，注入 system prompt。
        r2_mode=True 時只包含繪圖相關函式，大幅減少 token。
        """
        funcs = FUNCTION_DEFINITIONS
        if r2_mode:
            funcs = [
                fd for fd in FUNCTION_DEFINITIONS
                if (fd.get("function", fd)).get("name", "") in ClaudeSubscriptionAdapter._R2_ALLOWED_FUNCTIONS
            ]

        lines = [
            "\n\n【可用工具 — Function Call】",
            "當你需要執行操作時，用以下 XML 格式呼叫工具：",
            "<tool_call>{\"name\": \"函式名\", \"arguments\": {參數}}</tool_call>",
            "",
            "可用函式列表：",
        ]
        for i, fd in enumerate(funcs, 1):
            func = fd.get("function", fd)
            name = func.get("name", "")
            desc = func.get("description", "")
            lines.append(f"\n{i}. {name} — {desc}")
            params = func.get("parameters", {})
            props = params.get("properties", {})
            required = set(params.get("required", []))
            if props:
                lines.append("   參數:")
                for pname, pinfo in props.items():
                    ptype = pinfo.get("type", "string")
                    pdesc = pinfo.get("description", "")
                    enum_vals = pinfo.get("enum")
                    req_mark = " (必填)" if pname in required else ""
                    enum_str = f", 可選值: {enum_vals}" if enum_vals else ""
                    lines.append(f"   - {pname} ({ptype}{req_mark}): {pdesc}{enum_str}")
        if r2_mode:
            lines.append("\n你只能使用以上列出的工具來標記圖表和管理指標。")
        else:
            lines.append("\n重要：你必須主動呼叫這些工具來獲取真實數據，不要憑空猜測。")
        return "\n".join(lines)

    def _build_system_message(
        self,
        chart_state: Optional[dict] = None,
        system_prompt: Optional[str] = None,
        include_tools: bool = True,
        r2_mode: bool = False,
    ) -> str:
        """覆寫父類：附加工具定義文字。r2_mode=True 時只附加繪圖函式。"""
        prompt = super()._build_system_message(chart_state, system_prompt)
        if include_tools:
            prompt += self._format_tools_for_prompt(r2_mode=r2_mode)
        else:
            # Round 3：工具已執行完畢，指示 LLM 直接生成文字分析
            prompt += (
                "\n\n【重要提示】所有工具呼叫已在前面的步驟中執行完畢，數據結果已包含在對話歷史中。"
                "你現在的任務是：根據已獲得的數據和計算結果，用文字詳細回答使用者的問題。"
                "不要再嘗試呼叫任何工具或函式，直接提供分析結論、關鍵數據解讀和交易建議。"
            )
        return prompt

    def _build_user_prompt(self, messages: list[dict]) -> str:
        """組裝對話歷史 + 當前訊息為單一 prompt"""
        prompt_parts: list[str] = []
        last_user_msg = ""
        for msg in messages:
            role_label = "使用者" if msg["role"] == "user" else "助手"
            content = msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
            if msg["role"] == "user":
                last_user_msg = content
            prompt_parts.append(f"{role_label}: {content}")
        return "\n\n".join(prompt_parts) if len(messages) > 1 else last_user_msg

    @staticmethod
    def _parse_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
        """解析 <tool_call> XML 並回傳 (clean_text, function_calls)"""
        tool_call_re = re.compile(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>")
        function_calls: list[dict[str, Any]] = []
        for match in tool_call_re.finditer(text):
            try:
                obj = json.loads(match.group(1))
                fc = {"name": obj.get("name", ""), "arguments": obj.get("arguments", {})}
                if fc["name"]:
                    function_calls.append(fc)
            except json.JSONDecodeError:
                pass
        clean_text = tool_call_re.sub("", text).strip()
        return clean_text, function_calls

    @staticmethod
    def _build_usage(usage_data: dict, model: str) -> TokenUsage:
        inp = usage_data.get("input_tokens", 0) or 0
        out = usage_data.get("output_tokens", 0) or 0
        cache_read = usage_data.get("cache_read_input_tokens", 0) or 0
        cache_create = usage_data.get("cache_creation_input_tokens", 0) or 0
        return TokenUsage(
            prompt_tokens=inp + cache_read + cache_create,
            completion_tokens=out,
            total_tokens=inp + out + cache_read + cache_create,
            model=model,
            provider="claude_subscription",
        )

    async def chat(
        self,
        messages: list[dict],
        chart_state: Optional[dict] = None,
        force_text: bool = False,
        system_prompt: Optional[str] = None,
        chart_screenshot: Optional[str] = None,
        r2_mode: bool = False,
    ) -> LLMResponse:
        """使用 stream-json 模式逐行讀取，避免整體 timeout"""
        import os
        import tempfile

        try:
            sys_prompt = self._build_system_message(
                chart_state, system_prompt, include_tools=not force_text, r2_mode=r2_mode,
            )
            user_prompt = self._build_user_prompt(messages)

            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write(sys_prompt)
                sys_prompt_file = f.name

            try:
                proc = await asyncio.create_subprocess_exec(
                    "claude", "-p",
                    "--output-format", "stream-json",
                    "--verbose",
                    "--include-partial-messages",
                    "--model", self.model,
                    "--system-prompt-file", sys_prompt_file,
                    # "--no-session-persistence",  # 啟用 prompt caching 加速連續對話
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    limit=1024 * 1024,  # 1MB buffer（預設 64KB，長回應會溢出）
                )

                proc.stdin.write(user_prompt.encode("utf-8"))
                await proc.stdin.drain()
                proc.stdin.close()

                full_text = ""
                usage = None
                result_data = None
                _line_timeout = 60  # 每行最多等 60 秒

                while True:
                    try:
                        line = await asyncio.wait_for(
                            proc.stdout.readline(), timeout=_line_timeout,
                        )
                    except asyncio.TimeoutError:
                        logger.warning("Claude CLI stream 逐行讀取超時（60s 無新輸出）")
                        break

                    if not line:
                        break

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    evt_type = data.get("type")

                    if evt_type == "stream_event":
                        event = data.get("event", {})
                        if event.get("type") == "content_block_delta":
                            delta = event.get("delta", {})
                            if delta.get("type") == "text_delta":
                                full_text += delta.get("text", "")

                    elif evt_type == "result":
                        full_text = data.get("result", full_text)
                        usage_data = data.get("usage", {})
                        usage = self._build_usage(usage_data, self.model)
                        result_data = data

                await proc.wait()
            finally:
                os.unlink(sys_prompt_file)

            if not full_text:
                return LLMResponse(message="Claude CLI 回應為空")

            if not usage:
                usage = TokenUsage(
                    prompt_tokens=0, completion_tokens=0, total_tokens=0,
                    model=self.model, provider="claude_subscription",
                )

            clean_text, function_calls = self._parse_tool_calls(full_text)

            return LLMResponse(
                message=clean_text,
                function_calls=function_calls,
                raw_response=result_data,
                usage=usage,
            )

        except Exception as e:
            logger.error(f"Claude CLI 請求失敗: {e}")
            return LLMResponse(message=f"Claude 訂閱制請求失敗: {str(e)}")

    async def chat_stream(
        self, messages: list[dict], chart_state: Optional[dict] = None, system_prompt: Optional[str] = None,
        r2_mode: bool = False,
    ):
        """真正的 streaming — 逐 chunk yield 文字"""
        import os
        import tempfile

        try:
            sys_prompt = self._build_system_message(chart_state, system_prompt, r2_mode=r2_mode)
            user_prompt = self._build_user_prompt(messages)

            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write(sys_prompt)
                sys_prompt_file = f.name

            try:
                proc = await asyncio.create_subprocess_exec(
                    "claude", "-p",
                    "--output-format", "stream-json",
                    "--verbose",
                    "--include-partial-messages",
                    "--model", self.model,
                    "--system-prompt-file", sys_prompt_file,
                    # "--no-session-persistence",  # 啟用 prompt caching 加速連續對話
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    limit=1024 * 1024,  # 1MB buffer（預設 64KB，長回應會溢出）
                )

                proc.stdin.write(user_prompt.encode("utf-8"))
                await proc.stdin.drain()
                proc.stdin.close()

                _line_timeout = 60

                while True:
                    try:
                        line = await asyncio.wait_for(
                            proc.stdout.readline(), timeout=_line_timeout,
                        )
                    except asyncio.TimeoutError:
                        break
                    if not line:
                        break

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if data.get("type") == "stream_event":
                        event = data.get("event", {})
                        if event.get("type") == "content_block_delta":
                            delta = event.get("delta", {})
                            text = delta.get("text", "")
                            if text:
                                yield text

                await proc.wait()
            finally:
                os.unlink(sys_prompt_file)

        except Exception as e:
            logger.error(f"Claude CLI streaming 失敗: {e}")
            yield f"Claude 訂閱制請求失敗: {str(e)}"


class OllamaAdapter(BaseLLMAdapter):
    """Ollama 本地模型適配器"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def chat(self, messages: list[dict], chart_state: Optional[dict] = None, force_text: bool = False, system_prompt: Optional[str] = None, chart_screenshot: Optional[str] = None) -> LLMResponse:
        try:
            import httpx

            sys_msg = {"role": "system", "content": self._build_system_message(chart_state, system_prompt)}
            all_messages = [sys_msg] + messages

            # Ollama 多模態模型（llava 等）支援 images 欄位
            if chart_screenshot:
                b64_data, _ = self._extract_base64(chart_screenshot)
                for i in range(len(all_messages) - 1, -1, -1):
                    if all_messages[i]["role"] == "user":
                        all_messages[i] = {
                            **all_messages[i],
                            "images": [b64_data],
                        }
                        break

            async with httpx.AsyncClient(timeout=float(_LLM_TIMEOUT)) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": all_messages,
                        "stream": False,
                        "options": {"temperature": settings.llm_temperature},
                    },
                )
                data = response.json()
                # 提取 token 用量
                usage = TokenUsage(
                    prompt_tokens=data.get("prompt_eval_count", 0) or 0,
                    completion_tokens=data.get("eval_count", 0) or 0,
                    total_tokens=(data.get("prompt_eval_count", 0) or 0) + (data.get("eval_count", 0) or 0),
                    model=self.model,
                    provider="ollama",
                )
                return LLMResponse(
                    message=data.get("message", {}).get("content", ""),
                    raw_response=data,
                    usage=usage,
                )
        except Exception as e:
            logger.error(f"Ollama 請求失敗: {e}")
            return LLMResponse(message=f"Ollama 請求失敗: {str(e)}")

    async def chat_stream(self, messages: list[dict], chart_state: Optional[dict] = None, system_prompt: Optional[str] = None):
        try:
            import httpx

            sys_msg = {"role": "system", "content": self._build_system_message(chart_state, system_prompt)}
            all_messages = [sys_msg] + messages

            async with httpx.AsyncClient(timeout=float(_LLM_STREAM_TIMEOUT)) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": all_messages,
                        "stream": True,
                    },
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                content = data.get("message", {}).get("content", "")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                pass
        except Exception as e:
            yield f"[錯誤] {str(e)}"


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
        from app.core.auth.claude_oauth import check_claude_cli_available
        status = check_claude_cli_available()
        if not status["available"]:
            raise ValueError(status["error"] or "Claude CLI 不可用")
        return ClaudeSubscriptionAdapter(model=model_name)

    elif provider == "ollama":
        return OllamaAdapter(
            base_url=base_url or "http://localhost:11434",
            model=model_name,
        )

    else:
        raise ValueError(f"不支援的 LLM 供應商: {provider}")
