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
from app.core.llm.function_defs import FUNCTION_DEFINITIONS, SYSTEM_PROMPT, SYSTEM_PROMPT_STATIC

# 各 LLM 供應商統一超時（秒）— 從 settings 讀取，可透過 .env 覆蓋
_LLM_TIMEOUT = settings.llm_timeout
_LLM_STREAM_TIMEOUT = settings.llm_stream_timeout

# v139：Anthropic prompt caching TTL — 預設用 1h（beta），讓「一鍵連跑」6 條訊息
# 都能命中 CORE cache。Anthropic 接受 ttl: "1h" + extended-cache-ttl beta header；
# 若不支援會 silently fallback 到 5min（系統行為一致）。
# 若想暫時關閉，把下方 dict 改回 {"type": "ephemeral"} 即可。
_CACHE_CONTROL_1H = {"type": "ephemeral", "ttl": "1h"}
_ANTHROPIC_CACHE_BETA_HEADER = {"anthropic-beta": "extended-cache-ttl-2025-04-11"}

# v149：限流/過載特徵字串（從 CLI stderr 偵測，用於退避重試判斷）
_RATE_LIMIT_MARKERS = (
    "rate limit", "rate_limit", "ratelimit", "overloaded", "429", "529",
    "usage limit", "too many requests", "quota", "limit reached",
)


def _looks_rate_limited(text: str) -> bool:
    """stderr 是否含限流/過載特徵。"""
    if not text:
        return False
    t = text.lower()
    return any(k in t for k in _RATE_LIMIT_MARKERS)


def _minimal_r2_chart_state(chart_state: Optional[dict]) -> Optional[dict]:
    """R1：round2 用的精簡 chart_state。

    Round 1 已用過完整 chart_state（含 indicators / signal_history /
    external_signals 等 ~200KB 內容）；round 2 僅做文字綜合，重傳完整資料是
    冗餘且讓 Claude CLI TTFT 過長（v118-v120 累積疊加導致 4 分鐘無回應的根因）。

    保留：核心識別欄位 + Round 1 已參考的關鍵統計摘要
    移除：indicators 完整 JSON、signal_history、external_signals 詳情、
          recent_accuracy 內 regime_warning/direction_balance/combo_stats 細節
          （改成單行摘要注入）
    """
    if not chart_state:
        return chart_state

    keep_keys = (
        "symbol", "timeframe", "currentPrice", "currentRegime",
        "regime", "regimeConfidence", "regimeSubtype",
    )
    minimal: dict = {k: chart_state[k] for k in keep_keys if k in chart_state}

    # 摘要 recent_accuracy（Round 1 已詳讀，這裡只保留判讀提示）
    ra = chart_state.get("recent_accuracy") or {}
    if ra:
        ra_summary: dict = {}
        for k in ("win_rate_30d", "win_rate", "total_predictions"):
            if k in ra:
                ra_summary[k] = ra[k]
        # 選項2：decided 勝率摘要（去 expired 稀釋），避免 round2 只看加權值再次誤導
        if ra.get("win_rate_decided_30d") is not None:
            ra_summary["win_rate_decided_30d_summary"] = (
                f"decided={ra.get('win_rate_decided_30d')}% "
                f"(n_dec={ra.get('n_decided_30d')}, expired={ra.get('expired_30d')}, "
                f"ci={ra.get('ci_30d')})"
            )
        # regime_warning / direction_balance / combo_stats 各取一行 verdict（不複製細節）
        rw = ra.get("regime_warning") or {}
        if rw and rw.get("win_rate") is not None:
            ra_summary["regime_warning_summary"] = (
                f"win_rate={rw.get('win_rate')}% n={rw.get('samples', 0)}"
            )
        db = ra.get("direction_balance") or {}
        if db:
            ra_summary["direction_balance_summary"] = (
                f"long_pct={db.get('long_pct', 0)}% biased_long={db.get('biased_long', False)} "
                f"biased_short={db.get('biased_short', False)}"
            )
        sh = ra.get("signal_history") or {}
        cs = (sh.get("combo_stats") or {})
        if cs and cs.get("win_rate") is not None:
            ra_summary["combo_stats_summary"] = (
                f"win_rate={cs.get('win_rate')}% n={cs.get('samples', 0)}"
            )
        # v124：機率三聯壓單行 summary + 保留警示列（round1 已詳讀完整 triplet）
        pt = ra.get("probability_triplet") or {}
        if pt:
            base = pt.get("baseline_unconditional") or {}
            ta = pt.get("ta_conditional") or {}
            tr = pt.get("track_record") or {}
            ra_summary["probability_triplet_summary"] = (
                f"baseline={base.get('prob_pct')}% (n={base.get('n')}) "
                f"ta={ta.get('prob_pct')}% (src={ta.get('source')}) "
                f"track={tr.get('win_rate_raw_pct')}% (n_dec={tr.get('n_decided')}, "
                f"ci={tr.get('ci_pct')})"
            )
            warn = (pt.get("significance") or {}).get("warning_lines") or []
            if warn:
                ra_summary["probability_triplet_warnings"] = warn
        if ra_summary:
            minimal["recent_accuracy"] = ra_summary

    # external_signals 改放摘要（不傳 macro / orderbook / etf 等大欄位）
    ext = chart_state.get("external_signals") or {}
    if ext:
        ext_summary: dict = {}
        deriv = ext.get("derivatives") or {}
        if "funding_rate_pct" in deriv:
            ext_summary["funding_rate_pct"] = deriv["funding_rate_pct"]
        sent = ext.get("sentiment") or {}
        if "fear_greed_value" in sent:
            ext_summary["fear_greed_value"] = sent["fear_greed_value"]
        if ext_summary:
            minimal["external_signals_summary"] = ext_summary

    # v145：保留 rl_strategic_insight（RL 戰略結論）— 它已是精簡格式，Round 2 報告
    # 的 #6.5 RL 戰略結論段需要它，否則只能寫「資料不可得」。直接 copy 不展開。
    rl = chart_state.get("rl_strategic_insight")
    if rl:
        minimal["rl_strategic_insight"] = rl

    minimal["_r2_note"] = (
        "Round 2 精簡狀態：完整 indicators / signal_history / external_signals 已在 Round 1 提供，"
        "此處僅保留核心識別與摘要。請依對話歷史中的 Round 1 結果做綜合分析。"
    )
    return minimal


class TokenUsage:
    """Token 用量統計（含 Anthropic prompt caching 細項）"""

    def __init__(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        model: str = "",
        provider: str = "",
        cache_creation_tokens: int = 0,  # 寫入 cache 的 token 數（首次 + TTL 失效後重寫）
        cache_read_tokens: int = 0,      # 從 cache 讀的 token 數（命中）
    ):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.model = model
        self.provider = provider
        self.cache_creation_tokens = cache_creation_tokens
        self.cache_read_tokens = cache_read_tokens

    def to_dict(self) -> dict:
        cost = estimate_cost(self.provider, self.model, self.prompt_tokens, self.completion_tokens)
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "provider": self.provider,
            "estimated_cost_usd": cost,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_read_tokens": self.cache_read_tokens,
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
        stop_reason: str = "end_turn",
    ):
        self.message = message
        self.function_calls = function_calls or []
        self.raw_response = raw_response
        self.usage = usage
        self.stop_reason = stop_reason  # "end_turn" / "length" / "tool_use"


class StreamEvent:
    """LLM 真串流事件統一格式（chat_stream_events yield 的物件）。

    type 可能值：
      - "text_delta"    : 文字增量（即時逐字 yield）
      - "function_call" : 完整的 function call（args 已 parse 完整）
      - "usage"         : token 用量（通常在串流結尾出現）
      - "stop"          : 結束信號 + stop_reason
    """

    def __init__(
        self,
        type: str,
        text: str = "",
        function_call: Optional[dict] = None,
        usage: Optional[TokenUsage] = None,
        stop_reason: str = "",
    ):
        self.type = type
        self.text = text
        self.function_call = function_call
        self.usage = usage
        self.stop_reason = stop_reason


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
        r2_mode: bool = False,
    ) -> LLMResponse:
        """發送對話請求

        Args:
            force_text: 若為 True，不傳送 tools 給 LLM，強制只產生文字回應。
            system_prompt: 動態組裝的 SYSTEM_PROMPT，傳入後取代預設完整版。
            chart_screenshot: base64 JPEG 圖表截圖（data:image/jpeg;base64,...）。
            r2_mode: 第二輪模式（給 ClaudeSubscriptionAdapter 用，限制工具集）。
        """
        pass

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[dict],
        chart_state: Optional[dict] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """串流對話（舊介面，僅 yield 文字 chunk）

        Args:
            system_prompt: 動態組裝的 SYSTEM_PROMPT，傳入後取代預設完整版。
        """
        pass

    async def chat_stream_events(
        self,
        messages: list[dict],
        chart_state: Optional[dict] = None,
        force_text: bool = False,
        system_prompt: Optional[str] = None,
        chart_screenshot: Optional[str] = None,
        r2_mode: bool = False,
    ) -> AsyncGenerator["StreamEvent", None]:
        """真串流（新介面）— yield 結構化 StreamEvent。

        預設實作：呼叫 chat() 等完整回應後一次性 yield 所有事件（仍是「假串流」）。
        各 adapter 應 override 為真串流，邊收到 token 邊 yield 給呼叫端。
        """
        response = await self.chat(
            messages,
            chart_state=chart_state,
            force_text=force_text,
            system_prompt=system_prompt,
            chart_screenshot=chart_screenshot,
            r2_mode=r2_mode,
        )
        if response.message:
            yield StreamEvent(type="text_delta", text=response.message)
        for fc in response.function_calls:
            yield StreamEvent(type="function_call", function_call=fc)
        if response.usage:
            yield StreamEvent(type="usage", usage=response.usage)
        yield StreamEvent(type="stop", stop_reason=response.stop_reason)

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

    def _build_system_blocks(
        self,
        chart_state: Optional[dict] = None,
        system_prompt: Optional[str] = None,
    ) -> tuple[str, str]:
        """建立 (cacheable_static, dynamic_suffix) 兩段，供 prompt caching 用。

        cacheable_static：純靜態系統 prompt（CORE + 模組），跨請求穩定 → 適合放 cache
        dynamic_suffix：時間戳 + chart_state，每次都不同 → 必須在 cached 段之後

        若呼叫端傳入動態組裝的 system_prompt（含時間戳），則整體當靜態處理
        （由呼叫端負責確保穩定性）。否則用全域 SYSTEM_PROMPT_STATIC。
        """
        if system_prompt:
            cacheable = system_prompt
        else:
            cacheable = SYSTEM_PROMPT_STATIC

        dynamic_parts = []
        # 時間戳：放動態段（每分鐘變）
        from datetime import datetime
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        dynamic_parts.append(f"\n【目前時間】{now_str}（台北時區 UTC+8）")

        if chart_state:
            dynamic_parts.append(
                f"\n\n目前圖表狀態：\n{json.dumps(chart_state, ensure_ascii=False, indent=2)}"
            )

        return cacheable, "".join(dynamic_parts)

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
        """覆寫父類：附加工具定義文字。r2_mode=True 時只附加繪圖函式 + 精簡 chart_state。"""
        # R1: r2_mode=True 時用精簡 chart_state（移除 signal_history 等已在 round1 用過的欄位）
        effective_chart_state = (
            _minimal_r2_chart_state(chart_state) if r2_mode and chart_state else chart_state
        )
        prompt = super()._build_system_message(effective_chart_state, system_prompt)
        if include_tools:
            prompt += self._format_tools_for_prompt(r2_mode=r2_mode)
        else:
            # Round 3：工具已執行完畢，指示 LLM 直接生成文字分析
            prompt += (
                "\n\n【重要提示】所有工具呼叫已在前面的步驟中執行完畢，數據結果已包含在對話歷史中。"
                "你現在的任務是：根據已獲得的數據和計算結果，用文字詳細回答使用者的問題。"
                "不要再嘗試呼叫任何工具或函式，直接提供分析結論、關鍵數據解讀和交易建議。"
            )
        # P2: log prompt size 便於追蹤 chart_state 是否再次膨脹
        sys_kb = len(prompt.encode("utf-8")) / 1024
        cs_kb = (
            len(json.dumps(effective_chart_state, ensure_ascii=False).encode("utf-8")) / 1024
            if effective_chart_state else 0
        )
        if sys_kb >= 50:  # >50KB 才印，避免無關 log 噪音
            logger.info(
                f"[prompt_size] sys={sys_kb:.1f}KB chart_state={cs_kb:.1f}KB r2_mode={r2_mode}"
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
        """使用 stream-json 模式逐行讀取。

        v149：首個輸出快速失敗（120s）+ 限流偵測退避重試。
        舊版對「首個輸出」也套用 1200s timeout → LLM 限流/卡住時要硬等 20 分鐘。
        改成：首 token 120s 無回應即視為卡住/限流，並行 drain stderr 拿原因，
        指數退避重試最多 2 次，最終回傳清楚錯誤訊息給用戶（而非靜默 20 分）。
        """
        import os
        import tempfile

        sys_prompt_file = None
        try:
            sys_prompt = self._build_system_message(
                chart_state, system_prompt, include_tools=not force_text, r2_mode=r2_mode,
            )
            user_prompt = self._build_user_prompt(messages)

            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write(sys_prompt)
                sys_prompt_file = f.name

            _BACKOFFS = [5, 15]          # 限流退避秒數（最多重試 len 次）
            _MAX_ATTEMPTS = len(_BACKOFFS) + 1
            for attempt in range(_MAX_ATTEMPTS):
                full_text, usage, result_data, stderr_text, no_output = (
                    await self._run_cli_attempt(user_prompt, sys_prompt_file)
                )

                if full_text:
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

                # 空回應：判斷是否限流/卡住 → 退避重試
                rate_limited = no_output or _looks_rate_limited(stderr_text)
                if rate_limited and attempt < _MAX_ATTEMPTS - 1:
                    wait_s = _BACKOFFS[attempt]
                    logger.warning(
                        f"Claude CLI 無回應/疑似限流（第 {attempt + 1} 次），{wait_s}s 後退避重試。"
                        f"stderr={stderr_text.strip()[:300]!r}"
                    )
                    await asyncio.sleep(wait_s)
                    continue

                reason = (
                    stderr_text.strip()[:200]
                    or ("首個輸出逾時無回應（疑似限流/忙碌）" if no_output else "回應為空")
                )
                logger.error(f"Claude CLI 最終失敗（attempt {attempt + 1}）：{reason}")
                return LLMResponse(
                    message=f"⚠️ LLM 服務無回應或限流，請稍後重試（原因：{reason}）"
                )

            return LLMResponse(message="⚠️ LLM 服務無回應，請稍後重試")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Claude CLI 請求失敗: {e}")
            return LLMResponse(message=f"Claude 訂閱制請求失敗: {str(e)}")
        finally:
            if sys_prompt_file:
                try:
                    os.unlink(sys_prompt_file)
                except OSError:
                    pass

    async def _run_cli_attempt(
        self, user_prompt: str, sys_prompt_file: str,
    ) -> tuple[str, Optional["TokenUsage"], Optional[dict], str, bool]:
        """跑一次 Claude CLI。回傳 (full_text, usage, result_data, stderr_text, no_output)。

        no_output=True 表示「首個輸出 120s 內零回應」（疑似限流/卡住）。
        並行 drain stderr：(a) 避免 stderr pipe-buffer 填滿導致 CLI 阻塞，
        (b) 拿到真正的失敗原因（限流訊息常寫在 stderr）。
        """
        _FIRST_LINE_TIMEOUT = 120   # 尚未收到任何輸出時：快速失敗
        _IDLE_TIMEOUT = 300         # 已開始輸出後：行與行間最大間隔

        proc = None
        stderr_task = None
        stderr_chunks: list[bytes] = []
        try:
            proc = await asyncio.create_subprocess_exec(
                "claude", "-p",
                "--output-format", "stream-json",
                "--verbose",
                "--include-partial-messages",
                "--model", self.model,
                "--system-prompt-file", sys_prompt_file,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=1024 * 1024,
            )

            proc.stdin.write(user_prompt.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()

            async def _drain_stderr() -> None:
                try:
                    while True:
                        chunk = await proc.stderr.read(4096)
                        if not chunk:
                            break
                        stderr_chunks.append(chunk)
                except Exception:
                    pass

            stderr_task = asyncio.create_task(_drain_stderr())

            full_text = ""
            usage = None
            result_data = None
            got_first = False
            no_output = False

            while True:
                timeout = _IDLE_TIMEOUT if got_first else _FIRST_LINE_TIMEOUT
                try:
                    line = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout)
                except asyncio.TimeoutError:
                    if not got_first:
                        no_output = True
                        logger.warning(
                            f"Claude CLI 首個輸出 {_FIRST_LINE_TIMEOUT}s 無回應（疑似限流/忙碌）"
                        )
                    else:
                        logger.warning(f"Claude CLI 行間 {_IDLE_TIMEOUT}s 無新輸出，結束讀取")
                    break

                if not line:
                    break
                got_first = True

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
                    usage = self._build_usage(data.get("usage", {}), self.model)
                    result_data = data

            # 讓 stderr 收尾（拿完整錯誤原因，尤其限流訊息），最多等 2s
            try:
                await asyncio.wait_for(stderr_task, timeout=2)
            except (asyncio.TimeoutError, Exception):
                pass

            stderr_text = b"".join(stderr_chunks).decode("utf-8", "replace")
            return full_text, usage, result_data, stderr_text, no_output

        finally:
            if stderr_task and not stderr_task.done():
                stderr_task.cancel()
            if proc and proc.returncode is None:
                try:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        proc.kill()
                        await proc.wait()
                except Exception:
                    pass

    async def chat_stream(
        self, messages: list[dict], chart_state: Optional[dict] = None, system_prompt: Optional[str] = None,
        r2_mode: bool = False,
    ):
        """真正的 streaming — 逐 chunk yield 文字"""
        import os
        import tempfile

        proc = None
        sys_prompt_file = None

        try:
            sys_prompt = self._build_system_message(chart_state, system_prompt, r2_mode=r2_mode)
            user_prompt = self._build_user_prompt(messages)

            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write(sys_prompt)
                sys_prompt_file = f.name

            proc = await asyncio.create_subprocess_exec(
                "claude", "-p",
                "--output-format", "stream-json",
                "--verbose",
                "--include-partial-messages",
                "--model", self.model,
                "--system-prompt-file", sys_prompt_file,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=1024 * 1024,
            )

            proc.stdin.write(user_prompt.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()

            # v149：首個輸出快速失敗（120s）；已開始輸出後行間放寬到 300s
            # （舊版首 token 也套 1200s → 限流/卡住時硬等 20 分鐘）
            _FIRST_LINE_TIMEOUT = 120
            _IDLE_TIMEOUT = 300
            got_first = False

            while True:
                timeout = _IDLE_TIMEOUT if got_first else _FIRST_LINE_TIMEOUT
                try:
                    line = await asyncio.wait_for(
                        proc.stdout.readline(), timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    if not got_first:
                        logger.warning(
                            f"Claude CLI streaming 首個輸出 {_FIRST_LINE_TIMEOUT}s 無回應（疑似限流/忙碌）"
                        )
                    break
                if not line:
                    break
                got_first = True

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

        except (asyncio.CancelledError, GeneratorExit):
            pass  # 清理在 finally 中統一處理
        except Exception as e:
            logger.error(f"Claude CLI streaming 失敗: {e}")
            yield f"Claude 訂閱制請求失敗: {str(e)}"
        finally:
            # 確保子進程不會殘留
            if proc and proc.returncode is None:
                try:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=5)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        proc.kill()
                        await proc.wait()
                    logger.info("Claude CLI streaming 子進程已終止")
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            if sys_prompt_file:
                try:
                    os.unlink(sys_prompt_file)
                except OSError:
                    pass

    async def chat_stream_events(
        self,
        messages: list[dict],
        chart_state: Optional[dict] = None,
        force_text: bool = False,
        system_prompt: Optional[str] = None,
        chart_screenshot: Optional[str] = None,
        r2_mode: bool = False,
    ):
        """訂閱版 Claude CLI 真串流：邊讀 stdout 邊 yield，並處理 <tool_call> XML。

        策略：
        - 累積完整 text buffer
        - 每次新 delta 進來時，先掃描已累積的 buffer 找完整 <tool_call>
        - 只 yield「不含 tool_call XML」的安全文字（保留尾端可能正在形成 tag 的部分）
        - 串流結束後 flush 剩餘 + yield usage event
        """
        import os
        import tempfile

        TOOL_OPEN = "<tool_call>"
        TOOL_CLOSE_RE = re.compile(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>")

        proc = None
        sys_prompt_file = None
        text_buffer = ""
        yielded_until = 0  # text_buffer[:yielded_until] 已 yield 過或屬於 tool_call
        usage: Optional[TokenUsage] = None

        try:
            sys_prompt = self._build_system_message(
                chart_state, system_prompt, include_tools=not force_text, r2_mode=r2_mode,
            )
            user_prompt = self._build_user_prompt(messages)

            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write(sys_prompt)
                sys_prompt_file = f.name

            proc = await asyncio.create_subprocess_exec(
                "claude", "-p",
                "--output-format", "stream-json",
                "--verbose",
                "--include-partial-messages",
                "--model", self.model,
                "--system-prompt-file", sys_prompt_file,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=1024 * 1024,
            )

            proc.stdin.write(user_prompt.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()

            # v149：首個輸出快速失敗（120s）；已開始輸出後行間放寬到 300s
            # （舊版首 token 也套 1200s → 限流/卡住時硬等 20 分鐘）
            _FIRST_LINE_TIMEOUT = 120
            _IDLE_TIMEOUT = 300
            got_first = False

            while True:
                timeout = _IDLE_TIMEOUT if got_first else _FIRST_LINE_TIMEOUT
                try:
                    line = await asyncio.wait_for(
                        proc.stdout.readline(), timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    if not got_first:
                        logger.warning(
                            f"Claude CLI stream 首個輸出 {_FIRST_LINE_TIMEOUT}s 無回應（疑似限流/忙碌）"
                        )
                    else:
                        logger.warning(f"Claude CLI stream 行間 {_IDLE_TIMEOUT}s 無新輸出，結束讀取")
                    break
                if not line:
                    break
                got_first = True

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
                            text_buffer += delta.get("text", "")

                            # 嘗試提取完整 tool_call
                            while True:
                                match = TOOL_CLOSE_RE.search(text_buffer, yielded_until)
                                if not match:
                                    break
                                # yield tool_call 之前的安全文字
                                if match.start() > yielded_until:
                                    yield StreamEvent(
                                        type="text_delta",
                                        text=text_buffer[yielded_until:match.start()],
                                    )
                                # 解析並 yield function_call
                                try:
                                    obj = json.loads(match.group(1))
                                    fc = {
                                        "name": obj.get("name", ""),
                                        "arguments": obj.get("arguments", {}),
                                    }
                                    if fc["name"]:
                                        yield StreamEvent(type="function_call", function_call=fc)
                                except json.JSONDecodeError:
                                    pass
                                yielded_until = match.end()

                            # yield 安全文字（保留可能正在形成 <tool_call> 的尾端）
                            partial_open = text_buffer.rfind(TOOL_OPEN, yielded_until)
                            safe_until = (
                                partial_open if partial_open >= 0 else len(text_buffer)
                            )
                            # 若沒看到 <tool_call> 開頭，但結尾可能是 "<tool" 之類的部分匹配 → 也要保留
                            if partial_open < 0:
                                tail = text_buffer[max(yielded_until, len(text_buffer) - 11):]
                                # 結尾若可能是 <tool_call> 的部分前綴，hold 那段
                                for i in range(len(tail), 0, -1):
                                    if TOOL_OPEN.startswith(tail[-i:]):
                                        safe_until = len(text_buffer) - i
                                        break

                            if safe_until > yielded_until:
                                yield StreamEvent(
                                    type="text_delta",
                                    text=text_buffer[yielded_until:safe_until],
                                )
                                yielded_until = safe_until

                elif evt_type == "result":
                    usage_data = data.get("usage", {})
                    usage = self._build_usage(usage_data, self.model)

            await proc.wait()

            # Flush 剩餘 buffer（嘗試最後一次 tool_call 提取）
            while True:
                match = TOOL_CLOSE_RE.search(text_buffer, yielded_until)
                if not match:
                    break
                if match.start() > yielded_until:
                    yield StreamEvent(
                        type="text_delta",
                        text=text_buffer[yielded_until:match.start()],
                    )
                try:
                    obj = json.loads(match.group(1))
                    fc = {"name": obj.get("name", ""), "arguments": obj.get("arguments", {})}
                    if fc["name"]:
                        yield StreamEvent(type="function_call", function_call=fc)
                except json.JSONDecodeError:
                    pass
                yielded_until = match.end()

            # 剩餘文字：若是 unclosed tool_call 就丟掉，否則 yield
            remaining = text_buffer[yielded_until:]
            if remaining:
                if TOOL_OPEN in remaining and "</tool_call>" not in remaining:
                    pass  # 不完整的 tool_call，丟棄
                else:
                    yield StreamEvent(type="text_delta", text=remaining)

            # yield usage + stop
            if usage:
                yield StreamEvent(type="usage", usage=usage)
            yield StreamEvent(type="stop", stop_reason="end_turn")

        except (asyncio.CancelledError, GeneratorExit):
            # v110 修正：之前訊息 [client_disconnect] 誤導 — CancelledError 來源除了 client 真的斷
            # 還包括「上層 wait_for timeout」「正常 generator close」等情況。改中性訊息。
            logger.debug("Claude CLI stream 被取消（可能：client 斷線 / wait_for timeout / generator close）")
            raise
        except Exception as e:
            logger.error(f"Claude CLI streaming events 失敗: {e}")
            yield StreamEvent(type="text_delta", text=f"\n\n[Claude 訂閱版串流錯誤] {str(e)}")
            yield StreamEvent(type="stop", stop_reason="error")
        finally:
            if proc and proc.returncode is None:
                try:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=5)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        proc.kill()
                        await proc.wait()
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            if sys_prompt_file:
                try:
                    os.unlink(sys_prompt_file)
                except OSError:
                    pass


class OllamaAdapter(BaseLLMAdapter):
    """Ollama 本地模型適配器"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def chat(self, messages: list[dict], chart_state: Optional[dict] = None, force_text: bool = False, system_prompt: Optional[str] = None, chart_screenshot: Optional[str] = None, r2_mode: bool = False) -> LLMResponse:
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
