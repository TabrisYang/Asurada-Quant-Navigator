"""LLM adapter 共用基底 — 常數 / TokenUsage / LLMResponse / BaseLLMAdapter

v154 由 adapter.py 純搬家拆出（原 L1-398），邏輯零改動。
"""

import json
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Optional


from app.core.config.settings import settings
from app.core.llm.function_defs import SYSTEM_PROMPT, SYSTEM_PROMPT_STATIC

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

    # v153：保留「當前指標值」摘要 — R2 綜合掃描結果時必須能定位當前值（如當前
    # BB_Width 落在哪個機率 bin），否則模型只能反問使用者要數字。indicatorValues 是
    # 當前值＋趨勢的小 dict（非 signal_history 大宗）；仍加體積保險絲防 TTFT 回歸。
    iv = chart_state.get("indicatorValues")
    if iv:
        try:
            if len(json.dumps(iv, ensure_ascii=False)) <= 8000:
                minimal["indicatorValues"] = iv
        except Exception:
            pass
    for k in ("donchian_position_pct", "donchian_upper", "donchian_lower"):
        if chart_state.get(k) is not None:
            minimal[k] = chart_state[k]

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


