"""阿斯拉量化系統 — LLM 對話路由（Streaming + 對話歷史 + Function Call 二輪回應）

核心改進：
1. 支援完整對話歷史（前端傳送最近 N 輪，LLM 能記住上下文）
2. Function call 執行後，將結果回傳 LLM 做第二輪分析回應
3. SSE 串流式回傳，體感像即時打字
4. 四層快取：知識快取 → 分析快取(hash) → 語意快取(向量) → LLM
"""

import asyncio
import json
import re
import time
import uuid
from collections import defaultdict
from typing import Optional
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse
from loguru import logger

from app.core.llm.adapter import create_adapter
from app.core.llm.executor import execute_function_calls, check_input_safety
from app.core.llm.function_defs import detect_intents, assemble_system_prompt
from app.core.security.key_manager import key_manager
from app.core.usage_tracker import usage_tracker
from app.core.chat_history import chat_history
from app.core.knowledge_cache import knowledge_cache
from app.core.analysis_cache import analysis_cache
from app.core.semantic_cache import semantic_cache
from app.core.knowledge_distiller import knowledge_distiller
from app.core.knowledge_fragments import (
    fragment_store, parse_key_insights, strip_key_insights,
    parse_system_distill, strip_system_distill,
)
from app.core.symbol_extractor import extract_symbol_from_text
from app.core.user_strategies import get_enabled_strategies_prompt
from app.core.prediction_tracker import (
    prediction_tracker, parse_predictions, strip_predictions,
)
from app.core.prediction_validator import validate_all_active
from app.core.prediction_feedback import (
    generate_feedback_prompt, get_active_predictions_summary,
)
from app.core.backtest.parameter_optimizer import format_calibration_for_prompt
from app.models.schemas import ChatRequest, ChatResponse, TokenUsageResponse

router = APIRouter()

# 對話歷史最多保留的訊息數（避免 token 過多）
MAX_HISTORY_MESSAGES = 20

# ─── 簡易速率限制（每 IP 每分鐘最多 30 次 chat 請求）───
_RATE_LIMIT_WINDOW = 60  # 秒
_RATE_LIMIT_MAX = 30     # 每窗口最大請求數
_rate_limit_store: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(client_ip: str) -> bool:
    """檢查是否超過速率限制，返回 True 表示允許"""
    now = time.time()
    timestamps = _rate_limit_store[client_ip]
    # 清除過期記錄
    _rate_limit_store[client_ip] = [t for t in timestamps if now - t < _RATE_LIMIT_WINDOW]
    if len(_rate_limit_store[client_ip]) >= _RATE_LIMIT_MAX:
        return False
    _rate_limit_store[client_ip].append(now)
    return True

# 指標名稱到 indicator_id 的映射（用於自動偵測 LLM 文字中提到的指標）
_INDICATOR_TEXT_MAP: dict[str, tuple[str, str]] = {
    # (文字關鍵字, (indicator_id, display_mode))
    "rsi": ("rsi", "sub_chart"),
    "相對強弱": ("rsi", "sub_chart"),
    "macd": ("macd", "sub_chart"),
    "布林": ("bb", "overlay"),
    "bollinger": ("bb", "overlay"),
    "sma": ("sma", "overlay"),
    "簡單移動平均": ("sma", "overlay"),
    "ema": ("ema", "overlay"),
    "指數移動平均": ("ema", "overlay"),
    "adx": ("adx", "sub_chart"),
    "平均趨向": ("adx", "sub_chart"),
    "atr": ("atr", "sub_chart"),
    "真實波動": ("atr", "sub_chart"),
    "obv": ("obv", "sub_chart"),
    "能量潮": ("obv", "sub_chart"),
    "rel_vol": ("rel_vol", "sub_chart"),
    "相對量能": ("rel_vol", "sub_chart"),
    "relative volume": ("rel_vol", "sub_chart"),
    "stochrsi": ("stochrsi", "sub_chart"),
    "隨機相對強弱": ("stochrsi", "sub_chart"),
    "supertrend": ("supertrend", "overlay"),
    "超級趨勢": ("supertrend", "overlay"),
    "ichimoku": ("ichimoku", "overlay"),
    "一目均衡": ("ichimoku", "overlay"),
    "vwap": ("vwap", "overlay"),
    "psar": ("psar", "overlay"),
    "拋物線": ("psar", "overlay"),
    "donchian": ("donchian", "overlay"),
    "唐奇安": ("donchian", "overlay"),
    "keltner": ("keltner", "overlay"),
    "肯特納": ("keltner", "overlay"),
    "roc": ("roc", "sub_chart"),
    "變動率": ("roc", "sub_chart"),
    "bias": ("bias", "sub_chart"),
    "乖離率": ("bias", "sub_chart"),
    "vol_switch": ("vol_switch", "sub_chart"),
    "量價背離": ("vol_switch", "sub_chart"),
    "trailing_stop": ("trailing_stop", "overlay"),
    "追蹤止損": ("trailing_stop", "overlay"),
    "kelly": ("kelly", "sub_chart"),
    "凱利": ("kelly", "sub_chart"),
    "max_drawdown": ("max_drawdown", "sub_chart"),
    "最大回撤": ("max_drawdown", "sub_chart"),
    "fear_greed": ("fear_greed", "sub_chart"),
    "恐懼貪婪": ("fear_greed", "sub_chart"),
    "funding": ("funding", "sub_chart"),
    "資金費率": ("funding", "sub_chart"),
    "market_structure": ("market_structure", "overlay"),
    "市場結構": ("market_structure", "overlay"),
    "harmonic": ("harmonic", "overlay"),
    "諧波": ("harmonic", "overlay"),
    "cvd": ("cvd", "sub_chart"),
    "累計量差": ("cvd", "sub_chart"),
    "poc": ("poc", "overlay"),
    "成交量密集": ("poc", "overlay"),
    "hv": ("hv", "sub_chart"),
    "歷史波動率": ("hv", "sub_chart"),
    "vol_squeeze": ("vol_squeeze", "sub_chart"),
    "波動壓縮": ("vol_squeeze", "sub_chart"),
    "rsi_divergence": ("rsi_divergence", "sub_chart"),
    "rsi背離": ("rsi_divergence", "sub_chart"),
    "rsi 背離": ("rsi_divergence", "sub_chart"),
    "macd_divergence": ("macd_divergence", "sub_chart"),
    "macd背離": ("macd_divergence", "sub_chart"),
    "macd 背離": ("macd_divergence", "sub_chart"),
    "vol_divergence": ("vol_divergence", "sub_chart"),
    "成交量背離": ("vol_divergence", "sub_chart"),
    "leading_composite": ("leading_composite", "sub_chart"),
    "綜合先行": ("leading_composite", "sub_chart"),
    "先行訊號": ("leading_composite", "sub_chart"),
    "mtf_mss": ("mtf_mss", "sub_chart"),
    "mss": ("mtf_mss", "sub_chart"),
    "結構轉變": ("mtf_mss", "sub_chart"),
    "多時間框架": ("mtf_mss", "sub_chart"),
}

# ─── JSON function call 過濾（方案 A 安全網）───────────

_KNOWN_FC_KEYS = {"group_name", "annotations", "annotation_type", "pattern_name", "points"}

# fenced code blocks: ```json ... ``` or ``` ... ```
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*\n(\{[\s\S]*?\})\s*\n```")
# inline JSON with known FC keys
_INLINE_JSON_RE = re.compile(
    r'(\{[^{}]*"(?:group_name|annotation_type|pattern_name|annotations)"[\s\S]*?\}(?:\s*\})?)'
)


def _extract_json_function_calls(text: str) -> tuple[str, list[dict]]:
    """偵測並提取 LLM 文字中誤輸出的 JSON function call。

    Returns:
        (cleaned_text, extracted_function_calls)
        cleaned_text: 移除 JSON 區塊後的文字
        extracted_function_calls: 可執行的 function call 列表
    """
    if not text:
        return text, []

    extracted: list[dict] = []
    spans_to_remove: list[tuple[int, int]] = []

    # 先嘗試 fenced code blocks，再嘗試 inline JSON
    all_matches = list(_FENCED_JSON_RE.finditer(text)) + list(_INLINE_JSON_RE.finditer(text))
    # 按起始位置排序，並去除重疊
    all_matches.sort(key=lambda m: m.start())
    seen_spans: set[tuple[int, int]] = set()
    for match in all_matches:
        span = (match.start(), match.end())
        if any(s[0] <= span[0] < s[1] for s in seen_spans):
            continue
        seen_spans.add(span)
        raw = match.group(1)
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            # 嘗試修復常見的 JSON 問題（尾逗號等）
            cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
            try:
                obj = json.loads(cleaned)
            except json.JSONDecodeError:
                continue

        if not isinstance(obj, dict):
            continue

        # 判斷是否為已知的 function call 格式
        fc = _try_parse_as_function_call(obj)
        if fc:
            extracted.append(fc)
            spans_to_remove.append((match.start(), match.end()))

    if not spans_to_remove:
        return text, []

    # 從後往前移除，避免 offset 錯位
    cleaned = text
    for start, end in reversed(spans_to_remove):
        cleaned = cleaned[:start] + cleaned[end:]

    # 清理多餘空行
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    logger.info(f"從 LLM 文字中提取 {len(extracted)} 個 JSON function call 並移除原始 JSON")
    return cleaned, extracted


def _try_parse_as_function_call(obj: dict) -> dict | None:
    """嘗試將 JSON 物件解析為可執行的 function call。"""
    # annotate_chart 格式
    if "group_name" in obj and ("annotations" in obj or "annotation_type" in obj):
        return {"name": "annotate_chart", "arguments": obj}

    # draw_pattern 格式
    if "pattern_name" in obj and "points" in obj:
        return {"name": "draw_pattern", "arguments": obj}

    # manage_indicator 格式
    if "action" in obj and "indicator_id" in obj:
        return {"name": "manage_indicator", "arguments": obj}

    return None


def _detect_mentioned_indicators(text: str, existing_ids: set[str]) -> list[dict]:
    """從 LLM 文字中偵測提到但未透過 function call 添加的指標"""
    if not text:
        return []
    text_lower = text.lower()
    detected: dict[str, str] = {}  # indicator_id → display_mode
    for keyword, (ind_id, display_mode) in _INDICATOR_TEXT_MAP.items():
        if keyword in text_lower and ind_id not in existing_ids and ind_id not in detected:
            detected[ind_id] = display_mode
    return [
        {"action": "add", "indicator_id": ind_id, "display_mode": dm}
        for ind_id, dm in detected.items()
    ]


# 數值相關關鍵詞（避免只是問定義時觸發計算）
_VALUE_KEYWORDS = {"多少", "是多少", "目前", "現在", "幾", "數值", "分析", "趨勢", "看"}

# ─── 按意圖等級定義核心指標集 ────────────────────────────
# analysis: 覆蓋 regime 判定 + 4 維度最小必要集
_ANALYSIS_CORE_INDICATORS = [
    "adx", "atr", "rsi", "macd", "bb", "obv",
]
# deep analysis: 覆蓋全 7 維度 + 先行訊號
_DEEP_ANALYSIS_INDICATORS = [
    "adx", "atr", "rsi", "macd", "bb", "obv",
    "supertrend", "stochrsi", "donchian", "rel_vol",
    "leading_composite", "mtf_mss",
]

_MAX_AUTO_CALC = 15


def _auto_calc_indicator_values(
    user_msg: str,
    chart_state: dict | None,
    intents: set[str] | None = None,
) -> dict:
    """根據意圖等級 + 使用者提及的指標，自動計算數值注入 chart_state。

    三種觸發模式（按優先順序）：
    1. 深度分析意圖 → 自動計算 _DEEP_ANALYSIS_INDICATORS（12 個）
    2. 一般分析意圖 → 自動計算 _ANALYSIS_CORE_INDICATORS（6 個）
    3. 關鍵字比對   → 只計算使用者提到的指標（原始邏輯）
    """
    if not chart_state or not user_msg:
        return chart_state or {}

    _intents = intents or set()
    msg_lower = user_msg.lower()
    existing_keys = set((chart_state.get("indicatorValues") or {}).keys())
    need_calc: list[str] = []

    is_deep = bool(_intents & {"backtest", "quant_research", "event_analysis", "calibrate"})
    is_analysis = "analysis" in _intents

    if is_deep:
        for ind_id in _DEEP_ANALYSIS_INDICATORS:
            if ind_id not in existing_keys and ind_id not in need_calc:
                need_calc.append(ind_id)
    elif is_analysis:
        for ind_id in _ANALYSIS_CORE_INDICATORS:
            if ind_id not in existing_keys and ind_id not in need_calc:
                need_calc.append(ind_id)

    has_value_intent = is_analysis or is_deep or any(kw in user_msg for kw in _VALUE_KEYWORDS)
    if has_value_intent:
        for keyword, (ind_id, _dm) in _INDICATOR_TEXT_MAP.items():
            if keyword in msg_lower and ind_id not in need_calc:
                if ind_id not in existing_keys:
                    need_calc.append(ind_id)
            if len(need_calc) >= _MAX_AUTO_CALC:
                break

    if not need_calc:
        return chart_state

    symbol = chart_state.get("symbol", "")
    timeframe = chart_state.get("timeframe", "4h")

    try:
        from app.data.fetchers.crypto_engine import crypto_engine
        from app.core.indicators import registry as ind_registry

        df = crypto_engine.load_local_data(symbol, timeframe)
        if df.empty or len(df) < 10:
            return chart_state

        auto_values = dict(chart_state.get("indicatorValues") or {})

        for ind_id in need_calc:
            try:
                calc_result = ind_registry.calculate(ind_id, df)
                if not calc_result:
                    continue
                for series_name, series_data in calc_result.items():
                    if not series_data:
                        continue
                    tail = series_data[-5:]
                    rounded = [round(v, 4) if v is not None else None for v in tail]
                    valid = [v for v in rounded if v is not None]
                    trend = "→"
                    if len(valid) >= 2:
                        diff = valid[-1] - valid[0]
                        thr = abs(valid[0]) * 0.02 or 0.001
                        if diff > thr:
                            trend = "↑"
                        elif diff < -thr:
                            trend = "↓"
                    key = series_name if series_name != "value" else ind_id
                    auto_values[key] = {"values": rounded, "trend": trend, "auto": True}
            except Exception:
                pass

        chart_state = {**chart_state, "indicatorValues": auto_values}
        if need_calc:
            logger.info(f"自動計算指標值 (intent={_intents}): {need_calc}")
    except Exception as e:
        logger.warning(f"自動計算指標值失敗: {e}")

    return chart_state


# 觸發摘要壓縮的閾值（歷史訊息超過此數就壓縮舊的部分）
SUMMARY_THRESHOLD = 10

# 壓縮後保留最近 N 輪的原文
KEEP_RECENT_MESSAGES = 6


def _resolve_api_key(request: ChatRequest) -> tuple[str, str | None, str | None, str | None]:
    """從請求中解析出 provider、api_key、base_url、model_name"""
    provider = (request.llm_provider or "openai").value if request.llm_provider else "openai"
    api_key = None
    base_url = None
    model_name = None

    if request.session_id:
        session_info = key_manager.get_session_info(request.session_id)
        if session_info:
            provider = session_info["provider"]
            base_url = session_info.get("base_url")
            model_name = session_info.get("model_name")
            api_key = key_manager.get_key(request.session_id)
        else:
            api_key = None

    if not api_key and request.api_key:
        api_key = request.api_key

    return provider, api_key, base_url, model_name


def _inject_ml_prediction(
    chart_state: dict | None,
    intents: set[str] | None = None,
) -> dict:
    """在 chart_state 中注入 ML 預測信號（若可用且啟用）"""
    if not chart_state:
        return chart_state or {}

    _intents = intents or set()
    is_analysis = bool(_intents & {
        "analysis", "backtest", "quant_research", "event_analysis", "calibrate",
    })
    if not is_analysis:
        return chart_state

    symbol = chart_state.get("symbol", "")
    timeframe = chart_state.get("timeframe", "4h")
    if not symbol:
        return chart_state

    try:
        from app.core.ml.model_manager import model_manager

        should_enable, reason = model_manager.should_enable_ml(symbol, timeframe)
        if not should_enable:
            logger.debug(f"ML 未啟用: {reason}")
            return chart_state

        from app.data.fetchers.crypto_engine import crypto_engine
        df = crypto_engine.load_local_data(symbol, timeframe)
        if df.empty or len(df) < 100:
            return chart_state

        result = model_manager.predict_consensus(df, symbol, timeframe)
        if result.get("status") == "success":
            chart_state["mlPrediction"] = result["prediction"]
            mode = result["prediction"].get("consensus_mode", "best")
            n_models = result["prediction"].get("models_considered", 1)
            logger.info(
                f"ML 預測注入: {symbol} {timeframe} → "
                f"{result['prediction']['direction']} "
                f"(prob={result['prediction']['probability']:.1%}, "
                f"consensus={mode}, models={n_models})"
            )
    except Exception as e:
        logger.debug(f"ML 預測注入失敗（不影響分析）: {e}")

    return chart_state


def _build_messages(
    request: ChatRequest,
    rag_fragments: Optional[list[dict]] = None,
    intents: Optional[set[str]] = None,
) -> list[dict]:
    """
    從請求中建立完整的 messages 列表。
    根據偵測到的意圖（intents）決定注入哪些背景資料，
    所有背景資料合併為**一個** user/assistant pair，大幅節省 token。

    結構：
    1. [合併的系統背景資料]（按需）
    2. [歷史訊息摘要 或 原文]
    3. [當前標的意識]
    4. [當前使用者訊息]
    """
    messages: list[dict] = []
    _intents = intents or {"general"}

    chart_symbol = (request.chart_state or {}).get("symbol")
    needs_deep_context = bool(
        _intents & {"analysis", "backtest", "quant_research", "event_analysis", "calibrate"}
    )

    # ★ 按需合併背景資料為單一注入（舊版每項各一組 user/assistant，最多 12 條訊息；新版只有 2 條）
    if needs_deep_context:
        context_parts: list[str] = []

        strategies_prompt = get_enabled_strategies_prompt()
        if strategies_prompt:
            context_parts.append(f"【使用者自訂分析方法論】\n{strategies_prompt}")

        distilled_context = knowledge_distiller.get_context_for_symbol(chart_symbol)
        if distilled_context:
            context_parts.append(f"【歷史分析記憶】\n{distilled_context}")

        if _intents & {"analysis", "backtest", "quant_research"}:
            _feedback_prompt = generate_feedback_prompt(symbol=chart_symbol)
            if _feedback_prompt:
                context_parts.append(f"【預測績效反饋】\n{_feedback_prompt}")
            _active_summary = get_active_predictions_summary(chart_symbol)
            if _active_summary:
                context_parts.append(f"【進行中的預測】\n{_active_summary}")

        if chart_symbol and (_intents & {"analysis", "backtest", "calibrate"}):
            _calibration_prompt = format_calibration_for_prompt(chart_symbol)
            if _calibration_prompt:
                context_parts.append(f"【指標參數校準數據】\n{_calibration_prompt}")

        if rag_fragments:
            frag_texts = [
                f"• [{f['type']}] {f['content']}（相關度 {f['similarity']:.0%}）"
                for f in rag_fragments
            ]
            context_parts.append(f"【歷史分析經驗碎片】\n" + "\n".join(frag_texts))

        if context_parts:
            merged = "\n\n---\n\n".join(context_parts)
            messages.append({
                "role": "user",
                "content": f"[系統背景資料 — 供分析參考，不要直接複述]\n\n{merged}",
            })
            messages.append({
                "role": "assistant",
                "content": "已收到背景資料，我會在分析時參考。",
            })

    # 過濾有效歷史訊息
    history: list[dict] = []
    if request.messages:
        for msg in request.messages[-MAX_HISTORY_MESSAGES:]:
            if msg.role in ("user", "assistant") and msg.content.strip():
                history.append({"role": msg.role, "content": msg.content})

    if len(history) > SUMMARY_THRESHOLD:
        old_messages = history[:-KEEP_RECENT_MESSAGES]
        recent_messages = history[-KEEP_RECENT_MESSAGES:]
        summary = _compress_to_summary(old_messages)
        if summary:
            messages.append({
                "role": "user",
                "content": f"[對話摘要 — 以下是之前對話的重點整理，非原文]\n{summary}",
            })
            messages.append({
                "role": "assistant",
                "content": "好的，我已了解之前的對話內容，請繼續。",
            })
        messages.extend(recent_messages)
    else:
        messages.extend(history)

    # ★ 注入當前標的意識指令（精簡版，防止跨標的混淆）
    chart_timeframe_ctx = (request.chart_state or {}).get("timeframe", "")
    if chart_symbol:
        messages.append({
            "role": "user",
            "content": (
                f"[系統提醒] 目前標的：{chart_symbol}（{chart_timeframe_ctx}），"
                f"所有分析和函式呼叫針對此標的，忽略歷史中其他標的殘留內容。"
            ),
        })
        messages.append({
            "role": "assistant",
            "content": f"了解，專注分析 {chart_symbol}。",
        })

    # 加入當前使用者訊息
    if request.mode == "quant_research":
        quant_prefix = (
            "[系統指令：使用者點擊了「回測分析」按鈕，要求對以下內容進行量化驗證]\n"
            "你必須：\n"
            "1. 先用 query_chart_data 取得當前指標數據，分析目前的市場環境\n"
            "2. 如果使用者提到具體開倉價位（如「在 0.255 做多」），按照【價位策略轉換指引】：\n"
            "   - 把當前市場指標狀態轉換為進場條件（RSI 區間、ADX、趨勢方向等）\n"
            "   - 可加入 close 的寬鬆區間匹配（±3%）\n"
            "   - 把止損價換算成 stop_loss_pct 百分比\n"
            "   - 使用者提到的槓桿倍數傳入 leverage 參數\n"
            "3. 呼叫 run_quant_research（完整研究）或 run_backtest（快速回測）\n"
            "4. 用回測數據回答：勝率、報酬率、Sharpe、Sortino、Expectancy、最大回撤\n"
            "5. 如果有足夠交易次數，Monte Carlo 模擬評估破產風險\n"
            "6. 給出明確結論：策略是否可行、風險等級、建議倉位和槓桿\n"
            "**禁止只用文字估算，必須呼叫函式用數據驗證。**\n\n"
            f"使用者的策略描述：{request.message}"
        )
        messages.append({"role": "user", "content": quant_prefix})
    elif request.mode == "calibrate":
        calibrate_prefix = (
            "[系統指令：使用者點擊了「校準指標」按鈕，要求對當前標的執行指標參數校準]\n"
            "你必須：\n"
            "1. 呼叫 optimize_indicator_params 對當前標的進行參數校準\n"
            "2. 校準完成後，用表格清楚列出每個指標的：\n"
            "   - 校準用途（做多閾值、做空閾值、動能門檻等）\n"
            "   - 穩健區間（而非單一值）\n"
            "   - 可信度等級（★~★★★）\n"
            "   - 與教科書通用值的對比\n"
            "3. 說明哪些校準結果可信度高可以直接使用，哪些需要交叉參考\n"
            "4. 校準結果已自動儲存，告知使用者之後的分析會自動參考這些校準值\n\n"
            f"使用者備註：{request.message if request.message.strip() else '請校準全部可用指標'}"
        )
        messages.append({"role": "user", "content": calibrate_prefix})
    else:
        messages.append({"role": "user", "content": request.message})

    return messages


def _compress_to_summary(old_messages: list[dict]) -> str:
    """
    將舊對話壓縮成精簡摘要（純規則，不需額外 LLM 呼叫）。

    策略：提取每輪使用者問題的關鍵資訊 + 助手回答的核心數字/結論。
    保留：幣種、時間範圍、數字、日期、指標名稱。
    """
    summary_parts = []
    current_user_msg = None

    for msg in old_messages:
        if msg["role"] == "user":
            current_user_msg = msg["content"]
        elif msg["role"] == "assistant" and current_user_msg:
            # 壓縮一組 Q&A
            q = current_user_msg[:80] + ("..." if len(current_user_msg) > 80 else "")
            # 取助手回答的前 150 字（保留數字和結論）
            a = msg["content"][:150] + ("..." if len(msg["content"]) > 150 else "")
            summary_parts.append(f"Q: {q}\nA: {a}")
            current_user_msg = None

    if not summary_parts:
        return ""

    return "\n---\n".join(summary_parts)


def _format_function_results(function_calls: list[dict], exec_result: dict) -> str:
    """將 function call 執行結果格式化為 LLM 可讀的文字摘要"""
    parts = []

    results = exec_result.get("results", [])
    chart_updates = exec_result.get("chart_updates", {})

    for i, fc in enumerate(function_calls):
        fname = fc.get("name", "unknown")
        fargs = fc.get("arguments", {})
        result = results[i] if i < len(results) else {}

        parts.append(f"### 函式 {i+1}: {fname}")
        parts.append(f"參數: {json.dumps(fargs, ensure_ascii=False)}")

        if "error" in result:
            parts.append(f"錯誤: {result['error']}")
        elif "result" in result:
            r = result["result"]
            # 格式化不同類型的結果
            if fname == "query_chart_data":
                cu = r.get("chart_updates", {})
                parts.append(f"結果: {cu.get('symbol', '?')} {cu.get('timeframe', '?')}，"
                             f"共 {cu.get('dataPoints', 0)} 根 K 線")
                ps = r.get("price_summary")
                if ps:
                    parts.append(f"期間最高: {ps.get('period_high')} ({ps.get('period_high_date')})")
                    parts.append(f"期間最低: {ps.get('period_low')} ({ps.get('period_low_date')})")
                    parts.append(f"首日開盤: {ps.get('first_open')} ({ps.get('first_date')})")
                    parts.append(f"最後收盤: {ps.get('last_close')} ({ps.get('last_date')})")
                    for mo in (ps.get("monthly_ohlc") or []):
                        parts.append(f"  {mo['m']}: H={mo['h']} L={mo['l']} C={mo['c']}")
                    for day in (ps.get("daily_ohlc") or []):
                        parts.append(f"  {day['d']}: H={day['h']} L={day['l']} C={day['c']}")
                    for c in (ps.get("candles") or []):
                        parts.append(f"  {c['t']}: H={c['h']} L={c['l']} C={c['c']}")
            elif fname == "find_conditions":
                parts.append(f"結果: 找到 {r.get('matched_periods', 0)} 個匹配時間點")
                if r.get("summary"):
                    parts.append(f"摘要: {r['summary']}")
            elif fname == "manage_indicator":
                parts.append(f"結果: {r.get('action', '?')} 指標 {r.get('indicator_name', r.get('indicator_id', '?'))}")
            elif fname == "suggest_indicators":
                recs = r.get("recommended", [])
                names = ", ".join(rec.get("name", rec.get("id", "?")) for rec in recs)
                parts.append(f"推薦指標: {names}")
            elif fname == "scan_conditional_probability":
                parts.append(f"目標: {r.get('target', '?')}")
                parts.append(f"數據範圍: {r.get('data_range', '?')}，共 {r.get('total_bars', 0)} 根 K 線")
                ob = r.get("overall_best", {})
                if ob:
                    parts.append(f"★ 最佳區間: {ob.get('indicator', '?')} = {ob.get('range', '?')} → "
                                 f"機率 {ob.get('prob_pct', 0)}%（基線 {ob.get('baseline_pct', 0)}%，提升 {ob.get('lift', 0)} 個百分點）")
                for key, ind_data in r.get("indicators", {}).items():
                    parts.append(f"\n指標 {key}（樣本 {ind_data.get('total_valid_samples', 0)}）:")
                    parts.append(f"  基線機率: {ind_data.get('baseline_prob_pct', 0)}%")
                    parts.append(f"  最佳區間: {ind_data.get('best_range', '?')} → {ind_data.get('best_prob_pct', 0)}%")
                    for b in ind_data.get("bins", []):
                        if b.get("prob_pct") is not None:
                            parts.append(f"  {b['range']}: {b['prob_pct']}% ({b['hit']}/{b['count']})")
                        elif b.get("note"):
                            parts.append(f"  {b['range']}: {b['note']} ({b['count']})")
            else:
                # 通用格式化（截斷過長內容）
                result_str = json.dumps(r, ensure_ascii=False)
                if len(result_str) > 500:
                    result_str = result_str[:500] + "..."
                parts.append(f"結果: {result_str}")
        parts.append("")

    if chart_updates:
        parts.append(f"圖表更新: {json.dumps(chart_updates, ensure_ascii=False)[:300]}")

    return "\n".join(parts)


def _sse_event(event_type: str, data: dict) -> str:
    """格式化 SSE event"""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ─── 同步端點（保留向下相容）────────────────────

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """處理 LLM 對話請求（同步模式）"""

    if not check_input_safety(request.message):
        return ChatResponse(
            message="偵測到不安全的輸入，請重新輸入你的分析需求。",
            function_calls=[], chart_updates=None,
            conversation_id=request.conversation_id or str(uuid.uuid4()),
        )

    provider, api_key, base_url, model_name = _resolve_api_key(request)

    if not api_key and provider not in ("ollama", "claude_subscription"):
        msg = (
            "Session 已過期，請重新在設定中輸入 API Key。"
            if request.session_id
            else f"請先在設定中輸入 {provider.upper()} 的 API Key。點擊右上角「設定」進行配置。"
        )
        return ChatResponse(
            message=msg, function_calls=[], chart_updates=None,
            conversation_id=request.conversation_id or str(uuid.uuid4()),
        )

    try:
        adapter = create_adapter(provider=provider, api_key=api_key, model_name=model_name, base_url=base_url)
    except Exception as e:
        return ChatResponse(
            message=f"無法連接 LLM: {str(e)}", function_calls=[], chart_updates=None,
            conversation_id=request.conversation_id or str(uuid.uuid4()),
        )

    messages = _build_messages(request)
    try:
        response = await adapter.chat(messages, chart_state=request.chart_state, chart_screenshot=request.chart_screenshot)
    except Exception as e:
        logger.error(f"LLM 呼叫失敗: {e}")
        return ChatResponse(
            message=f"LLM 回應失敗: {str(e)}", function_calls=[], chart_updates=None,
            conversation_id=request.conversation_id or str(uuid.uuid4()),
        )

    chart_updates = None
    if response.function_calls:
        exec_result = await execute_function_calls(response.function_calls, chart_state=request.chart_state)
        chart_updates = exec_result.get("chart_updates")

    usage_resp = None
    if response.usage:
        usage_resp = TokenUsageResponse(**response.usage.to_dict())

    return ChatResponse(
        message=response.message,
        function_calls=response.function_calls,
        chart_updates=chart_updates,
        conversation_id=request.conversation_id or str(uuid.uuid4()),
        usage=usage_resp,
    )


# ─── Streaming 端點（主要對話模式）────────────────

@router.post("/stream")
async def chat_stream(request: ChatRequest, raw_request: Request):
    """
    處理 LLM 對話請求（Streaming 模式 + 對話歷史 + Function Call 二輪）

    流程：
    1. 立即發送 "thinking" 事件
    2. 建立完整 messages（歷史 + 當前）
    3. 呼叫 LLM 取得第一輪結果
    4. 如果有 function calls → 執行 → 結果回傳 LLM → 第二輪回應
    5. 串流文字、function calls、chart updates、usage
    6. 發送 done 事件
    """

    # 速率限制檢查
    client_ip = raw_request.client.host if raw_request.client else "unknown"
    if not _check_rate_limit(client_ip):
        return JSONResponse(
            status_code=429,
            content={"detail": "請求過於頻繁，請稍後再試（每分鐘上限 30 次）"},
        )

    # 安全檢查
    if not check_input_safety(request.message):
        async def error_gen():
            yield _sse_event("error", {"error": "偵測到不安全的輸入"})
            yield _sse_event("done", {})
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    # ★ L1 知識快取：純知識類問題直接回傳，完全不消耗 token
    cached_answer = knowledge_cache.try_answer(request.message)
    if cached_answer:
        conv_id = request.conversation_id or str(uuid.uuid4())
        async def cache_gen():
            # L1 也補發指標（從答案文字偵測）
            _auto_inds = _detect_mentioned_indicators(cached_answer, set())
            if _auto_inds:
                yield _sse_event("chart", {"chart_updates": {"indicator_actions": _auto_inds}})
            for chunk in _split_text_for_streaming(cached_answer):
                yield _sse_event("token", {"content": chunk})
                await asyncio.sleep(0.02)
            yield _sse_event("done", {"conversation_id": conv_id})
        chat_history.save_message(conversation_id=conv_id, role="user", content=request.message)
        chat_history.save_message(conversation_id=conv_id, role="assistant", content=cached_answer)
        return StreamingResponse(cache_gen(), media_type="text/event-stream")

    # ★ L2/L3 快取已停用回傳功能（投資分析時效性高，快取舊答案風險大於省 token 的好處）
    # L2/L3 仍會「儲存」LLM 回答，用於知識碎片提取和蒸餾，但不再攔截使用者問題。
    # 所有非 L1 知識問題一律走 LLM 即時分析。

    # ★ L3.5 知識融合準備（中等相似度 0.75~0.92 → RAG 注入碎片，低 token）
    _rag_context_fragments: list[dict] = []
    semantic_match = semantic_cache.try_get_with_score(request.message, request.chart_state)
    if semantic_match and semantic_match["similarity"] >= 0.75:
        logger.info(
            f"L3.5 語意匹配（中等置信度 {semantic_match['similarity']:.2%}）→ 作為 RAG 參考"
        )

    chart_symbol = (request.chart_state or {}).get("symbol", "")
    mentioned_symbol = extract_symbol_from_text(request.message)
    rag_symbol = mentioned_symbol or chart_symbol
    if rag_symbol:
        _rag_context_fragments = fragment_store.retrieve_relevant(
            question=request.message,
            symbol=rag_symbol,
            top_k=5,
            min_similarity=0.45,
        )
        if _rag_context_fragments:
            logger.info(
                f"L3.5 知識碎片命中 {len(_rag_context_fragments)} 筆 "
                f"(最高相似度 {_rag_context_fragments[0]['similarity']:.2%})"
            )

    provider, api_key, base_url, model_name = _resolve_api_key(request)

    if not api_key and provider not in ("ollama", "claude_subscription"):
        async def no_key_gen():
            msg = "Session 已過期，請重新在設定中輸入 API Key。" if request.session_id \
                else f"請先在設定中輸入 {provider.upper()} 的 API Key。點擊右上角「設定」進行配置。"
            yield _sse_event("token", {"content": msg})
            yield _sse_event("done", {})
        return StreamingResponse(no_key_gen(), media_type="text/event-stream")

    try:
        adapter = create_adapter(provider=provider, api_key=api_key, model_name=model_name, base_url=base_url)
    except Exception as e:
        async def err_gen():
            yield _sse_event("error", {"error": f"無法連接 LLM: {str(e)}"})
            yield _sse_event("done", {})
        return StreamingResponse(err_gen(), media_type="text/event-stream")

    # ★ 意圖偵測 → 決定載入哪些 SYSTEM_PROMPT 模組與背景資料
    _intents = detect_intents(request.message, mode=request.mode)
    from app.api.routes.config import load_system_settings
    _teaching = load_system_settings().get("teaching_mode", False)
    _dynamic_prompt = assemble_system_prompt(_intents, teaching_mode=_teaching)
    logger.info(f"意圖偵測: {_intents}, 教學模式={'ON' if _teaching else 'OFF'} → SYSTEM_PROMPT 模組已動態組裝")

    # ★ 自動計算分析所需 + 使用者提到的指標值，注入 chart_state
    request.chart_state = _auto_calc_indicator_values(request.message, request.chart_state, intents=_intents)

    # ★ ML 增強：自動注入 ML 預測信號
    request.chart_state = _inject_ml_prediction(request.chart_state, _intents)

    messages = _build_messages(request, rag_fragments=_rag_context_fragments, intents=_intents)
    conversation_id = request.conversation_id or str(uuid.uuid4())

    # ★ 保存使用者訊息到歷史
    chart_symbol_for_save = (request.chart_state or {}).get("symbol")
    chart_timeframe = (request.chart_state or {}).get("timeframe")
    chart_timeframe_ctx = chart_timeframe or ""
    chat_history.save_message(
        conversation_id=conversation_id,
        role="user",
        content=request.message,
        symbol=chart_symbol_for_save,
        timeframe=chart_timeframe,
    )

    _MAX_FINAL_TEXT = 50_000  # 累積文字最大長度（~50KB），防止記憶體爆炸

    async def stream_gen():
        total_usage = None
        final_text = ""
        _active_tasks: list[asyncio.Task] = []

        # 1. 立即告知前端「正在思考」
        yield _sse_event("thinking", {})
        yield _sse_event("status", {"message": "正在分析您的問題..."})

        try:
            # 2. 第一輪 LLM 呼叫（使用意圖驅動的動態 SYSTEM_PROMPT，帶心跳）
            _llm_task = asyncio.create_task(adapter.chat(
                messages, chart_state=request.chart_state,
                system_prompt=_dynamic_prompt,
                chart_screenshot=request.chart_screenshot,
            ))
            _active_tasks.append(_llm_task)
            _hb_sec = 0
            while not _llm_task.done():
                await asyncio.sleep(3)
                _hb_sec += 3
                yield _sse_event("status", {"message": f"正在分析您的問題... ({_hb_sec}秒)"})
            response = _llm_task.result()

            if response.usage:
                total_usage = response.usage

            logger.info(
                f"第一輪結果: 文字長度={len(response.message)}, "
                f"function_calls={len(response.function_calls)}"
            )

            # 3. 如果 LLM 回傳了 function calls → 執行 → 二輪回應
            if response.function_calls:
                # 先串流第一輪的文字（如果有，移除 KEY_INSIGHTS / PREDICTIONS + JSON 過濾）
                if response.message:
                    display_msg = strip_system_distill(strip_predictions(strip_key_insights(response.message)))
                    display_msg, _ = _extract_json_function_calls(display_msg)
                    if len(final_text) < _MAX_FINAL_TEXT:
                        final_text += response.message[:_MAX_FINAL_TEXT - len(final_text)]
                    for chunk in _split_text_for_streaming(display_msg):
                        yield _sse_event("token", {"content": chunk})
                        await asyncio.sleep(0.02)

                # 發送 function calls 事件
                yield _sse_event("function", {"function_calls": response.function_calls})
                yield _sse_event("status", {"message": "正在執行圖表操作..."})

                # 執行 function calls（帶心跳，避免使用者以為當機）
                try:
                    _fc_task = asyncio.create_task(execute_function_calls(
                        response.function_calls, chart_state=request.chart_state,
                    ))
                    _active_tasks.append(_fc_task)
                    _hb_sec = 0
                    while not _fc_task.done():
                        await asyncio.sleep(3)
                        _hb_sec += 3
                        yield _sse_event("status", {"message": f"正在執行分析運算... ({_hb_sec}秒)"})
                    exec_result = _fc_task.result()

                    chart_updates = exec_result.get("chart_updates")
                    if chart_updates:
                        yield _sse_event("chart", {"chart_updates": chart_updates})
                        _ann_count1 = len(chart_updates.get("annotations", []))
                        if _ann_count1 > 0:
                            yield _sse_event("status", {"message": f"已添加 {_ann_count1} 筆圖表標記"})

                    # ★ 核心改進：將結果回傳 LLM 做第二輪分析
                    fc_summary = _format_function_results(response.function_calls, exec_result)
                    logger.info(f"Function call 結果摘要：{fc_summary[:200]}...")

                    # 建立第二輪 messages：精簡版（只保留使用者原文 + 標的提醒 + 助手回應 + 工具結果）
                    # ★ 不再複製完整第一輪 messages（含背景資料、歷史），大幅節省 token
                    round2_messages: list[dict] = []
                    _user_original = messages[-1]["content"] if messages else request.message
                    _r2_symbol_hint = ""
                    if chart_symbol:
                        _r2_symbol_hint = f"（標的：{chart_symbol} {chart_timeframe_ctx}）"
                    round2_messages.append({
                        "role": "user",
                        "content": f"{_r2_symbol_hint}{_user_original}",
                    })
                    if response.message:
                        round2_messages.append({"role": "assistant", "content": response.message})
                    round2_messages.append({
                        "role": "user",
                        "content": (
                            f"[系統自動回傳] 以下是你剛才呼叫的函式執行結果：\n\n"
                            f"{fc_summary}\n\n"
                            f"請根據以上數據結果回答使用者的問題。\n"
                            f"如果使用者要求在圖表上畫線、標記、型態等，你**必須**呼叫 annotate_chart 或 draw_pattern 函式來繪製。\n"
                            f"你也可以呼叫 manage_indicator 來添加分析中用到的指標到圖表上。\n"
                            f"除了 annotate_chart、draw_pattern 和 manage_indicator 以外，不要呼叫其他函式。"
                        ),
                    })

                    # 第二輪 LLM 呼叫（帶心跳，沿用同一份動態 prompt）
                    yield _sse_event("status", {"message": "正在整理分析結果..."})
                    _r2_task = asyncio.create_task(adapter.chat(
                        round2_messages, chart_state=request.chart_state,
                        system_prompt=_dynamic_prompt,
                    ))
                    _active_tasks.append(_r2_task)
                    _hb_sec = 0
                    while not _r2_task.done():
                        await asyncio.sleep(3)
                        _hb_sec += 3
                        yield _sse_event("status", {"message": f"正在整理分析結果... ({_hb_sec}秒)"})
                    response2 = _r2_task.result()

                    if response2.usage and total_usage:
                        total_usage.prompt_tokens += response2.usage.prompt_tokens
                        total_usage.completion_tokens += response2.usage.completion_tokens
                        total_usage.total_tokens += response2.usage.total_tokens
                    elif response2.usage:
                        total_usage = response2.usage

                    # ★ 處理第二輪的 function calls（annotate_chart / draw_pattern / manage_indicator）
                    _has_draw_calls = False
                    _r2_indicator_ids: set[str] = set()
                    if response2.function_calls:
                        _has_draw_calls = any(
                            fc.get("name") in ("annotate_chart", "draw_pattern")
                            for fc in response2.function_calls
                        )
                        for fc in response2.function_calls:
                            if fc.get("name") == "manage_indicator":
                                _r2_indicator_ids.add(fc.get("arguments", {}).get("indicator_id", ""))

                        allowed_r2 = [
                            fc for fc in response2.function_calls
                            if fc.get("name") in ("annotate_chart", "draw_pattern", "manage_indicator")
                        ]
                        if allowed_r2:
                            yield _sse_event("status", {"message": "正在更新圖表..."})
                            try:
                                exec_result2 = await execute_function_calls(
                                    allowed_r2, chart_state=request.chart_state,
                                )
                                chart_updates2 = exec_result2.get("chart_updates")
                                if chart_updates2:
                                    yield _sse_event("chart", {"chart_updates": chart_updates2})
                                    _ann_count = len(chart_updates2.get("annotations", []))
                                    _ind_count = len(chart_updates2.get("indicator_actions", []))
                                    parts = []
                                    if _ann_count > 0:
                                        parts.append(f"{_ann_count} 筆圖表標記")
                                    if _ind_count > 0:
                                        parts.append(f"{_ind_count} 個指標")
                                    if parts:
                                        yield _sse_event("status", {"message": f"已添加 {'、'.join(parts)}"})
                            except Exception as e2:
                                logger.warning(f"第二輪 function call 執行失敗: {e2}")

                    # ★ 處理第二輪文字回應
                    _r2_text = response2.message or ""
                    logger.info(
                        f"第二輪結果: 文字長度={len(_r2_text)}, "
                        f"function_calls={len(response2.function_calls)}"
                    )

                    # ★★ 關鍵修復：若第二輪只有 function calls 沒有文字
                    #    （OpenAI 回傳 tool_calls 時 content=null），
                    #    自動追加第三輪純文字生成。
                    if not _r2_text.strip() and response2.function_calls:
                        logger.warning(
                            "第二輪無文字（只有 function calls）→ 啟動第三輪純文字生成"
                        )
                        yield _sse_event("status", {"message": "正在生成分析報告..."})

                        round3_messages = list(round2_messages)
                        round3_messages.append({
                            "role": "assistant",
                            "content": "(已執行圖表標記操作)",
                        })
                        round3_messages.append({
                            "role": "user",
                            "content": (
                                "圖表標記已完成。現在請用文字詳細回答使用者的問題，"
                                "包含完整的分析結論、關鍵數據和交易建議。"
                                "不要再呼叫任何函式，只需要文字回答。"
                            ),
                        })

                        _r3_task = asyncio.create_task(
                            adapter.chat(
                                round3_messages, chart_state=request.chart_state,
                                force_text=True, system_prompt=_dynamic_prompt,
                            )
                        )
                        _active_tasks.append(_r3_task)
                        _hb_sec = 0
                        while not _r3_task.done():
                            await asyncio.sleep(3)
                            _hb_sec += 3
                            yield _sse_event("status", {
                                "message": f"正在生成分析報告... ({_hb_sec}秒)"
                            })
                        response3 = _r3_task.result()

                        if response3.usage and total_usage:
                            total_usage.prompt_tokens += response3.usage.prompt_tokens
                            total_usage.completion_tokens += response3.usage.completion_tokens
                            total_usage.total_tokens += response3.usage.total_tokens
                        elif response3.usage:
                            total_usage = response3.usage

                        _r2_text = response3.message or ""
                        logger.info(f"第三輪結果: 文字長度={len(_r2_text)}")

                    # 串流文字回應（移除 KEY_INSIGHTS / PREDICTIONS + JSON 過濾）
                    if _r2_text.strip():
                        display_msg2 = strip_system_distill(strip_predictions(strip_key_insights(_r2_text)))

                        # ★ 方案 A：偵測並提取文字中誤輸出的 JSON function call
                        display_msg2, _rescued_fcs = _extract_json_function_calls(display_msg2)
                        if _rescued_fcs:
                            logger.info(f"從第二/三輪文字中搶救 {len(_rescued_fcs)} 個 JSON function call")
                            try:
                                _rescue_result = await execute_function_calls(
                                    _rescued_fcs, chart_state=request.chart_state,
                                )
                                _rescue_updates = _rescue_result.get("chart_updates")
                                if _rescue_updates:
                                    yield _sse_event("chart", {"chart_updates": _rescue_updates})
                                    _has_draw_calls = True
                            except Exception as _re:
                                logger.warning(f"搶救的 JSON function call 執行失敗: {_re}")

                        if len(final_text) < _MAX_FINAL_TEXT:
                            final_text += _r2_text[:_MAX_FINAL_TEXT - len(final_text)]

                        _draw_keywords = (
                            "畫出", "繪製", "標記了", "已標註", "已在圖表", "畫了", "繪出",
                            "標示出", "繪畫", "畫上", "標出了",
                        )
                        _text_claims_draw = any(kw in display_msg2 for kw in _draw_keywords)
                        if _text_claims_draw and not _has_draw_calls:
                            logger.warning("LLM 文字聲稱繪圖但未產生 annotate_chart/draw_pattern function call")
                            display_msg2 += (
                                "\n\n⚠️ **注意**：AI 描述了繪圖操作，但未成功產生繪圖指令。"
                                "如果圖表上沒有看到標記，請重新描述您希望畫的內容，"
                                "例如：「請在圖表上畫出趨勢線」。"
                            )

                        _all_text = (response.message or "") + _r2_text
                        _r1_indicator_ids = {
                            fc.get("arguments", {}).get("indicator_id", "")
                            for fc in (response.function_calls or [])
                            if fc.get("name") == "manage_indicator"
                        }
                        _all_added_ids = _r1_indicator_ids | _r2_indicator_ids
                        _auto_indicators = _detect_mentioned_indicators(_all_text, _all_added_ids)
                        if _auto_indicators:
                            _auto_names = [ind["indicator_id"] for ind in _auto_indicators]
                            logger.info(f"自動補加指標: {_auto_names}")
                            yield _sse_event("chart", {"chart_updates": {"indicator_actions": _auto_indicators}})
                            yield _sse_event("status", {"message": f"已自動添加 {len(_auto_indicators)} 個分析指標"})

                        for chunk in _split_text_for_streaming(display_msg2):
                            yield _sse_event("token", {"content": chunk})
                            await asyncio.sleep(0.02)
                    else:
                        logger.error("所有輪次都未產生文字回應")
                        yield _sse_event("token", {
                            "content": "⚠️ AI 分析完成但未能產生文字報告，請嘗試重新提問。"
                        })

                except Exception as e:
                    logger.error(f"Function call 執行或二輪回應失敗: {e}")
                    yield _sse_event("error", {"error": f"指令執行失敗: {str(e)}"})

            else:
                # 沒有 function calls，直接串流文字（移除 KEY_INSIGHTS / PREDICTIONS）
                text = response.message
                if text:
                    if len(final_text) < _MAX_FINAL_TEXT:
                        final_text += text[:_MAX_FINAL_TEXT - len(final_text)]
                    display_text = strip_system_distill(strip_predictions(strip_key_insights(text)))

                    # ★ 方案 A：偵測並提取文字中誤輸出的 JSON function call
                    display_text, _rescued_fcs = _extract_json_function_calls(display_text)
                    if _rescued_fcs:
                        logger.info(f"從純文字回覆中搶救 {len(_rescued_fcs)} 個 JSON function call")
                        try:
                            _rescue_result = await execute_function_calls(
                                _rescued_fcs, chart_state=request.chart_state,
                            )
                            _rescue_updates = _rescue_result.get("chart_updates")
                            if _rescue_updates:
                                yield _sse_event("chart", {"chart_updates": _rescue_updates})
                        except Exception as _re:
                            logger.warning(f"搶救的 JSON function call 執行失敗: {_re}")

                    # 即使沒有 function call，也自動偵測文字中提到的指標
                    _auto_indicators = _detect_mentioned_indicators(text, set())
                    if _auto_indicators:
                        _auto_names = [ind["indicator_id"] for ind in _auto_indicators]
                        logger.info(f"(純文字回覆) 自動補加指標: {_auto_names}")
                        yield _sse_event("chart", {"chart_updates": {"indicator_actions": _auto_indicators}})

                    for chunk in _split_text_for_streaming(display_text):
                        yield _sse_event("token", {"content": chunk})
                        await asyncio.sleep(0.02)

            # 4. token 用量 + 持久化記錄
            if total_usage:
                usage_dict = total_usage.to_dict()
                yield _sse_event("usage", {"usage": usage_dict})

                # ★ 非同步寫入 SQLite（fire-and-forget，不阻塞串流）
                if api_key:
                    usage_tracker.record_usage(
                        api_key=api_key,
                        provider=total_usage.provider or provider,
                        model=total_usage.model or model_name or "unknown",
                        prompt_tokens=total_usage.prompt_tokens,
                        completion_tokens=total_usage.completion_tokens,
                        total_tokens=total_usage.total_tokens,
                        estimated_cost_usd=usage_dict.get("estimated_cost_usd", 0.0),
                        conversation_id=conversation_id,
                        request_type="chat_stream",
                    )

        except asyncio.TimeoutError:
            logger.error("Streaming chat 超時")
            yield _sse_event("error", {"error": "LLM 回應超時，請稍後再試"})
        except asyncio.CancelledError:
            logger.warning("Streaming chat 被取消（客戶端斷線）")
            return
        except Exception as e:
            logger.error(f"Streaming chat 錯誤: {e}")
            yield _sse_event("error", {"error": str(e)})
        finally:
            for t in _active_tasks:
                if not t.done():
                    t.cancel()
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass

        # 5. 保存助手回應到歷史 + 快取 + 知識碎片（全部包在 try 內防止阻斷 done）
        try:
            if final_text.strip():
                insights = parse_key_insights(final_text)
                distill_fragments = parse_system_distill(final_text)
                clean_text = strip_system_distill(strip_predictions(strip_key_insights(final_text)))

                usage_dict_for_save = total_usage.to_dict() if total_usage else None
                chat_history.save_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=clean_text,
                    token_usage=usage_dict_for_save,
                )

                analysis_cache.store(
                    question=request.message,
                    answer=clean_text,
                    chart_state=request.chart_state,
                )

                semantic_cache.store(
                    question=request.message,
                    answer=clean_text,
                    chart_state=request.chart_state,
                )

                frag_symbol = extract_symbol_from_text(request.message) or chart_symbol_for_save or ""
                if insights:
                    stored = fragment_store.store_batch(
                        fragments=insights,
                        symbol=frag_symbol,
                        source_question=request.message,
                    )
                    if stored:
                        logger.info(f"自動提取 {stored} 筆知識碎片（{frag_symbol}）")

                # ★ SYSTEM_DISTILL 碎片也存入知識庫
                if distill_fragments:
                    stored_d = fragment_store.store_batch(
                        fragments=distill_fragments,
                        symbol=frag_symbol,
                        source_question=request.message,
                    )
                    if stored_d:
                        logger.info(f"自動提取 {stored_d} 筆蒸餾碎片（{frag_symbol}）")

                # ★ 提取並存儲預測
                predictions = parse_predictions(final_text)
                if predictions:
                    pred_symbol = extract_symbol_from_text(request.message) or chart_symbol_for_save or ""
                    pred_tf = (request.chart_state or {}).get("timeframe", "4h")
                    for pred in predictions:
                        try:
                            prediction_tracker.store(
                                symbol=pred_symbol,
                                timeframe=pred_tf,
                                prediction=pred,
                                source_question=request.message,
                            )
                        except Exception as pe:
                            logger.warning(f"儲存預測失敗: {pe}")
                    logger.info(f"自動提取 {len(predictions)} 筆預測（{pred_symbol}）")

                # ★ 順便驗證已到期的預測
                try:
                    val_result = validate_all_active()
                    if val_result.get("validated", 0) > 0:
                        logger.info(f"自動驗證 {val_result['validated']} 筆預測")
                except Exception as ve:
                    logger.warning(f"自動驗證預測失敗: {ve}")

        except Exception as save_err:
            logger.error(f"保存歷史/快取時發生錯誤（不影響回應）: {save_err}")

        # 6. 完成（★ 保證 done 事件一定發送，即使前面全部出錯）
        done_data: dict = {"conversation_id": conversation_id}
        try:
            if chat_history._conn:
                distill_info = knowledge_distiller.get_distill_status(chat_history._conn)
                if distill_info.get("should_distill"):
                    done_data["distill_hint"] = True
                    done_data["distill_days"] = distill_info.get("undistilled_days", 0)
        except Exception:
            pass
        yield _sse_event("done", done_data)

    return StreamingResponse(
        stream_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ─── 對話歷史端點 ──────────────────────────────

@router.get("/history")
async def list_conversations(limit: int = 20, offset: int = 0):
    """列出最近的對話記錄"""
    conversations = chat_history.list_conversations(limit=limit, offset=offset)
    return {"status": "ok", "conversations": conversations, "total": len(conversations)}


@router.get("/history/{conversation_id}")
async def get_conversation(conversation_id: str):
    """取得特定對話的完整訊息"""
    messages = chat_history.get_conversation_messages(conversation_id)
    if not messages:
        return {"status": "ok", "messages": [], "note": "對話不存在或已清除"}
    return {"status": "ok", "conversation_id": conversation_id, "messages": messages}


@router.delete("/history/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """刪除一個對話"""
    success = chat_history.delete_conversation(conversation_id)
    return {"status": "ok" if success else "not_found"}


# ─── 知識蒸餾端點 ──────────────────────────────

@router.get("/distill/status")
async def get_distill_status():
    """查詢蒸餾狀態（前端用來判斷是否顯示蒸餾按鈕）"""
    if not chat_history._conn:
        return {"status": "error", "message": "對話歷史未初始化"}
    status = knowledge_distiller.get_distill_status(chat_history._conn)
    return {"status": "ok", **status}


@router.post("/distill/preview")
async def preview_distill(request: ChatRequest):
    """
    預覽蒸餾結果（不實際執行）。

    呼叫 LLM 生成摘要，讓使用者確認後才存入。
    """
    provider, api_key, base_url, model_name = _resolve_api_key(request)
    if not api_key and provider not in ("ollama", "claude_subscription"):
        return {"status": "error", "message": "需要有效的 LLM session 才能執行蒸餾"}

    # 準備材料
    if not chat_history._conn:
        return {"status": "error", "message": "對話歷史未初始化"}
    material = knowledge_distiller.prepare_distill_material(chat_history._conn)
    if material["total_messages"] < 4:
        return {"status": "error", "message": "對話數量不足（至少需要 4 條以上訊息）"}

    try:
        adapter = create_adapter(provider=provider, api_key=api_key, model_name=model_name, base_url=base_url)
    except Exception as e:
        return {"status": "error", "message": f"無法連接 LLM: {str(e)}"}

    previews = []
    total_tokens_used = 0

    # 為每個幣種生成摘要
    for symbol, qa_pairs in material["groups"].items():
        if len(qa_pairs) < 2:
            continue

        prompt = knowledge_distiller.build_distill_prompt(symbol, qa_pairs)
        try:
            response = await adapter.chat(
                [{"role": "user", "content": prompt}],
            )
            summary = response.message or ""
            tokens = 0
            if response.usage:
                tokens = response.usage.total_tokens
                total_tokens_used += tokens

            # 找出時間範圍
            times = [qa["time"][:10] for qa in qa_pairs if qa.get("time")]
            period_start = min(times) if times else ""
            period_end = max(times) if times else ""

            previews.append({
                "symbol": symbol,
                "period_start": period_start,
                "period_end": period_end,
                "summary": summary,
                "source_count": len(qa_pairs),
                "original_chars": sum(len(qa["q"]) + len(qa["a"]) for qa in qa_pairs),
                "distilled_chars": len(summary),
                "tokens_used": tokens,
            })
        except Exception as e:
            logger.error(f"蒸餾 {symbol} 失敗: {e}")
            previews.append({
                "symbol": symbol,
                "error": str(e),
            })

    # 生成使用者風格分析
    all_qa = []
    for pairs in material["groups"].values():
        all_qa.extend(pairs)

    profile_preview = None
    if len(all_qa) >= 5:
        try:
            profile_prompt = knowledge_distiller.build_profile_prompt(all_qa)
            response = await adapter.chat(
                [{"role": "user", "content": profile_prompt}],
            )
            profile_preview = response.message or ""
            if response.usage:
                total_tokens_used += response.usage.total_tokens
        except Exception as e:
            logger.warning(f"使用者風格分析失敗: {e}")

    return {
        "status": "ok",
        "previews": previews,
        "profile_preview": profile_preview,
        "total_tokens_used": total_tokens_used,
        "total_messages": material["total_messages"],
        "total_chars": material["total_chars"],
    }


@router.post("/distill/confirm")
async def confirm_distill(request: ChatRequest):
    """
    確認並執行蒸餾。

    接收前端傳來的預覽結果（可能被使用者微調），正式存入知識庫。
    存入後，標記原始對話為「已蒸餾」。
    """
    # request.chart_state 裡夾帶蒸餾確認資料
    distill_data = request.chart_state or {}
    previews = distill_data.get("previews", [])
    profile = distill_data.get("profile", "")
    total_tokens = distill_data.get("total_tokens_used", 0)

    if not previews:
        return {"status": "error", "message": "沒有蒸餾內容"}

    saved_count = 0
    for p in previews:
        if p.get("error") or not p.get("summary"):
            continue

        kid = knowledge_distiller.save_distilled_knowledge(
            symbol=p["symbol"],
            period_start=p.get("period_start", ""),
            period_end=p.get("period_end", ""),
            summary=p["summary"],
            key_numbers=p.get("key_numbers", ""),
            source_count=p.get("source_count", 0),
            original_chars=p.get("original_chars", 0),
        )
        if kid > 0:
            saved_count += 1

    # 保存使用者風格
    if profile:
        knowledge_distiller.save_user_profile(profile)

    # 記錄蒸餾歷史
    knowledge_distiller.save_distill_history(
        distill_type="full",
        original_data=json.dumps({"previews_count": len(previews)}, ensure_ascii=False),
        result=json.dumps({"saved": saved_count}, ensure_ascii=False),
        tokens=total_tokens,
    )

    return {
        "status": "ok",
        "saved_knowledge": saved_count,
        "message": f"已成功蒸餾 {saved_count} 筆知識摘要",
    }


@router.get("/distill/knowledge")
async def get_distilled_knowledge():
    """取得所有蒸餾知識（供前端展示）"""
    knowledge = knowledge_distiller.get_all_knowledge()
    profile = knowledge_distiller.get_user_profile()
    return {
        "status": "ok",
        "knowledge": knowledge,
        "user_profile": profile,
        "total": len(knowledge),
    }


# ─── 知識碎片端點 ──────────────────────────────

@router.get("/fragments/stats")
async def get_fragment_stats():
    """取得知識碎片統計"""
    stats = fragment_store.get_stats()
    return {"status": "ok", **stats}


@router.get("/fragments")
async def list_fragments(
    symbol: Optional[str] = None,
    fragment_type: Optional[str] = None,
    sort_by: str = "hit_count",
    limit: int = 100,
    offset: int = 0,
):
    """列出所有知識碎片（供前端知識庫瀏覽器使用）"""
    data = fragment_store.list_all(
        symbol=symbol,
        fragment_type=fragment_type,
        sort_by=sort_by,
        limit=limit,
        offset=offset,
    )
    return {"status": "ok", **data}


@router.delete("/fragments/{fragment_id}")
async def delete_fragment(fragment_id: int):
    """刪除指定知識碎片"""
    ok = fragment_store.delete_by_id(fragment_id)
    return {"status": "ok" if ok else "not_found"}


@router.post("/fragments/note")
async def add_user_note(request: ChatRequest):
    """使用者手動添加學習筆記到知識庫。

    前端傳送 { message: "筆記內容", symbol: "BTC/USDT" }
    """
    note_text = (request.message or "").strip()
    if not note_text or len(note_text) < 5:
        return {"status": "error", "message": "筆記內容太短（至少 5 字）"}
    if len(note_text) > 2000:
        return {"status": "error", "message": "筆記內容太長（上限 2000 字）"}

    symbol = request.symbol or "GENERAL"

    fid = fragment_store.store_fragment(
        content=note_text,
        fragment_type="user_note",
        symbol=symbol,
        source_question=f"[使用者筆記] {note_text[:60]}",
        is_seed=True,
    )

    return {
        "status": "ok",
        "message": "筆記已儲存到知識庫",
        "fragment_id": fid,
    }


# ─── 預測追蹤端點 ──────────────────────────────

@router.get("/predictions/stats")
async def get_prediction_stats(symbol: Optional[str] = None):
    """取得預測績效統計。"""
    stats = prediction_tracker.get_stats(symbol=symbol)
    return {"status": "ok", **stats}


@router.get("/predictions/active")
async def get_active_predictions(symbol: Optional[str] = None):
    """取得尚未到期的預測。"""
    active = prediction_tracker.get_active(symbol)
    return {"status": "ok", "predictions": active, "total": len(active)}


@router.get("/predictions/history")
async def get_prediction_history(symbol: Optional[str] = None, limit: int = 30):
    """取得已驗證的預測歷史。"""
    history = prediction_tracker.get_validated(symbol=symbol, limit=limit)
    return {"status": "ok", "predictions": history, "total": len(history)}


@router.post("/predictions/validate")
async def trigger_validation():
    """手動觸發預測驗證。"""
    result = validate_all_active()
    return {"status": "ok", **result}


# ─── 工具函式 ──────────────────────────────────

def _split_text_for_streaming(text: str) -> list[str]:
    """
    將文字切割成適合串流的 chunks。

    策略：按標點/換行自然斷句，每 chunk 約 20-80 字元。
    這樣串流看起來像真人在打字。
    """
    if len(text) <= 30:
        return [text]

    chunks: list[str] = []
    current = ""

    for char in text:
        current += char
        # 在標點符號或換行處斷開
        if char in "。！？\n；：，、」）】" and len(current) >= 8:
            chunks.append(current)
            current = ""
        elif len(current) >= 60:
            # 太長了，強制斷開
            chunks.append(current)
            current = ""

    if current:
        chunks.append(current)

    return chunks
