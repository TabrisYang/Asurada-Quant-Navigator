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
from app.core.llm.executor import execute_function_calls, check_input_safety, ProgressTracker
from app.core.llm.function_defs import detect_intents, assemble_system_prompt
from app.core.security.key_manager import key_manager
from app.core.config.settings import settings  # v101: feature flags
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
from app.utils.timezone import taipei_now
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
# analysis: 趨勢 + 動量 + 波動 + 量能 + 風控（15 個）
_ANALYSIS_CORE_INDICATORS = [
    # 趨勢
    "adx", "supertrend", "ema", "psar",
    # 動量
    "rsi", "macd", "stochrsi",
    # 波動
    "bb", "atr", "donchian",
    # 量能
    "obv", "rel_vol", "vwap",
    # 背離
    "rsi_divergence",
    # 風控
    "kelly",
]
# deep analysis: 全維度覆蓋 + 先行訊號 + 情緒 + 結構（25 個）
_DEEP_ANALYSIS_INDICATORS = [
    # 趨勢
    "adx", "supertrend", "ema", "psar", "ichimoku", "market_structure", "mtf_mss",
    # 動量
    "rsi", "macd", "stochrsi", "roc", "leading_composite",
    # 波動
    "bb", "atr", "donchian", "keltner", "hv", "vol_squeeze",
    # 量能
    "obv", "rel_vol", "vwap", "cvd",
    # 背離
    "rsi_divergence", "macd_divergence",
    # 風控
    "kelly",
]

_MAX_AUTO_CALC = 30


def _mark_status(
    chart_state: dict | None,
    key: str,
    status: str,
    reason: str = "",
    **extra,
) -> None:
    """v123：把欄位注入狀態寫入 chart_state['data_status'][key]，讓 LLM 看見
    「曾嘗試但失敗 + 原因」而非完全省略。

    status 約定值：
      - "ok"                    成功注入
      - "partial"               部分欄位成功（如 basket 部分成員、衍生品部分 API）
      - "skipped"               條件不符（如台股不抓 social_sentiment、ranging 才跑 subtype）
      - "failed"                exception 或 API 全失敗
      - "insufficient_samples"  樣本不足（如 historical_insights n < 200）
      - "insufficient_data"     資料量不足（如 df < 100 / df < 60）
      - "no_model"              ML 模型不存在
      - "guard_failed"          多層守衛任一失敗（如 RL 6 層守衛）
      - "stale"                 用 stale cache fallback

    調用方式：
      _mark_status(chart_state, "crossStockSignals", "partial", "basket_size=2 < 3")
      _mark_status(chart_state, "external_signals", "ok")
    """
    if chart_state is None:
        return
    ds = chart_state.get("data_status")
    if not isinstance(ds, dict):
        ds = {}
        chart_state["data_status"] = ds
    payload = {"status": status}
    if reason:
        payload["reason"] = reason
    if extra:
        payload.update(extra)
    ds[key] = payload


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

    # ── 核心修復：分析意圖時「強制重算」所有核心指標 ──
    # 前端傳來的 indicatorValues 可能是舊幣種/舊時間框架的殘留值，
    # 不能信任。只要是分析意圖，一律由後端從最新數據重新計算。
    force_recalc = is_deep or is_analysis

    if is_deep:
        for ind_id in _DEEP_ANALYSIS_INDICATORS:
            if ind_id not in need_calc:
                need_calc.append(ind_id)
    elif is_analysis:
        for ind_id in _ANALYSIS_CORE_INDICATORS:
            if ind_id not in need_calc:
                need_calc.append(ind_id)

    has_value_intent = is_analysis or is_deep or any(kw in user_msg for kw in _VALUE_KEYWORDS)
    if has_value_intent:
        for keyword, (ind_id, _dm) in _INDICATOR_TEXT_MAP.items():
            if keyword in msg_lower and ind_id not in need_calc:
                if force_recalc or ind_id not in existing_keys:
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

        # 快取 DataFrame 供後續 ML 預測使用，避免重複載入
        chart_state["_cached_df"] = df
        chart_state["_cached_df_key"] = f"{symbol}_{timeframe}"

        # 注入數據可用性資訊，讓 LLM 知道本地實際有多少數據
        if len(df) > 0 and "timestamp" in df.columns:
            ts = df["timestamp"]
            chart_state["data_availability"] = {
                "total_bars": len(df),
                "timeframe": timeframe,
                "start_date": str(ts.iloc[0]),
                "end_date": str(ts.iloc[-1]),
            }

        # 注入 STL 時序分解：給 LLM 看到趨勢/季節/殘差立體結構
        try:
            from app.core.timeseries_decomposition import decompose_price, is_available
            if not is_available():
                _mark_status(chart_state, "decomposition", "skipped", "stl_module_unavailable")
            elif len(df) < 60:
                _mark_status(chart_state, "decomposition", "insufficient_data",
                             f"df_len={len(df)} < 60")
            else:
                # 日線用 period=20（約一個月），其他用 14
                _stl_period = 20 if timeframe == "1d" else 14
                decomp = decompose_price(df, period=_stl_period)
                if "error" not in decomp:
                    chart_state["decomposition"] = decomp
                    _mark_status(chart_state, "decomposition", "ok")
                else:
                    _mark_status(chart_state, "decomposition", "failed",
                                 str(decomp.get("error")))
        except Exception as _stl_err:
            logger.info(f"STL 分解失敗（不影響主流程）: {_stl_err}")
            _mark_status(chart_state, "decomposition", "failed", str(_stl_err))

        # 注入當前 regime + 低信心警告（Phase 1A + Phase 2.3）+ Level 1 自動記錄
        try:
            from app.core.regime_filter import summarize_current_regime
            regime_info = summarize_current_regime(df)
            chart_state["currentRegime"] = regime_info
            _mark_status(chart_state, "currentRegime", "ok",
                         confidence=regime_info.get("confidence"))
            if regime_info.get("confidence", 0) < 0.5:
                chart_state["regimeWarning"] = {
                    "level": "high",
                    "message": f"目前市場處於 {regime_info.get('regime', 'unknown')} regime（信心 {regime_info.get('confidence', 0)*100:.0f}%），所有策略建議僅供參考",
                    "auto_position_multiplier": 0.5,
                }
                # Level 1：累積 unknown regime 樣本給未來 audit / classifier 升級用
                try:
                    from app.core.unknown_regime_logger import log_unknown_regime
                    log_unknown_regime(symbol, timeframe, regime_info)
                except Exception:
                    pass  # 記錄失敗不影響主流程
        except Exception as _rg_err:
            logger.info(f"regime 分類失敗（不影響主流程）: {_rg_err}")
            _mark_status(chart_state, "currentRegime", "failed", str(_rg_err))

        # v104 Q1 + v119.2：注入外部訊號快照（funding / OI / 多空比 / Fear&Greed / FRED 總體）
        # v119.2 修法：條件放寬為「永遠注入」（即使部分欄位空），讓 LLM 知道「系統有嘗試
        # 抓資料」。配合 v119.1 的 stale cache fallback，現在「無衍生品快照」應該很少見。
        try:
            from app.core.external_signals import get_signals_snapshot, format_signals_summary
            _signals = get_signals_snapshot(symbol)
            if _signals:
                chart_state["external_signals"] = _signals
                chart_state["external_signals_summary"] = format_signals_summary(_signals)
                _deriv_n = len(_signals.get('derivatives') or {})
                _sent_n = len(_signals.get('sentiment') or {})
                _macro_n = len(_signals.get('macro') or {})
                _stale_tag = " STALE" if _signals.get("stale") else ""
                logger.info(
                    f"[external_signals]{_stale_tag} {symbol} "
                    f"deriv={_deriv_n} sentiment={_sent_n} macro={_macro_n}"
                )
                # v123：部分欄位空 → partial；全空 → failed_all；stale → stale；其餘 ok
                if _signals.get("stale"):
                    _es_status = "stale"
                elif _deriv_n == 0 and _sent_n == 0 and _macro_n == 0:
                    _es_status = "failed_all"
                elif _deriv_n == 0 or _sent_n == 0:
                    _es_status = "partial"
                else:
                    _es_status = "ok"
                _mark_status(chart_state, "external_signals", _es_status,
                             f"derivatives={_deriv_n} sentiment={_sent_n} macro={_macro_n}")
            else:
                _mark_status(chart_state, "external_signals", "failed_all",
                             "get_signals_snapshot returned empty")
        except Exception as _ex_err:
            logger.info(f"external_signals 失敗（不影響主流程）: {_ex_err}")
            _mark_status(chart_state, "external_signals", "failed", str(_ex_err))

        # v103 6A + v105.4：注入未來 72h 高影響事件 + calendar_meta（過期警示）
        try:
            from app.core.event_injector import get_upcoming_events, get_calendar_meta
            from app.utils.symbol import is_tw_stock as _is_tw
            scope = "equities" if _is_tw(symbol) else "crypto"
            events = get_upcoming_events(within_hours=72, min_severity="medium", scope_match=scope)
            cal_meta = get_calendar_meta()
            if events:
                chart_state["upcoming_events"] = events
                logger.info(
                    f"[event_injector] 注入 {len(events)} 筆 72h 內事件 (scope={scope}) "
                    f"calendar age={cal_meta.get('age_days')} 天 stale={cal_meta.get('is_stale')}"
                )
                _mark_status(chart_state, "upcoming_events", "ok",
                             f"n={len(events)} scope={scope}")
            else:
                _mark_status(chart_state, "upcoming_events", "skipped",
                             f"no_high_severity_events_in_72h (scope={scope})")
            # 即使沒事件也注入 calendar_meta（給 prompt 判斷是否過期）
            chart_state["calendar_meta"] = cal_meta
            if cal_meta.get("is_stale"):
                _mark_status(chart_state, "calendar_meta", "stale",
                             f"age_days={cal_meta.get('age_days')}")
            else:
                _mark_status(chart_state, "calendar_meta", "ok")
        except Exception as _ev_err:
            logger.info(f"event_injector 失敗（不影響主流程）: {_ev_err}")
            _mark_status(chart_state, "upcoming_events", "failed", str(_ev_err))
            _mark_status(chart_state, "calendar_meta", "failed", str(_ev_err))

        # 注入跨股票群體訊號（台股 + 加密）：給 LLM 看到所屬集合 + 龍頭 + breadth
        # v123: 即使 basket < 3 也注入 partial（含 note），讓 LLM 在 #1 段顯示「資料不可得」而非省略段落
        try:
            from app.utils.symbol import is_tw_stock
            if is_tw_stock(symbol):
                from app.core.cross_stock_signals import compute_signals
                cs_signals = compute_signals(symbol, timeframe)
            else:
                from app.core.cross_stock_signals import compute_signals_crypto
                cs_signals = compute_signals_crypto(symbol, timeframe)
            if cs_signals:
                # 即使只有 note（partial）也注入，讓 LLM 看到 reason
                chart_state["crossStockSignals"] = cs_signals
                _basket_size = cs_signals.get("basket_size") or 0
                _has_full = bool(
                    cs_signals.get("sector")
                    or _basket_size >= 3
                    or cs_signals.get("market_regime")
                )
                if _has_full:
                    _mark_status(chart_state, "crossStockSignals", "ok",
                                 f"basket_size={_basket_size}")
                else:
                    _mark_status(chart_state, "crossStockSignals", "partial",
                                 cs_signals.get("note") or f"basket_size={_basket_size} < 3")
                logger.info(
                    f"[crossStockSignals] {symbol}: basket_size={_basket_size} "
                    f"status={'ok' if _has_full else 'partial'}"
                )
            else:
                _mark_status(chart_state, "crossStockSignals", "failed",
                             "compute_signals returned None")
        except Exception as _cs_err:
            logger.info(f"跨股票訊號計算失敗（不影響主流程）: {_cs_err}")
            _mark_status(chart_state, "crossStockSignals", "failed", str(_cs_err))

        # v106 A3：注入社群情緒（Reddit + CryptoPanic RSS，graceful fallback）
        try:
            from app.core.social_sentiment import get_social_sentiment
            from app.utils.symbol import is_tw_stock as _is_tw_a3
            # 只對加密 symbol 抓社群情緒（台股的 Reddit 沒參考價值）
            if _is_tw_a3(symbol):
                _mark_status(chart_state, "social_sentiment", "skipped",
                             "tw_stock_not_supported")
            else:
                _sent = get_social_sentiment(symbol)
                if _sent and not _sent.get("stale_warning"):
                    chart_state["social_sentiment"] = _sent
                    _mark_status(chart_state, "social_sentiment", "ok",
                                 f"overall={_sent.get('overall_label')}")
                    logger.info(
                        f"[social_sentiment] {symbol}: "
                        f"overall={_sent.get('overall_label')} "
                        f"({_sent.get('overall_sentiment', 0):+.2f})"
                    )
                elif _sent and _sent.get("stale_warning"):
                    _mark_status(chart_state, "social_sentiment", "stale",
                                 str(_sent.get("stale_warning")))
                else:
                    _mark_status(chart_state, "social_sentiment", "failed",
                                 "no_sentiment_data")
        except Exception as _sent_err:
            logger.info(f"social_sentiment 注入失敗（不影響主流程）: {_sent_err}")
            _mark_status(chart_state, "social_sentiment", "failed", str(_sent_err))

        # v107.2：只注入 portfolio_summary（客觀組合風控），不注入 user_positions（避免 LLM 偏向使用者立場）
        try:
            from app.core.position_tracker import position_tracker
            _portfolio = position_tracker.get_summary()
            if _portfolio.get("total_positions", 0) > 0:
                chart_state["portfolio_summary"] = _portfolio
                _mark_status(chart_state, "portfolio_summary", "ok",
                             f"n_positions={_portfolio.get('total_positions')}")
                logger.info(
                    f"[portfolio_summary] {symbol}: "
                    f"{_portfolio.get('total_positions')} positions, "
                    f"${_portfolio.get('total_exposure_usd', 0):.0f}"
                )
            else:
                _mark_status(chart_state, "portfolio_summary", "skipped",
                             "no_open_positions")
        except Exception as _pos_err:
            logger.info(f"portfolio_summary 注入失敗（不影響主流程）: {_pos_err}")
            _mark_status(chart_state, "portfolio_summary", "failed", str(_pos_err))

        # v104 Fix B：ranging / unknown 子類型分類
        # 必須在 currentRegime + crossStockSignals + external_signals 注入完才跑
        try:
            _ri = chart_state.get("currentRegime") or {}
            _regime_label = _ri.get("regime")
            if _regime_label in ("ranging", "unknown"):
                from app.core.regime_subtype import classify_ranging_subtype
                _sub = classify_ranging_subtype(df, _ri, chart_state, symbol=symbol)
                if _sub and _sub.get("subtype"):
                    chart_state["regime_subtype"] = _sub
                    _mark_status(chart_state, "regime_subtype", "ok",
                                 f"subtype={_sub['subtype']}")
                    logger.info(
                        f"[regime_subtype] {symbol} {timeframe}: "
                        f"{_sub['subtype']} (conf={_sub.get('confidence',0):.2f}) — {_sub.get('reason','')}"
                    )
                else:
                    _mark_status(chart_state, "regime_subtype", "failed",
                                 "classify returned no subtype")
            else:
                _mark_status(chart_state, "regime_subtype", "skipped",
                             f"regime={_regime_label}, only_ranging_unknown_applicable")
        except Exception as _sub_err:
            logger.info(f"regime_subtype 分類失敗（不影響主流程）: {_sub_err}")
            _mark_status(chart_state, "regime_subtype", "failed", str(_sub_err))

        # v108 Phase 2：注入 donchian_position_pct + bilateral_plan
        # 解決 LLM 編造客觀數值問題：區間位置 % 與雙向計劃進場分批價必須由後端算好
        try:
            # 1. donchian_position_pct（標準 Donchian-20，範圍 0-100）
            if len(df) >= 20:
                _don_u = float(df["high"].iloc[-20:].max())
                _don_l = float(df["low"].iloc[-20:].min())
                _close_now = float(df["close"].iloc[-1])
                _don_range = _don_u - _don_l
                if _don_range > 0:
                    _pct = ((_close_now - _don_l) / _don_range) * 100.0
                    _pct = max(0.0, min(100.0, _pct))
                    chart_state["donchian_position_pct"] = round(_pct, 1)
                    chart_state["donchian_upper"] = round(_don_u, 6)
                    chart_state["donchian_lower"] = round(_don_l, 6)
                    _mark_status(chart_state, "donchian_position", "ok")
                else:
                    _mark_status(chart_state, "donchian_position", "skipped",
                                 "zero_range")
            else:
                _mark_status(chart_state, "donchian_position", "insufficient_data",
                             f"df_len={len(df)} < 20")

            # 2. bilateral_plan — 僅 ranging/unknown 場景才算（其他 regime 不需雙向計劃）
            _ri_bp = chart_state.get("currentRegime") or {}
            _regime_bp = _ri_bp.get("regime", "")
            if _regime_bp in ("ranging", "unknown"):
                from app.core.laddered_entries import compute_laddered_entries
                _conf_bp = float(_ri_bp.get("confidence", 0.0))
                if _conf_bp >= 0.7:
                    _conf_label_bp = "high"
                elif _conf_bp >= 0.4:
                    _conf_label_bp = "medium"
                else:
                    _conf_label_bp = "low"
                _bp = compute_laddered_entries(
                    df=df, direction="both", regime=_regime_bp,
                    regime_confidence=_conf_bp, n_tranches=3,
                    timeframe_str=timeframe, confidence_label=_conf_label_bp,
                )
                if _bp:
                    chart_state["bilateral_plan"] = _bp
                    _mark_status(chart_state, "bilateral_plan", "ok",
                                 f"regime={_regime_bp} conf={_conf_bp:.2f}")
                    logger.info(
                        f"[bilateral_plan] {symbol} {timeframe}: "
                        f"enabled={_bp.get('enabled')} "
                        f"long={len(_bp.get('long_entries') or [])} "
                        f"short={len(_bp.get('short_entries') or [])}"
                    )
                else:
                    _mark_status(chart_state, "bilateral_plan", "failed",
                                 "compute_laddered_entries returned empty")
            else:
                _mark_status(chart_state, "bilateral_plan", "skipped",
                             f"regime={_regime_bp}, only_ranging_unknown_applicable")
        except Exception as _bp_err:
            logger.info(f"donchian_position / bilateral_plan 注入失敗（不影響主流程）: {_bp_err}")
            _mark_status(chart_state, "bilateral_plan", "failed", str(_bp_err))

        # v106 C3：注入歷史洞察庫（從 200 筆驗證樣本萃取 patterns）
        try:
            from app.core.strategy_insights import get_insights_for
            _ri_insight = chart_state.get("currentRegime") or {}
            _regime_label = _ri_insight.get("regime", "unknown")
            _insight = get_insights_for(symbol, timeframe, _regime_label)
            if _insight:
                chart_state["historical_insights"] = _insight
                _mark_status(chart_state, "historical_insights", "ok",
                             f"n={_insight.get('n_samples')} regime={_regime_label}")
                logger.info(
                    f"[strategy_insights] {symbol} {timeframe} {_regime_label}: "
                    f"n={_insight['n_samples']} winrate={_insight['winrate']*100:.1f}%"
                )
            else:
                _mark_status(chart_state, "historical_insights", "insufficient_samples",
                             f"regime={_regime_label}, n < 200_required")
        except Exception as _ins_err:
            logger.info(f"strategy_insights 注入失敗（不影響主流程）: {_ins_err}")
            _mark_status(chart_state, "historical_insights", "failed", str(_ins_err))

        # 強制重算模式：清除前端傳來的舊值，完全以後端計算為準
        if force_recalc:
            auto_values = {}
        else:
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
            logger.info(f"自動計算指標值 (intent={_intents}, force={force_recalc}): {need_calc}")
    except Exception as e:
        logger.warning(f"自動計算指標值失敗: {e}")

    # 注入自動掃描預警（含波動機率和依據）
    try:
        from app.core.auto_scanner import auto_scanner
        import json as _json

        active_alerts = auto_scanner.get_active_alerts(symbol)
        if active_alerts:
            enriched = []
            for a in active_alerts[:5]:
                item = {
                    "symbol": a["symbol"],
                    "alert_type": a["alert_type"],
                    "direction": a["direction"],
                    "confidence": a["confidence"],
                    "move_probability": a.get("move_probability"),
                    "created_at": a["created_at"],
                    "expires_at": a["expires_at"],
                }
                # 從 trigger_conditions 解析 evidence_summary
                tc = a.get("trigger_conditions")
                if tc and isinstance(tc, str):
                    try:
                        parsed = _json.loads(tc)
                        prob = parsed.get("probability")
                        if prob:
                            item["evidence_summary"] = prob.get("evidence_summary")
                    except (ValueError, AttributeError):
                        pass
                enriched.append(item)
            chart_state["active_alerts"] = enriched
    except Exception:
        pass

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
    # ML 模組總開關：未啟用則直接返回，跳過所有 ML 邏輯
    from app.core.ml._settings import ML_ENABLED
    if not ML_ENABLED:
        return chart_state or {}

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
            logger.info(f"ML 未啟用: {reason}")
            _mark_status(chart_state, "mlPrediction", "no_model", reason)
            return chart_state

        # 優先使用 _auto_calc_indicator_values 已載入的 DataFrame
        cached_key = chart_state.get("_cached_df_key")
        if cached_key == f"{symbol}_{timeframe}" and "_cached_df" in chart_state:
            df = chart_state["_cached_df"]
        else:
            from app.data.fetchers.crypto_engine import crypto_engine
            df = crypto_engine.load_local_data(symbol, timeframe)
        if df.empty or len(df) < 100:
            _mark_status(chart_state, "mlPrediction", "insufficient_data",
                         f"df_len={len(df)} < 100")
            return chart_state

        result = model_manager.predict_consensus(df, symbol, timeframe)
        if result.get("status") == "success":
            chart_state["mlPrediction"] = result["prediction"]
            mode = result["prediction"].get("consensus_mode", "best")
            n_models = result["prediction"].get("models_considered", 1)
            _mark_status(chart_state, "mlPrediction", "ok",
                         f"consensus={mode} n_models={n_models}")
            logger.info(
                f"ML 預測注入: {symbol} {timeframe} → "
                f"{result['prediction']['direction']} "
                f"(prob={result['prediction']['probability']:.1%}, "
                f"consensus={mode}, models={n_models})"
            )
        else:
            _mark_status(chart_state, "mlPrediction", "failed",
                         str(result.get("status") or "unknown"))
    except Exception as e:
        logger.info(f"ML 預測注入失敗（不影響分析）: {e}")
        _mark_status(chart_state, "mlPrediction", "failed", str(e))

    return chart_state


# v124：機率三聯 (baseline / TA-conditional / track-record) 注入用 timeframe-aware 預設參數
# 取代「從預測卡讀」的不可行設計（注入時 LLM 尚未產出預測卡）
_TF_BASELINE_PARAMS: dict[str, tuple[int, float]] = {
    # timeframe → (forward_bars, target_pct)
    "1m": (12, 1.0), "5m": (12, 1.5), "15m": (8, 2.0), "30m": (8, 2.0),
    "1h": (6, 2.0), "2h": (6, 2.5), "4h": (6, 3.0),
    "6h": (5, 4.0), "8h": (5, 5.0), "12h": (5, 5.0),
    "1d": (5, 5.0), "1w": (4, 10.0), "1mo": (3, 15.0),
}


def _build_triplet_warnings(triplet: dict) -> list[str]:
    """v124：依機率三聯資料產生顯著性警示列。

    後端產生而非 LLM 算，因為比 CI 邊界 LLM 容易錯（會用「直覺」判斷重不重疊）。
    """
    lines: list[str] = []
    base = triplet.get("baseline_unconditional") or {}
    tr = triplet.get("track_record") or {}

    if tr.get("status") == "ok":
        cw = tr.get("ci_width_pp", 0)
        if cw > 30:
            lines.append(f"⚠️ track record CI 寬度 {cw}pp，點估值僅供參考")
        n_dec = tr.get("n_decided", 0)
        if n_dec < 10:
            lines.append(f"⚠️ track record 已決出樣本不足 (n_decided={n_dec} < 10)，CI 寬，僅作參考")
        if base.get("status") == "ok":
            tr_raw = tr.get("win_rate_raw_pct")
            base_p = base.get("prob_pct")
            if (tr_raw is not None and base_p is not None
                    and tr_raw < base_p and n_dec >= 10):
                lines.append(
                    f"⚠️ 此 symbol 你的歷史命中率 {tr_raw}% 低於純價格基線 {base_p}% "
                    f"(n={n_dec}) — 強烈建議降倉 / 觀望 / 改用反向視角"
                )

    if base.get("status") == "ok" and base.get("n", 0) < 200:
        lines.append(f"⚠️ baseline 樣本不足 (n={base.get('n')} < 200)，baseline 本身也不準")

    return lines


def _inject_probability_triplet(
    chart_state: dict,
    df,
    symbol: str,
    timeframe: str,
) -> None:
    """v124：把「純價格 baseline / TA 條件化機率 / track record」三聯注入 recent_accuracy。

    使用者疑問「系統是不是只看價格」的對應措施：把三類數字並列顯示，讓 LLM
    在報告中明確標示「TA 條件化機率」≠「純價格 baseline」≠「歷史 track record」。

    掛在 recent_accuracy.probability_triplet 下（不增 top-level 欄位，維持
    CHART_STATE_SCHEMA.md 26 欄位上限）。
    """
    from app.core.config.settings import settings as _settings
    if not getattr(_settings, "probability_triplet_enabled", True):
        _mark_status(chart_state, "probability_triplet", "skipped", "flag_disabled")
        return

    if "recent_accuracy" not in chart_state or not isinstance(
        chart_state["recent_accuracy"], dict
    ):
        _mark_status(chart_state, "probability_triplet", "skipped",
                     "recent_accuracy_missing")
        return

    triplet: dict = {}

    # ── Track A：純價格 baseline ──
    fb, tp = _TF_BASELINE_PARAMS.get(timeframe, (6, 3.0))
    try:
        from app.core.probability_baseline import calc_unconditional_baseline
        baseline_a = calc_unconditional_baseline(df, fb, tp, direction="up")
        baseline_a["params_source"] = "timeframe_default"
        triplet["baseline_unconditional"] = baseline_a
    except Exception as _err:
        triplet["baseline_unconditional"] = {
            "status": "failed", "reason": str(_err),
            "params": {"forward_bars": fb, "target_pct": tp, "direction": "up"},
        }

    # ── Track B：TA 條件化機率（無 CI，依使用者決策） ──
    sub = (chart_state.get("regime_subtype") or {}).get("metrics") or {}
    bias_score = sub.get("bias_score")
    bias_reasons = sub.get("bias_reasons") or []
    if bias_score is not None:
        # P(多) = clip(0.5 + bias_score × 0.5, 0.10, 0.90)，與 function_defs.py 規則一致
        p_long_pct = round(max(10.0, min(90.0, (0.5 + bias_score * 0.5) * 100)), 1)
        triplet["ta_conditional"] = {
            "status": "ok",
            "prob_pct": p_long_pct,
            "bias_score": bias_score,
            "bias_reasons": bias_reasons[:3],
            "source": "bias_score_9dim",
        }
    else:
        # fallback：用 currentRegime.bullish_score（非 ranging regime 時）
        cr = chart_state.get("currentRegime") or {}
        bs = cr.get("bullish_score")
        if bs is not None:
            triplet["ta_conditional"] = {
                "status": "ok",
                "prob_pct": round(float(bs) * 100, 1),
                "bias_score": None,
                "bias_reasons": [],
                "source": "currentRegime_bullish_score",
            }
        else:
            triplet["ta_conditional"] = {
                "status": "skipped",
                "reason": "no_bias_score_and_no_bullish_score",
            }

    # ── Track C：該 symbol track record ──
    try:
        tr_data = prediction_tracker.get_winrate_with_ci(symbol=symbol, days=90)
        triplet["track_record"] = tr_data
    except Exception as _err:
        triplet["track_record"] = {"status": "failed", "reason": str(_err)}

    # ── 顯著性警示 ──
    triplet["significance"] = {"warning_lines": _build_triplet_warnings(triplet)}

    chart_state["recent_accuracy"]["probability_triplet"] = triplet
    _mark_status(chart_state, "probability_triplet", "ok",
                 f"baseline={triplet['baseline_unconditional'].get('status')} "
                 f"ta={triplet['ta_conditional'].get('status')} "
                 f"track={triplet['track_record'].get('status')}")


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

        # 自動調整規則（最高優先級，插入最前面）
        try:
            from app.core.auto_adjuster import generate_hard_constraints
            _hard_constraints = generate_hard_constraints(symbol=chart_symbol)
            if _hard_constraints:
                context_parts.append(_hard_constraints)
        except Exception:
            pass

        try:
            strategies_prompt = get_enabled_strategies_prompt()
            if strategies_prompt:
                context_parts.append(f"【使用者自訂分析方法論】\n{strategies_prompt}")
        except Exception as e:
            logger.warning(f"策略 prompt 載入失敗: {e}")

        try:
            distilled_context = knowledge_distiller.get_context_for_symbol(chart_symbol)
            if distilled_context:
                context_parts.append(f"【歷史分析記憶】\n{distilled_context}")
        except Exception as e:
            logger.warning(f"蒸餾知識載入失敗: {e}")

        if _intents & {"analysis", "backtest", "quant_research"}:
            try:
                _current_regime = (request.chart_state or {}).get("regime")
                if not _current_regime:
                    _recent_active = prediction_tracker.get_active(chart_symbol)
                    if _recent_active:
                        _current_regime = _recent_active[0].get("regime")
                _feedback_prompt = generate_feedback_prompt(
                    symbol=chart_symbol, current_regime=_current_regime,
                )
                if _feedback_prompt:
                    context_parts.append(f"【預測績效反饋】\n{_feedback_prompt}")

                # ★ v100：把命中率也以結構化形式注入 chart_state.recent_accuracy
                # 讓 LLM 不只看純文字反饋，還能精確引用數字校準自己的信心
                try:
                    _stats_30d = prediction_tracker.get_stats(symbol=chart_symbol, days=30)
                    _stats_90d = prediction_tracker.get_stats(symbol=chart_symbol, days=90)
                    _n30 = _stats_30d.get("total", 0)
                    _n90 = _stats_90d.get("total", 0)
                    if _n30 < 3 and _n90 < 3:
                        _mark_status(request.chart_state, "recent_accuracy",
                                     "insufficient_predictions",
                                     f"n_30d={_n30}, n_90d={_n90}, min=3")
                    if _stats_30d.get("total", 0) >= 3 or _stats_90d.get("total", 0) >= 3:
                        _bayes = _stats_30d.get("bayesian", {})
                        # v112 fix：3 個連環 bug
                        # (1) get_stats 回傳 key 是 "indicator_performance"（不是 "indicator_stats"），導致 _ind_stats 永遠空
                        # (2) 原本 `for k, _ in _best[:3] if _[1].get(...)` 的 _[1] 在 dict 上會 KeyError（_ 是 value dict 不是 tuple）
                        # (3) 邏輯順序錯：先排序再 filter samples >= 3 → top 3 全是 samples=1 的 → 過濾後空
                        # 正確順序：先 filter samples >= 3 再排序，避免 best/worst 被「樣本不足但碰巧勝率高/低」的 indicator 佔據
                        _ind_stats = _stats_30d.get("indicator_performance", {})
                        # 只考慮 samples >= 3 的 indicator（樣本不足的 win_rate 沒統計意義）
                        _filtered_inds = [(k, v) for k, v in _ind_stats.items() if v.get("samples", 0) >= 3]
                        _best = sorted(_filtered_inds, key=lambda x: x[1].get("win_rate", 0), reverse=True)
                        _worst = sorted(_filtered_inds, key=lambda x: x[1].get("win_rate", 0))
                        _best_inds = [k for k, _v in _best[:3]]
                        _worst_inds = [k for k, _v in _worst[:2]]
                        if request.chart_state is None:
                            request.chart_state = {}
                        request.chart_state["recent_accuracy"] = {
                            "symbol": chart_symbol,
                            "regime": _current_regime,
                            "win_rate_30d": _stats_30d.get("win_rate_weighted"),
                            "win_rate_90d": _stats_90d.get("win_rate_weighted"),
                            "n_30d": _stats_30d.get("total"),
                            "n_90d": _stats_90d.get("total"),
                            "bayesian_ci_95": _bayes.get("credible_interval_95"),
                            "calibration_brier": _stats_30d.get("calibration", {}).get("brier_score"),
                            "best_indicators": _best_inds,
                            "worst_indicators": _worst_inds,
                        }
                        _mark_status(request.chart_state, "recent_accuracy", "ok",
                                     f"n_30d={_n30} n_90d={_n90}")

                        # ★ v118：注入 regime_warning + direction_balance（修「看漲說漲」bias）
                        # Diagnose 發現：BULLISH regime 100% 看多但命中率僅 21.7%（賠錢領域），
                        # 系統有強烈順勢 bias 反而最差。注入這兩個欄位讓 LLM 強制權衡 contrarian。
                        try:
                            _regime_class_stats = prediction_tracker.get_regime_class_stats(
                                symbol=chart_symbol, days=90,
                            )
                            # regime_warning：當前 regime class 命中率 < 50% 且 n >= 10
                            if _current_regime:
                                _current_class = prediction_tracker.classify_regime(_current_regime)
                                _class_data = _regime_class_stats.get(_current_class)
                                if (
                                    _class_data
                                    and _class_data.get("samples", 0) >= 10
                                    and _class_data.get("win_rate", 0) < 50
                                ):
                                    _wr = _class_data["win_rate"]
                                    _n = _class_data["samples"]
                                    _lp = _class_data.get("long_pct", 0)
                                    request.chart_state["recent_accuracy"]["regime_warning"] = {
                                        "regime_class": _current_class,
                                        "win_rate": _wr,
                                        "samples": _n,
                                        "long_pct": _lp,
                                        "warning_text": (
                                            f"⚠️ 該 symbol 在 {_current_class} regime 過去 90 天歷史命中率僅 "
                                            f"{_wr}% (n={_n}，過去多空比 {_lp}% long)。禁止盲目順勢；"
                                            f"必須權衡 contrarian 視角，最高給 medium 信心，或改建議觀望。"
                                        ),
                                    }
                            # direction_balance：過去 30 天該 symbol 多空分布
                            _dir_stats = prediction_tracker.get_direction_stats(
                                symbol=chart_symbol, days=30,
                            )
                            if _dir_stats:
                                _long_n = _dir_stats.get("long", {}).get("samples", 0)
                                _short_n = _dir_stats.get("short", {}).get("samples", 0)
                                _total_dir = _long_n + _short_n
                                if _total_dir >= 10:
                                    _long_pct = round(_long_n / _total_dir * 100, 1)
                                    request.chart_state["recent_accuracy"]["direction_balance"] = {
                                        "long_n": _long_n,
                                        "short_n": _short_n,
                                        "long_pct": _long_pct,
                                        "biased_long": _long_pct > 75,
                                        "biased_short": _long_pct < 25,
                                    }
                        except Exception as _v118_err:
                            logger.debug(f"v118 regime_warning/direction_balance 注入失敗: {_v118_err}")

                        # ★ v120.5：注入 signal_history（訊號組合歷史命中率）
                        # 把當下 chart_state.external_signals.derivatives 訊號 classify 成 buckets，
                        # 查歷史「相同 bucket 組合」命中率，讓 LLM 判斷此次訊號組合是否有 alpha。
                        try:
                            _ext_sig = (request.chart_state or {}).get("external_signals") or {}
                            _ext_deriv = _ext_sig.get("derivatives") or {}
                            _ext_sentiment = _ext_sig.get("sentiment") or {}
                            if not (_ext_deriv or _ext_sentiment):
                                _mark_status(request.chart_state, "signal_history",
                                             "skipped", "no_derivatives_or_sentiment_data")
                            if _ext_deriv or _ext_sentiment:
                                from app.core.signal_buckets import classify_all_signals
                                _current_buckets = classify_all_signals(_ext_deriv, _ext_sentiment)
                                # 過濾 UNKNOWN，只留實際有資料的訊號
                                _active_buckets = {
                                    k: v for k, v in _current_buckets.items()
                                    if v and v != "UNKNOWN"
                                }
                                if _active_buckets:
                                    _combo = prediction_tracker.get_signal_combo_stats(
                                        symbol=chart_symbol,
                                        current_buckets=_active_buckets,
                                        days=90,
                                    )
                                    # 單一訊號 stats（給 LLM 看每個訊號的個別命中率）
                                    _single_stats = {}
                                    for _sig_name, _sig_bucket in _active_buckets.items():
                                        _ss = prediction_tracker.get_single_signal_stats(
                                            symbol=chart_symbol,  # 該 symbol 樣本可能不夠，後續可加 None fallback
                                            signal_name=_sig_name,
                                            bucket=_sig_bucket,
                                            days=90,
                                        )
                                        # 樣本太少（< 5）改用全 symbol 統計
                                        if _ss.get("samples", 0) < 5:
                                            _ss = prediction_tracker.get_single_signal_stats(
                                                symbol=None,
                                                signal_name=_sig_name,
                                                bucket=_sig_bucket,
                                                days=180,  # 拉長窗口
                                            )
                                            _ss["scope"] = "all_symbols_180d"
                                        else:
                                            _ss["scope"] = "this_symbol_90d"
                                        _single_stats[f"{_sig_name}_{_sig_bucket}"] = _ss
                                    request.chart_state["recent_accuracy"]["signal_history"] = {
                                        "current_buckets": _active_buckets,
                                        "combo_stats": _combo,
                                        "single_signal_stats": _single_stats,
                                    }
                                    _mark_status(request.chart_state, "signal_history",
                                                 "ok", f"n_buckets={len(_active_buckets)}")
                                else:
                                    _mark_status(request.chart_state, "signal_history",
                                                 "skipped", "all_buckets_UNKNOWN")
                        except Exception as _v120_err:
                            logger.info(f"v120.5 signal_history 注入失敗: {_v120_err}")
                            _mark_status(request.chart_state, "signal_history",
                                         "failed", str(_v120_err))

                        # ★ v124：機率三聯（baseline / TA 條件化 / track record）注入
                        try:
                            _df_t = request.chart_state.get("_cached_df")
                            _tf_t = request.chart_state.get("timeframe", "4h")
                            if _df_t is not None and chart_symbol:
                                _inject_probability_triplet(
                                    request.chart_state, _df_t, chart_symbol, _tf_t,
                                )
                            else:
                                _mark_status(request.chart_state, "probability_triplet",
                                             "skipped",
                                             f"df={'ok' if _df_t is not None else 'none'} "
                                             f"symbol={'ok' if chart_symbol else 'none'}")
                        except Exception as _v124_err:
                            logger.info(f"v124 probability_triplet 注入失敗: {_v124_err}")
                            _mark_status(request.chart_state, "probability_triplet",
                                         "failed", str(_v124_err))
                except Exception as _ra_err:
                    logger.info(f"recent_accuracy 注入失敗: {_ra_err}")
                    _mark_status(request.chart_state, "recent_accuracy", "failed", str(_ra_err))
            except Exception as e:
                logger.warning(f"預測反饋載入失敗: {e}")

            try:
                _active_summary = get_active_predictions_summary(chart_symbol)
                if _active_summary:
                    context_parts.append(f"【進行中的預測】\n{_active_summary}")
            except Exception as e:
                logger.warning(f"活躍預測摘要載入失敗: {e}")

        # ★ v102 Phase 2.3：模仿學習推論（subprocess 隔離）
        # 主進程**永不載 lightgbm/shap**，避免 macOS native lib 衝突 segfault
        # 6 層守衛（任一未過 → 不暴露 v101 給 user，等同 v100）：
        #   1. chart_symbol 非空
        #   2. intent 命中（comprehensive_analysis / deep_phase3）
        #   3. chart_state 非 None
        #   4. imitation_learning_enabled = True OR imitation_shadow_mode = True
        #   5. (learning) quality_gate 通過 + canary % 命中
        if (
            chart_symbol
            and (_intents & {"comprehensive_analysis", "deep_phase3"})
            and request.chart_state is not None
            and (settings.imitation_learning_enabled or settings.imitation_shadow_mode)
        ):
            try:
                from app.core.canary import use_v101
                # ★ v102: 用 ml_client 而非直接 import imitation_predictor
                #   → 主進程不載 lightgbm/shap
                from app.core.ml_client import predict_via_subprocess
                # ★ v102.1: 用 feature_extractor 抓全 39 特徵（不只 4 個）
                #   → SHAP 顯示真實值；KNN similar_paths 找到真正相似的歷史
                from app.core.feature_extractor import extract_features_at
                from app.data.fetchers.crypto_engine import crypto_engine

                _chart_tf = (request.chart_state or {}).get("timeframe") or "4h"
                _df_for_features = crypto_engine.load_local_data(chart_symbol, _chart_tf)

                if _df_for_features is not None and not _df_for_features.empty:
                    # Placeholder prediction（LLM 還沒生 entry/target/stop）
                    # 用 current price + 預設 target/stop 作 RR 估算
                    _current_price = float(_df_for_features["close"].iloc[-1])
                    _placeholder_pred = {
                        "symbol": chart_symbol,
                        "direction": "long",
                        "entry_price": _current_price,
                        "target_price": _current_price * 1.05,  # +5% target placeholder
                        "stop_price": _current_price * 0.97,    # -3% stop placeholder
                        "timeframe_hours": 48,
                        "confidence": "medium",
                        "regime": (
                            ((request.chart_state or {}).get("currentRegime") or {}).get("regime", "unknown")
                        ),
                    }
                    _current_features = extract_features_at(
                        _df_for_features, request.chart_state, _placeholder_pred
                    )
                else:
                    # OHLCV 載入失敗 fallback：只送 4 個必要特徵（其餘 35 個會被填 0）
                    _ra = (request.chart_state or {}).get("recent_accuracy")
                    _current_features = {
                        "direction_long": 1,
                        "confidence_score": 1,
                        "regime_confidence": float(
                            ((request.chart_state or {}).get("currentRegime") or {}).get("confidence", 0)
                        ),
                        "recent_30d_winrate": (_ra or {}).get("win_rate_30d") if isinstance(_ra, dict) else None,
                    }

                if use_v101(chart_symbol):
                    # 通過所有守衛 → subprocess 推論 + 注入給 user
                    insight = predict_via_subprocess(
                        _current_features, request.chart_state, timeout_sec=20
                    )
                    if insight:
                        request.chart_state["rl_strategic_insight"] = insight
                        _mark_status(request.chart_state, "rl_strategic_insight", "ok",
                                     f"mode={insight.get('mode')}")
                        logger.info(
                            f"[v101] 注入 rl_strategic_insight: "
                            f"mode={insight.get('mode')} p={insight.get('p_hit_target')}"
                        )
                    else:
                        _mark_status(request.chart_state, "rl_strategic_insight",
                                     "failed", "predict_via_subprocess returned empty")
                elif settings.imitation_shadow_mode:
                    # SHADOW MODE：subprocess 推論但不注入給 user
                    # （目的：驗證 subprocess 流程穩定 + 累積資料給 quality gate）
                    insight = predict_via_subprocess(
                        _current_features, request.chart_state, timeout_sec=20
                    )
                    if insight:
                        logger.debug(
                            f"[v101 shadow] mode={insight.get('mode')} "
                            f"p={insight.get('p_hit_target')}"
                        )
                    _mark_status(request.chart_state, "rl_strategic_insight",
                                 "guard_failed", "shadow_mode_only_no_user_inject")
                else:
                    _mark_status(request.chart_state, "rl_strategic_insight",
                                 "guard_failed", "canary_not_hit_or_quality_gate_failed")
            except Exception as _ie:
                logger.info(f"v101 subprocess 失敗（不影響 v100 流程）: {_ie}")
                _mark_status(request.chart_state, "rl_strategic_insight",
                             "failed", str(_ie))

        if chart_symbol and (_intents & {"analysis", "backtest", "calibrate"}):
            try:
                _calibration_prompt = format_calibration_for_prompt(chart_symbol)
                if _calibration_prompt:
                    context_parts.append(
                        f"【★★ 指標參數校準數據 — 必須使用】\n"
                        f"以下是根據歷史回測優化過的指標參數。你在分析中引用這些指標時，"
                        f"「必須」使用校準後的最佳參數值，而非教科書預設值。\n{_calibration_prompt}"
                    )
            except Exception as e:
                logger.warning(f"校準數據載入失敗: {e}")

        if rag_fragments:
            frag_texts = [
                f"• [{f['type']}] {f['content']}（相關度 {f['similarity']:.0%}）"
                for f in rag_fragments
            ]
            context_parts.append("【歷史分析經驗碎片】\n" + "\n".join(frag_texts))

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
    elif request.mode == "factor_validation":
        fv_prefix = (
            "[系統指令：使用者點擊了「因子驗證」按鈕]\n"
            "你必須呼叫 run_quant_research 取得因子 IC 數據。\n"
            "專注報告：因子 IC 排名、雙因子組合 IC、Bucket 評分、共線性警告。\n"
            "不需要詳細的策略回測分析。\n\n"
            f"使用者備註：{request.message if request.message.strip() else '請分析當前標的的因子有效性'}"
        )
        messages.append({"role": "user", "content": fv_prefix})
    elif request.mode == "strategy_backtest":
        sb_prefix = (
            "[系統指令：使用者點擊了「策略回測」按鈕]\n"
            "你必須呼叫 run_quant_research 取得回測 + MC + WF + CPCV 數據。\n"
            "專注報告：回測績效、Monte Carlo、Walk Forward、CPCV 交叉驗證、倉位建議。\n"
            "不需要詳細的因子 IC 分析。\n\n"
            f"使用者備註：{request.message if request.message.strip() else '請驗證當前標的的策略穩定性'}"
        )
        messages.append({"role": "user", "content": sb_prefix})
    elif request.mode == "regime_analysis":
        ra_prefix = (
            "[系統指令：使用者點擊了「市場體制」按鈕]\n"
            "你必須呼叫 generate_scenarios 取得 GMM/GARCH/HMM 市場體制數據。\n"
            "專注報告：GMM regime 分類、GARCH 波動率預測、HMM 狀態轉移、Bucket 評分。\n"
            "說明當前 regime 適合什麼策略。\n\n"
            f"使用者備註：{request.message if request.message.strip() else '請分析當前市場體制'}"
        )
        messages.append({"role": "user", "content": ra_prefix})
    elif request.mode == "fundamental_analysis":
        fa_prefix = (
            "[系統指令：使用者點擊了「基本面」按鈕]\n"
            "你必須呼叫 analyze_fundamentals 取得基本面數據。\n"
            "專注報告：月營收趨勢、法人動向、財報指標、綜合評分。\n\n"
            f"使用者備註：{request.message if request.message.strip() else '請分析當前標的的基本面'}"
        )
        messages.append({"role": "user", "content": fa_prefix})
    elif request.mode == "momentum_analysis":
        ma_prefix = (
            "[系統指令：使用者點擊了「動能分析」按鈕]\n"
            "你必須呼叫 analyze_momentum 取得完整動量數據。\n"
            "專注報告：多週期動量、加速/減速、相對強弱、反轉信號、動量策略回測、綜合評分。\n\n"
            f"使用者備註：{request.message if request.message.strip() else '請分析當前標的的動能狀態'}"
        )
        messages.append({"role": "user", "content": ma_prefix})
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

        # v110：顯示 result 內的 function name 而非 fc.name
        # 主因：executor v110 之前 results 順序跟 function_calls 不對應，會把 error 訊息掛到別的 function 上
        # v110 已改 by-index 但顯示層也用 result.function 雙重保險
        actual_name = result.get("function") if isinstance(result, dict) and result.get("function") else fname
        if actual_name != fname:
            logger.warning(f"[fc_results] 函式 {i+1} fc.name={fname!r} 但 result.function={actual_name!r} (順序對應錯)")

        parts.append(f"### 函式 {i+1}: {actual_name}")
        parts.append(f"參數: {json.dumps(fargs, ensure_ascii=False, default=_json_safe_default)}")

        if "error" in result:
            parts.append(f"錯誤: {result['error']}")
        elif "result" in result:
            r = result["result"]
            # 格式化不同類型的結果
            if fname == "query_chart_data":
                cu = r.get("chart_updates", {})
                sym = cu.get("symbol", "?")
                # 計算是否為多幣查詢（超過 1 個 query_chart_data）
                chart_call_count = sum(1 for fc2 in function_calls if fc2.get("name") == "query_chart_data")
                is_multi = chart_call_count > 1

                parts.append(f"========== {sym} 價格數據 ==========")
                parts.append(f"幣種: {sym}，時間框架: {cu.get('timeframe', '?')}，"
                             f"共 {cu.get('dataPoints', 0)} 根 K 線")
                # ★ 沒有數據時強制提示 LLM 呼叫下載
                if r.get("no_data") or cu.get("dataPoints", 0) == 0:
                    hint = r.get("hint", "")
                    if hint:
                        parts.append(f"⚠️ {hint}")
                    else:
                        parts.append(f"⚠️ 沒有本地數據。請呼叫 sync_symbol_data 下載 {sym} 的數據，不要只回覆沒有數據。")
                ps = r.get("price_summary")
                if ps:
                    parts.append(f"[{sym}] 期間最高: {ps.get('period_high')} ({ps.get('period_high_date')})")
                    parts.append(f"[{sym}] 期間最低: {ps.get('period_low')} ({ps.get('period_low_date')})")
                    parts.append(f"[{sym}] 首日開盤: {ps.get('first_open')} ({ps.get('first_date')})")
                    parts.append(f"[{sym}] 最後收盤: {ps.get('last_close')} ({ps.get('last_date')})")
                    for mo in (ps.get("monthly_ohlc") or []):
                        parts.append(f"  [{sym}] {mo['m']}: H={mo['h']} L={mo['l']} C={mo['c']}")
                    daily = ps.get("daily_ohlc") or []
                    if is_multi and len(daily) > 7:
                        daily = daily[-7:]  # 多幣查詢時只保留最近 7 天，避免上下文過長混淆
                        parts.append(f"  （{sym} 僅顯示最近 7 天）")
                    for day in daily:
                        parts.append(f"  [{sym}] {day['d']}: H={day['h']} L={day['l']} C={day['c']}")
                    candles = ps.get("candles") or []
                    if is_multi and len(candles) > 10:
                        candles = candles[-10:]
                        parts.append(f"  （{sym} 僅顯示最近 10 根 K 線）")
                    for c in candles:
                        parts.append(f"  [{sym}] {c['t']}: H={c['h']} L={c['l']} C={c['c']}")
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
            elif fname == "generate_scenarios":
                parts.append(f"當前價格: {r.get('current_price', '?')}")
                parts.append(f"數據量: {r.get('data_points', 0)} 根 K 線")
                for sc in r.get("scenarios", []):
                    arrow = {"bullish": "▲", "neutral": "▬", "bearish": "▼"}.get(sc.get("direction", ""), "●")
                    parts.append(f"\n{arrow} {sc['label']} ({sc.get('probability_pct', '?')})")
                    pt = sc.get("price_target", {})
                    parts.append(f"  目標區間: {pt.get('low', '?')} ~ {pt.get('high', '?')}")
                    parts.append(f"  風險: {sc.get('risk_level', '?')}")
                    if sc.get("invalidation"):
                        parts.append(f"  失效條件: {sc['invalidation']}")
                    for sig in sc.get("supporting_signals", []):
                        parts.append(f"  ↳ {sig.get('name', '?')}: {sig.get('interpretation', '')}")
                sources = r.get("signal_sources", {})
                w = sources.get("weights", {})
                if w:
                    w_str = ", ".join(f"{k}({v*100:.0f}%)" for k, v in w.items())
                    parts.append(f"\n信心來源權重: {w_str}")
                # Bucket 因子群評分
                bucket = r.get("bucket_scores", {})
                if bucket:
                    parts.append(f"\n因子群 Bucket 評分（合計 {bucket.get('total', 0)}/{bucket.get('max_possible', 10)}）→ {bucket.get('direction', '?')}")
                    for group_name, group_score in bucket.get("scores", {}).items():
                        indicator = "▲" if group_score > 0 else ("▼" if group_score < 0 else "▬")
                        parts.append(f"  {group_name}: {indicator} {group_score:+d}")
                # 附加 GMM/GARCH/HMM（基礎分析也有）
                gmm_s = r.get("gmm_regime", {})
                if gmm_s and gmm_s.get("status") == "success":
                    parts.append(f"\nGMM 市場體制: {gmm_s.get('current_regime', '?')}")
                    for rn, rp in gmm_s.get("regime_probabilities", {}).items():
                        parts.append(f"  {rn}: {rp}%")
                garch_s = r.get("garch_volatility", {})
                if garch_s and garch_s.get("status") == "success":
                    parts.append(f"GARCH 波動率: {garch_s.get('vol_direction', '?')} | 止損倍率 {garch_s.get('suggested_sl_multiplier', '?')}x")
                hmm_s = r.get("hmm_regime", {})
                if hmm_s and hmm_s.get("status") == "success":
                    parts.append(f"HMM 狀態: {hmm_s.get('current_state', '?')} | {hmm_s.get('interpretation', '?')}")
                # 歷史準確率
                ha = r.get("historical_accuracy", {})
                if ha:
                    parts.append(f"情境預測歷史準確率: {ha.get('direction_accuracy_pct', '?')}%（{ha.get('n_evaluations', '?')} 次評估）")
            elif fname == "analyze_fundamentals":
                parts.append(f"📊 基本面分析 — {r.get('code', '?')} {r.get('name', '?')}")
                # 營收
                rev = r.get("revenue", {})
                if rev.get("available"):
                    parts.append(f"\n營收趨勢: {rev.get('trend', '?')}")
                    parts.append(f"  最新月: {rev.get('latest_month', '?')} 營收={rev.get('latest_revenue', '?')}")
                    if rev.get("mom_pct") is not None:
                        parts.append(f"  MoM: {rev['mom_pct']}% | YoY: {rev.get('yoy_pct', '?')}%")
                    if rev.get("consecutive_growth_months"):
                        parts.append(f"  連續成長: {rev['consecutive_growth_months']} 個月")
                # 法人
                inst = r.get("institutional", {})
                if inst.get("available"):
                    parts.append(f"\n法人動向: {inst.get('direction', '?')}")
                    parts.append(f"  外資近20日: {inst.get('foreign_net_20d', '?')} 張")
                    parts.append(f"  投信近20日: {inst.get('trust_net_20d', '?')} 張")
                    if inst.get("foreign_consecutive_buy"):
                        parts.append(f"  外資連買: {inst['foreign_consecutive_buy']} 天")
                # 財報
                fin = r.get("financials", {})
                if fin.get("available"):
                    parts.append(f"\n財報指標:")
                    if fin.get("pe_ratio"):
                        parts.append(f"  本益比: {fin['pe_ratio']} ({fin.get('pe_assessment', '')})")
                    if fin.get("eps_trailing"):
                        parts.append(f"  EPS: {fin['eps_trailing']}")
                    if fin.get("dividend_yield"):
                        parts.append(f"  殖利率: {fin['dividend_yield']}% ({fin.get('yield_assessment', '')})")
                # 評分
                score = r.get("fundamental_score", {})
                if score:
                    parts.append(f"\n綜合基本面評分: {score.get('score', '?')}/{score.get('max_possible', 10)} → {score.get('direction', '?')}")
            elif fname == "sync_symbol_data":
                if r.get("status") == "success":
                    parts.append(f"✅ 數據下載完成: {r.get('symbol', '?')} {r.get('timeframe', '?')}")
                    parts.append(f"  共 {r.get('bars', 0)} 根 K 線")
                    parts.append(f"  範圍: {r.get('range', '?')}")
                else:
                    parts.append(f"❌ 下載失敗: {r.get('message', '?')}")
            elif fname == "sync_sector_data":
                parts.append(f"📦 族群批次下載: {r.get('sector_name', '?')}（{r.get('success', 0)}/{r.get('total', 0)} 檔成功）")
                for d in r.get("details", []):
                    icon = "✅" if d.get("status") == "ok" and d.get("bars", 0) > 0 else "❌"
                    parts.append(f"  {icon} {d.get('name', '?')}（{d.get('symbol', '?')}）: {d.get('bars', 0)} 根")
            elif fname == "analyze_momentum":
                parts.append(f"📊 動能分析 — {r.get('symbol', '?')} {r.get('timeframe', '?')}（{r.get('total_bars', 0)} 根）")
                # 動量因子
                mf = r.get("momentum_factors", {})
                for k, v in mf.items():
                    if k.startswith("mom_"):
                        parts.append(f"  {v.get('label', k)}: {v.get('return_pct', '?')}%")
                cm = mf.get("classic_momentum", {})
                if cm:
                    parts.append(f"  經典動量: {cm.get('value', '?')} ({cm.get('interpretation', '')})")
                cons = mf.get("consistency", {})
                if cons:
                    parts.append(f"  方向一致性: {cons.get('interpretation', '?')}")
                # 動量加速
                ms = r.get("momentum_shift", {})
                if ms:
                    parts.append(f"\n動量狀態: {ms.get('state', '?')} | ROC5={ms.get('roc_5', '?')}% | 加速度={ms.get('acceleration', '?')}")
                    if ms.get("shift_detected"):
                        parts.append(f"  ⚡ 轉折: {ms.get('shift_type', '?')}")
                # 相對動量
                rm = r.get("relative_momentum", {})
                if rm and rm.get("available"):
                    parts.append(f"\n相對動量（vs BTC）: {rm.get('interpretation', '?')}")
                    for pk, pv in rm.get("periods", {}).items():
                        parts.append(f"  {pk}: 標的 {pv.get('target_return', '?')}% vs BTC {pv.get('benchmark_return', '?')}% → RS={pv.get('relative_strength', '?')}")
                # 反轉
                rev = r.get("reversal_detection", {})
                if rev:
                    parts.append(f"\n反轉信號: {rev.get('net_direction', '?')}（多 {rev.get('bullish_signals', 0)} / 空 {rev.get('bearish_signals', 0)}）")
                    for sig in rev.get("signals", []):
                        parts.append(f"  {'🟢' if sig.get('direction') == 'bullish' else '🔴'} {sig.get('type', '?')}")
                # 策略回測
                sb = r.get("strategy_backtest", {})
                if sb:
                    parts.append(f"\n動量策略回測: 最佳={sb.get('best_strategy_name', '?')}")
                    for sk, sv in sb.get("strategies", {}).items():
                        parts.append(f"  {sv.get('name', sk)}: 勝率={sv.get('win_rate', '?')}% Sharpe={sv.get('sharpe', '?')} 報酬={sv.get('return_pct', '?')}%")
                # 綜合評分
                mscore = r.get("momentum_score", {})
                if mscore:
                    parts.append(f"\n綜合動量評分: {mscore.get('score', '?')}/{mscore.get('max_possible', 10)} → {mscore.get('direction', '?')}")
            elif fname == "detect_smc_structure":
                parts.append(f"📊 SMC 訂單流分析 — {r.get('symbol', '?')} {r.get('timeframe', '?')}")
                parts.append(f"結構方向: HTF={r.get('trend_htf', '?')} / LTF={r.get('trend_ltf', '?')} / 共振={'✅' if r.get('mtf_aligned') else '❌'}")
                parts.append(f"溢價/折價: {r.get('premium_discount', '?')} (Fib 0.5 = {r.get('fib_50', '?')})")
                # BOS/CHoCH 事件
                for evt in r.get("bos_points", []) + r.get("choch_points", []):
                    if evt.get("filtered"):
                        continue
                    tag = "🔴" if evt.get("type") == "CHoCH" else "🔵"
                    parts.append(f"  {tag} {evt['type']} @ {evt['price']} (量比 {evt.get('vol_ratio', '?')}x)")
                # Sweep 事件
                for sw in r.get("sweep_events", []):
                    if not sw.get("filtered"):
                        parts.append(f"  💧 {sw['type']} Sweep @ {sw['price']} (量比 {sw.get('vol_ratio', '?')}x)")
                # FVG
                unfilled_fvg = [f for f in r.get("fvg_zones", []) if not f.get("filled")]
                if unfilled_fvg:
                    parts.append(f"  未填補 FVG: {len(unfilled_fvg)} 個")
                # 交易建議
                bias = r.get("bias", "WAIT")
                parts.append(f"\n建議: {bias}")
                if r.get("entry"):
                    parts.append(f"  Entry: {r['entry']} / SL: {r.get('stop_loss', '?')} / TP: {r.get('take_profit', '?')}")
                    parts.append(f"  RR: {r.get('rr_ratio', '?')} / 信心: {r.get('confidence', '?')}%")
                # 信心分項
                bd = r.get("confidence_breakdown", {})
                if bd:
                    bd_str = ", ".join(f"{k}({v:+d})" for k, v in bd.items())
                    parts.append(f"  評分明細: {bd_str}")
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
            elif fname == "run_quant_research":
                parts.append(f"📊 量化研究報告 — {r.get('symbol', '?')} {r.get('timeframe', '?')}（{r.get('total_bars', 0)} 根 K 線）")
                if r.get("notice"):
                    parts.append(f"⚠️ {r['notice']}")
                # 因子 IC 排名
                ranking = r.get("factor_ic", {}).get("ranking", [])
                if ranking:
                    parts.append(f"\n因子預測力排名（共分析 {r.get('factor_ic', {}).get('total_analyzed', 0)} 個因子）:")
                    for f in ranking[:8]:
                        parts.append(f"  {f.get('factor', '?')}: IC={f.get('best_ic', '?')} [{f.get('power', '?')}] decay={f.get('decay_trend', '?')}")
                # 因子相關性
                corr = r.get("factor_correlation", {})
                if corr.get("recommendation"):
                    parts.append(f"因子相關性: {corr['recommendation']}")
                # 組合因子 IC（combo_top）
                fscan = r.get("factor_scan", {})
                combo_top = fscan.get("combo_top", [])
                if combo_top and isinstance(combo_top, list):
                    parts.append(f"\n雙因子組合 IC（top {min(5, len(combo_top))}）:")
                    for combo in combo_top[:5]:
                        if isinstance(combo, dict):
                            parts.append(f"  {combo.get('factor_a', '?')} + {combo.get('factor_b', '?')}: combo_IC={combo.get('combo_ic', '?')}")
                # Bucket 因子群評分
                bucket_qr = r.get("bucket_scores", {})
                if bucket_qr:
                    parts.append(f"\n因子群 Bucket 評分（合計 {bucket_qr.get('total', 0)}/{bucket_qr.get('max_possible', 10)}）→ {bucket_qr.get('direction', '?')}")
                    for grp_name, grp_score in bucket_qr.get("scores", {}).items():
                        ind = "▲" if grp_score > 0 else ("▼" if grp_score < 0 else "▬")
                        parts.append(f"  {grp_name}: {ind} {grp_score:+d}")
                # 回測
                bt = r.get("backtest", {})
                if bt and not bt.get("error"):
                    parts.append("\n回測績效:")
                    parts.append(f"  勝率: {bt.get('win_rate', '?')}% | PF: {bt.get('profit_factor', '?')} | 交易: {bt.get('total_trades', '?')} 筆")
                    parts.append(f"  Sharpe: {bt.get('sharpe_ratio', '?')} | Sortino: {bt.get('sortino_ratio', '?')}")
                    parts.append(f"  Expectancy: {bt.get('expectancy_pct', '?')}% | 最大回撤: {bt.get('max_drawdown_pct', '?')}%")
                    parts.append(f"  總報酬: {bt.get('total_return_pct', '?')}%")
                # Monte Carlo
                mc = r.get("monte_carlo", {})
                if mc and mc.get("status") == "success":
                    parts.append("\nMonte Carlo 模擬:")
                    parts.append(f"  獲利機率: {mc.get('profit_probability', '?')}% | 破產風險: {mc.get('ruin_probability', '?')}%")
                    parts.append(f"  報酬分布 p25={mc.get('p25_return', '?')}% / p50={mc.get('p50_return', '?')}% / p75={mc.get('p75_return', '?')}%")
                    parts.append(f"  策略穩健: {'✅' if mc.get('strategy_robust') else '❌'}")
                # Walk Forward
                wf = r.get("walk_forward", {})
                if wf:
                    assessment = wf.get("assessment", {})
                    summary = wf.get("summary", {})
                    if assessment:
                        parts.append("\nWalk Forward 驗證:")
                        parts.append(f"  Alpha: {'✅' if assessment.get('has_alpha') else '❌'} | 評分: {assessment.get('score', '?')}/100")
                        if summary:
                            parts.append(f"  視窗數: {summary.get('n_windows', '?')} | OOS 平均報酬: {summary.get('avg_oos_return', '?')}%")
                # GMM Regime
                gmm = r.get("gmm_regime", {})
                if gmm and gmm.get("status") == "success":
                    parts.append(f"\nGMM 市場體制分類:")
                    parts.append(f"  當前 Regime: {gmm.get('current_regime', '?')}")
                    for rname, prob in gmm.get("regime_probabilities", {}).items():
                        parts.append(f"  {rname}: {prob}%")
                    for rname, rstats in gmm.get("regime_stats", {}).items():
                        parts.append(f"  {rname}: 歷史佔比 {rstats.get('pct', '?')}%, 平均報酬 {rstats.get('avg_return', '?')}%")
                # GARCH 波動率
                garch = r.get("garch_volatility", {})
                if garch and garch.get("status") == "success":
                    parts.append(f"\nGARCH 波動率預測:")
                    parts.append(f"  當前波動率: {garch.get('current_volatility', '?')}")
                    parts.append(f"  方向: {garch.get('vol_direction', '?')} | 止損倍率建議: {garch.get('suggested_sl_multiplier', '?')}x")
                    parts.append(f"  解讀: {garch.get('interpretation', '?')}")
                # HMM 狀態轉移
                hmm = r.get("hmm_regime", {})
                if hmm and hmm.get("status") == "success":
                    parts.append(f"\nHMM 狀態轉移:")
                    parts.append(f"  當前狀態: {hmm.get('current_state', '?')}")
                    parts.append(f"  解讀: {hmm.get('interpretation', '?')}")
                    for state_name, dur in hmm.get("expected_duration_bars", {}).items():
                        parts.append(f"  {state_name}: 預期持續 {dur} 根 K 線")
                # CPCV
                cpcv_data = r.get("cpcv", {})
                if cpcv_data:
                    cpcv_assess = cpcv_data.get("assessment", {})
                    cpcv_met = cpcv_data.get("metrics_distribution", {})
                    if cpcv_assess:
                        parts.append(f"\nCPCV 組合淨化交叉驗證（{cpcv_data.get('n_combinations', '?')} 組合）:")
                        parts.append(f"  一致性: {cpcv_met.get('consistency_pct', '?')}%")
                        cpcv_ret = cpcv_met.get("return_pct", {})
                        parts.append(f"  報酬: 均值={cpcv_ret.get('mean', '?')}%, P25={cpcv_ret.get('p25', '?')}%, 中位數={cpcv_ret.get('median', '?')}%")
                        parts.append(f"  評分: {cpcv_assess.get('score', '?')}/100 | 有邊際: {'✅' if cpcv_assess.get('has_edge') else '❌'}")
                        parts.append(f"  結論: {cpcv_assess.get('verdict', '?')}")
                # OOS Monte Carlo
                mc_oos = r.get("monte_carlo_oos", {})
                if mc_oos and mc_oos.get("status") == "success":
                    parts.append(f"\nOOS Monte Carlo（Walk Forward OOS 交易）:")
                    parts.append(f"  獲利機率: {mc_oos.get('profit_probability', '?')}% | 破產風險: {mc_oos.get('ruin_probability', '?')}%")
                    parts.append(f"  策略穩健: {'✅' if mc_oos.get('strategy_robust') else '❌'}")
                # 倉位建議
                pos = r.get("position_sizing", {})
                if pos and not pos.get("error"):
                    parts.append(f"\n倉位建議: {pos.get('recommendation', pos.get('summary', '?'))}")
                    mc_adj = pos.get("mc_adjustment", {})
                    if mc_adj:
                        parts.append(f"  MC 調整: {mc_adj.get('reason', '?')}")
                # 結論
                conclusion = r.get("conclusion", {})
                if conclusion:
                    parts.append(f"\n結論（綜合評分: {conclusion.get('score', '?')}/100）:")
                    for finding in conclusion.get("findings", []):
                        parts.append(f"  {finding}")
                    if conclusion.get("recommendation"):
                        parts.append(f"  建議: {conclusion['recommendation']}")
            elif fname == "run_backtest":
                m = r.get("metrics", {})
                parts.append(f"回測結果（{r.get('total_trades', 0)} 筆交易）:")
                parts.append(f"  勝率: {m.get('win_rate', '?')}% | PF: {m.get('profit_factor', '?')} | Sharpe: {m.get('sharpe_ratio', '?')}")
                parts.append(f"  總報酬: {m.get('total_return_pct', '?')}% | 最大回撤: {m.get('max_drawdown_pct', '?')}%")
                parts.append(f"  Sortino: {m.get('sortino_ratio', '?')} | Expectancy: {m.get('expectancy_pct', '?')}%")
                if r.get("warnings"):
                    parts.append(f"  ⚠️ 警告: {'; '.join(r['warnings'][:3])}")
            elif fname == "compare_strategies":
                parts.append(f"策略比較（{r.get('symbol', '?')} {r.get('timeframe', '?')}，共 {r.get('total_strategies', 0)} 個策略）:")
                for c in r.get("comparison", []):
                    if c.get("status") == "success":
                        m = c.get("metrics", {})
                        rank = c.get("rank", "?")
                        parts.append(f"  #{rank} {c['name']}: 勝率={m.get('win_rate', '?')}% PF={m.get('profit_factor', '?')} "
                                     f"Sharpe={m.get('sharpe_ratio', '?')} 報酬={m.get('total_return_pct', '?')}%")
                    else:
                        parts.append(f"  ✗ {c.get('name', '?')}: {c.get('message', '錯誤')}")
            elif fname == "compute_laddered_entries":
                # v106 B1：分批進場專用 formatter（取代 raw JSON dump）
                if not r.get("enabled"):
                    parts.append(f"分批進場：disabled — {r.get('warning', '?')}")
                    if r.get("sl_mult_hint"):
                        parts.append(f"  建議 SL ≈ {r['sl_mult_hint']} × {r.get('timeframe_used','?')} ATR")
                        parts.append(f"  建議 TP ≈ {r.get('tp_mult_hint','?')} × {r.get('timeframe_used','?')} ATR")
                else:
                    parts.append(
                        f"分批進場（regime={r.get('regime_used','?')} {r.get('regime_confidence',0):.2f} | "
                        f"配比 {r.get('ratio_strategy','?')}）"
                    )
                    parts.append(f"  ATR: {r.get('atr_used','?')} | 當前: {r.get('current_price','?')}")
                    if r.get("long_entries"):
                        parts.append("  Long entries:")
                        for e in r["long_entries"]:
                            parts.append(f"    {e['size_pct']}% @ ${e['price']} ({e.get('source','')})")
                        parts.append(
                            f"    avg=${r.get('weighted_avg_entry_long','?')} | "
                            f"SL=${r.get('stop_loss_long','?')} (≈{r.get('sl_mult_used','?')}×ATR) | "
                            f"TP=${r.get('take_profit_long','?')} (≈{r.get('tp_mult_used','?')}×ATR) | "
                            f"RR={r.get('rr_long','?')}"
                        )
                    if r.get("short_entries"):
                        parts.append("  Short entries:")
                        for e in r["short_entries"]:
                            parts.append(f"    {e['size_pct']}% @ ${e['price']} ({e.get('source','')})")
                        parts.append(
                            f"    avg=${r.get('weighted_avg_entry_short','?')} | "
                            f"SL=${r.get('stop_loss_short','?')} | "
                            f"TP=${r.get('take_profit_short','?')} | "
                            f"RR={r.get('rr_short','?')}"
                        )
                    if r.get("missing_indicators"):
                        parts.append(f"  ⚠️ 缺失指標: {', '.join(r['missing_indicators'])}")
            elif fname == "analyze_sector":
                # v106 B1：族群分析專用 formatter
                parts.append(f"族群分析：{r.get('sector_name', '?')}")
                if r.get("status") == "error":
                    parts.append(f"  錯誤：{r.get('message', '?')}")
                else:
                    parts.append(f"  成員: {r.get('member_count', '?')} 檔")
                    parts.append(f"  廣度: {r.get('breadth_pct', '?')}% advancing")
                    if r.get("leader"):
                        parts.append(f"  龍頭: {r['leader']} (5K 線 {r.get('leader_change_5d','?')}%)")
                    if r.get("interpretation"):
                        parts.append(f"  解讀: {r['interpretation'][:300]}")
            else:
                # 通用格式化（截斷過長內容）
                # v105.7：limit 3000→8000，且改為「前 6000 + 後 2000」截斷
                result_str = json.dumps(r, ensure_ascii=False, default=_json_safe_default)
                if len(result_str) > 8000:
                    result_str = (
                        result_str[:6000]
                        + f"\n\n... [中段省略 {len(result_str) - 8000} 字以節省 token] ...\n\n"
                        + result_str[-2000:]
                    )
                parts.append(f"結果: {result_str}")
        parts.append("")

    if chart_updates:
        parts.append(f"圖表更新: {json.dumps(chart_updates, ensure_ascii=False, default=_json_safe_default)[:300]}")

    return "\n".join(parts)


def _json_safe_default(o):
    """json.dumps 用的 default callback：把 numpy / pandas 型別轉成 Python 原生。

    numpy.bool_ 不是 Python bool 的子類別，會讓標準 json 拋 TypeError；
    numpy 數值 / ndarray / pandas Timestamp 同樣不被原生 json 認識。
    """
    # numpy 標量
    try:
        import numpy as _np
        if isinstance(o, _np.bool_):
            return bool(o)
        if isinstance(o, _np.integer):
            return int(o)
        if isinstance(o, _np.floating):
            f = float(o)
            return f if f == f else None  # NaN → None
        if isinstance(o, _np.ndarray):
            return o.tolist()
    except Exception:
        pass
    # pandas Timestamp / Timedelta
    try:
        import pandas as _pd
        if isinstance(o, (_pd.Timestamp, _pd.Timedelta)):
            return str(o)
        if hasattr(o, "to_dict"):  # DataFrame / Series
            return o.to_dict()
    except Exception:
        pass
    # set / frozenset
    if isinstance(o, (set, frozenset)):
        return list(o)
    # 最後手段：str()
    return str(o)


def _sse_event(event_type: str, data: dict) -> str:
    """格式化 SSE event"""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False, default=_json_safe_default)}\n\n"


# v110：SSE 心跳機制 — 防止 LLM 思考期間（無 token）SSE 連線被 OS / browser / proxy 任何層判定 idle 而斷
# 業界標準：long-lived SSE 連線每 N 秒送 heartbeat 維持 active
# 演進：15s（首版）→ 5s（實測仍會斷）→ 2s（v110 final）
# 實測 5s 仍不夠：v108 後 system prompt 變大（含 bilateral_plan / indicators_snapshot / 各規則段）
# 加上對話歷史累積 → LLM TTFT 達 4-5 秒，5s timeout「剛好」太晚 → client 在第 5 秒已斷
# 2s 給最積極保護：每 2 秒至少一個心跳，遠低於任何 OS / browser / fetch 內部 idle 偵測
_HEARTBEAT_INTERVAL = 2.0  # 秒
_HEARTBEAT_SENTINEL = object()  # 用 sentinel 物件標記心跳事件


async def _stream_with_heartbeat(stream_iter, interval: float = _HEARTBEAT_INTERVAL):
    """包裝 LLM adapter 的 async iterator，無事件達 interval 秒 → yield 心跳 sentinel。

    使用方式：
        async for evt in _stream_with_heartbeat(adapter.chat_stream_events(...)):
            if evt is _HEARTBEAT_SENTINEL:
                # 心跳：呼叫端應 yield SSE heartbeat event 給 client
                continue
            # 處理真實 evt（既有邏輯）

    ★★★ 關鍵設計（v110 fix）：用 asyncio.shield 保護 pending task ★★★
    舊版直接 `await asyncio.wait_for(_iter.__anext__(), timeout=interval)` 在 timeout 時
    會 cancel inner coroutine（即 LLM adapter 的 __anext__），導致 adapter generator 內部
    state（subprocess / readline 等）被破壞。下次 __anext__ 立刻 StopAsyncIteration → 主
    流程看到 streaming 結束 → _r2_text_buf 為空 → 走「未能產生文字報告」 fallback。

    正確做法：把 pending task 存起來，每輪 wait_for 用 asyncio.shield 包它。timeout 取消
    的是 shield wrapper 不是 pending 本身，pending 繼續活著等 LLM 下一個 event。
    """
    _iter = stream_iter.__aiter__() if hasattr(stream_iter, "__aiter__") else stream_iter
    pending: asyncio.Task | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(_iter.__anext__())
            try:
                evt = await asyncio.wait_for(asyncio.shield(pending), timeout=interval)
                # 拿到 event，pending 已 done，下輪重新建立
                pending = None
                yield evt
            except asyncio.TimeoutError:
                # pending 仍活著（被 shield 保護），下輪迴圈繼續 await 同一個 pending
                yield _HEARTBEAT_SENTINEL
            except StopAsyncIteration:
                pending = None
                return
    finally:
        # 確保 pending 在 generator 結束時被清理（避免 task leak）
        if pending is not None and not pending.done():
            pending.cancel()
            try:
                await pending
            except (asyncio.CancelledError, StopAsyncIteration, Exception):
                pass


# ─── v100/v103 1B：結論卡「📈 系統參考」自動注入歷史命中率 ────────
# v103 1B 寬鬆化：容忍 LLM 寫不同字眼（將/由/會/即將...）
_PLACEHOLDER_PATTERN = re.compile(
    r"📈\s*系統參考[：:][^\n]*",
)


def _inject_recent_accuracy(final_text: str, symbol: str, regime: str) -> str:
    """v100：替換結論卡「📈 系統參考：」佔位行為實際歷史命中率。

    若 final_text 不含此佔位行（不是結論卡格式），原樣返回。
    若樣本不足（< 3 筆驗證），顯示「樣本不足」訊息。
    """
    matched = _PLACEHOLDER_PATTERN.search(final_text)
    if not matched:
        logger.debug("[v103 1B] _inject_recent_accuracy: 沒找到 placeholder，可能 LLM 沒產出結論卡")
        return final_text
    logger.info(f"[v103 1B] 命中 placeholder: '{matched.group(0)[:80]}'")
    try:
        stats = prediction_tracker.get_stats(symbol=symbol, regime=regime, days=30)
    except Exception as e:
        logger.warning(f"_inject_recent_accuracy 取統計失敗：{e}")
        return final_text

    total = stats.get("total", 0)

    # v108 Phase 3：另查 invalidated 筆數，附加到 hit-rate 後讓使用者知道
    invalidated_n = 0
    try:
        prediction_tracker._ensure_db()
        with prediction_tracker._lock:
            from datetime import timedelta as _td
            cutoff = (taipei_now() - _td(days=30)).isoformat()
            _q = "SELECT COUNT(*) FROM predictions WHERE status='invalidated' AND created_at > ?"
            _params: list = [cutoff]
            if symbol:
                _q += " AND symbol = ?"
                _params.append(symbol)
            if regime:
                _q += " AND regime = ?"
                _params.append(regime)
            row = prediction_tracker._conn.execute(_q, _params).fetchone()
            if row:
                invalidated_n = int(row[0] or 0)
    except Exception as _inv_err:
        logger.debug(f"_inject_recent_accuracy 取 invalidated 數失敗：{_inv_err}")

    if total < 3:
        replacement = f"📈 系統參考：樣本不足（n={total}，需 ≥ 3 筆已驗證），命中率尚不可估"
    else:
        wr = stats.get("win_rate_weighted", 0)
        bayesian = stats.get("bayesian", {})
        ci = bayesian.get("credible_interval_95", [None, None])
        ci_str = f"，CI {ci[0]}-{ci[1]}%" if ci[0] is not None else ""
        replacement = (
            f"📈 系統參考：你近 30 天類似條件（{regime}）命中率 {wr}%（n={total}{ci_str}）"
        )

    # 若有 invalidated，附加說明
    if invalidated_n > 0:
        replacement += f"｜📛 另 {invalidated_n} 筆因失效條件觸發已排除（不計入命中率）"

    return _PLACEHOLDER_PATTERN.sub(replacement, final_text)


def _execute_function_calls_in_thread(*args, **kwargs):
    """在 worker thread 內開新 event loop 跑 execute_function_calls。

    ★ 為什麼需要這個：
    execute_function_calls 內部會呼叫重 CPU 的 sync 操作（ML 訓練、回測、
    Monte Carlo、Walk Forward 等），這些是 NumPy/pandas/sklearn 的 sync
    呼叫，會阻塞 asyncio event loop。

    若直接 await execute_function_calls(...)，主 event loop 在重操作期間
    無法處理任何其他協程 → 心跳 yield SSE event 排不上時程 → 前端誤判
    超時斷線。

    解法：用 asyncio.to_thread + 新 event loop，讓 execute 在獨立 thread
    執行，主 event loop 完全不被佔用。

    用法：
        _fc_task = asyncio.create_task(asyncio.to_thread(
            _execute_function_calls_in_thread, fcs, chart_state=..., progress=...
        ))
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(execute_function_calls(*args, **kwargs))
    finally:
        loop.close()


def _post_process_chat_message(
    *,
    final_text: str,
    request_message: str,
    chart_state: Optional[dict],
    chart_symbol_for_save: Optional[str],
    conversation_id: str,
    total_usage,  # TokenUsage | None — 不嚴格型別避免循環 import
) -> None:
    """串流結束後的所有 DB 寫入和知識提取（v117：純 sync 跑在 thread pool）。

    ★ v117 改動：本函式內部全是 sync 操作（chat_history / analysis_cache /
       semantic_cache / fragment_store / prediction_tracker 都是 sync SQLite），
       原本 `async def` 但內部沒 `await` → 仍跑在主 event loop 上。當 SQLite
       fsync / embedding encode 慢時，會阻塞所有其他 endpoints，造成 backend
       hang（已實測 reproduce）。改成 sync function 由 caller 用
       `asyncio.create_task(asyncio.to_thread(...))` 丟到 thread pool 跑。

    ★ 設計理由：原本這段同步跑在 stream_gen 內，會造成 30-300 秒沉默
       （embedding 計算、predictions 驗證、OHLCV 重抓），導致前端誤判
       「分析回應超時」。改成 background task 後 SSE 可以立刻 yield done，
       使用者體驗即時、後處理悄悄完成。

    所有錯誤只 log 不 raise（不影響使用者已收到的回應）。
    """
    try:
        if not final_text.strip():
            return

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
            question=request_message,
            answer=clean_text,
            chart_state=chart_state,
        )

        semantic_cache.store(
            question=request_message,
            answer=clean_text,
            chart_state=chart_state,
        )

        frag_symbol = extract_symbol_from_text(request_message) or chart_symbol_for_save or ""
        if insights:
            stored = fragment_store.store_batch(
                fragments=insights,
                symbol=frag_symbol,
                source_question=request_message,
            )
            if stored:
                logger.info(f"[bg] 自動提取 {stored} 筆知識碎片（{frag_symbol}）")

        if distill_fragments:
            stored_d = fragment_store.store_batch(
                fragments=distill_fragments,
                symbol=frag_symbol,
                source_question=request_message,
            )
            if stored_d:
                logger.info(f"[bg] 自動提取 {stored_d} 筆蒸餾碎片（{frag_symbol}）")

        # 提取並存儲預測（v100：低信心過濾，避免汙染命中率統計）
        predictions = parse_predictions(final_text)
        # v100：過濾掉信心="low" 的預測（雙保險，prompt 已要求低信心改用「建議觀望」格式）
        before_filter = len(predictions)
        predictions = [p for p in predictions if p.get("confidence") != "low"]
        if before_filter > len(predictions):
            logger.info(f"[bg] 過濾 {before_filter - len(predictions)} 筆低信心預測（不追蹤）")

        if predictions:
            pred_symbol = extract_symbol_from_text(request_message) or chart_symbol_for_save or ""
            pred_tf = (chart_state or {}).get("timeframe", "4h")
            for pred in predictions:
                try:
                    pred_id = prediction_tracker.store(
                        symbol=pred_symbol,
                        timeframe=pred_tf,
                        prediction=pred,
                        source_question=request_message,
                        chart_state=chart_state,  # v120.3: capture external_signals
                    )
                    # ★ Phase 2.0c：即時記錄 39 個特徵快照（feature flag 保護）
                    if pred_id and settings.feature_recording_enabled:
                        try:
                            from app.core.feature_extractor import record_features
                            from app.data.fetchers.crypto_engine import crypto_engine
                            df = crypto_engine.load_local_data(pred_symbol, pred_tf)
                            if df is not None and not df.empty:
                                pred_for_features = {**pred, "symbol": pred_symbol}
                                record_features(pred_id, df, chart_state, pred_for_features)
                        except Exception as fe:
                            logger.debug(f"[bg] 特徵記錄失敗（不影響預測）: {fe}")
                except Exception as pe:
                    logger.warning(f"[bg] 儲存預測失敗: {pe}")
            logger.info(f"[bg] 自動提取 {len(predictions)} 筆預測（{pred_symbol}）")

        # 追蹤 ML 預測（若存在）
        ml_pred = (chart_state or {}).get("mlPrediction")
        if ml_pred and isinstance(ml_pred, dict) and ml_pred.get("probability"):
            try:
                _ml_symbol = chart_symbol_for_save or ""
                _ml_tf = (chart_state or {}).get("timeframe", "4h")
                _ml_direction = ml_pred.get("direction", "long")
                _ml_prob = ml_pred.get("probability", 0.5)
                _ml_entry = (chart_state or {}).get("current_price", 0)
                if _ml_direction == "long":
                    _ml_target = _ml_entry * 1.03
                    _ml_stop = _ml_entry * 0.97
                else:
                    _ml_target = _ml_entry * 0.97
                    _ml_stop = _ml_entry * 1.03
                _ml_conf = "high" if _ml_prob >= 0.7 else ("medium" if _ml_prob >= 0.5 else "low")
                ml_pred_data = {
                    "direction": _ml_direction,
                    "entry_price": _ml_entry,
                    "target_price": round(_ml_target, 2),
                    "stop_price": round(_ml_stop, 2),
                    "timeframe_hours": 24,
                    "confidence": _ml_conf,
                    "regime": ml_pred.get("regime", "unknown"),
                    "indicators": "ml_model",
                }
                _ml_pid = prediction_tracker.store(
                    symbol=_ml_symbol, timeframe=_ml_tf,
                    prediction=ml_pred_data, source_question="ml_auto",
                    chart_state=chart_state,  # v120.3
                )
                prediction_tracker._ensure_db()
                prediction_tracker._conn.execute(
                    "UPDATE predictions SET ml_enhanced=1 WHERE id=?", (_ml_pid,)
                )
                prediction_tracker._conn.commit()
                logger.info(f"[bg] ML 預測已追蹤 #{_ml_pid}: {_ml_symbol} {_ml_direction}")
            except Exception as me:
                logger.warning(f"[bg] 儲存 ML 預測失敗: {me}")

        # 順便驗證已到期的預測
        try:
            val_result = validate_all_active()
            if val_result.get("validated", 0) > 0:
                logger.info(f"[bg] 自動驗證 {val_result['validated']} 筆預測")
        except Exception as ve:
            logger.warning(f"[bg] 自動驗證預測失敗: {ve}")

    except Exception as save_err:
        logger.error(f"[bg] post-processing 整體失敗（不影響使用者）: {save_err}")


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

    # 速率限制檢查（per-IP）
    client_ip = raw_request.client.host if raw_request.client else "unknown"
    if not _check_rate_limit(client_ip):
        return JSONResponse(
            status_code=429,
            content={"detail": "請求過於頻繁，請稍後再試（每分鐘上限 30 次）"},
        )

    # v106 D1：per-session 進階 rate limit（更嚴）
    from app.core.security import (
        check_session_rate_limit,
        detect_prompt_injection,
        log_security_event,
    )
    sess_ok, sess_remaining = check_session_rate_limit(request.session_id)
    if not sess_ok:
        log_security_event("session_rate_limited", {"session_id": (request.session_id or "")[:8]})
        return JSONResponse(
            status_code=429,
            content={"detail": "此 session 請求過於頻繁，請稍候再試（每分鐘上限 20 次）"},
        )

    # v106 D1：prompt injection 偵測（不阻擋，只 log + 後續 prompt 加 fence）
    injection_check = detect_prompt_injection(request.message)
    if injection_check["detected"]:
        log_security_event("prompt_injection_detected", {
            "severity": injection_check["severity"],
            "patterns": injection_check["patterns"],
            "session_id": (request.session_id or "")[:8],
        })

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

    # ★ 輕量前置處理（< 50ms）：API key 解析 + adapter 建立
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
    except Exception as exc:
        _exc_msg = str(exc)
        async def err_gen():
            yield _sse_event("error", {"error": f"無法連接 LLM: {_exc_msg}"})
            yield _sse_event("done", {})
        return StreamingResponse(err_gen(), media_type="text/event-stream")

    conversation_id = request.conversation_id or str(uuid.uuid4())

    _MAX_FINAL_TEXT = 50_000  # 累積文字最大長度（~50KB），防止記憶體爆炸

    async def stream_gen():
        total_usage = None
        final_text = ""
        _active_tasks: list[asyncio.Task] = []

        # 1. 立即告知前端「正在思考」
        yield _sse_event("thinking", {})
        yield _sse_event("status", {"message": "正在準備分析環境..."})

        # 2. 重量級前置處理（移入 streaming 內部，讓前端立即有回應）
        # ★ L3.5 知識融合準備
        _rag_context_fragments: list[dict] = []
        try:
            semantic_match = semantic_cache.try_get_with_score(request.message, request.chart_state)
            if semantic_match and semantic_match["similarity"] >= 0.75:
                logger.info(
                    f"L3.5 語意匹配（中等置信度 {semantic_match['similarity']:.2%}）→ 作為 RAG 參考"
                )
        except Exception as e:
            logger.warning(f"語意快取查詢失敗: {e}")

        chart_symbol = (request.chart_state or {}).get("symbol", "")
        try:
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
        except Exception as e:
            logger.warning(f"知識碎片檢索失敗: {e}")

        # ★ 意圖偵測 → prompt 組裝 → 指標計算 → ML 預測 → 訊息建構
        try:
            _intents = detect_intents(request.message, mode=request.mode)
            from app.api.routes.config import load_system_settings
            _teaching = load_system_settings().get("teaching_mode", False)
            _dynamic_prompt = assemble_system_prompt(_intents, teaching_mode=_teaching)
            logger.info(f"意圖偵測: {_intents}, 教學模式={'ON' if _teaching else 'OFF'} → SYSTEM_PROMPT 模組已動態組裝")

            yield _sse_event("status", {"message": "正在載入分析數據..."})

            # ★ 自動校準：分析意圖 + 該幣種無校準數據或已過期（>7天）→ 快速校準
            if "analysis" in _intents and chart_symbol:
                try:
                    from pathlib import Path
                    import time as _time
                    from app.core.config.settings import settings as _cal_settings
                    _cal_path = Path(_cal_settings.db_path) / "calibration" / f"{chart_symbol}.json"
                    _needs_cal = not _cal_path.exists()
                    if not _needs_cal:
                        _age_days = (_time.time() - _cal_path.stat().st_mtime) / 86400
                        _needs_cal = _age_days > 7
                    if _needs_cal:
                        from app.core.backtest.parameter_optimizer import run_calibration
                        chart_timeframe = (request.chart_state or {}).get("timeframe", "4h")
                        _msg = "首次分析，正在校準指標參數..." if not _cal_path.exists() else "校準參數已過期，正在更新..."
                        yield _sse_event("status", {"message": _msg})
                        # 包成 task + 心跳，避免校準時 SSE 沉默
                        _cal_task = asyncio.create_task(
                            asyncio.to_thread(run_calibration, chart_symbol, chart_timeframe)
                        )
                        _cal_hb = 0
                        while not _cal_task.done():
                            await asyncio.sleep(5)
                            _cal_hb += 5
                            yield _sse_event("status", {
                                "message": f"{_msg.rstrip('...')}... ({_cal_hb}秒)"
                            })
                        _cal_task.result()  # 觸發 raise 若有錯
                        logger.info(f"自動校準完成: {chart_symbol}")
                except Exception as e:
                    logger.warning(f"自動校準失敗（不影響分析）: {e}")

            # ★ 自動計算分析所需 + 使用者提到的指標值，注入 chart_state
            request.chart_state = _auto_calc_indicator_values(request.message, request.chart_state, intents=_intents)

            # ★ ML 增強：自動注入 ML 預測信號
            request.chart_state = _inject_ml_prediction(request.chart_state, _intents)

            # v106 D3：依 intent 壓縮 chart_state（節省 token / 提升 cache hit）
            try:
                from app.core.context_compressor import (
                    compress_chart_state, estimate_token_savings,
                )
                _orig_state = request.chart_state
                _compressed = compress_chart_state(request.chart_state, _intents)
                if _compressed and _orig_state:
                    _stats = estimate_token_savings(_orig_state, _compressed)
                    if _stats["saved_chars"] > 0:
                        logger.info(
                            f"[context_compress] intents={sorted(_intents)} "
                            f"kept={len(_compressed)-1}/{len(_orig_state)} fields "
                            f"saved≈{_stats['estimated_token_savings']} tokens "
                            f"({_stats['savings_pct']}%)"
                        )
                    request.chart_state = _compressed
            except Exception as _comp_err:
                logger.debug(f"context_compress 失敗（不影響主流程）: {_comp_err}")

            messages = _build_messages(request, rag_fragments=_rag_context_fragments, intents=_intents)
        except Exception as e:
            logger.exception(f"對話前置處理失敗: {e}")
            yield _sse_event("error", {"error": f"系統初始化錯誤: {str(e)}"})
            yield _sse_event("done", {})
            return

        # ★ 清理暫存的 DataFrame（不傳給 LLM）
        if request.chart_state:
            request.chart_state.pop("_cached_df", None)
            request.chart_state.pop("_cached_df_key", None)

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

        # ★ 預回測：分析意圖時，在 LLM 呼叫前先跑多空策略回測，注入上下文
        # ★ A1: 加 deep_phase3 + comprehensive_analysis（讓「全部分析」也跑 6 策略對比）
        _PRE_BT_INTENTS = {
            "analysis", "deep_analysis",
            "deep_phase1", "deep_phase2", "deep_phase3",
            "comprehensive_analysis",
        }
        if (_intents & _PRE_BT_INTENTS) and chart_symbol:
            yield _sse_event("status", {"message": "正在預跑策略回測..."})
            _pre_bt_calls = [{
                "name": "compare_strategies",
                "arguments": {
                    "symbol": chart_symbol,
                    "timeframe": chart_timeframe_ctx or "1d",
                    "strategies": [
                        # ★ v103 Phase 2A：dict 格式（engine 才認得）+ 正確 decimal 百分比（0.05 = 5%）
                        # 用 cross_above/cross_below 確保會真的觸發 trade（修舊版 0 trades bug）
                        {
                            "name": "均值回歸(多)",
                            "entry_conditions": [{"indicator": "rsi", "series": "RSI", "operator": "cross_above", "value": 30}],
                            "exit_conditions": [{"indicator": "rsi", "series": "RSI", "operator": "cross_above", "value": 70}],
                            "direction": "long", "stop_loss_pct": 0.05, "take_profit_pct": 0.10,
                            "compatible_regimes": ["ranging", "low_vol"],
                        },
                        {
                            "name": "均值回歸(空)",
                            "entry_conditions": [{"indicator": "rsi", "series": "RSI", "operator": "cross_below", "value": 70}],
                            "exit_conditions": [{"indicator": "rsi", "series": "RSI", "operator": "cross_below", "value": 30}],
                            "direction": "short", "stop_loss_pct": 0.05, "take_profit_pct": 0.10,
                            "compatible_regimes": ["ranging", "low_vol"],
                        },
                        {
                            "name": "MACD動量(多)",
                            "entry_conditions": [{"indicator": "macd", "series": "MACD_Hist", "operator": "cross_above", "value": 0}],
                            "exit_conditions": [{"indicator": "macd", "series": "MACD_Hist", "operator": "cross_below", "value": 0}],
                            "direction": "long", "stop_loss_pct": 0.06, "take_profit_pct": 0.12,
                            "compatible_regimes": ["trending_up"],
                        },
                        {
                            "name": "MACD動量(空)",
                            "entry_conditions": [{"indicator": "macd", "series": "MACD_Hist", "operator": "cross_below", "value": 0}],
                            "exit_conditions": [{"indicator": "macd", "series": "MACD_Hist", "operator": "cross_above", "value": 0}],
                            "direction": "short", "stop_loss_pct": 0.06, "take_profit_pct": 0.12,
                            "compatible_regimes": ["trending_down"],
                        },
                        {
                            "name": "ROC動量(多)",
                            "entry_conditions": [{"indicator": "roc", "series": "ROC", "operator": "cross_above", "value": 3}],
                            "exit_conditions": [{"indicator": "roc", "series": "ROC", "operator": "cross_below", "value": 0}],
                            "direction": "long", "stop_loss_pct": 0.06, "take_profit_pct": 0.12,
                            "compatible_regimes": ["trending_up", "high_vol"],
                        },
                        {
                            "name": "ROC動量(空)",
                            "entry_conditions": [{"indicator": "roc", "series": "ROC", "operator": "cross_below", "value": -3}],
                            "exit_conditions": [{"indicator": "roc", "series": "ROC", "operator": "cross_above", "value": 0}],
                            "direction": "short", "stop_loss_pct": 0.06, "take_profit_pct": 0.12,
                            "compatible_regimes": ["trending_down", "high_vol"],
                        },
                        {
                            "name": "StochRSI 超賣反彈(多)",
                            "entry_conditions": [{"indicator": "stochrsi", "series": "StochRSI_K", "operator": "cross_above", "value": 0.2}],
                            "exit_conditions": [{"indicator": "stochrsi", "series": "StochRSI_K", "operator": "cross_above", "value": 0.8}],
                            "direction": "long", "stop_loss_pct": 0.05, "take_profit_pct": 0.10,
                            "compatible_regimes": ["ranging", "low_vol"],
                        },
                        {
                            "name": "StochRSI 超買反轉(空)",
                            "entry_conditions": [{"indicator": "stochrsi", "series": "StochRSI_K", "operator": "cross_below", "value": 0.8}],
                            "exit_conditions": [{"indicator": "stochrsi", "series": "StochRSI_K", "operator": "cross_below", "value": 0.2}],
                            "direction": "short", "stop_loss_pct": 0.05, "take_profit_pct": 0.10,
                            "compatible_regimes": ["ranging", "low_vol"],
                        },
                    ],
                },
            }]
            try:
                # 預回測 6 個策略累計 60-180+ 秒，跑在 worker thread 避免阻塞主 event loop
                # （主 loop 必須空著才能即時 yield 心跳 SSE event）
                _pre_bt_task = asyncio.create_task(asyncio.to_thread(
                    _execute_function_calls_in_thread,
                    _pre_bt_calls, chart_state=request.chart_state,
                ))
                _active_tasks.append(_pre_bt_task)
                _pre_bt_hb = 0
                while not _pre_bt_task.done():
                    await asyncio.sleep(5)
                    _pre_bt_hb += 5
                    yield _sse_event("status", {
                        "message": f"正在預跑 8 個策略回測... ({_pre_bt_hb}秒，通常需 80-240 秒)"
                    })
                _pre_bt_result = _pre_bt_task.result()
                _pre_bt_summary = _format_function_results(_pre_bt_calls, _pre_bt_result)
                if _pre_bt_summary:
                    # 注入到 messages 的使用者訊息之前（倒數第一條是使用者訊息）
                    messages.insert(-1, {
                        "role": "user",
                        "content": (
                            "[系統預回測結果 — 必須參考]\n"
                            "以下是系統自動對 8 種策略（做多 4 + 做空 4，含 ROC 動量）執行的歷史回測結果。\n"
                            "你在分析時必須參考這些數據，結論必須與回測結果一致。\n"
                            "如果所有做多策略回測均虧損，不可建議做多；反之亦然。\n"
                            "如果所有策略均虧損，結論必須為「觀望」或「僅適合小倉位試單」。\n\n"
                            f"{_pre_bt_summary}"
                        ),
                    })
                    messages.insert(-1, {
                        "role": "assistant",
                        "content": "已收到回測數據，我會根據回測結果調整分析方向。",
                    })
                    logger.info(f"預回測完成，已注入 {len(_pre_bt_summary)} 字元上下文")
            except Exception as e:
                logger.warning(f"預回測失敗（不影響分析）：{e}")

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

            # v130: comprehensive_analysis 強制 call function 兜底（最多重試 1 次）
            # 若 LLM 第一輪沒 call function 卻寫了實質內容，必然走純文字單段路徑、
            # 跳過所有 v129 的完整性保護機制（seg1/seg2 / 段落驗證 / function 補齊），
            # 且 LLM 在沒 function 數據下會編造 winrate/PF/IC 數字（fact_checker 抓到的場景）。
            if (
                "comprehensive_analysis" in _intents
                and not response.function_calls
                and len(response.message or "") > 200
            ):
                logger.warning(
                    f"[v130] comprehensive_analysis 第一輪 function_calls=0 "
                    f"(text_len={len(response.message)}), 強制重試 1 次"
                )
                yield _sse_event("status", {
                    "message": "[重試] LLM 未呼叫必要函式，重新請求中...",
                })
                _retry_messages = list(messages) + [
                    {"role": "assistant", "content": (response.message or "")[:500]},
                    {"role": "user", "content": (
                        "[系統強制] 你的上一輪回覆沒有呼叫任何 function，"
                        "但 comprehensive_analysis 模式必須先呼叫以下函式取得真實數據："
                        "detect_smc_structure、generate_scenarios、analyze_momentum、"
                        "scan_conditional_probability、run_quant_research、compute_laddered_entries。"
                        "**請現在呼叫這些 function**，再根據結果重寫完整分析。"
                        "**禁止在沒有 function 結果的情況下編造任何具體數字**"
                        "（winrate / PF / IC / Sharpe / MDD / RSI 數值 等）。"
                    )},
                ]
                _retry_task = asyncio.create_task(adapter.chat(
                    _retry_messages, chart_state=request.chart_state,
                    system_prompt=_dynamic_prompt,
                    chart_screenshot=request.chart_screenshot,
                ))
                _active_tasks.append(_retry_task)
                _hb_sec = 0
                while not _retry_task.done():
                    await asyncio.sleep(3)
                    _hb_sec += 3
                    yield _sse_event("status", {
                        "message": f"重試中... ({_hb_sec}秒)",
                    })
                _retry_response = _retry_task.result()
                logger.info(
                    f"[v130] 重試結果: 文字長度={len(_retry_response.message)}, "
                    f"function_calls={len(_retry_response.function_calls)}"
                )
                if _retry_response.usage:
                    if total_usage:
                        total_usage.prompt_tokens += _retry_response.usage.prompt_tokens
                        total_usage.completion_tokens += _retry_response.usage.completion_tokens
                        total_usage.total_tokens += _retry_response.usage.total_tokens
                    else:
                        total_usage = _retry_response.usage
                # 無論重試是否成功都用結果取代（避免無限循環）
                response = _retry_response

            # 3. 如果 LLM 回傳了 function calls → 執行 → 二輪回應
            if response.function_calls:
                # 第一輪若 LLM 同時寫了實質內容（例如 #1-#6 初稿），當 seg1 串給前端。
                # 早期版本會丟棄此段假設它是「佔位文字」，但 LLM 行為不保證，
                # 直接 stream 是穩健契約。同時不放入 round2_messages（見下方 v129）
                # 以免 seg2 LLM 看到後從中段續寫、漏掉前半。
                if response.message:
                    yield _sse_event("token", {
                        "content": response.message,
                        "segment": 1,
                    })
                    if len(final_text) < _MAX_FINAL_TEXT:
                        final_text += response.message[:_MAX_FINAL_TEXT - len(final_text)]

                # 發送 function calls 事件
                yield _sse_event("function", {"function_calls": response.function_calls})
                yield _sse_event("status", {"message": "正在執行圖表操作..."})

                # 執行 function calls（無超時限制，以進度百分比回報）
                # ★ 跑在 worker thread 避免 ML/回測等重 sync 操作阻塞主 event loop
                try:
                    _fc_progress = ProgressTracker()
                    _fc_task = asyncio.create_task(asyncio.to_thread(
                        _execute_function_calls_in_thread,
                        response.function_calls, chart_state=request.chart_state,
                        progress=_fc_progress,
                    ))
                    _active_tasks.append(_fc_task)
                    _hb_sec = 0
                    _last_pct = -1
                    _last_status_text = ""
                    while not _fc_task.done():
                        await asyncio.sleep(2)
                        _hb_sec += 2
                        _pct = _fc_progress.percentage
                        _status = _fc_progress.status_text
                        # v110：每 2 秒就強制發 progress（從 6s 提高），確保 SSE 連線在長 function call 期間
                        # 也不會因 idle 被 OS / browser 任何層判定假死而斷
                        _last_pct = _pct
                        # v121：心跳訊息附加累積耗時，且若 status_text 卡在同一句 ≥ 10 秒就加「⏳ 計算中」標記
                        # → 用戶能明確看到「系統還在跑」，避免誤以為畫面凍結而切標的
                        if _status == _last_status_text and _hb_sec > 0 and _hb_sec % 10 == 0:
                            _status_with_time = f"⏳ {_status} ({_hb_sec}s — 系統仍在計算中)"
                        else:
                            _status_with_time = f"{_status} ({_hb_sec}s)" if _status else f"分析中... ({_hb_sec}s)"
                        _last_status_text = _status
                        yield _sse_event("progress", {
                            "percentage": _pct,
                            "completed": _fc_progress.completed,
                            "total": _fc_progress.total,
                            "current_task": _fc_progress.current_task,
                            "message": _status_with_time,
                        })
                        yield _sse_event("status", {"message": _status_with_time})
                    exec_result = _fc_task.result()
                    yield _sse_event("progress", {
                        "percentage": 100,
                        "completed": _fc_progress.total,
                        "total": _fc_progress.total,
                        "current_task": "",
                        "message": "分析完成 (100%)",
                    })
                    logger.info(f"Function call 執行完成 ({_hb_sec}s)，結果數: {len(exec_result.get('results', []))}")

                    # （預回測已在 LLM 呼叫前完成，此處不再需要事後回測注入）
                    _llm_called_funcs = {fc.get("name") for fc in response.function_calls} if response.function_calls else set()

                    # ★ 分析意圖必要函式補齊：檢查 prompt 要求的函式是否已執行
                    # 注意：comprehensive_analysis 放最前面（dict 迭代順序 = 插入順序，loop 用 break）
                    _REQUIRED_ANALYSIS_FUNCS: dict[str, list[str]] = {
                        "comprehensive_analysis": [
                            "detect_smc_structure", "generate_scenarios",
                            "analyze_momentum", "compare_strategies",
                            "scan_conditional_probability", "run_quant_research",
                            "compute_laddered_entries",  # ★ v99：分批進場價位
                        ],
                        "analysis": ["generate_scenarios", "detect_smc_structure"],
                        "deep_phase1": ["generate_scenarios", "detect_smc_structure"],
                        "deep_analysis": ["generate_scenarios", "detect_smc_structure"],
                        "deep_phase2": ["compare_strategies", "scan_conditional_probability"],
                        "deep_phase3": ["run_quant_research"],
                        "factor_validation": ["run_quant_research"],
                        "strategy_backtest": ["run_quant_research"],
                        "regime_analysis": ["generate_scenarios"],
                        "momentum_analysis": ["analyze_momentum"],
                        "fundamental_analysis": ["analyze_fundamentals"],
                    }
                    _already_executed = _llm_called_funcs | {
                        r.get("function") for r in exec_result.get("results", []) if isinstance(r, dict)
                    }
                    _missing_funcs: list[str] = []
                    for _ik, _rf in _REQUIRED_ANALYSIS_FUNCS.items():
                        if _ik in _intents:
                            _missing_funcs = [fn for fn in _rf if fn not in _already_executed]
                            break

                    if _missing_funcs and chart_symbol:
                        logger.info(f"自動補齊缺少的分析函式：{_missing_funcs}")
                        yield _sse_event("status", {"message": f"正在補充分析數據..."})
                        _fill_calls = [
                            {"name": fn, "arguments": {"symbol": chart_symbol, "timeframe": chart_timeframe_ctx or "1d"}}
                            for fn in _missing_funcs
                        ]
                        try:
                            # 補齊函式可能含 run_quant_research 等重操作，
                            # 同樣丟 worker thread 跑 + 主 loop 心跳
                            _fill_task = asyncio.create_task(asyncio.to_thread(
                                _execute_function_calls_in_thread,
                                _fill_calls, chart_state=request.chart_state,
                            ))
                            _active_tasks.append(_fill_task)
                            _fill_hb = 0
                            while not _fill_task.done():
                                await asyncio.sleep(5)
                                _fill_hb += 5
                                yield _sse_event("status", {
                                    "message": f"正在補齊缺失分析函式... ({_fill_hb}秒，{', '.join(_missing_funcs)})"
                                })
                            _fill_result = _fill_task.result()
                            if "results" in _fill_result and _fill_result["results"]:
                                _fitems = _fill_result["results"] if isinstance(_fill_result["results"], list) else [_fill_result["results"]]
                                exec_result.setdefault("results", []).extend(_fitems)
                            logger.info(f"補齊完成：{_missing_funcs}")
                        except Exception as e:
                            logger.warning(f"補齊分析函式失敗（不影響主流程）：{e}")

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
                    # v129：不把第一輪 assistant 文字放進 round2。若放會讓 seg2 LLM
                    # 看到 #1-#6 已寫過、自行從 #7 開始續寫，造成前端看到「跳段」。
                    # 第一輪文字已透過 token event(segment=1) 串給前端，這裡只給 seg2
                    # 看 user 原文 + tool 結果，由它完整重寫 #1-#11。
                    _r2_func_rule = (
                        "你可以繼續呼叫分析函式補充數據（如 detect_smc_structure、generate_scenarios 等），"
                        "也可以呼叫 annotate_chart、draw_pattern、manage_indicator 進行圖表繪製。"
                    ) if (_intents & {"analysis", "deep_analysis", "deep_phase1", "deep_phase2", "deep_phase3"}) else (
                        "除了 annotate_chart、draw_pattern 和 manage_indicator 以外，不要呼叫其他函式。"
                    )
                    round2_messages.append({
                        "role": "user",
                        "content": (
                            f"[系統自動回傳] 以下是你剛才呼叫的函式執行結果：\n\n"
                            f"{fc_summary}\n\n"
                            f"請根據以上數據結果回答使用者的問題。\n"
                            f"如果使用者要求在圖表上畫線、標記、型態等，你**必須**呼叫 annotate_chart 或 draw_pattern 函式來繪製。\n"
                            f"你也可以呼叫 manage_indicator 來添加分析中用到的指標到圖表上。\n"
                            f"{_r2_func_rule}"
                        ),
                    })

                    # 第二輪 LLM 呼叫（真串流：邊收 token 邊 yield 給前端）
                    yield _sse_event("status", {"message": "AI 正在撰寫分析報告..."})

                    _r2_text_buf = ""
                    _r2_function_calls: list[dict] = []
                    _r2_usage = None
                    _r2_stop_reason = "end_turn"
                    _r2_yielded_until = 0
                    _r2_marker_seen = False
                    _r2_streamed = True  # 旗標：表示文字已經邊串流邊 yield 過，後續顯示段不要重複
                    # v125-B: 追蹤每段結束時 _r2_text_buf 的長度，用於計算每段字數做完整性驗證
                    _seg_end_positions: dict[int, int] = {}

                    # KEY_INSIGHTS / PREDICTIONS / SYSTEM_DISTILL 標記偵測（不 yield 給使用者）
                    _MARKER_RE = re.compile(r'\n---(?:KEY_INSIGHTS|PREDICTIONS|SYSTEM_DISTILL)---')

                    # S1: 分段判定 — 「全部分析」拆成兩段（30 秒結論卡 + 完整詳細分析）
                    # 其他 intent 走原單段邏輯（segment=0）
                    # v132: comprehensive_pipeline flag 開 → seg1 照跑、seg2 monolith 改走編排管線
                    _is_comprehensive = "comprehensive_analysis" in _intents
                    _use_pipeline = _is_comprehensive and settings.comprehensive_pipeline_enabled
                    if _use_pipeline:
                        _segments_to_run = [1]            # seg1 仍由本迴圈跑，seg2 交給 pipeline
                    elif _is_comprehensive:
                        _segments_to_run = [1, 2]
                    else:
                        _segments_to_run = [0]
                    _seg_labels = {
                        0: "完整分析",
                        1: "第 1 段（核心結論）",
                        2: "第 2 段（完整詳細分析）",
                    }

                    for _seg_idx, _seg_no in enumerate(_segments_to_run):
                        # 每段重組 system_prompt（segment=1/2 會替換 comprehensive_analysis 模組）
                        if _seg_no == 0:
                            _seg_prompt = _dynamic_prompt
                        else:
                            _seg_prompt = assemble_system_prompt(
                                _intents, teaching_mode=_teaching, segment=_seg_no,
                            )

                        # 第 2 段加「接續」hint，讓 LLM 知道這是 continuation
                        if _seg_no == 2:
                            _seg_messages = list(round2_messages) + [{
                                "role": "user",
                                "content": (
                                    "（系統提示）請輸出第 2 段：完整詳細分析。"
                                    "在開頭加「（接續第 1 段）」標示。"
                                    "**不可重複 30 秒結論卡與分批進場表**（已在第 1 段輸出）。"
                                    "依序輸出全 12 段（v123 強制）："
                                    "#1 市場環境（含跨市場群體 + 衍生品矩陣完整 funding/OI/OB/LS/premium/fear_greed）"
                                    " → #2 結構分析 → #3 動能特徵（含條件機率掃描 bin）"
                                    " → #4 多策略回測比較（具體 PF/Sharpe/MDD/MC/CPCV）"
                                    " → #5 量化研究（IC + Decay + WF + MC）"
                                    " → #5.5 Alpha 動態監控分級表（🟢/⬆️/⬇️/❌/👁️/❓）"
                                    " → #6 跨維度交叉驗證表 → #6.5 RL 戰略結論"
                                    " → #7 正式結論卡 → #8 風險清單（含嚴重度欄）"
                                    " → #9 延伸學習 → #10 摘要表 → #11 4 策略附錄。"
                                    "**任一段資料不可得時必須輸出段落標題 + 「⚠️ 資料不可得：[原因]」，禁止省略段落**。"
                                    "**禁止編造任何具體數字**（RSI/winrate/PF/IC/LS ratio 等），所有數字必須對應 chart_state 欄位或 function call 結果，否則寫「無資料」。"
                                    "撰寫前先檢查 chart_state.data_status 是否標記該段對應欄位為非 ok。"
                                ),
                            }]
                        else:
                            _seg_messages = round2_messages

                        # 進度事件 — 每段啟動時通知前端
                        # R2: 立即 emit，避免用戶看長時間空白誤觸發 abort（v118-v120 chart_state 增大後 TTFT 拉長的根因）
                        yield _sse_event("status", {
                            "message": f"完整分析生成中（{_seg_labels[_seg_no]}）...",
                        })

                        # 每段重置 marker_seen，避免第一段意外標記吞掉第二段內容
                        _r2_marker_seen = False
                        import time as _time_mod
                        _seg_t0 = _time_mod.time()

                        # ★★★ v122 修正：重新啟用 _stream_with_heartbeat ★★★
                        #
                        # v110 註解說「心跳對 client 真斷線無效」是誤判 — 真正的根因是：
                        # 第 1 段 → 第 2 段切換時，adapter.chat_stream_events 內部會 spawn
                        # 新 Claude CLI subprocess（5-30s 啟動 + LLM TTFT），這段時間 SSE 完全 idle
                        # → 中間層（瀏覽器 HTTP/2 idle timeout / WiFi 路由器）判定 idle 斷線
                        # → uvicorn 偵測到 client_disconnect → cancel ASGI task
                        # → 用戶看到「分析跑到一半停了」（即使用戶什麼都沒動）
                        #
                        # _stream_with_heartbeat 用 asyncio.shield 保護 pending task，
                        # timeout 取消的是 shield wrapper、不是 LLM adapter，所以不會殺 LLM。
                        # 設計正確（line 1684-1727），v110 回退是錯的。
                        async for _evt in _stream_with_heartbeat(adapter.chat_stream_events(
                            _seg_messages, chart_state=request.chart_state,
                            system_prompt=_seg_prompt,
                            r2_mode=True,
                        ), interval=2.0):
                            if _evt is _HEARTBEAT_SENTINEL:
                                # 心跳：每 2 秒 emit SSE status 維持連線活躍、避免中間層判 idle
                                yield _sse_event("status", {
                                    "message": f"⏳ {_seg_labels[_seg_no]}生成中... ({int(_time_mod.time()-_seg_t0)}s)",
                                })
                                continue
                            if _evt.type == "text_delta":
                                _r2_text_buf += _evt.text
                                if not _r2_marker_seen:
                                    # v129：限制 marker 只在文末 200 字內搜尋
                                    # 避免 LLM 在中段（如 #5/#6）意外輸出 ---KEY_INSIGHTS---
                                    # 把後續所有段落（#7-#11）全吞掉
                                    _marker_search_start = max(
                                        _r2_yielded_until,
                                        len(_r2_text_buf) - 200,
                                    )
                                    _m = _MARKER_RE.search(_r2_text_buf, _marker_search_start)
                                    if _m:
                                        # yield 標記之前的文字，然後停止 yield（本段內）
                                        if _m.start() > _r2_yielded_until:
                                            yield _sse_event("token", {
                                                "content": _r2_text_buf[_r2_yielded_until:_m.start()],
                                                "segment": _seg_no,
                                            })
                                        _r2_yielded_until = len(_r2_text_buf)
                                        _r2_marker_seen = True
                                    else:
                                        # 保留尾端 30 字（可能正在形成標記）
                                        _safe = max(_r2_yielded_until, len(_r2_text_buf) - 30)
                                        if _safe > _r2_yielded_until:
                                            yield _sse_event("token", {
                                                "content": _r2_text_buf[_r2_yielded_until:_safe],
                                                "segment": _seg_no,
                                            })
                                            _r2_yielded_until = _safe
                            elif _evt.type == "function_call":
                                _r2_function_calls.append(_evt.function_call)
                            elif _evt.type == "usage":
                                # 跨段累積 usage
                                if _r2_usage:
                                    _r2_usage.prompt_tokens += _evt.usage.prompt_tokens
                                    _r2_usage.completion_tokens += _evt.usage.completion_tokens
                                    _r2_usage.total_tokens += _evt.usage.total_tokens
                                else:
                                    _r2_usage = _evt.usage
                            elif _evt.type == "stop":
                                _r2_stop_reason = _evt.stop_reason

                        # 每段結束 flush 剩餘文字（若沒看過標記）
                        if not _r2_marker_seen and _r2_yielded_until < len(_r2_text_buf):
                            yield _sse_event("token", {
                                "content": _r2_text_buf[_r2_yielded_until:],
                                "segment": _seg_no,
                            })
                            _r2_yielded_until = len(_r2_text_buf)

                        # v125-B: 記錄本段結束時的 _r2_text_buf 長度（給 seg2 字數驗證用）
                        _seg_end_positions[_seg_no] = len(_r2_text_buf)

                        # 段間：emit segment_complete（除最後一段）
                        # 前端可據此插入「【第 N 段：xxx】」分隔或新建 message bubble
                        if _seg_no >= 1 and _seg_idx < len(_segments_to_run) - 1:
                            _next_seg = _segments_to_run[_seg_idx + 1]
                            yield _sse_event("segment_complete", {
                                "segment": _seg_no,
                                "next_segment": _next_seg,
                                "next_label": _seg_labels.get(_next_seg, ""),
                            })

                            # v125-B → v129: seg1→seg2 切換 SSE idle 風險點 — 主動 emit 心跳
                            # adapter.chat_stream_events 對 seg2 會 spawn 新 Claude CLI subprocess
                            # （5-30s 啟動 + LLM TTFT），這段 SSE idle 是 v122 心跳機制覆蓋不到的縫
                            # （_stream_with_heartbeat 只在 adapter 已啟動後生效）。
                            # v129：心跳次數從 3 拉到 20（4.5s → 30s），覆蓋 subprocess 最壞啟動情況
                            for _hb_i in range(20):
                                yield _sse_event("status", {
                                    "message": f"準備第 {_next_seg} 段（{_seg_labels.get(_next_seg, '完整詳細分析')}）... ({_hb_i+1}/20)",
                                    "phase": "seg_warmup",
                                })
                                await asyncio.sleep(1.5)

                    # ═══ v132 編排管線：seg1 之後跑 map-reduce pipeline 取代 seg2 monolith ═══
                    # 5 維度 focused call（並行）+ synthesis call。每維度品質對齊「單獨問」。
                    # pipeline 自行 emit status/token/segment_complete/warning，並以
                    # ("_pipeline_result", {...}) 終結事件帶回全文與 usage。
                    if _use_pipeline:
                        _seg1_card_text = _r2_text_buf  # seg1 是本迴圈唯一跑過的段
                        _pipeline_full_text = ""
                        try:
                            from app.core.llm.comprehensive_pipeline import run_pipeline
                            async for _evt_type, _payload in run_pipeline(
                                adapter=adapter,
                                user_message=_user_original,
                                chart_state=request.chart_state,
                                function_calls=response.function_calls,
                                exec_result=exec_result,
                                seg1_card_text=_seg1_card_text,
                                pre_backtest_summary=locals().get("_pre_bt_summary", "") or "",
                                teaching_mode=_teaching,
                                chart_symbol=chart_symbol or "",
                                chart_timeframe=chart_timeframe_ctx or "",
                            ):
                                if _evt_type == "_pipeline_result":
                                    _pipeline_full_text = _payload.get("full_text", "") or ""
                                    _pl_usage = _payload.get("usage") or {}
                                    if _pl_usage and total_usage:
                                        total_usage.prompt_tokens += _pl_usage.get("prompt_tokens", 0)
                                        total_usage.completion_tokens += _pl_usage.get("completion_tokens", 0)
                                        total_usage.total_tokens += _pl_usage.get("total_tokens", 0)
                                    continue
                                yield _sse_event(_evt_type, _payload)
                        except Exception as _pl_err:
                            import traceback as _pl_tb
                            logger.error(
                                f"[v132 pipeline] 執行失敗: {_pl_err}\n{_pl_tb.format_exc()}"
                            )
                            yield _sse_event("warning", {
                                "message": f"編排管線執行失敗：{_pl_err}",
                                "type": "pipeline_failed",
                            })
                        # pipeline 全文併入 _r2_text_buf（給持久化 / fact-check / 預測追蹤用）
                        if _pipeline_full_text:
                            _r2_text_buf += "\n\n" + _pipeline_full_text
                        logger.info(
                            f"[v132 pipeline] 完成，全文長度={len(_pipeline_full_text)}，"
                            f"_r2_text_buf 總長={len(_r2_text_buf)}"
                        )

                    # v125-B: seg2 字數完整性檢查
                    # 預期 seg2 ≥ 4000 字（涵蓋 12 段詳細分析）；不足表示 streaming 中斷或 LLM 早收
                    if 2 in _segments_to_run and 2 in _seg_end_positions:
                        _seg1_end = _seg_end_positions.get(1, 0)
                        _seg2_len = _seg_end_positions[2] - _seg1_end
                        if _seg2_len < 4000:
                            logger.warning(
                                f"[v125-B] seg2 字數過短 {_seg2_len} < 4000，疑似 streaming 中斷或 LLM 早收"
                            )
                            yield _sse_event("warning", {
                                "message": f"第 2 段內容可能不完整（{_seg2_len} 字、預期 ≥ 4000）",
                                "type": "seg2_incomplete",
                                "seg2_len": _seg2_len,
                            })
                        else:
                            logger.info(f"[v125-B] seg2 字數驗證通過：{_seg2_len} 字 ≥ 4000")

                        # v129: seg2 段落完整性檢測 — 字數夠但段落仍可能缺
                        # 用「#N」標題正則掃描 seg2，比對預期 12 段（含 #5.5、#6.5）
                        _seg2_text_only = _r2_text_buf[_seg1_end:]
                        _section_pattern = re.compile(r'(?:^|\n)\s*#\s*(\d+(?:\.5)?)\b')
                        _found_sections = set(_section_pattern.findall(_seg2_text_only))
                        _expected_sections = {'1', '2', '3', '4', '5', '5.5', '6', '6.5', '7', '8', '9', '10', '11'}
                        _missing_sections = _expected_sections - _found_sections
                        if _missing_sections:
                            _missing_sorted = sorted(_missing_sections, key=lambda x: float(x))
                            _found_sorted = sorted(_found_sections, key=lambda x: float(x) if x.replace('.', '').isdigit() else 99)
                            logger.warning(
                                f"[seg2 完整性] 缺少段落: {_missing_sorted}（found={_found_sorted}）"
                            )
                            yield _sse_event("warning", {
                                "message": f"分析報告缺少段落 #{', #'.join(_missing_sorted)}",
                                "type": "seg2_missing_sections",
                                "missing": _missing_sorted,
                                "found": _found_sorted,
                            })
                        else:
                            logger.info(f"[seg2 完整性] 全 13 段（含 #5.5、#6.5）皆到齊")

                    # 建構 response2 物件相容後續既有邏輯
                    class _StreamedResponse:
                        pass
                    response2 = _StreamedResponse()
                    response2.message = _r2_text_buf
                    response2.function_calls = _r2_function_calls
                    response2.usage = _r2_usage
                    response2.stop_reason = _r2_stop_reason

                    logger.info(f"第二輪 LLM 完成 (streaming)，文字長度={len(_r2_text_buf)}")

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

                        _r2_draw_funcs = {"annotate_chart", "draw_pattern", "manage_indicator"}
                        _is_analysis_intent = bool(_intents & {"analysis", "deep_analysis", "deep_phase1", "deep_phase2", "deep_phase3"})
                        allowed_r2 = [
                            fc for fc in response2.function_calls
                            if fc.get("name") in _r2_draw_funcs or _is_analysis_intent
                        ]
                        if allowed_r2:
                            _r2_has_analysis = any(fc.get("name") not in _r2_draw_funcs for fc in allowed_r2)
                            _r2_status = "正在補充分析數據並更新圖表..." if _r2_has_analysis else "正在更新圖表..."
                            yield _sse_event("status", {"message": _r2_status})
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
                                # 如果第二輪有分析函式，將結果追加到摘要供第三輪使用
                                if _r2_has_analysis and exec_result2.get("results"):
                                    _r2_analysis_summary = _format_function_results(
                                        [fc for fc in allowed_r2 if fc.get("name") not in _r2_draw_funcs],
                                        exec_result2,
                                    )
                                    round2_messages.append({
                                        "role": "user",
                                        "content": f"[系統補充] 以下是你剛才補充呼叫的函式結果：\n\n{_r2_analysis_summary}\n\n請將這些數據整合到你的回覆中。",
                                    })
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

                        # 第三輪 LLM 呼叫（真串流）
                        _r3_text_buf = ""
                        _r3_usage = None
                        _r3_stop_reason = "end_turn"
                        _r3_yielded_until = 0
                        _r3_marker_seen = False
                        _r2_streamed = True  # 標記第二/三輪文字已即時 yield，避免後段重複

                        # v110 回退：raw streaming
                        async for _evt in adapter.chat_stream_events(
                            round3_messages, chart_state=request.chart_state,
                            force_text=True, system_prompt=_dynamic_prompt,
                        ):
                            if _evt.type == "text_delta":
                                _r3_text_buf += _evt.text
                                if not _r3_marker_seen:
                                    _m = _MARKER_RE.search(_r3_text_buf, _r3_yielded_until)
                                    if _m:
                                        if _m.start() > _r3_yielded_until:
                                            yield _sse_event("token", {"content": _r3_text_buf[_r3_yielded_until:_m.start()]})
                                        _r3_yielded_until = len(_r3_text_buf)
                                        _r3_marker_seen = True
                                    else:
                                        _safe = max(_r3_yielded_until, len(_r3_text_buf) - 30)
                                        if _safe > _r3_yielded_until:
                                            yield _sse_event("token", {"content": _r3_text_buf[_r3_yielded_until:_safe]})
                                            _r3_yielded_until = _safe
                            elif _evt.type == "usage":
                                _r3_usage = _evt.usage
                            elif _evt.type == "stop":
                                _r3_stop_reason = _evt.stop_reason
                            # function_call 在 force_text=True 下不應該出現，忽略

                        # Flush 剩餘
                        if not _r3_marker_seen and _r3_yielded_until < len(_r3_text_buf):
                            yield _sse_event("token", {"content": _r3_text_buf[_r3_yielded_until:]})

                        # 建構 response3 物件兼容後續邏輯
                        class _StreamedR3:
                            pass
                        response3 = _StreamedR3()
                        response3.message = _r3_text_buf
                        response3.function_calls = []
                        response3.usage = _r3_usage
                        response3.stop_reason = _r3_stop_reason

                        if response3.usage and total_usage:
                            total_usage.prompt_tokens += response3.usage.prompt_tokens
                            total_usage.completion_tokens += response3.usage.completion_tokens
                            total_usage.total_tokens += response3.usage.total_tokens
                        elif response3.usage:
                            total_usage = response3.usage

                        _r2_text = response3.message or ""
                        logger.info(f"第三輪結果 (streaming): 文字長度={len(_r2_text)}")

                    # ★★ 截斷偵測 + 自動續寫 ★★
                    # 判斷最終回應是否被 token 上限截斷
                    _final_response = response3 if (not (response2.message or "").strip() and response2.function_calls) else response2
                    _was_truncated = getattr(_final_response, "stop_reason", "end_turn") in ("length", "max_tokens")

                    if _was_truncated and _r2_text.strip():
                        logger.warning(f"LLM 回覆被截斷（stop_reason={_final_response.stop_reason}），啟動自動續寫")
                        # 先串流第一段 + 提示
                        _part1 = strip_system_distill(strip_predictions(strip_key_insights(_r2_text)))
                        _part1 += "\n\n⏳ **分析內容較長，後續正在生成...**\n"
                        for chunk in _split_text_for_streaming(_part1):
                            yield _sse_event("token", {"content": chunk})
                            await asyncio.sleep(0.02)

                        # 追加續寫呼叫
                        yield _sse_event("status", {"message": "正在生成後續分析..."})
                        _cont_messages = list(round2_messages)
                        _cont_messages.append({"role": "assistant", "content": _r2_text})
                        _cont_messages.append({
                            "role": "user",
                            "content": "你的回覆被截斷了，請從截斷處繼續完成分析。不要重複已說過的內容，直接接續。",
                        })
                        # 續寫呼叫（真串流）
                        _cont_text_buf = ""
                        _cont_usage = None
                        _cont_yielded_until = 0
                        _cont_marker_seen = False

                        # v110 回退：raw streaming
                        async for _evt in adapter.chat_stream_events(
                            _cont_messages, chart_state=request.chart_state,
                            force_text=True, system_prompt=_dynamic_prompt,
                        ):
                            if _evt.type == "text_delta":
                                _cont_text_buf += _evt.text
                                if not _cont_marker_seen:
                                    _m = _MARKER_RE.search(_cont_text_buf, _cont_yielded_until)
                                    if _m:
                                        if _m.start() > _cont_yielded_until:
                                            yield _sse_event("token", {"content": _cont_text_buf[_cont_yielded_until:_m.start()]})
                                        _cont_yielded_until = len(_cont_text_buf)
                                        _cont_marker_seen = True
                                    else:
                                        _safe = max(_cont_yielded_until, len(_cont_text_buf) - 30)
                                        if _safe > _cont_yielded_until:
                                            yield _sse_event("token", {"content": _cont_text_buf[_cont_yielded_until:_safe]})
                                            _cont_yielded_until = _safe
                            elif _evt.type == "usage":
                                _cont_usage = _evt.usage

                        if not _cont_marker_seen and _cont_yielded_until < len(_cont_text_buf):
                            yield _sse_event("token", {"content": _cont_text_buf[_cont_yielded_until:]})

                        if _cont_usage and total_usage:
                            total_usage.prompt_tokens += _cont_usage.prompt_tokens
                            total_usage.completion_tokens += _cont_usage.completion_tokens
                            total_usage.total_tokens += _cont_usage.total_tokens

                        _cont_text = _cont_text_buf
                        if _cont_text.strip():
                            logger.info(f"續寫完成 (streaming): 文字長度={len(_cont_text)}")
                            _r2_text += "\n" + _cont_text  # 合併完整文字

                        if len(final_text) < _MAX_FINAL_TEXT:
                            final_text += _r2_text[:_MAX_FINAL_TEXT - len(final_text)]

                    # 串流文字回應（移除 KEY_INSIGHTS / PREDICTIONS + JSON 過濾）
                    elif _r2_text.strip():
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
                        _draw_warning_suffix = ""
                        if _text_claims_draw and not _has_draw_calls:
                            logger.warning("LLM 文字聲稱繪圖但未產生 annotate_chart/draw_pattern function call")
                            _draw_warning_suffix = (
                                "\n\n⚠️ **注意**：AI 描述了繪圖操作，但未成功產生繪圖指令。"
                                "如果圖表上沒有看到標記，請重新描述您希望畫的內容，"
                                "例如：「請在圖表上畫出趨勢線」。"
                            )
                            display_msg2 += _draw_warning_suffix

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

                        # 真串流模式：主文字已經邊收邊 yield，這裡只 yield 後續 append 的警告
                        if locals().get("_r2_streamed"):
                            if _draw_warning_suffix:
                                yield _sse_event("token", {"content": _draw_warning_suffix})
                        else:
                            # 假串流 fallback：把整段 display_msg2 切碎慢慢 yield（舊行為）
                            for chunk in _split_text_for_streaming(display_msg2):
                                yield _sse_event("token", {"content": chunk})
                                await asyncio.sleep(0.02)
                    else:
                        logger.error("所有輪次都未產生文字回應")
                        yield _sse_event("token", {
                            "content": "⚠️ AI 分析完成但未能產生文字報告，請嘗試重新提問。"
                        })

                except asyncio.CancelledError:
                    # v114 bug 修正：以前 except (Exception, CancelledError) 一起抓會吞掉 cancel,
                    # 導致外層 line 3091 的 partial-save 永遠不會被觸發、用戶 r2_text_buf 內容遺失。
                    # 現在拆開：CancelledError re-raise 給外層處理 partial-save，
                    # 真實 exception 仍在下方 Exception 分支處理。
                    import traceback
                    _r2_len = len(locals().get("_r2_text_buf", "") or "")
                    _r2_fc_count = len(locals().get("_r2_function_calls", []) or [])
                    _has_response = "response" in locals()
                    _round2_started = "round2_messages" in locals()
                    logger.warning(
                        f"[client_disconnect] HTTP 連線在 stream_gen 內被 cancel\n"
                        f"  狀態：round2_started={_round2_started}, "
                        f"r2_text_len={_r2_len}, r2_function_calls={_r2_fc_count}, "
                        f"has_response={_has_response}\n"
                        f"  常見原因：(1) 用戶切換 symbol/timeframe (2) browser tab 斷連 "
                        f"(3) 前端 abort 其他原因\n"
                        f"  → 將 re-raise 給外層 partial-save 把 r2_text_buf 存進 DB\n"
                        f"{traceback.format_exc()}"
                    )
                    raise
                except Exception as e:
                    import traceback
                    logger.error(f"Function call 執行或二輪回應失敗: {e}\n{traceback.format_exc()}")
                    err_msg = str(e) or "未知錯誤"
                    yield _sse_event("token", {
                        "content": f"\n\n⚠️ 分析報告產生失敗: {err_msg}\n請嘗試重新提問，若持續發生請檢查後端日誌。"
                    })
                    yield _sse_event("error", {"error": f"指令執行失敗: {err_msg}"})

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

                    # v130: 純文字單段路徑下對 comprehensive_analysis 做完整性檢測
                    # （走到這代表方案 B 重試後 LLM 仍不 call function，至少留下警告線索）
                    if "comprehensive_analysis" in _intents and display_text:
                        if len(display_text) < 4000:
                            logger.warning(
                                f"[v130 純文字] comprehensive_analysis 字數過短 "
                                f"{len(display_text)} < 4000，疑似 LLM 早收"
                            )
                            yield _sse_event("warning", {
                                "message": f"完整分析內容偏短（{len(display_text)} 字、預期 ≥ 4000）",
                                "type": "plain_text_too_short",
                                "text_len": len(display_text),
                            })
                        _section_pattern = re.compile(r'(?:^|\n)\s*#\s*(\d+(?:\.5)?)\b')
                        _found_sections = set(_section_pattern.findall(display_text))
                        # 純文字單段走的是舊 6 段結構（function_defs.py:1576-1582）
                        _expected_sections = {'1', '2', '3', '4', '5', '6'}
                        _missing_sections = _expected_sections - _found_sections
                        if _missing_sections:
                            _missing_sorted = sorted(_missing_sections, key=lambda x: float(x))
                            logger.warning(
                                f"[v130 純文字] comprehensive_analysis 缺段落: {_missing_sorted}"
                            )
                            yield _sse_event("warning", {
                                "message": f"分析報告缺少段落 #{', #'.join(_missing_sorted)}",
                                "type": "plain_text_missing_sections",
                                "missing": _missing_sorted,
                            })

            # 4a. 三階段完整分析：程式化強制附加接續提示（不依賴 LLM）
            _deep_reminder = ""
            if "deep_phase1" in _intents or "deep_analysis" in _intents:
                _deep_reminder = (
                    "\n\n---\n"
                    "📋 **完整分析進度：[1/3]**\n"
                    "✅ 第一階段完成：市場環境 + 情境預測 + SMC 結構\n"
                    "➡️ 輸入「**完整分析二**」→ 多策略回測驗證 + 條件機率掃描\n"
                    "➡️ 輸入「**完整分析三**」→ 量化研究 + Monte Carlo + 倉位管理\n"
                    "---"
                )
            elif "deep_phase2" in _intents:
                _deep_reminder = (
                    "\n\n---\n"
                    "📋 **完整分析進度：[2/3]**\n"
                    "✅ 第一階段：市場環境 + 情境預測 + SMC 結構\n"
                    "✅ 第二階段完成：多策略回測 + 條件機率\n"
                    "➡️ 輸入「**完整分析三**」→ 因子驗證 + Monte Carlo 壓力測試 + 倉位管理\n"
                    "---"
                )
            elif "deep_phase3" in _intents:
                _deep_reminder = (
                    "\n\n---\n"
                    "📋 **完整分析進度：[3/3] — 全部完成**\n"
                    "✅ 第一階段：市場環境 + 情境預測 + SMC 結構\n"
                    "✅ 第二階段：多策略回測 + 條件機率\n"
                    "✅ 第三階段：量化研究 + Monte Carlo + 倉位管理\n"
                    "---"
                )
            if _deep_reminder:
                if "完整分析進度" not in final_text and "完整分析二" not in final_text:
                    for chunk in _split_text_for_streaming(_deep_reminder):
                        yield _sse_event("token", {"content": chunk})
                        await asyncio.sleep(0.02)
                    final_text += _deep_reminder

            # 4b. token 用量 + 持久化記錄
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
            # v114：partial-save — 把已生成的部分內容存進 DB
            # 即使 client 斷線（瀏覽器/系統/網路層原因，外部不可控），
            # 使用者重整頁面也能看到 partial 報告，不必打「請繼續」重跑浪費 token + 8 分鐘
            # 用 sync 呼叫 chat_history.save_message（不 await），避免被 cancel scope 連坐
            #
            # v122 bug 修正：以前 partial-save 只存 final_text，但 round2 內容在 _r2_text_buf
            # → 用戶在 round2 第 2 段被斷線時 2095 字 r2_text_buf 全遺失。
            #   現合併兩者：優先 final_text、若沒則用 _r2_text_buf；若兩者都有則 final_text + r2_text_buf
            _partial_final = locals().get("final_text") or ""
            _partial_r2 = locals().get("_r2_text_buf") or ""
            _partial_conv_id = locals().get("conversation_id") or ""
            # 合併：若 r2_text_buf 內容不在 final_text 內，就 append（避免重複）
            if _partial_r2 and _partial_r2 not in _partial_final:
                _partial_combined = (_partial_final + "\n\n" + _partial_r2).strip() if _partial_final else _partial_r2
            else:
                _partial_combined = _partial_final
            if _partial_combined.strip() and _partial_conv_id:
                try:
                    _partial_clean = (
                        strip_system_distill(strip_predictions(strip_key_insights(_partial_combined)))
                        + "\n\n---\n⚠️ **報告生成中斷**（網路抖動 / 系統省電）— 已存上方部分內容。請輸入「請繼續」接續。"
                    )
                    _partial_total_usage = locals().get("total_usage")
                    _partial_usage_dict = _partial_total_usage.to_dict() if _partial_total_usage else None
                    chat_history.save_message(
                        conversation_id=_partial_conv_id,
                        role="assistant",
                        content=_partial_clean,
                        token_usage=_partial_usage_dict,
                    )
                    logger.info(
                        f"[client_disconnect] partial 內容已存 DB，長度={len(_partial_clean)}"
                        f"（final_text={len(_partial_final)}, r2_text_buf={len(_partial_r2)}）"
                    )
                except Exception as _save_err:
                    logger.error(f"[client_disconnect] partial 存 DB 失敗: {_save_err}")
            return
        except Exception as e:
            logger.error(f"Streaming chat 錯誤: {e}")
            yield _sse_event("error", {"error": str(e)})
        finally:
            # v117：加 timeout 防 event loop 卡死
            # 原 `await t` 沒 timeout，若 task 內部是 asyncio.to_thread(...) 包的
            # CPU-bound sync 操作（pandas backtest / SHAP / Monte Carlo），cancel
            # 訊號等到 thread 跑完才生效。期間整個 event loop 卡住，所有 endpoint
            # 都 timeout（已實測 reproduce）。改成 2 秒 timeout 強制略過。
            for t in _active_tasks:
                if not t.done():
                    t.cancel()
                    try:
                        await asyncio.wait_for(t, timeout=2.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                        # task 不理 cancel（thread 內死循環）也要放手，避免拖死整個 loop
                        pass

        # 4c. v100：結論卡「📈 系統參考」自動注入歷史命中率（必須在 done 之前，讓前端能即時替換）
        try:
            if final_text and "📈" in final_text and "系統參考" in final_text:
                _regime_info = (request.chart_state or {}).get("currentRegime") or {}
                _regime = _regime_info.get("regime", "unknown") if isinstance(_regime_info, dict) else "unknown"
                injected_text = _inject_recent_accuracy(final_text, chart_symbol_for_save or "", _regime)
                if injected_text != final_text:
                    yield _sse_event("accuracy_inject", {
                        "old_pattern": "📈 系統參考：",
                        "new_text": _PLACEHOLDER_PATTERN.search(injected_text).group(0),
                    })
                    final_text = injected_text
        except Exception as e:
            logger.warning(f"accuracy 注入失敗（不影響主流程）：{e}")

        # 4c-2. v104 Q3：LLM 數值編造偵測（fact-checker）
        # v108 Phase 2：mismatch 時附加可見區塊到報告底部（不再只發 SSE event）
        try:
            if final_text and request.chart_state:
                from app.core.fact_checker import check_text_against_chart_state
                fc = check_text_against_chart_state(final_text, request.chart_state)
                if fc.get("checked_count", 0) > 0:
                    yield _sse_event("fact_check", {
                        "checked_count": fc["checked_count"],
                        "mismatches": fc["mismatches"],
                        "summary": fc["summary"],
                    })
                    mismatches = fc.get("mismatches") or []
                    if mismatches:
                        logger.warning(
                            f"[fact_check] {len(mismatches)} mismatches in final_text "
                            f"(checked {fc['checked_count']})"
                        )
                        # v130: 同步發 warning event 讓前端 toast / UI 警告也能聽到
                        # （現有 fact_check event 與下方可見區塊都保留，職責不同：
                        #  fact_check event → 給專用 UI 顯示完整 mismatch 列表；
                        #  warning event → 給通用 toast / status bar 一句話警告；
                        #  可見區塊 → 直接附加到報告底部讓用戶必看到）
                        yield _sse_event("warning", {
                            "message": f"⚠️ 報告含 {len(mismatches)} 處數字與真實數據不符，請務必複核",
                            "type": "fact_check_mismatch",
                            "count": len(mismatches),
                        })
                        # v108 Phase 2：將 mismatch 列表組成可見區塊串流給使用者
                        _fc_lines = ["", "", "═══ ⚠️ 數值校驗異常（系統 fact-check）═══"]
                        for _m in mismatches[:8]:  # 最多顯示 8 條，避免淹沒
                            _t = _m.get("type", "?")
                            _claimed = _m.get("claimed", "?")
                            _actual = _m.get("actual", "?")
                            _name = _m.get("name", "")
                            _tol = _m.get("tolerance", "")
                            _label = f"{_t}{('/'+_name) if _name else ''}"
                            _fc_lines.append(
                                f"  • {_label}: 報告寫 {_claimed}，系統實際 {_actual}"
                                + (f"（容忍 {_tol}）" if _tol else "")
                            )
                        if len(mismatches) > 8:
                            _fc_lines.append(f"  ... 還有 {len(mismatches) - 8} 條未列出")
                        _fc_lines.append("⚠️ 標記區塊的數值不可採信，請以系統實際值為準")
                        _fc_lines.append("═══════════════════════════════════════")
                        _fc_block = "\n".join(_fc_lines)
                        for chunk in _split_text_for_streaming(_fc_block):
                            yield _sse_event("token", {"content": chunk})
                            await asyncio.sleep(0.02)
                        final_text += _fc_block
        except Exception as _fc_err:
            logger.debug(f"fact_check 失敗（不影響主流程）: {_fc_err}")

        # 4d. v103 Phase 2B：用 LLM 產出的真實 entry/target/stop 重做 ML 推論
        # 初始推論用 placeholder（current+5%/-3%），這裡用真實值算 refined SHAP，前端 append 一行
        try:
            _cs = request.chart_state or {}
            _cached_df = _cs.get("_cached_df")
            if final_text and _cached_df is not None and not _cached_df.empty:
                parsed = parse_predictions(final_text)
                if parsed:
                    real_pred = parsed[0]
                    # parse_predictions 回的 dict 沒有 symbol（chat 流程後填），手動補
                    real_pred = {**real_pred, "symbol": chart_symbol_for_save or real_pred.get("symbol", "")}
                    from app.core.feature_extractor import extract_features_at as _eft
                    from app.core.ml_client import predict_via_subprocess as _pvs
                    refined_features = _eft(_cached_df, _cs, real_pred)
                    refined_insight = _pvs(refined_features, _cs, timeout_sec=15)
                    if refined_insight and refined_insight.get("top_features"):
                        yield _sse_event("shap_refine", {
                            "top_features": refined_insight["top_features"],
                            "p_hit_target": refined_insight.get("p_hit_target"),
                            "model_regime": refined_insight.get("model_regime"),
                            "real_entry": real_pred.get("entry_price"),
                            "real_target": real_pred.get("target_price"),
                            "real_stop": real_pred.get("stop_price"),
                            "real_direction": real_pred.get("direction"),
                        })
                        logger.info(
                            f"[shap_refine] direction={real_pred.get('direction')} "
                            f"top={[f['name'] for f in refined_insight['top_features'][:3]]}"
                        )
        except Exception as _shap_err:
            logger.debug(f"shap_refine 失敗（不影響主流程）: {_shap_err}")

        # v107.1：機械審查（純 Python，<50ms，不打 LLM）
        try:
            from app.core.mechanical_audit import audit_final_text
            _audit = audit_final_text(final_text, request.chart_state)
            if _audit.get("n_checks", 0) > 0:
                logger.info(
                    f"[mechanical_audit] {_audit['summary']}"
                    + (f" issues={_audit['issues']}" if _audit['issues'] else "")
                )
            yield _sse_event("audit", _audit)
        except Exception as _audit_err:
            logger.debug(f"mechanical_audit 失敗（不影響主流程）: {_audit_err}")

        # 5. 立刻發 done event，post-processing 改在背景跑（避免 SSE 沉默觸發前端 timeout）
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

        # 6. Post-processing 在 thread pool 跑（v117：避免 sync DB ops 阻塞主 event loop）
        # 原本 async def 但內部沒 await → 仍在主 loop 上跑 sync SQLite/embedding，
        # 會阻塞其他 endpoints。改用 to_thread 丟 thread pool。
        asyncio.create_task(asyncio.to_thread(
            _post_process_chat_message,
            final_text=final_text,
            request_message=request.message,
            chart_state=request.chart_state,
            chart_symbol_for_save=chart_symbol_for_save,
            conversation_id=conversation_id,
            total_usage=total_usage,
        ))

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
    預覽蒸餾結果 — SSE 串流版（v100）。

    為什麼要 SSE：
    - 蒸餾要 N 個 symbol × 1 次 LLM 呼叫，加上 1 次 user profile 呼叫
    - N 可能 5-15 個，每次 10-30 秒 → 總共 3-7 分鐘，會超過 axios 預設 timeout
    - 改 SSE 後：邊跑邊推進度（"正在蒸餾 ADA/USDT (1/12)..."）+ 完成後推 done event
    """

    async def event_gen():
        provider, api_key, base_url, model_name = _resolve_api_key(request)
        if not api_key and provider not in ("ollama", "claude_subscription"):
            yield _sse_event("error", {"message": "需要有效的 LLM session 才能執行蒸餾"})
            yield _sse_event("done", {})
            return

        if not chat_history._conn:
            yield _sse_event("error", {"message": "對話歷史未初始化"})
            yield _sse_event("done", {})
            return

        material = knowledge_distiller.prepare_distill_material(chat_history._conn)
        if material["total_messages"] < 4:
            yield _sse_event("error", {"message": "對話數量不足（至少需要 4 條以上訊息）"})
            yield _sse_event("done", {})
            return

        try:
            adapter = create_adapter(provider=provider, api_key=api_key, model_name=model_name, base_url=base_url)
        except Exception as e:
            yield _sse_event("error", {"message": f"無法連接 LLM: {str(e)}"})
            yield _sse_event("done", {})
            return

        # 過濾 < 2 組 Q&A 的 symbol（也計入總數時排除）
        groups = {sym: qas for sym, qas in material["groups"].items() if len(qas) >= 2}
        all_qa = []
        for pairs in groups.values():
            all_qa.extend(pairs)
        # 蒸餾總任務 = N 個 symbol + 1 個 user profile（若 all_qa >= 5）
        total_tasks = len(groups) + (1 if len(all_qa) >= 5 else 0)
        yield _sse_event("status", {
            "message": f"準備蒸餾 {len(groups)} 個 symbol，預估需 {total_tasks * 20}-{total_tasks * 40} 秒",
            "total": total_tasks, "current": 0,
        })

        previews = []
        total_tokens_used = 0
        completed = 0

        for symbol, qa_pairs in groups.items():
            completed += 1
            display_name = symbol if symbol != "_general" else "一般問題"
            yield _sse_event("progress", {
                "message": f"正在蒸餾 {display_name}（{completed}/{total_tasks}）",
                "current": completed, "total": total_tasks,
                "current_symbol": symbol,
            })

            prompt = knowledge_distiller.build_distill_prompt(symbol, qa_pairs)
            try:
                response = await adapter.chat([{"role": "user", "content": prompt}])
                summary = response.message or ""
                tokens = response.usage.total_tokens if response.usage else 0
                total_tokens_used += tokens

                times = [qa["time"][:10] for qa in qa_pairs if qa.get("time")]
                period_start = min(times) if times else ""
                period_end = max(times) if times else ""

                preview_item = {
                    "symbol": symbol,
                    "period_start": period_start,
                    "period_end": period_end,
                    "summary": summary,
                    "source_count": len(qa_pairs),
                    "original_chars": sum(len(qa["q"]) + len(qa["a"]) for qa in qa_pairs),
                    "distilled_chars": len(summary),
                    "tokens_used": tokens,
                }
                previews.append(preview_item)
                yield _sse_event("preview_item", preview_item)
            except Exception as e:
                logger.error(f"蒸餾 {symbol} 失敗: {e}")
                err_item = {"symbol": symbol, "error": str(e)}
                previews.append(err_item)
                yield _sse_event("preview_item", err_item)

        # 使用者風格分析
        profile_preview = None
        if len(all_qa) >= 5:
            completed += 1
            yield _sse_event("progress", {
                "message": f"正在分析使用者風格（{completed}/{total_tasks}）",
                "current": completed, "total": total_tasks,
                "current_symbol": "_user_profile",
            })
            try:
                profile_prompt = knowledge_distiller.build_profile_prompt(all_qa)
                response = await adapter.chat([{"role": "user", "content": profile_prompt}])
                profile_preview = response.message or ""
                if response.usage:
                    total_tokens_used += response.usage.total_tokens
            except Exception as e:
                logger.warning(f"使用者風格分析失敗: {e}")

        yield _sse_event("done", {
            "status": "ok",
            "previews": previews,
            "profile_preview": profile_preview,
            "total_tokens_used": total_tokens_used,
            "total_messages": material["total_messages"],
            "total_chars": material["total_chars"],
        })

    return StreamingResponse(event_gen(), media_type="text/event-stream")


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
