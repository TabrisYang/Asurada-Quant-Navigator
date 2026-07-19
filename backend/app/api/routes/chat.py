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
import uuid
from typing import Optional
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse
from loguru import logger

from app.core.llm.adapter import create_adapter
from app.core.llm.executor import execute_function_calls, check_input_safety, ProgressTracker
from app.core.llm.function_result_formatter import format_function_results as _format_function_results
from app.core.llm.function_defs import detect_intents, assemble_system_prompt
from app.core.security.key_manager import key_manager
from app.core.config.settings import settings  # v101: feature flags
from app.core.usage_tracker import usage_tracker
from app.core.chat_history import chat_history
from app.core.knowledge_cache import knowledge_cache
from app.core.semantic_cache import semantic_cache
from app.core.knowledge_distiller import knowledge_distiller
from app.core.knowledge_fragments import (
    fragment_store, strip_key_insights,
    strip_system_distill,
)
from app.core.symbol_extractor import extract_symbol_from_text
from app.core.prediction_tracker import (
    prediction_tracker, parse_predictions, strip_predictions,
)
from app.core.prediction_validator import validate_all_active
from app.models.schemas import ChatRequest, ChatResponse, TokenUsageResponse

# ── v157 拆分：helper 依職責搬到同層 4 個模組（純搬家，邏輯零改動）──
# 這裡一併 re-export，讓既有的 `from app.api.routes.chat import X` 呼叫端
# （comprehensive_pipeline.py 與 6 個測試檔）不需改動。
from app.api.routes.chat_text import (  # noqa: F401
    _INDICATOR_TEXT_MAP, _extract_json_function_calls, _try_parse_as_function_call,
    _detect_segments_v138, _detect_mentioned_indicators,
)
from app.api.routes.chat_context import (  # noqa: F401
    MAX_HISTORY_MESSAGES, SUMMARY_THRESHOLD, KEEP_RECENT_MESSAGES,
    _mark_status, _auto_calc_indicator_values, _inject_ml_prediction,
    _build_triplet_warnings, _inject_probability_triplet,
    _build_messages, _compress_to_summary,
)
from app.api.routes.chat_stream_utils import (  # noqa: F401
    _HEARTBEAT_INTERVAL, _HEARTBEAT_SENTINEL, _check_rate_limit, _sse_event,
    _stream_with_heartbeat, _execute_function_calls_in_thread,
)
from app.api.routes.chat_post import (  # noqa: F401
    _PLACEHOLDER_PATTERN, _inject_recent_accuracy, _post_process_chat_message,
    _split_text_for_streaming,
)

router = APIRouter()







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

    if not api_key and provider not in ("ollama", "claude_subscription", "codex_subscription"):
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

    if not api_key and provider not in ("ollama", "claude_subscription", "codex_subscription"):
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
        # v141.1：保留 pre-compress 版 chart_state 給 _post_process_chat_message。
        # 必要：context_compressor 對部分 intent（backtest/regime/calibrate）會剝掉
        # external_signals dict，導致 _capture_signals 寫入的 buckets_json 全 UNKNOWN
        # → shadow_mode 的 combo_stats 永遠 samples=0 → P3 等待路線 deadlock。
        _chart_state_for_post_process: Optional[dict] = None

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

            # v139：sequence 後續訊息 + 某些 intent → 走精簡 chart_state 省 token
            # opt-in safe intents（這些不依賴完整 indicators，可走 r2_mode）
            _R2_SAFE_INTENTS = {"fundamental_analysis", "sector_analysis", "calibrate"}
            _use_r2_for_sequence = bool(
                request.is_sequence_follow
                and (_intents & _R2_SAFE_INTENTS)
                # 排除：event_analysis（需要 indicators 算 pattern 相似度）
                # 排除：compute_laddered / scenario / smc 等需精確 indicator 值的 mode
                and not (_intents & {
                    "event_analysis", "scenario", "smc", "conditional_prob",
                    "deep_analysis", "deep_phase1", "deep_phase2", "deep_phase3",
                    "comprehensive_analysis",
                })
            )
            if _use_r2_for_sequence:
                logger.info(
                    f"[v139] sequence 後續訊息 + safe intent {_intents & _R2_SAFE_INTENTS}, "
                    f"走 r2_mode 精簡 chart_state（省 token）"
                )

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
            # v141.1：壓縮前保存完整版給 _post_process_chat_message，避免 prediction
            # _capture_signals 拿不到 external_signals.derivatives/sentiment → buckets 全 UNKNOWN
            try:
                from app.core.context_compressor import (
                    compress_chart_state, estimate_token_savings,
                )
                _orig_state = request.chart_state
                _chart_state_for_post_process = _orig_state  # v141.1: pre-compress ref
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

            # 移到 worker thread：_build_messages 內含同步的 prediction_tracker（threading.Lock）
            # 與 DB 查詢，直接跑在事件迴圈上會在鎖競爭時凍住所有端點（根因修復）。
            messages = await asyncio.to_thread(
                _build_messages, request,
                rag_fragments=_rag_context_fragments, intents=_intents,
            )
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
            # v139：sequence 後續訊息 + safe intent 走 r2_mode 精簡 chart_state 省 token
            _llm_task = asyncio.create_task(adapter.chat(
                messages, chart_state=request.chart_state,
                system_prompt=_dynamic_prompt,
                chart_screenshot=request.chart_screenshot,
                r2_mode=_use_r2_for_sequence,
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
                text_len = len(response.message or "")
                # v134 修復 A：純文字 ≥ 3000 字 → LLM 已給完整分析，跳過重試避免卡頓
                # 原邏輯硬要重試 → CLI 大 prompt + 同 prompt 重試常 fail → 卡 28+ 分鐘
                if text_len >= 3000:
                    logger.warning(
                        f"[v130/v134] 第一輪 function_calls=0 但純文字 {text_len} 字 ≥ 3000，"
                        f"跳過重試直接用純文字回應"
                    )
                    yield _sse_event("status", {
                        "message": f"[提示] LLM 給出 {text_len} 字純文字分析（未呼叫 function），跳過重試",
                    })
                    # 不重試，用原始 response 繼續走後續流程
                else:
                    logger.warning(
                        f"[v130] comprehensive_analysis 第一輪 function_calls=0 "
                        f"(text_len={text_len}), 強制重試 1 次"
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
                    # v134 修復 B：5 分鐘 hard timeout，超過 cancel + 用原始 response
                    _RETRY_TIMEOUT_SEC = 300
                    _retry_timed_out = False
                    _hb_sec = 0
                    while not _retry_task.done():
                        await asyncio.sleep(3)
                        _hb_sec += 3
                        # v134 修復 C：心跳訊息標明 timeout 上限
                        yield _sse_event("status", {
                            "message": f"重試中... ({_hb_sec}秒 / 最多 {_RETRY_TIMEOUT_SEC} 秒)",
                        })
                        if _hb_sec >= _RETRY_TIMEOUT_SEC:
                            logger.warning(
                                f"[v134] 重試超過 {_RETRY_TIMEOUT_SEC}s，取消重試使用原始回應"
                            )
                            _retry_task.cancel()
                            _retry_timed_out = True
                            yield _sse_event("status", {
                                "message": f"[超時] 重試超過 {_RETRY_TIMEOUT_SEC}s，使用原始回應",
                            })
                            break

                    if not _retry_timed_out and _retry_task.done() and not _retry_task.cancelled():
                        try:
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
                        except Exception as _retry_err:
                            logger.warning(f"[v130] 重試結果取得失敗: {_retry_err}，使用原始回應")
                    # else: 超時 / cancelled，保持原 response 不動

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
                        # quant_research 須排在 analysis 前：quant_research mode 的 intents 同時含
                        # {quant_research, analysis}，迴圈取首個命中 → 不放前面會誤補 analysis 函式
                        # 而非 run_quant_research（造成回測/MC/WF/CPCV/regime「未提供」）。
                        "quant_research": ["run_quant_research"],
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
                        "crypto_fundamental": ["analyze_crypto_fundamentals"],
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

                    # Q2：若本次跑了 SMC 且產出進場價，把 setup 類型記到 post-process 用的
                    # chart_state，讓 prediction _capture_signals 標進 buckets_json.smc_setup，
                    # 之後可累積各 SMC setup（fvg/fib 進場）的歷史命中率。
                    try:
                        for _r in exec_result.get("results", []):
                            if isinstance(_r, dict) and _r.get("function") == "detect_smc_structure":
                                _smc_setup = (_r.get("result") or {}).get("entry_setup")
                                if _smc_setup and _chart_state_for_post_process is not None:
                                    _chart_state_for_post_process["smc_setup"] = _smc_setup
                                break
                    except Exception:
                        pass

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
                        # v138: 改用多 pattern 偵測（支援 ## / 第一 / **1. 等多種結構）
                        _seg2_text_only = _r2_text_buf[_seg1_end:]
                        _found_sections = _detect_segments_v138(_seg2_text_only)
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
                        # v138: 用多 pattern 段落偵測（支援 # / ## / ### / 第一 / **1. / 段落 #N 等）
                        _found_sections = _detect_segments_v138(display_text)
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
                # v149：把 exec_result 一起傳入，讓回測數字（PF/Sharpe/MDD/MC）也納入驗證。
                # exec_result 僅在有 function call 時定義，用 locals().get 防 NameError（純文字回應時為 None）。
                fc = check_text_against_chart_state(
                    final_text, request.chart_state,
                    exec_result=locals().get("exec_result"),
                )
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

        # 4d-2. LLM 覆核：低一階模型交叉檢查數據/邏輯（core/llm/verifier.py）
        # 事後、graceful、不阻擋 — 抓 fact_check 規則層抓不到的方向矛盾/編造/推理錯誤
        try:
            from app.core.llm.verifier import should_verify, verify_answer, format_verify_block
            if settings.verify_enabled and should_verify(final_text, _intents):
                yield _sse_event("verify", {"status": "checking"})
                _v = await verify_answer(
                    final_text, request.chart_state, locals().get("exec_result"),
                    provider, api_key, base_url, model_name or "",
                    override=settings.verify_model_override,
                    timeout_sec=settings.verify_timeout_sec,
                )
                if _v:
                    yield _sse_event("verify", {
                        "status": _v["status"], "model": _v["model"], "issues": _v["issues"],
                    })
                    if _v["issues"]:
                        yield _sse_event("warning", {
                            "message": f"⚠️ AI 覆核發現 {len(_v['issues'])} 個可疑問題，請複核",
                            "type": "verify_issues",
                            "count": len(_v["issues"]),
                        })
                        _v_block = format_verify_block(_v)
                        for chunk in _split_text_for_streaming(_v_block):
                            yield _sse_event("token", {"content": chunk})
                            await asyncio.sleep(0.02)
                        final_text += _v_block  # 併入 final_text → post-process 存 DB
                    _v_usage = _v.get("usage")
                    if api_key and _v_usage:
                        _v_usage_dict = _v_usage.to_dict()
                        usage_tracker.record_usage(
                            api_key=api_key,
                            provider=provider,
                            model=_v["model"],
                            prompt_tokens=_v_usage.prompt_tokens,
                            completion_tokens=_v_usage.completion_tokens,
                            total_tokens=_v_usage.total_tokens,
                            estimated_cost_usd=_v_usage_dict.get("estimated_cost_usd", 0.0),
                            conversation_id=conversation_id,
                            request_type="verify",
                        )
                else:
                    yield _sse_event("verify", {"status": "skipped"})
        except Exception as _v_err:
            logger.debug(f"verify 失敗（不影響主流程）: {_v_err}")

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
            # v141.1: 用 pre-compress 版本，否則 _capture_signals 拿不到 external_signals.derivatives
            chart_state=_chart_state_for_post_process or request.chart_state,
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
        if not api_key and provider not in ("ollama", "claude_subscription", "codex_subscription"):
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

