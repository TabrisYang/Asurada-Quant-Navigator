"""阿斯拉量化系統 v132 — 完整分析「編排管線」（map-reduce）。

把舊 comprehensive_analysis seg2 monolith（一次 LLM 呼叫產 13 段、字數被稀釋）拆成：
  [Map]    5 個維度 focused call（並行，每維度載自己的專用 _mode 模組）
  [Reduce] 1 個 synthesis call（跨維度整合 + 決策卡，串流）

每維度品質對齊「單獨問該維度」；synthesis 保留完整分析獨有的跨維度交叉驗證。
由 settings.comprehensive_pipeline_enabled 灰度切換，舊 seg2 monolith 路徑保留。

對外只暴露 run_pipeline()，為 async generator，yield (event_type, payload) tuple，
由 chat.py 轉成 SSE。特殊終結事件 ("_pipeline_result", {...}) 帶回全文與 usage。

segment 編號：seg1=1（chat.py 負責）、5 維度=2..6、synthesis=7。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncGenerator, Optional

from app.core.llm.function_defs import (
    assemble_dimension_prompt,
    assemble_synthesis_prompt,
    _PIPELINE_DIMENSION_ORDER,
)

logger = logging.getLogger(__name__)

# 維度 → 顯示標籤
_DIMENSION_LABELS = {
    "regime": "市場環境",
    "structure": "結構分析",
    "momentum": "動能特徵",
    "backtest": "多策略回測",
    "quant": "量化研究",
}

# 維度 → 需要的 function 結果（從 Round 1 的 exec_result 切片）
_DIMENSION_FUNCTIONS: dict[str, set[str]] = {
    "regime": {"generate_scenarios"},
    "structure": {"detect_smc_structure", "generate_scenarios"},
    "momentum": {"analyze_momentum", "scan_conditional_probability"},
    "backtest": {"compare_strategies", "run_quant_research"},
    "quant": {"run_quant_research"},
}

_SEG_BASE = 2  # 第一個維度的 segment 編號（seg1=1 由 chat.py 負責）
_SYNTHESIS_SEG = _SEG_BASE + len(_PIPELINE_DIMENSION_ORDER)  # = 7

_MAP_CONCURRENCY = 3        # 並行 map call 上限（避免過多 Claude CLI subprocess 並起）
_DIMENSION_MIN_CHARS = 600  # 維度輸出最低字數，不足重試一次（連貫性保險 #3）
_HEARTBEAT_INTERVAL = 2.0


# ─── 文字清理（維度模組如 alpha_monitor 可能輸出 SYSTEM_DISTILL 等標記）────

def _strip_markers(text: str) -> str:
    """移除 LLM 輸出中的 KEY_INSIGHTS / PREDICTIONS / SYSTEM_DISTILL 標記段。"""
    if not text:
        return text
    try:
        from app.core.knowledge_fragments import strip_key_insights, strip_system_distill
        from app.core.prediction_tracker import strip_predictions
        return strip_system_distill(strip_predictions(strip_key_insights(text)))
    except Exception:  # noqa: BLE001 — 清理失敗不阻斷主流程
        return text


# ─── usage 累加（統一用 dict）────────────────────────────

def _usage_to_dict(usage) -> dict:
    if not usage:
        return {}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        "provider": getattr(usage, "provider", None),
        "model": getattr(usage, "model", None),
    }


def _merge_usage(acc: dict, other: dict) -> None:
    if not other:
        return
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        acc[k] = acc.get(k, 0) + (other.get(k, 0) or 0)
    if other.get("provider") and not acc.get("provider"):
        acc["provider"] = other["provider"]
    if other.get("model") and not acc.get("model"):
        acc["model"] = other["model"]


# ─── function 結果切片 ──────────────────────────────────

def _slice_function_results(function_calls, exec_result, wanted: set[str]) -> str:
    """從 Round 1 的 exec_result 切出某維度需要的 function 結果，格式化成文字。

    注意 exec_result["results"] 在 chat.py 的「自動補齊」後可能比 function_calls 長，
    且補齊項只有 result.function、無對應 function_call → 以 result.function 為主判定。
    """
    from app.api.routes.chat import _format_function_results  # 延遲匯入避免循環

    results = (exec_result or {}).get("results", []) or []
    function_calls = function_calls or []
    sub_fcs, sub_results = [], []
    for i, res in enumerate(results):
        if not isinstance(res, dict):
            continue
        name = res.get("function")
        if not name and i < len(function_calls):
            name = function_calls[i].get("name")
        if name not in wanted:
            continue
        if i < len(function_calls) and function_calls[i].get("name") == name:
            fc = function_calls[i]
        else:
            fc = {"name": name, "arguments": {}}
        sub_fcs.append(fc)
        sub_results.append(res)
    if not sub_fcs:
        return ""
    try:
        return _format_function_results(sub_fcs, {"results": sub_results})
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[pipeline] _slice_function_results 格式化失敗: {e}")
        return ""


# ─── 單一維度 focused call ──────────────────────────────

async def _call_dimension(
    adapter, dimension: str, *,
    user_message: str, chart_state: Optional[dict],
    function_calls: list, exec_result: dict,
    pre_backtest_summary: str, teaching_mode: bool,
    chart_symbol: str, chart_timeframe: str,
) -> tuple[str, dict, bool]:
    """跑單一維度的 focused call，回傳 (text, usage_dict, ok)。

    ok=False 代表兩次嘗試後輸出仍過短 / 失敗（連貫性保險 #3）。
    """
    wanted = _DIMENSION_FUNCTIONS.get(dimension, set())
    fc_summary = _slice_function_results(function_calls, exec_result, wanted)
    if dimension == "backtest" and pre_backtest_summary:
        fc_summary = (fc_summary + "\n\n" + pre_backtest_summary).strip()

    label = _DIMENSION_LABELS.get(dimension, dimension)
    symbol_hint = f"（標的：{chart_symbol} {chart_timeframe}）" if chart_symbol else ""
    if fc_summary:
        data_block = f"[系統自動回傳] 以下是「{label}」維度相關的函式結果：\n\n{fc_summary}\n\n"
    else:
        data_block = "[系統提示] 本維度無對應 function 結果，請依 chart_state 與系統 prompt 規格分析。\n\n"

    messages = [
        {"role": "user", "content": f"{symbol_hint}{user_message}"},
        {"role": "user", "content": (
            f"{data_block}"
            f"請只產出「{label}」這一個維度的深度分析，"
            f"依系統 prompt 指定的專屬輸出格式，做到單獨深度分析的品質與篇幅。"
        )},
    ]
    system_prompt = assemble_dimension_prompt(dimension, teaching_mode=teaching_mode)

    usage_acc: dict = {}
    text = ""
    for attempt in (1, 2):
        try:
            resp = await adapter.chat(
                messages, chart_state=chart_state,
                system_prompt=system_prompt, force_text=True, r2_mode=True,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[pipeline] 維度 {dimension} 第 {attempt} 次呼叫失敗: {e}")
            text = ""
            continue
        _merge_usage(usage_acc, _usage_to_dict(getattr(resp, "usage", None)))
        text = _strip_markers((resp.message or "").strip())
        if len(text) >= _DIMENSION_MIN_CHARS:
            return text, usage_acc, True
        logger.warning(
            f"[pipeline] 維度 {dimension} 輸出過短（{len(text)} < {_DIMENSION_MIN_CHARS}）"
            f"{'，重試' if attempt == 1 else '，仍不足→標記不可得'}"
        )
    return text, usage_acc, False


# ─── 主編排 ─────────────────────────────────────────────

async def run_pipeline(
    *,
    adapter,
    user_message: str,
    chart_state: Optional[dict],
    function_calls: list,
    exec_result: dict,
    seg1_card_text: str = "",
    pre_backtest_summary: str = "",
    teaching_mode: bool = False,
    chart_symbol: str = "",
    chart_timeframe: str = "",
) -> AsyncGenerator[tuple, None]:
    """完整分析編排管線。yield (event_type, payload) tuple 給 chat.py 轉 SSE。

    事件：status / token / segment_complete / warning，
    最後一個必為 ("_pipeline_result", {"full_text": str, "usage": dict})。
    """
    total_usage: dict = {}
    full_text_parts: list[str] = []
    symbol_hint = f"（標的：{chart_symbol} {chart_timeframe}）" if chart_symbol else ""

    # ═══ Map 階段：5 維度並行 focused call ═══
    sem = asyncio.Semaphore(_MAP_CONCURRENCY)

    async def _guarded(dim: str):
        async with sem:
            return dim, await _call_dimension(
                adapter, dim,
                user_message=user_message, chart_state=chart_state,
                function_calls=function_calls, exec_result=exec_result,
                pre_backtest_summary=pre_backtest_summary,
                teaching_mode=teaching_mode,
                chart_symbol=chart_symbol, chart_timeframe=chart_timeframe,
            )

    yield ("status", {"message": "完整分析 — 5 維度深度分析啟動中..."})
    tasks = {asyncio.create_task(_guarded(d)): d for d in _PIPELINE_DIMENSION_ORDER}

    # 並行等待 + 心跳（避免 map 收集期間 SSE idle 斷線）
    t0 = time.time()
    while not all(t.done() for t in tasks):
        await asyncio.sleep(_HEARTBEAT_INTERVAL)
        done_n = sum(1 for t in tasks if t.done())
        yield ("status", {
            "message": f"⏳ 維度深度分析中（{done_n}/{len(tasks)} 完成，{int(time.time() - t0)}s）",
        })

    dim_results: dict[str, tuple[str, bool]] = {}
    for t, dim in tasks.items():
        try:
            _dim, (text, usage, ok) = t.result()
        except Exception as e:  # noqa: BLE001
            logger.error(f"[pipeline] 維度 {dim} task 例外: {e}")
            text, usage, ok = "", {}, False
        _merge_usage(total_usage, usage)
        dim_results[dim] = (text, ok)

    # ═══ 依固定順序 emit 每維度 bubble ═══
    for idx, dim in enumerate(_PIPELINE_DIMENSION_ORDER):
        seg_no = _SEG_BASE + idx
        label = _DIMENSION_LABELS[dim]
        text, ok = dim_results.get(dim, ("", False))

        # segment_complete：切到本段（含 seg1→第一維度 的首次切換）
        yield ("segment_complete", {
            "segment": seg_no - 1,
            "next_segment": seg_no,
            "next_label": label,
        })

        if not ok or not text:
            text = (
                f"## {label}\n\n"
                f"⚠️ 資料不可得：本維度分析未能產生有效輸出（已自動重試一次仍不足）。"
            )
            yield ("warning", {
                "message": f"完整分析「{label}」維度未能產生有效輸出",
                "type": "dimension_incomplete",
                "dimension": dim,
            })

        yield ("token", {"content": text, "segment": seg_no})
        full_text_parts.append(text)

    # ═══ Reduce 階段：synthesis 串流 ═══
    yield ("segment_complete", {
        "segment": _SYNTHESIS_SEG - 1,
        "next_segment": _SYNTHESIS_SEG,
        "next_label": "跨維度整合",
    })

    dim_report_block = "\n\n".join(
        f"───── 維度：{_DIMENSION_LABELS[d]} ─────\n{dim_results.get(d, ('（無輸出）', False))[0]}"
        for d in _PIPELINE_DIMENSION_ORDER
    )
    synth_input = (
        "[系統自動回傳] 以下是 5 個維度的深度分析全文，以及第 1 段的 30 秒結論卡。\n\n"
        f"{dim_report_block}\n\n"
        "───── 第 1 段 30 秒結論卡 ─────\n"
        f"{seg1_card_text.strip() or '（第 1 段未提供）'}\n\n"
        "請依系統 prompt 規格做「跨維度整合 + 決策」（#6-#11）：主動標示維度間矛盾，"
        "#7 正式結論卡的方向 / 信心 / 倉位必須與上方 30 秒結論卡完全一致。"
    )
    synth_messages = [
        {"role": "user", "content": f"{symbol_hint}{user_message}"},
        {"role": "user", "content": synth_input},
    ]
    system_prompt = assemble_synthesis_prompt(teaching_mode=teaching_mode)
    yield ("status", {"message": "完整分析 — 跨維度整合中..."})

    # 延遲匯入避免循環依賴（chat.py 會匯入本模組）
    from app.api.routes.chat import _stream_with_heartbeat, _HEARTBEAT_SENTINEL

    synth_text = ""
    s0 = time.time()
    try:
        async for evt in _stream_with_heartbeat(adapter.chat_stream_events(
            synth_messages, chart_state=chart_state,
            system_prompt=system_prompt, force_text=True, r2_mode=True,
        )):
            if evt is _HEARTBEAT_SENTINEL:
                yield ("status", {"message": f"⏳ 跨維度整合中... ({int(time.time() - s0)}s)"})
                continue
            if evt.type == "text_delta":
                synth_text += evt.text
                yield ("token", {"content": evt.text, "segment": _SYNTHESIS_SEG})
            elif evt.type == "usage":
                _merge_usage(total_usage, _usage_to_dict(evt.usage))
            # function_call / stop 在 force_text 下忽略
    except Exception as e:  # noqa: BLE001
        logger.error(f"[pipeline] synthesis 串流失敗: {e}")
        yield ("warning", {
            "message": f"跨維度整合階段失敗：{e}",
            "type": "synthesis_failed",
        })

    if synth_text.strip():
        full_text_parts.append(_strip_markers(synth_text))
    else:
        yield ("warning", {
            "message": "跨維度整合未產生輸出",
            "type": "synthesis_empty",
        })

    # ═══ 終結事件 ═══
    yield ("_pipeline_result", {
        "full_text": "\n\n".join(p for p in full_text_parts if p),
        "usage": total_usage,
    })
