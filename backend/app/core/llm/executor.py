"""阿斯拉量化系統 — LLM Function Call 執行器

接收 LLM 回傳的 function calls，執行對應操作，
返回前端需要的圖表更新指令。
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from loguru import logger


# ── 進度追蹤器 ────────────────────────────────────────────

# 函式中文顯示名稱
_FUNC_DISPLAY_NAMES: dict[str, str] = {
    "query_chart_data": "載入價格數據",
    "manage_indicator": "設定指標",
    "find_conditions": "條件掃描",
    "annotate_chart": "圖表標記",
    "draw_pattern": "繪製型態",
    "generate_analysis": "文字分析",
    "suggest_indicators": "指標建議",
    "run_backtest": "策略回測",
    "compare_strategies": "多策略比較",
    "analyze_event_patterns": "事件型態分析",
    "run_quant_research": "量化研究",
    "optimize_indicator_params": "參數優化",
    "scan_conditional_probability": "條件機率掃描",
    "generate_scenarios": "情境預測",
    "detect_smc_structure": "SMC 結構分析",
}


@dataclass
class ProgressTracker:
    """追蹤 function call 執行進度，供 SSE 心跳讀取"""
    total: int = 0
    completed: int = 0
    current_task: str = ""
    phase: str = "init"       # init → sequential → parallel → done

    @property
    def percentage(self) -> int:
        if self.total == 0:
            return 0
        return min(int(self.completed / self.total * 100), 99)

    @property
    def status_text(self) -> str:
        name = _FUNC_DISPLAY_NAMES.get(self.current_task, self.current_task)
        if self.phase == "done":
            return "分析完成 (100%)"
        return f"[{self.completed}/{self.total}] {name}... ({self.percentage}%)"

from app.core.indicators import registry
from app.data.fetchers.crypto_engine import crypto_engine
from app.data.fetchers.tw_stock_engine import tw_stock_engine
from app.utils.symbol import is_tw_stock


def _load_local_data(symbol: str, timeframe: str, start: str = None, end: str = None):
    """根據 symbol 類型選擇正確的數據引擎載入本地資料"""
    engine = tw_stock_engine if is_tw_stock(symbol) else crypto_engine
    return engine.load_local_data(symbol, timeframe, start, end)


# ========== 安全過濾 ==========

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous",
    r"忽略.*指令",
    r"forget\s+(all|your)",
    r"you\s+are\s+now",
    r"new\s+instructions?",
    r"system\s*prompt",
    r"override",
    r"jailbreak",
    r"<script",
    r"\\x[0-9a-fA-F]",
]

ALLOWED_FUNCTIONS = {
    "query_chart_data",
    "manage_indicator",
    "find_conditions",
    "annotate_chart",
    "draw_pattern",
    "generate_analysis",
    "suggest_indicators",
    "run_backtest",
    "compare_strategies",
    "analyze_event_patterns",
    "run_quant_research",
    "optimize_indicator_params",
    "scan_conditional_probability",
    "generate_scenarios",
    "detect_smc_structure",
}


def check_input_safety(text: str) -> bool:
    """檢查使用者輸入是否安全"""
    lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lower):
            logger.warning(f"偵測到可疑輸入: {text[:100]}")
            return False
    return True


def validate_function_call(func_name: str, args: dict) -> bool:
    """驗證 function call 是否在白名單內"""
    if func_name not in ALLOWED_FUNCTIONS:
        logger.warning(f"LLM 嘗試呼叫未授權函式: {func_name}")
        return False
    return True


# ========== 函式執行器 ==========

async def execute_function_calls(
    function_calls: list[dict[str, Any]],
    chart_state: Optional[dict] = None,
    progress: Optional[ProgressTracker] = None,
) -> dict[str, Any]:
    """
    執行 LLM 回傳的 function calls
    重量級分析函式（回測、SMC、情境預測等）會並行執行以避免超時。

    Args:
        progress: 可選的進度追蹤器，供外部輪詢進度百分比

    Returns:
        {
            "chart_updates": {...},  # 前端需要的圖表更新
            "results": [...],        # 各函式的執行結果
        }
    """
    chart_updates: dict[str, Any] = {}
    results: list[dict] = []

    # 從 chart_state 取得預設值
    default_symbol = (chart_state or {}).get("symbol", "BTC/USDT")
    default_timeframe = (chart_state or {}).get("timeframe", "1d")

    # 可並行執行的重量級非同步函式
    _PARALLEL_FUNCS = {
        "run_backtest", "compare_strategies", "analyze_event_patterns",
        "run_quant_research", "optimize_indicator_params",
        "scan_conditional_probability", "generate_scenarios",
        "detect_smc_structure",
    }

    # 分離：輕量同步/依序 vs 重量級可並行
    sequential_calls = []
    parallel_calls = []
    for fc in function_calls:
        name = fc.get("name", "")
        if not validate_function_call(name, fc.get("arguments", {})):
            results.append({"function": name, "error": "未授權的函式呼叫"})
            continue
        if name in _PARALLEL_FUNCS:
            parallel_calls.append(fc)
        else:
            sequential_calls.append(fc)

    # 初始化進度追蹤
    _total = len(sequential_calls) + len(parallel_calls)
    if progress is not None:
        progress.total = _total
        progress.completed = 0
        progress.phase = "sequential" if sequential_calls else "parallel"

    # --- Phase 1: 依序執行輕量函式 ---
    for fc in sequential_calls:
        name = fc["name"]
        args = fc.get("arguments", {})
        if progress is not None:
            progress.current_task = name
        try:
            if name == "query_chart_data":
                result = await _exec_query_chart(args, default_symbol, default_timeframe)
                cu = result.get("chart_updates", {})
                if not chart_updates.get("symbol"):
                    chart_updates.update(cu)
                else:
                    chart_updates.setdefault("multi_symbol_data", []).append(cu)
                results.append({"function": name, "result": result})

            elif name == "manage_indicator":
                result = _exec_manage_indicator(args)
                chart_updates.setdefault("indicator_actions", []).append(result)
                results.append({"function": name, "result": result})

            elif name == "find_conditions":
                result = await _exec_find_conditions(args, default_symbol, default_timeframe)
                chart_updates.setdefault("annotations", []).extend(result.get("annotations", []))
                results.append({"function": name, "result": result})

            elif name == "annotate_chart":
                ann_list = _exec_annotate(args)
                chart_updates.setdefault("annotations", []).extend(ann_list)
                results.append({"function": name, "result": {"count": len(ann_list), "group_name": args.get("group_name", "AI 標記")}})

            elif name == "draw_pattern":
                ann_list = _exec_draw_pattern(args)
                chart_updates.setdefault("annotations", []).extend(ann_list)
                results.append({"function": name, "result": {"pattern": args.get("pattern_name"), "points": len(args.get("points", [])), "lines": len(ann_list)}})

            elif name == "generate_analysis":
                results.append({"function": name, "result": {"type": "text_analysis"}})

            elif name == "suggest_indicators":
                result = _exec_suggest_indicators(args)
                results.append({"function": name, "result": result})

        except Exception as e:
            logger.error(f"執行 {name} 失敗: {e}")
            results.append({"function": name, "error": str(e)})
        finally:
            if progress is not None:
                progress.completed += 1

    # --- Phase 2: 並行執行重量級函式 ---
    if parallel_calls:
        if progress is not None:
            progress.phase = "parallel"
            # 顯示並行中最具代表性的任務名
            progress.current_task = parallel_calls[0]["name"]

        async def _run_one(fc: dict) -> dict:
            """執行單一重量級函式，完成後更新進度"""
            name = fc["name"]
            args = fc.get("arguments", {})
            try:
                if name == "run_backtest":
                    result = await _exec_backtest(args, default_symbol, default_timeframe)
                    return {"function": name, "result": result}
                elif name == "compare_strategies":
                    result = await _exec_compare_strategies(args, default_symbol, default_timeframe)
                    return {"function": name, "result": result}
                elif name == "analyze_event_patterns":
                    result = await _exec_analyze_event_patterns(args, default_symbol, default_timeframe)
                    return {"function": name, "result": result}
                elif name == "run_quant_research":
                    result = await _exec_quant_research(args, default_symbol, default_timeframe)
                    return {"function": name, "result": result}
                elif name == "optimize_indicator_params":
                    result = await _exec_optimize_params(args, default_symbol, default_timeframe)
                    return {"function": name, "result": result}
                elif name == "scan_conditional_probability":
                    result = await _exec_conditional_prob_scan(args, default_symbol, default_timeframe)
                    return {"function": name, "result": result}
                elif name == "generate_scenarios":
                    result = await _exec_generate_scenarios(args, default_symbol, default_timeframe)
                    return {"function": name, "result": result}
                elif name == "detect_smc_structure":
                    result = await _exec_detect_smc(args, default_symbol, default_timeframe)
                    return {"function": name, "result": result}
                return {"function": name, "error": f"未知的並行函式: {name}"}
            except Exception as e:
                logger.error(f"執行 {name} 失敗: {e}")
                return {"function": name, "error": str(e)}
            finally:
                if progress is not None:
                    progress.completed += 1
                    # 更新 current_task 為尚未完成的並行任務中第一個
                    _remaining = [
                        pc["name"] for pc in parallel_calls
                        if pc["name"] != name
                    ]
                    if _remaining:
                        progress.current_task = _remaining[0]

        parallel_results = await asyncio.gather(
            *[_run_one(fc) for fc in parallel_calls],
            return_exceptions=False,
        )

        # 合併並行結果到 chart_updates 和 results
        for pr in parallel_results:
            results.append(pr)
            r = pr.get("result", {})
            if not isinstance(r, dict):
                continue
            fname = pr.get("function", "")
            if fname == "run_backtest" and r.get("trade_annotations"):
                chart_updates.setdefault("annotations", []).extend(r["trade_annotations"])
            elif fname == "analyze_event_patterns" and r.get("annotations"):
                chart_updates.setdefault("annotations", []).extend(r.pop("annotations"))

    if progress is not None:
        progress.phase = "done"
        progress.completed = progress.total

    return {"chart_updates": chart_updates, "results": results}


async def _exec_query_chart(args: dict, default_symbol: str, default_tf: str) -> dict:
    """執行 query_chart_data — 回傳壓縮價格摘要供 LLM 精確回答歷史問題"""
    import numpy as np
    import pandas as pd

    symbol = args.get("symbol", default_symbol)
    timeframe = args.get("timeframe", default_tf)
    start = args.get("start_date")
    end = args.get("end_date")

    df = _load_local_data(symbol, timeframe, start, end)

    result: dict[str, Any] = {
        "chart_updates": {
            "symbol": symbol,
            "timeframe": timeframe,
            # ★ 不回傳 startDate/endDate 給前端，避免 LLM 的查詢範圍
            #   覆蓋前端圖表的日期設定，導致圖表只顯示部分數據。
            #   前端圖表應始終顯示全量數據，LLM 的日期篩選僅用於本次分析。
            "dataLoaded": not df.empty,
            "dataPoints": len(df),
        }
    }

    if df.empty:
        return result

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    volumes = df["volume"].values
    timestamps = df["timestamp"].values

    high_idx = int(np.argmax(highs))
    low_idx = int(np.argmin(lows))

    result["price_summary"] = {
        "period_high": round(float(highs[high_idx]), 4),
        "period_high_date": str(timestamps[high_idx])[:16],
        "period_low": round(float(lows[low_idx]), 4),
        "period_low_date": str(timestamps[low_idx])[:16],
        "first_open": round(float(df["open"].iloc[0]), 4),
        "last_close": round(float(closes[-1]), 4),
        "first_date": str(timestamps[0])[:16],
        "last_date": str(timestamps[-1])[:16],
        "avg_volume": int(np.mean(volumes)),
    }

    ts_series = pd.to_datetime(df["timestamp"])
    n = len(df)

    if n > 60 and timeframe in ("15m", "1h", "4h"):
        day_labels = ts_series.dt.strftime("%Y-%m-%d")
        daily = []
        for day, idx_list in sorted(ts_series.groupby(day_labels).groups.items()):
            pos = idx_list.to_numpy()
            daily.append({"d": day,
                          "h": round(float(highs[pos].max()), 4),
                          "l": round(float(lows[pos].min()), 4),
                          "c": round(float(closes[pos[-1]]), 4)})
        if len(daily) > 90:
            mo_labels = ts_series.dt.strftime("%Y-%m")
            monthly = []
            for mo, idx_list in sorted(ts_series.groupby(mo_labels).groups.items()):
                pos = idx_list.to_numpy()
                monthly.append({"m": mo,
                                "h": round(float(highs[pos].max()), 4),
                                "l": round(float(lows[pos].min()), 4),
                                "c": round(float(closes[pos[-1]]), 4)})
            result["price_summary"]["monthly_ohlc"] = monthly
        else:
            result["price_summary"]["daily_ohlc"] = daily
    elif n > 60 and timeframe in ("1d", "1w"):
        mo_labels = ts_series.dt.strftime("%Y-%m")
        monthly = []
        for mo, idx_list in sorted(ts_series.groupby(mo_labels).groups.items()):
            pos = idx_list.to_numpy()
            monthly.append({"m": mo,
                            "h": round(float(highs[pos].max()), 4),
                            "l": round(float(lows[pos].min()), 4),
                            "c": round(float(closes[pos[-1]]), 4)})
        result["price_summary"]["monthly_ohlc"] = monthly
    else:
        result["price_summary"]["candles"] = [
            {"t": str(timestamps[i])[:16], "h": round(float(highs[i]), 4),
             "l": round(float(lows[i]), 4), "c": round(float(closes[i]), 4)}
            for i in range(n)
        ]

    return result


def _exec_manage_indicator(args: dict) -> dict:
    """執行 manage_indicator"""
    action = args.get("action", "add")
    indicator_id = args.get("indicator_id", "")
    params = args.get("parameters", {})

    # 驗證指標存在
    indicator = registry.get(indicator_id)
    if not indicator:
        return {"error": f"找不到指標: {indicator_id}"}

    return {
        "action": action,
        "indicator_id": indicator_id,
        "indicator_name": indicator.name,
        "parameters": params,
        "display_mode": indicator.display_mode,
    }


async def _exec_find_conditions(args: dict, default_symbol: str, default_tf: str) -> dict:
    """執行 find_conditions"""
    symbol = args.get("symbol", default_symbol)
    timeframe = args.get("timeframe", default_tf)
    conditions = args.get("conditions", [])
    start = args.get("start_date")
    end = args.get("end_date")

    df = _load_local_data(symbol, timeframe, start, end)
    if df.empty:
        return {"matched_periods": [], "summary": "找不到數據", "annotations": []}

    import pandas as pd
    condition_masks = []
    for cond in conditions:
        indicator_id = cond.get("indicator", "").lower()
        calc_result = registry.calculate(indicator_id, df, cond.get("parameters"))
        if not calc_result:
            continue

        series_name = list(calc_result.keys())[0]
        values = pd.Series(calc_result[series_name])
        op = cond.get("operator", ">")
        val = cond.get("value", 0)

        if op == ">":
            mask = values > val
        elif op == "<":
            mask = values < val
        elif op == ">=":
            mask = values >= val
        elif op == "<=":
            mask = values <= val
        elif op == "==":
            mask = values == val
        elif op == "cross_above":
            mask = (values > val) & (values.shift(1) <= val)
        elif op == "cross_below":
            mask = (values < val) & (values.shift(1) >= val)
        elif op == "between":
            mask = (values >= val) & (values <= cond.get("value2", val))
        else:
            continue
        condition_masks.append(mask)

    if not condition_masks:
        return {"matched_periods": [], "summary": "無有效條件", "annotations": []}

    logical = args.get("logical_operator", "AND")
    combined = condition_masks[0]
    for m in condition_masks[1:]:
        combined = combined & m if logical == "AND" else combined | m

    matched = df[combined]
    annotations = []
    if not matched.empty:
        timestamps = matched["timestamp"].tolist()
        for ts in timestamps:
            annotations.append({
                "type": "vertical_line",
                "time": str(ts),
                "color": "#f85149",
            })

    return {
        "matched_periods": len(matched),
        "summary": f"找到 {len(matched)} 個匹配點",
        "annotations": annotations,
    }


_ANNOTATE_ALLOWED_TYPES = {"horizontal_line", "text_label"}


def _exec_annotate(args: dict) -> list[dict]:
    """執行 annotate_chart — 支援批量繪圖，回傳 annotation 列表。
    白名單過濾：只允許 horizontal_line 和 text_label，
    trend_line / highlight_range / vertical_line 一律丟棄。
    """
    import uuid
    group_id = str(uuid.uuid4())[:8]
    group_name = args.get("group_name", "AI 標記")

    def _build_one(a: dict) -> dict | None:
        ann_type = a.get("annotation_type", "horizontal_line")
        if ann_type not in _ANNOTATE_ALLOWED_TYPES:
            return None
        return {
            "type": ann_type,
            "startTime": a.get("start_time"),
            "endTime": a.get("end_time"),
            "price": a.get("price"),
            "endPrice": a.get("end_price"),
            "text": a.get("text"),
            "color": a.get("color", "#58a6ff"),
            "lineWidth": a.get("line_width", 2),
            "lineStyle": a.get("line_style", 0),
            "groupId": group_id,
            "groupName": group_name,
        }

    batch = args.get("annotations")
    if batch and isinstance(batch, list):
        return [r for item in batch if (r := _build_one(item)) is not None]

    result = _build_one(args)
    return [result] if result is not None else []


def _exec_draw_pattern(args: dict) -> list[dict]:
    """draw_pattern — 根據關鍵點自動連線和標注，回傳 annotation 列表"""
    import uuid
    pattern_name = args.get("pattern_name", "Pattern")
    points = args.get("points", [])
    color = args.get("color", "#f0b90b")
    line_width = args.get("line_width", 2)
    bullish = args.get("bullish")

    if bullish is True:
        color = args.get("color", "#26a69a")
    elif bullish is False:
        color = args.get("color", "#ef5350")

    group_id = str(uuid.uuid4())[:8]
    group_name = f"{pattern_name}"
    annotations: list[dict] = []

    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        annotations.append({
            "type": "trend_line",
            "startTime": p1.get("time"),
            "endTime": p2.get("time"),
            "price": p1.get("price"),
            "endPrice": p2.get("price"),
            "color": color,
            "lineWidth": line_width,
            "lineStyle": 0,
            "groupId": group_id,
            "groupName": group_name,
        })

    for pt in points:
        label = pt.get("label", "")
        annotations.append({
            "type": "text_label",
            "startTime": pt.get("time"),
            "price": pt.get("price"),
            "text": label,
            "color": color,
            "groupId": group_id,
            "groupName": group_name,
        })

    return annotations


def _exec_suggest_indicators(args: dict) -> dict:
    """執行 suggest_indicators — 根據分析目標推薦多維度指標組合"""
    goal = args.get("analysis_goal", "").lower()

    # 關鍵字 → 推薦指標（涵蓋更多指標，不再只有老三樣）
    suggestions = {
        # 趨勢判斷
        "趨勢": ["ema", "adx", "supertrend", "market_structure", "ichimoku"],
        "trend": ["ema", "adx", "supertrend", "market_structure", "ichimoku"],
        "方向": ["adx", "supertrend", "psar", "ema"],
        # 動量 / 超買超賣
        "超買超賣": ["rsi", "stochrsi", "bb", "bias"],
        "overbought": ["rsi", "stochrsi", "bb", "bias"],
        "動量": ["macd", "roc", "rsi", "stochrsi"],
        "momentum": ["macd", "roc", "rsi", "stochrsi"],
        # 波動率
        "波動": ["bb", "atr", "keltner", "donchian", "vol_switch"],
        "volatility": ["bb", "atr", "keltner", "donchian", "vol_switch"],
        # 量能
        "量能": ["obv", "rel_vol", "vol_switch", "vwap"],
        "volume": ["obv", "rel_vol", "vol_switch", "vwap"],
        "成交量": ["obv", "rel_vol", "vol_switch"],
        # 風險管理
        "風險": ["atr", "trailing_stop", "max_drawdown", "kelly"],
        "risk": ["atr", "trailing_stop", "max_drawdown", "kelly"],
        "止損": ["trailing_stop", "atr", "psar"],
        # 情緒
        "情緒": ["fear_greed", "funding"],
        "sentiment": ["fear_greed", "funding"],
        # 型態
        "型態": ["harmonic", "market_structure", "bb"],
        "pattern": ["harmonic", "market_structure", "bb"],
        # 進出場
        "進場": ["rsi", "stochrsi", "supertrend", "bb", "vwap"],
        "出場": ["trailing_stop", "psar", "atr", "rsi"],
        "entry": ["rsi", "stochrsi", "supertrend", "bb", "vwap"],
        "exit": ["trailing_stop", "psar", "atr", "rsi"],
    }

    recommended = []
    seen_ids = set()
    for keyword, ids in suggestions.items():
        if keyword in goal:
            for ind_id in ids:
                if ind_id not in seen_ids:
                    ind = registry.get(ind_id)
                    if ind:
                        recommended.append({
                            "id": ind.id,
                            "name": ind.name,
                            "description": ind.description,
                        })
                        seen_ids.add(ind_id)

    if not recommended:
        # 預設：多維度綜合分析組合（涵蓋趨勢+動量+量能+波動率+情緒）
        default_ids = ["adx", "supertrend", "rsi", "macd", "obv", "bb", "atr", "fear_greed"]
        for ind_id in default_ids:
            ind = registry.get(ind_id)
            if ind:
                recommended.append({"id": ind.id, "name": ind.name, "description": ind.description})

    return {"recommended": recommended, "goal": goal}


async def _exec_compare_strategies(args: dict, default_symbol: str, default_tf: str) -> dict:
    """執行多策略比較回測"""
    from app.core.backtest import run_backtest

    strategies = args.get("strategies", [])
    if not strategies or not isinstance(strategies, list):
        return {"status": "error", "message": "缺少 strategies 陣列"}
    if len(strategies) > 5:
        return {"status": "error", "message": "最多比較 5 個策略"}

    symbol = args.get("symbol", default_symbol)
    timeframe = args.get("timeframe", default_tf)
    start = args.get("start_date")
    end = args.get("end_date")

    _MIN_BARS = 60
    _PREFER_MIN_BARS = 500
    df = _load_local_data(symbol, timeframe, start, end)
    if (start or end) and len(df) < _PREFER_MIN_BARS:
        df_full = _load_local_data(symbol, timeframe)
        if len(df_full) > len(df):
            logger.info(f"策略比較 [{symbol}]: 指定範圍 {len(df)} 根不足建議量，擴大至全部（{len(df_full)} 根）")
            df = df_full
    if df.empty or len(df) < _MIN_BARS:
        return {"status": "error", "message": f"找不到 {symbol} {timeframe} 的本地數據，請先同步。"}

    comparison: list[dict] = []
    for i, strat in enumerate(strategies):
        name = strat.get("name", f"策略 {i + 1}")
        entry_conds = strat.get("entry_conditions", [])
        exit_conds = strat.get("exit_conditions", [])
        direction = strat.get("direction", "long")
        sl = strat.get("stop_loss_pct")
        tp = strat.get("take_profit_pct")

        if not entry_conds or not exit_conds:
            comparison.append({"name": name, "status": "error", "message": "缺少進場或出場條件"})
            continue

        result = run_backtest(
            df=df,
            entry_conditions=entry_conds,
            exit_conditions=exit_conds,
            direction=direction,
            stop_loss_pct=sl,
            take_profit_pct=tp,
        )
        comparison.append({
            "name": name,
            "status": "success",
            "metrics": result.metrics,
            "warnings_count": len(result.warnings),
        })

    # 排名（按 Sharpe 或 total_return 排序）
    valid = [c for c in comparison if c.get("status") == "success" and c.get("metrics")]
    if valid:
        valid.sort(key=lambda x: x["metrics"].get("sharpe_ratio", 0), reverse=True)
        for rank, item in enumerate(valid, 1):
            item["rank"] = rank

    return {
        "status": "success",
        "symbol": symbol,
        "timeframe": timeframe,
        "total_strategies": len(strategies),
        "comparison": comparison,
    }


async def _exec_backtest(args: dict, default_symbol: str, default_tf: str) -> dict:
    """執行策略回測"""
    from app.core.backtest import run_backtest

    symbol = args.get("symbol", default_symbol)
    timeframe = args.get("timeframe", default_tf)
    start = args.get("start_date")
    end = args.get("end_date")

    _MIN_BARS_BACKTEST = 60
    _PREFER_MIN_BARS = 500
    df = _load_local_data(symbol, timeframe, start, end)
    if (start or end) and len(df) < _PREFER_MIN_BARS:
        df_full = _load_local_data(symbol, timeframe)
        if len(df_full) > len(df):
            df = df_full
            logger.info(
                f"回測 [{symbol}]: 指定範圍不足建議量，已自動擴大至全部本地數據（{len(df)} 根）"
            )
    if df.empty:
        return {"status": "error", "message": f"找不到 {symbol} {timeframe} 的本地數據，請先同步。"}

    entry_conditions = args.get("entry_conditions", [])
    exit_conditions = args.get("exit_conditions", [])

    if not entry_conditions:
        return {"status": "error", "message": "缺少進場條件 (entry_conditions)"}
    if not exit_conditions:
        return {"status": "error", "message": "缺少出場條件 (exit_conditions)"}

    direction = args.get("direction", "long")
    stop_loss = args.get("stop_loss_pct")
    take_profit = args.get("take_profit_pct")
    capital = args.get("initial_capital", 10000)
    leverage = args.get("leverage", 1.0)

    result = run_backtest(
        df=df,
        entry_conditions=entry_conditions,
        exit_conditions=exit_conditions,
        direction=direction,
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit,
        initial_capital=capital,
        leverage=leverage,
    )
    return result.to_dict()


async def _exec_analyze_event_patterns(args: dict, default_symbol: str, default_tf: str) -> dict:
    """事件回溯統計分析 — 找出特定事件前的指標共通性"""
    import numpy as np
    import pandas as pd
    import uuid

    symbol = args.get("symbol", default_symbol)
    timeframe = args.get("timeframe", default_tf)
    event_type = args.get("event_type", "price_surge")
    threshold = args.get("threshold", 10.0)
    lookback = args.get("lookback_bars", 5)
    n_bars = args.get("n_bars", 1)
    indicator_ids = args.get("indicators", ["rsi", "macd", "adx", "rel_vol", "bb", "atr"])
    start = args.get("start_date")
    end = args.get("end_date")

    _min_bars_event = lookback + n_bars + 20
    df = _load_local_data(symbol, timeframe, start, end)
    if len(df) < _min_bars_event and (start or end):
        df_full = _load_local_data(symbol, timeframe)
        if len(df_full) >= _min_bars_event:
            df = df_full
            logger.info(f"事件分析 [{symbol}]: 指定日期範圍數據不足，已自動擴大至全部本地數據（{len(df)} 根）")
    if df.empty or len(df) < _min_bars_event:
        return {"status": "error", "message": f"數據不足（需至少 {_min_bars_event} 根 K 線）。請先同步更多歷史數據。"}

    closes = df["close"].values
    volumes = df["volume"].values
    timestamps = df["timestamp"].values

    # ── 1. 找出事件 ──
    event_indices = []

    if event_type in ("price_surge", "price_drop"):
        for i in range(n_bars, len(df)):
            pct = (closes[i] - closes[i - n_bars]) / closes[i - n_bars] * 100
            if event_type == "price_surge" and pct >= threshold:
                event_indices.append(i)
            elif event_type == "price_drop" and pct <= -threshold:
                event_indices.append(i)

    elif event_type == "volume_spike":
        vol_ma = pd.Series(volumes).rolling(20, min_periods=1).mean().values
        for i in range(20, len(df)):
            if vol_ma[i] > 0 and volumes[i] / vol_ma[i] >= threshold:
                event_indices.append(i)

    elif event_type == "volatility_expansion":
        atr_data = registry.calculate("atr", df, {"period": 14})
        if atr_data:
            atr_vals = list(atr_data.values())[0]
            atr_ma = pd.Series(atr_vals).rolling(20, min_periods=1).mean().values
            for i in range(34, len(df)):
                if atr_vals[i] is not None and atr_ma[i] and atr_ma[i] > 0:
                    if atr_vals[i] / atr_ma[i] >= threshold:
                        event_indices.append(i)

    event_indices = [i for i in event_indices if i >= lookback]

    if not event_indices:
        return {
            "status": "no_events",
            "message": f"在 {symbol} {timeframe} 中未找到符合條件的事件（{event_type}, 閾值={threshold}）",
            "suggestion": "嘗試降低閾值或使用更長的時間範圍",
        }

    # ── 2. 計算每個事件前 lookback 根 K 線的指標 ──
    indicator_data = {}
    for ind_id in indicator_ids:
        try:
            calc = registry.calculate(ind_id, df)
            if calc:
                indicator_data[ind_id] = calc
        except Exception:
            pass

    # ── 3. 統計共通性 ──
    common_patterns = {}

    for ind_id, series_dict in indicator_data.items():
        for series_name, values in series_dict.items():
            pre_event_values = []
            for idx in event_indices:
                window = values[max(0, idx - lookback):idx]
                valid = [v for v in window if v is not None and not (isinstance(v, float) and np.isnan(v))]
                if valid:
                    pre_event_values.append(np.mean(valid))

            if not pre_event_values:
                continue

            arr = np.array(pre_event_values)
            avg = float(np.mean(arr))
            med = float(np.median(arr))
            std = float(np.std(arr))
            lo = float(np.min(arr))
            hi = float(np.max(arr))

            key = f"{ind_id}_{series_name}" if series_name != ind_id else ind_id
            common_patterns[key] = {
                "indicator": ind_id,
                "series": series_name,
                "avg": round(avg, 4),
                "median": round(med, 4),
                "std": round(std, 4),
                "range": f"{round(lo, 4)} ~ {round(hi, 4)}",
                "samples": len(pre_event_values),
            }

    # 價格和量能的基礎統計
    pre_price_changes = []
    pre_volume_ratios = []
    vol_ma_20 = pd.Series(volumes).rolling(20, min_periods=1).mean().values

    for idx in event_indices:
        if idx >= lookback + 1:
            pc = (closes[idx - 1] - closes[idx - lookback - 1]) / closes[idx - lookback - 1] * 100
            pre_price_changes.append(pc)
        if vol_ma_20[idx - 1] > 0:
            pre_volume_ratios.append(volumes[idx - 1] / vol_ma_20[idx - 1])

    if pre_price_changes:
        arr = np.array(pre_price_changes)
        common_patterns["_price_change_before"] = {
            "description": f"事件前 {lookback} 根 K 線的價格變化%",
            "avg": round(float(np.mean(arr)), 2),
            "median": round(float(np.median(arr)), 2),
            "range": f"{round(float(np.min(arr)), 2)}% ~ {round(float(np.max(arr)), 2)}%",
            "samples": len(pre_price_changes),
        }

    if pre_volume_ratios:
        arr = np.array(pre_volume_ratios)
        common_patterns["_volume_ratio_before"] = {
            "description": "事件前一根 K 線的相對量能（vs 20MA）",
            "avg": round(float(np.mean(arr)), 2),
            "median": round(float(np.median(arr)), 2),
            "range": f"{round(float(np.min(arr)), 2)}x ~ {round(float(np.max(arr)), 2)}x",
            "samples": len(pre_volume_ratios),
        }

    # ── 4. 生成圖表標記 ──
    group_id = str(uuid.uuid4())[:8]
    event_label = {"price_surge": "大漲", "price_drop": "大跌", "volume_spike": "爆量", "volatility_expansion": "波動擴張"}.get(event_type, "事件")
    annotations = []

    for idx in event_indices:
        ts = str(timestamps[idx])
        pct = round((closes[idx] - closes[idx - n_bars]) / closes[idx - n_bars] * 100, 1) if event_type in ("price_surge", "price_drop") else threshold
        annotations.append({
            "type": "vertical_line",
            "startTime": ts,
            "text": f"{event_label} {pct}%",
            "color": "#f85149" if "drop" in event_type else "#3fb950",
            "groupId": group_id,
            "groupName": f"{event_label}事件 ({len(event_indices)}次)",
        })

    event_dates = [str(timestamps[i])[:10] for i in event_indices]

    return {
        "status": "success",
        "symbol": symbol,
        "timeframe": timeframe,
        "event_type": event_type,
        "threshold": threshold,
        "lookback_bars": lookback,
        "events_found": len(event_indices),
        "event_dates": event_dates[:30],
        "common_patterns": common_patterns,
        "data_range": f"{str(timestamps[0])[:10]} ~ {str(timestamps[-1])[:10]}",
        "total_bars": len(df),
        "annotations": annotations,
        "warning": "統計共通性不代表因果關係，僅供參考。樣本數越多結論越可靠。" if len(event_indices) < 10 else None,
    }


async def _exec_quant_research(args: dict, default_symbol: str, default_tf: str) -> dict:
    """完整量化研究流程：因子分析 + 回測 + Monte Carlo + Walk Forward + 倉位建議"""
    from app.core.backtest.engine import run_backtest
    from app.core.backtest.monte_carlo import run_monte_carlo
    from app.core.backtest.walk_forward import run_walk_forward
    from app.core.backtest.factor_analysis import (
        compute_factor_ic, run_factor_scan, SCANNABLE_INDICATORS,
    )
    from app.core.backtest.position_sizing import calculate_dynamic_positions

    symbol = args.get("symbol", default_symbol)
    timeframe = args.get("timeframe", default_tf)
    start = args.get("start_date")
    end = args.get("end_date")
    indicator_ids = args.get("indicators", SCANNABLE_INDICATORS)
    entry_conditions = args.get("entry_conditions", [])
    exit_conditions = args.get("exit_conditions", [])
    direction = args.get("direction", "long")
    stop_loss = args.get("stop_loss_pct")
    take_profit = args.get("take_profit_pct")
    leverage = args.get("leverage", 1.0)

    _MIN_BARS_RESEARCH = 100
    _MAX_BARS_NO_TRIM = 5000  # 日線級別不截斷，僅小時級別以下考慮
    # 量化研究優先使用全部數據，確保統計顯著性
    df_full = _load_local_data(symbol, timeframe)
    date_expanded = False
    if (start or end):
        df = _load_local_data(symbol, timeframe, start, end)
        # 只要全量數據不超過上限，就使用全量（LLM 常自作主張帶日期限制）
        if len(df_full) <= _MAX_BARS_NO_TRIM and len(df_full) > len(df):
            logger.info(
                f"量化研究 [{symbol}]: 指定範圍 {len(df)} 根，全量 {len(df_full)} 根在合理範圍內，使用全量數據"
            )
            df = df_full
            date_expanded = True
    else:
        df = df_full
    if df.empty or len(df) < _MIN_BARS_RESEARCH:
        return {"status": "error", "message": f"數據不足（{len(df)} 根 K 線），至少需要 {_MIN_BARS_RESEARCH} 根。請先同步更多歷史數據。"}

    report: dict = {"symbol": symbol, "timeframe": timeframe, "total_bars": len(df)}
    if date_expanded:
        report["notice"] = "因指定日期範圍數據不足，已自動擴大至全部本地數據進行分析。"

    # ── 1. 因子掃描（含近期 IC、Alpha Decay、衍生因子、組合 IC、分位數、相關性）──
    logger.info(f"量化研究 [{symbol}]: 因子掃描中（含衍生因子）...")
    try:
        scan = run_factor_scan(df, timeframe=timeframe, indicator_ids=indicator_ids)
        if scan.get("status") == "success":
            report["factor_scan"] = {
                "regime": scan.get("regime"),
                "total_scanned": scan.get("total_factors_scanned"),
                "effective_count": scan.get("effective_count"),
                "ic_threshold_used": scan.get("ic_threshold_used"),
                "p_value_cutoff": scan.get("p_value_cutoff"),
                "oos_split_ratio": scan.get("oos_split_ratio"),
                "positive_top": scan.get("positive_top", []),
                "negative_top": scan.get("negative_top", []),
                "combo_top": scan.get("combo_top", []),
                "quantile_analysis": scan.get("quantile_analysis"),
                "high_correlation_warnings": scan.get("high_correlation_warnings", []),
                "scan_warnings": scan.get("scan_warnings", []),
            }
            # 同時提供向下相容的簡化排名
            all_tops = scan.get("positive_top", []) + scan.get("negative_top", [])
            all_tops.sort(key=lambda x: abs(x.get("ic_recent", 0)), reverse=True)
            report["factor_ic"] = {
                "ranking": [
                    {
                        "factor": f["factor"],
                        "best_ic": f["ic_recent"],
                        "power": f["status_label"],
                        "decay_trend": f.get("decay_trend", "unknown"),
                    }
                    for f in all_tops[:10]
                ],
                "total_analyzed": scan.get("total_factors_scanned", 0),
            }
            report["factor_correlation"] = {
                "high_pairs": scan.get("high_correlation_warnings", [])[:5],
                "recommendation": (
                    "高相關因子：" + ", ".join(
                        f"{p['factor_a']}↔{p['factor_b']}"
                        for p in scan.get("high_correlation_warnings", [])[:3]
                    )
                ) if scan.get("high_correlation_warnings") else "各因子相關性低",
            }
    except Exception as e:
        logger.warning(f"因子掃描失敗，退回傳統 IC: {e}")
        try:
            ic_result = compute_factor_ic(df, indicator_ids, forward_periods=[1, 3, 5, 10, 20])
            if ic_result.get("status") == "success":
                report["factor_ic"] = {
                    "ranking": ic_result.get("ranking", [])[:8],
                    "total_analyzed": ic_result.get("total_factors_analyzed", 0),
                }
        except Exception as e2:
            report["factor_ic"] = {"error": str(e2)}

    # ── 3. 策略回測（如有條件）──
    if entry_conditions and exit_conditions:
        logger.info(f"量化研究 [{symbol}]: 策略回測中...")
        try:
            bt_result = run_backtest(
                df, entry_conditions, exit_conditions,
                direction=direction, stop_loss_pct=stop_loss, take_profit_pct=take_profit,
                leverage=leverage,
            )
            metrics = bt_result.metrics
            report["backtest"] = metrics

            # ── 4. Monte Carlo ──
            if bt_result.trades and len(bt_result.trades) >= 10:
                logger.info(f"量化研究 [{symbol}]: Monte Carlo 模擬中...")
                pnls = [t.pnl_pct for t in bt_result.trades]
                # Regime labels: 用 ATR 中位數分高/低波動
                atr_simple = (df["high"] - df["low"]).values
                atr_median = float(np.median(atr_simple))
                regime_per_bar = (atr_simple > atr_median).astype(int)
                regime_labels = [
                    int(regime_per_bar[t.entry_idx])
                    for t in bt_result.trades
                ]
                mc = run_monte_carlo(
                    pnls, n_simulations=1000, regime_labels=regime_labels,
                )
                report["monte_carlo"] = mc

            # ── 5. Walk Forward ──
            if len(df) >= 200:
                logger.info(f"量化研究 [{symbol}]: Walk Forward 分析中...")
                wf = run_walk_forward(
                    df, entry_conditions, exit_conditions,
                    direction=direction, stop_loss_pct=stop_loss, take_profit_pct=take_profit,
                    n_windows=5, leverage=leverage,
                )
                if wf.get("status") == "success":
                    report["walk_forward"] = {
                        "summary": wf.get("summary"),
                        "assessment": wf.get("assessment"),
                    }
                else:
                    report["walk_forward"] = wf
        except Exception as e:
            report["backtest"] = {"error": str(e)}

    # ── 6. 動態倉位建議（MC 回饋調控 Kelly） ──
    logger.info(f"量化研究 [{symbol}]: 動態倉位計算中...")
    try:
        win_rate = report.get("backtest", {}).get("win_rate", 50) / 100
        avg_win = report.get("backtest", {}).get("avg_win_pct", 1.5)
        avg_loss = abs(report.get("backtest", {}).get("avg_loss_pct", 1.0))
        wl_ratio = avg_win / avg_loss if avg_loss > 0 else 1.5

        # MC → Kelly cap: P5 最大回撤 > 30% 時按比例縮減
        mc_kelly_cap = 1.0
        mc_result = report.get("monte_carlo", {})
        if mc_result.get("status") == "success":
            mc_p5_dd = abs(mc_result.get("max_drawdown", {}).get("worst_5pct", 0))
            if mc_p5_dd > 30:
                mc_kelly_cap = 30.0 / mc_p5_dd

        pos = calculate_dynamic_positions(
            df, method="kelly_dynamic",
            win_rate=win_rate, avg_win_loss_ratio=wl_ratio,
            kelly_fraction=0.25 * mc_kelly_cap,
        )
        report["position_sizing"] = {
            "summary": pos.get("summary"),
            "recommendation": pos.get("recommendation"),
            "mc_adjustment": {
                "kelly_cap_factor": round(mc_kelly_cap, 3),
                "reason": f"MC P5 回撤 {mc_p5_dd:.1f}% 超過 30%，Kelly 縮減至 {mc_kelly_cap:.1%}"
                if mc_kelly_cap < 1.0
                else "無需調整",
            },
        }
    except Exception as e:
        report["position_sizing"] = {"error": str(e)}

    # ── 7. 整合結論 ──
    report["conclusion"] = _generate_conclusion(report)
    report["status"] = "success"

    return report


def _generate_conclusion(report: dict) -> dict:
    """根據所有分析結果生成整合性結論。"""
    score = 50
    findings = []

    # 因子品質
    ranking = report.get("factor_ic", {}).get("ranking", [])
    strong_factors = [f for f in ranking if "強" in f.get("power", "")]
    if strong_factors:
        score += 10
        findings.append(f"✅ 發現 {len(strong_factors)} 個強預測力因子：{', '.join(f['factor'] for f in strong_factors[:3])}")
    elif ranking:
        findings.append("⚠️ 因子預測力普遍偏弱")

    # 回測績效
    bt = report.get("backtest", {})
    if bt.get("win_rate", 0) > 55:
        score += 10
    if bt.get("sharpe_ratio", 0) > 1:
        score += 10
        findings.append(f"✅ Sharpe {bt['sharpe_ratio']} > 1，風險調整報酬佳")
    elif bt.get("sharpe_ratio", 0) > 0.5:
        score += 5
    if bt.get("sortino_ratio", 0) > 1.5:
        score += 5
        findings.append(f"✅ Sortino {bt['sortino_ratio']}，下行風險控制好")
    if bt.get("expectancy_pct", 0) > 0:
        score += 5
        findings.append(f"✅ Expectancy {bt['expectancy_pct']}% > 0，長期期望值正")
    elif bt.get("expectancy_pct", 0) <= 0 and bt.get("total_trades", 0) > 5:
        score -= 15
        findings.append(f"❌ Expectancy {bt.get('expectancy_pct', 0)}% ≤ 0，長期期望值負")

    # Monte Carlo
    mc = report.get("monte_carlo", {})
    if mc.get("confidence_level") == "low":
        findings.append("⚠️ Monte Carlo 信心度偏低（交易筆數不足 30）")
    if mc.get("strategy_robust"):
        score += 10
        findings.append(f"✅ Monte Carlo 驗證通過（獲利機率 {mc.get('profit_probability', 0)}%）")
    elif mc.get("status") == "success":
        score -= 10
        findings.append("⚠️ Monte Carlo 未通過（25%分位報酬為負）")
    if mc.get("ruin_probability", 0) > 5:
        score -= 15
        findings.append(f"❌ 破產風險 {mc['ruin_probability']}%，建議降低槓桿")
    # 壓力測試
    stress = mc.get("stress_test", {})
    if stress.get("ruin_probability", 0) > 10:
        score -= 10
        findings.append(f"❌ 壓力測試破產風險 {stress['ruin_probability']}%")
    # 回撤機率
    dd_probs = mc.get("drawdown_probabilities", {})
    if dd_probs.get("exceed_50pct", 0) > 10:
        score -= 5
        findings.append(f"⚠️ 超過 50% 回撤的機率為 {dd_probs['exceed_50pct']}%")

    # Walk Forward
    wf = report.get("walk_forward", {})
    assessment = wf.get("assessment", {})
    if assessment.get("has_alpha"):
        score += 10
        findings.append("✅ Walk Forward 驗證具備 Alpha")
    elif assessment.get("score", 0) < 40:
        score -= 10
        findings.append("❌ Walk Forward 未通過，策略可能過擬合")

    score = max(0, min(100, score))

    # 建議
    has_alpha = score >= 60
    if score >= 75:
        stability = "高"
        leverage = "1~3x（視風險承受度）"
    elif score >= 50:
        stability = "中"
        leverage = "1~2x（建議保守）"
    else:
        stability = "低"
        leverage = "不建議開槓桿"

    pos_summary = report.get("position_sizing", {}).get("summary", {})
    suggested_position = pos_summary.get("current_position_pct", "N/A")

    return {
        "overall_score": score,
        "has_alpha": has_alpha,
        "stability": stability,
        "suggested_leverage": leverage,
        "suggested_position_pct": suggested_position,
        "best_timeframe": report.get("timeframe", "N/A"),
        "findings": findings,
    }


async def _exec_optimize_params(args: dict, default_symbol: str, default_tf: str) -> dict:
    """執行指標參數校準。"""
    from app.core.backtest.parameter_optimizer import run_calibration

    symbol = args.get("symbol", default_symbol)
    timeframe = args.get("timeframe", default_tf)
    start = args.get("start_date")
    end = args.get("end_date")
    indicator_ids = args.get("indicators")
    forward_bars = args.get("forward_bars", 5)

    df = _load_local_data(symbol, timeframe, start, end)
    if df.empty or len(df) < 200:
        return {"status": "error", "message": f"數據不足（{len(df)} 根 K 線），至少需要 200 根。請先同步更多歷史數據。"}

    logger.info(f"開始校準 {symbol} {timeframe} 的指標參數...")
    result = run_calibration(
        df=df,
        symbol=symbol,
        indicator_ids=indicator_ids,
        forward_bars=forward_bars,
    )
    return result


async def _exec_conditional_prob_scan(args: dict, default_symbol: str, default_tf: str) -> dict:
    """條件機率掃描：掃描指標數值區間，計算每個區間後續 N 根 K 線漲/跌 ≥ X% 的機率"""
    import numpy as np

    symbol = args.get("symbol", default_symbol)
    timeframe = args.get("timeframe", default_tf)
    indicator_ids = args.get("indicators", ["rsi"])
    forward_bars = args.get("forward_bars", 6)
    target_pct = args.get("target_pct", 3.0)
    direction = args.get("direction", "up")
    n_bins = args.get("n_bins", 10)
    start = args.get("start_date")
    end = args.get("end_date")

    df = _load_local_data(symbol, timeframe, start, end)

    # 使用全量歷史數據以提升統計顯著性；另外保留近期 15% 子集做對比
    df_recent = None
    if df is not None and len(df) > forward_bars + 60:
        recent_bars = max(60, len(df) // 7)  # 約 15% 的數據，最少 60 根
        df_recent = df.tail(recent_bars).copy()

    if df.empty or len(df) < forward_bars + 50:
        return {"status": "error", "message": f"數據不足（{len(df)} 根 K 線），至少需要 {forward_bars + 50} 根。"}

    closes = df["close"].values.astype(float)
    n = len(closes)

    future_hit = np.zeros(n, dtype=bool)
    for i in range(n - forward_bars):
        pct = (closes[i + forward_bars] - closes[i]) / closes[i] * 100
        if direction == "up":
            future_hit[i] = pct >= target_pct
        else:
            future_hit[i] = pct <= -target_pct
    valid_range = n - forward_bars

    results_by_indicator = {}

    for ind_id in indicator_ids:
        try:
            calc = registry.calculate(ind_id, df)
            if not calc:
                continue
        except Exception:
            continue

        for series_name, values in calc.items():
            key = f"{ind_id}_{series_name}" if series_name != ind_id else ind_id
            arr = np.array([float(v) if v is not None else np.nan for v in values], dtype=float)
            valid_mask = ~np.isnan(arr[:valid_range])

            if valid_mask.sum() < 20:
                continue

            valid_vals = arr[:valid_range][valid_mask]
            lo, hi = float(np.percentile(valid_vals, 2)), float(np.percentile(valid_vals, 98))
            if hi - lo < 1e-8:
                continue

            bin_edges = np.linspace(lo, hi, n_bins + 1)
            bins = []

            best_prob = 0.0
            best_bin_label = ""

            for b in range(n_bins):
                bl, bh = bin_edges[b], bin_edges[b + 1]
                if b == n_bins - 1:
                    in_bin = valid_mask & (arr[:valid_range] >= bl) & (arr[:valid_range] <= bh)
                else:
                    in_bin = valid_mask & (arr[:valid_range] >= bl) & (arr[:valid_range] < bh)

                count = int(in_bin.sum())
                if count < 3:
                    bins.append({
                        "range": f"{bl:.2f}~{bh:.2f}", "count": count,
                        "hit": 0, "prob_pct": None, "note": "樣本不足",
                    })
                    continue

                hit_count = int(future_hit[:valid_range][in_bin].sum())
                prob = hit_count / count * 100

                label = f"{bl:.2f}~{bh:.2f}"
                bins.append({
                    "range": label, "count": count,
                    "hit": hit_count, "prob_pct": round(prob, 1),
                })

                if prob > best_prob and count >= 5:
                    best_prob = prob
                    best_bin_label = label

            baseline_prob = float(future_hit[:valid_range][valid_mask].sum()) / valid_mask.sum() * 100

            results_by_indicator[key] = {
                "indicator": ind_id,
                "series": series_name,
                "total_valid_samples": int(valid_mask.sum()),
                "baseline_prob_pct": round(baseline_prob, 1),
                "best_range": best_bin_label,
                "best_prob_pct": round(best_prob, 1),
                "lift_vs_baseline": round(best_prob - baseline_prob, 1) if best_prob > 0 else 0,
                "bins": bins,
            }

    if not results_by_indicator:
        return {"status": "error", "message": "指定的指標無法計算或數據不足"}

    # --- 近期對比（最近 120 根 K 線）---
    recent_comparison: dict | None = None
    if df_recent is not None and len(df_recent) >= forward_bars + 20:
        rc_closes = df_recent["close"].values.astype(float)
        rc_n = len(rc_closes)
        rc_hit = np.zeros(rc_n, dtype=bool)
        for i in range(rc_n - forward_bars):
            pct = (rc_closes[i + forward_bars] - rc_closes[i]) / rc_closes[i] * 100
            if direction == "up":
                rc_hit[i] = pct >= target_pct
            else:
                rc_hit[i] = pct <= -target_pct
        rc_valid = rc_n - forward_bars

        recent_comparison = {"bars": len(df_recent), "indicators": {}}
        for ind_id in indicator_ids:
            try:
                calc = registry.calculate(ind_id, df_recent)
                if not calc:
                    continue
            except Exception:
                continue
            for series_name, values in calc.items():
                key = f"{ind_id}_{series_name}" if series_name != ind_id else ind_id
                if key not in results_by_indicator:
                    continue
                arr = np.array([float(v) if v is not None else np.nan for v in values], dtype=float)
                valid_mask = ~np.isnan(arr[:rc_valid])
                if valid_mask.sum() < 10:
                    continue
                rc_baseline = float(rc_hit[:rc_valid][valid_mask].sum()) / valid_mask.sum() * 100
                recent_comparison["indicators"][key] = {
                    "recent_baseline_prob_pct": round(rc_baseline, 1),
                    "full_baseline_prob_pct": results_by_indicator[key]["baseline_prob_pct"],
                    "samples": int(valid_mask.sum()),
                }

    overall_best = max(
        results_by_indicator.values(),
        key=lambda x: x["best_prob_pct"],
    )

    dir_label = f"上漲≥{target_pct}%" if direction == "up" else f"下跌≥{target_pct}%"

    result = {
        "status": "success",
        "symbol": symbol,
        "timeframe": timeframe,
        "target": f"後續 {forward_bars} 根 K 線{dir_label}",
        "direction": direction,
        "target_pct": target_pct,
        "forward_bars": forward_bars,
        "data_range": f"{str(df['timestamp'].iloc[0])[:10]} ~ {str(df['timestamp'].iloc[-1])[:10]}",
        "total_bars": n,
        "indicators": results_by_indicator,
        "overall_best": {
            "indicator": overall_best["indicator"],
            "range": overall_best["best_range"],
            "prob_pct": overall_best["best_prob_pct"],
            "baseline_pct": overall_best["baseline_prob_pct"],
            "lift": overall_best["lift_vs_baseline"],
        },
        "warning": "條件機率不代表因果關係，高機率區間可能樣本較少。建議搭配其他分析交叉驗證。",
    }
    if recent_comparison:
        result["recent_comparison"] = recent_comparison
    return result


async def _exec_generate_scenarios(args: dict, default_symbol: str, default_tf: str) -> dict:
    """產出三大情境預測 — 整合 ML、技術指標、歷史相似度、市場結構"""
    from app.core.ml.scenario_predictor import scenario_predictor

    symbol = args.get("symbol", default_symbol)
    timeframe = args.get("timeframe", default_tf)
    forward_bars = args.get("forward_bars", 5)

    df = _load_local_data(symbol, timeframe)
    if df is None or df.empty:
        return {"status": "error", "message": f"找不到 {symbol} {timeframe} 的本地數據，請先同步"}
    if len(df) < 30:
        return {"status": "error", "message": f"數據不足（{len(df)} 根），需要至少 30 根 K 線"}

    try:
        result = scenario_predictor.predict_scenarios(
            df=df, symbol=symbol, timeframe=timeframe, forward_bars=forward_bars,
        )
        return {"status": "success", **result.to_dict()}
    except Exception as e:
        return {"status": "error", "message": f"情境預測失敗: {str(e)}"}


async def _exec_detect_smc(args: dict, default_symbol: str, default_tf: str) -> dict:
    """SMC 訂單流結構分析 — BOS/CHoCH/FVG/Sweep/MTF 全量化計算"""
    from app.core.ml.smc_detector import smc_detector

    symbol = args.get("symbol", default_symbol)
    timeframe = args.get("timeframe", default_tf)
    lookback = args.get("lookback", 120)
    htf_tf = args.get("htf")

    df = _load_local_data(symbol, timeframe)
    if df is None or df.empty:
        return {"status": "error", "message": f"找不到 {symbol} {timeframe} 的本地數據，請先同步"}
    if len(df) < 30:
        return {"status": "error", "message": f"數據不足（{len(df)} 根），需要至少 30 根 K 線"}

    # 載入 HTF 數據（多時區共振）
    df_htf = None
    if htf_tf:
        df_htf = _load_local_data(symbol, htf_tf)
    else:
        # 自動推斷 HTF
        _htf_map = {"15m": "1h", "1h": "4h", "4h": "1d", "1d": "1w"}
        inferred_htf = _htf_map.get(timeframe)
        if inferred_htf and inferred_htf != timeframe:
            df_htf = _load_local_data(symbol, inferred_htf)

    try:
        result = smc_detector.detect(
            df=df, symbol=symbol, timeframe=timeframe,
            df_htf=df_htf, lookback=lookback,
        )
        return {"status": "success", **result.to_dict()}
    except Exception as e:
        return {"status": "error", "message": f"SMC 分析失敗: {str(e)}"}
