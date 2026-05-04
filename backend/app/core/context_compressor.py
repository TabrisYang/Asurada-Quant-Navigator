"""阿斯拉量化系統 — v106 D3：Context compression。

chart_state 注入完整時 ~5000 tokens，多數欄位對「簡單查詢」沒意義。
依 intent 動態挑欄位 → token 節省 30-50%，cache hit 也更穩。

設計原則：
- **強制保留核心**：symbol / timeframe / current_price / regime / regime_confidence
  → 任何意圖都不能丟（防 LLM 失去基礎方向感）
- 其餘欄位依 intent 加白名單（additive）
- 若 intent 集為空或未知，回退「保守完整版」
- 不修改原 dict，回傳新 dict（避免汙染後續 logic）
"""

from __future__ import annotations

from typing import Optional


# ─── 各 intent 對應的 chart_state 欄位白名單 ────────────

# 任何 intent 都必須保留的核心欄位
_ALWAYS_KEEP = {
    "symbol", "timeframe", "current_price", "currentPrice", "lastClose",
    "currentRegime", "regime_subtype", "regimeWarning",
    "auto_position_multiplier",
    "priceOverview", "dataPoints",
    "calendar_meta",  # 日曆過期警告必須保留
    "stale_warning",
}

# 加密/台股共用
_ANALYSIS_FIELDS = {
    "indicatorValues", "smcStructure",
    "external_signals", "external_signals_summary",
    "social_sentiment",
    "portfolio_summary",  # v107.2：保留客觀組合風控；user_positions 已停用
    "historical_insights",
    "rl_strategic_insight",
    "upcoming_events",
    "fundamentals_summary", "drift_status",
    "crossStockSignals",
    "recent_accuracy",
    "shap_refine",
    "knowledge_fragments_context",
}

_BACKTEST_FIELDS = {
    "indicatorValues", "smcStructure",
    "calibration_summary", "drift_status",
    "external_signals_summary",
}

_REGIME_FIELDS = {
    "indicatorValues",
    "external_signals_summary",
    "drift_status",
}

_QUANT_FIELDS = _ANALYSIS_FIELDS | {"calibration_summary", "shap_refine"}


_INTENT_FIELD_MAP: dict[str, set[str]] = {
    "analysis": _ANALYSIS_FIELDS,
    "comprehensive_analysis": _ANALYSIS_FIELDS | _QUANT_FIELDS,
    "deep_analysis": _ANALYSIS_FIELDS | _QUANT_FIELDS,
    "deep_phase1": _ANALYSIS_FIELDS,
    "deep_phase2": _ANALYSIS_FIELDS,
    "deep_phase3": _ANALYSIS_FIELDS,
    "backtest": _BACKTEST_FIELDS,
    "quant_research": _QUANT_FIELDS,
    "regime_analysis": _REGIME_FIELDS,
    "momentum_analysis": _ANALYSIS_FIELDS,
    "factor_validation": _QUANT_FIELDS,
    "event_analysis": _ANALYSIS_FIELDS | {"upcoming_events", "calendar_meta"},
    "calibrate": _BACKTEST_FIELDS,
    "strategy_backtest": _BACKTEST_FIELDS,
}

# 純查詢類意圖（簡單問題、知識）→ 給最小 chart_state
_LIGHT_INTENTS = {"general", "simple_query", "knowledge_query", "system_query"}


def compress_chart_state(
    chart_state: Optional[dict],
    intents: Optional[set[str]] = None,
) -> Optional[dict]:
    """依 intent 過濾 chart_state，只留必要欄位。

    Args:
        chart_state: 完整 chart_state（可能 ~100 fields, ~5000 tokens）
        intents: 由 detect_intents 產生的意圖集

    Returns:
        新 dict（淺複製）；若 intents 為空或全是 light → 回傳極簡版
    """
    if not chart_state:
        return chart_state

    intents = intents or set()

    # 計算合集白名單
    keep = set(_ALWAYS_KEEP)

    is_light_only = bool(intents) and intents <= _LIGHT_INTENTS
    if is_light_only:
        # 只保留 ALWAYS_KEEP（已含核心欄位）
        pass
    else:
        for it in intents:
            if it in _INTENT_FIELD_MAP:
                keep |= _INTENT_FIELD_MAP[it]

        # 若 intents 包含未知標籤（保守起見）→ 全保留
        unknown = {it for it in intents if it not in _INTENT_FIELD_MAP and it not in _LIGHT_INTENTS}
        if unknown:
            return dict(chart_state)

    # 篩選
    out: dict = {}
    for k, v in chart_state.items():
        if k in keep:
            out[k] = v

    # 標記壓縮（讓 LLM 知道是精簡版，需要更多細節時可呼叫工具）
    out["_compressed"] = {
        "applied": True,
        "intents": sorted(intents) if intents else [],
        "kept_fields": len(out) - 1,
        "original_fields": len(chart_state),
    }

    return out


def estimate_token_savings(
    original: dict, compressed: dict, chars_per_token: float = 3.0,
) -> dict:
    """概算壓縮節省的 token（給 logging / observability 用）。"""
    import json
    orig_chars = len(json.dumps(original, ensure_ascii=False))
    comp_chars = len(json.dumps(compressed, ensure_ascii=False))
    return {
        "original_chars": orig_chars,
        "compressed_chars": comp_chars,
        "saved_chars": orig_chars - comp_chars,
        "estimated_token_savings": int((orig_chars - comp_chars) / chars_per_token),
        "savings_pct": round((orig_chars - comp_chars) / orig_chars * 100, 1) if orig_chars else 0,
    }
