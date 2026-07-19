"""chat 路由的上下文組裝層（v157 拆分：從 chat.py 機械搬移，邏輯零改動）。

職責：把後端算得出來的客觀數據注入 chart_state（指標值 / regime / 外部訊號 /
機率三聯 / ML 預測），以及依意圖組裝送給 LLM 的 messages。

⚠️ 本檔含大量 chart_state[X] 注入 — 新增欄位前必讀 backend/docs/CHART_STATE_SCHEMA.md，
check_repo_health.py 與 test_contracts_chart_state_schema.py 都會掃描本檔。
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from app.core.config.settings import settings
from app.core.knowledge_distiller import knowledge_distiller
from app.core.user_strategies import get_enabled_strategies_prompt
from app.core.prediction_tracker import prediction_tracker
from app.core.prediction_feedback import (
    generate_feedback_prompt, get_active_predictions_summary,
)
from app.core.backtest.parameter_optimizer import format_calibration_for_prompt
from app.models.schemas import ChatRequest
from app.api.routes.chat_text import _INDICATOR_TEXT_MAP

# 對話歷史最多保留的訊息數（避免 token 過多）
MAX_HISTORY_MESSAGES = 20


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

    # v156.2：補全觸發清單 — 原本 conditional_prob/scenario/smc 等分析型意圖不在
    # 清單內 → 後端不補算指標 → AI 只能回「未啟用 RSI/EMA/ATR」或靠自覺開指標
    #（機率性，Codex 等模型服從度不一）。改為所有分析型意圖都確定性補算。
    is_deep = bool(_intents & {
        "backtest", "quant_research", "event_analysis", "calibrate",
        "deep_analysis", "deep_phase1", "deep_phase2", "deep_phase3",
        "comprehensive_analysis",
    })
    is_analysis = bool(_intents & {
        "analysis", "conditional_prob", "scenario", "smc",
        "momentum_analysis", "factor_validation", "strategy_backtest",
        "regime_analysis",
    })

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

                        # ★ 選項2：注入 30d decided 勝率（排除 expired）。
                        # win_rate_30d=win_rate_weighted 把 expired 放進分母當「非勝」，
                        # expired 佔比高時數字結構性偏低，誤觸 v100「命中率嚴重偏低」警示。
                        # decided 勝率 = hit_target/(hit_target+hit_stop) 才是真實方向準度。複用 v124 函式。
                        try:
                            _wci_30 = prediction_tracker.get_winrate_with_ci(
                                symbol=chart_symbol, days=30,
                            )
                            if _wci_30.get("status") == "ok":
                                request.chart_state["recent_accuracy"].update({
                                    "win_rate_decided_30d": _wci_30.get("win_rate_raw_pct"),
                                    "n_decided_30d": _wci_30.get("n_decided"),
                                    "expired_30d": _wci_30.get("expired"),
                                    "ci_30d": _wci_30.get("ci_pct"),
                                })
                        except Exception as _dec_err:
                            logger.debug(f"decided 勝率注入失敗: {_dec_err}")

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
            # v145：碎片標註來源標的（避免把別標的的勝率當成當前標的的歷史勝率 = Q8 前視偏差）
            frag_texts = [
                f"• [{f['type']}｜來源:{f.get('symbol', '通用')}] {f['content']}（相關度 {f['similarity']:.0%}）"
                for f in rag_fragments
            ]
            context_parts.append(
                "【歷史分析經驗碎片】\n" + "\n".join(frag_texts) + "\n"
                "⚠️ 碎片隔離鐵則：碎片內的勝率/IC/統計數字**只屬於該碎片的來源標的與當時行情**，"
                "**嚴禁**當成「當前標的的歷史勝率」引用。當前標的的真實勝率/命中率以 "
                "chart_state.recent_accuracy / probability_triplet（track_record）為唯一依據。"
                "引用碎片數字時必須標明來源標的（如「BTC 歷史上…」），不可裸寫「歷史勝率 67%」。"
            )

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
    elif request.mode == "smc_structure":
        smc_prefix = (
            "[系統指令：使用者點擊了「SMC 結構分析」按鈕]\n"
            "你必須呼叫 detect_smc_structure 取得 SMC 智慧資金結構數據。\n"
            "專注報告：BOS / CHoCH 結構破壞、訂單塊 (Order Block) 位置、"
            "公平價值缺口 (FVG)、流動性掃蕩 (Liquidity Sweep)、"
            "並依結構提出明確進場 / 止損 / 止盈位。\n\n"
            f"使用者備註：{request.message if request.message.strip() else '請分析當前標的的 SMC 智慧資金結構'}"
        )
        messages.append({"role": "user", "content": smc_prefix})
    elif request.mode == "scenario_predict":
        sp_prefix = (
            "[系統指令：使用者點擊了「三情境預測」按鈕]\n"
            "你必須呼叫 generate_scenarios 取得三情境機率預測。\n"
            "專注報告：看漲 / 中性 / 看跌三情境的機率分布、各情境的價格目標、"
            "歷史相似度匹配、ML 模型輔助結論、各情境對應建議倉位。\n\n"
            f"使用者備註：{request.message if request.message.strip() else '請預測當前標的未來三種可能情境'}"
        )
        messages.append({"role": "user", "content": sp_prefix})
    elif request.mode == "conditional_prob":
        cp_prefix = (
            "[系統指令：使用者點擊了「條件機率掃描」按鈕]\n"
            "你必須呼叫 scan_conditional_probability 取得指標條件機率數據。\n"
            "專注報告：各指標在不同數值區間下，後續 N 根 K 線達到 X% 漲跌的歷史機率、"
            "Wilson 95% 信賴區間、Bayesian shrinkage 後的穩健機率、"
            "並找出 lift 最大且樣本充足的條件區間作為進場參考。\n\n"
            f"使用者備註：{request.message if request.message.strip() else '請掃描當前標的各指標的條件機率，找出最佳進場區間'}"
        )
        messages.append({"role": "user", "content": cp_prefix})
    elif request.mode == "event_pattern":
        # v135：強制兩步驟（先 query_chart_data 取最新價，再 analyze_event_patterns）
        # + 主動計算相似度（禁止反問使用者）+ 禁互動式延伸引導（連跑模式不該需要再輸入）
        ep_prefix = (
            "[系統指令：使用者點擊了「事件型態分析」按鈕]\n"
            "你必須執行以下兩步驟：\n"
            "1. **先呼叫 query_chart_data** 取得當前最新價格與 30 天區間"
            "（不可只用 chart_state 的快照，快照可能滯後幾分鐘到幾十分鐘）\n"
            "2. **再呼叫 analyze_event_patterns** 取得歷史事件前的 K 線指標共通特徵\n"
            "報告必須完成：\n"
            "  - 歷史大漲 / 大跌 / 爆量事件前 N 根 K 線的指標分布\n"
            "  - 共通技術特徵（如 RSI / MACD / 量能 / 結構）\n"
            "  - **主動計算當前 vs 歷史 pattern 的相似度 %**，給出觸發機率評估"
            "（不可只列歷史不算當前）\n"
            "**禁止**輸出「可以說 X」「可以問我 Y」「想進一步可以...」「你可以告訴我...」"
            "這類等候使用者再次輸入的延伸提示 — 本次必須一次完成所有相關計算與結論。\n\n"
            f"使用者備註：{request.message if request.message.strip() else '請分析當前標的歷史大漲大跌前的共通特徵'}"
        )
        messages.append({"role": "user", "content": ep_prefix})
    elif request.mode == "compute_laddered":
        cl_prefix = (
            "[系統指令：使用者點擊了「分批進場規劃」按鈕]\n"
            "你必須呼叫 compute_laddered_entries 取得分批進場價位規劃。\n"
            "專注報告：依當前 regime 自動選擇配比策略（金字塔加碼 / 倒金字塔 / 均分）、"
            "各檔進場價位（含技術依據如 SMC OB / EMA / BB 中軌）、"
            "加權均價、止損 / 止盈、風險回報比、建議倉位大小。\n\n"
            f"使用者備註：{request.message if request.message.strip() else '請規劃當前標的的分批進場價位'}"
        )
        messages.append({"role": "user", "content": cl_prefix})
    elif request.mode == "bollinger_full":
        # v137：完整布林通道符合度分析（綜合 8 維特徵 + 4 策略評分 + 最佳策略 entry/exit）
        bf_prefix = (
            "[系統指令：使用者點擊了「布林通道完整度分析」按鈕]\n"
            "你必須執行以下步驟：\n"
            "1. 先呼叫 query_chart_data 取得當前最新 OHLCV + BB / SMA20 / ATR / 量能數據\n"
            "2. 從數據算 8 個布林通道核心特徵：\n"
            "   - PctB (bb_position): 當前 %B 值（0-100 區間位置）\n"
            "   - PctB_lag1: 前一根 %B（跨軌瞬間判斷）\n"
            "   - Bandwidth_ROC: bb_width 近 4 根變動率（波動爆發）\n"
            "   - Z_Score_20: (close - sma20) / std20（σ 偏離倍數）\n"
            "   - ATR_Ratio: atr14 / 過去 50 根平均 atr14\n"
            "   - OBV_Slope_10: 近 10 根 OBV 線性回歸斜率（量能動能）\n"
            "   - is_squeeze: BB 是否收進 Keltner（squeeze 標準定義）\n"
            "   - squeeze_duration: squeeze 連續根數\n"
            "3. 評分 4 個布林策略各自的當下匹配度（0-100 分）：\n"
            "   - The Squeeze 待發程度（is_squeeze=True 加分 / squeeze_duration 越長越高）\n"
            "   - Squeeze Breakout 剛發生程度（prev is_squeeze + 本根釋放 + bandwidth_roc>0 + 量配合）\n"
            "   - Walk the Band 進行程度（ADX>25 + 連續觸軌）\n"
            "   - Mean Reversion 觸發程度（ranging regime + 觸軌後回到通道）\n"
            "4. 選最高分策略，依該策略給對應 entry / stop / target / RR\n"
            "5. 標明當前 regime + 該策略在此 regime 的歷史適配度\n"
            "報告主結論必須含「**綜合完整度評分 X / 100**」（4 個策略最高分），\n"
            "並列出 8 個特徵當下值與符合方向（✅/⚠️/❌）。\n"
            "**禁止**反問使用者或輸出延伸引導句型，本次必須一次完成所有計算。\n\n"
            f"使用者備註：{request.message if request.message.strip() else '請評估當前標的的完整布林通道符合度'}"
        )
        messages.append({"role": "user", "content": bf_prefix})
    elif request.mode == "sector_analysis":
        sec_prefix = (
            "[系統指令：使用者點擊了「族群分析」按鈕]\n"
            "你必須呼叫 analyze_sector 取得族群指數技術分析數據。\n"
            "專注報告：族群指數的 Regime、Breadth（多少成分股呈現多頭）、"
            "族群內個股相對強弱排名、族群龍頭股辨識、"
            "並判斷該族群是否具備族群行情（族群 RS > 大盤）。\n\n"
            f"使用者備註：{request.message if request.message.strip() else '請分析當前標的所屬族群的技術面與內部強弱排名'}"
        )
        messages.append({"role": "user", "content": sec_prefix})
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
