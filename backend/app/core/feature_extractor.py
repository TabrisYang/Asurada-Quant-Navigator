"""阿斯拉量化系統 — v101 統一特徵抽取器（Phase 2.0b）

抽取 39 個特徵作為模仿學習訓練資料，分層設計（Q4 解法）：
- 結構性 / 離散事件特徵 → 給規則融合用
- 統計性 / 連續特徵 → 給 LightGBM 用
兩者誤差來源不同（規則錯在「結構誤判」，ML 錯在「數值雜訊」），
相關性從 ~0.7 降到 ~0.4，ensemble 收益增加 100-200%。

特徵分類（39 個）：
  動量超買超賣 (4) / 趨勢 (4) / 波動率 (4) / 量能 (3) / 結構 (4)
  regime one-hot (7) / 跨股廣度 (4) / 預測 metadata (6) / 歷史校準 (3)

回填可從 OHLCV 重算，**不需要當時的 chart_state 即時快照**（指標都 deterministic）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from app.core.indicators import registry
from app.core.prediction_tracker import prediction_tracker

# v101 特徵 schema 順序（必須跟 prediction_features 表的欄位一致）
FEATURE_COLUMNS = [
    # 動量 (4)
    "rsi_14", "stoch_rsi", "macd_hist", "macd_signal_diff",
    # 趨勢 (4)
    "adx_14", "ema_20_60_diff_pct", "supertrend_dir", "plus_minus_di",
    # 波動率 (4)
    "atr_pct", "bb_width", "bb_position", "hv_30",
    # 量能 (3)
    "volume_ratio", "obv_slope_10", "cvd_change",
    # 結構 (4)
    "smc_bias", "decomp_trend_slope", "decomp_residual_vol", "fvg_count",
    # regime one-hot + confidence (7)
    "regime_trending_up", "regime_trending_down", "regime_ranging",
    "regime_high_vol", "regime_low_vol", "regime_unknown", "regime_confidence",
    # 跨股廣度 (4)
    "breadth_pct", "btc_dominance_pct", "rs_vs_basket", "leadership_score",
    # 預測 metadata (6)
    "direction_long", "confidence_score", "timeframe_hours",
    "target_distance_pct", "stop_distance_pct", "rr_ratio",
    # 歷史校準 (3)
    "recent_30d_winrate", "recent_n", "calibration_brier",
]


def extract_features_at(
    df: pd.DataFrame,
    chart_state: Optional[dict],
    prediction: dict,
) -> dict:
    """從 OHLCV df + 可選的 chart_state + 預測本身 抽 39 個特徵。

    df 應該是預測時點以前的歷史 K 線（不含未來）。
    chart_state 可選 — 若有就優先用，否則從 df 重算。
    prediction 必含 entry_price / target_price / stop_price / direction / confidence / timeframe_hours。

    回傳 dict（39 個 key），失敗欄位用 None 填。
    """
    features: dict = {col: None for col in FEATURE_COLUMNS}

    if df is None or df.empty or len(df) < 30:
        return features

    try:
        last_close = float(df["close"].iloc[-1])
    except Exception:
        return features

    # ─── 動量 (4) ───
    rsi = _last_indicator(df, "rsi", "RSI", {"period": 14})
    features["rsi_14"] = rsi
    stoch_rsi = _last_indicator(df, "stochrsi", "StochRSI_K")
    features["stoch_rsi"] = stoch_rsi
    macd = _calc_indicator(df, "macd")
    if macd:
        hist = _last_valid(macd.get("MACD_Hist") or macd.get("Histogram") or [])
        features["macd_hist"] = hist
        macd_line = _last_valid(macd.get("MACD") or [])
        signal_line = _last_valid(macd.get("Signal") or macd.get("MACD_Signal") or [])
        if macd_line is not None and signal_line is not None:
            features["macd_signal_diff"] = macd_line - signal_line

    # ─── 趨勢 (4) ───
    adx_calc = _calc_indicator(df, "adx")
    if adx_calc:
        features["adx_14"] = _last_valid(adx_calc.get("ADX") or [])
        plus_di = _last_valid(adx_calc.get("+DI") or [])
        minus_di = _last_valid(adx_calc.get("-DI") or [])
        if plus_di is not None and minus_di is not None:
            features["plus_minus_di"] = plus_di - minus_di
    ema_20 = _last_indicator(df, "ema", "EMA(20)", {"period": 20})
    ema_60 = _last_indicator(df, "ema", "EMA(60)", {"period": 60})
    if ema_20 and ema_60 and last_close:
        features["ema_20_60_diff_pct"] = (ema_20 - ema_60) / last_close
    st_calc = _calc_indicator(df, "supertrend")
    if st_calc:
        st_dir = _last_valid(st_calc.get("Direction") or st_calc.get("Trend") or [])
        features["supertrend_dir"] = int(st_dir) if st_dir is not None else None

    # ─── 波動率 (4) ───
    atr = _last_indicator(df, "atr", "ATR", {"period": 14})
    if atr is not None and last_close:
        features["atr_pct"] = atr / last_close
    bb_calc = _calc_indicator(df, "bb")
    if bb_calc:
        upper = _last_valid(bb_calc.get("BB_Upper") or [])
        lower = _last_valid(bb_calc.get("BB_Lower") or [])
        middle = _last_valid(bb_calc.get("BB_Middle") or [])
        if upper is not None and lower is not None and middle:
            features["bb_width"] = (upper - lower) / middle
            if upper != lower:
                features["bb_position"] = (last_close - lower) / (upper - lower)
    # HV30 用 close pct change std
    if len(df) >= 30:
        ret = df["close"].pct_change().dropna().tail(30)
        features["hv_30"] = float(ret.std() * np.sqrt(252))

    # ─── 量能 (3) ───
    if "volume" in df.columns and len(df) >= 20:
        vol_sma = df["volume"].tail(20).mean()
        if vol_sma > 0:
            features["volume_ratio"] = float(df["volume"].iloc[-1] / vol_sma)
    obv_calc = _calc_indicator(df, "obv")
    if obv_calc and len(df) >= 10:
        obv_series = obv_calc.get("OBV") or []
        valid = [v for v in obv_series[-10:] if v is not None]
        if len(valid) >= 5:
            features["obv_slope_10"] = float((valid[-1] - valid[0]) / max(abs(valid[0]), 1))
    # CVD change（簡化：tail(5) - tail(20)）
    cvd_calc = _calc_indicator(df, "cvd")
    if cvd_calc:
        cvd_series = cvd_calc.get("CVD") or []
        if len(cvd_series) >= 20:
            recent = _last_valid(cvd_series[-5:])
            past = _last_valid(cvd_series[-20:-15])
            if recent is not None and past is not None:
                features["cvd_change"] = float(recent - past)

    # ─── 結構 (4) ───
    # smc_bias 從 chart_state 取（若有，否則 0）
    if chart_state:
        smc = chart_state.get("smcStructure") or {}
        bias_str = smc.get("bias", "").upper()
        features["smc_bias"] = (
            1 if bias_str in {"BUY", "LONG", "BULLISH"}
            else -1 if bias_str in {"SELL", "SHORT", "BEARISH"}
            else 0
        )
        # 從 chart_state.decomposition 取 STL 結果
        decomp = chart_state.get("decomposition") or {}
        features["decomp_trend_slope"] = decomp.get("trend_slope_pct_recent_10bars")
        features["decomp_residual_vol"] = decomp.get("residual_volatility_pct")
        features["fvg_count"] = (smc.get("fvg_count") or 0)
    else:
        features["smc_bias"] = 0
        features["fvg_count"] = 0

    # ─── regime one-hot + confidence (7) ───
    regime = "unknown"
    regime_conf = 0.0
    if chart_state and chart_state.get("currentRegime"):
        regime = chart_state["currentRegime"].get("regime", "unknown")
        regime_conf = float(chart_state["currentRegime"].get("confidence", 0))
    else:
        # 從 df 重算
        try:
            from app.core.regime_filter import classify_regime_at_bar
            r = classify_regime_at_bar(df, len(df) - 1)
            regime = r.get("regime", "unknown")
            regime_conf = float(r.get("confidence", 0))
        except Exception:
            pass
    features["regime_trending_up"] = 1 if regime == "trending_up" else 0
    features["regime_trending_down"] = 1 if regime == "trending_down" else 0
    features["regime_ranging"] = 1 if regime == "ranging" else 0
    features["regime_high_vol"] = 1 if regime == "high_vol" else 0
    features["regime_low_vol"] = 1 if regime == "low_vol" else 0
    features["regime_unknown"] = 1 if regime == "unknown" else 0
    features["regime_confidence"] = regime_conf

    # ─── 跨股廣度 (4)（從 chart_state.crossStockSignals 取）───
    if chart_state:
        css = chart_state.get("crossStockSignals") or {}
        features["breadth_pct"] = css.get("breadth_pct_advancing")
        features["btc_dominance_pct"] = css.get("btc_dominance_proxy_pct")
        rs_field = css.get("relative_strength_5d") or css.get("relative_strength")
        features["rs_vs_basket"] = rs_field
        # leadership_score：強勢=1 弱勢=-1 中性=0
        leader = (css.get("leadership_signal") or "").lower()
        features["leadership_score"] = (
            1 if leader in {"strong", "強勢"}
            else -1 if leader in {"weak", "弱勢"}
            else 0
        )

    # ─── 預測 metadata (6) ───
    direction = prediction.get("direction", "long")
    features["direction_long"] = 1 if direction == "long" else 0
    conf = prediction.get("confidence", "medium")
    features["confidence_score"] = (
        2 if conf == "high" else 1 if conf == "medium" else 0
    )
    features["timeframe_hours"] = int(prediction.get("timeframe_hours", 48))
    entry = float(prediction.get("entry_price", 0))
    target = float(prediction.get("target_price", 0))
    stop = float(prediction.get("stop_price", 0))
    if entry > 0:
        features["target_distance_pct"] = (target - entry) / entry
        features["stop_distance_pct"] = (stop - entry) / entry
        risk = abs(entry - stop)
        reward = abs(target - entry)
        features["rr_ratio"] = reward / risk if risk > 0 else None

    # ─── 歷史校準 (3) ───
    try:
        symbol = prediction.get("symbol") or ""
        stats_30d = prediction_tracker.get_stats(symbol=symbol, days=30) if symbol else {}
        features["recent_30d_winrate"] = stats_30d.get("win_rate_weighted")
        features["recent_n"] = stats_30d.get("total")
        features["calibration_brier"] = stats_30d.get("calibration", {}).get("brier_score")
    except Exception:
        pass

    return features


def record_features(prediction_id: int, df: pd.DataFrame, chart_state: Optional[dict],
                    prediction: dict, source: str = "realtime") -> bool:
    """抽特徵 + 寫入 prediction_features 表。

    Returns: True 成功 / False 失敗（不冒泡 — 失敗只 log，不影響預測流程）。
    """
    try:
        features = extract_features_at(df, chart_state, prediction)
        _insert_to_db(prediction_id, features, source)
        return True
    except Exception as e:
        logger.warning(f"record_features 失敗（不影響預測）: {e}")
        return False


def _insert_to_db(prediction_id: int, features: dict, source: str) -> None:
    """寫入 prediction_features 表（INSERT OR REPLACE 容忍重複）。"""
    if not prediction_tracker._conn:
        return

    cols = ["prediction_id"] + FEATURE_COLUMNS + ["snapshot_at", "source"]
    values = [prediction_id] + [features.get(c) for c in FEATURE_COLUMNS] + [
        datetime.now().isoformat(), source,
    ]
    placeholders = ", ".join("?" * len(cols))
    cols_sql = ", ".join(cols)

    prediction_tracker._conn.execute(
        f"INSERT OR REPLACE INTO prediction_features ({cols_sql}) VALUES ({placeholders})",
        values,
    )
    prediction_tracker._conn.commit()


# ─── helper ───────────────────────────────────────────────


def _calc_indicator(df: pd.DataFrame, indicator_id: str, params: Optional[dict] = None):
    """安全包裝 registry.calculate"""
    try:
        return registry.calculate(indicator_id, df, params)
    except Exception:
        return None


def _last_indicator(df: pd.DataFrame, indicator_id: str, series_key: str,
                    params: Optional[dict] = None) -> Optional[float]:
    """取某 indicator 的某 series 最後一個有效值"""
    calc = _calc_indicator(df, indicator_id, params)
    if not calc or series_key not in calc:
        return None
    return _last_valid(calc[series_key])


def _last_valid(values: list) -> Optional[float]:
    if not values:
        return None
    for v in reversed(values):
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f != f:  # NaN
            continue
        return f
    return None
