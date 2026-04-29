"""阿斯拉量化系統 — v104 Fix B：ranging 子類型分類器。

把「ranging」regime 進一步拆成 4 個子類型，後端先算好標籤注入 chart_state，
LLM 看標籤直接選結論卡，不用自己算百分位（避免不一致）：

- true_ranging       — 真震盪（ADX 低 + BB 窄 + ATR 低 + 高低點重疊）→ 雙向計劃
- lean_long          — ranging 但偏多（大型結構多 + RSI 偏強 + breadth 多 + funding 空頭擠壓）
- lean_short         — ranging 但偏空（對稱 lean_long）
- breakout_pending   — BB 收窄 + 量縮但未明確方向 → 雙向窄區間等突破
- neutral_ranging    — 不符上述任何條件，看大型結構決定（fallback）

每條件用後端規則計算（不靠 LLM 推論），確定可重現。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from app.core.indicators import registry


# ─── 條件閾值（保留為模組常數方便調整）───
_ADX_LOW = 15           # ADX < 15 → 趨勢非常弱
_BB_WIDTH_PCTL_LOW = 30 # BB 寬度 30 百分位以下 → 窄
_ATR_PCT_PCTL_LOW = 30  # ATR% 30 百分位以下 → 低波動
_OVERLAP_HIGH = 0.7     # 高低點重疊比例 > 0.7 → 真震盪（窄區間反覆）
# v104.1：bias_score 從 5 分量擴成 9 分量後，更容易跨 0.3 →
# 升到 0.4 對應「至少 2-3 個中等訊號同向」才觸發 lean，避免雜訊放大誤觸。
# settings.bias_score_extended_dimensions=False 時（5 分量），會自動降回 0.3 baseline。
_BIAS_THRESHOLD_BASELINE = 0.3
_BIAS_THRESHOLD_EXTENDED = 0.4


def _percentile_of_last(series: pd.Series, lookback: int = 100) -> Optional[float]:
    """最後一根 K 線在過去 lookback 根中的百分位（0-100）。"""
    if series is None or len(series) < 30:
        return None
    s = series.dropna().tail(lookback)
    if len(s) < 30:
        return None
    last_val = s.iloc[-1]
    rank = (s < last_val).sum() / max(len(s) - 1, 1) * 100
    return float(rank)


def _compute_overlap_ratio(df: pd.DataFrame, lookback: int = 20) -> Optional[float]:
    """過去 lookback 根 K 線高低點重疊比例。

    range_overlap = (重疊區間長度) / (總範圍長度)
    重疊區間 = max(min_low, ...) 跟 min(max_high, ...) 的距離 / 總高低範圍
    簡化：用「最後一根 K 線高低點」跟「過去 lookback-1 根高低點區間」的重疊度。
    """
    if df is None or len(df) < lookback + 1:
        return None
    recent = df.tail(lookback)
    range_high = recent["high"].max()
    range_low = recent["low"].min()
    total_range = range_high - range_low
    if total_range <= 0:
        return None
    last_high = float(df["high"].iloc[-1])
    last_low = float(df["low"].iloc[-1])
    last_range = last_high - last_low
    # 重疊（粗估）：最後一根 range 佔總 range 的比例倒過來看 — 越小代表越窄、越在中間
    # 改用：(範圍中段佔比) — 取 25-75 百分位高低點 vs 總高低點
    q25_low = recent["low"].quantile(0.25)
    q75_high = recent["high"].quantile(0.75)
    middle_range = q75_high - q25_low
    return float(middle_range / total_range) if total_range > 0 else None


def _compute_bias_score(
    df: pd.DataFrame,
    chart_state: Optional[dict],
    symbol: Optional[str] = None,
) -> tuple[float, list[str], dict]:
    """偏多 / 偏空 score，回傳 (score, top_reasons, full_metrics)。

    score 範圍 [-1, +1]，正值偏多、負值偏空。

    分量總覽（settings.bias_score_extended_dimensions=True 時 9 個；False 時前 5 個）：
      原 5 分量（v104 baseline）：
        - 大型結構（EMA60 趨勢）：±0.3
        - 短期 RSI 數值：±0.2
        - 跨股 breadth：±0.2（v104.1 改線性 ±0.10-0.20）
        - funding rate 極端：±0.15
        - 多空持倉比極端：±0.15
      新增 4 分量（v104.1 階段 1+2）：
        - 個股 RS（vs basket）：±0.10
        - market_regime × symbol_type：±0.15
        - RSI 雙背離 + MACD 雙背離：±0.20（合計）
        - （breadth 已升級成軟加，不算新分量）

    回傳格式變更：tuple[score, top_reasons (≤3), full_metrics (9 dims)]
    full_metrics 給 debugging 用，top_reasons 給 LLM prompt 用（避免超字）。

    TODO: 權重為經驗值 — 等 verified predictions 累積到 100+ 後 walk-forward 量化最優組合。
    """
    # contributions: list of (label_short, value, abs_value) — 排序後取 top 3
    contributions: list[tuple[str, float]] = []
    full_metrics: dict = {}

    # ─── v104.1 settings flag：False 時走原 5 分量行為 ───
    try:
        from app.core.config.settings import settings as _s
        extended = getattr(_s, "bias_score_extended_dimensions", True)
        use_learned = getattr(_s, "bias_score_data_driven_weights", True)
    except Exception:
        extended = True
        use_learned = True
    full_metrics["extended_dimensions"] = extended

    # v105 Phase B：嘗試載 learned weights（若通過 quality gate 才存在）
    learned_weights = None
    if use_learned:
        try:
            from app.core.factor_weight_learner import load_learned_weights
            learned_weights = load_learned_weights()
        except Exception:
            learned_weights = None
    full_metrics["weights_source"] = "learned" if learned_weights else "heuristic"

    # ─── 分量 1：EMA60 斜率 ───
    try:
        ema_calc = registry.calculate("ema", df, {"period": 60})
        if ema_calc:
            ema_series = ema_calc.get("EMA(60)") or ema_calc.get("EMA") or []
            valid = [v for v in ema_series if v is not None]
            if len(valid) >= 10:
                slope = (valid[-1] - valid[-10]) / valid[-10]
                full_metrics["ema60_slope_pct"] = round(slope * 100, 3)
                if slope > 0.01:
                    contributions.append((f"EMA60斜率 +{slope*100:.1f}%（多）", 0.30))
                elif slope < -0.01:
                    contributions.append((f"EMA60斜率 {slope*100:.1f}%（空）", -0.30))
    except Exception:
        pass

    # ─── 分量 2：短期 RSI 數值 ───
    try:
        rsi_calc = registry.calculate("rsi", df, {"period": 14})
        if rsi_calc:
            rsi_series = rsi_calc.get("RSI") or rsi_calc.get("rsi") or []
            valid = [v for v in rsi_series if v is not None]
            if valid:
                rsi = float(valid[-1])
                full_metrics["rsi_14"] = round(rsi, 2)
                if rsi >= 60:
                    contributions.append((f"RSI={rsi:.0f}（強）", 0.20))
                elif rsi <= 40:
                    contributions.append((f"RSI={rsi:.0f}（弱）", -0.20))
    except Exception:
        pass

    # ─── 分量 3：跨股 breadth（v104.1 階段 1：軟加分尺度）───
    css = (chart_state or {}).get("crossStockSignals") or {}
    breadth = css.get("breadth_pct_advancing")
    if breadth is not None:
        try:
            b = float(breadth)
            full_metrics["breadth_pct"] = round(b, 1)
            if extended:
                # 軟加：65/55/45/35 四檔線性
                if b >= 65:
                    contributions.append((f"breadth {b:.0f}%（強多）", 0.20))
                elif b >= 55:
                    val = 0.10 + (b - 55) / 10 * 0.10  # 55→0.10, 65→0.20
                    contributions.append((f"breadth {b:.0f}%（偏多）", round(val, 3)))
                elif b <= 35:
                    contributions.append((f"breadth {b:.0f}%（強空）", -0.20))
                elif b <= 45:
                    val = -0.10 - (45 - b) / 10 * 0.10
                    contributions.append((f"breadth {b:.0f}%（偏空）", round(val, 3)))
            else:
                # 舊 5 分量行為
                if b > 60:
                    contributions.append((f"breadth {b:.0f}%（多）", 0.20))
                elif b < 40:
                    contributions.append((f"breadth {b:.0f}%（空）", -0.20))
        except Exception:
            pass

    # ─── 分量 4-5：external_signals（funding + LS）───
    ext = (chart_state or {}).get("external_signals") or {}
    deriv = ext.get("derivatives") or {}
    if "funding_rate_pct" in deriv:
        try:
            fr = float(deriv["funding_rate_pct"])
            full_metrics["funding_rate_pct"] = fr
            if fr < -0.05:
                contributions.append((f"funding={fr:.3f}%（空頭擠壓→多）", 0.15))
            elif fr > 0.05:
                contributions.append((f"funding={fr:.3f}%（多頭擠壓→空）", -0.15))
        except Exception:
            pass
    if "global_long_short_ratio" in deriv:
        try:
            r = float(deriv["global_long_short_ratio"])
            full_metrics["ls_ratio"] = round(r, 2)
            if r < 0.5:
                contributions.append((f"LS比={r:.2f}（散戶極空→多）", 0.15))
            elif r > 2.5:
                contributions.append((f"LS比={r:.2f}（散戶極多→空）", -0.15))
        except Exception:
            pass

    # ─── 階段 1 新分量 6：個股 RS（vs basket）───
    if extended:
        rs = css.get("relative_strength_vs_basket") or css.get("relative_strength_5d") or css.get("relative_strength")
        if rs is not None:
            try:
                rs_val = float(rs)
                full_metrics["rs_vs_basket"] = round(rs_val, 2)
                if rs_val > 1.2:
                    contributions.append((f"RS={rs_val:.2f}（強於 basket）", 0.10))
                elif rs_val < 0.8:
                    contributions.append((f"RS={rs_val:.2f}（弱於 basket）", -0.10))
            except Exception:
                pass

    # ─── 階段 2 分量 7：market_regime × symbol_type 矩陣 ───
    if extended and symbol:
        market_regime = css.get("market_regime")
        if market_regime:
            try:
                from app.utils.symbol import is_btc_pair, is_alt_pair
                is_btc = is_btc_pair(symbol)
                is_alt = is_alt_pair(symbol)
                full_metrics["market_regime"] = market_regime
                full_metrics["is_btc"] = is_btc
                full_metrics["is_alt"] = is_alt

                # 6-cell 矩陣
                contrib = 0.0
                label = ""
                if market_regime == "alt_season":
                    if is_btc:
                        contrib, label = -0.15, "alt_season×BTC（資金外流→空）"
                    elif is_alt:
                        contrib, label = 0.15, "alt_season×alt（資金流入→多）"
                elif market_regime == "btc_led":
                    if is_btc:
                        contrib, label = 0.15, "btc_led×BTC（領漲→多）"
                    elif is_alt:
                        contrib, label = -0.15, "btc_led×alt（落後→空）"
                elif market_regime == "bearish":
                    contrib, label = -0.15, "bearish 整體偏空"
                # mixed → 0

                if contrib != 0:
                    contributions.append((label, contrib))
            except Exception:
                pass

    # ─── 階段 2 分量 8-9：RSI/MACD 雙背離 ───
    if extended:
        rsi_div = _get_divergence_value(df, chart_state, "RSI_Div", "rsi", {"period": 14})
        macd_div = _get_divergence_value(df, chart_state, "MACD_Div", "macd", None)
        full_metrics["rsi_divergence"] = rsi_div
        full_metrics["macd_divergence"] = macd_div

        if rsi_div is not None and macd_div is not None:
            if rsi_div > 0 and macd_div > 0:
                contributions.append(("RSI+MACD 雙背離↑", 0.20))
            elif rsi_div < 0 and macd_div < 0:
                contributions.append(("RSI+MACD 雙背離↓", -0.20))
            elif rsi_div > 0 and macd_div == 0:
                contributions.append(("RSI 背離↑", 0.10))
            elif rsi_div == 0 and macd_div > 0:
                contributions.append(("MACD 背離↑", 0.10))
            elif rsi_div < 0 and macd_div == 0:
                contributions.append(("RSI 背離↓", -0.10))
            elif rsi_div == 0 and macd_div < 0:
                contributions.append(("MACD 背離↓", -0.10))
            # 不一致（一正一負）→ 抵銷不加分

    # ─── 加總 + 排序 ───
    score = sum(c[1] for c in contributions)
    score = max(-1.0, min(1.0, score))

    # top 3 by absolute contribution
    contributions.sort(key=lambda c: abs(c[1]), reverse=True)
    top_reasons = [c[0] for c in contributions[:3]]

    full_metrics["all_contributions"] = [{"label": c[0], "value": c[1]} for c in contributions]
    full_metrics["score"] = round(score, 3)

    return score, top_reasons, full_metrics


def _get_divergence_value(
    df: pd.DataFrame,
    chart_state: Optional[dict],
    field_name: str,
    indicator_id: str,
    params: Optional[dict] = None,
) -> Optional[float]:
    """背離訊號取得：先讀 chart_state.indicatorValues，找不到 fallback 直接計算。

    回傳 +1 / -1 / 0，或 None（兩條都失敗）。
    """
    # 首選：chart_state.indicatorValues 已注入
    iv = (chart_state or {}).get("indicatorValues") or {}
    if field_name in iv:
        try:
            return float(iv[field_name])
        except Exception:
            pass

    # 備援：直接呼叫 registry 計算
    try:
        result = registry.calculate(indicator_id, df, params)
        if result and field_name in result:
            series = result[field_name] or []
            valid = [v for v in series if v is not None]
            if valid:
                return float(valid[-1])
    except Exception:
        pass

    return None


def classify_ranging_subtype(
    df: pd.DataFrame,
    regime_info: dict,
    chart_state: Optional[dict] = None,
    symbol: Optional[str] = None,
) -> dict:
    """v104 Fix B：ranging 子類型分類。

    僅在 regime == "ranging" 時呼叫；其他 regime 直接 return None。
    回傳 dict：
      {
        "subtype": "true_ranging" | "lean_long" | "lean_short" |
                   "breakout_pending" | "neutral_ranging",
        "confidence": 0-1,
        "reason": "短描述",
        "metrics": {adx, bb_width_pctl, atr_pct_pctl, overlap_ratio, bias_score},
      }
    """
    # v104：ranging + unknown 兩種模糊情境都跑子分類
    # （unknown = 結構分類器無法判定 HH/HL/LH/LL，本質跟 ranging 同樣需要拆細）
    if regime_info.get("regime") not in ("ranging", "unknown"):
        return {"subtype": None, "reason": "regime != ranging/unknown"}

    if df is None or len(df) < 60:
        return {"subtype": "neutral_ranging", "reason": "資料不足", "confidence": 0.3}

    metrics: dict = {}

    # 1. ADX（從 indicator registry）
    adx_value: Optional[float] = None
    try:
        adx_calc = registry.calculate("adx", df, {"period": 14})
        if adx_calc:
            adx_series = adx_calc.get("ADX") or []
            valid = [v for v in adx_series if v is not None]
            if valid:
                adx_value = float(valid[-1])
    except Exception:
        pass
    metrics["adx"] = adx_value

    # 2. BB 寬度百分位
    bb_width_pctl: Optional[float] = None
    try:
        bb_calc = registry.calculate("bb", df)
        if bb_calc:
            upper = pd.Series(bb_calc.get("BB_Upper") or [])
            lower = pd.Series(bb_calc.get("BB_Lower") or [])
            middle = pd.Series(bb_calc.get("BB_Middle") or [])
            if len(upper) > 0 and len(lower) > 0 and len(middle) > 0:
                width = (upper - lower) / middle
                bb_width_pctl = _percentile_of_last(width, 100)
    except Exception:
        pass
    metrics["bb_width_pctl"] = bb_width_pctl

    # 3. ATR% 百分位
    atr_pct_pctl: Optional[float] = None
    try:
        atr_calc = registry.calculate("atr", df, {"period": 14})
        if atr_calc:
            atr_series = pd.Series(atr_calc.get("ATR") or [])
            close_series = df["close"].reset_index(drop=True)
            if len(atr_series) > 0 and len(close_series) == len(atr_series):
                atr_pct = atr_series / close_series
                atr_pct_pctl = _percentile_of_last(atr_pct, 100)
    except Exception:
        pass
    metrics["atr_pct_pctl"] = atr_pct_pctl

    # 4. 高低點重疊比例
    overlap = _compute_overlap_ratio(df, 20)
    metrics["overlap_ratio"] = overlap

    # 5. 偏向 score（給 lean_long/short 判定用 — v104.1：9 分量含 symbol-aware）
    bias_score, bias_reasons, bias_full = _compute_bias_score(df, chart_state, symbol=symbol)
    metrics["bias_score"] = round(bias_score, 3)
    metrics["bias_reasons"] = bias_reasons  # top 3，給 LLM prompt 用
    metrics["bias_full_metrics"] = bias_full  # 完整 9 分量，給除錯 / UI 展開用

    # ─── 判定（先檢真震盪 → 突破待發 → 偏向 → 中性）───

    # 真震盪：4 條件至少 3 條成立
    true_ranging_count = 0
    if adx_value is not None and adx_value < _ADX_LOW:
        true_ranging_count += 1
    if bb_width_pctl is not None and bb_width_pctl < _BB_WIDTH_PCTL_LOW:
        true_ranging_count += 1
    if atr_pct_pctl is not None and atr_pct_pctl < _ATR_PCT_PCTL_LOW:
        true_ranging_count += 1
    if overlap is not None and overlap > _OVERLAP_HIGH:
        true_ranging_count += 1

    if true_ranging_count >= 3:
        return {
            "subtype": "true_ranging",
            "confidence": min(0.9, 0.5 + 0.1 * true_ranging_count),
            "reason": f"4 條件中 {true_ranging_count} 條成立（真震盪）",
            "metrics": metrics,
        }

    # v104.1：依 extended flag 動態切 threshold（baseline 0.3 / extended 0.4）
    bias_threshold = _BIAS_THRESHOLD_EXTENDED if bias_full.get("extended_dimensions") else _BIAS_THRESHOLD_BASELINE

    # 突破待發：BB 收窄到極致（pctl < 20）但 ADX 未起，方向不明
    if (bb_width_pctl is not None and bb_width_pctl < 20
            and (adx_value is None or adx_value < 20)
            and abs(bias_score) < bias_threshold):
        return {
            "subtype": "breakout_pending",
            "confidence": 0.65,
            "reason": f"BB 寬度 {bb_width_pctl:.0f} 百分位（極窄）+ ADX={adx_value or 0:.0f}（弱），等突破",
            "metrics": metrics,
        }

    # 偏多 / 偏空
    if bias_score >= bias_threshold:
        return {
            "subtype": "lean_long",
            "confidence": min(0.85, 0.5 + abs(bias_score) * 0.5),
            "reason": f"bias={bias_score:+.2f}：" + "、".join(bias_reasons[:3]),
            "metrics": metrics,
        }
    if bias_score <= -bias_threshold:
        return {
            "subtype": "lean_short",
            "confidence": min(0.85, 0.5 + abs(bias_score) * 0.5),
            "reason": f"bias={bias_score:+.2f}：" + "、".join(bias_reasons[:3]),
            "metrics": metrics,
        }

    # 中性 ranging（fallback）
    return {
        "subtype": "neutral_ranging",
        "confidence": 0.5,
        "reason": f"條件不明確：true_ranging={true_ranging_count}/4、bias={bias_score:+.2f}",
        "metrics": metrics,
    }
