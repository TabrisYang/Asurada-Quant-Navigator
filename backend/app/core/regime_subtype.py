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
_BIAS_THRESHOLD = 0.5   # bias score 絕對值 > 0.5 才標為 lean_long/lean_short


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


def _compute_bias_score(df: pd.DataFrame, chart_state: Optional[dict]) -> tuple[float, list[str]]:
    """偏多 / 偏空 score，回傳 (score, reasons)。

    score 範圍 [-1, +1]，正值偏多、負值偏空。
    各分量加權：
      - 大型結構（EMA60 趨勢）：±0.3
      - 短期 RSI：±0.2
      - 跨股 breadth：±0.2
      - funding rate（多空擠壓潛力）：±0.15
      - 多空持倉比極端：±0.15
    """
    score = 0.0
    reasons: list[str] = []

    # 1. 大型結構 — EMA60 斜率
    try:
        ema_calc = registry.calculate("ema", df, {"period": 60})
        if ema_calc:
            ema_series = ema_calc.get("EMA(60)") or ema_calc.get("EMA") or []
            valid = [v for v in ema_series if v is not None]
            if len(valid) >= 10:
                slope = (valid[-1] - valid[-10]) / valid[-10]
                if slope > 0.01:  # +1% over 10 bars
                    score += 0.3
                    reasons.append(f"EMA60 斜率 +{slope*100:.1f}%（多）")
                elif slope < -0.01:
                    score -= 0.3
                    reasons.append(f"EMA60 斜率 {slope*100:.1f}%（空）")
    except Exception:
        pass

    # 2. 短期 RSI
    try:
        rsi_calc = registry.calculate("rsi", df, {"period": 14})
        if rsi_calc:
            rsi_series = rsi_calc.get("RSI") or rsi_calc.get("rsi") or []
            valid = [v for v in rsi_series if v is not None]
            if valid:
                rsi = float(valid[-1])
                if rsi >= 60:
                    score += 0.2
                    reasons.append(f"RSI={rsi:.0f}（強）")
                elif rsi <= 40:
                    score -= 0.2
                    reasons.append(f"RSI={rsi:.0f}（弱）")
    except Exception:
        pass

    # 3. 跨股 breadth（chart_state 已有）
    if chart_state:
        css = chart_state.get("crossStockSignals") or {}
        breadth = css.get("breadth_pct_advancing")
        if breadth is not None:
            try:
                b = float(breadth)
                if b > 60:
                    score += 0.2
                    reasons.append(f"breadth {b:.0f}%（多）")
                elif b < 40:
                    score -= 0.2
                    reasons.append(f"breadth {b:.0f}%（空）")
            except Exception:
                pass

    # 4. funding rate（極端偏空 → 空頭擠壓潛力 → 偏多）
    if chart_state:
        ext = chart_state.get("external_signals") or {}
        deriv = ext.get("derivatives") or {}
        if "funding_rate_pct" in deriv:
            fr = deriv["funding_rate_pct"]
            if fr < -0.05:  # < -0.05%（空方付錢）→ 空頭擠壓潛力 → 偏多
                score += 0.15
                reasons.append(f"funding={fr:.3f}%（空頭擠壓→偏多）")
            elif fr > 0.05:
                score -= 0.15
                reasons.append(f"funding={fr:.3f}%（多頭擠壓→偏空）")

        # 5. 多空持倉比極端
        if "global_long_short_ratio" in deriv:
            r = deriv["global_long_short_ratio"]
            if r < 0.5:  # 散戶極端空 → 反向偏多
                score += 0.15
                reasons.append(f"LS比={r:.2f}（散戶極空→反向偏多）")
            elif r > 2.5:
                score -= 0.15
                reasons.append(f"LS比={r:.2f}（散戶極多→反向偏空）")

    # clip 到 [-1, 1]
    score = max(-1.0, min(1.0, score))
    return score, reasons


def classify_ranging_subtype(
    df: pd.DataFrame,
    regime_info: dict,
    chart_state: Optional[dict] = None,
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
    if regime_info.get("regime") != "ranging":
        return {"subtype": None, "reason": "regime != ranging"}

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

    # 5. 偏向 score（給 lean_long/short 判定用）
    bias_score, bias_reasons = _compute_bias_score(df, chart_state)
    metrics["bias_score"] = round(bias_score, 3)
    metrics["bias_reasons"] = bias_reasons

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

    # 突破待發：BB 收窄到極致（pctl < 20）但 ADX 未起，方向不明
    if (bb_width_pctl is not None and bb_width_pctl < 20
            and (adx_value is None or adx_value < 20)
            and abs(bias_score) < _BIAS_THRESHOLD):
        return {
            "subtype": "breakout_pending",
            "confidence": 0.65,
            "reason": f"BB 寬度 {bb_width_pctl:.0f} 百分位（極窄）+ ADX={adx_value or 0:.0f}（弱），等突破",
            "metrics": metrics,
        }

    # 偏多 / 偏空
    if bias_score >= _BIAS_THRESHOLD:
        return {
            "subtype": "lean_long",
            "confidence": min(0.85, 0.5 + abs(bias_score) * 0.5),
            "reason": f"bias={bias_score:+.2f}：" + "、".join(bias_reasons[:3]),
            "metrics": metrics,
        }
    if bias_score <= -_BIAS_THRESHOLD:
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
