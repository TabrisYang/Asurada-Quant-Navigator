"""阿斯拉量化系統 — 自動掃描預警引擎

基於 ADA/USDT 5 次大波動事件（3%~10%）前 6 根 K 線的實證分析，
提取 35+ 個特徵並與正常時期基準對比，篩選出具區分力的前兆特徵。

核心前兆特徵（百分位偏離 + 五次一致性）：
  強信號: vol_range_desync（量幅不同步度）> 1.3  → 百分位 86.7%
  中信號: direction_consistency（方向一致性）>= 0.667 → 百分位 82.2%
          ma5_cross_count（MA5穿越次數）<= 1 → 百分位 17.2%
  弱信號: price_efficiency > 0.5, vol_momentum_pressure > 0.8,
          max_streak >= 3, vol_concentration < 0.26
"""

import json
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from app.core.config.settings import settings
from app.core.stats_utils import wilson_ci as _wilson_ci
from app.core.bollinger_signals import classify_bollinger_signal as _classify_bollinger


def _derive_regime_from_features(features: dict) -> str:
    """v136：依當下 ADX + bb_position 簡易推斷 regime（不依賴 regime_filter）。

    這是 bollinger_signals 用的「inline regime」，不是系統正式 regime。
    正式 regime 判定在 regime_filter.py，但 auto_scanner 流程未呼叫它。
    """
    adx = features.get("adx", 0)
    bb_pos = features.get("bb_position", 50)
    pct_6bar = features.get("pct_6bar", 0)

    if adx >= 25:
        if pct_6bar > 0.5 or bb_pos > 60:
            return "trending_up"
        elif pct_6bar < -0.5 or bb_pos < 40:
            return "trending_down"
    return "ranging"


def _compute_bollinger_status(ind: dict, features: dict, idx: list[int]) -> dict | None:
    """v136：依當下特徵 + 前一根特徵 + 近 5 根 bb_position 偵測 Bollinger 訊號。

    回傳 classify_bollinger_signal 的結果，或 None。
    """
    if not getattr(settings, "bollinger_signals_enabled", True):
        return None

    last = idx[-1]
    bb_pos_arr = ind.get("bb_pos")
    if bb_pos_arr is None or last < 1:
        return None

    # 取前一根的精簡特徵（給 breakout / reversion detector 用）
    prev_features = {
        "bb_position": float(bb_pos_arr[last - 1]) if not np.isnan(bb_pos_arr[last - 1]) else 50,
        "bb_position_lag1": float(bb_pos_arr[last - 2]) if last >= 2 and not np.isnan(bb_pos_arr[last - 2]) else 50,
        "is_squeeze": bool(ind["is_squeeze"][last - 1]) if "is_squeeze" in ind else False,
        "squeeze_duration": int(ind["squeeze_duration"][last - 1]) if "squeeze_duration" in ind else 0,
    }

    # 近 5 根 bb_position（給 walk_the_band detector 用）
    recent_bb_positions = []
    for i in range(max(0, last - 4), last + 1):
        v = bb_pos_arr[i]
        if not np.isnan(v):
            recent_bb_positions.append(float(v))

    # 推斷 regime
    regime = _derive_regime_from_features(features)

    # 取當下價格、通道值、ATR（給 entry/exit 計算用）
    close = float(ind["closes"][last])
    sma20 = float(ind["ma20"][last]) if not np.isnan(ind["ma20"][last]) else close
    bb_upper = float(ind.get("bb_upper", [close])[last]) if "bb_upper" in ind else close * 1.02
    bb_lower = float(ind.get("bb_lower", [close])[last]) if "bb_lower" in ind else close * 0.98
    atr = float(ind["atr14"][last]) if not np.isnan(ind["atr14"][last]) else close * 0.01

    try:
        result = _classify_bollinger(
            features, prev_features, recent_bb_positions, regime,
            close, sma20, bb_upper, bb_lower, atr,
        )
        if result:
            result["regime_used"] = regime
        return result
    except Exception as e:
        logger.debug(f"[v136] bollinger_signal 偵測失敗: {e}")
        return None


# ─── 常量 ────────────────────────────────────────────

_LOOKBACK = 6  # 前兆窗口 K 線數

# 預警有效時間（根據 timeframe）
_ALERT_DURATION = {
    "15m": timedelta(hours=4),
    "1h": timedelta(hours=12),
    "4h": timedelta(hours=48),
    "1d": timedelta(days=7),
    "1w": timedelta(days=14),
}

# ─── 特徵計算 ─────────────────────────────────────────


def _compute_features(
    closes: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    ma5: np.ndarray,
    ma20: np.ndarray,
    vol_ma20: np.ndarray,
    rsi14: np.ndarray,
    bb_pos: np.ndarray,
    bb_width: np.ndarray,
    atr14: np.ndarray,
    adx: np.ndarray,
    plus_di: np.ndarray,
    minus_di: np.ndarray,
    idx: list[int],
    # v136：布林通道完整策略系統 — 進階特徵（向後相容，可選參數）
    bb_std: np.ndarray | None = None,
    obv: np.ndarray | None = None,
    keltner_upper: np.ndarray | None = None,
    keltner_lower: np.ndarray | None = None,
    is_squeeze: np.ndarray | None = None,
    squeeze_duration: np.ndarray | None = None,
) -> dict:
    """計算一個 6 根窗口的全部前兆特徵。"""
    c = closes[idx]
    o = opens[idx]
    h = highs[idx]
    l = lows[idx]
    v = volumes[idx]
    last = idx[-1]

    body = c - o
    body_pct = body / (o + 1e-10) * 100
    upper_wick = h - np.maximum(c, o)
    lower_wick = np.minimum(c, o) - l
    full_range = h - l

    f: dict = {}

    # ── K 線形態 ──
    bullish = sum(1 for b in body if b > 0)
    f["direction_consistency"] = max(bullish, 6 - bullish) / 6
    f["bullish_ratio"] = bullish / 6
    f["body_range_ratio"] = float(np.mean(np.abs(body) / (full_range + 1e-10)))

    # ── 成交量 ──
    f["vol_acceleration"] = float(np.mean(v[-3:]) / (np.mean(v[:3]) + 1e-10))
    f["vol_concentration"] = float(max(v) / (sum(v) + 1e-10))
    f["max_vol_spike"] = float(max(v) / (np.mean(v) + 1e-10))
    f["vol_ratio_20ma"] = float(v[-1] / (vol_ma20[last] + 1e-10))

    price_dir = 1 if c[-1] > c[0] else -1
    vol_dir = 1 if v[-1] > v[0] else -1
    f["vol_price_divergence"] = 1 if price_dir != vol_dir else 0

    # ── 量幅不同步度（強信號）──
    vol_rank = np.argsort(np.argsort(v)).astype(float)
    range_rank = np.argsort(np.argsort(full_range)).astype(float)
    f["vol_range_desync"] = float(np.mean(np.abs(vol_rank - range_rank)))

    # ── 價格效率 ──
    net_move = abs(c[-1] - c[0])
    total_path = sum(abs(c[j] - c[j - 1]) for j in range(1, len(c)))
    f["price_efficiency"] = float(net_move / (total_path + 1e-10))

    # ── 連續同向 ──
    max_streak = 1
    curr = 1
    for j in range(1, _LOOKBACK):
        if (body[j] > 0) == (body[j - 1] > 0):
            curr += 1
            max_streak = max(max_streak, curr)
        else:
            curr = 1
    f["max_streak"] = max_streak

    # ── 影線 ──
    long_wick = sum(
        1
        for j in range(_LOOKBACK)
        if upper_wick[j] > abs(body[j]) * 1.5
        or lower_wick[j] > abs(body[j]) * 1.5
    )
    f["long_wick_count"] = long_wick

    # ── MA 穿越 ──
    cross_count = 0
    for j in range(1, len(idx)):
        if (closes[idx[j]] > ma5[idx[j]]) != (closes[idx[j - 1]] > ma5[idx[j - 1]]):
            cross_count += 1
    f["ma5_cross_count"] = cross_count

    # ── 波動率 ──
    r_first3 = float(np.mean(full_range[:3]))
    r_last3 = float(np.mean(full_range[-3:]))
    f["range_contraction"] = r_last3 / (r_first3 + 1e-10)

    atr_50_start = max(0, idx[0] - 50)
    atr_50_avg = float(np.mean(atr14[atr_50_start : idx[0]])) if idx[0] > 50 else float(atr14[last])
    f["atr_relative"] = float(atr14[last] / (atr_50_avg + 1e-10))

    # ── 組合特徵 ──
    f["vol_momentum_pressure"] = f["vol_acceleration"] * f["direction_consistency"]

    lower_wick_ratio = float(np.mean(lower_wick / (full_range + 1e-10)))
    f["accumulation_signal"] = lower_wick_ratio * f["vol_acceleration"]

    # ── 傳統指標（方向性判斷用）──
    f["rsi"] = float(rsi14[last])
    f["bb_position"] = float(bb_pos[last])
    f["bb_width"] = float(bb_width[last])
    f["adx"] = float(adx[last])
    f["di_spread"] = float(plus_di[last] - minus_di[last])
    f["price_vs_ma20"] = float((c[-1] - ma20[last]) / (ma20[last] + 1e-10) * 100)

    # 6 根漲跌
    f["pct_6bar"] = float((c[-1] - c[0]) / (c[0] + 1e-10) * 100)

    # ── v136 布林通道完整策略 — 進階特徵 ──
    # PctB_lag1：前一根的 %B（跨軌瞬間判斷）
    if last >= 1 and not np.isnan(bb_pos[last - 1]):
        f["bb_position_lag1"] = float(bb_pos[last - 1])
    else:
        f["bb_position_lag1"] = float(bb_pos[last])

    # Bandwidth_ROC：bb_width 近 4 根變動率（抓波動爆發）
    if last >= 4 and not np.isnan(bb_width[last - 4]) and bb_width[last - 4] > 1e-10:
        f["bb_width_roc"] = float(bb_width[last] / bb_width[last - 4] - 1) * 100
    else:
        f["bb_width_roc"] = 0.0

    # Z_Score：(close - sma20) / std20（價格偏離均線 σ 倍數）
    if bb_std is not None and last < len(bb_std) and not np.isnan(bb_std[last]) and bb_std[last] > 1e-10:
        f["z_score_20"] = float((c[-1] - ma20[last]) / bb_std[last])
    else:
        f["z_score_20"] = 0.0

    # OBV_Slope：近 10 根 OBV 線性回歸斜率（量能動能）
    if obv is not None and last >= 10:
        recent_obv = obv[last - 9 : last + 1]
        if not np.any(np.isnan(recent_obv)):
            x = np.arange(10)
            try:
                slope = float(np.polyfit(x, recent_obv, 1)[0])
                # 標準化到 -1~+1 區間，除以平均 OBV 量級避免數值爆炸
                obv_mean = float(np.mean(np.abs(recent_obv)) + 1e-10)
                f["obv_slope_10"] = slope / obv_mean
            except (np.linalg.LinAlgError, ValueError):
                f["obv_slope_10"] = 0.0
        else:
            f["obv_slope_10"] = 0.0
    else:
        f["obv_slope_10"] = 0.0

    # Keltner Channel + Squeeze 偵測（The Squeeze 策略的標準做法）
    if keltner_upper is not None and keltner_lower is not None:
        f["keltner_upper"] = float(keltner_upper[last]) if not np.isnan(keltner_upper[last]) else 0.0
        f["keltner_lower"] = float(keltner_lower[last]) if not np.isnan(keltner_lower[last]) else 0.0
    if is_squeeze is not None:
        f["is_squeeze"] = bool(is_squeeze[last])
    if squeeze_duration is not None:
        f["squeeze_duration"] = int(squeeze_duration[last])

    return f


def _precompute_indicators(df: pd.DataFrame) -> dict:
    """預計算所有技術指標，返回 numpy 陣列字典。"""
    closes = df["close"].values.astype(float)
    opens = df["open"].values.astype(float)
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    volumes = df["volume"].values.astype(float)

    ma5 = pd.Series(closes).rolling(5).mean().values
    ma20 = pd.Series(closes).rolling(20).mean().values
    vol_ma20 = pd.Series(volumes).rolling(20).mean().values

    # RSI(14)
    delta = np.diff(closes, prepend=closes[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean().values
    avg_loss = pd.Series(loss).rolling(14).mean().values
    rsi14 = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-10)))

    # BB
    bb_ma = pd.Series(closes).rolling(20).mean().values
    bb_std = pd.Series(closes).rolling(20).std().values
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / (bb_ma + 1e-10) * 100
    bb_pos = (closes - bb_lower) / (bb_upper - bb_lower + 1e-10) * 100

    # ATR(14)
    tr = np.maximum(
        highs - lows,
        np.maximum(
            np.abs(highs - np.roll(closes, 1)),
            np.abs(lows - np.roll(closes, 1)),
        ),
    )
    atr14 = pd.Series(tr).rolling(14).mean().values

    # ADX
    plus_dm = np.maximum(np.diff(highs, prepend=highs[0]), 0)
    minus_dm = np.maximum(-np.diff(lows, prepend=lows[0]), 0)
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    atr_s = pd.Series(tr).rolling(14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(14).mean().values / (atr_s + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(14).mean().values / (atr_s + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(14).mean().values

    # v136 布林通道完整策略 — 進階指標
    # OBV：On-Balance Volume 累積值
    price_change_sign = np.sign(np.diff(closes, prepend=closes[0]))
    obv = np.cumsum(price_change_sign * volumes)

    # Keltner Channel：sma20 ± 1.5 × atr14（squeeze 偵測標準）
    keltner_upper = bb_ma + 1.5 * atr14
    keltner_lower = bb_ma - 1.5 * atr14

    # is_squeeze：BB 收進 Keltner 內（John Bollinger 標準定義）
    is_squeeze = (bb_upper < keltner_upper) & (bb_lower > keltner_lower)

    # squeeze_duration：is_squeeze 連續 True 根數（squeeze 越久爆發越強）
    squeeze_duration = np.zeros(len(is_squeeze), dtype=int)
    run = 0
    for i in range(len(is_squeeze)):
        if is_squeeze[i]:
            run += 1
        else:
            run = 0
        squeeze_duration[i] = run

    return {
        "closes": closes,
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "volumes": volumes,
        "ma5": ma5,
        "ma20": ma20,
        "vol_ma20": vol_ma20,
        "rsi14": rsi14,
        "bb_pos": bb_pos,
        "bb_width": bb_width,
        "bb_std": bb_std,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "atr14": atr14,
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
        # v136 布林通道進階特徵
        "obv": obv,
        "keltner_upper": keltner_upper,
        "keltner_lower": keltner_lower,
        "is_squeeze": is_squeeze,
        "squeeze_duration": squeeze_duration,
    }


# ─── 信號偵測（數據驅動路徑）─────────────────────────────


def _detect_signals_data_driven(
    features: dict,
    calibrations: dict | None,
    feature_profiles: dict,
) -> list[dict]:
    """用全量歷史統計的權重和門檻評分。"""
    signals: list[dict] = []
    meta = feature_profiles["_meta"]

    # ── 數據驅動評分 ──
    score = 0.0
    triggered: list[str] = []

    fw = (calibrations or {}).get("feature_weights", {})

    for feat_name, profile in feature_profiles.items():
        if feat_name.startswith("_"):
            continue
        if profile.get("weight", 0) < 0.01:
            continue

        val = features.get(feat_name)
        if val is None:
            continue

        threshold = profile["optimal_threshold"]
        direction = profile["threshold_direction"]
        weight = profile["weight"]
        p50 = profile.get("p50", threshold)

        # 疊加 calibrator 微調
        cal_data = fw.get(feat_name, {})
        if cal_data.get("adjusted") and cal_data.get("default") and cal_data["default"] > 0:
            mult = cal_data["adjusted"] / cal_data["default"]
            weight = weight * mult

        crosses = (
            (val > threshold) if direction == "above"
            else (val < threshold)
        )
        if crosses:
            score += weight
            triggered.append(feat_name)
        elif (
            (direction == "above" and val > p50)
            or (direction == "below" and val < p50)
        ):
            score += weight * 0.33  # partial credit

    # ── 最低門檻 ──
    threshold_score = meta.get("score_threshold", 5.5)
    # calibrator 門檻覆蓋
    cal_threshold = (calibrations or {}).get("score_threshold", {}).get("adjusted")
    if cal_threshold is not None and abs(cal_threshold - 5.5) > 0.01:
        # calibrator 的門檻是基於舊尺度的，需做比例換算
        ratio = cal_threshold / 5.5
        threshold_score = threshold_score * ratio

    if score < threshold_score:
        return signals

    # ── 判斷信號類型和方向（邏輯同硬編碼路徑）──

    short_score = 0.0
    short_triggers: list[str] = []
    if features["rsi"] > 70:
        short_score += 2.0
        short_triggers.append(f"RSI={features['rsi']:.1f}>70")
    elif features["rsi"] > 55:
        short_score += 0.5 + (features["rsi"] - 55) / 30
    if features["bb_position"] >= 75:
        bp = min(features["bb_position"], 100)
        short_score += 1.0 + (bp - 75) / 25
        short_triggers.append(f"BB={features['bb_position']:.0f}%≥75%")
    if features["price_vs_ma20"] > 1.5:
        short_score += min(features["price_vs_ma20"] / 2.0, 1.5)
        short_triggers.append(f"price>MA20+{features['price_vs_ma20']:.1f}%")
    if features["bullish_ratio"] > 0.6:
        short_score += 0.5
    if features["di_spread"] > 5:
        short_score += 0.5
        short_triggers.append("+DI偏強")

    if short_score >= 2.5:
        signals.append({
            "alert_type": "overbought_reversal",
            "direction": "short",
            "score": score + short_score,
            "triggered": triggered + short_triggers,
        })

    long_score = 0.0
    long_triggers: list[str] = []
    if features["rsi"] < 38:
        long_score += 2.0
        long_triggers.append(f"RSI={features['rsi']:.1f}<38")
    elif features["rsi"] < 45:
        long_score += 0.5 + (45 - features["rsi"]) / 14
    if features["bb_position"] <= 25:
        bp = max(features["bb_position"], 0)
        long_score += 1.0 + (25 - bp) / 25
        long_triggers.append(f"BB={features['bb_position']:.0f}%≤25%")
    elif features["bb_position"] < 35:
        long_score += 0.5
    if features["price_vs_ma20"] < -1.5:
        long_score += min(abs(features["price_vs_ma20"]) / 2.0, 1.5)
        long_triggers.append(f"price<MA20{features['price_vs_ma20']:+.1f}%")
    if features["bullish_ratio"] < 0.33:
        long_score += 1.0
        long_triggers.append("bearish_dominant")
    elif features["bullish_ratio"] < 0.5:
        long_score += 0.5
    if features["vol_acceleration"] > 1.1:
        long_score += 0.5

    if long_score >= 2.5:
        signals.append({
            "alert_type": "oversold_bounce",
            "direction": "long",
            "score": score + long_score,
            "triggered": triggered + long_triggers,
        })

    # 波動擴張（無明確方向時的後備）
    if not signals:
        vol_triggers: list[str] = []
        vol_ok = False

        if features["adx"] > 20:
            vol_triggers.append(f"ADX={features['adx']:.1f}")
            vol_ok = True
        elif score >= threshold_score * 1.3:
            vol_triggers.append(f"high_score={score:.1f}")
            vol_ok = True

        if vol_ok:
            direction = "long" if features["pct_6bar"] > 0 else "short"
            if features["di_spread"] > 10:
                vol_triggers.append("+DI主導")
            elif features["di_spread"] < -10:
                vol_triggers.append("-DI主導")
            signals.append({
                "alert_type": "volatility_expansion",
                "direction": direction,
                "score": score,
                "triggered": triggered + vol_triggers,
            })

    # 信心等級
    high_boundary = meta.get("high_boundary", 12)
    medium_boundary = meta.get("medium_boundary", 8)
    for sig in signals:
        s = sig["score"]
        if s >= high_boundary:
            sig["confidence"] = "high"
        elif s >= medium_boundary:
            sig["confidence"] = "medium"
        else:
            sig["confidence"] = "low"

    return signals


# ─── 信號偵測（硬編碼 fallback 路徑）─────────────────────


def _detect_signals(
    features: dict,
    calibrations: dict | None = None,
    feature_profiles: dict | None = None,
) -> list[dict]:
    """根據特徵判斷是否觸發預警信號。

    三層架構：
    1. feature_profiles（數據驅動）：全量歷史統計的 IC 權重 + 最佳門檻
    2. calibrations（預警回饋）：疊加在數據驅動權重上的微調乘數
    3. 硬編碼（fallback）：當以上兩者不可用時的安全後備
    """
    # === 數據驅動路徑 ===
    if feature_profiles and feature_profiles.get("_meta"):
        return _detect_signals_data_driven(features, calibrations, feature_profiles)

    # === 硬編碼路徑（fallback）===
    signals: list[dict] = []
    fw = (calibrations or {}).get("feature_weights", {})

    def _w(name: str, default: float) -> float:
        """讀取校準後的特徵權重，fallback 為預設值。"""
        return fw.get(name, {}).get("adjusted", default)

    # ── 加權評分（每個特徵根據偏離程度計分）──
    score = 0.0
    triggered: list[str] = []

    # 強信號（百分位 86.7%，事件均值 1.6 vs 正常 1.0）
    if features["vol_range_desync"] > 1.3:
        score += _w("vol_range_desync", 3.0)
        triggered.append("vol_range_desync")
    elif features["vol_range_desync"] > 1.0:
        score += _w("vol_range_desync", 3.0) * 0.5  # 部分分數

    # 中信號（百分位 82.2%）
    if features["direction_consistency"] >= 0.833:
        score += _w("direction_consistency_high", 2.5)
        triggered.append("direction_consistency_high")
    elif features["direction_consistency"] >= 0.667:
        score += _w("direction_consistency", 1.5)
        triggered.append("direction_consistency")

    # 中信號（百分位 17.2%，低值 = 趨勢未被打斷）
    if features["ma5_cross_count"] == 0:
        score += _w("ma5_no_cross", 2.5)
        triggered.append("ma5_no_cross")
    elif features["ma5_cross_count"] <= 1:
        score += _w("ma5_cross_low", 1.5)
        triggered.append("ma5_cross_low")

    # 弱信號群
    if features["price_efficiency"] > 0.6:
        score += _w("price_efficiency", 1.5)
        triggered.append("price_efficiency")
    elif features["price_efficiency"] > 0.4:
        score += _w("price_efficiency", 1.5) * 0.33

    if features["vol_momentum_pressure"] > 0.8:
        score += _w("vol_momentum_pressure", 1.0)
        triggered.append("vol_momentum_pressure")

    if features["max_streak"] >= 4:
        score += _w("max_streak", 1.5)
        triggered.append("max_streak")
    elif features["max_streak"] >= 3:
        score += _w("max_streak", 1.5) * 0.33

    if features["vol_concentration"] < 0.25:
        score += _w("vol_even_distribution", 1.0)
        triggered.append("vol_even_distribution")

    if features["accumulation_signal"] > 0.35:
        score += _w("accumulation_signal", 1.0)
        triggered.append("accumulation_signal")

    if features["vol_acceleration"] > 1.2:
        score += _w("vol_accelerating", 1.0)
        triggered.append("vol_accelerating")
    elif features["vol_acceleration"] > 1.05:
        score += _w("vol_accelerating", 1.0) * 0.5

    # ATR 相對膨脹（波動率擴張前兆）
    if features["atr_relative"] > 1.2:
        score += _w("atr_expanding", 1.0)
        triggered.append("atr_expanding")
    elif features["atr_relative"] > 1.05:
        score += _w("atr_expanding", 1.0) * 0.5

    # K線實體佔全幅比例高 = 強勢趨勢
    if features["body_range_ratio"] > 0.65:
        score += _w("strong_body_ratio", 1.0)
        triggered.append("strong_body_ratio")
    elif features["body_range_ratio"] > 0.55:
        score += _w("strong_body_ratio", 1.0) * 0.5

    # 量能相對 MA20 放大
    if features["vol_ratio_20ma"] > 1.5:
        score += _w("vol_above_ma20", 1.0)
        triggered.append("vol_above_ma20")
    elif features["vol_ratio_20ma"] > 1.2:
        score += _w("vol_above_ma20", 1.0) * 0.5

    # ── 最低門檻（可校準）──
    threshold = (calibrations or {}).get("score_threshold", {}).get("adjusted", 5.5)
    if score < threshold:
        return signals

    # ── 判斷信號類型和方向 ──

    # 做空信號（超買反轉）— 使用漸進式評分
    short_score = 0.0
    short_triggers: list[str] = []
    if features["rsi"] > 70:
        short_score += 2.0
        short_triggers.append(f"RSI={features['rsi']:.1f}>70")
    elif features["rsi"] > 55:
        short_score += 0.5 + (features["rsi"] - 55) / 30  # 漸進：55→70 得 0.5~1.0
    if features["bb_position"] >= 75:
        bp = min(features["bb_position"], 100)
        short_score += 1.0 + (bp - 75) / 25  # 75%→100% 得 1.0~2.0
        short_triggers.append(f"BB={features['bb_position']:.0f}%≥75%")
    if features["price_vs_ma20"] > 1.5:
        short_score += min(features["price_vs_ma20"] / 2.0, 1.5)  # 漸進上限 1.5
        short_triggers.append(f"price>MA20+{features['price_vs_ma20']:.1f}%")
    if features["bullish_ratio"] > 0.6:
        short_score += 0.5
    if features["di_spread"] > 5:
        short_score += 0.5
        short_triggers.append("+DI偏強")

    if short_score >= 2.5:
        signals.append({
            "alert_type": "overbought_reversal",
            "direction": "short",
            "score": score + short_score,
            "triggered": triggered + short_triggers,
        })

    # 做多信號（超賣反彈）
    long_score = 0.0
    long_triggers: list[str] = []
    if features["rsi"] < 38:
        long_score += 2.0
        long_triggers.append(f"RSI={features['rsi']:.1f}<38")
    elif features["rsi"] < 45:
        long_score += 0.5 + (45 - features["rsi"]) / 14  # 漸進：45→38 得 0.5~1.0
    if features["bb_position"] <= 25:
        bp = max(features["bb_position"], 0)
        long_score += 1.0 + (25 - bp) / 25  # 25%→0% 得 1.0~2.0
        long_triggers.append(f"BB={features['bb_position']:.0f}%≤25%")
    elif features["bb_position"] < 35:
        long_score += 0.5
    if features["price_vs_ma20"] < -1.5:
        long_score += min(abs(features["price_vs_ma20"]) / 2.0, 1.5)
        long_triggers.append(f"price<MA20{features['price_vs_ma20']:+.1f}%")
    if features["bullish_ratio"] < 0.33:
        long_score += 1.0
        long_triggers.append("bearish_dominant")
    elif features["bullish_ratio"] < 0.5:
        long_score += 0.5
    if features["vol_acceleration"] > 1.1:
        long_score += 0.5

    if long_score >= 2.5:
        signals.append({
            "alert_type": "oversold_bounce",
            "direction": "long",
            "score": score + long_score,
            "triggered": triggered + long_triggers,
        })

    # 波動擴張信號（當沒有明確方向性信號時的後備）
    # 放寬 ADX 門檻，或當 general score 足夠高時也觸發
    if not signals:
        vol_triggers: list[str] = []
        vol_ok = False

        if features["adx"] > 20:
            vol_triggers.append(f"ADX={features['adx']:.1f}")
            vol_ok = True
        elif score >= 7.0:
            # 高分但無方向性信號 → 仍然值得預警
            vol_triggers.append(f"high_score={score:.1f}")
            vol_ok = True

        if vol_ok:
            direction = "long" if features["pct_6bar"] > 0 else "short"
            if features["di_spread"] > 10:
                vol_triggers.append("+DI主導")
            elif features["di_spread"] < -10:
                vol_triggers.append("-DI主導")

            signals.append({
                "alert_type": "volatility_expansion",
                "direction": direction,
                "score": score,
                "triggered": triggered + vol_triggers,
            })

    # 信心等級（可校準分界）
    cb = (calibrations or {}).get("confidence_boundaries", {})
    high_boundary = cb.get("high", 12)
    medium_boundary = cb.get("medium", 8)
    for sig in signals:
        s = sig["score"]
        if s >= high_boundary:
            sig["confidence"] = "high"
        elif s >= medium_boundary:
            sig["confidence"] = "medium"
        else:
            sig["confidence"] = "low"

    return signals


# ─── 歷史相似情境機率估算 ─────────────────────────────


def _features_to_vector(features: dict) -> np.ndarray:
    """將特徵 dict 轉換為正規化向量，用於餘弦相似度計算。"""
    # 選取核心數值特徵（排除方向性指標，因為它們是方向判斷用的）
    keys = [
        "vol_range_desync", "direction_consistency", "price_efficiency",
        "ma5_cross_count", "vol_momentum_pressure", "max_streak",
        "vol_concentration", "accumulation_signal", "vol_acceleration",
        "body_range_ratio", "atr_relative", "vol_ratio_20ma",
        "range_contraction", "max_vol_spike",
    ]
    vec = np.array([features.get(k, 0.0) for k in keys], dtype=float)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """計算兩個向量的餘弦相似度。"""
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def _estimate_movement_probability(
    ind: dict,
    current_features: dict,
    current_score: float,
    current_triggered: list[str],
    direction: str,
    n: int,
    move_threshold: float = 3.0,
    calibrations: dict | None = None,
) -> dict:
    """遍歷歷史數據，找出特徵向量相似的窗口，統計後續 3~6 根 K 線的漲跌機率。

    改進：
    - 使用餘弦相似度匹配（而非單一分數容差），能區分「相同分數但不同特徵組合」
    - STEP=_LOOKBACK 確保窗口不重疊，樣本獨立
    - 附帶 Wilson score 信賴區間

    Args:
        ind: _precompute_indicators() 的輸出
        current_features: 當前窗口的完整特徵 dict
        current_score: 當前信號分數
        current_triggered: 當前觸發的特徵名稱列表
        direction: 信號方向
        n: 總 K 線數
        move_threshold: 漲跌幅門檻（%）
        calibrations: 校準參數（含 similarity_threshold）
    """
    SIMILARITY_THRESHOLD = (calibrations or {}).get(
        "similarity_threshold", {}
    ).get("adjusted", 0.85)
    FORWARD_BARS = [3, 4, 5, 6]
    WARMUP = 50
    STEP = _LOOKBACK  # 不重疊窗口，確保樣本獨立
    max_fb = max(FORWARD_BARS)

    closes = ind["closes"]
    current_vec = _features_to_vector(current_features)

    # 統計容器
    similar_windows: list[dict] = []

    for i in range(WARMUP, n - max_fb - _LOOKBACK, STEP):
        idx = list(range(i, i + _LOOKBACK))

        hist_features = _compute_features(
            closes, ind["opens"], ind["highs"], ind["lows"], ind["volumes"],
            ind["ma5"], ind["ma20"], ind["vol_ma20"],
            ind["rsi14"], ind["bb_pos"], ind["bb_width"],
            ind["atr14"], ind["adx"], ind["plus_di"], ind["minus_di"],
            idx,
            bb_std=ind.get("bb_std"),
            obv=ind.get("obv"),
            keltner_upper=ind.get("keltner_upper"),
            keltner_lower=ind.get("keltner_lower"),
            is_squeeze=ind.get("is_squeeze"),
            squeeze_duration=ind.get("squeeze_duration"),
        )

        # 特徵向量相似度篩選
        hist_vec = _features_to_vector(hist_features)
        sim = _cosine_similarity(current_vec, hist_vec)
        if sim < SIMILARITY_THRESHOLD:
            continue

        # 此窗口特徵相似，統計未來走勢
        entry_price = closes[i + _LOOKBACK - 1]
        outcomes: dict[int, float] = {}
        for fb in FORWARD_BARS:
            future_idx = i + _LOOKBACK - 1 + fb
            if future_idx < n:
                outcomes[fb] = (closes[future_idx] - entry_price) / entry_price * 100

        # 收集此窗口觸發了哪些特徵
        hist_signals = _detect_signals(hist_features)
        hist_triggered: set[str] = set()
        for s in hist_signals:
            hist_triggered.update(s.get("triggered", []))

        similar_windows.append({
            "outcomes": outcomes,
            "triggered": hist_triggered,
            "similarity": sim,
        })

    sample_count = len(similar_windows)
    if sample_count == 0:
        return {
            "sample_count": 0,
            "match_method": "cosine_similarity≥0.85",
            "probability": {},
            "direction_bias": direction,
            "confidence_note": "insufficient",
            "feature_attribution": [],
            "evidence_summary": "歷史數據中未找到特徵相似的情境，無法估算機率。",
        }

    # 統計各 forward horizon 的漲跌比例 + 信賴區間
    prob: dict[str, dict] = {}
    for fb in FORWARD_BARS:
        valid_outcomes = [w["outcomes"][fb] for w in similar_windows if fb in w["outcomes"]]
        valid = len(valid_outcomes)
        if valid == 0:
            continue
        up_count = sum(1 for o in valid_outcomes if o >= move_threshold)
        down_count = sum(1 for o in valid_outcomes if o <= -move_threshold)
        any_count = up_count + down_count

        up_ci = _wilson_ci(up_count, valid)
        down_ci = _wilson_ci(down_count, valid)
        any_ci = _wilson_ci(any_count, valid)

        prob[f"{fb}_bars"] = {
            "up_pct": round(up_count / valid * 100, 1),
            "down_pct": round(down_count / valid * 100, 1),
            "any_move_pct": round(any_count / valid * 100, 1),
            "up_ci": up_ci,
            "down_ci": down_ci,
            "any_ci": any_ci,
            "sample_n": valid,
        }

    # Feature attribution
    hit_windows = [
        w for w in similar_windows
        if any(abs(w["outcomes"].get(fb, 0)) >= move_threshold for fb in [6, 5, 4, 3])
    ]
    hit_count = len(hit_windows)

    attribution: list[dict] = []
    for feat_name in current_triggered:
        base_name = feat_name.split("=")[0] if "=" in feat_name else feat_name
        presence_all = sum(1 for w in similar_windows if any(base_name in t for t in w["triggered"]))
        presence_hit = sum(1 for w in hit_windows if any(base_name in t for t in w["triggered"])) if hit_count > 0 else 0
        rate_all = presence_all / sample_count * 100 if sample_count > 0 else 0
        rate_hit = presence_hit / hit_count * 100 if hit_count > 0 else 0
        lift = rate_hit / rate_all if rate_all > 0 else 0

        attribution.append({
            "feature": feat_name,
            "presence_in_hits": round(rate_hit, 1),
            "presence_in_all": round(rate_all, 1),
            "lift": round(lift, 2),
        })

    attribution.sort(key=lambda x: x["lift"], reverse=True)

    # 信心等級（基於獨立樣本數）
    if sample_count >= 30:
        confidence_note = "adequate"
    elif sample_count >= 10:
        confidence_note = "moderate"
    else:
        confidence_note = "limited"

    # 生成中文 evidence_summary（含信賴區間）
    best_horizon = "6_bars" if "6_bars" in prob else (list(prob.keys())[-1] if prob else None)

    if best_horizon and prob.get(best_horizon):
        bp = prob[best_horizon]
        bars_n = best_horizon.replace("_bars", "")
        ci = bp["any_ci"]
        top_feat = attribution[0]["feature"] if attribution else "N/A"
        top_rate = attribution[0]["presence_in_hits"] if attribution else 0

        summary = (
            f"在{sample_count}個獨立相似情境中（餘弦相似度≥0.85），"
            f"未來{bars_n}根K線內出現≥{move_threshold:.0f}%波動的機率為"
            f"{bp['any_move_pct']:.1f}%（95%信賴區間：{ci[0]:.0f}%~{ci[1]:.0f}%）"
            f"（上漲{bp['up_pct']:.1f}%，下跌{bp['down_pct']:.1f}%）。"
        )
        if top_rate > 0:
            summary += f"主要依據：{top_feat}在成功信號中出現率達{top_rate:.0f}%。"
        if confidence_note == "limited":
            summary += "（注意：獨立樣本數<10，機率僅供參考）"
        elif confidence_note == "moderate":
            summary += "（樣本數適中，機率具一定參考價值）"
    else:
        summary = "歷史數據不足以計算可靠的機率估算。"

    headline_prob = prob.get("6_bars", prob.get("3_bars", {})).get("any_move_pct", 0)

    return {
        "sample_count": sample_count,
        "match_method": "cosine_similarity≥0.85",
        "probability": prob,
        "headline_probability": headline_prob,
        "direction_bias": direction,
        "confidence_note": confidence_note,
        "feature_attribution": attribution[:8],
        "evidence_summary": summary,
    }


# ─── AutoScanner 類 ──────────────────────────────────


class AutoScanner:
    """後台自動掃描引擎：定期掃描所有幣種，偵測異常前兆信號。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            db_path = settings.db_path / "predictions.db"
            from app.core.db_utils import open_sqlite
            self._conn = open_sqlite(db_path, check_same_thread=False, row_factory=True)
        return self._conn

    def _init_db(self):
        conn = self._get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                direction TEXT,
                confidence TEXT DEFAULT 'low',
                trigger_conditions TEXT,
                signal_score REAL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                outcome_pct REAL,
                validated_at TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_alerts_symbol ON alerts(symbol, status)"
        )
        # 新增 move_probability 欄位（向後相容）
        try:
            conn.execute("ALTER TABLE alerts ADD COLUMN move_probability REAL")
        except sqlite3.OperationalError:
            pass  # 欄位已存在
        conn.commit()

    def scan_symbol(self, symbol: str, timeframe: str = "4h") -> list[dict]:
        """掃描單一幣種，返回觸發的預警列表。"""
        from app.data.fetchers.crypto_engine import crypto_engine

        try:
            df = crypto_engine.load_local_data(symbol, timeframe)
            if df.empty or len(df) < 80:
                return []

            df = df.reset_index(drop=True)
            ind = _precompute_indicators(df)

            # 取最後 6 根 K 線
            n = len(df)
            idx = list(range(n - _LOOKBACK, n))

            features = _compute_features(
                ind["closes"], ind["opens"], ind["highs"], ind["lows"], ind["volumes"],
                ind["ma5"], ind["ma20"], ind["vol_ma20"],
                ind["rsi14"], ind["bb_pos"], ind["bb_width"],
                ind["atr14"], ind["adx"], ind["plus_di"], ind["minus_di"],
                idx,
            )

            # 載入校準值和特徵分析
            from app.core.scanner_calibrator import scanner_calibrator
            from app.core.scanner_feature_profiler import scanner_feature_profiler
            calibrations = scanner_calibrator.get_active_calibrations()
            feature_profiles = scanner_feature_profiler.get_feature_profiles()

            signals = _detect_signals(features, calibrations, feature_profiles)

            # v136：布林通道完整策略系統 — 偵測 Bollinger signal 並附到所有 alerts
            bollinger_result = _compute_bollinger_status(ind, features, idx)

            if not signals:
                return []

            # 記錄使用的評分路徑
            if feature_profiles and feature_profiles.get("_meta"):
                meta = feature_profiles["_meta"]
                logger.debug(
                    f"[掃描] {symbol} 使用數據驅動路徑 "
                    f"(threshold={meta.get('score_threshold')}, "
                    f"samples={meta.get('total_samples')})"
                )
            elif calibrations.get("calibration_active"):
                adj_count = sum(
                    1 for f in calibrations.get("feature_weights", {}).values()
                    if abs(f.get("adjusted", f.get("default", 0)) - f.get("default", 0)) > 0.01
                )
                if adj_count > 0:
                    logger.debug(
                        f"[校準] 使用校準值: threshold="
                        f"{calibrations.get('score_threshold', {}).get('adjusted', 5.5)}, "
                        f"{adj_count} features adjusted"
                    )

            # 為每個信號計算歷史相似情境機率
            for sig in signals:
                try:
                    prob_result = _estimate_movement_probability(
                        ind, features, sig["score"], sig["triggered"],
                        sig["direction"], n, calibrations=calibrations,
                    )
                    sig["probability"] = prob_result
                except Exception as e:
                    logger.warning(f"[自動掃描] {symbol} 機率估算失敗: {e}")
                    sig["probability"] = None

            # 存入 DB
            now = datetime.now()
            duration = _ALERT_DURATION.get(timeframe, timedelta(hours=48))
            expires = now + duration

            results = []
            with self._lock:
                conn = self._get_conn()

                # 避免重複：檢查是否已有同幣種同類型的 active alert
                for sig in signals:
                    existing = conn.execute(
                        "SELECT id FROM alerts WHERE symbol=? AND alert_type=? AND status='active'",
                        (symbol, sig["alert_type"]),
                    ).fetchone()

                    if existing:
                        continue

                    headline_prob = (
                        sig["probability"]["headline_probability"]
                        if sig.get("probability") else None
                    )

                    conn.execute(
                        """INSERT INTO alerts
                        (symbol, timeframe, alert_type, direction, confidence,
                         trigger_conditions, signal_score, created_at, expires_at,
                         status, move_probability)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)""",
                        (
                            symbol,
                            timeframe,
                            sig["alert_type"],
                            sig["direction"],
                            sig["confidence"],
                            json.dumps(
                                {
                                    "triggered": sig["triggered"],
                                    "features": features,
                                    "probability": sig.get("probability"),
                                    # v136：附 Bollinger 訊號狀態給前端 / LLM 報告引用
                                    "bollinger": bollinger_result,
                                },
                                ensure_ascii=False,
                            ),
                            sig["score"],
                            now.isoformat(),
                            expires.isoformat(),
                            headline_prob,
                        ),
                    )
                    results.append(
                        {
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "alert_type": sig["alert_type"],
                            "direction": sig["direction"],
                            "confidence": sig["confidence"],
                            "score": sig["score"],
                            "move_probability": headline_prob,
                            "evidence_summary": (
                                sig["probability"]["evidence_summary"]
                                if sig.get("probability") else None
                            ),
                            # v136：Bollinger 訊號狀態 (signal / emoji / label / strategy / entry_exit)
                            "bollinger_status": bollinger_result.get("signal") if bollinger_result else None,
                            "bollinger_emoji": bollinger_result.get("emoji") if bollinger_result else None,
                            "bollinger_label": bollinger_result.get("label") if bollinger_result else None,
                            "bollinger_strategy": bollinger_result.get("strategy") if bollinger_result else None,
                            "bollinger_entry_exit": bollinger_result.get("entry_exit") if bollinger_result else None,
                            "bollinger_regime": bollinger_result.get("regime_used") if bollinger_result else None,
                        }
                    )

                conn.commit()

            if results:
                for r in results:
                    logger.info(
                        f"[自動掃描] 🚨 {r['symbol']} {r['alert_type']} "
                        f"({r['direction']}, {r['confidence']}, score={r['score']:.1f})"
                    )

            return results

        except Exception as e:
            logger.error(f"[自動掃描] {symbol} 掃描失敗: {e}")
            return []

    def scan_all_symbols(self) -> list[dict]:
        """掃描所有幣種（優先讀取用戶自訂設定，否則用預設）。"""
        scan_cfg = self._load_scan_config()
        symbols = scan_cfg.get("scan_symbols", settings.default_symbols)
        timeframe = scan_cfg.get("scan_timeframe", "4h")

        all_results = []
        skipped: list[str] = []
        for symbol in symbols:
            sym = symbol.replace("/", "")
            # 數據可用性檢查
            if not self._has_data(sym, timeframe):
                skipped.append(symbol)
                continue
            results = self.scan_symbol(sym, timeframe)
            all_results.extend(results)

        if skipped:
            logger.warning(f"[自動掃描] 以下幣種無本地數據，已跳過: {', '.join(skipped)}")

        return all_results

    def check_symbols_data(self, symbols: list[str], timeframe: str = "4h") -> dict[str, bool]:
        """檢查多個幣種是否有足夠的本地數據。供 API 呼叫。"""
        result = {}
        for symbol in symbols:
            sym = symbol.replace("/", "")
            result[symbol] = self._has_data(sym, timeframe)
        return result

    @staticmethod
    def _has_data(symbol: str, timeframe: str) -> bool:
        """檢查指定幣種是否有足夠的本地 OHLCV 數據（至少 80 根）。"""
        try:
            from app.data.fetchers.crypto_engine import crypto_engine
            df = crypto_engine.load_local_data(symbol, timeframe)
            return not df.empty and len(df) >= 80
        except Exception:
            return False

    @staticmethod
    def _load_scan_config() -> dict:
        """讀取用戶自訂掃描設定。"""
        try:
            from app.api.routes.config import load_system_settings
            return load_system_settings()
        except Exception:
            return {}

    def get_active_alerts(self, symbol: Optional[str] = None) -> list[dict]:
        """取得進行中的預警。"""
        with self._lock:
            conn = self._get_conn()
            now = datetime.now().isoformat()

            if symbol:
                rows = conn.execute(
                    "SELECT * FROM alerts WHERE status='active' AND expires_at > ? AND symbol=? ORDER BY signal_score DESC",
                    (now, symbol),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM alerts WHERE status='active' AND expires_at > ? ORDER BY signal_score DESC",
                    (now,),
                ).fetchall()

            return [dict(r) for r in rows]

    def get_history(self, limit: int = 50, symbol: Optional[str] = None) -> list[dict]:
        """取得歷史預警。"""
        with self._lock:
            conn = self._get_conn()
            if symbol:
                rows = conn.execute(
                    "SELECT * FROM alerts WHERE symbol=? ORDER BY created_at DESC LIMIT ?",
                    (symbol, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    def dismiss_alert(self, alert_id: int):
        """忽略某個預警。"""
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "UPDATE alerts SET status='dismissed' WHERE id=?", (alert_id,)
            )
            conn.commit()

    def validate_expired_alerts(self):
        """驗證已過期的預警，計算實際漲跌幅。"""
        from app.data.fetchers.crypto_engine import crypto_engine

        with self._lock:
            conn = self._get_conn()
            now = datetime.now().isoformat()

            expired = conn.execute(
                "SELECT * FROM alerts WHERE status='active' AND expires_at <= ?",
                (now,),
            ).fetchall()

        for row in expired:
            try:
                df = crypto_engine.load_local_data(row["symbol"], row["timeframe"])
                if df.empty:
                    continue

                created = pd.Timestamp(row["created_at"])
                expires = pd.Timestamp(row["expires_at"])

                mask = (df["timestamp"] >= created) & (df["timestamp"] <= expires)
                window = df[mask]

                if len(window) < 2:
                    status = "expired"
                    outcome = 0.0
                else:
                    entry_price = window.iloc[0]["close"]
                    if row["direction"] == "long":
                        best = window["high"].max()
                        outcome = (best - entry_price) / entry_price * 100
                    elif row["direction"] == "short":
                        best = window["low"].min()
                        outcome = (entry_price - best) / entry_price * 100
                    else:
                        max_move = max(
                            abs(window["high"].max() - entry_price),
                            abs(entry_price - window["low"].min()),
                        )
                        outcome = max_move / entry_price * 100

                    status = "triggered" if outcome > 2.0 else "expired"

                with self._lock:
                    conn = self._get_conn()
                    conn.execute(
                        "UPDATE alerts SET status=?, outcome_pct=?, validated_at=? WHERE id=?",
                        (status, round(outcome, 2), now, row["id"]),
                    )
                    conn.commit()

                logger.info(
                    f"[預警驗證] {row['symbol']} {row['alert_type']}: "
                    f"{status} (outcome={outcome:+.2f}%)"
                )

            except Exception as e:
                logger.error(f"[預警驗證] {row['symbol']} 失敗: {e}")

        # 驗證完成後觸發校��
        try:
            from app.core.scanner_calibrator import scanner_calibrator
            scanner_calibrator.run_calibration_cycle()
        except Exception as e:
            logger.error(f"[校準] 失敗: {e}")

    def estimate_probability(
        self, symbol: str, timeframe: str = "4h", move_threshold: float = 3.0,
    ) -> dict:
        """On-demand 計算指定幣種的波動機率（含歷史依據）。

        即使當前分數低於預警門檻也能回傳機率。
        """
        from app.data.fetchers.crypto_engine import crypto_engine

        df = crypto_engine.load_local_data(symbol, timeframe)
        if df.empty or len(df) < 80:
            return {"status": "error", "message": f"數據不足（{len(df) if not df.empty else 0} 根 K 線）"}

        df = df.reset_index(drop=True)
        ind = _precompute_indicators(df)
        n = len(df)
        idx = list(range(n - _LOOKBACK, n))

        features = _compute_features(
            ind["closes"], ind["opens"], ind["highs"], ind["lows"], ind["volumes"],
            ind["ma5"], ind["ma20"], ind["vol_ma20"],
            ind["rsi14"], ind["bb_pos"], ind["bb_width"],
            ind["atr14"], ind["adx"], ind["plus_di"], ind["minus_di"],
            idx,
            bb_std=ind.get("bb_std"),
            obv=ind.get("obv"),
            keltner_upper=ind.get("keltner_upper"),
            keltner_lower=ind.get("keltner_lower"),
            is_squeeze=ind.get("is_squeeze"),
            squeeze_duration=ind.get("squeeze_duration"),
        )

        from app.core.scanner_calibrator import scanner_calibrator
        from app.core.scanner_feature_profiler import scanner_feature_profiler
        calibrations = scanner_calibrator.get_active_calibrations()
        feature_profiles = scanner_feature_profiler.get_feature_profiles()

        signals = _detect_signals(features, calibrations, feature_profiles)
        if signals:
            best = max(signals, key=lambda s: s["score"])
            score = best["score"]
            triggered = best["triggered"]
            direction = best["direction"]
        else:
            score = 0.0
            triggered = []
            direction = "neutral"

        prob_result = _estimate_movement_probability(
            ind, features, score, triggered, direction, n, move_threshold,
            calibrations=calibrations,
        )

        return {
            "status": "success",
            "symbol": symbol,
            "timeframe": timeframe,
            "move_threshold": move_threshold,
            "current_score": score,
            "has_active_signal": len(signals) > 0,
            "signal_direction": direction,
            "features": features,
            **prob_result,
        }

    def backtest_scan(self, symbol: str, timeframe: str, timestamp: str) -> list[dict]:
        """回測模式：在指定時間點執行掃描（用於驗證歷史事件）。"""
        from app.data.fetchers.crypto_engine import crypto_engine

        df = crypto_engine.load_local_data(symbol, timeframe)
        if df.empty or len(df) < 80:
            return []

        df = df.reset_index(drop=True)
        mask = df["timestamp"] <= timestamp
        if not mask.any():
            return []

        end_idx = df[mask].index[-1]
        if end_idx < _LOOKBACK + 50:
            return []

        # 只用到指定時間之前的數據
        df_slice = df.iloc[: end_idx + 1].reset_index(drop=True)
        ind = _precompute_indicators(df_slice)
        n = len(df_slice)
        idx = list(range(n - _LOOKBACK, n))

        features = _compute_features(
            ind["closes"], ind["opens"], ind["highs"], ind["lows"], ind["volumes"],
            ind["ma5"], ind["ma20"], ind["vol_ma20"],
            ind["rsi14"], ind["bb_pos"], ind["bb_width"],
            ind["atr14"], ind["adx"], ind["plus_di"], ind["minus_di"],
            idx,
            bb_std=ind.get("bb_std"),
            obv=ind.get("obv"),
            keltner_upper=ind.get("keltner_upper"),
            keltner_lower=ind.get("keltner_lower"),
            is_squeeze=ind.get("is_squeeze"),
            squeeze_duration=ind.get("squeeze_duration"),
        )

        from app.core.scanner_calibrator import scanner_calibrator
        from app.core.scanner_feature_profiler import scanner_feature_profiler
        calibrations = scanner_calibrator.get_active_calibrations()
        feature_profiles = scanner_feature_profiler.get_feature_profiles()

        signals = _detect_signals(features, calibrations, feature_profiles)
        for sig in signals:
            sig["features"] = features
            sig["timestamp"] = timestamp
            # 回測也計算機率
            try:
                prob_result = _estimate_movement_probability(
                    ind, features, sig["score"], sig["triggered"], sig["direction"], n,
                    calibrations=calibrations,
                )
                sig["probability"] = prob_result
            except Exception:
                sig["probability"] = None
        return signals


# 全域單例
auto_scanner = AutoScanner()
