"""v136 完整布林通道策略系統 — 3 個核心策略偵測 + regime-aware 選擇 + entry/exit/stop。

涵蓋 John Bollinger 經典書中的核心交易型態：
1. The Squeeze（BB 收進 Keltner，等爆發）
2. Squeeze Breakout（squeeze 結束 + bandwidth 爆發 + 量價配合）
3. Walk the Band（強趨勢中連續觸軌跟單）
4. Mean Reversion（盤整中觸軌反轉）

設計原則：
- 每個策略只在對應 regime 觸發（trending vs ranging 用不同策略）
- 依當下 features dict 判定，不需歷史價格序列（除了 lag1）
- 回傳結構化 dict 讓 auto_scanner 整合到掃描結果
- 所有 threshold 從 settings 讀（方便 calibration），fallback 用合理預設值
"""
from __future__ import annotations

from typing import Optional


# ─── 訊號類型常數 ────────────────────────────────────────

SIGNAL_SQUEEZE_ACTIVE = "SQUEEZE_ACTIVE"
SIGNAL_SQUEEZE_BREAKOUT_UP = "SQUEEZE_BREAKOUT_UP"
SIGNAL_SQUEEZE_BREAKOUT_DOWN = "SQUEEZE_BREAKOUT_DOWN"
SIGNAL_WALKING_UPPER = "WALKING_UPPER_BAND"
SIGNAL_WALKING_LOWER = "WALKING_LOWER_BAND"
SIGNAL_REVERSION_FROM_UPPER = "MEAN_REVERSION_FROM_UPPER"
SIGNAL_REVERSION_FROM_LOWER = "MEAN_REVERSION_FROM_LOWER"

EMOJI_MAP: dict[str, str] = {
    SIGNAL_SQUEEZE_ACTIVE: "⚡",
    SIGNAL_SQUEEZE_BREAKOUT_UP: "🟢",
    SIGNAL_SQUEEZE_BREAKOUT_DOWN: "🔴",
    SIGNAL_WALKING_UPPER: "⬆️",
    SIGNAL_WALKING_LOWER: "⬇️",
    SIGNAL_REVERSION_FROM_UPPER: "↩️🔻",
    SIGNAL_REVERSION_FROM_LOWER: "↪️🟢",
}

LABEL_MAP: dict[str, str] = {
    SIGNAL_SQUEEZE_ACTIVE: "Squeeze 進行中",
    SIGNAL_SQUEEZE_BREAKOUT_UP: "Squeeze 突破上軌",
    SIGNAL_SQUEEZE_BREAKOUT_DOWN: "Squeeze 突破下軌",
    SIGNAL_WALKING_UPPER: "Walking 上軌",
    SIGNAL_WALKING_LOWER: "Walking 下軌",
    SIGNAL_REVERSION_FROM_UPPER: "上軌反轉",
    SIGNAL_REVERSION_FROM_LOWER: "下軌反轉",
}

STRATEGY_MAP: dict[str, str] = {
    SIGNAL_SQUEEZE_ACTIVE: "the_squeeze",
    SIGNAL_SQUEEZE_BREAKOUT_UP: "squeeze_breakout",
    SIGNAL_SQUEEZE_BREAKOUT_DOWN: "squeeze_breakout",
    SIGNAL_WALKING_UPPER: "walk_the_band",
    SIGNAL_WALKING_LOWER: "walk_the_band",
    SIGNAL_REVERSION_FROM_UPPER: "mean_reversion",
    SIGNAL_REVERSION_FROM_LOWER: "mean_reversion",
}


# ─── Thresholds（從 settings 讀，否則用預設值）─────────────

def _get_thresholds() -> dict:
    """從 settings 讀 threshold，缺則用預設值。"""
    defaults = {
        "squeeze_min_duration": 5,        # squeeze 至少持續 5 根才算「active」
        "breakout_bandwidth_roc": 5.0,    # bandwidth 變動率 > 5% 算爆發
        "walk_band_min_touches": 3,       # 近 5 根至少 3 根觸軌
        "walk_band_min_adx": 25.0,        # ADX > 25 才算強趨勢
        "upper_band_threshold": 0.9,      # bb_position > 0.9 算觸上軌（百分比/100）
        "lower_band_threshold": 0.1,      # bb_position < 0.1 算觸下軌
    }
    try:
        from app.core.config.settings import settings
        for key, default in defaults.items():
            attr = f"bollinger_{key}"
            if hasattr(settings, attr):
                defaults[key] = getattr(settings, attr)
    except Exception:
        pass
    return defaults


# ─── 個別策略 detector ────────────────────────────────────

def detect_squeeze_state(features: dict, thresholds: dict) -> Optional[str]:
    """The Squeeze：BB 收進 Keltner，等爆發。

    啟動條件：is_squeeze=True 且 squeeze_duration >= min_duration
    """
    if not features.get("is_squeeze"):
        return None
    if features.get("squeeze_duration", 0) < thresholds["squeeze_min_duration"]:
        return None
    return SIGNAL_SQUEEZE_ACTIVE


def detect_squeeze_breakout(features: dict, prev_features: dict, thresholds: dict) -> Optional[str]:
    """Squeeze Breakout：squeeze 結束的瞬間 + bandwidth 爆發 + 量價配合。

    啟動條件：
    - 前一根仍 squeeze、本根已釋放
    - bandwidth_roc > threshold（波動爆發）
    - obv_slope_10 > 0（量配合）
    - bb_position 突破上/下軌
    """
    if not prev_features.get("is_squeeze"):
        return None
    if features.get("is_squeeze"):
        return None  # 還在 squeeze，未爆發

    if features.get("bb_width_roc", 0) < thresholds["breakout_bandwidth_roc"]:
        return None

    bb_pos = features.get("bb_position", 50) / 100.0  # bb_position 在 auto_scanner 是百分比 0-100
    bb_pos_lag1 = features.get("bb_position_lag1", 50) / 100.0

    # 上突破：前根還在通道內 (lag1 < 1.0)，本根衝出上軌 (pos > upper_threshold)
    if bb_pos_lag1 < 1.0 and bb_pos > thresholds["upper_band_threshold"]:
        if features.get("obv_slope_10", 0) > 0:
            return SIGNAL_SQUEEZE_BREAKOUT_UP

    # 下突破：前根還在通道內 (lag1 > 0.0)，本根跌破下軌 (pos < lower_threshold)
    if bb_pos_lag1 > 0.0 and bb_pos < thresholds["lower_band_threshold"]:
        if features.get("obv_slope_10", 0) < 0:
            return SIGNAL_SQUEEZE_BREAKOUT_DOWN

    return None


def detect_walk_the_band(features: dict, recent_bb_positions: list[float], thresholds: dict) -> Optional[str]:
    """Walk the Band：連續觸軌 + 強趨勢（ADX > 25）。

    啟動條件：
    - ADX > min_adx
    - 近 5 根至少 N 根觸上/下軌
    """
    if features.get("adx", 0) < thresholds["walk_band_min_adx"]:
        return None
    if len(recent_bb_positions) < 5:
        return None

    last5 = recent_bb_positions[-5:]
    upper_threshold_pct = thresholds["upper_band_threshold"] * 100  # auto_scanner bb_position 是 0-100
    lower_threshold_pct = thresholds["lower_band_threshold"] * 100

    touches_upper = sum(1 for p in last5 if p >= upper_threshold_pct)
    touches_lower = sum(1 for p in last5 if p <= lower_threshold_pct)

    if touches_upper >= thresholds["walk_band_min_touches"]:
        return SIGNAL_WALKING_UPPER
    if touches_lower >= thresholds["walk_band_min_touches"]:
        return SIGNAL_WALKING_LOWER
    return None


def detect_mean_reversion(features: dict, prev_features: dict, regime: str, thresholds: dict) -> Optional[str]:
    """Mean Reversion：盤整中觸軌反轉。

    啟動條件：
    - regime 為 ranging
    - 前根突破軌外（bb_position_lag1 > 100 或 < 0），本根回到通道內
    """
    if regime not in ("ranging", "unknown"):
        return None

    bb_pos = features.get("bb_position", 50)
    bb_pos_lag1 = features.get("bb_position_lag1", 50)
    upper_threshold_pct = thresholds["upper_band_threshold"] * 100
    lower_threshold_pct = thresholds["lower_band_threshold"] * 100

    # 上軌觸後反轉：前根 > 上軌、本根回到通道內偏中
    if bb_pos_lag1 > 100 and bb_pos < upper_threshold_pct:
        return SIGNAL_REVERSION_FROM_UPPER

    # 下軌觸後反轉：前根 < 下軌、本根回到通道內偏中
    if bb_pos_lag1 < 0 and bb_pos > lower_threshold_pct:
        return SIGNAL_REVERSION_FROM_LOWER

    return None


# ─── Regime-aware 策略選擇 ────────────────────────────────

# regime → 該 regime 適用的策略訊號優先順序（前面優先）
_REGIME_SIGNAL_PRIORITY: dict[str, list[str]] = {
    "trending_up": [SIGNAL_WALKING_UPPER, SIGNAL_SQUEEZE_BREAKOUT_UP, SIGNAL_SQUEEZE_ACTIVE],
    "trending_down": [SIGNAL_WALKING_LOWER, SIGNAL_SQUEEZE_BREAKOUT_DOWN, SIGNAL_SQUEEZE_ACTIVE],
    "ranging": [
        SIGNAL_REVERSION_FROM_UPPER, SIGNAL_REVERSION_FROM_LOWER,
        SIGNAL_SQUEEZE_BREAKOUT_UP, SIGNAL_SQUEEZE_BREAKOUT_DOWN,
        SIGNAL_SQUEEZE_ACTIVE,
    ],
    "unknown": [
        SIGNAL_SQUEEZE_ACTIVE, SIGNAL_SQUEEZE_BREAKOUT_UP, SIGNAL_SQUEEZE_BREAKOUT_DOWN,
        SIGNAL_REVERSION_FROM_UPPER, SIGNAL_REVERSION_FROM_LOWER,
    ],
}


# ─── Entry / Exit / Stop 規則 ─────────────────────────────

def get_entry_exit_stop(
    signal_type: str,
    features: dict,
    atr: float,
    close: float,
    sma20: float,
    bb_upper: float,
    bb_lower: float,
) -> dict:
    """每個訊號類型對應的進出場 / 停損規則。

    回傳：{entry, stop, target_1, target_2, rr_1, rr_2}
    """
    result = {"entry": close, "stop": None, "target_1": None, "target_2": None}

    if signal_type == SIGNAL_SQUEEZE_BREAKOUT_UP:
        # 突破上軌：跌破中軌出，target 用 2-4 ATR
        result["stop"] = sma20
        result["target_1"] = close + 2 * atr
        result["target_2"] = close + 4 * atr

    elif signal_type == SIGNAL_SQUEEZE_BREAKOUT_DOWN:
        # 突破下軌（空單）：漲過中軌出，target 用 2-4 ATR
        result["stop"] = sma20
        result["target_1"] = close - 2 * atr
        result["target_2"] = close - 4 * atr

    elif signal_type == SIGNAL_WALKING_UPPER:
        # 跟趨勢：trailing 用中軌，target 開放（用 N×ATR 動態）
        result["stop"] = sma20
        result["target_1"] = close + 3 * atr
        result["target_2"] = close + 6 * atr

    elif signal_type == SIGNAL_WALKING_LOWER:
        result["stop"] = sma20
        result["target_1"] = close - 3 * atr
        result["target_2"] = close - 6 * atr

    elif signal_type == SIGNAL_REVERSION_FROM_UPPER:
        # 上軌反轉做空：止損上軌、target 中軌
        result["stop"] = bb_upper
        result["target_1"] = sma20
        result["target_2"] = bb_lower

    elif signal_type == SIGNAL_REVERSION_FROM_LOWER:
        # 下軌反轉做多：止損下軌、target 中軌
        result["stop"] = bb_lower
        result["target_1"] = sma20
        result["target_2"] = bb_upper

    elif signal_type == SIGNAL_SQUEEZE_ACTIVE:
        # Squeeze 進行中無進場規則，只是「待爆發」狀態
        result["stop"] = None
        result["target_1"] = None
        result["target_2"] = None

    # 計算 RR（若可算）
    if result["stop"] is not None and result["stop"] != close:
        risk = abs(close - result["stop"])
        if risk > 1e-10:
            if result["target_1"] is not None:
                reward_1 = abs(result["target_1"] - close)
                result["rr_1"] = round(reward_1 / risk, 2)
            if result["target_2"] is not None:
                reward_2 = abs(result["target_2"] - close)
                result["rr_2"] = round(reward_2 / risk, 2)

    return result


# ─── 主入口：classify_bollinger_signal ────────────────────

def classify_bollinger_signal(
    features: dict,
    prev_features: Optional[dict],
    recent_bb_positions: Optional[list[float]],
    regime: str,
    close: float,
    sma20: float,
    bb_upper: float,
    bb_lower: float,
    atr: float,
) -> Optional[dict]:
    """主入口：依 regime 選擇可能適用策略，回傳第一個觸發的訊號。

    Args:
        features: 當前根的特徵 dict（含 bb_position, is_squeeze, squeeze_duration, etc.）
        prev_features: 前一根的特徵 dict（給 breakout / reversion 偵測用）
        recent_bb_positions: 近 5 根的 bb_position 序列（給 walk_the_band 偵測用）
        regime: 當前 regime label（trending_up / trending_down / ranging / unknown）
        close / sma20 / bb_upper / bb_lower / atr: 當下價格與通道值（給 entry/exit 計算）

    Returns:
        dict 或 None：
          {
            "signal": SIGNAL_*,            # 訊號類型常數
            "label": "Squeeze 突破上軌",   # 中文標籤
            "emoji": "🟢",                  # 視覺 emoji
            "strategy": "squeeze_breakout", # 策略名稱
            "entry_exit": {...},            # entry / stop / target / rr
            "features_used": {...},         # 觸發訊號用到的特徵值（給 LLM 解釋用）
          }
    """
    if not features or not regime:
        return None

    thresholds = _get_thresholds()

    # 依 regime 取得策略優先順序
    priority = _REGIME_SIGNAL_PRIORITY.get(regime, _REGIME_SIGNAL_PRIORITY["unknown"])

    # 依優先順序評估各 detector
    detected_signal: Optional[str] = None
    for candidate in priority:
        if candidate == SIGNAL_SQUEEZE_ACTIVE:
            detected_signal = detect_squeeze_state(features, thresholds)
        elif candidate in (SIGNAL_SQUEEZE_BREAKOUT_UP, SIGNAL_SQUEEZE_BREAKOUT_DOWN):
            if prev_features:
                detected_signal = detect_squeeze_breakout(features, prev_features, thresholds)
                # detect_squeeze_breakout 可能回 UP 或 DOWN，跟 candidate 不一定一樣 — 都接受
        elif candidate in (SIGNAL_WALKING_UPPER, SIGNAL_WALKING_LOWER):
            if recent_bb_positions:
                detected_signal = detect_walk_the_band(features, recent_bb_positions, thresholds)
        elif candidate in (SIGNAL_REVERSION_FROM_UPPER, SIGNAL_REVERSION_FROM_LOWER):
            if prev_features:
                detected_signal = detect_mean_reversion(features, prev_features, regime, thresholds)

        if detected_signal:
            break

    if not detected_signal:
        return None

    entry_exit = get_entry_exit_stop(
        detected_signal, features, atr, close, sma20, bb_upper, bb_lower,
    )

    return {
        "signal": detected_signal,
        "label": LABEL_MAP[detected_signal],
        "emoji": EMOJI_MAP[detected_signal],
        "strategy": STRATEGY_MAP[detected_signal],
        "entry_exit": entry_exit,
        "features_used": {
            "bb_position": features.get("bb_position"),
            "bb_position_lag1": features.get("bb_position_lag1"),
            "bb_width": features.get("bb_width"),
            "bb_width_roc": features.get("bb_width_roc"),
            "z_score_20": features.get("z_score_20"),
            "atr_relative": features.get("atr_relative"),
            "obv_slope_10": features.get("obv_slope_10"),
            "is_squeeze": features.get("is_squeeze"),
            "squeeze_duration": features.get("squeeze_duration"),
            "adx": features.get("adx"),
        },
    }
