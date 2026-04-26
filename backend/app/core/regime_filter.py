"""Regime-aware 策略過濾器：給回測引擎決定「策略此刻能不能進場」。

設計原則：
- 規則式分類（不依賴 ML），可解釋、不易過擬合
- 重用既有 _detect_market_regime + 加 ATR 波動率分支
- 每根 K 線可分類，回測引擎用此 mask 過濾進場
- 同時輸出 confidence，給 unknown 警告機制用

Regime 類型（6 種）：
  - trending_up    : 上升趨勢（HH+HL + ADX>25 + SMA20>SMA50）
  - trending_down  : 下降趨勢（LH+LL + ADX>25 + SMA20<SMA50）
  - ranging        : 震盪（混合結構 + ADX<20 + 低波動）
  - high_vol       : 高波動（ATR/close 高於 60 日中位數 1.5x）
  - low_vol        : 低波動（ATR/close 低於 60 日中位數 0.5x）
  - unknown        : 以上都不夠明確（confidence < 0.5）

策略 → regime 對應表：
  - 趨勢跟蹤類 → trending_up / trending_down
  - 均值回歸類 → ranging / low_vol
  - 動量突破類 → trending_up + high_vol（需要趨勢 + 動能）
  - 波動率交易 → high_vol（保留給未來）
"""

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# 策略類型 → 相容 regime 對應表
# 策略名稱使用「關鍵字」匹配，避免每個策略都要明確標註
_STRATEGY_REGIME_MAP = {
    "趨勢跟蹤": {"trending_up", "trending_down"},
    "趨勢追蹤": {"trending_up", "trending_down"},  # 同義
    "trend": {"trending_up", "trending_down"},
    "均值回歸": {"ranging", "low_vol"},
    "mean_reversion": {"ranging", "low_vol"},
    "動量突破": {"trending_up", "high_vol"},
    "突破": {"trending_up", "high_vol"},
    "momentum": {"trending_up", "high_vol"},
    "breakout": {"trending_up", "high_vol"},
    "波動率": {"high_vol"},
}


def _slice_for_classification(df: pd.DataFrame, bar_idx: int, lookback: int = 60) -> pd.DataFrame:
    """切出某根 K 線之前 `lookback` 根用於分類（避免 look-ahead）。"""
    start = max(0, bar_idx - lookback)
    end = bar_idx + 1  # 包含當前
    return df.iloc[start:end]


def _compute_adx_simple(df_window: pd.DataFrame, period: int = 14) -> Optional[float]:
    """輕量 ADX 計算（最後一根值）— 避免依賴 indicator registry 的 key 命名。

    跟 tw_bb_scanner._compute_adx 同邏輯。
    """
    if len(df_window) < period * 2:
        return None
    high = df_window["high"]
    low = df_window["low"]
    close = df_window["close"]
    up_move = high.diff()
    down_move = -low.diff()
    # 用 high.index 確保所有 Series 共用同一 index（否則 rolling/division 會 NaN）
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * plus_dm.rolling(period).mean() / atr
    minus_di = 100 * minus_dm.rolling(period).mean() / atr
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denom
    adx = dx.rolling(period).mean()
    if adx.empty or np.isnan(adx.iloc[-1]):
        return None
    return float(adx.iloc[-1])


def _classify_volatility(df_window: pd.DataFrame) -> Optional[str]:
    """根據 ATR/close 比率判斷波動率 regime。

    Returns: "high_vol" / "low_vol" / None（中性）
    """
    if len(df_window) < 30 or "high" not in df_window.columns:
        return None
    high = df_window["high"].values
    low = df_window["low"].values
    close = df_window["close"].values

    # 簡化 ATR：(high - low) 14 日平均
    tr = high - low
    atr = pd.Series(tr).rolling(14).mean().values
    if np.isnan(atr[-1]) or close[-1] <= 0:
        return None
    current_atr_pct = atr[-1] / close[-1]

    # 跟最近 60 根的中位數比較
    atr_pct_series = atr / close
    valid = atr_pct_series[~np.isnan(atr_pct_series)]
    if len(valid) < 20:
        return None
    median = float(np.median(valid))
    if median <= 0:
        return None

    ratio = current_atr_pct / median
    if ratio > 1.5:
        return "high_vol"
    if ratio < 0.5:
        return "low_vol"
    return None


def classify_regime_at_bar(df: pd.DataFrame, bar_idx: int) -> dict:
    """單根 K 線 regime 分類，回傳 6 類之一 + confidence。

    Args:
        df: 完整 OHLCV
        bar_idx: 要分類的 K 線索引

    Returns:
        {
            "regime": "trending_up" / "trending_down" / "ranging" / "high_vol" / "low_vol" / "unknown",
            "confidence": 0.0~1.0,
            "details": {...}
        }
    """
    from app.core.ml.scenario_predictor import _detect_market_regime

    df_window = _slice_for_classification(df, bar_idx, lookback=60)
    if len(df_window) < 50:
        return {"regime": "unknown", "confidence": 0.0, "details": {"reason": "insufficient_data"}}

    # 1. 既有的 trend regime 分類
    base = _detect_market_regime(df_window)
    base_regime = base.get("regime", "unknown")
    bullish_score = base.get("bullish_score", 0.5)
    details = base.get("details", {})

    # 2. 自行算 ADX（既有 _detect_market_regime 取 ADX 有 bug 永遠拿 None）
    adx = _compute_adx_simple(df_window, period=14)
    details["adx"] = round(adx, 1) if adx is not None else None

    # 3. 波動率分支
    vol_regime = _classify_volatility(df_window)

    # 4. 結合判斷 + confidence 計算
    regime = "unknown"
    confidence = 0.5

    has_strong_trend = adx is not None and adx > 25

    # 趨勢類 regime（有強趨勢才算）
    if base_regime == "uptrend" and has_strong_trend:
        regime = "trending_up"
        confidence = 0.65 + (min(adx, 50) - 25) / 100  # adx 越高 confidence 越高
    elif base_regime == "downtrend" and has_strong_trend:
        regime = "trending_down"
        confidence = 0.65 + (min(adx, 50) - 25) / 100

    # 震盪類 regime（弱趨勢 + 結構混合）
    elif base_regime == "ranging" and (adx is None or adx < 20):
        regime = "ranging"
        confidence = 0.6 if vol_regime != "high_vol" else 0.4

    # 波動率 regime（有明確高低波動 + 弱趨勢）
    elif vol_regime is not None and (adx is None or adx < 25):
        regime = vol_regime
        confidence = 0.55

    # 都不夠明確 → unknown
    else:
        regime = "unknown"
        confidence = 0.3

    confidence = max(0.0, min(1.0, confidence))

    return {
        "regime": regime,
        "confidence": round(confidence, 3),
        "bullish_score": bullish_score,
        "details": {
            **details,
            "vol_regime": vol_regime,
        },
    }


def classify_regime_series(df: pd.DataFrame, step: int = 5) -> list[dict]:
    """整個 series 每根 K 線分類（給回測 mask 用）。

    為了效能，每 `step` 根分類一次，中間填上一個的結果。

    Returns:
        list of dict（長度 = len(df)），每個含 regime / confidence
    """
    n = len(df)
    if n == 0:
        return []

    results: list[dict] = [{}] * n
    last_classification = {"regime": "unknown", "confidence": 0.0, "details": {}}

    for i in range(0, n, step):
        if i < 50:
            # 前 50 根資料不足，全標 unknown
            last_classification = {"regime": "unknown", "confidence": 0.0, "details": {"reason": "warmup"}}
        else:
            try:
                last_classification = classify_regime_at_bar(df, i)
            except Exception as e:
                logger.debug(f"classify_regime_at_bar bar={i} failed: {e}")
                last_classification = {"regime": "unknown", "confidence": 0.0, "details": {"reason": str(e)[:30]}}

        # 填補 step 範圍
        end = min(i + step, n)
        for j in range(i, end):
            results[j] = last_classification

    return results


def get_compatible_regimes(strategy_name: str) -> Optional[set[str]]:
    """根據策略名稱關鍵字推斷相容 regime。

    Returns:
        set of regime names；若沒匹配任何關鍵字回傳 None（表示「all」，不過濾）
    """
    if not strategy_name:
        return None
    name_lower = strategy_name.lower()
    matched: set[str] = set()
    for keyword, regimes in _STRATEGY_REGIME_MAP.items():
        if keyword in strategy_name or keyword.lower() in name_lower:
            matched.update(regimes)
    return matched if matched else None


def is_strategy_compatible(strategy_name: str, regime: str) -> bool:
    """策略名稱 vs 當前 regime 相容性判斷。"""
    compatible = get_compatible_regimes(strategy_name)
    if compatible is None:
        return True  # 沒匹配任何關鍵字 → 不過濾，所有 regime 都相容
    return regime in compatible


def summarize_current_regime(df: pd.DataFrame) -> dict:
    """給 chart_state 注入用的「當前 regime」摘要。

    Returns:
        {
            "regime": "...",
            "confidence": 0.xx,
            "interpretation": "文字解讀（給 LLM 直接引用）",
            "details": {...}
        }
    """
    if df is None or len(df) < 50:
        return {"regime": "unknown", "confidence": 0.0, "interpretation": "資料不足無法判斷 regime"}

    last_idx = len(df) - 1
    result = classify_regime_at_bar(df, last_idx)

    # 文字解讀
    regime_label = {
        "trending_up": "上升趨勢",
        "trending_down": "下降趨勢",
        "ranging": "震盪盤整",
        "high_vol": "高波動",
        "low_vol": "低波動",
        "unknown": "未明",
    }.get(result["regime"], result["regime"])

    interp = f"目前 regime: {regime_label}（信心 {result['confidence']*100:.0f}%）"
    if result["confidence"] < 0.5:
        interp += "。⚠️ 低信心狀態，所有策略建議僅供參考"
    elif result["confidence"] < 0.7:
        interp += "。中等信心，建議搭配多重指標確認"
    else:
        interp += "。高信心，可依此 regime 對應的策略類型操作"

    result["interpretation"] = interp
    return result
