"""阿斯拉量化系統 — 信號融合層

將技術指標信號 + ML 預測 + 跨時間框架確認整合為單一綜合信號。
採用加權平均方式，權重可配置。
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from loguru import logger


@dataclass
class SignalSource:
    """單一信號來源"""
    name: str
    direction: str          # "long" / "short" / "neutral"
    confidence: float       # 0.0 ~ 1.0
    weight: float = 1.0     # 權重


@dataclass
class FusedSignal:
    """融合後的綜合信號"""
    direction: str          # "long" / "short" / "neutral"
    strength: float         # 0.0 ~ 1.0（信號強度）
    agreement_ratio: float  # 各來源方向一致性 0.0 ~ 1.0
    sources: list[dict] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "strength": round(self.strength, 4),
            "agreement_ratio": round(self.agreement_ratio, 4),
            "sources": self.sources,
            "recommendation": self.recommendation,
        }


# 預設信號權重
DEFAULT_WEIGHTS = {
    "technical": 0.35,      # 技術指標
    "ml_prediction": 0.40,  # ML 模型預測
    "multi_timeframe": 0.25,  # 跨時間框架確認
}


def fuse_signals(
    sources: list[SignalSource],
    weights: Optional[dict[str, float]] = None,
) -> FusedSignal:
    """融合多個信號來源為單一綜合信號

    Args:
        sources: 信號來源列表
        weights: 各來源類型的權重覆寫

    Returns:
        FusedSignal: 融合後的綜合信號
    """
    if not sources:
        return FusedSignal(
            direction="neutral",
            strength=0.0,
            agreement_ratio=0.0,
            recommendation="無信號來源",
        )

    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    # 計算加權分數：long=+1, short=-1, neutral=0
    weighted_score = 0.0
    total_weight = 0.0
    source_details = []

    for s in sources:
        signal_val = {"long": 1.0, "short": -1.0, "neutral": 0.0}.get(s.direction, 0.0)
        effective_weight = s.weight * w.get(s.name, 1.0)
        weighted_score += signal_val * s.confidence * effective_weight
        total_weight += effective_weight

        source_details.append({
            "name": s.name,
            "direction": s.direction,
            "confidence": round(s.confidence, 4),
            "weight": round(effective_weight, 4),
        })

    if total_weight == 0:
        return FusedSignal(direction="neutral", strength=0.0, agreement_ratio=0.0,
                           sources=source_details, recommendation="權重總和為零")

    # 正規化分數到 [-1, 1]
    normalized = weighted_score / total_weight

    # 判定方向
    if normalized > 0.15:
        direction = "long"
    elif normalized < -0.15:
        direction = "short"
    else:
        direction = "neutral"

    strength = min(abs(normalized), 1.0)

    # 計算方向一致性
    directions = [s.direction for s in sources if s.direction != "neutral"]
    if directions:
        most_common = max(set(directions), key=directions.count)
        agreement_ratio = directions.count(most_common) / len(directions)
    else:
        agreement_ratio = 0.0

    # 生成建議
    recommendation = _generate_recommendation(direction, strength, agreement_ratio)

    return FusedSignal(
        direction=direction,
        strength=strength,
        agreement_ratio=agreement_ratio,
        sources=source_details,
        recommendation=recommendation,
    )


def create_technical_signal(
    indicator_values: dict[str, float],
) -> SignalSource:
    """從技術指標值創建信號

    常見指標閾值：
    - RSI < 30: oversold (long), > 70: overbought (short)
    - MACD > 0: bullish, < 0: bearish
    """
    long_count = 0
    short_count = 0
    total = 0

    rsi = indicator_values.get("rsi")
    if rsi is not None:
        total += 1
        if rsi < 30:
            long_count += 1
        elif rsi > 70:
            short_count += 1

    macd = indicator_values.get("macd")
    if macd is not None:
        total += 1
        if macd > 0:
            long_count += 1
        elif macd < 0:
            short_count += 1

    adx = indicator_values.get("adx")
    if adx is not None and adx > 25:
        # ADX > 25 表示趨勢明確，增強信號信心
        total += 1
        # ADX 不決定方向，但增加信心

    if total == 0:
        return SignalSource(name="technical", direction="neutral", confidence=0.0)

    if long_count > short_count:
        direction = "long"
        confidence = long_count / total
    elif short_count > long_count:
        direction = "short"
        confidence = short_count / total
    else:
        direction = "neutral"
        confidence = 0.3

    return SignalSource(name="technical", direction=direction, confidence=confidence)


def create_ml_signal(
    probability: float,
    reliability_level: str = "medium",
) -> SignalSource:
    """從 ML 預測機率創建信號"""
    # 根據可靠性調整信心
    reliability_discount = {
        "high": 1.0,
        "medium": 0.8,
        "low": 0.5,
        "unreliable": 0.2,
    }.get(reliability_level, 0.5)

    if probability >= 0.6:
        direction = "long"
        confidence = (probability - 0.5) * 2 * reliability_discount
    elif probability <= 0.4:
        direction = "short"
        confidence = (0.5 - probability) * 2 * reliability_discount
    else:
        direction = "neutral"
        confidence = 0.0

    return SignalSource(name="ml_prediction", direction=direction, confidence=min(confidence, 1.0))


def create_multi_timeframe_signal(
    signals_by_tf: dict[str, str],
) -> SignalSource:
    """從多時間框架方向創建信號

    Args:
        signals_by_tf: {"15m": "long", "1h": "long", "4h": "short", ...}
    """
    if not signals_by_tf:
        return SignalSource(name="multi_timeframe", direction="neutral", confidence=0.0)

    # 高時間框架權重更大
    tf_weights = {"1w": 4.0, "1d": 3.0, "4h": 2.0, "1h": 1.5, "15m": 1.0}
    weighted_sum = 0.0
    total_weight = 0.0

    for tf, direction in signals_by_tf.items():
        w = tf_weights.get(tf, 1.0)
        val = {"long": 1.0, "short": -1.0, "neutral": 0.0}.get(direction, 0.0)
        weighted_sum += val * w
        total_weight += w

    if total_weight == 0:
        return SignalSource(name="multi_timeframe", direction="neutral", confidence=0.0)

    score = weighted_sum / total_weight
    if score > 0.2:
        direction = "long"
    elif score < -0.2:
        direction = "short"
    else:
        direction = "neutral"

    confidence = min(abs(score), 1.0)
    return SignalSource(name="multi_timeframe", direction=direction, confidence=confidence)


def _generate_recommendation(direction: str, strength: float, agreement: float) -> str:
    """生成人類可讀的建議"""
    if direction == "neutral":
        return "信號不明確，建議觀望"

    dir_text = "做多" if direction == "long" else "做空"

    if strength > 0.7 and agreement > 0.8:
        return f"強烈{dir_text}信號（強度 {strength:.0%}，一致性 {agreement:.0%}）"
    elif strength > 0.4:
        return f"中等{dir_text}信號（強度 {strength:.0%}，一致性 {agreement:.0%}），建議搭配其他確認"
    else:
        return f"弱{dir_text}信號（強度 {strength:.0%}），不建議單獨依據此信號操作"
