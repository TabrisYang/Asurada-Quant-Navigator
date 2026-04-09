"""阿斯拉量化系統 — 動態倉位管理

根據市場波動率 (ATR) 和 Kelly Criterion 動態調整倉位大小。
波動大時自動減倉，波動小時自動加倉。
"""

import numpy as np
import pandas as pd


def calculate_dynamic_positions(
    df: pd.DataFrame,
    base_risk_pct: float = 0.02,
    atr_period: int = 14,
    atr_ma_period: int = 50,
    max_position_pct: float = 1.0,
    min_position_pct: float = 0.1,
    method: str = "atr_volatility",
    win_rate: float = 0.5,
    avg_win_loss_ratio: float = 1.5,
    kelly_fraction: float = 0.25,
) -> dict:
    """計算每根 K 線的建議倉位大小。

    Args:
        df: OHLCV DataFrame
        base_risk_pct: 每筆交易的基礎風險（佔總資金百分比，如 0.02 = 2%）
        atr_period: ATR 計算週期
        atr_ma_period: ATR 移動平均（用於判斷波動是否高於正常）
        max_position_pct: 最大倉位（佔總資金）
        min_position_pct: 最小倉位
        method: 倉位計算方法
            - "atr_volatility": 固定風險百分比法（ATR 決定止損距離 → 反算倉位）
            - "atr_adaptive": 波動率自適應法（波動大減倉，波動小加倉）
            - "kelly_dynamic": Kelly + ATR 混合法
        win_rate: 策略勝率（Kelly 用）
        avg_win_loss_ratio: 平均盈虧比（Kelly 用）
        kelly_fraction: Kelly 分數倍數（建議 0.25 = quarter Kelly）

    Returns:
        dict: 包含每根 K 線的建議倉位 + 統計摘要
    """
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    # 計算 ATR
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1]),
        ),
    )
    tr = np.concatenate([[highs[0] - lows[0]], tr])

    atr = pd.Series(tr).rolling(atr_period, min_periods=1).mean().values
    atr_ma = pd.Series(atr).rolling(atr_ma_period, min_periods=1).mean().values

    positions = np.full(n, base_risk_pct)

    if method == "atr_volatility":
        # 固定風險法：position_size = risk_amount / (ATR * multiplier)
        # 用 ATR 作為止損距離，反算能承受的倉位
        for i in range(atr_period, n):
            if atr[i] > 0 and closes[i] > 0:
                stop_distance_pct = (atr[i] * 2) / closes[i]
                position = base_risk_pct / stop_distance_pct if stop_distance_pct > 0 else base_risk_pct
                positions[i] = np.clip(position, min_position_pct, max_position_pct)

    elif method == "atr_adaptive":
        # 波動自適應：ATR 高於均值 → 減倉，低於均值 → 加倉
        for i in range(max(atr_period, atr_ma_period), n):
            if atr_ma[i] > 0:
                vol_ratio = atr[i] / atr_ma[i]
                # vol_ratio > 1 = 高波動 → 減倉；< 1 = 低波動 → 加倉
                adaptive_factor = 1.0 / vol_ratio if vol_ratio > 0 else 1.0
                position = base_risk_pct * adaptive_factor
                positions[i] = np.clip(position, min_position_pct, max_position_pct)

    elif method == "kelly_dynamic":
        # Kelly（連續收益型）+ ATR 混合
        # 離散型 Kelly: (b*p - q) / b 假設只有贏/輸兩種結果
        # 連續型更適合交易：f = E[R] / E[L^2]，近似為 (p*W - q*L) / L
        p = win_rate
        q = 1 - p
        b = avg_win_loss_ratio
        if b > 0 and 0 < p < 1:
            # 連續收益 Kelly: 考慮平均獲利和平均虧損的比例
            # avg_win = b * avg_loss (by definition of win/loss ratio)
            # f = (p * avg_win - q * avg_loss) / avg_loss = p*b - q
            # 但需要除以 b 以得到佔比: f = (p*b - q) / b
            # 額外安全措施：加入二階修正項，避免高波動時過度下注
            raw_kelly = (p * b - q) / b
            # 二階修正：Kelly 乘以 (1 - variance_penalty)
            # variance_penalty 近似為 kelly^2 * variance_ratio
            variance_penalty = min(0.3, raw_kelly ** 2)
            full_kelly = raw_kelly * (1 - variance_penalty)
        else:
            full_kelly = 0
        base_kelly = max(0, full_kelly * kelly_fraction)

        for i in range(max(atr_period, atr_ma_period), n):
            if atr_ma[i] > 0:
                vol_ratio = atr[i] / atr_ma[i]
                adaptive_factor = 1.0 / vol_ratio if vol_ratio > 0 else 1.0
                position = base_kelly * adaptive_factor
                positions[i] = np.clip(position, min_position_pct, max_position_pct)

    positions_pct = (positions * 100).tolist()

    return {
        "method": method,
        "positions_pct": [round(p, 2) for p in positions_pct],
        "summary": {
            "avg_position_pct": round(float(np.mean(positions)) * 100, 2),
            "median_position_pct": round(float(np.median(positions)) * 100, 2),
            "min_position_pct": round(float(np.min(positions)) * 100, 2),
            "max_position_pct": round(float(np.max(positions)) * 100, 2),
            "current_position_pct": round(float(positions[-1]) * 100, 2),
            "current_atr": round(float(atr[-1]), 4),
            "current_atr_vs_avg": round(float(atr[-1] / atr_ma[-1]) if atr_ma[-1] > 0 else 1.0, 3),
        },
        "recommendation": _position_recommendation(
            float(positions[-1]),
            float(atr[-1] / atr_ma[-1]) if atr_ma[-1] > 0 else 1.0,
        ),
    }


def _position_recommendation(current_pos: float, vol_ratio: float) -> str:
    """產生倉位建議。"""
    if vol_ratio > 1.5:
        return f"⚠️ 當前波動率偏高（{vol_ratio:.1f}x），建議倉位 {current_pos*100:.1f}%，謹慎操作"
    elif vol_ratio > 1.2:
        return f"當前波動率略高（{vol_ratio:.1f}x），建議倉位 {current_pos*100:.1f}%"
    elif vol_ratio < 0.7:
        return f"當前波動率偏低（{vol_ratio:.1f}x），可適度加倉至 {current_pos*100:.1f}%"
    else:
        return f"波動率正常（{vol_ratio:.1f}x），建議倉位 {current_pos*100:.1f}%"
