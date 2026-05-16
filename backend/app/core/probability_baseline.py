"""純價格 unconditional baseline 機率計算。

回答的問題：「在過去 N 根 K 線中，任意時刻 forward_bars 根內，
價格達到 target_pct% 漲幅（或跌幅）的歷史頻率是多少？」

特性：
- Path-dependent：掃整段 forward_bars 內最高/最低，不是固定終點
- Unconditional：不過濾任何技術指標、regime、時段
- 用 Wilson CI 給樣本量誤差範圍

用途：作為 P1 機率三聯區塊中的「📊 純價格 baseline」，與 TA 條件化機率
並列顯示，讓使用者看見「TA 條件化帶來多少 lift」。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from app.core.stats_utils import wilson_ci


def calc_unconditional_baseline(
    df: pd.DataFrame,
    forward_bars: int,
    target_pct: float,
    direction: str = "up",
) -> dict:
    """計算純價格 unconditional baseline 機率（含 Wilson 95% CI）。

    Args:
        df: 含 high/low/close 欄位的 DataFrame
        forward_bars: 前瞻視窗根數
        target_pct: 目標漲幅 / 跌幅百分比（正數）
        direction: "up" = 看 high 達 target；"down" = 看 low 跌 target

    Returns:
        dict with: status, prob_pct, n, ci_pct, ci_width_pp, params
        status ∈ {"ok", "insufficient_data"}
    """
    params = {
        "forward_bars": forward_bars,
        "target_pct": target_pct,
        "direction": direction,
    }

    if df is None or len(df) < forward_bars + 20:
        return {
            "status": "insufficient_data",
            "reason": f"df_len={len(df) if df is not None else 0} < {forward_bars + 20}",
            "params": params,
        }

    try:
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
    except (KeyError, ValueError) as e:
        return {"status": "insufficient_data", "reason": f"df_columns_error: {e}", "params": params}

    n = len(closes)
    valid_range = n - forward_bars
    if valid_range <= 0:
        return {"status": "insufficient_data", "reason": "valid_range<=0", "params": params}

    future_hit = np.zeros(valid_range, dtype=bool)
    for i in range(valid_range):
        c0 = closes[i]
        if c0 <= 0 or np.isnan(c0):
            continue
        if direction == "up":
            window_max = float(np.max(highs[i + 1 : i + forward_bars + 1]))
            pct = (window_max - c0) / c0 * 100
        else:
            window_min = float(np.min(lows[i + 1 : i + forward_bars + 1]))
            pct = (c0 - window_min) / c0 * 100
        future_hit[i] = pct >= target_pct

    valid_mask = ~np.isnan(closes[:valid_range]) & (closes[:valid_range] > 0)
    total_valid = int(valid_mask.sum())
    if total_valid == 0:
        return {"status": "insufficient_data", "reason": "no_valid_samples", "params": params}

    total_hits = int(future_hit[valid_mask].sum())
    prob_pct = round(total_hits / total_valid * 100, 1)
    ci_lo, ci_hi = wilson_ci(total_hits, total_valid)

    return {
        "status": "ok",
        "prob_pct": prob_pct,
        "n": total_valid,
        "hits": total_hits,
        "ci_pct": [ci_lo, ci_hi],
        "ci_width_pp": round(ci_hi - ci_lo, 1),
        "params": params,
    }
