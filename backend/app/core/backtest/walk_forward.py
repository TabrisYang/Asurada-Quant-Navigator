"""阿斯拉量化系統 — Walk Forward Analysis

將數據分成多個滑動窗口，每個窗口用訓練集優化、測試集驗證，
確保策略在未見數據上持續有效，是最嚴格的過擬合檢測方法。
"""

import numpy as np
import pandas as pd
from typing import Optional

from app.core.backtest.engine import run_backtest


def run_walk_forward(
    df: pd.DataFrame,
    entry_conditions: list[dict],
    exit_conditions: list[dict],
    direction: str = "long",
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    n_windows: int = 5,
    train_ratio: float = 0.7,
    initial_capital: float = 10000.0,
    leverage: float = 1.0,
) -> dict:
    """執行 Walk Forward Analysis。

    將數據切成 n_windows 個重疊窗口，每個窗口：
    - 前 train_ratio 用來回測（in-sample）
    - 後 (1-train_ratio) 用來驗證（out-of-sample）

    Args:
        df: 完整 OHLCV DataFrame
        entry_conditions / exit_conditions: 進出場條件
        n_windows: 窗口數量（5~10）
        train_ratio: 訓練集佔比（0.6~0.8）

    Returns:
        dict: 包含每個窗口的績效和整體穩定性評估
    """
    total_bars = len(df)
    if total_bars < 100:
        return {"status": "error", "message": f"數據不足（{total_bars} 根 K 線），至少需要 100 根"}

    min_bars_per_window = max(50, total_bars // (n_windows + 1))
    step_size = (total_bars - min_bars_per_window) // max(n_windows - 1, 1)
    window_size = min(min_bars_per_window + step_size, total_bars)

    if window_size < 50:
        return {"status": "error", "message": "每個窗口的數據量不足，請增加歷史數據或減少窗口數"}

    windows = []
    for i in range(n_windows):
        start = i * step_size
        end = min(start + window_size, total_bars)
        if end - start < 40:
            break

        split_idx = int((end - start) * train_ratio) + start
        train_df = df.iloc[start:split_idx].copy().reset_index(drop=True)
        test_df = df.iloc[split_idx:end].copy().reset_index(drop=True)

        if len(train_df) < 20 or len(test_df) < 10:
            continue

        train_result = run_backtest(
            train_df, entry_conditions, exit_conditions,
            direction=direction, stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct, initial_capital=initial_capital,
            leverage=leverage,
        )
        test_result = run_backtest(
            test_df, entry_conditions, exit_conditions,
            direction=direction, stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct, initial_capital=initial_capital,
            leverage=leverage,
        )

        train_m = train_result.metrics
        test_m = test_result.metrics

        windows.append({
            "window": i + 1,
            "period": f"bar {start}~{end}",
            "train_bars": len(train_df),
            "test_bars": len(test_df),
            "train": {
                "trades": train_m.get("total_trades", 0),
                "win_rate": train_m.get("win_rate", 0),
                "return_pct": train_m.get("total_return_pct", 0),
                "sharpe": train_m.get("sharpe_ratio", 0),
                "mdd": train_m.get("max_drawdown_pct", 0),
            },
            "test": {
                "trades": test_m.get("total_trades", 0),
                "win_rate": test_m.get("win_rate", 0),
                "return_pct": test_m.get("total_return_pct", 0),
                "sharpe": test_m.get("sharpe_ratio", 0),
                "mdd": test_m.get("max_drawdown_pct", 0),
            },
        })

    if len(windows) < 2:
        return {"status": "error", "message": "有效窗口不足 2 個，無法進行 Walk Forward 分析"}

    # 整體統計
    test_returns = [w["test"]["return_pct"] for w in windows]
    test_win_rates = [w["test"]["win_rate"] for w in windows]
    test_sharpes = [w["test"]["sharpe"] for w in windows]
    train_returns = [w["train"]["return_pct"] for w in windows]

    profitable_windows = sum(1 for r in test_returns if r > 0)
    consistency_ratio = profitable_windows / len(windows)

    # 訓練集 vs 測試集的效能衰減
    decay_ratios = []
    for w in windows:
        if w["train"]["return_pct"] != 0:
            decay = w["test"]["return_pct"] / w["train"]["return_pct"]
            decay_ratios.append(decay)

    avg_decay = float(np.mean(decay_ratios)) if decay_ratios else 0

    return {
        "status": "success",
        "n_windows": len(windows),
        "windows": windows,
        "summary": {
            "test_avg_return": round(float(np.mean(test_returns)), 2),
            "test_median_return": round(float(np.median(test_returns)), 2),
            "test_std_return": round(float(np.std(test_returns)), 2),
            "test_avg_win_rate": round(float(np.mean(test_win_rates)), 2),
            "test_avg_sharpe": round(float(np.mean(test_sharpes)), 3),
            "train_avg_return": round(float(np.mean(train_returns)), 2),
            "consistency_ratio": round(consistency_ratio * 100, 1),
            "performance_decay": round(avg_decay, 3),
        },
        "assessment": _assess_walk_forward(
            consistency_ratio, avg_decay, test_returns, test_sharpes
        ),
    }


def _assess_walk_forward(
    consistency: float,
    decay: float,
    test_returns: list[float],
    test_sharpes: list[float],
) -> dict:
    """評估 Walk Forward 結果。"""
    issues = []
    score = 100

    if consistency < 0.5:
        issues.append(f"一致性不足：僅 {consistency*100:.0f}% 窗口獲利，策略穩定性差")
        score -= 30
    elif consistency < 0.7:
        issues.append(f"一致性中等：{consistency*100:.0f}% 窗口獲利")
        score -= 15

    if 0 < decay < 0.3:
        issues.append(f"嚴重過擬合：測試集績效僅為訓練集的 {decay*100:.0f}%")
        score -= 30
    elif 0 < decay < 0.6:
        issues.append(f"中度過擬合：測試集績效為訓練集的 {decay*100:.0f}%")
        score -= 15

    ret_std = float(np.std(test_returns)) if len(test_returns) > 1 else 0
    if ret_std > 20:
        issues.append(f"報酬波動大（std={ret_std:.1f}%），策略在不同時期表現差異大")
        score -= 10

    avg_sharpe = float(np.mean(test_sharpes))
    if avg_sharpe < 0:
        issues.append(f"測試集平均 Sharpe {avg_sharpe:.2f} < 0，策略無 Alpha")
        score -= 20
    elif avg_sharpe < 0.5:
        issues.append(f"測試集平均 Sharpe {avg_sharpe:.2f} 偏低")
        score -= 10

    score = max(0, score)

    if score >= 70:
        verdict = "策略通過 Walk Forward 驗證，具備一定穩定性"
    elif score >= 40:
        verdict = "策略穩定性中等，建議優化或降低倉位"
    else:
        verdict = "策略未通過 Walk Forward 驗證，不建議實盤使用"

    return {
        "score": score,
        "verdict": verdict,
        "issues": issues,
        "has_alpha": avg_sharpe > 0.5 and consistency >= 0.6,
    }
