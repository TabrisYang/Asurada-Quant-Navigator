"""阿斯拉量化系統 — Monte Carlo 模擬

打亂交易順序重新模擬 N 次，產生報酬率和回撤的分佈，
用於檢測策略穩定性和評估真實風險。
"""

import numpy as np
from typing import Optional


def run_monte_carlo(
    trade_pnl_pcts: list[float],
    initial_capital: float = 10000.0,
    n_simulations: int = 1000,
    confidence_levels: Optional[list[float]] = None,
) -> dict:
    """執行 Monte Carlo 模擬。

    Args:
        trade_pnl_pcts: 每筆交易的盈虧百分比列表（如 [0.05, -0.02, 0.08, ...]）
        initial_capital: 初始資金
        n_simulations: 模擬次數
        confidence_levels: 信賴區間（預設 [0.05, 0.25, 0.50, 0.75, 0.95]）

    Returns:
        dict: 包含分佈統計、信賴區間、破產機率等
    """
    if not trade_pnl_pcts or len(trade_pnl_pcts) < 3:
        return {
            "status": "insufficient_data",
            "message": "至少需要 3 筆交易才能執行 Monte Carlo 模擬",
        }

    if confidence_levels is None:
        confidence_levels = [0.05, 0.25, 0.50, 0.75, 0.95]

    pnls = np.array(trade_pnl_pcts)
    n_trades = len(pnls)

    final_returns = np.empty(n_simulations)
    max_drawdowns = np.empty(n_simulations)
    ruin_count = 0

    for i in range(n_simulations):
        shuffled = np.random.permutation(pnls)

        equity = np.empty(n_trades + 1)
        equity[0] = initial_capital
        for j in range(n_trades):
            new_val = equity[j] * (1 + shuffled[j])
            equity[j + 1] = max(new_val, 0.0)  # 資金不能為負
            if equity[j + 1] <= 0:
                equity[j + 1:] = 0.0
                break

        final_returns[i] = (equity[-1] / equity[0] - 1) * 100 if equity[0] > 0 else -100.0

        peak = np.maximum.accumulate(equity)
        # 避免 peak=0 時除零
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = np.where(peak > 0, (equity - peak) / peak, 0.0)
        max_drawdowns[i] = float(np.nanmin(dd)) * 100

        if equity[-1] < initial_capital * 0.1:
            ruin_count += 1

    # 統計
    return_percentiles = {}
    dd_percentiles = {}
    for cl in confidence_levels:
        pct = int(cl * 100)
        return_percentiles[f"p{pct}"] = round(float(np.percentile(final_returns, pct)), 2)
        dd_percentiles[f"p{pct}"] = round(float(np.percentile(max_drawdowns, pct)), 2)

    return {
        "status": "success",
        "n_simulations": n_simulations,
        "n_trades": n_trades,
        "final_return": {
            "mean": round(float(np.mean(final_returns)), 2),
            "median": round(float(np.median(final_returns)), 2),
            "std": round(float(np.std(final_returns)), 2),
            "min": round(float(np.min(final_returns)), 2),
            "max": round(float(np.max(final_returns)), 2),
            "percentiles": return_percentiles,
        },
        "max_drawdown": {
            "mean": round(float(np.mean(max_drawdowns)), 2),
            "median": round(float(np.median(max_drawdowns)), 2),
            "worst_5pct": round(float(np.percentile(max_drawdowns, 5)), 2),
            "percentiles": dd_percentiles,
        },
        "ruin_probability": round(ruin_count / n_simulations * 100, 2),
        "profit_probability": round(float(np.mean(final_returns > 0)) * 100, 2),
        "strategy_robust": bool(np.percentile(final_returns, 25) > 0),
        "interpretation": _interpret_results(
            float(np.median(final_returns)),
            float(np.percentile(final_returns, 5)),
            float(np.mean(max_drawdowns)),
            ruin_count / n_simulations,
        ),
    }


def _interpret_results(
    median_return: float,
    p5_return: float,
    avg_dd: float,
    ruin_pct: float,
) -> str:
    """自動產生結果解讀。"""
    parts = []

    if p5_return > 0:
        parts.append("✅ 策略穩健：即使在最差 5% 情境下仍能獲利")
    elif median_return > 0:
        parts.append("⚠️ 策略中等：中位數獲利，但最差情境可能虧損")
    else:
        parts.append("❌ 策略不穩：中位數報酬為負，長期期望值可能為負")

    if ruin_pct > 0.05:
        parts.append(f"❌ 破產風險偏高（{ruin_pct*100:.1f}%），建議降低槓桿或倉位")
    elif ruin_pct > 0.01:
        parts.append(f"⚠️ 有小幅破產風險（{ruin_pct*100:.1f}%）")
    else:
        parts.append("✅ 破產風險極低")

    if avg_dd < -30:
        parts.append(f"⚠️ 平均回撤 {avg_dd:.1f}% 偏大，心理承受壓力高")

    return "；".join(parts)
