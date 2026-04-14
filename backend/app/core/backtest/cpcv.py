"""阿斯拉量化系統 — Combinatorial Purged Cross-Validation (CPCV)

比 Walk Forward 更嚴格的過擬合檢測方法。
將數據分成 N 個不重疊 group，所有 C(N,2) 的 test 組合都跑回測，
並在 train/test 邊界加入 purge + embargo 隔離。
"""

import numpy as np
from itertools import combinations
from typing import Optional

from app.core.backtest.engine import run_backtest


# SL/TP Grid Search（複用 walk_forward 的範圍）
SL_RANGE = [0.02, 0.03, 0.05, 0.07, 0.10]
TP_RANGE = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]


def run_cpcv(
    df,
    entry_conditions: list[dict],
    exit_conditions: list[dict],
    n_groups: int = 5,
    embargo: int = 5,
    optimize_sl_tp: bool = True,
    direction: str = "long",
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    leverage: float = 1.0,
) -> dict:
    """Combinatorial Purged Cross-Validation.

    N=5 → C(5,2)=10 個 train/test 組合
    每個組合：3 groups 訓練、2 groups 測試
    Purge: train/test 邊界移除 embargo 根 K 線

    Args:
        df: 完整 OHLCV DataFrame
        n_groups: 分割數（預設 5 → 10 組合）
        embargo: 邊界隔離 K 線數
        optimize_sl_tp: 每組合在 train 上優化 SL/TP

    Returns:
        dict: 跨所有組合的績效分佈和評估
    """
    total_bars = len(df)
    min_required = n_groups * 50 + (n_groups - 1) * embargo

    if total_bars < min_required:
        return {
            "status": "error",
            "message": f"數據不足（{total_bars} 根），CPCV N={n_groups} 至少需要 {min_required} 根",
        }

    # 分成 N 個不重疊 group
    group_size = total_bars // n_groups
    groups = []
    for i in range(n_groups):
        start = i * group_size
        end = start + group_size if i < n_groups - 1 else total_bars
        groups.append((start, end))

    # 所有 C(N, 2) 組合
    test_combos = list(combinations(range(n_groups), 2))
    combo_results = []

    for combo_idx, test_group_ids in enumerate(test_combos):
        train_group_ids = [g for g in range(n_groups) if g not in test_group_ids]

        # 建構 train/test 索引（含 purge + embargo）
        train_indices = _get_purged_train_indices(groups, train_group_ids, test_group_ids, embargo)
        test_indices = _get_test_indices(groups, test_group_ids)

        if len(train_indices) < 30 or len(test_indices) < 10:
            continue

        train_df = df.iloc[train_indices].reset_index(drop=True)
        test_df = df.iloc[test_indices].reset_index(drop=True)

        # Per-combo SL/TP 優化
        best_sl = stop_loss_pct
        best_tp = take_profit_pct
        if optimize_sl_tp:
            best_sl, best_tp = _optimize_params(
                train_df, entry_conditions, exit_conditions,
                direction, leverage, stop_loss_pct, take_profit_pct,
            )

        # 測試集回測
        try:
            test_result = run_backtest(
                test_df, entry_conditions, exit_conditions,
                direction=direction, stop_loss_pct=best_sl,
                take_profit_pct=best_tp, leverage=leverage,
            )
            metrics = test_result.metrics
            combo_results.append({
                "combo": f"test={list(test_group_ids)}",
                "train_bars": len(train_indices),
                "test_bars": len(test_indices),
                "optimized_sl": best_sl,
                "optimized_tp": best_tp,
                "trades": metrics.get("total_trades", 0),
                "win_rate": metrics.get("win_rate", 0),
                "return_pct": metrics.get("total_return_pct", 0),
                "sharpe": metrics.get("sharpe_ratio", 0),
                "mdd": metrics.get("max_drawdown_pct", 0),
            })
        except Exception:
            continue

    if len(combo_results) < 3:
        return {"status": "error", "message": "有效組合不足 3 個，無法進行 CPCV 分析"}

    # 統計
    returns = [r["return_pct"] for r in combo_results]
    sharpes = [r["sharpe"] for r in combo_results]
    win_rates = [r["win_rate"] for r in combo_results]

    profitable = sum(1 for r in returns if r > 0)
    consistency = profitable / len(combo_results)

    assessment = _assess_cpcv(returns, sharpes, win_rates, consistency)

    return {
        "status": "success",
        "n_groups": n_groups,
        "n_combinations": len(test_combos),
        "valid_combinations": len(combo_results),
        "embargo_bars": embargo,
        "metrics_distribution": {
            "return_pct": {
                "mean": round(float(np.mean(returns)), 2),
                "median": round(float(np.median(returns)), 2),
                "std": round(float(np.std(returns)), 2),
                "min": round(float(np.min(returns)), 2),
                "max": round(float(np.max(returns)), 2),
                "p25": round(float(np.percentile(returns, 25)), 2),
                "p75": round(float(np.percentile(returns, 75)), 2),
            },
            "sharpe": {
                "mean": round(float(np.mean(sharpes)), 3),
                "median": round(float(np.median(sharpes)), 3),
            },
            "win_rate": {
                "mean": round(float(np.mean(win_rates)), 1),
                "std": round(float(np.std(win_rates)), 1),
            },
            "consistency_pct": round(consistency * 100, 1),
        },
        "combo_details": combo_results,
        "assessment": assessment,
    }


def _get_purged_train_indices(groups, train_ids, test_ids, embargo):
    """取得 purged 的 train 索引：移除與 test group 相鄰的 embargo 根。"""
    test_ranges = set()
    for tid in test_ids:
        start, end = groups[tid]
        test_ranges.update(range(start, end))

    # 擴展 embargo zone
    embargo_zones = set()
    for tid in test_ids:
        start, end = groups[tid]
        for e in range(max(0, start - embargo), start):
            embargo_zones.add(e)
        for e in range(end, min(end + embargo, max(g[1] for g in groups))):
            embargo_zones.add(e)

    indices = []
    for gid in train_ids:
        start, end = groups[gid]
        for i in range(start, end):
            if i not in test_ranges and i not in embargo_zones:
                indices.append(i)

    return sorted(indices)


def _get_test_indices(groups, test_ids):
    """取得 test 索引。"""
    indices = []
    for tid in test_ids:
        start, end = groups[tid]
        indices.extend(range(start, end))
    return sorted(indices)


def _optimize_params(train_df, entry_cond, exit_cond, direction, leverage, default_sl, default_tp):
    """在 train 上 grid search 最佳 SL/TP。"""
    best_sharpe = -999.0
    best_sl = default_sl
    best_tp = default_tp

    for sl in SL_RANGE:
        for tp in TP_RANGE:
            if tp <= sl:
                continue
            try:
                result = run_backtest(
                    train_df, entry_cond, exit_cond,
                    direction=direction, stop_loss_pct=sl,
                    take_profit_pct=tp, leverage=leverage,
                )
                sharpe = result.metrics.get("sharpe_ratio", -999)
                n_trades = result.metrics.get("total_trades", 0)
                if sharpe > best_sharpe and n_trades >= 3:
                    best_sharpe = sharpe
                    best_sl = sl
                    best_tp = tp
            except Exception:
                continue

    return best_sl, best_tp


def _assess_cpcv(returns, sharpes, win_rates, consistency):
    """評估 CPCV 結果。"""
    score = 100
    issues = []

    # 一致性
    if consistency < 0.5:
        score -= 30
        issues.append(f"CPCV 一致性不足：僅 {consistency*100:.0f}% 組合獲利")
    elif consistency < 0.7:
        score -= 15
        issues.append(f"CPCV 一致性中等：{consistency*100:.0f}% 組合獲利")

    # Sharpe
    avg_sharpe = float(np.mean(sharpes))
    if avg_sharpe < 0:
        score -= 25
        issues.append(f"CPCV 平均 Sharpe {avg_sharpe:.2f} < 0，策略無邊際")
    elif avg_sharpe < 0.5:
        score -= 10
        issues.append(f"CPCV 平均 Sharpe {avg_sharpe:.2f} 偏低")

    # 報酬波動
    ret_std = float(np.std(returns))
    if ret_std > 20:
        score -= 10
        issues.append(f"CPCV 報酬波動大（std={ret_std:.1f}%）")

    # P25 報酬
    p25 = float(np.percentile(returns, 25))
    if p25 < -5:
        score -= 10
        issues.append(f"CPCV P25 報酬 {p25:.1f}%，最差 25% 組合虧損明顯")

    score = max(0, score)

    if score >= 70:
        verdict = "CPCV 驗證通過：策略在多數組合中穩定獲利"
    elif score >= 40:
        verdict = "CPCV 驗證中等：策略有一定穩定性但不夠一致"
    else:
        verdict = "CPCV 驗證未通過：策略過擬合風險高，不建議實盤"

    return {
        "score": score,
        "verdict": verdict,
        "issues": issues,
        "has_edge": avg_sharpe > 0.8 and consistency >= 0.7,
    }
