"""阿斯拉量化系統 — Walk Forward Analysis（優化版）

將數據分成多個不重疊滑動窗口（含 embargo），每個窗口：
- 訓練集上可選擇重新優化 SL/TP 參數
- 測試集驗證

確保策略在未見數據上持續有效，是最嚴格的過擬合檢測方法。

優化項目：
- 不重疊窗口 + embargo period（避免持倉跨界洩漏）
- Per-window SL/TP grid search 優化
- 修正 decay ratio 計算
- 收集 OOS 交易供 Monte Carlo 使用
- 參數穩定性指標
"""

import numpy as np
import pandas as pd
from typing import Optional

from app.core.backtest.engine import run_backtest


# ── SL/TP Grid Search 範圍 ──
SL_RANGE = [0.02, 0.03, 0.05, 0.07, 0.10]
TP_RANGE = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]


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
    optimize_sl_tp: bool = True,
    embargo: int = 5,
) -> dict:
    """執行 Walk Forward Analysis（優化版）。

    將數據切成 n_windows 個不重疊窗口（中間隔 embargo），每個窗口：
    - 前 train_ratio 用來回測 / 優化 SL/TP（in-sample）
    - 後 (1-train_ratio) 用來驗證（out-of-sample）

    Args:
        df: 完整 OHLCV DataFrame
        entry_conditions / exit_conditions: 進出場條件
        n_windows: 窗口數量（3~10）
        train_ratio: 訓練集佔比（0.6~0.8）
        optimize_sl_tp: 是否在每窗口訓練集上重新優化 SL/TP
        embargo: 窗口間隔 K 線數（避免持倉跨界洩漏）

    Returns:
        dict: 包含每個窗口的績效、OOS 交易 PnL、參數穩定性
    """
    total_bars = len(df)
    if total_bars < 100:
        return {"status": "error", "message": f"數據不足（{total_bars} 根 K 線），至少需要 100 根"}

    # ── 不重疊窗口計算 ──
    total_embargo = (n_windows - 1) * embargo
    usable_bars = total_bars - total_embargo
    window_size = usable_bars // n_windows

    if window_size < 40:
        # 數據不夠分這麼多窗口，自動減少窗口數
        n_windows = max(2, (total_bars - embargo) // 50)
        total_embargo = (n_windows - 1) * embargo
        usable_bars = total_bars - total_embargo
        window_size = usable_bars // n_windows

    if window_size < 40 or n_windows < 2:
        return {"status": "error", "message": "數據量不足以進行 Walk Forward 分析，請增加歷史數據"}

    windows = []
    all_oos_pnls = []
    window_params = []

    for i in range(n_windows):
        start = i * (window_size + embargo)
        end = min(start + window_size, total_bars)
        if end - start < 40:
            break

        split_idx = int((end - start) * train_ratio) + start
        train_df = df.iloc[start:split_idx].copy().reset_index(drop=True)
        test_df = df.iloc[split_idx:end].copy().reset_index(drop=True)

        if len(train_df) < 20 or len(test_df) < 10:
            continue

        # ── Per-window SL/TP 優化 ──
        if optimize_sl_tp:
            best_sl, best_tp, best_sharpe = _optimize_on_train(
                train_df, entry_conditions, exit_conditions,
                direction, leverage, stop_loss_pct, take_profit_pct,
            )
        else:
            best_sl = stop_loss_pct
            best_tp = take_profit_pct
            best_sharpe = None

        # ── 訓練集回測（用優化後的參數）──
        train_result = run_backtest(
            train_df, entry_conditions, exit_conditions,
            direction=direction, stop_loss_pct=best_sl,
            take_profit_pct=best_tp, initial_capital=initial_capital,
            leverage=leverage,
        )

        # ── 測試集回測（用訓練集優化的參數）──
        test_result = run_backtest(
            test_df, entry_conditions, exit_conditions,
            direction=direction, stop_loss_pct=best_sl,
            take_profit_pct=best_tp, initial_capital=initial_capital,
            leverage=leverage,
        )

        train_m = train_result.metrics
        test_m = test_result.metrics

        # 收集 OOS 交易 PnL
        oos_pnls = [t.pnl_pct for t in test_result.trades]
        all_oos_pnls.extend(oos_pnls)

        window_info = {
            "window": i + 1,
            "period": f"bar {start}~{end}",
            "train_bars": len(train_df),
            "test_bars": len(test_df),
            "embargo_bars": embargo,
            "optimized_params": {
                "stop_loss_pct": best_sl,
                "take_profit_pct": best_tp,
                "train_sharpe": round(best_sharpe, 3) if best_sharpe is not None else None,
            },
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
            "oos_trades": len(oos_pnls),
        }
        windows.append(window_info)
        window_params.append({"sl": best_sl, "tp": best_tp})

    if len(windows) < 2:
        return {"status": "error", "message": "有效窗口不足 2 個，無法進行 Walk Forward 分析"}

    # ── 整體統計 ──
    test_returns = [w["test"]["return_pct"] for w in windows]
    test_win_rates = [w["test"]["win_rate"] for w in windows]
    test_sharpes = [w["test"]["sharpe"] for w in windows]
    train_returns = [w["train"]["return_pct"] for w in windows]

    profitable_windows = sum(1 for r in test_returns if r > 0)
    consistency_ratio = profitable_windows / len(windows)

    # ── 修正的 decay ratio ──
    decay_ratios = []
    for w in windows:
        train_ret = w["train"]["return_pct"]
        test_ret = w["test"]["return_pct"]
        if train_ret > 0:
            decay_ratios.append(test_ret / train_ret)
        else:
            # 訓練虧、測試也虧 = 0；訓練虧、測試賺 = 異常(-1)
            decay_ratios.append(0.0 if test_ret <= 0 else -1.0)

    avg_decay = float(np.mean(decay_ratios)) if decay_ratios else 0

    # ── 參數穩定性（CV） ──
    param_stability = _compute_param_stability(window_params)

    # ── OOS 交易數警告 ──
    oos_per_window = [w["oos_trades"] for w in windows]
    low_trade_windows = sum(1 for t in oos_per_window if t < 5)

    summary = {
        "test_avg_return": round(float(np.mean(test_returns)), 2),
        "test_median_return": round(float(np.median(test_returns)), 2),
        "test_std_return": round(float(np.std(test_returns)), 2),
        "test_avg_win_rate": round(float(np.mean(test_win_rates)), 2),
        "test_avg_sharpe": round(float(np.mean(test_sharpes)), 3),
        "train_avg_return": round(float(np.mean(train_returns)), 2),
        "consistency_ratio": round(consistency_ratio * 100, 1),
        "performance_decay": round(avg_decay, 3),
        "param_stability": param_stability,
        "oos_trades_total": len(all_oos_pnls),
        "oos_trades_per_window": oos_per_window,
        "low_trade_windows": low_trade_windows,
    }

    return {
        "status": "success",
        "n_windows": len(windows),
        "windows": windows,
        "summary": summary,
        "assessment": _assess_walk_forward(
            consistency_ratio, avg_decay, test_returns, test_sharpes, param_stability,
        ),
        "oos_trade_pnls": all_oos_pnls,
    }


# ═══════════════════════════════════════════════════════
#  Per-window SL/TP 優化
# ═══════════════════════════════════════════════════════

def _optimize_on_train(
    train_df: pd.DataFrame,
    entry_conditions: list[dict],
    exit_conditions: list[dict],
    direction: str,
    leverage: float,
    default_sl: Optional[float],
    default_tp: Optional[float],
) -> tuple[Optional[float], Optional[float], float]:
    """在訓練集上 grid search 最佳 SL/TP，回傳 (best_sl, best_tp, best_sharpe)。"""
    best_sharpe = -999.0
    best_sl = default_sl
    best_tp = default_tp

    # 先測試原始參數作為 baseline
    try:
        baseline = run_backtest(
            train_df, entry_conditions, exit_conditions,
            direction=direction, stop_loss_pct=default_sl,
            take_profit_pct=default_tp, leverage=leverage,
        )
        best_sharpe = baseline.metrics.get("sharpe_ratio", -999)
    except Exception:
        pass

    for sl in SL_RANGE:
        for tp in TP_RANGE:
            if tp <= sl:
                continue  # TP 應大於 SL
            try:
                result = run_backtest(
                    train_df, entry_conditions, exit_conditions,
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

    return best_sl, best_tp, best_sharpe


# ═══════════════════════════════════════════════════════
#  參數穩定性
# ═══════════════════════════════════════════════════════

def _compute_param_stability(window_params: list[dict]) -> dict:
    """計算各窗口最佳參數的變異係數（CV），CV < 0.3 = 穩定。"""
    if len(window_params) < 2:
        return {"sl_cv": None, "tp_cv": None, "stable": False}

    sls = [p["sl"] for p in window_params if p["sl"] is not None]
    tps = [p["tp"] for p in window_params if p["tp"] is not None]

    def _cv(values):
        if not values or len(values) < 2:
            return None
        mean = np.mean(values)
        if mean == 0:
            return None
        return round(float(np.std(values) / mean), 3)

    sl_cv = _cv(sls)
    tp_cv = _cv(tps)

    stable = True
    if sl_cv is not None and sl_cv > 0.3:
        stable = False
    if tp_cv is not None and tp_cv > 0.3:
        stable = False

    return {"sl_cv": sl_cv, "tp_cv": tp_cv, "stable": stable}


# ═══════════════════════════════════════════════════════
#  評估
# ═══════════════════════════════════════════════════════

def _assess_walk_forward(
    consistency: float,
    decay: float,
    test_returns: list[float],
    test_sharpes: list[float],
    param_stability: dict,
) -> dict:
    """評估 Walk Forward 結果。"""
    issues = []
    score = 100

    # 一致性
    if consistency < 0.5:
        issues.append(f"一致性不足：僅 {consistency*100:.0f}% 窗口獲利，策略穩定性差")
        score -= 30
    elif consistency < 0.7:
        issues.append(f"一致性中等：{consistency*100:.0f}% 窗口獲利")
        score -= 15

    # 過擬合偵測（修正後的 decay）
    valid_decays = [d for d in [decay] if d > 0]
    if valid_decays:
        d = valid_decays[0]
        if d < 0.3:
            issues.append(f"嚴重過擬合：測試集績效僅為訓練集的 {d*100:.0f}%")
            score -= 30
        elif d < 0.6:
            issues.append(f"中度過擬合：測試集績效為訓練集的 {d*100:.0f}%")
            score -= 15
    elif decay <= 0:
        issues.append("訓練集虧損或異常，decay 指標不適用")
        score -= 10

    # 報酬波動
    ret_std = float(np.std(test_returns)) if len(test_returns) > 1 else 0
    if ret_std > 20:
        issues.append(f"報酬波動大（std={ret_std:.1f}%），策略在不同時期表現差異大")
        score -= 10

    # Sharpe
    avg_sharpe = float(np.mean(test_sharpes))
    if avg_sharpe < 0:
        issues.append(f"測試集平均 Sharpe {avg_sharpe:.2f} < 0，策略無 Alpha")
        score -= 20
    elif avg_sharpe < 0.5:
        issues.append(f"測試集平均 Sharpe {avg_sharpe:.2f} 偏低")
        score -= 10

    # 參數穩定性
    if not param_stability.get("stable", True):
        issues.append("各窗口最佳參數差異大（CV > 0.3），策略對參數敏感")
        score -= 10

    score = max(0, score)

    if score >= 70:
        verdict = "策略通過 Walk Forward 驗證，具備一定穩定性"
    elif score >= 40:
        verdict = "策略穩定性中等，建議優化或降低倉位"
    else:
        verdict = "策略未通過 Walk Forward 驗證，不建議實盤使用"

    # 提高 has_alpha 門檻
    has_alpha = avg_sharpe > 0.8 and consistency >= 0.7
    if has_alpha and decay > 0:
        has_alpha = decay > 0.4

    return {
        "score": score,
        "verdict": verdict,
        "issues": issues,
        "has_alpha": has_alpha,
    }
