"""阿斯拉量化系統 — Monte Carlo 模擬（優化版）

使用 block bootstrap 重新模擬 N 次，產生報酬率和回撤的分佈，
用於檢測策略穩定性和評估真實風險。

優化項目：
- 自動最佳 block size（基於 Politis & White 2004 簡化版）
- 交易序列自相關偵測
- 信心度分級（依樣本量）
- Regime-aware bootstrap（可選）
- 壓力測試（尾部風險放大）
- 可配置破產門檻 + 回撤機率分佈
"""

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# 輔助函式
# ---------------------------------------------------------------------------

def _confidence_level(n: int) -> str:
    """依交易筆數回傳信心度等級。"""
    if n >= 100:
        return "high"
    if n >= 30:
        return "medium"
    return "low"


def _lag1_autocorrelation(pnls: np.ndarray) -> tuple[float, bool]:
    """計算 lag-1 自相關係數及其是否統計顯著。"""
    n = len(pnls)
    if n < 10:
        return 0.0, False
    mean = pnls.mean()
    denom = np.sum((pnls - mean) ** 2)
    if denom == 0:
        return 0.0, False
    numer = np.sum((pnls[1:] - mean) * (pnls[:-1] - mean))
    r = float(numer / denom)
    threshold = 2.0 / np.sqrt(n)
    return r, abs(r) > threshold


def _optimal_block_size(n_trades: int, lag1_r: float) -> int:
    """基於 Politis & White (2004) 簡化版估算最佳 block size。"""
    raw = int(np.ceil(n_trades ** (1 / 3) * (1 + 2 * abs(lag1_r))))
    return max(1, min(raw, n_trades // 3))


# ---------------------------------------------------------------------------
# 核心模擬引擎（向量化）
# ---------------------------------------------------------------------------

def _simulate(
    pnls: np.ndarray,
    initial_capital: float,
    n_simulations: int,
    block_size: int,
    ruin_threshold: float,
    regime_labels: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """核心模擬迴圈。回傳 (final_returns%, max_drawdowns%, ruin_count)。"""
    n_trades = len(pnls)
    n_blocks = int(np.ceil(n_trades / block_size))

    final_returns = np.empty(n_simulations)
    max_drawdowns = np.empty(n_simulations)
    ruin_count = 0

    for i in range(n_simulations):
        shuffled = _resample(pnls, n_trades, block_size, n_blocks, regime_labels)

        # 向量化 equity curve
        growth = 1 + shuffled
        equity = initial_capital * np.cumprod(growth)
        equity = np.clip(equity, 0.0, None)
        equity = np.concatenate([[initial_capital], equity])

        # 遇到歸零後全部歸零
        zero_mask = equity <= 0
        if zero_mask.any():
            first_zero = np.argmax(zero_mask)
            equity[first_zero:] = 0.0

        final_returns[i] = (equity[-1] / initial_capital - 1) * 100

        peak = np.maximum.accumulate(equity)
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = np.where(peak > 0, (equity - peak) / peak, 0.0)
        max_drawdowns[i] = float(np.nanmin(dd)) * 100

        if equity[-1] < initial_capital * ruin_threshold:
            ruin_count += 1

    return final_returns, max_drawdowns, ruin_count


def _resample(
    pnls: np.ndarray,
    n_trades: int,
    block_size: int,
    n_blocks: int,
    regime_labels: Optional[np.ndarray],
) -> np.ndarray:
    """Block bootstrap 取樣，支援 regime-aware 模式。"""
    if regime_labels is not None:
        return _resample_regime_aware(pnls, n_trades, block_size, regime_labels)

    block_starts = np.random.randint(0, n_trades - block_size + 1, size=n_blocks)
    blocks = [pnls[s : s + block_size] for s in block_starts]
    return np.concatenate(blocks)[:n_trades]


def _resample_regime_aware(
    pnls: np.ndarray,
    n_trades: int,
    block_size: int,
    regime_labels: np.ndarray,
) -> np.ndarray:
    """各 regime 內分別 block bootstrap，按原始比例拼接。"""
    unique_regimes = np.unique(regime_labels)
    parts: list[np.ndarray] = []

    for regime in unique_regimes:
        mask = regime_labels == regime
        regime_pnls = pnls[mask]
        n_regime = len(regime_pnls)
        target_count = int(np.round(n_regime / len(pnls) * n_trades))
        if target_count == 0 or n_regime == 0:
            continue

        bs = min(block_size, max(1, n_regime // 2))
        n_blk = int(np.ceil(target_count / bs))
        starts = np.random.randint(0, max(1, n_regime - bs + 1), size=n_blk)
        blocks = [regime_pnls[s : s + bs] for s in starts]
        concatenated = np.concatenate(blocks)[:target_count]
        parts.append(concatenated)

    if not parts:
        # fallback: 標準 block bootstrap
        n_blocks = int(np.ceil(n_trades / block_size))
        block_starts = np.random.randint(0, n_trades - block_size + 1, size=n_blocks)
        blocks = [pnls[s : s + block_size] for s in block_starts]
        return np.concatenate(blocks)[:n_trades]

    result = np.concatenate(parts)
    np.random.shuffle(result)
    # 確保長度正確
    if len(result) >= n_trades:
        return result[:n_trades]
    # 不足時補齊
    extra = n_trades - len(result)
    idx = np.random.randint(0, len(pnls), size=extra)
    return np.concatenate([result, pnls[idx]])


# ---------------------------------------------------------------------------
# 壓力測試
# ---------------------------------------------------------------------------

def _run_stress_test(
    pnls: np.ndarray,
    initial_capital: float,
    block_size: int,
    ruin_threshold: float,
    n_stress_sims: int = 200,
    tail_multiplier: float = 1.5,
) -> dict:
    """尾部風險放大壓力測試。"""
    p10 = np.percentile(pnls, 10)
    stressed = pnls.copy()
    stressed[stressed <= p10] *= tail_multiplier

    final_returns, max_drawdowns, ruin_count = _simulate(
        stressed, initial_capital, n_stress_sims, block_size, ruin_threshold,
    )

    return {
        "n_simulations": n_stress_sims,
        "tail_multiplier": tail_multiplier,
        "mean_return": round(float(np.mean(final_returns)), 2),
        "p5_return": round(float(np.percentile(final_returns, 5)), 2),
        "mean_max_drawdown": round(float(np.mean(max_drawdowns)), 2),
        "ruin_probability": round(ruin_count / n_stress_sims * 100, 2),
    }


# ---------------------------------------------------------------------------
# 主函式
# ---------------------------------------------------------------------------

def run_monte_carlo(
    trade_pnl_pcts: list[float],
    initial_capital: float = 10000.0,
    n_simulations: int = 1000,
    confidence_levels: Optional[list[float]] = None,
    block_size: Optional[int] = None,
    ruin_threshold: float = 0.5,
    regime_labels: Optional[list[int]] = None,
) -> dict:
    """執行 Monte Carlo 模擬（Block Bootstrap + 壓力測試）。

    Args:
        trade_pnl_pcts: 每筆交易的盈虧百分比列表（如 [0.05, -0.02, 0.08, ...]）
        initial_capital: 初始資金
        n_simulations: 模擬次數
        confidence_levels: 信賴區間（預設 [0.05, 0.25, 0.50, 0.75, 0.95]）
        block_size: block bootstrap 區塊大小（None = 自動估算）
        ruin_threshold: 破產門檻（佔初始資金比例，預設 0.5 = 資金剩 50%）
        regime_labels: 每筆交易的 regime 標籤（如 [0, 1, 0, ...]），None 則不分 regime

    Returns:
        dict: 包含分佈統計、信賴區間、破產機率、壓力測試等
    """
    if not trade_pnl_pcts or len(trade_pnl_pcts) < 10:
        return {
            "status": "insufficient_data",
            "message": "至少需要 10 筆交易才能執行 Monte Carlo 模擬",
        }

    if confidence_levels is None:
        confidence_levels = [0.05, 0.25, 0.50, 0.75, 0.95]

    pnls = np.array(trade_pnl_pcts)
    n_trades = len(pnls)
    conf = _confidence_level(n_trades)

    # 自相關偵測
    lag1_r, lag1_sig = _lag1_autocorrelation(pnls)

    # 自動或手動 block size
    if block_size is None:
        actual_block_size = _optimal_block_size(n_trades, lag1_r)
    else:
        actual_block_size = min(block_size, max(1, n_trades // 3))

    # Regime labels
    regime_arr = np.array(regime_labels) if regime_labels is not None else None

    # 主模擬
    final_returns, max_drawdowns, ruin_count = _simulate(
        pnls, initial_capital, n_simulations, actual_block_size,
        ruin_threshold, regime_arr,
    )

    # 統計
    return_percentiles = {}
    dd_percentiles = {}
    for cl in confidence_levels:
        pct = int(cl * 100)
        return_percentiles[f"p{pct}"] = round(float(np.percentile(final_returns, pct)), 2)
        dd_percentiles[f"p{pct}"] = round(float(np.percentile(max_drawdowns, pct)), 2)

    # 回撤機率
    drawdown_probabilities = {
        "exceed_20pct": round(float(np.mean(max_drawdowns < -20)) * 100, 2),
        "exceed_30pct": round(float(np.mean(max_drawdowns < -30)) * 100, 2),
        "exceed_50pct": round(float(np.mean(max_drawdowns < -50)) * 100, 2),
    }

    # 壓力測試
    stress_test = _run_stress_test(
        pnls, initial_capital, actual_block_size, ruin_threshold,
    )

    return {
        "status": "success",
        "n_simulations": n_simulations,
        "n_trades": n_trades,
        "confidence_level": conf,
        "block_size_used": actual_block_size,
        "autocorrelation": {
            "lag1": round(lag1_r, 4),
            "significant": lag1_sig,
        },
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
        "drawdown_probabilities": drawdown_probabilities,
        "ruin_probability": round(ruin_count / n_simulations * 100, 2),
        "ruin_threshold": ruin_threshold,
        "profit_probability": round(float(np.mean(final_returns > 0)) * 100, 2),
        "strategy_robust": bool(np.percentile(final_returns, 25) > 0),
        "stress_test": stress_test,
        "interpretation": _interpret_results(
            float(np.median(final_returns)),
            float(np.percentile(final_returns, 5)),
            float(np.mean(max_drawdowns)),
            ruin_count / n_simulations,
            conf,
        ),
    }


def _interpret_results(
    median_return: float,
    p5_return: float,
    avg_dd: float,
    ruin_pct: float,
    confidence: str,
) -> str:
    """自動產生結果解讀。"""
    parts = []

    if confidence == "low":
        parts.append("⚠️ 交易筆數不足 30 筆，模擬結論信心度偏低")

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
