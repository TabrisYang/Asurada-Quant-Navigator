"""阿斯拉量化系統 — 投資組合風險管理

提供跨資產相關性分析、部位集中度檢查、尾部風險警告。
使用現有 OHLCV 數據計算，無額外數據需求。
"""

from dataclasses import dataclass, field

import numpy as np
from loguru import logger


# ── 預設參數 ──────────────────────────────────────────
MAX_SINGLE_ASSET_WEIGHT = 0.30       # 單一標的最大佔比 30%
HIGH_CORRELATION_THRESHOLD = 0.70    # 高相關性警告閾值
TAIL_RISK_PERCENTILE = 5             # VaR 百分位（5% = 95% VaR）
MAX_PORTFOLIO_VAR_DAILY = 0.05       # 單日最大可接受 VaR 5%


@dataclass
class CorrelationResult:
    """相關性矩陣結果"""
    symbols: list[str]
    matrix: list[list[float]]          # N×N 相關性矩陣
    high_corr_pairs: list[dict]        # 高相關性配對警告

    def to_dict(self) -> dict:
        return {
            "symbols": self.symbols,
            "matrix": self.matrix,
            "high_corr_pairs": self.high_corr_pairs,
        }


@dataclass
class RiskReport:
    """投資組合風險報告"""
    total_value: float
    positions: list[dict]
    concentration_warnings: list[str]
    correlation_warnings: list[str]
    tail_risk: dict
    overall_risk_level: str            # "low" / "medium" / "high" / "critical"

    def to_dict(self) -> dict:
        return {
            "total_value": round(self.total_value, 2),
            "positions": self.positions,
            "concentration_warnings": self.concentration_warnings,
            "correlation_warnings": self.correlation_warnings,
            "tail_risk": self.tail_risk,
            "overall_risk_level": self.overall_risk_level,
        }


def compute_correlation_matrix(
    returns_by_symbol: dict[str, np.ndarray],
    min_overlap: int = 30,
) -> CorrelationResult:
    """計算多資產收益率相關性矩陣

    Args:
        returns_by_symbol: {symbol: daily_returns_array}
        min_overlap: 最少重疊天數才計算相關性

    Returns:
        CorrelationResult
    """
    symbols = sorted(returns_by_symbol.keys())
    n = len(symbols)

    if n < 2:
        return CorrelationResult(
            symbols=symbols,
            matrix=[[1.0]] if n == 1 else [],
            high_corr_pairs=[],
        )

    matrix = [[0.0] * n for _ in range(n)]
    high_corr_pairs = []

    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            ri = returns_by_symbol[symbols[i]]
            rj = returns_by_symbol[symbols[j]]

            # 取重疊部分（從尾端對齊）
            overlap = min(len(ri), len(rj))
            if overlap < min_overlap:
                matrix[i][j] = 0.0
                matrix[j][i] = 0.0
                continue

            a = ri[-overlap:]
            b = rj[-overlap:]

            # 過濾 NaN
            valid = ~(np.isnan(a) | np.isnan(b))
            if valid.sum() < min_overlap:
                matrix[i][j] = 0.0
                matrix[j][i] = 0.0
                continue

            corr = float(np.corrcoef(a[valid], b[valid])[0, 1])
            if np.isnan(corr):
                corr = 0.0

            matrix[i][j] = round(corr, 4)
            matrix[j][i] = round(corr, 4)

            if abs(corr) >= HIGH_CORRELATION_THRESHOLD:
                high_corr_pairs.append({
                    "pair": [symbols[i], symbols[j]],
                    "correlation": round(corr, 4),
                    "warning": (
                        f"{symbols[i]} 與 {symbols[j]} 相關性 {corr:.2f}，"
                        "分散風險效果有限"
                    ),
                })

    return CorrelationResult(
        symbols=symbols,
        matrix=matrix,
        high_corr_pairs=high_corr_pairs,
    )


def check_concentration(
    positions: dict[str, float],
    max_weight: float = MAX_SINGLE_ASSET_WEIGHT,
) -> list[str]:
    """檢查部位集中度

    Args:
        positions: {symbol: position_value_usd}
        max_weight: 單一標的最大佔比

    Returns:
        警告訊息列表
    """
    total = sum(abs(v) for v in positions.values())
    if total == 0:
        return []

    warnings = []
    for symbol, value in positions.items():
        weight = abs(value) / total
        if weight > max_weight:
            warnings.append(
                f"{symbol} 佔比 {weight:.1%} 超過上限 {max_weight:.1%}，"
                f"建議降低至 {max_weight:.1%} 以下"
            )

    return warnings


def compute_tail_risk(
    portfolio_returns: np.ndarray,
    percentile: int = TAIL_RISK_PERCENTILE,
) -> dict:
    """計算尾部風險指標

    Args:
        portfolio_returns: 投資組合日收益率序列
        percentile: VaR 百分位數

    Returns:
        dict with VaR, CVaR, max_drawdown
    """
    if len(portfolio_returns) < 10:
        return {
            "var": 0.0,
            "cvar": 0.0,
            "max_drawdown": 0.0,
            "warning": "數據不足，無法準確計算尾部風險",
        }

    clean = portfolio_returns[~np.isnan(portfolio_returns)]
    if len(clean) < 10:
        return {"var": 0.0, "cvar": 0.0, "max_drawdown": 0.0, "warning": "有效數據不足"}

    # VaR（歷史法）
    var = float(np.percentile(clean, percentile))

    # CVaR（條件 VaR = 超過 VaR 的平均損失）
    tail = clean[clean <= var]
    cvar = float(np.mean(tail)) if len(tail) > 0 else var

    # 最大回撤
    cumulative = np.cumprod(1 + clean)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = float(np.min(drawdowns))

    result = {
        "var": round(var, 4),
        "cvar": round(cvar, 4),
        "max_drawdown": round(max_drawdown, 4),
    }

    # 尾部風險警告
    if abs(var) > MAX_PORTFOLIO_VAR_DAILY:
        result["warning"] = (
            f"日 VaR({percentile}%) = {var:.2%}，超過 {MAX_PORTFOLIO_VAR_DAILY:.0%} 警戒線，"
            "建議降低槓桿或減少部位"
        )

    return result


def generate_risk_report(
    positions: dict[str, float],
    returns_by_symbol: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
) -> RiskReport:
    """生成完整風險報告

    Args:
        positions: {symbol: position_value_usd}
        returns_by_symbol: {symbol: daily_returns_array}
        weights: 可選的自訂部位權重

    Returns:
        RiskReport
    """
    total_value = sum(abs(v) for v in positions.values())

    # 部位明細
    position_details = []
    for symbol, value in positions.items():
        weight = abs(value) / total_value if total_value > 0 else 0
        position_details.append({
            "symbol": symbol,
            "value": round(value, 2),
            "weight": round(weight, 4),
            "direction": "long" if value > 0 else "short",
        })

    # 集中度檢查
    concentration_warnings = check_concentration(positions)

    # 相關性分析
    corr_result = compute_correlation_matrix(returns_by_symbol)
    correlation_warnings = [p["warning"] for p in corr_result.high_corr_pairs]

    # 組合收益率（等權或自訂權重）
    if total_value > 0 and returns_by_symbol:
        common_len = min(len(r) for r in returns_by_symbol.values())
        if common_len > 0:
            portfolio_returns = np.zeros(common_len)
            for symbol, rets in returns_by_symbol.items():
                w = abs(positions.get(symbol, 0)) / total_value if total_value > 0 else 0
                portfolio_returns += rets[-common_len:] * w
        else:
            portfolio_returns = np.array([])
    else:
        portfolio_returns = np.array([])

    # 尾部風險
    tail_risk = compute_tail_risk(portfolio_returns)

    # 綜合風險等級
    risk_score = 0
    if concentration_warnings:
        risk_score += len(concentration_warnings)
    if correlation_warnings:
        risk_score += len(correlation_warnings)
    if "warning" in tail_risk:
        risk_score += 2

    if risk_score >= 4:
        overall = "critical"
    elif risk_score >= 2:
        overall = "high"
    elif risk_score >= 1:
        overall = "medium"
    else:
        overall = "low"

    return RiskReport(
        total_value=total_value,
        positions=position_details,
        concentration_warnings=concentration_warnings,
        correlation_warnings=correlation_warnings,
        tail_risk=tail_risk,
        overall_risk_level=overall,
    )
