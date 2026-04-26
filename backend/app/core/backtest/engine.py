"""阿斯拉量化系統 — 回測引擎

NumPy 向量化回測核心，支援：
- 多條件進出場（AND/OR）
- 止損 / 止盈
- 動態成本模型（基於 ATR/成交量的滑點 + 交易所 maker/taker 費率）
- 自動 in-sample / out-of-sample 分割
- 完整績效統計（勝率、盈虧比、Sharpe、MDD 等）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger

from app.core.indicators import registry

# ─── 交易成本參數 ────────────────────────────────────
DEFAULT_SLIPPAGE_PCT = 0.0005   # 0.05% 保守滑點（靜態回退值）
DEFAULT_FEE_PCT = 0.001         # 0.1% 手續費（單邊，靜態回退值）
DEFAULT_FUNDING_RATE = 0.0001   # 0.01% 每 8 小時（永續合約資金費率）
DEFAULT_CAPITAL = 10000.0       # 預設初始資金 (USDT)
MIN_DATA_POINTS = 60            # 最低數據量需求
OOS_RATIO = 0.3                 # out-of-sample 佔比

# 各時間框架對應的 bar 數 per 8 小時（資金費率結算週期）
_BARS_PER_FUNDING_PERIOD: dict[str, float] = {
    "15m": 32.0,   # 8h / 15m
    "1h": 8.0,
    "4h": 2.0,
    "1d": 1 / 3,   # 8h / 24h
    "1w": 1 / 21,  # 8h / 168h
}

# 各交易所 maker/taker 費率（單邊）
EXCHANGE_FEES = {
    "binance":  {"maker": 0.0002, "taker": 0.0004},
    "bybit":    {"maker": 0.0002, "taker": 0.00055},
    "okx":      {"maker": 0.0002, "taker": 0.0005},
    "coinbase": {"maker": 0.004,  "taker": 0.006},
    "default":  {"maker": 0.0005, "taker": 0.001},
}

# 動態滑點參數
SLIPPAGE_ATR_SCALE = 0.1        # ATR 比例因子：slippage = ATR/close * scale
VOLUME_PERCENTILE_THRESHOLD = 0.2  # 低於 20 分位即視為低量

# 流動性分級滑點乘數（取代固定 2.0x）
_LIQUIDITY_SLIPPAGE_MULT = {
    "high": 1.0,    # BTC/ETH 等主流幣，日均交易額 > $1B
    "medium": 1.3,  # SOL/ADA 等中型幣，$100M-$1B
    "low": 1.5,     # 小幣 < $100M
}


def _estimate_liquidity_tier(close: np.ndarray, volume: np.ndarray) -> str:
    """根據成交量 × 價格估算流動性等級。"""
    if len(close) < 2 or len(volume) < 2:
        return "low"
    median_vol = float(np.nanmedian(volume))
    last_close = float(close[-1])
    daily_value = median_vol * last_close
    if daily_value > 1e9:
        return "high"
    elif daily_value > 1e8:
        return "medium"
    return "low"


def _compute_dynamic_cost(
    atr: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    bar_idx: int,
    exchange: str = "default",
    order_type: str = "taker",
    liquidity_tier: str = "medium",
) -> float:
    """計算單根 K 棒的動態交易成本（滑點 + 手續費，單邊）

    - 滑點基於 ATR/close 比率，低量時按流動性等級加成
    - 手續費依交易所 maker/taker 費率
    """
    # 手續費
    fees = EXCHANGE_FEES.get(exchange, EXCHANGE_FEES["default"])
    fee = fees.get(order_type, fees["taker"])

    # 動態滑點
    if np.isnan(atr[bar_idx]) or close[bar_idx] <= 0:
        slippage = DEFAULT_SLIPPAGE_PCT
    else:
        slippage = (atr[bar_idx] / close[bar_idx]) * SLIPPAGE_ATR_SCALE

        # 低成交量按流動性等級加成
        vol_mult = _LIQUIDITY_SLIPPAGE_MULT.get(liquidity_tier, 1.3)
        if bar_idx >= 20:
            vol_window = volume[max(0, bar_idx - 20):bar_idx]
            vol_pctile = np.percentile(vol_window, VOLUME_PERCENTILE_THRESHOLD * 100)
            if volume[bar_idx] < vol_pctile:
                slippage *= vol_mult

        # 設定地板和天花板
        slippage = max(slippage, 0.0001)  # 最低 0.01%
        slippage = min(slippage, 0.005)   # 最高 0.5%（從 1% 降低）

    return slippage + fee


@dataclass
class Trade:
    entry_idx: int
    exit_idx: int
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    direction: str          # "long" or "short"
    pnl_pct: float          # 含成本的淨收益%
    pnl_amount: float       # 金額
    bars_held: int
    exit_reason: str        # "signal" | "stop_loss" | "take_profit" | "end_of_data"


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    equity_curve: list[dict] = field(default_factory=list)
    trade_annotations: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    oos_metrics: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict:
        return {
            "status": "success",
            "total_trades": len(self.trades),
            "metrics": self.metrics,
            "oos_metrics": self.oos_metrics,
            "equity_curve_length": len(self.equity_curve),
            "trade_annotations": self.trade_annotations[:50],
            "warnings": self.warnings,
            "trades_summary": [
                {
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "direction": t.direction,
                    "entry_price": round(t.entry_price, 2),
                    "exit_price": round(t.exit_price, 2),
                    "pnl_pct": round(t.pnl_pct, 4),
                    "bars_held": t.bars_held,
                    "exit_reason": t.exit_reason,
                }
                for t in self.trades[:100]
            ],
        }


def _build_condition_mask(
    df: pd.DataFrame, conditions: list[dict], logic: str = "AND",
    indicator_cache: dict | None = None,
) -> np.ndarray:
    """將多個指標條件轉為 bool 遮罩向量（支援快取避免重複計算）"""
    masks: list[np.ndarray] = []
    cache = indicator_cache if indicator_cache is not None else {}

    for cond in conditions:
        indicator_id = cond.get("indicator", "").lower()
        params = cond.get("parameters")
        # 用 indicator_id + params 組成快取鍵
        cache_key = (indicator_id, str(sorted(params.items())) if params else "")
        if cache_key in cache:
            calc = cache[cache_key]
        else:
            calc = registry.calculate(indicator_id, df, params)
            cache[cache_key] = calc
        if not calc:
            logger.warning(f"回測：無法計算指標 {indicator_id}")
            continue

        # 支援 condition 中指定 series（如 ADX 的 "+DI"/"-DI"）
        series_name = cond.get("series") or list(calc.keys())[0]
        if series_name not in calc:
            logger.warning(f"回測：指標 {indicator_id} 沒有 series '{series_name}'，可用：{list(calc.keys())}")
            continue
        values = np.array(calc[series_name], dtype=float)

        op = cond.get("operator", ">")
        try:
            val = float(cond.get("value", 0))
        except (ValueError, TypeError):
            logger.warning(f"回測：condition value '{cond.get('value')}' 無法轉為數字")
            continue

        if op == ">":
            mask = values > val
        elif op == "<":
            mask = values < val
        elif op == ">=":
            mask = values >= val
        elif op == "<=":
            mask = values <= val
        elif op == "==":
            mask = np.isclose(values, val, atol=1e-6)
        elif op == "cross_above":
            prev = np.roll(values, 1)
            prev[0] = np.nan
            mask = (values > val) & (prev <= val)
        elif op == "cross_below":
            prev = np.roll(values, 1)
            prev[0] = np.nan
            mask = (values < val) & (prev >= val)
        elif op == "between":
            val2 = float(cond.get("value2", val))
            mask = (values >= val) & (values <= val2)
        else:
            continue

        mask = np.where(np.isnan(values), False, mask)
        masks.append(mask)

    if not masks:
        return np.zeros(len(df), dtype=bool)

    combined = masks[0]
    for m in masks[1:]:
        combined = combined & m if logic == "AND" else combined | m
    return combined


def _compute_metrics(trades: list[Trade], equity: np.ndarray, n_bars: int) -> dict:
    """從交易列表和資金曲線計算績效統計"""
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_pnl_pct": 0.0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "expectancy_pct": 0.0,
            "expectancy_ratio": 0.0,
            "avg_bars_held": 0,
            "note": "無交易產生，請檢查進場條件是否過於嚴格",
        }

    wins = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct <= 0]
    win_rate = len(wins) / len(trades)

    gross_profit = sum(t.pnl_amount for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl_amount for t in losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    pnl_pcts = np.array([t.pnl_pct for t in trades])
    avg_win = float(np.mean([t.pnl_pct for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([t.pnl_pct for t in losses])) if losses else 0.0

    # 資金曲線統計
    total_return = (equity[-1] / equity[0] - 1) * 100 if equity[0] > 0 else 0.0
    peak = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, (equity - peak) / peak, 0.0)
    max_dd = float(np.nanmin(dd)) * 100

    # Sharpe Ratio（年化）
    if len(equity) > 1:
        denom = equity[:-1].copy()
        denom[denom == 0] = np.nan
        returns = np.diff(equity) / denom
        mean_r = np.mean(returns)
        std_r = np.std(returns)
        sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0
    else:
        returns = np.array([])
        sharpe = 0.0

    # Sortino Ratio（只懲罰下行波動）
    if len(returns) > 1:
        downside = returns[returns < 0]
        downside_std = np.std(downside) if len(downside) > 0 else 0.0
        sortino = (mean_r / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0
    else:
        sortino = 0.0

    # Expectancy（期望值 = 勝率 × 平均獲利 - 敗率 × 平均虧損）
    expectancy = win_rate * avg_win - (1 - win_rate) * abs(avg_loss)
    # Expectancy per dollar risked
    avg_risk = abs(avg_loss) if avg_loss != 0 else 1.0
    expectancy_ratio = expectancy / avg_risk if avg_risk > 0 else 0.0

    avg_bars = int(np.mean([t.bars_held for t in trades]))

    # ★ Phase 2.1：Wilson 信心區間（小樣本下勝率的真實信心）
    n = len(trades)
    n_wins = len(wins)
    z = 1.96  # 95% confidence
    p_hat = win_rate
    if n > 0:
        z2_n = z * z / n
        denom_w = 1 + z2_n
        center = (p_hat + z2_n / 2) / denom_w
        margin = z * np.sqrt((p_hat * (1 - p_hat) + z2_n / 4) / n) / denom_w
        win_rate_ci_low = max(0.0, center - margin) * 100
        win_rate_ci_high = min(1.0, center + margin) * 100
    else:
        win_rate_ci_low = win_rate_ci_high = 0.0

    # ★ Phase 2.1：連續虧損次數（最大連虧）
    max_consecutive_losses = 0
    current_streak = 0
    for t in trades:
        if t.pnl_pct <= 0:
            current_streak += 1
            max_consecutive_losses = max(max_consecutive_losses, current_streak)
        else:
            current_streak = 0

    # ★ Phase 2.1：MDD 持續期間（從 peak 跌到 trough 共幾根 K 線）
    mdd_duration_bars = 0
    if len(equity) > 1:
        trough_idx = int(np.nanargmin(dd))
        # 找 trough 之前最後的 peak
        peak_idx = int(np.nanargmax(equity[:trough_idx + 1])) if trough_idx > 0 else 0
        mdd_duration_bars = trough_idx - peak_idx

    # ★ Phase 2.1：CVAR 5%（最差 5% 交易的平均報酬）
    cvar_5 = 0.0
    if len(pnl_pcts) >= 20:
        sorted_pnl = np.sort(pnl_pcts)
        worst_5pct_count = max(1, len(sorted_pnl) // 20)
        cvar_5 = float(np.mean(sorted_pnl[:worst_5pct_count])) * 100

    # ★ Phase 2.1：Calmar Ratio（年化報酬 / |MDD|）
    if equity[0] > 0 and n_bars > 0 and abs(max_dd) > 0.01:
        years = n_bars / 252  # 假設日線
        annualized_return = ((equity[-1] / equity[0]) ** (1 / years) - 1) * 100 if years > 0 else 0
        calmar = annualized_return / abs(max_dd)
    else:
        calmar = 0.0

    return {
        "total_trades": len(trades),
        "win_rate": round(float(win_rate * 100), 2),
        "win_rate_ci_95_low": round(float(win_rate_ci_low), 2),
        "win_rate_ci_95_high": round(float(win_rate_ci_high), 2),
        "profit_factor": round(float(min(profit_factor, 999.99)), 2),
        "avg_pnl_pct": round(float(np.mean(pnl_pcts)) * 100, 2),
        "avg_win_pct": round(float(avg_win * 100), 2),
        "avg_loss_pct": round(float(avg_loss * 100), 2),
        "total_return_pct": round(float(total_return), 2),
        "max_drawdown_pct": round(float(max_dd), 2),
        "max_drawdown_duration_bars": mdd_duration_bars,
        "max_consecutive_losses": max_consecutive_losses,
        "sharpe_ratio": round(float(sharpe), 3),
        "sortino_ratio": round(float(sortino), 3),
        "calmar_ratio": round(float(calmar), 3),
        "cvar_5_pct": round(float(cvar_5), 2),
        "expectancy_pct": round(float(expectancy * 100), 3),
        "expectancy_ratio": round(float(expectancy_ratio), 3),
        "avg_bars_held": int(avg_bars),
        "total_bars": int(n_bars),
    }


def _generate_warnings(
    metrics: dict, oos_metrics: Optional[dict], trades: list[Trade]
) -> list[str]:
    """自動產生風險警告"""
    warnings: list[str] = []

    if metrics["total_trades"] < 10:
        warnings.append("交易次數偏少（< 10），統計結果可能不具代表性。")

    if metrics["max_drawdown_pct"] < -30:
        warnings.append(f"最大回撤 {metrics['max_drawdown_pct']:.1f}%，風險偏高。實際交易中回撤可能更大。")

    if metrics["sharpe_ratio"] < 0.5 and metrics["total_trades"] > 5:
        warnings.append(f"Sharpe Ratio {metrics['sharpe_ratio']:.2f} 偏低，風險調整後報酬不佳。")

    if metrics.get("expectancy_pct", 0) <= 0 and metrics["total_trades"] > 5:
        warnings.append(f"Expectancy {metrics['expectancy_pct']:.2f}% ≤ 0，此策略長期期望值為負。")

    if oos_metrics and oos_metrics.get("total_trades", 0) > 3:
        is_diff = abs(metrics["win_rate"] - oos_metrics["win_rate"]) > 15
        ret_diff = abs(metrics["total_return_pct"] - oos_metrics["total_return_pct"])
        if is_diff or ret_diff > 20:
            warnings.append(
                f"樣本內/外績效差距大（勝率 {metrics['win_rate']}% vs {oos_metrics['win_rate']}%），"
                f"可能有過度擬合風險。"
            )

    if metrics.get("liquidation_count", 0) > 0:
        warnings.append(
            f"🚨 回測中出現 {metrics['liquidation_count']} 次爆倉（佔 {metrics.get('liquidation_rate', 0)}%）！"
            f"建議降低槓桿或放寬止損。"
        )

    if metrics.get("leverage", 1) > 1:
        warnings.append(
            f"⚠️ 此回測使用 {metrics['leverage']}x 槓桿。盈虧均按槓桿倍數放大，不含資金費率。"
        )

    warnings.append("⚠️ 回測結果僅供參考，已含動態滑點與手續費估算，但不含資金費率、極端行情流動性衝擊。實際交易請謹慎。")
    return warnings


def run_backtest(
    df: pd.DataFrame,
    entry_conditions: list[dict],
    exit_conditions: list[dict],
    direction: str = "long",
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    initial_capital: float = DEFAULT_CAPITAL,
    slippage_pct: float = DEFAULT_SLIPPAGE_PCT,
    fee_pct: float = DEFAULT_FEE_PCT,
    entry_logic: str = "AND",
    exit_logic: str = "AND",
    leverage: float = 1.0,
    dynamic_cost: bool = True,
    exchange: str = "default",
    timeframe: str = "4h",
    funding_rate: float = DEFAULT_FUNDING_RATE,
) -> BacktestResult:
    """
    執行向量化回測

    Args:
        df: 含 timestamp, open, high, low, close, volume 的 DataFrame
        entry_conditions: 進場條件列表
        exit_conditions: 出場條件列表
        direction: "long" | "short"
        stop_loss_pct: 止損百分比（如 0.02 = 2%）
        dynamic_cost: 啟用動態成本模型（基於 ATR/成交量），預設 True
        exchange: 交易所名稱，用於查詢 maker/taker 費率
        take_profit_pct: 止盈百分比
        initial_capital: 初始資金
        leverage: 槓桿倍數（預設 1.0 = 無槓桿）
        timeframe: K 線時間框架，用於計算資金費率週期
        funding_rate: 永續合約資金費率（每 8 小時），預設 0.01%
    """
    result = BacktestResult()

    if len(df) < MIN_DATA_POINTS:
        result.warnings.append(f"數據不足（{len(df)} 根 K 線），至少需要 {MIN_DATA_POINTS} 根。")
        result.metrics = {"total_trades": 0, "note": "數據不足"}
        return result

    # 準備價格數組
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    timestamps = df["timestamp"].astype(str).values

    # 建立進出場信號遮罩（共用指標快取，避免重複計算）
    _ind_cache: dict = {}
    entry_mask = _build_condition_mask(df, entry_conditions, entry_logic, _ind_cache)
    exit_mask = _build_condition_mask(df, exit_conditions, exit_logic, _ind_cache)

    if not entry_mask.any():
        result.warnings.append("進場條件在整個數據範圍內均未觸發，請放寬條件。")
        result.metrics = {"total_trades": 0, "note": "進場條件未觸發"}
        return result

    # ─── 逐 bar 模擬 ───────────────────────────────────
    leverage = max(1.0, leverage)
    static_cost_rate = slippage_pct + fee_pct
    capital = initial_capital

    # 資金費率：每 bar 的費率（永續合約槓桿 > 1 時生效）
    bars_per_funding = _BARS_PER_FUNDING_PERIOD.get(timeframe, 2.0)
    funding_per_bar = funding_rate / bars_per_funding if leverage > 1 and bars_per_funding > 0 else 0.0

    # 預計算 ATR 和 volume 用於動態成本
    volume = df["volume"].values.astype(float) if "volume" in df.columns else np.ones(len(close))
    liq_tier = _estimate_liquidity_tier(close, volume)
    if dynamic_cost and "atr" in df.columns:
        atr_arr = df["atr"].values.astype(float)
        use_dynamic = True
    elif dynamic_cost:
        # 現場計算 ATR(14)
        tr = np.maximum(
            high - low,
            np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
        )
        tr[0] = high[0] - low[0]
        atr_arr = pd.Series(tr).rolling(14).mean().values
        use_dynamic = True
    else:
        atr_arr = np.full(len(close), np.nan)
        use_dynamic = False

    def _cost_at(bar_idx: int) -> float:
        if use_dynamic:
            return _compute_dynamic_cost(atr_arr, close, volume, bar_idx, exchange, liquidity_tier=liq_tier)
        return static_cost_rate
    equity_arr = [capital]
    trades: list[Trade] = []
    in_position = False
    entry_idx = 0
    entry_price = 0.0
    entry_capital = 0.0

    # 強制平倉價（槓桿 > 1 時生效）：虧損達 ~95% 保證金即爆倉
    liq_margin = 0.98 / leverage if leverage > 1 else None

    open_prices = df["open"].values.astype(float) if "open" in df.columns else close

    for i in range(1, len(close)):
        cost_rate = _cost_at(i)
        if not in_position:
            if entry_mask[i]:
                raw_entry = open_prices[i] if i < len(open_prices) else close[i]
                entry_price = raw_entry * (1 + cost_rate if direction == "long" else 1 - cost_rate)
                entry_idx = i
                entry_capital = capital
                in_position = True
        else:
            # 計算持倉期間累計資金費率成本
            def _funding_cost(bars_held: int) -> float:
                return funding_per_bar * bars_held

            # 檢查強制平倉（槓桿爆倉）
            if liq_margin is not None:
                if direction == "long" and low[i] <= entry_price * (1 - liq_margin):
                    exit_price = entry_price * (1 - liq_margin)
                    trades.append(_make_trade(
                        entry_idx, i, timestamps, entry_price, exit_price,
                        direction, entry_capital, cost_rate, "liquidation", leverage,
                        _funding_cost(i - entry_idx),
                    ))
                    capital += trades[-1].pnl_amount
                    capital = max(capital, 0)
                    in_position = False
                    equity_arr.append(capital)
                    continue
                elif direction == "short" and high[i] >= entry_price * (1 + liq_margin):
                    exit_price = entry_price * (1 + liq_margin)
                    trades.append(_make_trade(
                        entry_idx, i, timestamps, entry_price, exit_price,
                        direction, entry_capital, cost_rate, "liquidation", leverage,
                        _funding_cost(i - entry_idx),
                    ))
                    capital += trades[-1].pnl_amount
                    capital = max(capital, 0)
                    in_position = False
                    equity_arr.append(capital)
                    continue

            # 檢查止損（含跳空處理：以 low/high 作為最壞情況出場價）
            if stop_loss_pct is not None:
                sl_target_long = entry_price * (1 - stop_loss_pct)
                sl_target_short = entry_price * (1 + stop_loss_pct)

                if direction == "long" and low[i] <= sl_target_long:
                    exit_price = min(sl_target_long, open_prices[i]) * (1 - cost_rate)
                    trades.append(_make_trade(
                        entry_idx, i, timestamps, entry_price, exit_price,
                        direction, entry_capital, cost_rate, "stop_loss", leverage,
                        _funding_cost(i - entry_idx),
                    ))
                    capital += trades[-1].pnl_amount
                    in_position = False
                    equity_arr.append(capital)
                    continue
                elif direction == "short" and high[i] >= sl_target_short:
                    exit_price = max(sl_target_short, open_prices[i]) * (1 + cost_rate)
                    trades.append(_make_trade(
                        entry_idx, i, timestamps, entry_price, exit_price,
                        direction, entry_capital, cost_rate, "stop_loss", leverage,
                        _funding_cost(i - entry_idx),
                    ))
                    capital += trades[-1].pnl_amount
                    in_position = False
                    equity_arr.append(capital)
                    continue

            # 檢查止盈（含跳空處理）
            if take_profit_pct is not None:
                tp_target_long = entry_price * (1 + take_profit_pct)
                tp_target_short = entry_price * (1 - take_profit_pct)

                if direction == "long" and high[i] >= tp_target_long:
                    exit_price = max(tp_target_long, open_prices[i]) * (1 - cost_rate)
                    trades.append(_make_trade(
                        entry_idx, i, timestamps, entry_price, exit_price,
                        direction, entry_capital, cost_rate, "take_profit", leverage,
                        _funding_cost(i - entry_idx),
                    ))
                    capital += trades[-1].pnl_amount
                    in_position = False
                    equity_arr.append(capital)
                    continue
                elif direction == "short" and low[i] <= tp_target_short:
                    exit_price = min(tp_target_short, open_prices[i]) * (1 + cost_rate)
                    trades.append(_make_trade(
                        entry_idx, i, timestamps, entry_price, exit_price,
                        direction, entry_capital, cost_rate, "take_profit", leverage,
                        _funding_cost(i - entry_idx),
                    ))
                    capital += trades[-1].pnl_amount
                    in_position = False
                    equity_arr.append(capital)
                    continue

            # 檢查出場信號
            if exit_mask[i]:
                exit_price = close[i] * (1 - cost_rate if direction == "long" else 1 + cost_rate)
                trades.append(_make_trade(
                    entry_idx, i, timestamps, entry_price, exit_price,
                    direction, entry_capital, cost_rate, "signal", leverage,
                    _funding_cost(i - entry_idx),
                ))
                capital += trades[-1].pnl_amount
                in_position = False

        equity_arr.append(capital)

    # 強制平倉
    if in_position:
        exit_price = close[-1] * (1 - cost_rate if direction == "long" else 1 + cost_rate)
        trades.append(_make_trade(
            entry_idx, len(close) - 1, timestamps, entry_price, exit_price,
            direction, entry_capital, cost_rate, "end_of_data", leverage,
            _funding_cost(len(close) - 1 - entry_idx),
        ))
        capital += trades[-1].pnl_amount
        equity_arr.append(capital)

    equity = np.array(equity_arr)

    # ─── 計算績效 ──────────────────────────────────────
    metrics = _compute_metrics(trades, equity, len(close))
    if leverage > 1:
        metrics["leverage"] = leverage
        liq_count = sum(1 for t in trades if t.exit_reason == "liquidation")
        if liq_count > 0:
            metrics["liquidation_count"] = liq_count
            metrics["liquidation_rate"] = round(liq_count / len(trades) * 100, 1)

    # ─── In-Sample / Out-of-Sample 時間制分割 ──────────
    # 數據已按 timestamp 排序，position-based split 即為 time-based split
    oos_metrics = None
    split_idx = int(len(close) * (1 - OOS_RATIO))
    if split_idx > MIN_DATA_POINTS and (len(close) - split_idx) > 30:
        is_trades = [t for t in trades if t.entry_idx < split_idx]
        oos_trades = [t for t in trades if t.entry_idx >= split_idx]
        if is_trades and oos_trades:
            is_eq = equity[: split_idx + 1]
            oos_eq = equity[split_idx:]
            is_metrics = _compute_metrics(is_trades, is_eq, split_idx)
            oos_metrics = _compute_metrics(oos_trades, oos_eq, len(close) - split_idx)
            oos_metrics["label"] = "out_of_sample"
            is_metrics["label"] = "in_sample"
            # 標記分割時間點
            is_metrics["period"] = f"{timestamps[0]} ~ {timestamps[split_idx - 1]}"
            oos_metrics["period"] = f"{timestamps[split_idx]} ~ {timestamps[-1]}"

    # ─── 風險警告 ──────────────────────────────────────
    warnings = _generate_warnings(metrics, oos_metrics, trades)

    # ─── 交易標記（前端圖表 annotations）─────────────────
    annotations: list[dict] = []
    for t in trades[:50]:
        color = "#3fb950" if t.pnl_pct > 0 else "#f85149"
        annotations.append({
            "type": "text_label",
            "time": t.entry_time,
            "startTime": t.entry_time,
            "price": t.entry_price,
            "text": f"{'▲' if t.direction == 'long' else '▼'} 進場",
            "color": "#58a6ff",
        })
        annotations.append({
            "type": "text_label",
            "time": t.exit_time,
            "startTime": t.exit_time,
            "price": t.exit_price,
            "text": f"{'✕' if t.pnl_pct <= 0 else '✓'} {t.pnl_pct * 100:+.1f}%",
            "color": color,
        })

    result.trades = trades
    result.metrics = metrics
    result.equity_curve = [{"bar": i, "equity": round(float(v), 2)} for i, v in enumerate(equity)]
    result.trade_annotations = annotations
    result.warnings = warnings
    result.oos_metrics = oos_metrics

    # ★ Phase 2.2：per-regime 績效拆解（給 LLM 看「策略在哪個 regime 才有效」）
    try:
        from app.core.regime_filter import classify_regime_series
        regime_series = classify_regime_series(df, step=10)  # 每 10 根分類一次節省時間
        per_regime_trades: dict[str, list] = {}
        for t in trades:
            if 0 <= t.entry_idx < len(regime_series):
                rg = regime_series[t.entry_idx].get("regime", "unknown")
                per_regime_trades.setdefault(rg, []).append(t)
        per_regime_metrics: dict[str, dict] = {}
        for rg, ts in per_regime_trades.items():
            wins_ = [t for t in ts if t.pnl_pct > 0]
            wr = len(wins_) / len(ts) * 100 if ts else 0
            avg_ret = float(np.mean([t.pnl_pct for t in ts])) * 100 if ts else 0
            per_regime_metrics[rg] = {
                "n_trades": len(ts),
                "win_rate": round(wr, 1),
                "avg_return_pct": round(avg_ret, 2),
            }
        metrics["per_regime_metrics"] = per_regime_metrics
    except Exception as _rg_err:
        logger.debug(f"per_regime_metrics 計算失敗（不影響主流程）: {_rg_err}")

    logger.info(
        f"回測完成: {len(trades)} 筆交易, 勝率 {metrics['win_rate']}%, "
        f"總報酬 {metrics['total_return_pct']}%, MDD {metrics['max_drawdown_pct']}%"
    )

    return result


def _make_trade(
    entry_idx: int,
    exit_idx: int,
    timestamps: np.ndarray,
    entry_price: float,
    exit_price: float,
    direction: str,
    capital: float,
    cost_rate: float,
    exit_reason: str,
    leverage: float = 1.0,
    funding_cost_pct: float = 0.0,
) -> Trade:
    if direction == "long":
        raw_pnl_pct = (exit_price / entry_price) - 1
    else:
        raw_pnl_pct = 1 - (exit_price / entry_price)

    # 扣除資金費率成本（永續合約持倉期間累計）
    pnl_pct = (raw_pnl_pct - funding_cost_pct) * leverage
    pnl_amount = capital * pnl_pct

    return Trade(
        entry_idx=entry_idx,
        exit_idx=exit_idx,
        entry_time=str(timestamps[entry_idx]),
        exit_time=str(timestamps[exit_idx]),
        entry_price=entry_price,
        exit_price=exit_price,
        direction=direction,
        pnl_pct=pnl_pct,
        pnl_amount=pnl_amount,
        bars_held=exit_idx - entry_idx,
        exit_reason=exit_reason,
    )
