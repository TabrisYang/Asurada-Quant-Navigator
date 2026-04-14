"""阿斯拉量化系統 — 動能交易分析引擎

完整的動量分析模組：
- 多週期動量因子（Price Momentum）
- 動量加速/減速偵測（Momentum Shift）
- 相對動量（vs BTC / 大盤）
- 動量反轉偵測（Momentum Reversal）
- 動量策略回測（追強勢、反轉進場）
"""

import numpy as np
import pandas as pd
from loguru import logger
from typing import Optional

from app.core.indicators import registry
from app.core.backtest.engine import run_backtest


async def analyze_momentum(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    benchmark_df: Optional[pd.DataFrame] = None,
) -> dict:
    """完整動能分析。

    Args:
        df: 標的 OHLCV DataFrame
        symbol: 交易對
        timeframe: 時間框架
        benchmark_df: 基準 OHLCV（BTC 或大盤），用於相對動量

    Returns:
        dict: 動量因子、加速度、相對強弱、反轉偵測、策略回測
    """
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    n = len(close)

    if n < 50:
        return {"status": "error", "message": f"數據不足（{n} 根），至少需要 50 根"}

    logger.info(f"動能分析 [{symbol} {timeframe}]: {n} 根 K 線")

    # ── 1. 多週期動量因子 ──
    momentum_factors = _compute_momentum_factors(close)

    # ── 2. 動量加速/減速 ──
    momentum_shift = _detect_momentum_shift(close, df)

    # ── 3. 相對動量 ──
    relative_momentum = None
    if benchmark_df is not None and len(benchmark_df) >= 50:
        relative_momentum = _compute_relative_momentum(close, benchmark_df["close"].values.astype(float))

    # ── 4. 動量反轉偵測 ──
    reversal = _detect_reversal(close, high, low, df)

    # ── 5. 動量策略回測 ──
    strategies = _backtest_momentum_strategies(df, timeframe)

    # ── 6. 綜合動量評分 ──
    score = _compute_momentum_score(momentum_factors, momentum_shift, reversal)

    return {
        "status": "success",
        "symbol": symbol,
        "timeframe": timeframe,
        "total_bars": n,
        "momentum_factors": momentum_factors,
        "momentum_shift": momentum_shift,
        "relative_momentum": relative_momentum,
        "reversal_detection": reversal,
        "strategy_backtest": strategies,
        "momentum_score": score,
    }


# ═══════════════════════════════════════════════════
#  1. 多週期動量因子
# ═══════════════════════════════════════════════════

def _compute_momentum_factors(close: np.ndarray) -> dict:
    """計算多週期價格動量。"""
    n = len(close)
    factors = {}

    for period, label in [(5, "5根"), (10, "10根"), (20, "20根"), (60, "60根")]:
        if n > period:
            ret = (close[-1] / close[-period] - 1) * 100
            factors[f"mom_{period}"] = {
                "period": period,
                "label": label,
                "return_pct": round(float(ret), 2),
            }

    # 經典動量因子：12 期報酬 - 1 期報酬（去除短期反轉效應）
    if n > 60:
        mom_long = (close[-1] / close[-60] - 1) * 100
        mom_short = (close[-1] / close[-5] - 1) * 100
        classic_mom = mom_long - mom_short
        factors["classic_momentum"] = {
            "value": round(float(classic_mom), 2),
            "interpretation": "正值=長期趨勢強且非短期過熱" if classic_mom > 0 else "負值=長期弱勢或短期反彈中",
        }

    # 動量方向一致性：多週期是否同方向
    signs = []
    for k, v in factors.items():
        if k.startswith("mom_"):
            signs.append(1 if v["return_pct"] > 0 else -1)
    if signs:
        consistency = abs(sum(signs)) / len(signs)
        all_positive = all(s > 0 for s in signs)
        all_negative = all(s < 0 for s in signs)
        factors["consistency"] = {
            "value": round(consistency, 2),
            "all_positive": all_positive,
            "all_negative": all_negative,
            "interpretation": (
                "全週期一致看多" if all_positive
                else "全週期一致看空" if all_negative
                else f"方向不一致（一致性 {consistency:.0%}）"
            ),
        }

    return factors


# ═══════════════════════════════════════════════════
#  2. 動量加速/減速偵測
# ═══════════════════════════════════════════════════

def _detect_momentum_shift(close: np.ndarray, df: pd.DataFrame) -> dict:
    """偵測動量從加速轉減速或反之。"""
    # ROC（變動率）
    roc_5 = np.zeros(len(close))
    roc_10 = np.zeros(len(close))
    for i in range(5, len(close)):
        roc_5[i] = (close[i] / close[i - 5] - 1) * 100
    for i in range(10, len(close)):
        roc_10[i] = (close[i] / close[i - 10] - 1) * 100

    # 動量加速度 = ROC 的 ROC
    acceleration = np.zeros(len(close))
    for i in range(10, len(close)):
        acceleration[i] = roc_5[i] - roc_5[i - 5]

    current_roc5 = float(roc_5[-1])
    current_roc10 = float(roc_10[-1])
    current_accel = float(acceleration[-1])
    prev_accel = float(acceleration[-2]) if len(acceleration) > 1 else 0

    # RSI 動量
    rsi_calc = registry.calculate("rsi", df)
    rsi_momentum = None
    if rsi_calc and "rsi" in rsi_calc:
        rsi_vals = [v for v in rsi_calc["rsi"] if v is not None]
        if len(rsi_vals) >= 10:
            rsi_momentum = {
                "current": round(rsi_vals[-1], 2),
                "change_5": round(rsi_vals[-1] - rsi_vals[-5], 2) if len(rsi_vals) >= 5 else 0,
                "accelerating": rsi_vals[-1] > rsi_vals[-3] > rsi_vals[-5] if len(rsi_vals) >= 5 else False,
            }

    # 判定動量狀態
    if current_accel > 0 and current_roc5 > 0:
        state = "加速上漲"
    elif current_accel < 0 and current_roc5 > 0:
        state = "減速上漲（動能衰退）"
    elif current_accel < 0 and current_roc5 < 0:
        state = "加速下跌"
    elif current_accel > 0 and current_roc5 < 0:
        state = "減速下跌（可能觸底）"
    else:
        state = "動量中性"

    # 轉折偵測：加速度從正變負或從負變正
    shift_detected = (current_accel > 0 and prev_accel < 0) or (current_accel < 0 and prev_accel > 0)

    return {
        "roc_5": round(current_roc5, 2),
        "roc_10": round(current_roc10, 2),
        "acceleration": round(current_accel, 4),
        "state": state,
        "shift_detected": shift_detected,
        "shift_type": (
            "動量由負轉正（底部反轉信號）" if shift_detected and current_accel > 0
            else "動量由正轉負（頂部反轉信號）" if shift_detected and current_accel < 0
            else "無轉折"
        ),
        "rsi_momentum": rsi_momentum,
    }


# ═══════════════════════════════════════════════════
#  3. 相對動量（vs BTC / 大盤）
# ═══════════════════════════════════════════════════

def _compute_relative_momentum(target_close: np.ndarray, benchmark_close: np.ndarray) -> dict:
    """計算相對動量（RS ratio）。"""
    n = min(len(target_close), len(benchmark_close))
    if n < 20:
        return {"available": False}

    tc = target_close[-n:]
    bc = benchmark_close[-n:]

    results = {}
    for period, label in [(5, "5根"), (10, "10根"), (20, "20根")]:
        if n > period:
            target_ret = (tc[-1] / tc[-period] - 1) * 100
            bench_ret = (bc[-1] / bc[-period] - 1) * 100
            rs = target_ret - bench_ret
            results[f"rs_{period}"] = {
                "target_return": round(float(target_ret), 2),
                "benchmark_return": round(float(bench_ret), 2),
                "relative_strength": round(float(rs), 2),
                "outperforming": rs > 0,
            }

    # 相對強弱趨勢
    rs_ratio = tc / bc
    rs_trend = "上升" if rs_ratio[-1] > rs_ratio[-5] else "下降"

    return {
        "available": True,
        "periods": results,
        "rs_trend": rs_trend,
        "interpretation": (
            "標的相對大盤走強" if rs_trend == "上升"
            else "標的相對大盤走弱"
        ),
    }


# ═══════════════════════════════════════════════════
#  4. 動量反轉偵測
# ═══════════════════════════════════════════════════

def _detect_reversal(close: np.ndarray, high: np.ndarray, low: np.ndarray, df: pd.DataFrame) -> dict:
    """偵測動量反轉信號。"""
    signals = []

    # RSI 背離
    rsi_calc = registry.calculate("rsi", df)
    if rsi_calc and "rsi" in rsi_calc:
        rsi_vals = [v for v in rsi_calc["rsi"] if v is not None]
        if len(rsi_vals) >= 20:
            # 看漲背離：價格創新低但 RSI 沒有
            price_lower = close[-1] < min(close[-20:-1])
            rsi_higher = rsi_vals[-1] > min(rsi_vals[-20:-1])
            if price_lower and rsi_higher:
                signals.append({
                    "type": "RSI 看漲背離",
                    "direction": "bullish",
                    "strength": "strong",
                    "detail": f"價格創 20 根新低但 RSI({rsi_vals[-1]:.1f}) 未跟隨",
                })

            # 看跌背離：價格創新高但 RSI 沒有
            price_higher = close[-1] > max(close[-20:-1])
            rsi_lower = rsi_vals[-1] < max(rsi_vals[-20:-1])
            if price_higher and rsi_lower:
                signals.append({
                    "type": "RSI 看跌背離",
                    "direction": "bearish",
                    "strength": "strong",
                    "detail": f"價格創 20 根新高但 RSI({rsi_vals[-1]:.1f}) 未跟隨",
                })

    # MACD 柱狀圖方向變化
    macd_calc = registry.calculate("macd", df)
    if macd_calc:
        keys = list(macd_calc.keys())
        if len(keys) >= 3:
            hist = [v for v in macd_calc[keys[2]] if v is not None]
            if len(hist) >= 3:
                if hist[-1] > 0 and hist[-2] < 0:
                    signals.append({"type": "MACD 柱狀圖翻正", "direction": "bullish", "strength": "medium"})
                elif hist[-1] < 0 and hist[-2] > 0:
                    signals.append({"type": "MACD 柱狀圖翻負", "direction": "bearish", "strength": "medium"})
                # 柱狀圖縮短
                if hist[-1] > 0 and abs(hist[-1]) < abs(hist[-2]):
                    signals.append({"type": "MACD 動能減弱（柱狀圖縮短）", "direction": "bearish", "strength": "weak"})
                elif hist[-1] < 0 and abs(hist[-1]) < abs(hist[-2]):
                    signals.append({"type": "MACD 空方動能減弱", "direction": "bullish", "strength": "weak"})

    # StochRSI 超買超賣反轉
    stoch_calc = registry.calculate("stochrsi", df)
    if stoch_calc:
        k_vals = list(stoch_calc.values())[0]
        k_clean = [v for v in k_vals if v is not None]
        if len(k_clean) >= 3:
            if k_clean[-1] < 0.2 and k_clean[-2] < 0.2 and k_clean[-1] > k_clean[-2]:
                signals.append({"type": "StochRSI 超賣反轉", "direction": "bullish", "strength": "medium"})
            if k_clean[-1] > 0.8 and k_clean[-2] > 0.8 and k_clean[-1] < k_clean[-2]:
                signals.append({"type": "StochRSI 超買反轉", "direction": "bearish", "strength": "medium"})

    bullish = [s for s in signals if s["direction"] == "bullish"]
    bearish = [s for s in signals if s["direction"] == "bearish"]

    return {
        "total_signals": len(signals),
        "bullish_signals": len(bullish),
        "bearish_signals": len(bearish),
        "signals": signals,
        "net_direction": (
            "偏多反轉" if len(bullish) > len(bearish)
            else "偏空反轉" if len(bearish) > len(bullish)
            else "無明確反轉信號"
        ),
    }


# ═══════════════════════════════════════════════════
#  5. 動量策略回測
# ═══════════════════════════════════════════════════

def _backtest_momentum_strategies(df: pd.DataFrame, timeframe: str) -> dict:
    """回測三種動量策略。"""
    _default_sl = {"15m": 0.03, "1h": 0.04, "4h": 0.08, "1d": 0.10, "1w": 0.15}
    sl = _default_sl.get(timeframe, 0.08)
    tp = sl * 2.5

    strategies = {}

    # 策略 1：RSI 動量追強（RSI > 60 進場，RSI < 40 出場）
    try:
        entry = [{"indicator": "rsi", "operator": ">", "value": 60}]
        exit_ = [{"indicator": "rsi", "operator": "<", "value": 40}]
        r = run_backtest(df, entry, exit_, direction="long", stop_loss_pct=sl, take_profit_pct=tp)
        m = r.metrics
        strategies["rsi_momentum"] = {
            "name": "RSI 動量追強（>60 進場）",
            "trades": m.get("total_trades", 0),
            "win_rate": m.get("win_rate", 0),
            "return_pct": m.get("total_return_pct", 0),
            "sharpe": m.get("sharpe_ratio", 0),
            "mdd": m.get("max_drawdown_pct", 0),
        }
    except Exception:
        pass

    # 策略 2：動量反轉（RSI < 30 超賣反彈）
    try:
        entry = [{"indicator": "rsi", "operator": "<", "value": 30}]
        exit_ = [{"indicator": "rsi", "operator": ">", "value": 55}]
        r = run_backtest(df, entry, exit_, direction="long", stop_loss_pct=sl, take_profit_pct=tp)
        m = r.metrics
        strategies["rsi_reversal"] = {
            "name": "RSI 超賣反轉（<30 進場）",
            "trades": m.get("total_trades", 0),
            "win_rate": m.get("win_rate", 0),
            "return_pct": m.get("total_return_pct", 0),
            "sharpe": m.get("sharpe_ratio", 0),
            "mdd": m.get("max_drawdown_pct", 0),
        }
    except Exception:
        pass

    # 策略 3：ADX 強趨勢追蹤（ADX > 30 + DI+ > DI-）
    try:
        entry = [
            {"indicator": "adx", "operator": ">", "value": 30},
        ]
        exit_ = [{"indicator": "adx", "operator": "<", "value": 20}]
        r = run_backtest(df, entry, exit_, direction="long", stop_loss_pct=sl, take_profit_pct=tp)
        m = r.metrics
        strategies["adx_trend"] = {
            "name": "ADX 強趨勢追蹤（>30 進場）",
            "trades": m.get("total_trades", 0),
            "win_rate": m.get("win_rate", 0),
            "return_pct": m.get("total_return_pct", 0),
            "sharpe": m.get("sharpe_ratio", 0),
            "mdd": m.get("max_drawdown_pct", 0),
        }
    except Exception:
        pass

    # 找最佳策略
    best = None
    best_sharpe = -999
    for key, s in strategies.items():
        if s.get("sharpe", -999) > best_sharpe and s.get("trades", 0) >= 5:
            best_sharpe = s["sharpe"]
            best = key

    return {
        "strategies": strategies,
        "best_strategy": best,
        "best_strategy_name": strategies[best]["name"] if best else "無適合的策略",
    }


# ═══════════════════════════════════════════════════
#  6. 綜合動量評分
# ═══════════════════════════════════════════════════

def _compute_momentum_score(factors: dict, shift: dict, reversal: dict) -> dict:
    """綜合動量評分 -10 ~ +10。"""
    score = 0

    # 多週期動量方向
    consistency = factors.get("consistency", {})
    if consistency.get("all_positive"):
        score += 3
    elif consistency.get("all_negative"):
        score -= 3
    else:
        cv = consistency.get("value", 0.5)
        mom_5 = factors.get("mom_5", {}).get("return_pct", 0)
        if mom_5 > 0:
            score += round(cv * 2)
        else:
            score -= round(cv * 2)

    # 動量加速度
    if shift.get("state") == "加速上漲":
        score += 2
    elif shift.get("state") == "減速上漲（動能衰退）":
        score += 0
    elif shift.get("state") == "加速下跌":
        score -= 2
    elif shift.get("state") == "減速下跌（可能觸底）":
        score += 1

    # RSI 動量
    rsi_m = shift.get("rsi_momentum", {})
    if rsi_m:
        if rsi_m.get("current", 50) > 60:
            score += 1
        elif rsi_m.get("current", 50) < 40:
            score -= 1

    # 反轉信號
    net_bull = reversal.get("bullish_signals", 0)
    net_bear = reversal.get("bearish_signals", 0)
    score += min(2, net_bull) - min(2, net_bear)

    score = max(-10, min(10, score))

    if score >= 5:
        direction = "強烈看多動能"
    elif score >= 2:
        direction = "偏多動能"
    elif score <= -5:
        direction = "強烈看空動能"
    elif score <= -2:
        direction = "偏空動能"
    else:
        direction = "動能中性"

    return {
        "score": score,
        "max_possible": 10,
        "direction": direction,
    }
