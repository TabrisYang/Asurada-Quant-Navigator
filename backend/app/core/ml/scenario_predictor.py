"""阿斯拉量化系統 — 三大情境預測引擎

整合多種分析方法（ML 模型、技術指標、歷史相似度、市場結構），
產出三個最有可能的價格情境，每個情境附帶統計計算的機率百分比。

設計原則：數字靠算，語言靠 LLM。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from app.core.indicators import registry


# ─── 資料結構 ────────────────────────────────────

@dataclass
class Scenario:
    """單一情境"""
    label: str                          # "看漲突破" / "震盪整理" / "回調修正"
    direction: str                      # "bullish" / "neutral" / "bearish"
    probability: float                  # 0.0~1.0
    price_target_low: float             # 預估價格區間下界
    price_target_high: float            # 預估價格區間上界
    timeframe_bars: int                 # 預測有效期（K 線根數）
    supporting_signals: list[dict] = field(default_factory=list)
    invalidation: Optional[str] = None  # 失效條件描述
    risk_level: str = "medium"          # "high" / "medium" / "low"

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "direction": self.direction,
            "probability": round(self.probability, 4),
            "probability_pct": f"{self.probability * 100:.1f}%",
            "price_target": {
                "low": round(self.price_target_low, 2),
                "high": round(self.price_target_high, 2),
            },
            "timeframe_bars": self.timeframe_bars,
            "supporting_signals": self.supporting_signals,
            "invalidation": self.invalidation,
            "risk_level": self.risk_level,
        }


@dataclass
class ScenarioResult:
    """情境預測完整結果"""
    symbol: str
    timeframe: str
    current_price: float
    scenarios: list[Scenario]
    signal_sources: dict               # 各訊號源的權重和值
    generated_at: str = ""
    data_points: int = 0               # 使用的歷史數據量

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "current_price": round(self.current_price, 2),
            "scenarios": [s.to_dict() for s in self.scenarios],
            "signal_sources": self.signal_sources,
            "generated_at": self.generated_at,
            "data_points": self.data_points,
        }


# ─── 訊號收集器 ──────────────────────────────────

def _collect_technical_signals(df: pd.DataFrame) -> dict:
    """從技術指標收集方向性訊號

    Returns:
        {
            "bullish_score": 0.0~1.0,
            "signals": [{"name": ..., "value": ..., "interpretation": ...}]
        }
    """
    signals = []
    bullish_count = 0
    total_count = 0

    closes = df["close"].values
    current = closes[-1]

    # RSI
    rsi_data = registry.calculate("rsi", df)
    if rsi_data and "rsi" in rsi_data:
        rsi_vals = [v for v in rsi_data["rsi"] if v is not None]
        if rsi_vals:
            rsi = rsi_vals[-1]
            total_count += 1
            if rsi < 30:
                bullish_count += 1
                signals.append({"name": "RSI", "value": round(rsi, 1), "interpretation": "超賣區，潛在反彈"})
            elif rsi > 70:
                signals.append({"name": "RSI", "value": round(rsi, 1), "interpretation": "超買區，潛在回調"})
            elif 40 <= rsi <= 60:
                bullish_count += 0.5
                signals.append({"name": "RSI", "value": round(rsi, 1), "interpretation": "中性區間"})
            else:
                if rsi > 50:
                    bullish_count += 0.7
                else:
                    bullish_count += 0.3
                signals.append({"name": "RSI", "value": round(rsi, 1), "interpretation": "偏多" if rsi > 50 else "偏空"})

    # MACD
    macd_data = registry.calculate("macd", df)
    if macd_data and "histogram" in macd_data:
        hist_vals = [v for v in macd_data["histogram"] if v is not None]
        if len(hist_vals) >= 2:
            total_count += 1
            hist = hist_vals[-1]
            hist_prev = hist_vals[-2]
            if hist > 0 and hist > hist_prev:
                bullish_count += 1
                signals.append({"name": "MACD", "value": round(hist, 4), "interpretation": "多頭動能增強"})
            elif hist > 0:
                bullish_count += 0.7
                signals.append({"name": "MACD", "value": round(hist, 4), "interpretation": "多頭但動能減弱"})
            elif hist < 0 and hist < hist_prev:
                signals.append({"name": "MACD", "value": round(hist, 4), "interpretation": "空頭動能增強"})
            else:
                bullish_count += 0.3
                signals.append({"name": "MACD", "value": round(hist, 4), "interpretation": "空頭但動能減弱"})

    # Bollinger Bands %B
    bb_data = registry.calculate("bb", df)
    if bb_data and "%b" in bb_data:
        bb_vals = [v for v in bb_data["%b"] if v is not None]
        if bb_vals:
            bb_pct = bb_vals[-1]
            total_count += 1
            if bb_pct < 0.2:
                bullish_count += 0.8
                signals.append({"name": "BB %B", "value": round(bb_pct, 3), "interpretation": "接近下軌，超賣"})
            elif bb_pct > 0.8:
                bullish_count += 0.2
                signals.append({"name": "BB %B", "value": round(bb_pct, 3), "interpretation": "接近上軌，超買"})
            else:
                bullish_count += 0.5
                signals.append({"name": "BB %B", "value": round(bb_pct, 3), "interpretation": "中間區域"})

    # ADX — 趨勢強度
    adx_data = registry.calculate("adx", df)
    if adx_data and "adx" in adx_data:
        adx_vals = [v for v in adx_data["adx"] if v is not None]
        if adx_vals:
            adx = adx_vals[-1]
            total_count += 1
            # ADX 不給方向，但高 ADX + 趨勢方向 = 更有信心
            trend_up = closes[-1] > closes[-5] if len(closes) >= 5 else True
            if adx > 25:
                bullish_count += 0.8 if trend_up else 0.2
                signals.append({"name": "ADX", "value": round(adx, 1), "interpretation": f"強趨勢（{'上升' if trend_up else '下降'}）"})
            else:
                bullish_count += 0.5
                signals.append({"name": "ADX", "value": round(adx, 1), "interpretation": "弱趨勢/盤整"})

    # 均線排列（用 close vs SMA20）
    if len(closes) >= 20:
        sma20 = float(np.mean(closes[-20:]))
        total_count += 1
        if current > sma20:
            bullish_count += 0.7
            signals.append({"name": "SMA20", "value": round(sma20, 2), "interpretation": f"價格在均線上方（+{(current/sma20-1)*100:.1f}%）"})
        else:
            bullish_count += 0.3
            signals.append({"name": "SMA20", "value": round(sma20, 2), "interpretation": f"價格在均線下方（{(current/sma20-1)*100:.1f}%）"})

    # 成交量趨勢
    rel_vol_data = registry.calculate("rel_vol", df)
    if rel_vol_data and "rel_vol" in rel_vol_data:
        rvol_vals = [v for v in rel_vol_data["rel_vol"] if v is not None]
        if rvol_vals:
            rvol = rvol_vals[-1]
            signals.append({"name": "相對量", "value": round(rvol, 2), "interpretation": "放量" if rvol > 1.5 else ("縮量" if rvol < 0.5 else "正常量")})

    bullish_score = bullish_count / total_count if total_count > 0 else 0.5

    return {
        "bullish_score": round(bullish_score, 4),
        "signals": signals,
        "indicators_used": total_count,
    }


def _collect_ml_signals(df: pd.DataFrame, symbol: str, timeframe: str) -> dict:
    """從 ML 模型收集預測訊號

    Returns:
        {
            "bullish_score": 0.0~1.0,
            "available": bool,
            "reliability": str,
            "details": dict
        }
    """
    try:
        from app.core.ml.model_manager import model_manager
        result = model_manager.predict_consensus(df, symbol, timeframe)

        if result.get("status") != "success":
            return {"bullish_score": 0.5, "available": False, "reliability": "none", "details": {}}

        pred = result["prediction"]
        prob = pred.get("probability", 0.5)
        quality = pred.get("model_quality", {})
        reliability = "medium"
        if quality.get("is_reliable") and quality.get("oos_accuracy", 0) > 0.55:
            reliability = "high"
        elif quality.get("oos_accuracy", 0) < 0.52:
            reliability = "low"

        return {
            "bullish_score": round(prob, 4),
            "available": True,
            "reliability": reliability,
            "details": {
                "probability": prob,
                "direction": pred.get("direction"),
                "confidence": pred.get("confidence"),
                "model_id": pred.get("model_id"),
                "top_features": pred.get("top_features", [])[:3],
                "models_considered": pred.get("models_considered", 1),
            },
        }
    except Exception as e:
        logger.debug(f"ML 預測不可用: {e}")
        return {"bullish_score": 0.5, "available": False, "reliability": "none", "details": {}}


def _collect_historical_similarity(df: pd.DataFrame, lookback: int = 20, n_similar: int = 30) -> dict:
    """歷史相似度分析 — 找出近期走勢最相似的歷史片段，統計後續走勢

    使用最近 lookback 根 K 線的報酬率序列做相似度匹配。

    Returns:
        {
            "bullish_score": 0.0~1.0,
            "available": bool,
            "stats": { "up_pct", "down_pct", "neutral_pct", "avg_return", "median_return" }
        }
    """
    closes = df["close"].values
    if len(closes) < lookback * 3 + 10:
        return {"bullish_score": 0.5, "available": False, "stats": {}}

    # 計算報酬率序列
    returns = np.diff(closes) / closes[:-1]

    # 目前模式（最近 lookback 根）
    current_pattern = returns[-lookback:]
    current_norm = (current_pattern - np.mean(current_pattern))
    current_std = np.std(current_norm)
    if current_std < 1e-10:
        return {"bullish_score": 0.5, "available": False, "stats": {}}
    current_norm = current_norm / current_std

    # 在歷史中搜尋相似片段
    forward_period = 5
    similarities = []
    # 從 lookback 開始，到倒數 lookback+forward_period 結束
    search_end = len(returns) - lookback - forward_period
    for i in range(lookback, search_end):
        hist_pattern = returns[i - lookback:i]
        hist_norm = (hist_pattern - np.mean(hist_pattern))
        hist_std = np.std(hist_norm)
        if hist_std < 1e-10:
            continue
        hist_norm = hist_norm / hist_std

        # 相關係數
        corr = float(np.corrcoef(current_norm, hist_norm)[0, 1])
        if np.isnan(corr):
            continue

        # 後續走勢（i 之後 forward_period 根的報酬）
        future_close = closes[i + forward_period]
        current_close = closes[i]
        future_return = (future_close - current_close) / current_close

        similarities.append((corr, future_return))

    if len(similarities) < 10:
        return {"bullish_score": 0.5, "available": False, "stats": {}}

    # 取最相似的 n_similar 個
    similarities.sort(key=lambda x: x[0], reverse=True)
    top_similar = similarities[:n_similar]
    future_returns = np.array([s[1] for s in top_similar])

    threshold = 0.005  # 0.5% 以內算中性
    up_count = np.sum(future_returns > threshold)
    down_count = np.sum(future_returns < -threshold)
    neutral_count = len(future_returns) - up_count - down_count

    n = len(future_returns)
    up_pct = float(up_count / n)
    down_pct = float(down_count / n)
    neutral_pct = float(neutral_count / n)

    return {
        "bullish_score": round(up_pct, 4),
        "available": True,
        "stats": {
            "up_pct": round(up_pct, 4),
            "down_pct": round(down_pct, 4),
            "neutral_pct": round(neutral_pct, 4),
            "avg_return": round(float(np.mean(future_returns)) * 100, 2),
            "median_return": round(float(np.median(future_returns)) * 100, 2),
            "similar_count": n,
            "avg_correlation": round(float(np.mean([s[0] for s in top_similar])), 3),
        },
    }


def _detect_market_regime(df: pd.DataFrame) -> dict:
    """市場結構/趨勢體制判定

    Returns:
        {
            "regime": "uptrend" / "downtrend" / "ranging",
            "bullish_score": 0.0~1.0,
            "details": { ... }
        }
    """
    closes = df["close"].values
    if len(closes) < 50:
        return {"regime": "unknown", "bullish_score": 0.5, "details": {}}

    # 1. 均線斜率
    sma20 = pd.Series(closes).rolling(20).mean().values
    sma50 = pd.Series(closes).rolling(50).mean().values

    recent_sma20_slope = (sma20[-1] - sma20[-5]) / sma20[-5] if sma20[-5] and not np.isnan(sma20[-5]) else 0
    _recent_sma50_slope = (sma50[-1] - sma50[-10]) / sma50[-10] if sma50[-10] and not np.isnan(sma50[-10]) else 0  # noqa: F841

    # 2. Higher highs / lower lows 判定
    highs = df["high"].values[-20:]
    lows = df["low"].values[-20:]

    # 簡單結構判定：比較前半 vs 後半的高低點
    half = len(highs) // 2
    first_half_high = np.max(highs[:half])
    second_half_high = np.max(highs[half:])
    first_half_low = np.min(lows[:half])
    second_half_low = np.min(lows[half:])

    hh = second_half_high > first_half_high  # Higher High
    hl = second_half_low > first_half_low    # Higher Low
    lh = second_half_high < first_half_high  # Lower High
    ll = second_half_low < first_half_low    # Lower Low

    # 3. 綜合判定
    score = 0.5
    if hh and hl:
        regime = "uptrend"
        score = 0.75
    elif lh and ll:
        regime = "downtrend"
        score = 0.25
    else:
        regime = "ranging"
        score = 0.5

    # 均線排列加分
    if not np.isnan(sma20[-1]) and not np.isnan(sma50[-1]):
        if sma20[-1] > sma50[-1] and recent_sma20_slope > 0:
            score = min(score + 0.1, 1.0)
        elif sma20[-1] < sma50[-1] and recent_sma20_slope < 0:
            score = max(score - 0.1, 0.0)

    # ADX 修正：強趨勢放大極端值，弱趨勢壓縮
    adx_data = registry.calculate("adx", df)
    adx_val = None
    if adx_data and "adx" in adx_data:
        adx_vals = [v for v in adx_data["adx"] if v is not None]
        if adx_vals:
            adx_val = adx_vals[-1]
            if adx_val > 25:
                # 強趨勢：放大偏離中性的程度
                score = 0.5 + (score - 0.5) * 1.2
                score = max(0.0, min(1.0, score))
            else:
                # 弱趨勢：壓縮回中性
                score = 0.5 + (score - 0.5) * 0.7

    return {
        "regime": regime,
        "bullish_score": round(score, 4),
        "details": {
            "structure": "HH+HL" if (hh and hl) else ("LH+LL" if (lh and ll) else "混合"),
            "sma20_slope": round(recent_sma20_slope * 100, 2),
            "sma20_above_sma50": bool(not np.isnan(sma20[-1]) and not np.isnan(sma50[-1]) and sma20[-1] > sma50[-1]),
            "adx": round(adx_val, 1) if adx_val else None,
        },
    }


# ─── 情境建構器 ──────────────────────────────────

def _build_scenarios(
    current_price: float,
    atr: float,
    bullish_prob: float,
    tech_signals: dict,
    ml_signals: dict,
    hist_signals: dict,
    regime_signals: dict,
    forward_bars: int = 5,
) -> list[Scenario]:
    """根據加權機率構建三個情境"""

    bearish_prob = 1.0 - bullish_prob
    # 中性機率：bullish 和 bearish 越接近，中性越高
    overlap = min(bullish_prob, bearish_prob)
    neutral_prob = overlap * 0.8  # 重疊越多 → 中性越高

    # 重新分配為三個情境
    raw_bull = max(bullish_prob - neutral_prob * 0.5, 0.05)
    raw_bear = max(bearish_prob - neutral_prob * 0.5, 0.05)
    raw_neutral = max(neutral_prob, 0.05)

    # 歸一化
    total = raw_bull + raw_bear + raw_neutral
    p_bull = raw_bull / total
    p_neutral = raw_neutral / total
    p_bear = raw_bear / total

    # 依機率排序
    scenarios_raw = [
        ("bullish", p_bull),
        ("neutral", p_neutral),
        ("bearish", p_bear),
    ]
    scenarios_raw.sort(key=lambda x: x[1], reverse=True)

    scenarios = []
    for direction, prob in scenarios_raw:
        if direction == "bullish":
            label = "看漲突破" if prob > 0.4 else "溫和上漲"
            target_low = current_price + atr * 0.5
            target_high = current_price + atr * 2.0
            invalidation = f"跌破 {current_price - atr * 1.0:.2f}"
            risk = "medium" if prob > 0.35 else "high"

            # 收集支撐訊號
            sigs = []
            for s in tech_signals.get("signals", []):
                if "多頭" in s.get("interpretation", "") or "上方" in s.get("interpretation", "") or "超賣" in s.get("interpretation", ""):
                    sigs.append(s)
            if ml_signals.get("available") and ml_signals.get("bullish_score", 0) > 0.6:
                sigs.append({"name": "ML 模型", "value": round(ml_signals["bullish_score"], 3), "interpretation": f"看漲機率 {ml_signals['bullish_score']*100:.1f}%"})
            if hist_signals.get("available") and hist_signals.get("stats", {}).get("up_pct", 0) > 0.5:
                sigs.append({"name": "歷史相似度", "value": hist_signals["stats"]["up_pct"], "interpretation": f"相似情境 {hist_signals['stats']['up_pct']*100:.0f}% 上漲"})

        elif direction == "bearish":
            label = "回調修正" if prob > 0.4 else "溫和下跌"
            target_low = current_price - atr * 2.0
            target_high = current_price - atr * 0.5
            invalidation = f"突破 {current_price + atr * 1.0:.2f} 並站穩"
            risk = "medium" if prob > 0.35 else "high"

            sigs = []
            for s in tech_signals.get("signals", []):
                if "空頭" in s.get("interpretation", "") or "下方" in s.get("interpretation", "") or "超買" in s.get("interpretation", ""):
                    sigs.append(s)
            if ml_signals.get("available") and ml_signals.get("bullish_score", 0) < 0.4:
                sigs.append({"name": "ML 模型", "value": round(1 - ml_signals["bullish_score"], 3), "interpretation": f"看跌機率 {(1-ml_signals['bullish_score'])*100:.1f}%"})
            if hist_signals.get("available") and hist_signals.get("stats", {}).get("down_pct", 0) > 0.3:
                sigs.append({"name": "歷史相似度", "value": hist_signals["stats"]["down_pct"], "interpretation": f"相似情境 {hist_signals['stats']['down_pct']*100:.0f}% 下跌"})

        else:  # neutral
            label = "震盪整理"
            target_low = current_price - atr * 0.8
            target_high = current_price + atr * 0.8
            invalidation = f"日線收盤突破 {current_price + atr * 1.2:.2f} 或跌破 {current_price - atr * 1.2:.2f}"
            risk = "low"

            sigs = []
            for s in tech_signals.get("signals", []):
                if "中性" in s.get("interpretation", "") or "盤整" in s.get("interpretation", "") or "弱趨勢" in s.get("interpretation", ""):
                    sigs.append(s)
            if regime_signals.get("regime") == "ranging":
                sigs.append({"name": "市場結構", "value": None, "interpretation": "盤整格局"})

        # 至少保留一個訊號
        if not sigs and tech_signals.get("signals"):
            sigs = tech_signals["signals"][:1]

        scenarios.append(Scenario(
            label=label,
            direction=direction,
            probability=round(prob, 4),
            price_target_low=round(target_low, 2),
            price_target_high=round(target_high, 2),
            timeframe_bars=forward_bars,
            supporting_signals=sigs[:5],
            invalidation=invalidation,
            risk_level=risk,
        ))

    return scenarios


# ─── 主入口 ──────────────────────────────────────

class ScenarioPredictor:
    """三大情境預測引擎"""

    def predict_scenarios(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        forward_bars: int = 5,
    ) -> ScenarioResult:
        """產出三大情境預測

        Args:
            df: OHLCV DataFrame
            symbol: 交易對
            timeframe: 時間週期
            forward_bars: 預測有效期（K 線根數）

        Returns:
            ScenarioResult 含三個情境及其機率
        """
        from app.utils.timezone import taipei_now

        if df is None or df.empty or len(df) < 30:
            raise ValueError(f"數據不足：需要至少 30 根 K 線，目前僅有 {len(df) if df is not None else 0} 根")

        current_price = float(df["close"].values[-1])

        # Step 1: 多源訊號收集
        logger.info(f"情境預測 [{symbol} {timeframe}]: 開始收集多源訊號...")
        tech = _collect_technical_signals(df)
        ml = _collect_ml_signals(df, symbol, timeframe)
        hist = _collect_historical_similarity(df)
        regime = _detect_market_regime(df)

        # Step 2: 加權計算整體看漲機率
        weights = {}
        scores = {}

        # ML 模型權重（依可靠性調整）
        if ml["available"]:
            ml_weight = {"high": 0.35, "medium": 0.25, "low": 0.10}.get(ml["reliability"], 0.10)
            weights["ml"] = ml_weight
            scores["ml"] = ml["bullish_score"]
        else:
            weights["ml"] = 0.0
            scores["ml"] = 0.5

        # 技術指標
        weights["technical"] = 0.25
        scores["technical"] = tech["bullish_score"]

        # 歷史相似度
        if hist["available"]:
            weights["historical"] = 0.25
            scores["historical"] = hist["bullish_score"]
        else:
            weights["historical"] = 0.0
            scores["historical"] = 0.5

        # 市場結構
        weights["regime"] = 0.15
        scores["regime"] = regime["bullish_score"]

        # 歸一化權重（排除不可用的來源）
        active_total = sum(w for w in weights.values() if w > 0)
        if active_total > 0:
            norm_weights = {k: v / active_total for k, v in weights.items()}
        else:
            norm_weights = {"technical": 0.5, "regime": 0.5, "ml": 0.0, "historical": 0.0}

        # 加權平均
        bullish_prob = sum(scores[k] * norm_weights[k] for k in scores)
        bullish_prob = max(0.05, min(0.95, bullish_prob))  # 限制極端值

        # ATR 用於價格目標計算
        atr_val = 0.0
        atr_data = registry.calculate("atr", df)
        if atr_data and "atr" in atr_data:
            atr_vals = [v for v in atr_data["atr"] if v is not None]
            if atr_vals:
                atr_val = atr_vals[-1]
        if atr_val == 0:
            # 用近 14 日波動替代
            atr_val = float(np.std(df["close"].values[-14:])) * 1.5

        # Step 3: 建構三個情境
        scenarios = _build_scenarios(
            current_price=current_price,
            atr=atr_val,
            bullish_prob=bullish_prob,
            tech_signals=tech,
            ml_signals=ml,
            hist_signals=hist,
            regime_signals=regime,
            forward_bars=forward_bars,
        )

        signal_sources = {
            "weights": {k: round(v, 3) for k, v in norm_weights.items() if v > 0},
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "composite_bullish": round(bullish_prob, 4),
            "technical": {
                "indicators_used": tech["indicators_used"],
                "bullish_score": tech["bullish_score"],
            },
            "ml": {
                "available": ml["available"],
                "reliability": ml.get("reliability", "none"),
                "details": ml.get("details", {}),
            },
            "historical": {
                "available": hist["available"],
                "stats": hist.get("stats", {}),
            },
            "regime": {
                "regime": regime["regime"],
                "details": regime.get("details", {}),
            },
        }

        logger.info(
            f"情境預測 [{symbol} {timeframe}]: 完成 — "
            f"看漲={bullish_prob:.1%}, 結構={regime['regime']}, "
            f"ML={'可用('+ml['reliability']+')' if ml['available'] else '不可用'}"
        )

        return ScenarioResult(
            symbol=symbol,
            timeframe=timeframe,
            current_price=current_price,
            scenarios=scenarios,
            signal_sources=signal_sources,
            generated_at=taipei_now().strftime("%Y-%m-%d %H:%M:%S"),
            data_points=len(df),
        )


# Singleton
scenario_predictor = ScenarioPredictor()
