"""阿斯拉量化系統 — SMC 訂單流結構偵測引擎

Smart Money Concepts (SMC) 量化分析模組。
所有計算在後端精確完成，LLM 僅負責解讀結果。

設計原則：數字靠算，語言靠 LLM。

核心功能：
- Swing High/Low 偵測
- BOS (Break of Structure) / CHoCH (Change of Character) 判定
- Fair Value Gap (FVG) 偵測
- 流動性 Sweep (BSL/SSL) 偵測
- Equal Highs/Lows 聚集流動性
- 多時區共振 (MTF Alignment)
- Premium/Discount 區域判定
- 信心評分（固定計分表）
- ML-Ready 特徵匯出
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from app.core.indicators import registry


# ─── 常量 ────────────────────────────────────────

# 信心計分表（硬編碼，不依賴 LLM）
CONFIDENCE_WEIGHTS = {
    "mtf_alignment":       +15,   # HTF/LTF 方向一致
    "volume_confirm":      +10,   # 關鍵事件成交量確認
    "fvg_support":         +10,   # 有 FVG 支持入場
    "premium_discount_ok": +15,   # 在正確的溢價/折價區
    "sweep_before_entry":  +10,   # 入場前有流動性 Sweep
    "bos_confirm":         +10,   # 有明確 BOS 確認
    "against_htf":         -20,   # 逆高時區方向
    "low_volume":          -10,   # 成交量不足
    "no_fvg":               -5,   # 缺乏 FVG
    "in_premium_for_buy":  -15,   # 在溢價區做多
}

# HTF 自動推斷表
_HTF_MAP = {
    "15m": "1h",
    "1h":  "4h",
    "4h":  "1d",
    "1d":  "1w",
    "1w":  "1w",  # 週線已是最高
}

# 台股 symbol 後綴
_TW_SUFFIX = "/TWD"


# ─── 資料結構 ────────────────────────────────────

@dataclass
class SwingPoint:
    """Swing High/Low 結構點"""
    index: int
    price: float
    timestamp: str
    swing_type: str  # "high" / "low"


@dataclass
class SMCResult:
    """SMC 分析完整結果"""
    symbol: str
    timeframe: str
    current_price: float

    # 市場結構
    trend_htf: str           # "bullish" / "bearish" / "ranging"
    trend_ltf: str
    mtf_aligned: bool

    # 結構事件
    bos_points: list[dict]   = field(default_factory=list)
    choch_points: list[dict] = field(default_factory=list)

    # 流動性
    fvg_zones: list[dict]    = field(default_factory=list)
    sweep_events: list[dict] = field(default_factory=list)
    equal_levels: list[dict] = field(default_factory=list)

    # 交易建議
    bias: str = "WAIT"       # "BUY" / "SELL" / "WAIT" / "NO_TRADE"
    premium_discount: str = "equilibrium"
    fib_50: float = 0.0
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    rr_ratio: Optional[float] = None

    # 信心評分
    confidence: int = 50
    confidence_breakdown: dict = field(default_factory=dict)

    # 參數透明度
    parameters_used: dict = field(default_factory=dict)

    # ML-Ready 特徵
    ml_features: dict = field(default_factory=dict)

    generated_at: str = ""
    data_points: int = 0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "current_price": round(self.current_price, 2),
            "trend_htf": self.trend_htf,
            "trend_ltf": self.trend_ltf,
            "mtf_aligned": self.mtf_aligned,
            "bos_points": self.bos_points,
            "choch_points": self.choch_points,
            "fvg_zones": self.fvg_zones,
            "sweep_events": self.sweep_events,
            "equal_levels": self.equal_levels,
            "bias": self.bias,
            "premium_discount": self.premium_discount,
            "fib_50": round(self.fib_50, 2),
            "entry": round(self.entry, 2) if self.entry else None,
            "stop_loss": round(self.stop_loss, 2) if self.stop_loss else None,
            "take_profit": round(self.take_profit, 2) if self.take_profit else None,
            "rr_ratio": round(self.rr_ratio, 2) if self.rr_ratio else None,
            "confidence": self.confidence,
            "confidence_breakdown": self.confidence_breakdown,
            "parameters_used": self.parameters_used,
            "ml_features": self.ml_features,
            "generated_at": self.generated_at,
            "data_points": self.data_points,
        }


# ─── Swing Point 偵測 ────────────────────────────

def _find_swing_points(df: pd.DataFrame, order: int = 5) -> list[SwingPoint]:
    """用 ±order 根比較法找出 Swing High / Swing Low

    Swing High: high[i] 是前後各 order 根中最高的
    Swing Low:  low[i] 是前後各 order 根中最低的
    """
    highs = df["high"].values
    lows = df["low"].values
    timestamps = df["timestamp"].values if "timestamp" in df.columns else [str(i) for i in range(len(df))]

    swings: list[SwingPoint] = []
    n = len(df)

    for i in range(order, n - order):
        # Swing High
        is_high = True
        for j in range(1, order + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_high = False
                break
        if is_high:
            swings.append(SwingPoint(
                index=i,
                price=float(highs[i]),
                timestamp=str(timestamps[i]),
                swing_type="high",
            ))

        # Swing Low
        is_low = True
        for j in range(1, order + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_low = False
                break
        if is_low:
            swings.append(SwingPoint(
                index=i,
                price=float(lows[i]),
                timestamp=str(timestamps[i]),
                swing_type="low",
            ))

    swings.sort(key=lambda s: s.index)
    return swings


# ─── BOS / CHoCH 偵測 ────────────────────────────

def _detect_bos_choch(
    df: pd.DataFrame,
    swings: list[SwingPoint],
    vol_threshold: float = 1.5,
) -> tuple[list[dict], list[dict]]:
    """追蹤市場結構方向，判定 BOS 和 CHoCH

    BOS: 順勢突破結構點（多頭中突破前高 = BOS）
    CHoCH: 逆勢突破結構點（多頭中跌破前低 = CHoCH → 趨勢轉折）

    僅以實體收盤價（Body Close）作為有效突破判定。
    影線穿透僅視為流動性抓取（Sweep），不算突破。
    """
    closes = df["close"].values
    volumes = df["volume"].values if "volume" in df.columns else None

    # 計算 MA20 成交量
    vol_ma20 = None
    if volumes is not None and len(volumes) >= 20:
        vol_ma20 = pd.Series(volumes).rolling(20).mean().values

    bos_events: list[dict] = []
    choch_events: list[dict] = []

    # 從 swings 中提取最近的高低點序列
    recent_highs = [s for s in swings if s.swing_type == "high"]
    recent_lows = [s for s in swings if s.swing_type == "low"]

    if len(recent_highs) < 2 or len(recent_lows) < 2:
        return bos_events, choch_events

    # 判定當前結構方向：比較最近兩組 swing
    last_highs = recent_highs[-3:]
    last_lows = recent_lows[-3:]

    # 簡化結構方向判定
    if len(last_highs) >= 2 and len(last_lows) >= 2:
        hh = last_highs[-1].price > last_highs[-2].price
        hl = last_lows[-1].price > last_lows[-2].price
        lh = last_highs[-1].price < last_highs[-2].price
        ll = last_lows[-1].price < last_lows[-2].price

        if hh and hl:
            structure_dir = "bullish"
        elif lh and ll:
            structure_dir = "bearish"
        else:
            structure_dir = "ranging"
    else:
        structure_dir = "ranging"

    timestamps = df["timestamp"].values if "timestamp" in df.columns else [str(i) for i in range(len(df))]

    # 掃描每根 K 線，檢查是否突破前一個 swing point
    for i in range(len(df)):
        close_i = float(closes[i])
        ts = str(timestamps[i])

        # 計算成交量比
        vol_ratio = None
        if volumes is not None and vol_ma20 is not None and i < len(vol_ma20):
            ma_val = vol_ma20[i]
            if ma_val and not np.isnan(ma_val) and ma_val > 0:
                vol_ratio = round(float(volumes[i]) / float(ma_val), 2)

        # 檢查是否以實體收盤突破前高
        for sh in recent_highs:
            if sh.index >= i:
                continue
            # 必須在前高之後至少 2 根才算
            if i - sh.index < 2:
                continue

            if close_i > sh.price:
                is_filtered = vol_ratio is not None and vol_ratio < vol_threshold

                event = {
                    "price": round(close_i, 2),
                    "level_price": round(sh.price, 2),
                    "timestamp": ts,
                    "vol_ratio": vol_ratio,
                    "filtered": is_filtered,
                    "bar_index": i,
                }

                if structure_dir == "bullish":
                    # 多頭中突破前高 = BOS
                    event["type"] = "BOS"
                    event["reasoning"] = (
                        f"多頭結構中，實體收盤 {close_i:.2f} 突破前高 {sh.price:.2f}"
                        + (f"，成交量 {vol_ratio:.2f}x MA20" if vol_ratio else "")
                        + ("，成交量不足" if is_filtered else "，確認 BOS")
                    )
                    bos_events.append(event)
                else:
                    # 空頭/盤整中突破前高 = CHoCH（結構轉折）
                    event["type"] = "CHoCH"
                    event["reasoning"] = (
                        f"{'空頭' if structure_dir == 'bearish' else '盤整'}結構中，"
                        f"實體收盤 {close_i:.2f} 突破前高 {sh.price:.2f}，結構轉多"
                        + (f"，成交量 {vol_ratio:.2f}x MA20" if vol_ratio else "")
                    )
                    choch_events.append(event)
                break  # 一根 K 線只算一次

        # 檢查是否以實體收盤跌破前低
        for sl in recent_lows:
            if sl.index >= i:
                continue
            if i - sl.index < 2:
                continue

            if close_i < sl.price:
                is_filtered = vol_ratio is not None and vol_ratio < vol_threshold

                event = {
                    "price": round(close_i, 2),
                    "level_price": round(sl.price, 2),
                    "timestamp": ts,
                    "vol_ratio": vol_ratio,
                    "filtered": is_filtered,
                    "bar_index": i,
                }

                if structure_dir == "bearish":
                    event["type"] = "BOS"
                    event["reasoning"] = (
                        f"空頭結構中，實體收盤 {close_i:.2f} 跌破前低 {sl.price:.2f}"
                        + (f"，成交量 {vol_ratio:.2f}x MA20" if vol_ratio else "")
                        + ("，成交量不足" if is_filtered else "，確認 BOS")
                    )
                    bos_events.append(event)
                else:
                    event["type"] = "CHoCH"
                    event["reasoning"] = (
                        f"{'多頭' if structure_dir == 'bullish' else '盤整'}結構中，"
                        f"實體收盤 {close_i:.2f} 跌破前低 {sl.price:.2f}，結構轉空"
                        + (f"，成交量 {vol_ratio:.2f}x MA20" if vol_ratio else "")
                    )
                    choch_events.append(event)
                break

    # 只保留最近的事件（避免太多歷史事件）
    bos_events = bos_events[-5:]
    choch_events = choch_events[-3:]

    return bos_events, choch_events


# ─── 結構方向判定 ─────────────────────────────────

def _determine_trend(swings: list[SwingPoint]) -> str:
    """根據 Swing 序列判定趨勢方向"""
    highs = [s for s in swings if s.swing_type == "high"]
    lows = [s for s in swings if s.swing_type == "low"]

    if len(highs) < 2 or len(lows) < 2:
        return "ranging"

    hh = highs[-1].price > highs[-2].price
    hl = lows[-1].price > lows[-2].price
    lh = highs[-1].price < highs[-2].price
    ll = lows[-1].price < lows[-2].price

    if hh and hl:
        return "bullish"
    elif lh and ll:
        return "bearish"
    return "ranging"


# ─── Fair Value Gap (FVG) 偵測 ───────────────────

def _detect_fvg(df: pd.DataFrame) -> list[dict]:
    """偵測 Fair Value Gap（失衡區）

    FVG 定義：三根 K 線中，第一根的 high 與第三根的 low 之間存在缺口。
    - Bullish FVG: Low[i] > High[i-2]（缺口向上，需求區）
    - Bearish FVG: High[i] < Low[i-2]（缺口向下，供給區）
    """
    highs = df["high"].values
    lows = df["low"].values
    timestamps = df["timestamp"].values if "timestamp" in df.columns else [str(i) for i in range(len(df))]

    fvg_zones: list[dict] = []

    for i in range(2, len(df)):
        # Bullish FVG: K[i] 的 low > K[i-2] 的 high
        if lows[i] > highs[i - 2]:
            gap_width = float(lows[i] - highs[i - 2])
            fvg_zones.append({
                "type": "bullish",
                "high": round(float(lows[i]), 2),       # FVG 上界
                "low": round(float(highs[i - 2]), 2),    # FVG 下界
                "width": round(gap_width, 2),
                "timestamp": str(timestamps[i - 1]),      # 中間 K 線時間
                "bar_index": i - 1,
                "filled": _is_fvg_filled(df, i, float(highs[i - 2]), float(lows[i]), "bullish"),
            })

        # Bearish FVG: K[i] 的 high < K[i-2] 的 low
        if highs[i] < lows[i - 2]:
            gap_width = float(lows[i - 2] - highs[i])
            fvg_zones.append({
                "type": "bearish",
                "high": round(float(lows[i - 2]), 2),    # FVG 上界
                "low": round(float(highs[i]), 2),         # FVG 下界
                "width": round(gap_width, 2),
                "timestamp": str(timestamps[i - 1]),
                "bar_index": i - 1,
                "filled": _is_fvg_filled(df, i, float(highs[i]), float(lows[i - 2]), "bearish"),
            })

    # 只保留最近未被填補的 FVG + 最近已填補的幾個
    unfilled = [f for f in fvg_zones if not f["filled"]][-5:]
    filled = [f for f in fvg_zones if f["filled"]][-2:]
    return unfilled + filled


def _is_fvg_filled(df: pd.DataFrame, fvg_start: int, low_bound: float, high_bound: float, fvg_type: str) -> bool:
    """檢查 FVG 是否已被後續 K 線填補"""
    for j in range(fvg_start + 1, len(df)):
        if fvg_type == "bullish":
            # Bullish FVG 被填補 = 價格下探到 FVG 區間內
            if df["low"].values[j] <= low_bound:
                return True
        else:
            # Bearish FVG 被填補 = 價格上漲到 FVG 區間內
            if df["high"].values[j] >= high_bound:
                return True
    return False


# ─── 流動性 Sweep 偵測 ───────────────────────────

def _detect_sweeps(
    df: pd.DataFrame,
    swings: list[SwingPoint],
    vol_threshold: float = 1.5,
) -> list[dict]:
    """偵測 BSL (Buy-Side Liquidity) / SSL (Sell-Side Liquidity) Sweep

    Sweep 定義：影線穿透結構點但收盤未突破
    - BSL Sweep: 影線（高點）穿過前高，但收盤低於前高 → 抓取買方流動性
    - SSL Sweep: 影線（低點）穿過前低，但收盤高於前低 → 抓取賣方流動性
    """
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    volumes = df["volume"].values if "volume" in df.columns else None
    timestamps = df["timestamp"].values if "timestamp" in df.columns else [str(i) for i in range(len(df))]

    vol_ma20 = None
    if volumes is not None and len(volumes) >= 20:
        vol_ma20 = pd.Series(volumes).rolling(20).mean().values

    sweep_events: list[dict] = []

    swing_highs = [s for s in swings if s.swing_type == "high"]
    swing_lows = [s for s in swings if s.swing_type == "low"]

    for i in range(len(df)):
        vol_ratio = None
        if volumes is not None and vol_ma20 is not None and i < len(vol_ma20):
            ma_val = vol_ma20[i]
            if ma_val and not np.isnan(ma_val) and ma_val > 0:
                vol_ratio = round(float(volumes[i]) / float(ma_val), 2)

        # BSL Sweep: 影線穿過前高，收盤低於前高
        for sh in swing_highs:
            if sh.index >= i or i - sh.index < 2:
                continue
            if highs[i] > sh.price and closes[i] <= sh.price:
                is_filtered = vol_ratio is not None and vol_ratio < vol_threshold
                sweep_events.append({
                    "type": "BSL",
                    "price": round(float(highs[i]), 2),
                    "level_price": round(sh.price, 2),
                    "timestamp": str(timestamps[i]),
                    "vol_ratio": vol_ratio,
                    "filtered": is_filtered,
                    "bar_index": i,
                    "reasoning": (
                        f"影線穿過前高 {sh.price:.2f}（最高 {highs[i]:.2f}），"
                        f"但收盤 {closes[i]:.2f} 回落，抓取買方流動性"
                        + (f"，成交量 {vol_ratio:.2f}x MA20" if vol_ratio else "")
                    ),
                })
                break

        # SSL Sweep: 影線穿過前低，收盤高於前低
        for sl in swing_lows:
            if sl.index >= i or i - sl.index < 2:
                continue
            if lows[i] < sl.price and closes[i] >= sl.price:
                is_filtered = vol_ratio is not None and vol_ratio < vol_threshold
                sweep_events.append({
                    "type": "SSL",
                    "price": round(float(lows[i]), 2),
                    "level_price": round(sl.price, 2),
                    "timestamp": str(timestamps[i]),
                    "vol_ratio": vol_ratio,
                    "filtered": is_filtered,
                    "bar_index": i,
                    "reasoning": (
                        f"影線穿過前低 {sl.price:.2f}（最低 {lows[i]:.2f}），"
                        f"但收盤 {closes[i]:.2f} 回升，抓取賣方流動性"
                        + (f"，成交量 {vol_ratio:.2f}x MA20" if vol_ratio else "")
                    ),
                })
                break

    return sweep_events[-8:]


# ─── Equal Highs/Lows 偵測 ──────────────────────

def _find_equal_levels(df: pd.DataFrame, swings: list[SwingPoint], tolerance: float = 0.001) -> list[dict]:
    """偵測 Equal Highs / Equal Lows（流動性聚集區）

    當多個 swing point 的價格極為接近時，代表該價位聚集了大量停損單。
    """
    equal_levels: list[dict] = []

    # 按類型分組
    for level_type, points in [("EQH", [s for s in swings if s.swing_type == "high"]),
                                ("EQL", [s for s in swings if s.swing_type == "low"])]:
        if len(points) < 2:
            continue

        # 找接近的價位群組
        used = set()
        for i, p1 in enumerate(points):
            if i in used:
                continue
            cluster = [p1]
            for j, p2 in enumerate(points):
                if j <= i or j in used:
                    continue
                diff = abs(p1.price - p2.price) / p1.price
                if diff <= tolerance:
                    cluster.append(p2)
                    used.add(j)

            if len(cluster) >= 2:
                avg_price = sum(p.price for p in cluster) / len(cluster)
                equal_levels.append({
                    "type": level_type,
                    "price": round(avg_price, 2),
                    "count": len(cluster),
                    "timestamps": [p.timestamp for p in cluster[-3:]],
                })
                used.add(i)

    return equal_levels[-5:]


# ─── Premium / Discount 判定 ────────────────────

def _calc_premium_discount(
    current_price: float,
    swings: list[SwingPoint],
) -> tuple[str, float]:
    """判定價格在溢價區還是折價區

    使用最近一個完整波段的 0.5 Fibonacci 位作為分界：
    - 價格 > 0.5 Fib = Premium（溢價區，適合做空）
    - 價格 < 0.5 Fib = Discount（折價區，適合做多）
    """
    highs = [s for s in swings if s.swing_type == "high"]
    lows = [s for s in swings if s.swing_type == "low"]

    if not highs or not lows:
        return "equilibrium", current_price

    # 取最近的 swing high 和 swing low
    recent_high = max(highs[-3:], key=lambda s: s.price)
    recent_low = min(lows[-3:], key=lambda s: s.price)

    fib_50 = (recent_high.price + recent_low.price) / 2.0

    if current_price > fib_50:
        zone = "premium"
    elif current_price < fib_50:
        zone = "discount"
    else:
        zone = "equilibrium"

    return zone, fib_50


# ─── ATR 止損計算 ────────────────────────────────

def _calc_atr_stoploss(
    df: pd.DataFrame,
    poi_price: float,
    direction: str,
    multiplier: float = 0.5,
) -> float:
    """計算動態止損位

    止損 = POI 邊界 ± 0.5 × ATR(14)
    """
    atr_data = registry.calculate("atr", df)
    atr_val = 0.0
    if atr_data and "atr" in atr_data:
        atr_vals = [v for v in atr_data["atr"] if v is not None]
        if atr_vals:
            atr_val = atr_vals[-1]

    if atr_val == 0:
        atr_val = float(np.std(df["close"].values[-14:])) * 1.5

    if direction == "BUY":
        return poi_price - atr_val * multiplier
    else:
        return poi_price + atr_val * multiplier


# ─── 信心評分 ────────────────────────────────────

def _build_confidence(
    trend_ltf: str,
    trend_htf: str,
    mtf_aligned: bool,
    bos_points: list[dict],
    sweep_events: list[dict],
    fvg_zones: list[dict],
    premium_discount: str,
    bias: str,
) -> tuple[int, dict]:
    """固定計分表計算信心分數"""
    breakdown: dict[str, int] = {}
    base = 50

    # MTF 共振
    if mtf_aligned:
        breakdown["+MTF共振"] = CONFIDENCE_WEIGHTS["mtf_alignment"]
    elif trend_ltf != trend_htf and trend_htf != "ranging":
        breakdown["-逆HTF方向"] = CONFIDENCE_WEIGHTS["against_htf"]

    # BOS 確認
    valid_bos = [b for b in bos_points if not b.get("filtered")]
    if valid_bos:
        breakdown["+BOS確認"] = CONFIDENCE_WEIGHTS["bos_confirm"]

    # 成交量確認
    valid_sweeps = [s for s in sweep_events if not s.get("filtered")]
    if valid_sweeps:
        breakdown["+成交量確認"] = CONFIDENCE_WEIGHTS["volume_confirm"]
        breakdown["+Sweep確認"] = CONFIDENCE_WEIGHTS["sweep_before_entry"]
    else:
        low_vol_events = [s for s in sweep_events if s.get("filtered")]
        if low_vol_events:
            breakdown["-成交量不足"] = CONFIDENCE_WEIGHTS["low_volume"]

    # FVG 支持
    unfilled_fvg = [f for f in fvg_zones if not f.get("filled")]
    if unfilled_fvg:
        breakdown["+FVG支持"] = CONFIDENCE_WEIGHTS["fvg_support"]
    else:
        breakdown["-無FVG"] = CONFIDENCE_WEIGHTS["no_fvg"]

    # Premium/Discount 適當性
    if bias == "BUY" and premium_discount == "discount":
        breakdown["+折價區做多"] = CONFIDENCE_WEIGHTS["premium_discount_ok"]
    elif bias == "SELL" and premium_discount == "premium":
        breakdown["+溢價區做空"] = CONFIDENCE_WEIGHTS["premium_discount_ok"]
    elif bias == "BUY" and premium_discount == "premium":
        breakdown["-溢價區做多"] = CONFIDENCE_WEIGHTS["in_premium_for_buy"]
    elif bias == "SELL" and premium_discount == "discount":
        breakdown["-折價區做空"] = CONFIDENCE_WEIGHTS["in_premium_for_buy"]

    total = base + sum(breakdown.values())
    total = max(0, min(100, total))

    return total, breakdown


# ─── 交易建議生成 ────────────────────────────────

def _generate_trade_bias(
    trend_ltf: str,
    trend_htf: str,
    mtf_aligned: bool,
    premium_discount: str,
    bos_points: list[dict],
    choch_points: list[dict],
    sweep_events: list[dict],
) -> str:
    """生成交易方向建議"""
    # 有 CHoCH 事件 = 結構轉折，等待確認
    recent_choch = [c for c in choch_points if not c.get("filtered")]
    if recent_choch:
        last_choch = recent_choch[-1]
        # CHoCH 轉多
        if "轉多" in last_choch.get("reasoning", ""):
            if premium_discount == "discount":
                return "BUY"
            return "WAIT"
        # CHoCH 轉空
        if "轉空" in last_choch.get("reasoning", ""):
            if premium_discount == "premium":
                return "SELL"
            return "WAIT"

    # MTF 共振 + 正確區域
    if mtf_aligned:
        if trend_htf == "bullish" and premium_discount == "discount":
            return "BUY"
        if trend_htf == "bearish" and premium_discount == "premium":
            return "SELL"
        if trend_htf == "bullish":
            return "WAIT"  # 多頭但在溢價區，等回調
        if trend_htf == "bearish":
            return "WAIT"

    # LTF 有方向但不共振
    if trend_ltf == "bullish" and premium_discount == "discount":
        return "BUY"
    if trend_ltf == "bearish" and premium_discount == "premium":
        return "SELL"

    return "WAIT"


# ─── ML-Ready 特徵計算 ──────────────────────────

def _compute_ml_features(
    df: pd.DataFrame,
    trend_htf: str,
    trend_ltf: str,
    vol_ratio: Optional[float],
    fvg_width: float,
    premium_discount: str,
    confidence: int,
) -> dict:
    """計算 ML-Ready 特徵"""
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    _ = closes[-1]  # current price reference

    # Body/Wick Ratio（最後一根 K 線）
    body = abs(closes[-1] - df["open"].values[-1])
    total_range = highs[-1] - lows[-1]
    body_wick_ratio = round(body / total_range, 4) if total_range > 0 else 0.0

    # ATR
    atr_data = registry.calculate("atr", df)
    atr_14 = 0.0
    if atr_data and "atr" in atr_data:
        atr_vals = [v for v in atr_data["atr"] if v is not None]
        if atr_vals:
            atr_14 = round(atr_vals[-1], 4)

    # RSI Relative（相對 50 的偏離）
    rsi_data = registry.calculate("rsi", df)
    rsi_relative = 0.0
    if rsi_data and "rsi" in rsi_data:
        rsi_vals = [v for v in rsi_data["rsi"] if v is not None]
        if rsi_vals:
            rsi_relative = round(rsi_vals[-1] - 50, 2)

    # Premium/Discount Position (0=discount, 0.5=equilibrium, 1=premium)
    pd_pos = {"discount": 0.0, "equilibrium": 0.5, "premium": 1.0}.get(premium_discount, 0.5)

    return {
        "trend_htf": trend_htf,
        "trend_ltf": trend_ltf,
        "vol_ratio": vol_ratio or 0.0,
        "body_wick_ratio": body_wick_ratio,
        "fvg_width": round(fvg_width, 4),
        "atr_14": atr_14,
        "rsi_relative": rsi_relative,
        "premium_discount_pos": pd_pos,
        "confidence": confidence,
    }


# ─── 主入口 ──────────────────────────────────────

class SMCDetector:
    """SMC 訂單流結構偵測引擎"""

    def detect(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        df_htf: Optional[pd.DataFrame] = None,
        lookback: int = 120,
    ) -> SMCResult:
        """執行完整 SMC 分析

        Args:
            df: LTF OHLCV DataFrame
            symbol: 交易對
            timeframe: LTF 時間框架
            df_htf: HTF OHLCV DataFrame（可選）
            lookback: 回溯 K 線數量

        Returns:
            SMCResult 含完整 SMC 分析結果
        """
        from app.utils.timezone import taipei_now

        if df is None or df.empty or len(df) < 30:
            raise ValueError(f"數據不足：需要至少 30 根 K 線，目前僅有 {len(df) if df is not None else 0} 根")

        # 限制分析範圍
        df_analysis = df.tail(lookback).copy()
        current_price = float(df_analysis["close"].values[-1])

        # 台股適配
        is_tw = symbol.endswith(_TW_SUFFIX)
        vol_threshold = 1.3 if is_tw else 1.5

        # ── Step 1: Swing Points ──
        swing_order = 3 if len(df_analysis) < 60 else 5
        swings = _find_swing_points(df_analysis, order=swing_order)
        logger.info(f"SMC [{symbol} {timeframe}]: 找到 {len(swings)} 個 Swing Points")

        # ── Step 2: LTF 趨勢 ──
        trend_ltf = _determine_trend(swings)

        # ── Step 3: HTF 趨勢（多時區共振）──
        trend_htf = trend_ltf  # 預設與 LTF 相同
        mtf_aligned = True

        if df_htf is not None and not df_htf.empty and len(df_htf) >= 30:
            htf_swings = _find_swing_points(df_htf, order=swing_order)
            trend_htf = _determine_trend(htf_swings)
            mtf_aligned = (trend_htf == trend_ltf) or trend_htf == "ranging"
            logger.info(f"SMC [{symbol}]: HTF={trend_htf}, LTF={trend_ltf}, 共振={'是' if mtf_aligned else '否'}")

        # ── Step 4: BOS / CHoCH ──
        bos_points, choch_points = _detect_bos_choch(df_analysis, swings, vol_threshold=vol_threshold)
        logger.info(f"SMC [{symbol}]: BOS={len(bos_points)}, CHoCH={len(choch_points)}")

        # ── Step 5: FVG ──
        fvg_zones = _detect_fvg(df_analysis)

        # ── Step 6: Sweep ──
        sweep_events = _detect_sweeps(df_analysis, swings, vol_threshold=vol_threshold)

        # ── Step 7: Equal Levels ──
        eq_tolerance = 0.002 if is_tw else 0.001
        equal_levels = _find_equal_levels(df_analysis, swings, tolerance=eq_tolerance)

        # ── Step 8: Premium / Discount ──
        premium_discount, fib_50 = _calc_premium_discount(current_price, swings)

        # ── Step 9: 交易建議 ──
        bias = _generate_trade_bias(
            trend_ltf, trend_htf, mtf_aligned,
            premium_discount, bos_points, choch_points, sweep_events,
        )

        # ── Step 10: Entry / SL / TP ──
        entry = None
        stop_loss = None
        take_profit = None
        rr_ratio = None

        if bias in ("BUY", "SELL"):
            # Entry: 使用最近未填補的 FVG 或 fib_50
            unfilled_fvg = [f for f in fvg_zones if not f.get("filled")]
            if bias == "BUY":
                # 做多入場：折價區 FVG 或 fib_50
                buy_fvgs = [f for f in unfilled_fvg if f.get("type") == "bullish"]
                if buy_fvgs:
                    entry = buy_fvgs[-1]["high"]  # FVG 上界
                else:
                    entry = round(fib_50, 2)

                # 止損：POI 下方 0.5 ATR
                recent_lows = [s for s in swings if s.swing_type == "low"]
                poi = recent_lows[-1].price if recent_lows else entry
                stop_loss = round(_calc_atr_stoploss(df_analysis, poi, "BUY"), 2)

                # 止盈：最近 swing high
                recent_highs = [s for s in swings if s.swing_type == "high"]
                if recent_highs:
                    take_profit = round(recent_highs[-1].price, 2)
                else:
                    atr_data = registry.calculate("atr", df_analysis)
                    atr_val = 0.0
                    if atr_data and "atr" in atr_data:
                        atr_vals = [v for v in atr_data["atr"] if v is not None]
                        if atr_vals:
                            atr_val = atr_vals[-1]
                    take_profit = round(entry + atr_val * 2, 2)

            else:  # SELL
                sell_fvgs = [f for f in unfilled_fvg if f.get("type") == "bearish"]
                if sell_fvgs:
                    entry = sell_fvgs[-1]["low"]
                else:
                    entry = round(fib_50, 2)

                recent_highs = [s for s in swings if s.swing_type == "high"]
                poi = recent_highs[-1].price if recent_highs else entry
                stop_loss = round(_calc_atr_stoploss(df_analysis, poi, "SELL"), 2)

                recent_lows = [s for s in swings if s.swing_type == "low"]
                if recent_lows:
                    take_profit = round(recent_lows[-1].price, 2)
                else:
                    atr_data = registry.calculate("atr", df_analysis)
                    atr_val = 0.0
                    if atr_data and "atr" in atr_data:
                        atr_vals = [v for v in atr_data["atr"] if v is not None]
                        if atr_vals:
                            atr_val = atr_vals[-1]
                    take_profit = round(entry - atr_val * 2, 2)

            # RR Ratio
            if entry and stop_loss and take_profit:
                risk = abs(entry - stop_loss)
                reward = abs(take_profit - entry)
                rr_ratio = round(reward / risk, 2) if risk > 0 else None

        # ── Step 11: 信心評分 ──
        confidence, conf_breakdown = _build_confidence(
            trend_ltf, trend_htf, mtf_aligned,
            bos_points, sweep_events, fvg_zones,
            premium_discount, bias,
        )

        # ── Step 12: ML Features ──
        latest_vol_ratio = None
        if sweep_events:
            latest_vol_ratio = sweep_events[-1].get("vol_ratio")

        max_fvg_width = max((f.get("width", 0) for f in fvg_zones), default=0.0)

        ml_features = _compute_ml_features(
            df_analysis, trend_htf, trend_ltf,
            latest_vol_ratio, max_fvg_width,
            premium_discount, confidence,
        )

        # ── 組裝結果 ──
        htf_label = _HTF_MAP.get(timeframe, "1d")
        parameters_used = {
            "lookback": lookback,
            "swing_order": swing_order,
            "vol_threshold": vol_threshold,
            "atr_multiplier": 0.5,
            "eq_tolerance": eq_tolerance,
            "htf": htf_label,
            "is_tw_stock": is_tw,
        }

        result = SMCResult(
            symbol=symbol,
            timeframe=timeframe,
            current_price=current_price,
            trend_htf=trend_htf,
            trend_ltf=trend_ltf,
            mtf_aligned=mtf_aligned,
            bos_points=bos_points,
            choch_points=choch_points,
            fvg_zones=fvg_zones,
            sweep_events=sweep_events,
            equal_levels=equal_levels,
            bias=bias,
            premium_discount=premium_discount,
            fib_50=fib_50,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            rr_ratio=rr_ratio,
            confidence=confidence,
            confidence_breakdown=conf_breakdown,
            parameters_used=parameters_used,
            ml_features=ml_features,
            generated_at=taipei_now().strftime("%Y-%m-%d %H:%M:%S"),
            data_points=len(df_analysis),
        )

        logger.info(
            f"SMC [{symbol} {timeframe}]: 完成 — "
            f"趨勢={trend_ltf}, 共振={'是' if mtf_aligned else '否'}, "
            f"BOS={len(bos_points)}, CHoCH={len(choch_points)}, "
            f"FVG={len(fvg_zones)}, Sweep={len(sweep_events)}, "
            f"建議={bias}, 信心={confidence}%"
        )

        return result


# Singleton
smc_detector = SMCDetector()
