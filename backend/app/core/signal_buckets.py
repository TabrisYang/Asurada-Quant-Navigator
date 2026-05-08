"""v120.2：把連續訊號值（funding %、OI 變化 %、premium % 等）normalize 成
有意義的 bucket（POSITIVE_HIGH / POSITIVE / NEUTRAL / NEGATIVE / NEGATIVE_HIGH）。

設計目的：
- prediction_tracker 寫入時記錄 raw 值 + bucket
- 算「過去 90 天該 bucket 組合 + 看多/空」命中率
- 連續值直接 group 樣本太散；分 5 bucket 能建立統計顯著樣本

每個 classifier 的門檻來自社群常識 + 歷史經驗，不是統計推導。
未來可基於回測校準調整門檻。
"""

from __future__ import annotations

from typing import Optional


# ─── Funding Rate ───────────────────────────────────
# Binance perp 8h 一期，正常範圍 ±0.01%；極端 ±0.05% 以上

def classify_funding(value: Optional[float]) -> str:
    """funding_rate_pct（已 ×100，例 0.01 表 0.01%）。"""
    if value is None:
        return "UNKNOWN"
    if value >= 0.05:
        return "POSITIVE_HIGH"      # 多頭付費過熱（contrarian sell signal）
    if value >= 0.01:
        return "POSITIVE"            # 多頭付費（情緒偏多）
    if value > -0.01:
        return "NEUTRAL"
    if value > -0.05:
        return "NEGATIVE"            # 空頭付費（散戶恐慌做空）
    return "NEGATIVE_HIGH"           # 極端負費（short squeeze 高機率）


# ─── Open Interest 24h 變化 ─────────────────────────
# OI 24h ±5% 算一般，±15% 是大幅調整

def classify_oi_change(pct: Optional[float]) -> str:
    if pct is None:
        return "UNKNOWN"
    if pct >= 15:
        return "RISING_FAST"         # 大量資金湧入（新部位）
    if pct >= 5:
        return "RISING"
    if pct > -5:
        return "FLAT"
    if pct > -15:
        return "FALLING"             # 部位收斂（避險平倉）
    return "FALLING_FAST"


# ─── Coinbase Premium ───────────────────────────────
# 對應 v119.4 內部的 label，獨立成 module 統一

def classify_premium(pct: Optional[float]) -> str:
    if pct is None:
        return "UNKNOWN"
    if pct >= 0.05:
        return "POSITIVE_HIGH"        # 美國機構強力承接
    if pct >= 0.01:
        return "POSITIVE"
    if pct > -0.01:
        return "NEUTRAL"
    if pct > -0.05:
        return "NEGATIVE"
    return "NEGATIVE_HIGH"            # 美國拋售壓力強


# ─── Long/Short Ratio（global）──────────────────────
# Binance global L/S ratio 中位數 ~1.5；散戶往往偏多

def classify_long_short(ratio: Optional[float]) -> str:
    if ratio is None:
        return "UNKNOWN"
    if ratio >= 3.0:
        return "BULLISH_HEAVY"        # 散戶過度多頭（contrarian sell）
    if ratio >= 1.5:
        return "BULLISH"
    if ratio >= 0.8:
        return "BALANCED"
    if ratio >= 0.5:
        return "BEARISH"
    return "BEARISH_HEAVY"            # 散戶過度空頭（contrarian buy / short squeeze 機會）


# ─── Order Book Imbalance ───────────────────────────
# imbalance_ratio = bid_depth / ask_depth；> 1.5 偏買壓 / < 0.67 偏賣壓

def classify_ob_imbalance(ratio: Optional[float]) -> str:
    if ratio is None:
        return "UNKNOWN"
    if ratio >= 2.0:
        return "BUY_HEAVY"            # 強買壓（限價買單堆疊）
    if ratio >= 1.5:
        return "BUY_PRESSURE"
    if ratio >= 0.67:
        return "BALANCED"
    if ratio >= 0.5:
        return "SELL_PRESSURE"
    return "SELL_HEAVY"


# ─── Fear & Greed ───────────────────────────────────
# alternative.me 0-100：< 25 極度恐懼 / > 75 極度貪婪

def classify_fear_greed(value: Optional[int]) -> str:
    if value is None:
        return "UNKNOWN"
    if value >= 75:
        return "EXTREME_GREED"        # 接近頂部（contrarian sell）
    if value >= 55:
        return "GREED"
    if value >= 45:
        return "NEUTRAL"
    if value >= 25:
        return "FEAR"
    return "EXTREME_FEAR"             # 接近底部（contrarian buy）


# ─── ETF Flow（v119.5 跳過，留 placeholder） ─────────
# 7 天累積流入 USD（單位百萬美元）

def classify_etf_flow(usd_7d: Optional[float]) -> str:
    if usd_7d is None:
        return "UNKNOWN"
    # 單位假設為美元（不是百萬）；BTC ETF 7 天 1B+ 算強流入
    if usd_7d >= 1_000_000_000:
        return "INFLOW_HIGH"
    if usd_7d >= 200_000_000:
        return "INFLOW"
    if usd_7d > -200_000_000:
        return "FLAT"
    if usd_7d > -1_000_000_000:
        return "OUTFLOW"
    return "OUTFLOW_HIGH"


# ─── 一鍵 classify 全部 ─────────────────────────────

def classify_all_signals(derivatives: dict, sentiment: dict | None = None) -> dict:
    """從 external_signals.derivatives + sentiment 一鍵 classify 全部訊號。

    Args:
        derivatives: chart_state.external_signals.derivatives dict
        sentiment: chart_state.external_signals.sentiment dict（含 fear_greed_value）

    Returns:
        {"funding": "POSITIVE", "oi_change": "RISING", "premium": "NEUTRAL", ...}
        每個 key 對應一個 bucket。值為 'UNKNOWN' 表示該訊號 fetcher 失敗或不適用。
    """
    sentiment = sentiment or {}
    return {
        "funding": classify_funding(derivatives.get("funding_rate_pct")),
        "oi_change": classify_oi_change(derivatives.get("open_interest_24h_change_pct")),
        "premium": classify_premium(derivatives.get("coinbase_premium_pct")),
        "long_short": classify_long_short(derivatives.get("global_long_short_ratio")),
        "ob_imbalance": classify_ob_imbalance(derivatives.get("ob_imbalance_ratio")),
        "fear_greed": classify_fear_greed(sentiment.get("fear_greed_value")),
        "etf_flow": classify_etf_flow(derivatives.get("etf_flow_7d_usd")),
    }
