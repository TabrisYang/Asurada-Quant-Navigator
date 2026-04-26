"""阿斯拉量化系統 — 分批進場（Laddered Entries）後端計算引擎。

設計原則（不可違反）：
- 所有 ladder 價位「必須」直接來自 indicator value（BB / EMA / Donchian / ATR），
  「禁止」自行算術推估，避免 LLM 拿到後再被質疑「編造數字」
- 倉位配比依 regime 自動切換（trending → 金字塔加碼，ranging → 倒金字塔接刀，
  high_vol → 對稱平均，confidence < 0.5 → 跳過）
- SL/TP 一律從 ATR 反推，保證 RR ≥ 2

對外只暴露 `compute_laddered_entries()` 一個函式，回傳 dict 給 LLM 引用 + 給回測 ladder_config。
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import pandas as pd
from loguru import logger

from app.core.indicators import registry

# ─── Regime → 倉位配比對應表 ──────────────────────────────────
# 寫死（不靠 LLM 決定），確保跨呼叫一致
_REGIME_RATIO_MAP: dict[str, dict[str, Any]] = {
    "trending_up": {
        "ratios": [50, 30, 20],
        "strategy": "金字塔加碼（trending — 趨勢確認後追進場）",
        "long_ok": True,
        "short_ok": False,  # 上升趨勢不開反向
    },
    "trending_down": {
        "ratios": [50, 30, 20],
        "strategy": "金字塔加碼（trending — 趨勢確認後追進場）",
        "long_ok": False,
        "short_ok": True,
    },
    "ranging": {
        "ratios": [25, 35, 40],
        "strategy": "倒金字塔接刀（ranging — 越接越加碼）",
        "long_ok": True,
        "short_ok": True,
    },
    "low_vol": {
        "ratios": [25, 35, 40],
        "strategy": "倒金字塔接刀（low_vol — 區間操作）",
        "long_ok": True,
        "short_ok": True,
    },
    "high_vol": {
        "ratios": [33, 33, 34],
        "strategy": "對稱平均（high_vol — ATR 等距分批）",
        "long_ok": True,
        "short_ok": True,
    },
}

_CONFIDENCE_THRESHOLD = 0.5  # 低於此值整個 ladder 跳過
_MIN_RR_RATIO = 2.0          # SL/TP 最低風險報酬比
_SL_ATR_MULT = 1.5           # SL = min_entry - ATR × 1.5
_TP_ATR_MULT = 2.0           # TP = weighted_avg ± ATR × 2（or risk × _MIN_RR_RATIO，取較大）


def _last_valid(values: list) -> Optional[float]:
    """從一個含 None/NaN 的 list 取最後一個有效數值。"""
    if not values:
        return None
    for v in reversed(values):
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f != f:  # NaN check
            continue
        return f
    return None


def _get_indicator_value(df: pd.DataFrame, indicator_id: str, series_key: str, params: Optional[dict] = None) -> Optional[float]:
    """從 registry 取最後一根有效值，失敗回 None。"""
    try:
        result = registry.calculate(indicator_id, df, params)
    except Exception as e:
        logger.debug(f"compute_laddered_entries: 指標 {indicator_id} 計算失敗：{e}")
        return None
    if not result or series_key not in result:
        return None
    return _last_valid(result[series_key])


def _build_long_ladder(
    df: pd.DataFrame, regime: str, current_price: float, atr: float,
    smc_entry: Optional[float],
) -> tuple[list[dict], list[str]]:
    """組 long 三檔 ladder，回傳 (entries, missing_indicators)。

    entries: [{"price", "size_pct", "source"}, ...] 已排序（最高價在前）
    missing_indicators: 缺失的指標 ID（給上層自動補回用）
    """
    missing: list[str] = []
    entries: list[dict] = []
    ratios = _REGIME_RATIO_MAP[regime]["ratios"]

    if regime in ("trending_up",):
        # current → ema_20 → bb_middle
        ema_20 = _get_indicator_value(df, "ema", "EMA(20)", {"period": 20})
        bb_mid = _get_indicator_value(df, "bb", "BB_Middle")
        first_price = smc_entry if smc_entry else current_price
        first_source = "current + SMC OB" if smc_entry else "current（趨勢確認追進場）"
        entries.append({"price": first_price, "size_pct": ratios[0], "source": first_source})
        if ema_20 is not None:
            entries.append({"price": ema_20, "size_pct": ratios[1], "source": "EMA20 動態支撐"})
        else:
            missing.append("ema")
        if bb_mid is not None:
            entries.append({"price": bb_mid, "size_pct": ratios[2], "source": "BB 中軌"})
        else:
            missing.append("bb")
    elif regime in ("ranging", "low_vol"):
        # bb_middle → bb_lower → donchian_low
        bb_mid = _get_indicator_value(df, "bb", "BB_Middle")
        bb_lower = _get_indicator_value(df, "bb", "BB_Lower")
        dc_lower = _get_indicator_value(df, "donchian", "DC_Lower")
        if bb_mid is not None:
            entries.append({"price": bb_mid, "size_pct": ratios[0], "source": "BB 中軌（首檔輕倉）"})
        else:
            missing.append("bb")
        if bb_lower is not None:
            entries.append({"price": bb_lower, "size_pct": ratios[1], "source": "BB 下軌（中檔加碼）"})
        if dc_lower is not None:
            entries.append({"price": dc_lower, "size_pct": ratios[2], "source": "Donchian 區間下緣（重倉接刀）"})
        else:
            missing.append("donchian")
    elif regime == "high_vol":
        # current → current - 1×ATR → current - 2×ATR（ATR 直接從 df 取）
        entries.append({"price": current_price, "size_pct": ratios[0], "source": "current（high_vol 首檔）"})
        if atr > 0:
            entries.append({"price": current_price - atr, "size_pct": ratios[1], "source": "current − 1×ATR"})
            entries.append({"price": current_price - 2 * atr, "size_pct": ratios[2], "source": "current − 2×ATR"})
        else:
            missing.append("atr")

    return entries, missing


def _build_short_ladder(
    df: pd.DataFrame, regime: str, current_price: float, atr: float,
    smc_entry: Optional[float],
) -> tuple[list[dict], list[str]]:
    """組 short 三檔 ladder（與 long 對稱）。"""
    missing: list[str] = []
    entries: list[dict] = []
    ratios = _REGIME_RATIO_MAP[regime]["ratios"]

    if regime == "trending_down":
        # current → ema_20 → bb_middle（做空在阻力位加碼）
        ema_20 = _get_indicator_value(df, "ema", "EMA(20)", {"period": 20})
        bb_mid = _get_indicator_value(df, "bb", "BB_Middle")
        first_price = smc_entry if smc_entry else current_price
        first_source = "current + SMC OB" if smc_entry else "current（趨勢確認追進場）"
        entries.append({"price": first_price, "size_pct": ratios[0], "source": first_source})
        if ema_20 is not None:
            entries.append({"price": ema_20, "size_pct": ratios[1], "source": "EMA20 動態壓力"})
        else:
            missing.append("ema")
        if bb_mid is not None:
            entries.append({"price": bb_mid, "size_pct": ratios[2], "source": "BB 中軌"})
        else:
            missing.append("bb")
    elif regime in ("ranging", "low_vol"):
        # bb_middle → bb_upper → donchian_high
        bb_mid = _get_indicator_value(df, "bb", "BB_Middle")
        bb_upper = _get_indicator_value(df, "bb", "BB_Upper")
        dc_upper = _get_indicator_value(df, "donchian", "DC_Upper")
        if bb_mid is not None:
            entries.append({"price": bb_mid, "size_pct": ratios[0], "source": "BB 中軌（首檔輕倉）"})
        else:
            missing.append("bb")
        if bb_upper is not None:
            entries.append({"price": bb_upper, "size_pct": ratios[1], "source": "BB 上軌（中檔加碼）"})
        if dc_upper is not None:
            entries.append({"price": dc_upper, "size_pct": ratios[2], "source": "Donchian 區間上緣（重倉接刀）"})
        else:
            missing.append("donchian")
    elif regime == "high_vol":
        entries.append({"price": current_price, "size_pct": ratios[0], "source": "current（high_vol 首檔）"})
        if atr > 0:
            entries.append({"price": current_price + atr, "size_pct": ratios[1], "source": "current + 1×ATR"})
            entries.append({"price": current_price + 2 * atr, "size_pct": ratios[2], "source": "current + 2×ATR"})
        else:
            missing.append("atr")

    return entries, missing


def _normalize_ratios(entries: list[dict]) -> list[dict]:
    """部分 entries 失敗時，按剩餘的 size_pct 重新歸一到 100%。"""
    if not entries:
        return entries
    total = sum(e["size_pct"] for e in entries)
    if total <= 0:
        return entries
    if total == 100:
        return entries
    for e in entries:
        e["size_pct"] = round(e["size_pct"] * 100 / total, 1)
    return entries


def _weighted_average(entries: list[dict]) -> Optional[float]:
    """按 size_pct 加權平均進場價。"""
    if not entries:
        return None
    total_weight = sum(e["size_pct"] for e in entries)
    if total_weight <= 0:
        return None
    return sum(e["price"] * e["size_pct"] for e in entries) / total_weight


def _compute_sl_tp(entries: list[dict], atr: float, side: Literal["long", "short"]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """從 entries + ATR 反推 SL / TP / RR。

    Long:  SL = min(entries) - 1.5 ATR ；TP = avg + max(2 ATR, risk × 2)
    Short: SL = max(entries) + 1.5 ATR ；TP = avg - max(2 ATR, risk × 2)
    """
    if not entries or atr <= 0:
        return None, None, None
    avg = _weighted_average(entries)
    if avg is None:
        return None, None, None
    prices = [e["price"] for e in entries]
    if side == "long":
        sl = min(prices) - _SL_ATR_MULT * atr
        risk = avg - sl
        tp = avg + max(_TP_ATR_MULT * atr, risk * _MIN_RR_RATIO)
    else:
        sl = max(prices) + _SL_ATR_MULT * atr
        risk = sl - avg
        tp = avg - max(_TP_ATR_MULT * atr, risk * _MIN_RR_RATIO)
    rr = max(_TP_ATR_MULT, _MIN_RR_RATIO)
    return sl, tp, rr


def _round_price(p: Optional[float]) -> Optional[float]:
    """根據量級自動小數位（價格 > 1000 取 0 位、> 10 取 2 位、其餘 4 位）。"""
    if p is None:
        return None
    if p >= 1000:
        return round(p, 0)
    if p >= 10:
        return round(p, 2)
    return round(p, 4)


def compute_laddered_entries(
    df: pd.DataFrame,
    direction: Literal["long", "short", "both"] = "both",
    regime: str = "unknown",
    regime_confidence: float = 0.0,
    smc_long_entry: Optional[float] = None,
    smc_short_entry: Optional[float] = None,
    n_tranches: int = 3,
) -> dict:
    """後端算分批進場價，全部從 indicator values 取，禁止推算。

    Args:
        df: OHLCV DataFrame（用最後一根當 current）
        direction: 要計算哪些方向
        regime: 從 chart_state.currentRegime 來，6 種之一
        regime_confidence: 0.0~1.0，< 0.5 直接跳過
        smc_long_entry / smc_short_entry: 若 SMC 已給 entry 則優先當第一檔
        n_tranches: 預設 3（本版固定，未來再做動態）

    Returns:
        dict — 含 enabled / long_entries / short_entries / SL / TP / 加權均價 / regime / strategy / warning。
        若 enabled = False，long_entries / short_entries 會是空 list。
    """
    if df is None or df.empty or len(df) < 30:
        return {"enabled": False, "warning": "資料不足（< 30 根 K 線），無法計算 ladder", "regime_used": regime, "n_tranches": n_tranches}

    # confidence 閾值 — 沿用既有 regimeWarning 邏輯
    if regime_confidence < _CONFIDENCE_THRESHOLD or regime not in _REGIME_RATIO_MAP:
        return {
            "enabled": False,
            "warning": f"regime={regime} confidence={regime_confidence:.2f} < {_CONFIDENCE_THRESHOLD}，跳過分批進場（建議單一進場 + 小倉位試單）",
            "regime_used": regime,
            "regime_confidence": regime_confidence,
            "n_tranches": n_tranches,
        }

    cfg = _REGIME_RATIO_MAP[regime]
    current_price = float(df["close"].iloc[-1])

    # ATR — 從 registry 取（給 high_vol 算分檔距離 + 給 SL/TP 用）
    atr = _get_indicator_value(df, "atr", "ATR", {"period": 14}) or 0.0

    long_entries: list[dict] = []
    short_entries: list[dict] = []
    missing_set: set[str] = set()

    if direction in ("long", "both") and cfg["long_ok"]:
        entries, missing = _build_long_ladder(df, regime, current_price, atr, smc_long_entry)
        long_entries = _normalize_ratios(entries)
        missing_set.update(missing)
    if direction in ("short", "both") and cfg["short_ok"]:
        entries, missing = _build_short_ladder(df, regime, current_price, atr, smc_short_entry)
        short_entries = _normalize_ratios(entries)
        missing_set.update(missing)

    # SL / TP
    long_sl, long_tp, long_rr = _compute_sl_tp(long_entries, atr, "long") if long_entries else (None, None, None)
    short_sl, short_tp, short_rr = _compute_sl_tp(short_entries, atr, "short") if short_entries else (None, None, None)

    # 對外輸出（價格全部四捨五入到合適小數位）
    out_long = [
        {"price": _round_price(e["price"]), "size_pct": e["size_pct"], "source": e["source"]}
        for e in long_entries
    ]
    out_short = [
        {"price": _round_price(e["price"]), "size_pct": e["size_pct"], "source": e["source"]}
        for e in short_entries
    ]

    return {
        "enabled": bool(long_entries or short_entries),
        "regime_used": regime,
        "regime_confidence": regime_confidence,
        "ratio_strategy": cfg["strategy"],
        "n_tranches": n_tranches,
        "current_price": _round_price(current_price),
        "atr_used": _round_price(atr) if atr > 0 else None,
        "long_entries": out_long,
        "short_entries": out_short,
        "weighted_avg_entry_long": _round_price(_weighted_average(long_entries)),
        "weighted_avg_entry_short": _round_price(_weighted_average(short_entries)),
        "stop_loss_long": _round_price(long_sl),
        "take_profit_long": _round_price(long_tp),
        "rr_long": round(long_rr, 2) if long_rr else None,
        "stop_loss_short": _round_price(short_sl),
        "take_profit_short": _round_price(short_tp),
        "rr_short": round(short_rr, 2) if short_rr else None,
        "missing_indicators": sorted(missing_set),  # 給上層自動補指標用
        "warning": None,
        "rule": "★ 後端強制：所有 price 直接來自 indicator value（BB / EMA / Donchian / ATR），LLM 必須引用，禁止自行推算",
    }
