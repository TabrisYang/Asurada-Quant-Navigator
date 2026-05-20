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
_SL_ATR_MULT = 1.5           # 舊固定值（feature flag OFF 時 fallback 用）
_TP_ATR_MULT = 2.0           # 舊固定值（feature flag OFF 時 fallback 用）

# v104 Fix A：timeframe-aware ATR 倍數基礎表（經驗值 — TODO: walk-forward 量化最優）
# 1d 持倉用 1.5×ATR 設 SL 通常 1-2 天就被打到（過緊）；timeframe 越長倍數越大
_TIMEFRAME_BASE_MULTS = {
    "15m": (0.8, 1.5),
    "30m": (1.0, 1.8),
    "1h":  (1.2, 2.0),
    "2h":  (1.4, 2.3),
    "4h":  (1.5, 2.5),
    "12h": (2.0, 3.2),
    "1d":  (2.5, 4.0),
    "3d":  (3.2, 5.0),
    "1w":  (4.0, 6.0),
}

# regime / 信心修正係數
_REGIME_MULT_ADJUST = {"high_vol": 1.3, "low_vol": 0.8}  # 其他 regime → 1.0
_CONFIDENCE_TP_ADJUST = {"high": 1.3, "medium": 1.0, "low": 0.8}

# Cap：最終倍數限制在基礎值 ±50% 內，避免疊加過度
_MULT_CAP_MIN = 0.7  # 最終 ≥ 基礎值 × 0.7
_MULT_CAP_MAX = 1.5  # 最終 ≤ 基礎值 × 1.5


def _get_atr_mults(
    timeframe: Optional[str] = None,
    regime: Optional[str] = None,
    confidence: Optional[str] = None,
) -> tuple[float, float]:
    """v104 Fix A：依 timeframe + regime + 信心算 ATR 倍數（含 cap）。

    feature flag adaptive_atr_mults_enabled 為 False → 直接回 (1.5, 2.0)。
    timeframe 沒給或不認識 → fallback (1.5, 2.0)。
    各維度疊加後用 cap 限在基礎值 ±50% 內。
    """
    try:
        from app.core.config.settings import settings as _s
        if not getattr(_s, "adaptive_atr_mults_enabled", True):
            return _SL_ATR_MULT, _TP_ATR_MULT
    except Exception:
        pass

    if not timeframe or timeframe not in _TIMEFRAME_BASE_MULTS:
        return _SL_ATR_MULT, _TP_ATR_MULT

    base_sl, base_tp = _TIMEFRAME_BASE_MULTS[timeframe]

    # regime 修正（影響 SL+TP 對稱）
    regime_adj = _REGIME_MULT_ADJUST.get(regime or "", 1.0) if regime else 1.0

    # 信心修正（只影響 TP — 高信心可以拉遠賺更多）
    conf_adj = _CONFIDENCE_TP_ADJUST.get(confidence or "", 1.0) if confidence else 1.0

    sl_mult = base_sl * regime_adj
    tp_mult = base_tp * regime_adj * conf_adj

    # Cap：最終值限制在 base × [0.7, 1.5]
    sl_mult = max(base_sl * _MULT_CAP_MIN, min(base_sl * _MULT_CAP_MAX, sl_mult))
    tp_mult = max(base_tp * _MULT_CAP_MIN, min(base_tp * _MULT_CAP_MAX, tp_mult))

    return round(sl_mult, 3), round(tp_mult, 3)


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


def _clamp_to_current_price(
    entries: list[dict],
    current_price: float,
    side: Literal["long", "short"],
) -> list[dict]:
    """v105.2 Bug A：long entries 必須 ≤ 現價、short entries 必須 ≥ 現價。

    根因（從 ADA 4h 真實案例）：價格在區間下半部時 BB 中軌會高於現價，
    但 ladder 仍用 BB 中軌當第 1 檔做多 → 「等漲了再買」反邏輯。

    處理：indicator 值若違反方向約束，clamp 到現價並標註。
    """
    if not entries or current_price <= 0:
        return entries
    out = []
    for e in entries:
        p = e["price"]
        if side == "long" and p > current_price:
            new_p = current_price
            e = {**e, "price": new_p, "source": e.get("source", "") + f"（原 ${p:.4f} > 現價，clamp 到現價）"}
        elif side == "short" and p < current_price:
            new_p = current_price
            e = {**e, "price": new_p, "source": e.get("source", "") + f"（原 ${p:.4f} < 現價，clamp 到現價）"}
        out.append(e)
    return out


def _reassign_ratios_by_position(
    entries: list[dict],
    regime_ratios: list[int],
    side: Literal["long", "short"],
) -> list[dict]:
    """v105.2 Bug B：依最終價格位置（不是建立順序）重新分配 size_pct。

    根因（從 ADA 4h 真實案例）：v105.1 _enforce_minimum_spacing 排序時 size 跟著
    source 走 → 25/40/35 不是設計的 25/35/40。

    處理：排序後依「越深倉位越重」（倒金字塔接刀）原則重新分配 ratios。
    long: 最貴的 → ratios[0]（最輕）/ 最便宜 → ratios[-1]（最重）
    short: 反之
    """
    if not entries or len(entries) != len(regime_ratios):
        return entries
    # long 由高到低（最便宜在最後加最多倉），short 由低到高（最貴在最後加最多倉）
    if side == "long":
        sorted_entries = sorted(entries, key=lambda e: -e["price"])
    else:
        sorted_entries = sorted(entries, key=lambda e: e["price"])
    for i, e in enumerate(sorted_entries):
        e["size_pct"] = regime_ratios[i]
    return sorted_entries


def _enforce_minimum_spacing(
    entries: list[dict],
    atr: float,
    side: Literal["long", "short"],
    min_atr_fraction: float = 0.5,
) -> list[dict]:
    """v105.1：強制各檔距離 ≥ 0.5×ATR，避免 BB 下軌跟 Donchian 下緣重合（差 0.04%）。

    實測 ADA 4h 案例：BB Lower=$0.2430、Donchian Lower=$0.2429 → 第 2/3 檔幾乎同價。
    根因：低波動環境下兩個 indicator 偶會重合，但 ladder 設計期望各檔有間距。

    解法：偵測相鄰兩檔距離 < min_atr_fraction × ATR 時，把較深那檔調整到「上一檔 - 0.5×ATR」
    （long）或「上一檔 + 0.5×ATR」（short），保證 spacing。
    若 ATR 太小（< 平均價的 0.1%），跳過調整避免微調無意義。
    """
    if not entries or atr <= 0 or len(entries) < 2:
        return entries
    avg_price = sum(e["price"] for e in entries) / len(entries)
    if atr / max(avg_price, 1e-9) < 0.001:  # ATR 太小，無法用 ATR 拉開
        return entries

    min_gap = atr * min_atr_fraction
    # entries 排序：long 第 1 檔最高、第 3 檔最低；short 反之
    if side == "long":
        # 多檔由高到低，下一檔應該 ≤ 上一檔 - min_gap
        sorted_entries = sorted(entries, key=lambda e: -e["price"])
        for i in range(1, len(sorted_entries)):
            prev = sorted_entries[i - 1]
            cur = sorted_entries[i]
            gap = prev["price"] - cur["price"]
            if gap < min_gap:
                cur["price"] = prev["price"] - min_gap
                cur["source"] = cur.get("source", "") + f"（拉開 ≥ {min_atr_fraction}×ATR）"
        return sorted_entries
    else:
        # 空檔由低到高，下一檔應該 ≥ 上一檔 + min_gap
        sorted_entries = sorted(entries, key=lambda e: e["price"])
        for i in range(1, len(sorted_entries)):
            prev = sorted_entries[i - 1]
            cur = sorted_entries[i]
            gap = cur["price"] - prev["price"]
            if gap < min_gap:
                cur["price"] = prev["price"] + min_gap
                cur["source"] = cur.get("source", "") + f"（拉開 ≥ {min_atr_fraction}×ATR）"
        return sorted_entries


def _weighted_average(entries: list[dict]) -> Optional[float]:
    """按 size_pct 加權平均進場價。"""
    if not entries:
        return None
    total_weight = sum(e["size_pct"] for e in entries)
    if total_weight <= 0:
        return None
    return sum(e["price"] * e["size_pct"] for e in entries) / total_weight


def _compute_sl_tp(
    entries: list[dict],
    atr: float,
    side: Literal["long", "short"],
    sl_mult: float = _SL_ATR_MULT,
    tp_mult: float = _TP_ATR_MULT,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """從 entries + ATR 反推 SL / TP / RR（v104：倍數可外部注入）。

    Long:  SL = min(entries) - sl_mult × ATR ；TP = avg + max(tp_mult × ATR, risk × MIN_RR)
    Short: SL = max(entries) + sl_mult × ATR ；TP = avg - max(tp_mult × ATR, risk × MIN_RR)
    """
    if not entries or atr <= 0:
        return None, None, None
    avg = _weighted_average(entries)
    if avg is None:
        return None, None, None
    prices = [e["price"] for e in entries]
    if side == "long":
        sl = min(prices) - sl_mult * atr
        risk = avg - sl
        tp = avg + max(tp_mult * atr, risk * _MIN_RR_RATIO)
        # v105.3 Bug C：算實際 RR（之前直接 max(tp_mult, MIN_RR) 跟實際距離比例不符）
        reward = tp - avg
    else:
        sl = max(prices) + sl_mult * atr
        risk = sl - avg
        tp = avg - max(tp_mult * atr, risk * _MIN_RR_RATIO)
        reward = avg - tp
    rr = (reward / risk) if risk > 0 else _MIN_RR_RATIO
    return sl, tp, rr


def _round_price(p: Optional[float]) -> Optional[float]:
    """價格精度對齊 K 線圖（共用 app.utils.price.round_price）。

    v145：原本 >1000→0位、>10→2位、其餘 4 位，與 K 線圖不一致
    （ADA $0.25 圖上顯示 6 位 0.251700，舊版只給 4 位）。改用共用 helper。
    """
    from app.utils.price import round_price
    return round_price(p)


def compute_laddered_entries(
    df: pd.DataFrame,
    direction: Literal["long", "short", "both"] = "both",
    regime: str = "unknown",
    regime_confidence: float = 0.0,
    smc_long_entry: Optional[float] = None,
    smc_short_entry: Optional[float] = None,
    n_tranches: int = 3,
    timeframe_str: Optional[str] = None,
    confidence_label: Optional[str] = None,
) -> dict:
    """後端算分批進場價，全部從 indicator values 取，禁止推算。

    Args:
        df: OHLCV DataFrame（用最後一根當 current）
        direction: 要計算哪些方向
        regime: 從 chart_state.currentRegime 來，6 種之一
        regime_confidence: 0.0~1.0，< 0.5 直接跳過
        smc_long_entry / smc_short_entry: 若 SMC 已給 entry 則優先當第一檔
        n_tranches: 預設 3（本版固定，未來再做動態）
        timeframe_str: v104 — "15m"/"1h"/"4h"/"1d"/... 讓 ATR 倍數隨 timeframe 自適應
        confidence_label: v104 — "high"/"medium"/"low" 影響 TP 倍數（高信心拉遠賺更多）

    Returns:
        dict — 含 enabled / long_entries / short_entries / SL / TP / 加權均價 / regime / strategy / warning。
        若 enabled = False，long_entries / short_entries 會是空 list。
    """
    if df is None or df.empty or len(df) < 30:
        return {"enabled": False, "warning": "資料不足（< 30 根 K 線），無法計算 ladder", "regime_used": regime, "n_tranches": n_tranches}

    # confidence 閾值 — 沿用既有 regimeWarning 邏輯
    if regime_confidence < _CONFIDENCE_THRESHOLD or regime not in _REGIME_RATIO_MAP:
        # v104.2：即使 ladder 不啟用，仍提供建議的 ATR 倍數給 LLM 用（避免它自己拍 1×ATR 太緊）
        sl_mult_hint, tp_mult_hint = _get_atr_mults(timeframe_str, regime, confidence_label)
        return {
            "enabled": False,
            "warning": f"regime={regime} confidence={regime_confidence:.2f} < {_CONFIDENCE_THRESHOLD}，跳過分批進場（建議單一進場 + 小倉位試單）",
            "regime_used": regime,
            "regime_confidence": regime_confidence,
            "n_tranches": n_tranches,
            # 提示倍數（即使 ladder disabled，LLM 自己定 entry 時應引用）
            "sl_mult_hint": sl_mult_hint,
            "tp_mult_hint": tp_mult_hint,
            "timeframe_used": timeframe_str,
            "confidence_label_used": confidence_label,
            "fallback_guidance": (
                f"ladder 雖 disabled，建議 LLM 自定 entry 時用 SL ≈ {sl_mult_hint} × {timeframe_str} ATR、"
                f"TP ≈ {tp_mult_hint} × {timeframe_str} ATR（regime/timeframe-aware 倍數，比 1×ATR 合理）"
            ),
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
        # v105.2 Bug A：clamp long entries ≤ current_price（避免「等漲了才買」）
        entries = _clamp_to_current_price(entries, current_price, "long")
        long_entries = _normalize_ratios(entries)
        # v105.1：強制最小 spacing 避免 BB/Donchian 重合（< 0.5×ATR）
        long_entries = _enforce_minimum_spacing(long_entries, atr, "long")
        # v105.2 Bug B：依最終位置重新分配 ratios（倒金字塔接刀越深倉位越重）
        long_entries = _reassign_ratios_by_position(long_entries, list(cfg["ratios"]), "long")
        missing_set.update(missing)
    if direction in ("short", "both") and cfg["short_ok"]:
        entries, missing = _build_short_ladder(df, regime, current_price, atr, smc_short_entry)
        entries = _clamp_to_current_price(entries, current_price, "short")
        short_entries = _normalize_ratios(entries)
        short_entries = _enforce_minimum_spacing(short_entries, atr, "short")
        short_entries = _reassign_ratios_by_position(short_entries, list(cfg["ratios"]), "short")
        missing_set.update(missing)

    # v104 Fix A：依 timeframe + regime + 信心算 ATR 倍數（含 cap + feature flag）
    sl_mult, tp_mult = _get_atr_mults(timeframe_str, regime, confidence_label)

    # SL / TP（用自適應倍數）
    long_sl, long_tp, long_rr = _compute_sl_tp(long_entries, atr, "long", sl_mult, tp_mult) if long_entries else (None, None, None)
    short_sl, short_tp, short_rr = _compute_sl_tp(short_entries, atr, "short", sl_mult, tp_mult) if short_entries else (None, None, None)

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
        # v104 Fix A：透明化使用的 ATR 倍數（給 LLM / UI 引用）
        "sl_mult_used": sl_mult,
        "tp_mult_used": tp_mult,
        "timeframe_used": timeframe_str,
        "confidence_label_used": confidence_label,
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
