"""阿斯拉量化系統 — 因子分析引擎 v2

功能：
1. 單因子 IC 分析（全域 + 近期窗口）
2. Alpha Decay 計算（rolling IC 曲線、半衰期）
3. 衍生因子自動生成（動量、偏離、交叉、背離等）
4. 因子相關性矩陣（冗餘排除）
5. 雙因子組合 IC
6. 分位數分析（單調性驗證）
7. 多因子加權合成信號

時間序列 IC（因子值 vs 未來 N 期報酬的相關性），
適用於加密貨幣單標的分析。
"""

import numpy as np
import pandas as pd
from typing import Optional

from app.core.indicators import registry

# ─── 適合做因子掃描的指標 ID ────────────────────────
SCANNABLE_INDICATORS: list[str] = [
    "rsi", "macd", "adx", "bb", "obv", "atr", "bias", "roc",
    "rel_vol", "stochrsi", "sma", "ema", "vwap", "vol_switch",
    "donchian", "keltner", "supertrend", "cvd", "hv", "psar", "poc",
]

# ─── 近期窗口映射（根據 K 線級別） ──────────────────
_RECENT_BARS: dict[str, int] = {
    "1m": 200, "3m": 200, "5m": 200, "15m": 200,
    "30m": 200, "1h": 200, "2h": 200,
    "4h": 200, "6h": 150, "8h": 120,
    "1d": 120, "3d": 80, "1w": 60, "1M": 40,
}

_MIN_SAMPLES = 60


def _to_array(values: list) -> np.ndarray:
    return np.array([
        v if v is not None and not (isinstance(v, float) and np.isnan(v))
        else np.nan for v in values
    ], dtype=float)


def _spearman_ic(f: np.ndarray, r: np.ndarray) -> float:
    from scipy.stats import spearmanr
    ic, _ = spearmanr(f, r)
    return float(ic) if not np.isnan(ic) else 0.0


def _compute_future_returns(closes: np.ndarray, forward_periods: list[int]) -> dict[int, np.ndarray]:
    n = len(closes)
    result = {}
    for fp in forward_periods:
        if fp >= n - 10:
            continue
        fwd = np.full(n, np.nan)
        safe = closes.copy()
        safe[safe == 0] = np.nan
        fwd[:n - fp] = (closes[fp:] - closes[:n - fp]) / safe[:n - fp]
        result[fp] = fwd
    return result


# ═══════════════════════════════════════════════════════
#  衍生因子生成
# ═══════════════════════════════════════════════════════

def compute_derived_factors(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """從已有指標衍生額外因子。"""
    n = len(df)
    derived: dict[str, np.ndarray] = {}
    close = df["close"].values.astype(float)

    def _safe_calc(indicator_id: str) -> Optional[dict[str, list]]:
        try:
            return registry.calculate(indicator_id, df)
        except Exception:
            return None

    def _first_series(calc: Optional[dict]) -> Optional[np.ndarray]:
        if not calc:
            return None
        vals = list(calc.values())[0]
        return _to_array(vals)

    def _roc(arr: np.ndarray, period: int) -> np.ndarray:
        result = np.full_like(arr, np.nan)
        shifted = np.roll(arr, period)
        shifted[:period] = np.nan
        safe = shifted.copy()
        safe[safe == 0] = np.nan
        result = (arr - shifted) / np.abs(safe)
        return result

    # RSI 動量
    rsi = _first_series(_safe_calc("rsi"))
    if rsi is not None:
        derived["rsi_momentum"] = _roc(rsi, 5)

    # MACD 加速度
    macd_calc = _safe_calc("macd")
    if macd_calc and "MACD_Hist" in macd_calc:
        hist = _to_array(macd_calc["MACD_Hist"])
        derived["macd_accel"] = _roc(hist, 3)

    # ADX 變化
    adx = _first_series(_safe_calc("adx"))
    if adx is not None:
        derived["adx_delta"] = _roc(adx, 5)

    # BB %B
    bb_calc = _safe_calc("bb")
    if bb_calc and "BB_Upper" in bb_calc and "BB_Lower" in bb_calc:
        upper = _to_array(bb_calc["BB_Upper"])
        lower = _to_array(bb_calc["BB_Lower"])
        width = upper - lower
        width[width == 0] = np.nan
        derived["bb_pctb"] = (close - lower) / width

        # BB 帶寬變化率
        bw = upper - lower
        derived["bb_width_change"] = _roc(bw, 10)

    # 價格 vs SMA (ATR 標準化)
    sma_arr = _first_series(_safe_calc("sma"))
    atr_arr = _first_series(_safe_calc("atr"))
    if sma_arr is not None and atr_arr is not None:
        safe_atr = atr_arr.copy()
        safe_atr[safe_atr == 0] = np.nan
        derived["price_vs_sma"] = (close - sma_arr) / safe_atr

    # 價格 vs VWAP (ATR 標準化)
    vwap_arr = _first_series(_safe_calc("vwap"))
    if vwap_arr is not None and atr_arr is not None:
        safe_atr = atr_arr.copy()
        safe_atr[safe_atr == 0] = np.nan
        derived["price_vs_vwap"] = (close - vwap_arr) / safe_atr

    # OBV 背離
    obv_arr = _first_series(_safe_calc("obv"))
    if obv_arr is not None:
        obv_roc = _roc(obv_arr, 10)
        price_roc = _roc(close, 10)
        derived["obv_divergence"] = obv_roc - price_roc

    # 相對成交量趨勢
    rel_vol = _first_series(_safe_calc("rel_vol"))
    if rel_vol is not None:
        kernel = np.ones(5) / 5
        padded = np.concatenate([np.full(4, np.nan), rel_vol])
        convolved = np.convolve(rel_vol, kernel, mode="full")[:n]
        convolved[:4] = np.nan
        derived["vol_ratio_trend"] = convolved

    # RSI - StochRSI 差距
    stoch_calc = _safe_calc("stochrsi")
    if rsi is not None and stoch_calc and "StochRSI_K" in stoch_calc:
        stoch_k = _to_array(stoch_calc["StochRSI_K"])
        derived["rsi_stoch_spread"] = rsi - stoch_k

    # Keltner 位置
    kelt_calc = _safe_calc("keltner")
    if kelt_calc and "Keltner_Mid" in kelt_calc and "Keltner_Upper" in kelt_calc:
        mid = _to_array(kelt_calc["Keltner_Mid"])
        upper = _to_array(kelt_calc["Keltner_Upper"])
        band = upper - mid
        band[band == 0] = np.nan
        derived["keltner_position"] = (close - mid) / band

    # ATR 比率 (短/長)
    if atr_arr is not None:
        atr_calc_long = registry.calculate("atr", df, {"period": 50})
        if atr_calc_long:
            atr_long = _to_array(list(atr_calc_long.values())[0])
            atr_long[atr_long == 0] = np.nan
            derived["atr_ratio"] = atr_arr / atr_long

    # CVD 動量
    cvd_arr = _first_series(_safe_calc("cvd"))
    if cvd_arr is not None:
        derived["cvd_momentum"] = _roc(cvd_arr, 10)

    # Supertrend 距離 (ATR 標準化)
    st_calc = _safe_calc("supertrend")
    if st_calc and "Supertrend" in st_calc and atr_arr is not None:
        st_line = _to_array(st_calc["Supertrend"])
        safe_atr = atr_arr.copy()
        safe_atr[safe_atr == 0] = np.nan
        derived["supertrend_dist"] = (close - st_line) / safe_atr

    # HV 比率 (短/長)
    hv_short = registry.calculate("hv", df, {"period": 10})
    hv_long = registry.calculate("hv", df, {"period": 30})
    if hv_short and hv_long:
        hv_s = _to_array(list(hv_short.values())[0])
        hv_l = _to_array(list(hv_long.values())[0])
        hv_l[hv_l == 0] = np.nan
        derived["hv_ratio"] = hv_s / hv_l

    return derived


# ═══════════════════════════════════════════════════════
#  核心 IC 計算
# ═══════════════════════════════════════════════════════

def _collect_all_factors(
    df: pd.DataFrame,
    indicator_ids: list[str],
    include_derived: bool = True,
) -> dict[str, np.ndarray]:
    """收集所有因子（原始 + 衍生）。"""
    factors: dict[str, np.ndarray] = {}

    for ind_id in indicator_ids:
        try:
            calc = registry.calculate(ind_id, df)
            if not calc:
                continue
            for series_name, values in calc.items():
                key = f"{ind_id}_{series_name}" if series_name != ind_id else ind_id
                factors[key] = _to_array(values)
        except Exception:
            pass

    if include_derived:
        derived = compute_derived_factors(df)
        factors.update(derived)

    return factors


def _ic_for_slice(
    factor: np.ndarray,
    fwd: np.ndarray,
    start: int,
    end: int,
) -> Optional[float]:
    """對指定切片計算 Spearman IC。"""
    f_slice = factor[start:end]
    r_slice = fwd[start:end]
    valid = ~np.isnan(f_slice) & ~np.isnan(r_slice)
    if np.sum(valid) < 15:
        return None
    try:
        return _spearman_ic(f_slice[valid], r_slice[valid])
    except Exception:
        return None


# ═══════════════════════════════════════════════════════
#  Alpha Decay
# ═══════════════════════════════════════════════════════

def _compute_decay_curve(
    factor: np.ndarray,
    fwd: np.ndarray,
    n_windows: int = 6,
) -> dict:
    """計算 Alpha Decay 曲線。

    把數據分成 n_windows 個等寬窗口（50% 重疊），
    從最舊到最新各算一次 IC，觀察衰減趨勢。
    """
    n = len(factor)
    window_size = max(n // (n_windows + 1), _MIN_SAMPLES)
    step = max((n - window_size) // max(n_windows - 1, 1), 1)

    ic_curve: list[Optional[float]] = []
    for i in range(n_windows):
        start = i * step
        end = min(start + window_size, n)
        if end - start < _MIN_SAMPLES:
            break
        ic = _ic_for_slice(factor, fwd, start, end)
        ic_curve.append(ic)

    valid_ics = [v for v in ic_curve if v is not None]
    if len(valid_ics) < 3:
        return {"curve": ic_curve, "half_life": None, "trend": "unknown", "cv": None}

    abs_ics = [abs(v) for v in valid_ics]
    recent_mean = np.mean(abs_ics[-2:])
    early_mean = np.mean(abs_ics[:2])

    if recent_mean > early_mean * 1.15:
        trend = "rising"
    elif recent_mean < early_mean * 0.7:
        trend = "decaying"
    else:
        trend = "stable"

    # 半衰期：從最新窗口的 |IC| 降到一半需要幾個窗口
    half_life = None
    if abs_ics[-1] > 0.01:
        target = abs_ics[-1] / 2
        for idx in range(len(abs_ics) - 2, -1, -1):
            if abs_ics[idx] <= target:
                half_life = len(abs_ics) - 1 - idx
                break
        if half_life is None:
            half_life = len(abs_ics)  # 全部都高於半值 → 非常穩定

    ic_mean = np.mean(valid_ics)
    ic_std = np.std(valid_ics)
    cv = round(float(ic_std / abs(ic_mean)), 4) if abs(ic_mean) > 1e-6 else 999.0

    return {
        "curve": [round(v, 4) if v is not None else None for v in ic_curve],
        "half_life": half_life,
        "trend": trend,
        "cv": round(cv, 4),
    }


# ═══════════════════════════════════════════════════════
#  分位數分析
# ═══════════════════════════════════════════════════════

def _quantile_analysis(
    factor: np.ndarray,
    fwd: np.ndarray,
    n_quantiles: int = 5,
) -> Optional[dict]:
    """把因子值分成 n 組，看每組的平均未來報酬，並回傳每組的因子值區間。"""
    valid = ~np.isnan(factor) & ~np.isnan(fwd)
    if np.sum(valid) < n_quantiles * 10:
        return None

    f_valid = factor[valid]
    r_valid = fwd[valid]

    try:
        quantile_edges = np.percentile(f_valid, np.linspace(0, 100, n_quantiles + 1))
        quantile_edges[-1] += 1e-10

        quantile_returns = []
        quantile_ranges = []
        for q in range(n_quantiles):
            lo = float(quantile_edges[q])
            hi = float(quantile_edges[q + 1])
            quantile_ranges.append({
                "low": round(lo, 4),
                "high": round(hi, 4),
                "label": f"Q{q+1}",
            })
            mask = (f_valid >= quantile_edges[q]) & (f_valid < quantile_edges[q + 1])
            if np.sum(mask) < 5:
                quantile_returns.append(None)
                continue
            avg_ret = float(np.mean(r_valid[mask]))
            quantile_returns.append(round(avg_ret * 100, 4))

        valid_returns = [r for r in quantile_returns if r is not None]
        if len(valid_returns) < 3:
            return None

        is_monotonic = all(
            valid_returns[i] <= valid_returns[i + 1]
            for i in range(len(valid_returns) - 1)
        ) or all(
            valid_returns[i] >= valid_returns[i + 1]
            for i in range(len(valid_returns) - 1)
        )

        spread = valid_returns[-1] - valid_returns[0] if len(valid_returns) >= 2 else 0

        # 找最佳進場區間
        best_q_idx = None
        best_ret = -999.0
        for i, r in enumerate(quantile_returns):
            if r is not None and r > best_ret:
                best_ret = r
                best_q_idx = i

        best_range = None
        entry_suggestion = None
        if best_q_idx is not None:
            best_range = quantile_ranges[best_q_idx]
            lo = best_range["low"]
            hi = best_range["high"]
            if best_q_idx == n_quantiles - 1:
                entry_suggestion = f"> {lo:.2f}"
            elif best_q_idx == 0:
                entry_suggestion = f"< {hi:.2f}"
            else:
                entry_suggestion = f"{lo:.2f} ~ {hi:.2f}"

        return {
            "quantile_returns_pct": quantile_returns,
            "quantile_ranges": quantile_ranges,
            "is_monotonic": is_monotonic,
            "spread_pct": round(spread, 4),
            "n_quantiles": n_quantiles,
            "best_quantile": {
                "index": best_q_idx,
                "range": best_range,
                "return_pct": best_ret,
                "entry_suggestion": entry_suggestion,
            } if best_q_idx is not None else None,
        }
    except Exception:
        return None


# ═══════════════════════════════════════════════════════
#  雙因子組合 IC
# ═══════════════════════════════════════════════════════

def _combo_ic(
    factors: dict[str, np.ndarray],
    fwd: np.ndarray,
    positive_keys: list[str],
    negative_keys: list[str],
    max_per_group: int = 3,
) -> dict:
    """對 top 因子做三類組合：正正、負負、正負對沖。

    正正：Z(A) + Z(B) → 同方向看多增強
    負負：Z(A) + Z(B) → 同方向看空增強
    正負對沖：Z(正) - Z(負) → 多空信號合併
    """

    def _zscore(arr: np.ndarray) -> np.ndarray:
        valid = ~np.isnan(arr)
        if np.sum(valid) < 20:
            return arr
        m = np.nanmean(arr)
        s = np.nanstd(arr)
        if s < 1e-10:
            return arr - m
        return (arr - m) / s

    def _calc_pairs(keys_a, keys_b, mode="add"):
        results = []
        seen = set()
        for ka in keys_a:
            for kb in keys_b:
                if ka == kb:
                    continue
                pair = tuple(sorted([ka, kb]))
                if pair in seen:
                    continue
                seen.add(pair)
                if ka not in factors or kb not in factors:
                    continue
                za = _zscore(factors[ka])
                zb = _zscore(factors[kb])
                combo = za + zb if mode == "add" else za - zb
                valid = ~np.isnan(combo) & ~np.isnan(fwd)
                if np.sum(valid) < 30:
                    continue
                try:
                    ic = _spearman_ic(combo[valid], fwd[valid])
                    results.append({
                        "factor_a": ka,
                        "factor_b": kb,
                        "combo_ic": round(ic, 4),
                        "combo_abs_ic": round(abs(ic), 4),
                    })
                except Exception:
                    pass
        results.sort(key=lambda x: x["combo_abs_ic"], reverse=True)
        return results[:max_per_group]

    return {
        "positive_combos": _calc_pairs(positive_keys, positive_keys, "add"),
        "negative_combos": _calc_pairs(negative_keys, negative_keys, "add"),
        "hedge_combos": _calc_pairs(positive_keys, negative_keys, "subtract"),
    }


# ═══════════════════════════════════════════════════════
#  因子狀態評級
# ═══════════════════════════════════════════════════════

def _rate_factor(
    abs_ic: float,
    trend: str,
    cv: Optional[float],
    half_life: Optional[int],
) -> dict:
    """綜合評級因子狀態。"""
    if abs_ic < 0.02:
        status = "inactive"
        stars = 0
        label = "✗ 無效"
    elif trend == "decaying":
        status = "decaying"
        stars = 1
        label = "↓ 衰退中"
    elif trend == "rising" and abs_ic > 0.05:
        status = "rising"
        stars = 2
        label = "★★ 升溫中"
        if abs_ic > 0.08:
            stars = 3
            label = "★★★ 強勢升溫"
    elif abs_ic > 0.1 and (cv is None or cv < 0.5):
        status = "strong"
        stars = 3
        label = "★★★ Strong"
    elif abs_ic > 0.05:
        if cv is not None and cv < 0.6:
            status = "validated"
            stars = 2
            label = "★★ Validated"
        else:
            status = "unstable"
            stars = 1
            label = "★ 不穩定"
    else:
        status = "weak"
        stars = 1
        label = "★ Weak"

    # 半衰期極短 → 降級
    if half_life is not None and half_life <= 1 and stars > 1:
        stars = max(stars - 1, 1)
        label = f"↓ 快速衰退 ({label})"
        status = "decaying"

    confidence = "高" if abs_ic > 0.08 and (cv is None or cv < 0.4) else \
                 "中" if abs_ic > 0.04 else "低"

    return {"status": status, "stars": stars, "label": label, "confidence": confidence}


# ═══════════════════════════════════════════════════════
#  主入口：因子掃描
# ═══════════════════════════════════════════════════════

def run_factor_scan(
    df: pd.DataFrame,
    timeframe: str = "4h",
    indicator_ids: Optional[list[str]] = None,
    include_derived: bool = True,
    top_n: int = 5,
    forward_period: int = 5,
) -> dict:
    """一鍵因子掃描：計算所有因子的近期 IC、長期 IC、Alpha Decay、
    分位數驗證、雙因子組合 IC。

    Args:
        df: OHLCV DataFrame
        timeframe: K 線級別（用於決定近期窗口大小）
        indicator_ids: 要掃描的指標 ID（預設 SCANNABLE_INDICATORS）
        include_derived: 是否包含衍生因子
        top_n: 正/負相關各取 top N
        forward_period: 未來報酬計算週期

    Returns:
        完整掃描結果（正相關 TOP、負相關 TOP、組合、警告等）
    """
    if indicator_ids is None:
        indicator_ids = SCANNABLE_INDICATORS

    n = len(df)
    if n < _MIN_SAMPLES + forward_period:
        return {"status": "error", "message": f"數據不足（{n} 根），至少需要 {_MIN_SAMPLES + forward_period} 根"}

    recent_bars = _RECENT_BARS.get(timeframe, 200)
    recent_bars = min(recent_bars, n)
    closes = df["close"].values.astype(float)

    # 計算全域和近期的未來報酬
    fwd_full = _compute_future_returns(closes, [forward_period]).get(forward_period)
    if fwd_full is None:
        return {"status": "error", "message": "無法計算未來報酬"}

    recent_start = max(0, n - recent_bars)
    fwd_recent = fwd_full.copy()
    fwd_recent[:recent_start] = np.nan

    # 收集所有因子
    factors = _collect_all_factors(df, indicator_ids, include_derived)
    if not factors:
        return {"status": "error", "message": "沒有可用的因子數據"}

    # 判斷市場體制
    regime = _detect_regime(df)

    # 逐因子計算
    factor_results = {}
    for key, arr in factors.items():
        # 全域 IC
        valid_full = ~np.isnan(arr) & ~np.isnan(fwd_full)
        if np.sum(valid_full) < 30:
            continue
        try:
            ic_full = _spearman_ic(arr[valid_full], fwd_full[valid_full])
        except Exception:
            continue

        # 近期 IC
        valid_recent = ~np.isnan(arr) & ~np.isnan(fwd_recent)
        samples_recent = int(np.sum(valid_recent))
        if samples_recent >= 30:
            try:
                ic_recent = _spearman_ic(arr[valid_recent], fwd_recent[valid_recent])
            except Exception:
                ic_recent = ic_full
        else:
            ic_recent = ic_full
            samples_recent = int(np.sum(valid_full))

        # Alpha Decay
        decay = _compute_decay_curve(arr, fwd_full, n_windows=6)

        # 評級
        rating = _rate_factor(abs(ic_recent), decay["trend"], decay["cv"], decay["half_life"])

        factor_results[key] = {
            "ic_recent": round(ic_recent, 4),
            "ic_full": round(ic_full, 4),
            "abs_ic_recent": round(abs(ic_recent), 4),
            "samples": samples_recent,
            "decay": decay,
            "rating": rating,
        }

    if not factor_results:
        return {"status": "error", "message": "所有因子計算失敗或樣本不足"}

    # 分正/負相關排序
    positive = sorted(
        [(k, v) for k, v in factor_results.items() if v["ic_recent"] > 0.02],
        key=lambda x: x[1]["ic_recent"], reverse=True,
    )[:top_n]

    negative = sorted(
        [(k, v) for k, v in factor_results.items() if v["ic_recent"] < -0.02],
        key=lambda x: x[1]["ic_recent"],
    )[:top_n]

    # 分位數分析（對所有 TOP 正相關和 TOP 負相關因子都做）
    quantile_results = {}
    for idx, (k, _v) in enumerate(positive[:3]):
        if k in factors:
            qa = _quantile_analysis(factors[k], fwd_full)
            if qa:
                quantile_results[f"positive_{idx+1}"] = {"factor": k, **qa}
    for idx, (k, _v) in enumerate(negative[:3]):
        if k in factors:
            qa = _quantile_analysis(factors[k], fwd_full)
            if qa:
                quantile_results[f"negative_{idx+1}"] = {"factor": k, **qa}

    # 雙因子組合 IC（正正/負負/正負對沖分開）
    pos_keys = [k for k, _ in positive[:5]]
    neg_keys = [k for k, _ in negative[:5]]
    combo_results = _combo_ic(factors, fwd_full, pos_keys, neg_keys, max_per_group=3)

    # 高相關警告
    high_corr = _find_high_correlations(factors, list(factor_results.keys()))

    # 組裝結果
    def _fmt(items):
        return [
            {
                "factor": k,
                "ic_recent": v["ic_recent"],
                "ic_full": v["ic_full"],
                "decay_trend": v["decay"]["trend"],
                "decay_curve": v["decay"]["curve"],
                "half_life": v["decay"]["half_life"],
                "confidence": v["rating"]["confidence"],
                "status_label": v["rating"]["label"],
                "stars": v["rating"]["stars"],
                "samples": v["samples"],
            }
            for k, v in items
        ]

    return {
        "status": "success",
        "symbol": None,  # 由呼叫方填入
        "timeframe": timeframe,
        "total_bars": n,
        "recent_bars": recent_bars,
        "forward_period": forward_period,
        "regime": regime,
        "total_factors_scanned": len(factor_results),
        "positive_top": _fmt(positive),
        "negative_top": _fmt(negative),
        "combo_top": combo_results,
        "quantile_analysis": quantile_results,
        "high_correlation_warnings": high_corr,
        "effective_count": sum(
            1 for v in factor_results.values()
            if v["abs_ic_recent"] > 0.05
        ),
    }


def _detect_regime(df: pd.DataFrame) -> dict:
    """快速判斷市場體制。"""
    try:
        adx_calc = registry.calculate("adx", df)
        atr_calc = registry.calculate("atr", df)
        adx_val = None
        atr_val = None
        if adx_calc:
            arr = _to_array(list(adx_calc.values())[0])
            valid = arr[~np.isnan(arr)]
            adx_val = float(valid[-1]) if len(valid) > 0 else None
        if atr_calc:
            arr = _to_array(list(atr_calc.values())[0])
            valid = arr[~np.isnan(arr)]
            atr_val = float(valid[-1]) if len(valid) > 0 else None

        if adx_val is None:
            return {"label": "未知", "adx": None, "atr": None}

        if adx_val > 25:
            label = "趨勢市場"
        elif adx_val < 20:
            label = "盤整市場"
        else:
            label = "過渡期"

        return {"label": label, "adx": round(adx_val, 1) if adx_val else None,
                "atr": round(atr_val, 4) if atr_val else None}
    except Exception:
        return {"label": "未知", "adx": None, "atr": None}


def _find_high_correlations(
    factors: dict[str, np.ndarray],
    keys: list[str],
    threshold: float = 0.75,
) -> list[dict]:
    """找出高相關因子對。"""
    warnings = []
    checked = set()
    key_list = [k for k in keys if k in factors]

    for i in range(len(key_list)):
        for j in range(i + 1, len(key_list)):
            pair = (key_list[i], key_list[j])
            if pair in checked:
                continue
            checked.add(pair)

            a, b = factors[pair[0]], factors[pair[1]]
            valid = ~np.isnan(a) & ~np.isnan(b)
            if np.sum(valid) < 30:
                continue
            try:
                c = float(np.corrcoef(a[valid], b[valid])[0, 1])
                if abs(c) > threshold:
                    warnings.append({
                        "factor_a": pair[0],
                        "factor_b": pair[1],
                        "correlation": round(c, 3),
                    })
            except Exception:
                pass

    warnings.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return warnings[:10]


# ═══════════════════════════════════════════════════════
#  向下相容：保留原有函式
# ═══════════════════════════════════════════════════════

def compute_factor_ic(
    df: pd.DataFrame,
    indicator_ids: list[str],
    forward_periods: list[int] = None,
    ic_method: str = "rank",
) -> dict:
    """計算每個指標因子的 IC（向下相容介面）。"""
    if forward_periods is None:
        forward_periods = [1, 3, 5, 10, 20]

    closes = df["close"].values.astype(float)
    future_returns = _compute_future_returns(closes, forward_periods)

    if not future_returns:
        return {"status": "error", "message": "數據不足以計算未來報酬"}

    results = {}
    for ind_id in indicator_ids:
        try:
            calc = registry.calculate(ind_id, df)
            if not calc:
                continue
            for series_name, values in calc.items():
                factor_arr = _to_array(values)
                key = f"{ind_id}_{series_name}" if series_name != ind_id else ind_id
                ic_values = {}

                for fp, fwd in future_returns.items():
                    valid = ~np.isnan(factor_arr) & ~np.isnan(fwd)
                    if np.sum(valid) < 30:
                        continue

                    if ic_method == "rank":
                        ic = _spearman_ic(factor_arr[valid], fwd[valid])
                        p_value = None
                    else:
                        ic = float(np.corrcoef(factor_arr[valid], fwd[valid])[0, 1])
                        p_value = None

                    ic_values[f"fwd_{fp}"] = {
                        "ic": round(ic, 4),
                        "abs_ic": round(abs(ic), 4),
                        "p_value": p_value,
                        "samples": int(np.sum(valid)),
                    }

                if ic_values:
                    decay = _compute_decay_curve(
                        factor_arr,
                        list(future_returns.values())[0],
                    )
                    best_period = max(ic_values.items(), key=lambda x: x[1]["abs_ic"])
                    rating = _rate_factor(
                        best_period[1]["abs_ic"], decay["trend"], decay["cv"], decay["half_life"],
                    )
                    results[key] = {
                        "indicator": ind_id,
                        "series": series_name,
                        "ic_by_period": ic_values,
                        "best_period": best_period[0],
                        "best_ic": best_period[1]["ic"],
                        "ic_stability": decay["cv"] or 0.0,
                        "predictive_power": rating["label"],
                        "decay": decay,
                    }
        except Exception as e:
            results[ind_id] = {"indicator": ind_id, "error": str(e)}

    if not results:
        return {"status": "error", "message": "沒有可分析的因子數據"}

    ranked = sorted(
        [(k, v) for k, v in results.items() if "best_ic" in v],
        key=lambda x: abs(x[1].get("best_ic", 0)),
        reverse=True,
    )

    return {
        "status": "success",
        "factors": results,
        "ranking": [{"factor": k, "best_ic": v["best_ic"], "power": v["predictive_power"]} for k, v in ranked[:10]],
        "total_factors_analyzed": len(results),
    }


def compute_factor_correlation(
    df: pd.DataFrame,
    indicator_ids: list[str],
) -> dict:
    """計算因子之間的相關性矩陣（向下相容）。"""
    factors = _collect_all_factors(df, indicator_ids, include_derived=False)

    if len(factors) < 2:
        return {"status": "error", "message": "至少需要 2 個因子才能計算相關性"}

    warnings = _find_high_correlations(factors, list(factors.keys()), threshold=0.7)

    return {
        "status": "success",
        "factors": list(factors.keys()),
        "high_correlation_pairs": warnings,
        "recommendation": (
            "以下因子高度相關，建議只保留預測力最強的：" +
            ", ".join(f"{p['factor_a']}↔{p['factor_b']}" for p in warnings[:5])
        ) if warnings else "各因子相關性低，可同時使用",
    }


def compute_composite_signal(
    df: pd.DataFrame,
    factor_weights: dict[str, float],
    normalize: bool = True,
) -> dict:
    """多因子加權合成信號（向下相容）。"""
    n = len(df)
    weighted_sum = np.zeros(n)
    total_weight = 0

    for ind_id, weight in factor_weights.items():
        try:
            calc = registry.calculate(ind_id, df)
            if not calc:
                continue
            first_series = list(calc.values())[0]
            arr = _to_array(first_series)

            if normalize:
                valid = ~np.isnan(arr)
                if np.sum(valid) > 20:
                    mean = np.nanmean(arr)
                    std = np.nanstd(arr)
                    if std > 0:
                        arr = (arr - mean) / std

            valid = ~np.isnan(arr)
            weighted_sum[valid] += arr[valid] * weight
            total_weight += abs(weight)
        except Exception:
            pass

    if total_weight == 0:
        return {"status": "error", "message": "沒有有效因子數據"}

    signal = weighted_sum / total_weight
    current = float(signal[-1]) if not np.isnan(signal[-1]) else 0

    return {
        "status": "success",
        "current_signal": round(current, 4),
        "signal_strength": _signal_strength(current),
        "signal_direction": "看多" if current > 0.5 else "看空" if current < -0.5 else "中性",
        "recent_5": [round(float(s), 4) for s in signal[-5:] if not np.isnan(s)],
    }


def _signal_strength(signal: float) -> str:
    if signal > 1.5:
        return "極強看多"
    elif signal > 0.5:
        return "看多"
    elif signal > -0.5:
        return "中性"
    elif signal > -1.5:
        return "看空"
    return "極強看空"
