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

from loguru import logger

from app.core.indicators import registry

# ─── 適合做因子掃描的指標 ID ────────────────────────
SCANNABLE_INDICATORS: list[str] = [
    "rsi", "macd", "adx", "bb", "obv", "atr", "bias", "roc",
    "rel_vol", "stochrsi", "sma", "ema", "vwap", "vol_switch",
    "donchian", "keltner", "supertrend", "cvd", "hv", "psar", "poc",
    "vol_squeeze", "rsi_divergence", "vol_divergence", "macd_divergence",
    "leading_composite", "mtf_mss",
]

# ─── 近期窗口映射（根據 K 線級別） ──────────────────
_RECENT_BARS: dict[str, int] = {
    "1m": 500, "3m": 400, "5m": 400, "15m": 350,
    "30m": 300, "1h": 300, "2h": 250,
    "4h": 300, "6h": 200, "8h": 180,
    "1d": 180, "3d": 100, "1w": 80, "1M": 50,
}

_MIN_SAMPLES_BY_TF: dict[str, int] = {
    "1m": 500, "3m": 500, "5m": 500, "15m": 400,
    "30m": 300, "1h": 250, "2h": 200,
    "4h": 200, "6h": 150, "8h": 150,
    "1d": 120, "3d": 80, "1w": 60, "1M": 40,
}
_MIN_SAMPLES = 60  # fallback


def _load_prediction_feedback(symbol: Optional[str] = None) -> dict[str, float]:
    """從預測追蹤器載入因子歷史表現，回傳 {indicator_name: accuracy_modifier}。

    modifier > 0 表示歷史預測準確，boost 因子權重
    modifier < 0 表示歷史預測不準確，降低因子權重
    modifier == 0 表示無數據
    """
    try:
        from app.core.prediction_tracker import prediction_tracker
        stats = prediction_tracker.get_stats(symbol=symbol, days=60)
        if stats.get("total", 0) < 8:
            return {}

        ind_perf = stats.get("indicator_performance", {})
        modifiers: dict[str, float] = {}
        for ind_name, perf in ind_perf.items():
            samples = perf.get("samples", 0)
            if samples < 3:
                continue
            win_rate = perf.get("win_rate", 50)
            modifiers[ind_name.lower()] = round((win_rate - 50) / 100, 3)
        return modifiers
    except Exception as e:
        logger.debug(f"載入預測反饋失敗（不影響掃描）: {e}")
        return {}


def _to_array(values: list) -> np.ndarray:
    return np.array([
        v if v is not None and not (isinstance(v, float) and np.isnan(v))
        else np.nan for v in values
    ], dtype=float)


def _spearman_ic(f: np.ndarray, r: np.ndarray) -> tuple[float, float]:
    """回傳 (ic, p_value)。p_value 越小越顯著。"""
    from scipy.stats import spearmanr
    ic, pval = spearmanr(f, r)
    ic = float(ic) if not np.isnan(ic) else 0.0
    pval = float(pval) if not np.isnan(pval) else 1.0
    return ic, pval


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
    df = df.copy()
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
        width = np.array(upper - lower, copy=True)
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
        convolved = np.convolve(rel_vol, kernel, mode="full")[:n].copy()
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
        band = np.array(upper - mid, copy=True)
        band[band == 0] = np.nan
        derived["keltner_position"] = (close - mid) / band

    # ATR 比率 (短/長)
    if atr_arr is not None:
        atr_calc_long = registry.calculate("atr", df, {"period": 50})
        if atr_calc_long:
            atr_long = _to_array(list(atr_calc_long.values())[0]).copy()
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
        hv_l = _to_array(list(hv_long.values())[0]).copy()
        hv_l[hv_l == 0] = np.nan
        derived["hv_ratio"] = hv_s / hv_l

    # ── 多參數版本指標 ────────────────────────
    # RSI 不同週期
    for period in [7, 21]:
        rsi_alt = _first_series(registry.calculate("rsi", df, {"period": period}))
        if rsi_alt is not None:
            derived[f"rsi_{period}"] = rsi_alt

    # SMA/EMA 不同週期的價格偏離
    for period in [10, 50, 100]:
        sma_alt = _first_series(registry.calculate("sma", df, {"period": period}))
        if sma_alt is not None and atr_arr is not None:
            safe_atr = atr_arr.copy()
            safe_atr[safe_atr == 0] = np.nan
            derived[f"price_vs_sma{period}"] = (close - sma_alt) / safe_atr

    # ADX 不同週期
    for period in [7, 21]:
        adx_alt = _first_series(registry.calculate("adx", df, {"period": period}))
        if adx_alt is not None:
            derived[f"adx_{period}"] = adx_alt

    # ── 斜率 / 加速度因子 ────────────────────
    # RSI 斜率（5 根線性回歸斜率）
    if rsi is not None and len(rsi) > 10:
        rsi_slope = pd.Series(rsi).rolling(5).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan,
            raw=False,
        ).to_numpy(copy=True)
        derived["rsi_slope"] = rsi_slope

    # MACD 零軸距離
    if macd_calc and "MACD_Line" in macd_calc:
        macd_line = _to_array(macd_calc["MACD_Line"])
        derived["macd_zero_dist"] = macd_line

    # 成交量加速度
    vol = df["volume"].values.astype(float)
    vol_ma5 = pd.Series(vol).rolling(5).mean().to_numpy(copy=True)
    vol_ma20 = pd.Series(vol).rolling(20).mean().to_numpy(copy=True)
    vol_ma20_safe = vol_ma20.copy()
    vol_ma20_safe[vol_ma20_safe == 0] = np.nan
    derived["vol_accel"] = vol_ma5 / vol_ma20_safe

    # 價格動量（不同週期）
    for period in [3, 10, 20]:
        mom = _roc(close, period)
        derived[f"price_mom_{period}"] = mom

    # ── 交叉/位置因子 ─────────────────────────
    # SMA 交叉信號（短均-長均 / ATR 標準化）
    sma20 = _first_series(registry.calculate("sma", df, {"period": 20}))
    sma50 = _first_series(registry.calculate("sma", df, {"period": 50}))
    if sma20 is not None and sma50 is not None and atr_arr is not None:
        safe_atr = atr_arr.copy()
        safe_atr[safe_atr == 0] = np.nan
        derived["sma_cross_dist"] = (sma20 - sma50) / safe_atr

    # Donchian 位置
    don_calc = _safe_calc("donchian")
    if don_calc and "Donchian_Upper" in don_calc and "Donchian_Lower" in don_calc:
        don_u = _to_array(don_calc["Donchian_Upper"])
        don_l = _to_array(don_calc["Donchian_Lower"])
        don_range = np.array(don_u - don_l, copy=True)
        don_range[don_range == 0] = np.nan
        derived["donchian_position"] = (close - don_l) / don_range

    # 價格相對 ATR 正規化波動
    if atr_arr is not None:
        close_safe = close.copy()
        close_safe[close_safe == 0] = np.nan
        derived["atr_pct"] = atr_arr / close_safe

    # RSI - 50 距離（衡量中性偏離）
    if rsi is not None:
        derived["rsi_neutral_dist"] = rsi - 50.0

    # HighLow 比率
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    low_safe = low.copy()
    low_safe[low_safe == 0] = np.nan
    derived["high_low_ratio"] = (high - low) / low_safe

    # 收盤位置（在當日 HL 範圍中的相對位置）
    hl_range = np.array(high - low, copy=True)
    hl_range[hl_range == 0] = np.nan
    derived["close_position"] = (close - low) / hl_range

    # EMA 偏離（短/長）
    for period in [10, 50]:
        ema_alt = _first_series(registry.calculate("ema", df, {"period": period}))
        if ema_alt is not None and atr_arr is not None:
            safe_atr = atr_arr.copy()
            safe_atr[safe_atr == 0] = np.nan
            derived[f"price_vs_ema{period}"] = (close - ema_alt) / safe_atr

    # OBV 斜率
    if obv_arr is not None and len(obv_arr) > 10:
        obv_slope = pd.Series(obv_arr).rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else np.nan,
            raw=False,
        ).to_numpy(copy=True)
        obv_std = pd.Series(obv_arr).rolling(20).std().to_numpy(copy=True)
        obv_std[obv_std == 0] = np.nan
        derived["obv_slope_norm"] = obv_slope / obv_std

    # BIAS 不同週期
    for period in [10, 30]:
        sma_b = _first_series(registry.calculate("sma", df, {"period": period}))
        if sma_b is not None:
            sma_safe = sma_b.copy()
            sma_safe[sma_safe == 0] = np.nan
            derived[f"bias_{period}"] = (close - sma_b) / sma_safe * 100

    # StochRSI 差值（%K - %D 如果有）
    if stoch_calc and "StochRSI_K" in stoch_calc and "StochRSI_D" in stoch_calc:
        stoch_k_arr = _to_array(stoch_calc["StochRSI_K"])
        stoch_d_arr = _to_array(stoch_calc["StochRSI_D"])
        derived["stochrsi_kd_spread"] = stoch_k_arr - stoch_d_arr

    # 連續漲跌天數
    price_change = np.diff(close, prepend=close[0])
    streak = np.zeros(n)
    for i in range(1, n):
        if price_change[i] > 0:
            streak[i] = max(0, streak[i - 1]) + 1
        elif price_change[i] < 0:
            streak[i] = min(0, streak[i - 1]) - 1
    derived["price_streak"] = streak

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
    """對指定切片計算 Spearman IC（只回傳 IC 值，用於 decay curve）。"""
    f_slice = factor[start:end]
    r_slice = fwd[start:end]
    valid = ~np.isnan(f_slice) & ~np.isnan(r_slice)
    if np.sum(valid) < 15:
        return None
    try:
        ic, _ = _spearman_ic(f_slice[valid], r_slice[valid])
        return ic
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
#  智慧策略分級（嚴格 / 建議 / 寬鬆）
# ═══════════════════════════════════════════════════════

def _generate_strategy_tiers(
    factors: dict[str, np.ndarray],
    factor_results: dict[str, dict],
    quantile_results: dict[str, dict],
    positive: list[tuple[str, dict]],
    negative: list[tuple[str, dict]],
    combo_results: dict,
    n_bars: int,
    prediction_feedback: Optional[dict[str, float]] = None,
) -> dict:
    """根據因子掃描的**全部結果** + 預測歷史反饋產生三級策略。

    資料來源優先級：
      1. 雙因子組合（combo IC）— 實際交易是多條件判斷，組合 IC 更有實戰意義
      2. 正相關 / 負相關 Top 因子排名 — 經 Alpha Decay 和評級過濾
      3. 分位數分析 — 用來決定每個因子的具體進場閾值
      4. 預測歷史反饋 — 從過去預測結果 boost/penalize 因子

    三級差異化：
      嚴格版：所有高品質因子（≥2 星 + 半衰期 ≥ 2）的 best_quantile AND
      建議版：只取最佳雙因子組合中的因子 + IC 最強 1~2 個單因子（放寬一級）
      寬鬆版：只保留最強的 2 個因子，再放寬一級
    """

    # ─── 收集有分位數分析的因子 → {factor_name: quantile_data} ─────
    qa_map: dict[str, dict] = {}
    for _label, qa in quantile_results.items():
        bq = qa.get("best_quantile")
        if not bq or not bq.get("entry_suggestion"):
            continue
        fname = qa["factor"]
        if fname not in factors:
            continue
        qa_map[fname] = qa

    # ─── 所有候選因子（正+負 Top，帶評級過濾）─────────────
    all_ranked: list[tuple[str, dict]] = []
    for k, v in list(positive) + list(negative):
        if k not in factors:
            continue
        rating = v.get("rating", {})
        stars = rating.get("stars", 0) if isinstance(rating, dict) else 0
        half_life = v.get("decay", {}).get("half_life")
        # 過濾：至少 1 星，且半衰期不能太短（< 2 表示衰退太快）
        if stars < 1:
            continue
        if half_life is not None and half_life < 2 and stars < 2:
            continue
        all_ranked.append((k, v))

    # 按 |IC| 強度排序（整合預測歷史反饋 boost/penalize）
    fb = prediction_feedback if isinstance(prediction_feedback, dict) else {}
    def _score(item: tuple[str, dict]) -> float:
        base_ic = item[1].get("abs_ic_recent", 0)
        factor_name_parts = item[0].split("_")
        modifier = 0.0
        for part in factor_name_parts:
            if part in fb:
                modifier = fb[part]
                break
        if item[0] in fb:
            modifier = fb[item[0]]
        return base_ic * (1 + modifier)

    all_ranked.sort(key=_score, reverse=True)

    if not all_ranked and not qa_map:
        return {}

    # ─── 工具函式 ────────────────────────────────────────

    def _build_cond(fname: str) -> dict | None:
        """從分位數分析中為某因子建構進場條件。"""
        qa = qa_map.get(fname)
        if not qa:
            return None
        bq = qa.get("best_quantile")
        if not bq:
            return None
        q_idx = bq["index"]
        n_q = qa.get("n_quantiles", 5)
        ranges = qa.get("quantile_ranges", [])
        r = bq.get("range", {})
        ic_val = factor_results.get(fname, {}).get("abs_ic_recent", 0)
        base = {
            "factor": fname,
            "best_q_idx": q_idx,
            "n_quantiles": n_q,
            "ranges": ranges,
            "best_range": r,
            "entry_suggestion": bq.get("entry_suggestion", ""),
            "return_pct": bq.get("return_pct", 0),
            "abs_ic": ic_val,
        }
        if q_idx == n_q - 1:
            return {**base, "use_low": r.get("low"), "use_high": None,
                    "description": f"> {r.get('low', 0):.2f}"}
        elif q_idx == 0:
            return {**base, "use_low": None, "use_high": r.get("high"),
                    "description": f"< {r.get('high', 0):.2f}"}
        else:
            return {**base, "use_low": r.get("low"), "use_high": r.get("high"),
                    "description": f"{r.get('low', 0):.2f} ~ {r.get('high', 0):.2f}"}

    def _relax(c: dict) -> dict:
        """向相鄰分位放寬一級。"""
        q_idx = c["best_q_idx"]
        n_q = c["n_quantiles"]
        ranges = c.get("ranges", [])
        if not ranges:
            return c
        if q_idx == n_q - 1 and q_idx >= 1:
            new_lo = ranges[q_idx - 1].get("low", c.get("use_low", 0))
            return {**c, "use_low": new_lo, "best_q_idx": q_idx - 1,
                    "description": f"> {new_lo:.2f} (放寬)"}
        elif q_idx == 0 and q_idx < n_q - 1:
            new_hi = ranges[q_idx + 1].get("high", c.get("use_high", 0))
            return {**c, "use_high": new_hi, "best_q_idx": q_idx + 1,
                    "description": f"< {new_hi:.2f} (放寬)"}
        elif c.get("use_low") is not None and c.get("use_high") is not None:
            lo_expand = ranges[max(0, q_idx - 1)].get("low", c["use_low"])
            hi_expand = ranges[min(n_q - 1, q_idx + 1)].get("high", c["use_high"])
            return {**c, "use_low": lo_expand, "use_high": hi_expand,
                    "description": f"{lo_expand:.2f} ~ {hi_expand:.2f} (放寬)"}
        return c

    def _count(conds: list[dict]) -> int:
        if not conds:
            return n_bars
        masks = []
        for c in conds:
            arr = factors.get(c["factor"])
            if arr is None:
                continue
            lo, hi = c.get("use_low"), c.get("use_high")
            if lo is not None and hi is not None:
                mask = (arr >= lo) & (arr <= hi)
            elif lo is not None:
                mask = arr >= lo
            elif hi is not None:
                mask = arr <= hi
            else:
                continue
            masks.append(np.where(np.isnan(arr), False, mask))
        if not masks:
            return n_bars
        combined = masks[0]
        for m in masks[1:]:
            combined = combined & m
        return int(np.nansum(combined))

    # ─── 從雙因子組合中提取最佳配對因子 ────────────────────
    combo_factor_pairs: list[tuple[str, str, float]] = []
    for group in ("positive_combos", "negative_combos", "hedge_combos"):
        for combo in combo_results.get(group, []):
            fa, fb = combo.get("factor_a", ""), combo.get("factor_b", "")
            cic = combo.get("combo_abs_ic", 0)
            if fa and fb and cic > 0.03:
                combo_factor_pairs.append((fa, fb, cic))
    combo_factor_pairs.sort(key=lambda x: x[2], reverse=True)

    # ─── 建構各版本的因子集合（去重） ──────────────────────

    def _unique_conds(names: list[str]) -> list[dict]:
        """按 name 列表建構唯一條件集，跳過沒有分位數資料的因子。"""
        seen, out = set(), []
        for n in names:
            if n in seen:
                continue
            seen.add(n)
            c = _build_cond(n)
            if c:
                out.append(c)
        return out

    # ─── 嚴格版：所有高品質因子（≥2 星 + 有分位數分析）──────
    strict_names = [k for k, v in all_ranked
                    if (v.get("rating", {}).get("stars", 0) if isinstance(v.get("rating"), dict) else 0) >= 2
                    and k in qa_map]
    if not strict_names:
        strict_names = [k for k, _ in all_ranked[:5] if k in qa_map]
    strict_conds = _unique_conds(strict_names)

    # ─── 建議版：最佳雙因子組合 + 補充 IC 最強單因子 ────────
    moderate_names: list[str] = []
    if combo_factor_pairs:
        best_pair = combo_factor_pairs[0]
        moderate_names.extend([best_pair[0], best_pair[1]])
    # 補充 IC 最強的因子（不跟組合重複）
    for k, _ in all_ranked:
        if k not in moderate_names and k in qa_map:
            moderate_names.append(k)
            if len(moderate_names) >= 4:
                break
    if not moderate_names:
        moderate_names = [k for k, _ in all_ranked[:3] if k in qa_map]

    moderate_conds = [_relax(c) for c in _unique_conds(moderate_names)]

    # ─── 寬鬆版：只取 IC 最強 2 個，放寬兩級 ────────────────
    loose_names: list[str] = []
    if combo_factor_pairs:
        loose_names = [combo_factor_pairs[0][0], combo_factor_pairs[0][1]]
    if len(loose_names) < 2:
        for k, _ in all_ranked:
            if k not in loose_names and k in qa_map:
                loose_names.append(k)
                if len(loose_names) >= 2:
                    break
    loose_conds = [_relax(_relax(c)) for c in _unique_conds(loose_names[:2])]

    # ─── 計算觸發數 ─────────────────────────────────────
    strict_triggers = _count(strict_conds)
    moderate_triggers = _count(moderate_conds)
    loose_triggers = _count(loose_conds)

    # 如果建議版觸發不比嚴格版多，額外放寬
    if moderate_triggers <= strict_triggers and moderate_conds:
        moderate_conds = [_relax(c) for c in moderate_conds]
        moderate_triggers = _count(moderate_conds)
        # 仍然不夠差異 → 移除最弱的因子
        if moderate_triggers <= strict_triggers and len(moderate_conds) > 2:
            moderate_conds.sort(key=lambda c: c.get("abs_ic", 0))
            moderate_conds = moderate_conds[1:]  # 移除最弱
            moderate_triggers = _count(moderate_conds)

    # 如果寬鬆版觸發不比建議版多，再放寬
    if loose_triggers <= moderate_triggers and loose_conds:
        loose_conds = [_relax(c) for c in loose_conds]
        loose_triggers = _count(loose_conds)

    # ─── 組裝輸出 ────────────────────────────────────────
    def _format_tier(conds, triggers, source_desc):
        return {
            "conditions": [
                {
                    "factor": c["factor"],
                    "description": c.get("description", c.get("entry_suggestion", "")),
                    "abs_ic": round(c.get("abs_ic", 0), 4),
                    "return_pct": round(c.get("return_pct", 0), 2),
                }
                for c in conds
            ],
            "trigger_count": triggers,
            "factor_count": len(conds),
            "source": source_desc,
        }

    # 標記有預測反饋 boost 的因子
    fb_note = ""
    if fb and isinstance(fb, dict):
        boosted = [k for k, v in fb.items() if isinstance(v, (int, float)) and v > 0.05]
        penalized = [k for k, v in fb.items() if isinstance(v, (int, float)) and v < -0.05]
        parts = []
        if boosted:
            parts.append(f"歷史驗證好: {','.join(boosted[:3])}")
        if penalized:
            parts.append(f"歷史驗證差: {','.join(penalized[:3])}")
        if parts:
            fb_note = " | " + " | ".join(parts)

    return {
        "strict": _format_tier(strict_conds, strict_triggers,
                               "高品質因子（≥2星 + 分位數最佳區間）全部 AND" + fb_note),
        "moderate": _format_tier(moderate_conds, moderate_triggers,
                                 "最佳雙因子組合 + IC 強因子（放寬一級）" + fb_note),
        "loose": _format_tier(loose_conds, loose_triggers,
                              "僅 IC 最強 2 因子（放寬兩級）" + fb_note),
        "prediction_feedback_applied": bool(fb),
    }


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
                    ic, pval = _spearman_ic(combo[valid], fwd[valid])
                    if pval > 0.10:
                        continue
                    results.append({
                        "factor_a": ka,
                        "factor_b": kb,
                        "combo_ic": round(ic, 4),
                        "combo_abs_ic": round(abs(ic), 4),
                        "p_value": round(pval, 4),
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
    """綜合評級因子狀態（已提高門檻）。"""
    if abs_ic < 0.03:
        status = "inactive"
        stars = 0
        label = "✗ 無效"
    elif trend == "decaying":
        status = "decaying"
        stars = 1
        label = "↓ 衰退中"
    elif trend == "rising" and abs_ic > 0.06:
        status = "rising"
        stars = 2
        label = "★★ 升溫中"
        if abs_ic > 0.10:
            stars = 3
            label = "★★★ 強勢升溫"
    elif abs_ic > 0.12 and (cv is None or cv < 0.5):
        status = "strong"
        stars = 3
        label = "★★★ Strong"
    elif abs_ic > 0.06:
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

    if half_life is not None and half_life <= 1 and stars > 1:
        stars = max(stars - 1, 1)
        label = f"↓ 快速衰退 ({label})"
        status = "decaying"

    confidence = "高" if abs_ic > 0.10 and (cv is None or cv < 0.4) else \
                 "中" if abs_ic > 0.05 else "低"

    return {"status": status, "stars": stars, "label": label, "confidence": confidence}


# ═══════════════════════════════════════════════════════
#  去重：自動剔除高相關因子
# ═══════════════════════════════════════════════════════

def _deduplicate_correlated_factors(
    factor_results: dict,
    raw_factors: dict[str, np.ndarray],
    threshold: float = 0.85,
) -> dict:
    """剔除因子間 Pearson 相關 > threshold 的冗餘因子，保留 abs_ic_recent 較高者。"""
    keys = list(factor_results.keys())
    if len(keys) <= 1:
        return factor_results

    sorted_keys = sorted(keys, key=lambda k: factor_results[k]["abs_ic_recent"], reverse=True)
    keep: set[str] = set()
    removed: set[str] = set()

    for k in sorted_keys:
        if k in removed:
            continue
        keep.add(k)
        arr_k = raw_factors.get(k)
        if arr_k is None:
            continue
        for other in sorted_keys:
            if other in keep or other in removed or other == k:
                continue
            arr_o = raw_factors.get(other)
            if arr_o is None:
                continue
            valid = ~np.isnan(arr_k) & ~np.isnan(arr_o)
            if np.sum(valid) < 20:
                continue
            corr = np.corrcoef(arr_k[valid], arr_o[valid])[0, 1]
            if abs(corr) > threshold:
                removed.add(other)

    return {k: v for k, v in factor_results.items() if k in keep}


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

    df = df.copy()
    n = len(df)
    min_samples_tf = _MIN_SAMPLES_BY_TF.get(timeframe, _MIN_SAMPLES)
    min_required = min_samples_tf + forward_period
    if n < min_required:
        return {
            "status": "error",
            "message": (
                f"數據不足（{n} 根 {timeframe} K 線），"
                f"至少需要 {min_required} 根才能產生可靠的因子掃描結果。"
                f"建議先載入更多歷史數據。"
            ),
        }

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

    # ── 逐因子計算 ──
    _IC_THRESHOLD = 0.05
    _IC_THRESHOLD_RELAXED = 0.03
    _P_VALUE_CUTOFF = 0.05

    factor_results = {}
    for key, arr in factors.items():
        valid_full = ~np.isnan(arr) & ~np.isnan(fwd_full)
        if np.sum(valid_full) < 30:
            continue
        try:
            ic_full, pval_full = _spearman_ic(arr[valid_full], fwd_full[valid_full])
        except Exception:
            continue

        valid_recent = ~np.isnan(arr) & ~np.isnan(fwd_recent)
        samples_recent = int(np.sum(valid_recent))
        if samples_recent >= 30:
            try:
                ic_recent, pval_recent = _spearman_ic(arr[valid_recent], fwd_recent[valid_recent])
            except Exception:
                ic_recent, pval_recent = ic_full, pval_full
        else:
            ic_recent, pval_recent = ic_full, pval_full
            samples_recent = int(np.sum(valid_full))

        # p-value 過濾：兩個 IC 都不顯著則跳過
        if pval_full > _P_VALUE_CUTOFF and pval_recent > _P_VALUE_CUTOFF:
            continue

        decay = _compute_decay_curve(arr, fwd_full, n_windows=6)
        rating = _rate_factor(abs(ic_recent), decay["trend"], decay["cv"], decay["half_life"])

        factor_results[key] = {
            "ic_recent": round(ic_recent, 4),
            "ic_full": round(ic_full, 4),
            "abs_ic_recent": round(abs(ic_recent), 4),
            "p_value_full": round(pval_full, 4),
            "p_value_recent": round(pval_recent, 4),
            "samples": samples_recent,
            "decay": decay,
            "rating": rating,
        }

    if not factor_results:
        return {"status": "error", "message": "所有因子計算失敗或樣本不足（或均未通過 p-value 顯著性檢定）"}

    # ── 自動剔除高相關因子 ──
    factor_results = _deduplicate_correlated_factors(factor_results, factors, threshold=0.85)

    # ── 分正/負相關排序（動態 IC 門檻） ──
    ic_threshold = _IC_THRESHOLD
    positive = sorted(
        [(k, v) for k, v in factor_results.items()
         if v["ic_recent"] > ic_threshold and v["p_value_recent"] <= _P_VALUE_CUTOFF],
        key=lambda x: x[1]["ic_recent"], reverse=True,
    )[:top_n]
    negative = sorted(
        [(k, v) for k, v in factor_results.items()
         if v["ic_recent"] < -ic_threshold and v["p_value_recent"] <= _P_VALUE_CUTOFF],
        key=lambda x: x[1]["ic_recent"],
    )[:top_n]

    if len(positive) + len(negative) < 3:
        ic_threshold = _IC_THRESHOLD_RELAXED
        positive = sorted(
            [(k, v) for k, v in factor_results.items()
             if v["ic_recent"] > ic_threshold and v["p_value_recent"] <= _P_VALUE_CUTOFF],
            key=lambda x: x[1]["ic_recent"], reverse=True,
        )[:top_n]
        negative = sorted(
            [(k, v) for k, v in factor_results.items()
             if v["ic_recent"] < -ic_threshold and v["p_value_recent"] <= _P_VALUE_CUTOFF],
            key=lambda x: x[1]["ic_recent"],
        )[:top_n]

    # ── 分位數分析（OOS: 70% 訓練 / 30% 測試） ──
    oos_split = int(n * 0.7)
    fwd_oos = fwd_full.copy()
    fwd_oos[:oos_split] = np.nan

    quantile_results = {}
    for idx, (k, _v) in enumerate(positive[:top_n]):
        if k in factors:
            qa = _quantile_analysis(factors[k], fwd_oos)
            if qa:
                quantile_results[f"positive_{idx+1}"] = {"factor": k, **qa}
    for idx, (k, _v) in enumerate(negative[:top_n]):
        if k in factors:
            qa = _quantile_analysis(factors[k], fwd_oos)
            if qa:
                quantile_results[f"negative_{idx+1}"] = {"factor": k, **qa}

    # 雙因子組合 IC（正正/負負/正負對沖分開）
    pos_keys = [k for k, _ in positive[:5]]
    neg_keys = [k for k, _ in negative[:5]]
    combo_results = _combo_ic(factors, fwd_full, pos_keys, neg_keys, max_per_group=3)

    # 高相關警告
    high_corr = _find_high_correlations(factors, list(factor_results.keys()))

    # 預測歷史反饋（用於 boost / penalize 因子）
    prediction_feedback = _load_prediction_feedback(symbol=None)

    # 智慧策略分級（整合雙因子組合 + IC 排名 + Alpha Decay + 預測反饋）
    strategy_tiers = _generate_strategy_tiers(
        factors, factor_results, quantile_results,
        positive, negative, combo_results, n,
        prediction_feedback=prediction_feedback,
    )

    # 組裝結果
    relaxed_mode = ic_threshold == _IC_THRESHOLD_RELAXED

    def _fmt(items):
        return [
            {
                "factor": k,
                "ic_recent": v["ic_recent"],
                "ic_full": v["ic_full"],
                "p_value_recent": v.get("p_value_recent"),
                "p_value_full": v.get("p_value_full"),
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

    warnings: list[str] = []
    if relaxed_mode:
        warnings.append("因子數量不足，已將 IC 門檻從 0.05 降級至 0.03，結果可靠度較低。")
    if n < min_samples_tf * 2:
        warnings.append(f"數據量偏低（{n} 根），建議載入 {min_samples_tf * 3}+ 根以獲得更穩定結果。")

    return {
        "status": "success",
        "symbol": None,
        "timeframe": timeframe,
        "total_bars": n,
        "recent_bars": recent_bars,
        "forward_period": forward_period,
        "regime": regime,
        "total_factors_scanned": len(factor_results),
        "ic_threshold_used": ic_threshold,
        "p_value_cutoff": _P_VALUE_CUTOFF,
        "oos_split_ratio": "70/30",
        "dedup_threshold": 0.85,
        "positive_top": _fmt(positive),
        "negative_top": _fmt(negative),
        "combo_top": combo_results,
        "quantile_analysis": quantile_results,
        "high_correlation_warnings": high_corr,
        "strategy_tiers": strategy_tiers,
        "effective_count": sum(
            1 for v in factor_results.values()
            if v["abs_ic_recent"] > 0.05
        ),
        "scan_warnings": warnings,
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
    df = df.copy()
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
                        ic, p_value = _spearman_ic(factor_arr[valid], fwd[valid])
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
    df = df.copy()
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
    df = df.copy()
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
