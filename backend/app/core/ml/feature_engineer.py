"""阿斯拉量化系統 — ML 特徵工程 Pipeline

從 OHLCV DataFrame + 技術指標計算出 ML 特徵矩陣。
三種特徵集規模：精簡(8) / 標準(15) / 完整(25+)。

支援：
- 自訂標籤方向（up/down）與幅度門檻
- 窗口特徵（lookback_window 統計摘要）
- 類別不平衡偵測與正樣本比率統計
"""

from __future__ import annotations


import numpy as np
import pandas as pd
from loguru import logger

from app.core.indicators import registry

# ─── 特徵集定義 ────────────────────────────────────

FEATURE_SETS: dict[str, list[str]] = {
    "compact": [
        "rsi", "adx", "macd", "bb", "atr", "obv", "rel_vol", "roc",
    ],
    "standard": [
        "rsi", "adx", "macd", "bb", "atr", "obv", "rel_vol", "roc",
        "stochrsi", "bias", "hv", "donchian", "supertrend", "ema", "cvd",
    ],
    "full": [
        "rsi", "adx", "macd", "bb", "atr", "obv", "rel_vol", "roc",
        "stochrsi", "bias", "hv", "donchian", "supertrend", "ema", "cvd",
        "keltner", "psar", "vol_squeeze", "rsi_divergence", "macd_divergence",
        "vol_divergence", "poc", "vwap", "ichimoku", "vol_switch",
    ],
}

_SERIES_MAP: dict[str, list[str] | None] = {
    "rsi": ["rsi"],
    "adx": ["adx"],
    "macd": ["macd", "signal", "histogram"],
    "bb": ["%b", "bandwidth"],
    "atr": ["atr"],
    "obv": ["obv"],
    "rel_vol": ["rel_vol"],
    "roc": ["roc"],
    "stochrsi": ["stochrsi_k", "stochrsi_d"],
    "bias": ["bias"],
    "hv": ["hv"],
    "donchian": ["upper", "lower", "middle"],
    "supertrend": ["direction"],
    "ema": ["ema"],
    "cvd": ["cvd"],
    "keltner": ["upper", "lower", "middle"],
    "psar": ["psar"],
    "vol_squeeze": ["squeeze_pct"],
    "rsi_divergence": ["divergence"],
    "macd_divergence": ["divergence"],
    "vol_divergence": ["divergence"],
    "poc": ["poc"],
    "vwap": ["vwap"],
    "ichimoku": ["tenkan", "kijun"],
    "vol_switch": ["regime"],
}


# ─── 主要入口 ────────────────────────────────────


def build_features(
    df: pd.DataFrame,
    feature_set: str = "standard",
    indicator_ids: list[str] | None = None,
    forward_period: int = 5,
    target_direction: str = "up",
    target_threshold: float = 0.03,
    lookback_window: int = 7,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, dict]:
    """從 OHLCV 數據建構窗口特徵矩陣 + 多種標籤

    Args:
        df: OHLCV DataFrame (需含 open/high/low/close/volume 欄位)
        feature_set: "compact" / "standard" / "full"
        indicator_ids: 自訂指標列表（覆蓋 feature_set）
        forward_period: 標籤計算的未來 K 線數
        target_direction: "up" = 預測上漲，"down" = 預測下跌
        target_threshold: 幅度門檻（例如 0.03 = 3%）
        lookback_window: 回看窗口（對每個特徵計算統計摘要的 K 線數）
        label_type: 標籤類型 "direction"/"amplitude"/"path"（預設 amplitude）

    Returns:
        (X, y, feature_names, valid_mask, label_stats)
        X: 特徵矩陣 (n_valid, n_window_features)
        y: 標籤 (n_valid,)  1=觸發事件, 0=未觸發
        feature_names: 窗口特徵名稱列表
        valid_mask: 布林陣列，標示哪些行有效
        label_stats: 標籤統計 {positive_count, positive_ratio, suggested_threshold, ...}
    """
    df = df.copy()
    n = len(df)
    close = df["close"].values.astype(float)

    ids = indicator_ids or FEATURE_SETS.get(feature_set, FEATURE_SETS["standard"])

    # ── 計算單點特徵 ──
    raw_cols: dict[str, np.ndarray] = {}

    raw_cols["return_1"] = _pct_change(close, 1)
    raw_cols["return_3"] = _pct_change(close, 3)
    raw_cols["return_5"] = _pct_change(close, 5)
    raw_cols["volatility_10"] = _rolling_std(close, 10)
    raw_cols["volume_ratio"] = _volume_ratio(df["volume"].values.astype(float), 20)

    for ind_id in ids:
        try:
            calc = registry.calculate(ind_id, df)
            if not calc:
                continue

            series_keys = _SERIES_MAP.get(ind_id)
            if series_keys is None:
                for sname, svals in calc.items():
                    col_name = f"{ind_id}_{sname}" if sname != ind_id else ind_id
                    raw_cols[col_name] = _to_float_array(svals, n)
            else:
                calc_keys = list(calc.keys())
                for i, skey in enumerate(series_keys):
                    if i < len(calc_keys):
                        col_name = f"{ind_id}_{skey}"
                        raw_cols[col_name] = _to_float_array(calc[calc_keys[i]], n)
        except Exception as e:
            logger.debug(f"特徵工程：指標 {ind_id} 計算失敗: {e}")

    raw_names = list(raw_cols.keys())
    if not raw_names:
        empty_stats = _empty_label_stats()
        return np.empty((0, 0)), np.empty(0), [], np.zeros(n, dtype=bool), empty_stats

    raw_matrix = np.column_stack([raw_cols[fn] for fn in raw_names])  # (n, n_raw)

    # ── 建構窗口統計特徵 ──
    window_features, window_names = _build_window_features(
        raw_matrix, raw_names, lookback_window,
    )

    # ── 計算標籤 ──
    # 支援四種標籤類型：direction / amplitude / path / 預設(legacy)
    label_type = kwargs.get("label_type", "amplitude")  # 向下相容：預設用幅度
    high = df["high"].values.astype(float) if "high" in df.columns else close
    low = df["low"].values.astype(float) if "low" in df.columns else close

    y_full = np.full(n, np.nan)
    for i in range(n - forward_period):
        if close[i] == 0:
            continue
        future = slice(i + 1, i + forward_period + 1)

        if label_type == "direction":
            # 方向標籤：終點收盤 vs 當前收盤
            pct = (close[i + forward_period] - close[i]) / close[i]
            y_full[i] = 1.0 if pct > 0 else 0.0

        elif label_type == "path":
            # 路徑標籤：先碰到 TP 還是先碰到 SL（First-Touch Probability）
            hit_tp = False
            for j in range(i + 1, min(i + forward_period + 1, n)):
                if target_direction == "up":
                    up_pct = (high[j] - close[i]) / close[i]
                    down_pct = (close[i] - low[j]) / close[i]
                else:
                    up_pct = (close[i] - low[j]) / close[i]
                    down_pct = (high[j] - close[i]) / close[i]
                if up_pct >= target_threshold:
                    hit_tp = True
                    break
                if down_pct >= target_threshold:
                    break
            y_full[i] = 1.0 if hit_tp else 0.0

        else:
            # amplitude 幅度標籤（預設）：N 根內最高漲幅是否達標
            if target_direction == "down":
                max_move = (close[i] - np.min(low[future])) / close[i]
            else:
                max_move = (np.max(high[future]) - close[i]) / close[i]
            y_full[i] = 1.0 if max_move >= target_threshold else 0.0

    # ── 過濾有效行 ──
    # 雙重保障：排除最後 forward_period 行（label 區域不完整）和 window 暖機行
    valid_mask = (
        ~np.isnan(y_full)
        & ~np.any(np.isnan(window_features), axis=1)
    )
    # 強制排除尾部，防止 partial forward window 洩漏
    if forward_period > 0:
        valid_mask[-(forward_period):] = False

    X = window_features[valid_mask]
    y = y_full[valid_mask]

    # ── 標籤統計 ──
    label_stats = _compute_label_stats(
        close, forward_period, target_direction, target_threshold, y,
    )

    logger.info(
        f"特徵工程完成: {len(window_names)} 窗口特徵 (回看={lookback_window}), "
        f"{int(valid_mask.sum())}/{n} 有效樣本, "
        f"正樣本 {label_stats['positive_count']}/{len(y)} ({label_stats['positive_ratio']:.1%})"
    )

    return X, y, window_names, valid_mask, label_stats


# ═══════════════════════════════════════════════════
#  因子淨化：VIF + 殘差正交化
# ═══════════════════════════════════════════════════

# 因子群分組（用於殘差正交化）
_FACTOR_GROUPS = {
    "trend": ["sma", "ema", "supertrend", "donchian", "ichimoku"],
    "momentum": ["rsi", "macd", "roc", "stochrsi", "bias"],
    "volatility": ["atr", "bb", "keltner", "hv", "vol_squeeze"],
    "volume": ["obv", "cvd", "rel_vol", "volume_ratio"],
    "structure": ["adx", "market_structure", "psar"],
}


def orthogonalize_features(
    X: np.ndarray,
    feature_names: list[str],
    vif_threshold: float = 10.0,
) -> tuple[np.ndarray, list[str], dict]:
    """因子淨化：VIF 共線性檢查 + 群內殘差正交化。

    Args:
        X: 特徵矩陣 (n_samples, n_features)
        feature_names: 特徵名稱列表
        vif_threshold: VIF 閾值，>10 視為高共線性

    Returns:
        (X_clean, clean_names, report)
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    if X.shape[1] < 2 or X.shape[0] < X.shape[1]:
        return X, feature_names, {"status": "skipped", "reason": "特徵數不足或樣本太少"}

    # 替換 NaN/Inf
    X_clean = X.copy()
    X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=0.0, neginf=0.0)

    # 1. VIF 檢查
    vif_scores = []
    for i in range(X_clean.shape[1]):
        try:
            v = variance_inflation_factor(X_clean, i)
            vif_scores.append(float(v) if np.isfinite(v) else 999.0)
        except Exception:
            vif_scores.append(999.0)

    high_vif = [
        {"feature": feature_names[i], "vif": round(vif_scores[i], 1)}
        for i in range(len(feature_names))
        if vif_scores[i] > vif_threshold
    ]

    # 2. 按群分組，群內保留 VIF 最低的，其餘用殘差取代
    keep_mask = np.ones(len(feature_names), dtype=bool)
    orthogonalized = []

    for group_name, group_prefixes in _FACTOR_GROUPS.items():
        # 找屬於這個群的特徵
        group_indices = []
        for i, name in enumerate(feature_names):
            base = name.split("_")[0]
            if base in group_prefixes:
                group_indices.append(i)

        if len(group_indices) < 2:
            continue

        # 保留 VIF 最低的作為主因子
        group_vifs = [(idx, vif_scores[idx]) for idx in group_indices]
        group_vifs.sort(key=lambda x: x[1])
        primary_idx = group_vifs[0][0]

        # 其餘因子用殘差正交化（去掉主因子的影響）
        for idx, _ in group_vifs[1:]:
            if vif_scores[idx] > vif_threshold:
                # 用 OLS 殘差取代：y = a*primary + b → residual = y - predicted
                primary_col = X_clean[:, primary_idx]
                target_col = X_clean[:, idx]
                denom = np.dot(primary_col, primary_col)
                if denom > 1e-10:
                    beta = np.dot(primary_col, target_col) / denom
                    residual = target_col - beta * primary_col
                    X_clean[:, idx] = residual
                    orthogonalized.append(feature_names[idx])

    # 3. 移除仍然極高 VIF（>50）的特徵
    final_keep = []
    for i in range(len(feature_names)):
        if vif_scores[i] > 50 and feature_names[i] not in orthogonalized:
            keep_mask[i] = False
        else:
            final_keep.append(i)

    X_result = X_clean[:, keep_mask]
    names_result = [feature_names[i] for i in range(len(feature_names)) if keep_mask[i]]

    report = {
        "status": "success",
        "original_features": len(feature_names),
        "retained_features": len(names_result),
        "removed_count": len(feature_names) - len(names_result),
        "high_vif_features": high_vif[:10],
        "orthogonalized": orthogonalized,
    }

    logger.info(
        f"因子淨化完成: {len(feature_names)} → {len(names_result)} 特徵 "
        f"(移除 {len(feature_names) - len(names_result)}，正交化 {len(orthogonalized)})"
    )

    return X_result, names_result, report


def build_latest_features(
    df: pd.DataFrame,
    feature_set: str = "standard",
    indicator_ids: list[str] | None = None,
    lookback_window: int = 7,
) -> tuple[np.ndarray, list[str]]:
    """只取最新一筆的窗口特徵向量（用於即時預測）

    Returns:
        (X_latest, feature_names)
        X_latest: (1, n_window_features)
        feature_names: 窗口特徵名稱列表
    """
    X, _, feature_names, valid_mask, _ = build_features(
        df, feature_set, indicator_ids,
        forward_period=1,
        target_direction="up",
        target_threshold=0.0,
        lookback_window=lookback_window,
    )
    if len(X) == 0:
        return np.empty((0, 0)), []

    return X[[-1]], feature_names


# ─── 窗口特徵建構 ────────────────────────────────────


def _build_window_features(
    raw_matrix: np.ndarray,
    raw_names: list[str],
    window: int,
) -> tuple[np.ndarray, list[str]]:
    """對每個原始特徵計算 lookback 窗口統計量

    統計量：mean / slope / std / last
    """
    n, n_raw = raw_matrix.shape
    stats_per_feat = 4  # mean, slope, std, last
    n_window_feats = n_raw * stats_per_feat
    result = np.full((n, n_window_feats), np.nan)
    names: list[str] = []

    for j, fname in enumerate(raw_names):
        names.extend([
            f"{fname}_mean_{window}",
            f"{fname}_slope_{window}",
            f"{fname}_std_{window}",
            f"{fname}_last",
        ])

    x_range = np.arange(window, dtype=float)
    x_mean = x_range.mean()
    x_var = np.sum((x_range - x_mean) ** 2)

    for i in range(window, n):
        segment = raw_matrix[i - window:i]  # (window, n_raw)
        col_mean = np.nanmean(segment, axis=0)
        col_std = np.nanstd(segment, axis=0)

        # 線性斜率 (最小二乘法): slope = Σ(x-x̄)(y-ȳ) / Σ(x-x̄)²
        if x_var > 0:
            centered_y = segment - np.nanmean(segment, axis=0, keepdims=True)
            col_slope = np.nansum(
                (x_range[:, None] - x_mean) * centered_y, axis=0
            ) / x_var
        else:
            col_slope = np.zeros(n_raw)

        col_last = segment[-1]

        for j in range(n_raw):
            base = j * stats_per_feat
            result[i, base] = col_mean[j]
            result[i, base + 1] = col_slope[j]
            result[i, base + 2] = col_std[j]
            result[i, base + 3] = col_last[j]

    return result, names


# ─── 標籤統計 ────────────────────────────────────


def _compute_label_stats(
    close: np.ndarray,
    forward_period: int,
    direction: str,
    threshold: float,
    y: np.ndarray,
) -> dict:
    """計算標籤分佈統計與建議門檻"""
    n = len(close)
    if n <= forward_period:
        return _empty_label_stats()

    # 歷史報酬率分佈
    returns = np.full(n, np.nan)
    for i in range(n - forward_period):
        if close[i] != 0:
            returns[i] = (close[i + forward_period] - close[i]) / close[i]
    valid_returns = returns[~np.isnan(returns)]

    if len(valid_returns) == 0:
        return _empty_label_stats()

    abs_returns = np.abs(valid_returns)
    p75 = float(np.percentile(abs_returns, 75))
    p50 = float(np.percentile(abs_returns, 50))

    pos_count = int(np.sum(y == 1.0))
    total = len(y)
    pos_ratio = pos_count / total if total > 0 else 0.0

    return {
        "positive_count": pos_count,
        "negative_count": total - pos_count,
        "total_samples": total,
        "positive_ratio": round(pos_ratio, 4),
        "target_direction": direction,
        "target_threshold": threshold,
        "suggested_threshold": round(p75, 4),
        "median_return": round(p50, 4),
        "p75_return": round(p75, 4),
        "imbalance_warning": pos_ratio < 0.05 or pos_ratio > 0.95,
        "insufficient_positives": pos_count < 50,
    }


def _empty_label_stats() -> dict:
    return {
        "positive_count": 0, "negative_count": 0, "total_samples": 0,
        "positive_ratio": 0.0, "target_direction": "up", "target_threshold": 0.0,
        "suggested_threshold": 0.0, "median_return": 0.0, "p75_return": 0.0,
        "imbalance_warning": True, "insufficient_positives": True,
    }


# ─── 標準化（不變） ────────────────────────────────────


def normalize_features(
    X: np.ndarray,
    method: str = "zscore",
) -> tuple[np.ndarray, dict]:
    """標準化特徵矩陣"""
    params = {}
    X_out = X.copy()

    if method == "zscore":
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        std[std == 0] = 1.0
        X_out = (X - mean) / std
        params = {"method": "zscore", "mean": mean.tolist(), "std": std.tolist()}
    elif method == "minmax":
        mn = np.nanmin(X, axis=0)
        mx = np.nanmax(X, axis=0)
        rng = mx - mn
        rng[rng == 0] = 1.0
        X_out = (X - mn) / rng
        params = {"method": "minmax", "min": mn.tolist(), "max": mx.tolist()}

    X_out = np.nan_to_num(X_out, nan=0.0)
    return X_out, params


def apply_normalization(X: np.ndarray, params: dict) -> np.ndarray:
    """用已有的 scaler params 標準化新數據"""
    X_out = X.copy()
    method = params.get("method", "zscore")

    if method == "zscore":
        mean = np.array(params["mean"])
        std = np.array(params["std"])
        X_out = (X - mean) / std
    elif method == "minmax":
        mn = np.array(params["min"])
        mx = np.array(params["max"])
        rng = mx - mn
        rng[rng == 0] = 1.0
        X_out = (X - mn) / rng

    return np.nan_to_num(X_out, nan=0.0)


# ─── 輔助函式 ────────────────────────────────────

def _pct_change(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full_like(arr, np.nan)
    if period >= len(arr):
        return out
    prev = arr[:-period] if period > 0 else arr
    safe_prev = np.where(prev == 0, np.nan, prev)
    out[period:] = (arr[period:] - prev) / safe_prev
    return out


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(arr, np.nan)
    for i in range(window, len(arr)):
        out[i] = np.std(arr[i - window:i])
    return out


def _volume_ratio(vol: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(vol, np.nan)
    ma = pd.Series(vol).rolling(window, min_periods=1).mean().values
    safe_ma = np.where(ma == 0, np.nan, ma)
    out = vol / safe_ma
    return out


def _to_float_array(values: list, expected_len: int) -> np.ndarray:
    arr = np.array([
        float(v) if v is not None and not (isinstance(v, float) and np.isnan(v))
        else np.nan
        for v in values
    ], dtype=float)
    if len(arr) < expected_len:
        arr = np.concatenate([np.full(expected_len - len(arr), np.nan), arr])
    return arr[:expected_len]
