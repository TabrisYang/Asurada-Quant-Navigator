"""阿斯拉量化系統 — ML 特徵工程 Pipeline

從 OHLCV DataFrame + 技術指標計算出 ML 特徵矩陣。
三種特徵集規模：精簡(8) / 標準(15) / 完整(25+)。

支援：
- 自訂標籤方向（up/down）與幅度門檻
- 窗口特徵（lookback_window 統計摘要）
- 類別不平衡偵測與正樣本比率統計
"""

from __future__ import annotations

from typing import Optional

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
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, dict]:
    """從 OHLCV 數據建構窗口特徵矩陣 + 幅度門檻標籤

    Args:
        df: OHLCV DataFrame (需含 open/high/low/close/volume 欄位)
        feature_set: "compact" / "standard" / "full"
        indicator_ids: 自訂指標列表（覆蓋 feature_set）
        forward_period: 標籤計算的未來 K 線數
        target_direction: "up" = 預測上漲，"down" = 預測下跌
        target_threshold: 幅度門檻（例如 0.03 = 3%）
        lookback_window: 回看窗口（對每個特徵計算統計摘要的 K 線數）

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

    # ── 計算標籤：未來 N 根是否達到幅度門檻 ──
    y_full = np.full(n, np.nan)
    for i in range(n - forward_period):
        pct = (close[i + forward_period] - close[i]) / close[i] if close[i] != 0 else 0.0
        if target_direction == "down":
            y_full[i] = 1.0 if pct <= -target_threshold else 0.0
        else:
            y_full[i] = 1.0 if pct >= target_threshold else 0.0

    # ── 過濾有效行 ──
    valid_mask = (
        ~np.isnan(y_full)
        & ~np.any(np.isnan(window_features), axis=1)
    )

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
