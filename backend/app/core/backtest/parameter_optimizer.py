"""阿斯拉量化系統 — 指標參數校準引擎

掃描每個指標在不同閾值下對未來報酬的預測力，
找出「穩健區間」而非單一最佳值，並用 Walk Forward 驗證防止過擬合。

輸出：每個指標的校準參數（做多閾值、做空閾值、動能閾值等）+ 可信度等級。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from app.core.indicators import registry
from app.core.config.settings import settings

# ─── 可校準指標及其掃描設定 ────────────────────────────
CALIBRATION_PROFILES: dict[str, dict] = {
    "rsi": {
        "name": "RSI",
        "series_key": 0,
        "scan_range": np.arange(20, 82, 2).tolist(),
        "roles": {
            "long_entry": {"direction": "below", "desc": "做多進場（低於此值超賣）"},
            "short_entry": {"direction": "above", "desc": "做空進場（高於此值超買）"},
        },
    },
    "adx": {
        "name": "ADX",
        "series_key": 0,
        "scan_range": np.arange(10, 55, 2.5).tolist(),
        "roles": {
            "trend_threshold": {"direction": "above", "desc": "趨勢確認（高於此值有動能）"},
        },
    },
    "stochrsi": {
        "name": "StochRSI",
        "series_key": 0,
        "scan_range": np.arange(5, 100, 5).tolist(),
        "roles": {
            "long_entry": {"direction": "below", "desc": "做多進場（低於此值超賣）"},
            "short_entry": {"direction": "above", "desc": "做空進場（高於此值超買）"},
        },
    },
    "macd": {
        "name": "MACD Histogram",
        "series_key": 2,  # histogram
        "scan_range": None,  # percentile-based
        "roles": {
            "bullish_threshold": {"direction": "above", "desc": "看多（histogram > 此值）"},
            "bearish_threshold": {"direction": "below", "desc": "看空（histogram < 此值）"},
        },
    },
    "roc": {
        "name": "ROC",
        "series_key": 0,
        "scan_range": None,
        "roles": {
            "bullish_threshold": {"direction": "above", "desc": "動能偏多"},
            "bearish_threshold": {"direction": "below", "desc": "動能偏空"},
        },
    },
    "bias": {
        "name": "Bias",
        "series_key": 0,
        "scan_range": None,
        "roles": {
            "overbought": {"direction": "above", "desc": "過熱（高於此值偏離過大）"},
            "oversold": {"direction": "below", "desc": "超跌（低於此值偏離過大）"},
        },
    },
    "rel_vol": {
        "name": "Relative Volume",
        "series_key": 0,
        "scan_range": np.arange(0.5, 4.1, 0.25).tolist(),
        "roles": {
            "volume_surge": {"direction": "above", "desc": "量能放大確認"},
        },
    },
    "atr": {
        "name": "ATR",
        "series_key": 0,
        "scan_range": None,
        "roles": {
            "high_volatility": {"direction": "above", "desc": "高波動環境"},
        },
    },
    "obv": {
        "name": "OBV",
        "series_key": 0,
        "scan_range": None,
        "roles": {
            "bullish_trend": {"direction": "above", "desc": "OBV 上升趨勢確認"},
        },
    },
}

# ─── Walk Forward 視窗數 ─────────────────────────────
N_WF_WINDOWS = 5
TRAIN_RATIO = 0.7

# ─── 結果儲存路徑 ────────────────────────────────────
def _calibration_dir() -> Path:
    p = settings.db_path / "calibration"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _symbol_key(symbol: str) -> str:
    return symbol.replace("/", "_").replace(" ", "_")


# ═══════════════════════════════════════════════════════
# 核心校準引擎
# ═══════════════════════════════════════════════════════

def run_calibration(
    df: pd.DataFrame,
    symbol: str,
    indicator_ids: Optional[list[str]] = None,
    forward_bars: int = 5,
    min_samples: int = 10,
) -> dict:
    """對指定標的執行指標參數校準。

    Args:
        df: OHLCV DataFrame（至少 200 根 K 線）
        symbol: 交易對名稱
        indicator_ids: 要校準的指標（None = 全部可校準指標）
        forward_bars: 計算未來報酬時看幾根 K 線
        min_samples: 最少觸發次數門檻

    Returns:
        dict: 校準結果，含每個指標的最佳區間和可信度
    """
    n = len(df)
    if n < 200:
        return {"status": "error", "message": f"數據不足（{n} 根），至少需要 200 根 K 線"}

    closes = df["close"].values.astype(float)
    future_ret = np.full(n, np.nan)
    for i in range(n - forward_bars):
        future_ret[i] = (closes[i + forward_bars] - closes[i]) / closes[i]

    targets = indicator_ids or list(CALIBRATION_PROFILES.keys())
    calibration_results: dict[str, dict] = {}

    for ind_id in targets:
        profile = CALIBRATION_PROFILES.get(ind_id)
        if not profile:
            continue

        try:
            calc = registry.calculate(ind_id, df)
            if not calc:
                continue

            series_name = list(calc.keys())[profile["series_key"]]
            values = np.array(calc[series_name], dtype=float)

            scan_range = profile["scan_range"]
            if scan_range is None:
                valid_vals = values[~np.isnan(values)]
                if len(valid_vals) < 50:
                    continue
                pcts = np.percentile(valid_vals, np.arange(5, 96, 5))
                scan_range = pcts.tolist()

            ind_result = _calibrate_indicator(
                ind_id, profile, values, future_ret, scan_range,
                df, forward_bars, min_samples,
            )
            if ind_result:
                calibration_results[ind_id] = ind_result

        except Exception as e:
            logger.warning(f"校準 {ind_id} 失敗: {e}")

    result = {
        "status": "success",
        "symbol": symbol,
        "total_bars": n,
        "forward_bars": forward_bars,
        "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_range": f"{df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}",
        "indicators": calibration_results,
        "total_calibrated": len(calibration_results),
    }

    save_calibration(symbol, result)
    return result


def _calibrate_indicator(
    ind_id: str,
    profile: dict,
    values: np.ndarray,
    future_ret: np.ndarray,
    scan_range: list[float],
    df: pd.DataFrame,
    forward_bars: int,
    min_samples: int,
) -> Optional[dict]:
    """校準單個指標的所有角色閾值。"""
    roles_result = {}
    valid_mask = ~np.isnan(values) & ~np.isnan(future_ret)

    for role_name, role_cfg in profile["roles"].items():
        direction = role_cfg["direction"]
        best = _scan_threshold(
            values, future_ret, valid_mask, scan_range, direction, min_samples,
        )
        if best is None:
            continue

        # Walk Forward 驗證
        wf_results = _walk_forward_validate(
            ind_id, df, best["threshold"], direction, forward_bars, min_samples,
        )

        confidence = _compute_confidence(best, wf_results, min_samples)

        # 從 Walk Forward 結果計算穩健區間
        if wf_results and len(wf_results) >= 3:
            wf_thresholds = [w["best_threshold"] for w in wf_results if w.get("best_threshold") is not None]
            if wf_thresholds:
                robust_low = float(np.percentile(wf_thresholds, 20))
                robust_high = float(np.percentile(wf_thresholds, 80))
                robust_median = float(np.median(wf_thresholds))
            else:
                robust_low = robust_high = robust_median = best["threshold"]
        else:
            robust_low = robust_high = robust_median = best["threshold"]

        roles_result[role_name] = {
            "description": role_cfg["desc"],
            "direction": direction,
            "optimal_value": round(best["threshold"], 2),
            "robust_range": [round(robust_low, 2), round(robust_high, 2)],
            "robust_median": round(robust_median, 2),
            "avg_return_pct": round(best["avg_return"] * 100, 3),
            "win_rate_pct": round(best["win_rate"] * 100, 1),
            "samples": best["samples"],
            "confidence": confidence,
            "confidence_stars": "★" * confidence,
            "wf_consistency": round(wf_results[0]["consistency"], 1) if wf_results and "consistency" in wf_results[0] else None,
        }

    if not roles_result:
        return None

    return {
        "name": profile["name"],
        "roles": roles_result,
    }


def _scan_threshold(
    values: np.ndarray,
    future_ret: np.ndarray,
    valid_mask: np.ndarray,
    scan_range: list[float],
    direction: str,
    min_samples: int,
) -> Optional[dict]:
    """掃描一系列閾值，找出預測力最強的閾值。"""
    best = None
    best_score = -np.inf

    for threshold in scan_range:
        if direction == "above":
            cond = (values > threshold) & valid_mask
        else:
            cond = (values < threshold) & valid_mask

        n_hits = int(np.sum(cond))
        if n_hits < min_samples:
            continue

        ret_when_hit = future_ret[cond]
        avg_ret = float(np.nanmean(ret_when_hit))
        win_rate = float(np.sum(ret_when_hit > 0) / n_hits)

        if direction == "above":
            score = avg_ret * np.sqrt(n_hits)
        else:
            score = (-avg_ret) * np.sqrt(n_hits)

        if score > best_score:
            best_score = score
            best = {
                "threshold": float(threshold),
                "avg_return": avg_ret,
                "win_rate": win_rate,
                "samples": n_hits,
                "score": float(score),
            }

    return best


def _walk_forward_validate(
    ind_id: str,
    df: pd.DataFrame,
    overall_threshold: float,
    direction: str,
    forward_bars: int,
    min_samples: int,
) -> list[dict]:
    """Walk Forward 驗證：在多個滑動窗口中重新尋找最佳閾值。"""
    n = len(df)
    window_size = n // N_WF_WINDOWS
    if window_size < 60:
        return []

    profile = CALIBRATION_PROFILES.get(ind_id)
    if not profile:
        return []

    results = []
    per_window_thresholds = []

    for i in range(N_WF_WINDOWS):
        start = i * (n - window_size) // max(N_WF_WINDOWS - 1, 1)
        end = min(start + window_size, n)
        window_df = df.iloc[start:end].copy().reset_index(drop=True)

        if len(window_df) < 60:
            continue

        try:
            calc = registry.calculate(ind_id, window_df)
            if not calc:
                continue

            series_name = list(calc.keys())[profile["series_key"]]
            vals = np.array(calc[series_name], dtype=float)
            closes = window_df["close"].values.astype(float)
            fret = np.full(len(window_df), np.nan)
            for j in range(len(window_df) - forward_bars):
                fret[j] = (closes[j + forward_bars] - closes[j]) / closes[j]

            valid = ~np.isnan(vals) & ~np.isnan(fret)

            scan_range = profile["scan_range"]
            if scan_range is None:
                valid_vals = vals[~np.isnan(vals)]
                if len(valid_vals) < 20:
                    continue
                scan_range = np.percentile(valid_vals, np.arange(10, 91, 10)).tolist()

            window_best = _scan_threshold(vals, fret, valid, scan_range, direction, max(3, min_samples // 3))
            if window_best:
                per_window_thresholds.append(window_best["threshold"])
                results.append({
                    "window": i + 1,
                    "best_threshold": window_best["threshold"],
                    "avg_return": window_best["avg_return"],
                    "win_rate": window_best["win_rate"],
                    "samples": window_best["samples"],
                })
        except Exception:
            continue

    if per_window_thresholds:
        std = float(np.std(per_window_thresholds))
        mean = float(np.mean(per_window_thresholds))
        cv = std / abs(mean) * 100 if abs(mean) > 1e-8 else 100
        consistency = max(0, 100 - cv)
        for r in results:
            r["consistency"] = consistency

    return results


def _compute_confidence(best: dict, wf_results: list[dict], min_samples: int) -> int:
    """計算可信度等級（1~3 星）。"""
    score = 0

    if best["samples"] >= min_samples * 5:
        score += 1
    elif best["samples"] >= min_samples * 2:
        score += 0.5

    if wf_results and len(wf_results) >= 3:
        consistency = wf_results[0].get("consistency", 0)
        if consistency >= 70:
            score += 1
        elif consistency >= 50:
            score += 0.5

    if abs(best["avg_return"]) > 0.002:
        score += 0.5
    if best["win_rate"] > 0.55 or best["win_rate"] < 0.45:
        score += 0.5

    return min(3, max(1, int(round(score))))


# ═══════════════════════════════════════════════════════
# 結果儲存 / 載入
# ═══════════════════════════════════════════════════════

def save_calibration(symbol: str, result: dict) -> Path:
    """將校準結果儲存到 JSON。"""
    path = _calibration_dir() / f"{_symbol_key(symbol)}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"校準結果已儲存: {path}")
    return path


def load_calibration(symbol: str) -> Optional[dict]:
    """載入已儲存的校準結果。"""
    path = _calibration_dir() / f"{_symbol_key(symbol)}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"載入校準結果失敗: {e}")
        return None


def format_calibration_for_prompt(symbol: str) -> Optional[str]:
    """將校準結果格式化為 LLM 可讀的上下文注入文字。"""
    cal = load_calibration(symbol)
    if not cal or cal.get("status") != "success":
        return None

    indicators = cal.get("indicators", {})
    if not indicators:
        return None

    lines = [
        f"【{symbol} 指標參數校準結果】",
        f"校準時間：{cal.get('calibrated_at', '未知')}",
        f"數據範圍：{cal.get('data_range', '未知')}（{cal.get('total_bars', 0)} 根 K 線）",
        f"前瞻期：{cal.get('forward_bars', 5)} 根 K 線",
        "",
    ]

    for ind_id, ind_data in indicators.items():
        ind_name = ind_data.get("name", ind_id.upper())
        for role_name, role_data in ind_data.get("roles", {}).items():
            direction = role_data["direction"]
            op = ">" if direction == "above" else "<"
            rng = role_data.get("robust_range", [0, 0])
            stars = role_data.get("confidence_stars", "★")
            desc = role_data.get("description", "")
            win = role_data.get("win_rate_pct", 0)
            samples = role_data.get("samples", 0)
            lines.append(
                f"- {ind_name} {desc}：{op} {rng[0]}~{rng[1]}（中位 {role_data.get('robust_median', 0)}）"
                f" | 勝率 {win}% | 樣本 {samples} | {stars}"
            )

    lines.append("")
    lines.append("請優先使用校準值分析，★ 可信度低的項目請與通用教科書值交叉參考。")

    return "\n".join(lines)
