"""阿斯拉量化系統 — v103 6B+6C：OHLCV 資料品質監控（本地、不付費）。

兩個檢查：
  1. **outlier 偵測**：單根 K 線收盤變化 > 3σ → log warning
  2. **重複資料**：(timestamp, symbol) 重複 → 自動去重 + log

寫到 backend/data/db/data_quality.log。
建議跟 daily_drift_check 一起跑（每日 03:00 後）。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

_BACKEND_DIR = Path(__file__).resolve().parents[2]
_OHLCV_DIR = _BACKEND_DIR / "data" / "ohlcv"
_LOG_PATH = _BACKEND_DIR / "data" / "db" / "data_quality.log"


def _append_log(line: str) -> None:
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] {line}\n")
    except Exception:
        pass


def detect_outliers_in_file(filepath: Path, sigma: float = 5.0) -> dict:
    """單一 CSV 檔的單 K 線異常偵測（收盤變化 > Nσ）。

    crypto 高 kurtosis 環境 3σ 太鬆（每檔 30+ 筆），預設 5σ 抓真極端事件。
    """
    try:
        df = pd.read_csv(filepath, on_bad_lines="skip", low_memory=False)
    except Exception as e:
        return {"file": filepath.name, "error": f"read failed: {e}"}

    if df.empty or "close" not in df.columns:
        return {"file": filepath.name, "error": "empty or missing close"}

    df = df.dropna(subset=["close"]).reset_index(drop=True)
    if len(df) < 30:
        return {"file": filepath.name, "skipped": "too few rows"}

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    if len(df) < 30:
        return {"file": filepath.name, "skipped": "too few valid"}

    pct_change = df["close"].pct_change().dropna()
    if pct_change.empty:
        return {"file": filepath.name, "skipped": "no pct_change"}

    mu = float(pct_change.mean())
    std = float(pct_change.std() or 0.0)
    if std == 0:
        return {"file": filepath.name, "skipped": "zero std"}

    z = (pct_change - mu) / std
    outlier_idx = np.where(np.abs(z) > sigma)[0]
    outliers = []
    for i in outlier_idx:
        ts = df.iloc[i + 1].get("timestamp", "?")  # +1 因 pct_change 少一筆
        outliers.append({
            "timestamp": str(ts),
            "pct_change": round(float(pct_change.iloc[i]) * 100, 2),
            "z": round(float(z.iloc[i]), 2),
        })

    return {
        "file": filepath.name,
        "n_rows": len(df),
        "n_outliers": len(outliers),
        "outliers": outliers[:5],  # 只取前 5 筆 sample
    }


def detect_duplicates_in_file(filepath: Path, auto_dedupe: bool = True) -> dict:
    """單一 CSV 檔的重複 timestamp 偵測 + 可選自動去重（保留最後一筆）。"""
    try:
        df = pd.read_csv(filepath, on_bad_lines="skip", low_memory=False)
    except Exception as e:
        return {"file": filepath.name, "error": f"read failed: {e}"}

    if df.empty or "timestamp" not in df.columns:
        return {"file": filepath.name, "error": "missing timestamp"}

    n_total = len(df)
    duplicated = df.duplicated(subset=["timestamp"], keep="last")
    n_dup = int(duplicated.sum())

    if n_dup == 0:
        return {"file": filepath.name, "n_total": n_total, "n_dup": 0}

    result = {"file": filepath.name, "n_total": n_total, "n_dup": n_dup, "dedupe_applied": False}

    if auto_dedupe:
        try:
            df_clean = df[~duplicated].reset_index(drop=True)
            df_clean.to_csv(filepath, index=False)
            result["dedupe_applied"] = True
            result["n_after"] = len(df_clean)
            _append_log(f"DEDUPE {filepath.name}: removed {n_dup} duplicates ({n_total} -> {len(df_clean)})")
        except Exception as e:
            result["error"] = f"dedupe failed: {e}"

    return result


def run_full_check(sigma: float = 5.0, auto_dedupe: bool = True) -> dict:
    """掃 backend/data/ohlcv/ 下所有 CSV，回傳摘要。"""
    if not _OHLCV_DIR.exists():
        return {"error": "OHLCV dir not found", "checked": 0}

    files = list(_OHLCV_DIR.glob("*.csv"))
    out: dict = {"checked": len(files), "outliers": [], "duplicates": []}
    total_outliers = 0
    total_dupes = 0

    for f in files:
        # outlier
        try:
            o = detect_outliers_in_file(f, sigma=sigma)
            if o.get("n_outliers", 0) > 0:
                total_outliers += o["n_outliers"]
                out["outliers"].append(o)
                _append_log(f"OUTLIERS {f.name}: {o['n_outliers']} bars > {sigma}σ")
        except Exception as e:
            logger.debug(f"outlier check {f.name} 失敗: {e}")

        # duplicates
        try:
            d = detect_duplicates_in_file(f, auto_dedupe=auto_dedupe)
            if d.get("n_dup", 0) > 0:
                total_dupes += d["n_dup"]
                out["duplicates"].append(d)
        except Exception as e:
            logger.debug(f"dupe check {f.name} 失敗: {e}")

    out["total_outliers"] = total_outliers
    out["total_dupes_removed"] = total_dupes if auto_dedupe else 0
    out["completed_at"] = datetime.now().isoformat()
    _append_log(
        f"SUMMARY: {len(files)} files | outliers across files={total_outliers} | dupes removed={total_dupes}"
    )
    return out


if __name__ == "__main__":
    import json as _json
    print(_json.dumps(run_full_check(), ensure_ascii=False, indent=2, default=str))
