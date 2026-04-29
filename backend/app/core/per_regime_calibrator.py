"""阿斯拉量化系統 — v105 Phase C：per-regime isotonic 校準器。

對 199 verified samples 按 regime_std 分桶，跑 isotonic regression 學每個 regime
的「raw probability → 真實 hit_target 機率」校準曲線。

樣本不足 30 的 regime 自動 skip（推論時 fallback all-in-one 校準）。

執行：
    .venv/bin/python -m app.core.per_regime_calibrator

存到：
- backend/models/calibrator_<regime>.pkl
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.isotonic import IsotonicRegression

_BACKEND_DIR = Path(__file__).resolve().parents[2]
_MODELS_DIR = _BACKEND_DIR / "models"
_MIN_SAMPLES = 30


def _build_per_regime_data() -> dict[str, tuple[pd.Series, pd.Series]]:
    """從 prediction_features × predictions 抽各 regime_std 的訓練資料。

    Returns dict: regime_std → (raw_probability_series, hit_target_series)。

    raw_probability：先用 RSI / MACD 簡單規則合成（缺真實 ML 推論結果時的 proxy）：
      - RSI < 30 + MACD 收斂 → high P
      - RSI > 70 + MACD 擴張下行 → low P
      - 中性 → 0.5
    這是因為 199 樣本沒有 v101 model 推論機率紀錄；之後 shadow_predictions 表
    累積後可用真實 p_ml 重訓。
    """
    from app.core.prediction_tracker import prediction_tracker
    prediction_tracker._ensure_db()
    if not prediction_tracker._conn:
        raise RuntimeError("DB 未初始化")

    rows = prediction_tracker._conn.execute(
        """SELECT pf.rsi_14, pf.macd_hist, pf.adx_14,
                  p.status, p.regime_std, p.direction
           FROM prediction_features pf
           JOIN predictions p ON pf.prediction_id = p.id
           WHERE p.status IN ('hit_target', 'hit_stop')"""
    ).fetchall()
    if not rows:
        return {}

    df = pd.DataFrame([dict(r) for r in rows])
    df["hit_target"] = (df["status"] == "hit_target").astype(int)
    df["rsi_14"] = pd.to_numeric(df["rsi_14"], errors="coerce").fillna(50)
    df["macd_hist"] = pd.to_numeric(df["macd_hist"], errors="coerce").fillna(0)
    df["adx_14"] = pd.to_numeric(df["adx_14"], errors="coerce").fillna(20)

    # Proxy raw probability：用 direction-aligned 動量訊號合成 [0, 1]
    def _raw_prob(row):
        # 多單：RSI 偏高 / MACD 正 / ADX 強 → 高 P
        # 空單：對稱
        rsi = row["rsi_14"]
        macd = row["macd_hist"]
        adx = row["adx_14"]
        score = 0.5
        if row["direction"] == "long":
            score += 0.3 * np.tanh((rsi - 50) / 20)
            score += 0.2 * np.tanh(macd / max(abs(macd) + 0.01, 0.01))
            score += 0.1 * np.tanh((adx - 20) / 20)
        else:
            score += 0.3 * np.tanh((50 - rsi) / 20)
            score += 0.2 * np.tanh(-macd / max(abs(macd) + 0.01, 0.01))
            score += 0.1 * np.tanh((adx - 20) / 20)
        return float(np.clip(score, 0.05, 0.95))

    df["raw_p"] = df.apply(_raw_prob, axis=1)

    out: dict[str, tuple[pd.Series, pd.Series]] = {}
    for regime, sub in df.groupby("regime_std"):
        if regime is None:
            continue
        out[regime] = (sub["raw_p"], sub["hit_target"])
    return out


def fit_per_regime_calibrators() -> dict:
    """對每個 regime 跑 isotonic regression。回傳 dict regime → status。"""
    data = _build_per_regime_data()
    if not data:
        return {"status": "no_data"}

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    for regime, (raw_p, hit) in data.items():
        n = len(hit)
        if n < _MIN_SAMPLES:
            results[regime] = {"status": "skipped_low_samples", "n": n, "min": _MIN_SAMPLES}
            logger.info(f"[per_regime_calibrator] {regime}: 樣本 {n} < {_MIN_SAMPLES}，skip")
            continue
        if hit.nunique() < 2:
            results[regime] = {"status": "skipped_single_class", "n": n}
            continue

        try:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.05, y_max=0.95)
            iso.fit(raw_p.values, hit.values)
            calibrated = iso.predict(raw_p.values)
            from sklearn.metrics import brier_score_loss
            brier_before = float(brier_score_loss(hit, raw_p))
            brier_after = float(brier_score_loss(hit, calibrated))

            path = _MODELS_DIR / f"calibrator_{regime}.pkl"
            joblib.dump({
                "model": iso,
                "regime": regime,
                "n_samples": n,
                "brier_before": brier_before,
                "brier_after": brier_after,
                "trained_at": datetime.now().isoformat(),
            }, path)
            results[regime] = {
                "status": "saved", "n": n,
                "brier_before": round(brier_before, 4),
                "brier_after": round(brier_after, 4),
                "improvement": round(brier_before - brier_after, 4),
                "path": str(path),
            }
            logger.info(
                f"[per_regime_calibrator] {regime}: n={n} brier "
                f"{brier_before:.3f}→{brier_after:.3f} (Δ {brier_before-brier_after:+.3f})"
            )
        except Exception as e:
            results[regime] = {"status": "error", "error": str(e)}
            logger.warning(f"[per_regime_calibrator] {regime} 失敗: {e}")

    return results


def load_calibrator(regime: str) -> Optional[object]:
    """讀某 regime 的 calibrator，沒檔回 None。"""
    path = _MODELS_DIR / f"calibrator_{regime}.pkl"
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def calibrate_probability(raw_p: float, regime: str) -> float:
    """推論時呼叫：把 raw probability 用 per-regime calibrator 校準。

    若該 regime 沒 calibrator（樣本不足）→ 直接回原值。
    """
    cal = load_calibrator(regime)
    if cal is None:
        return float(raw_p)
    try:
        return float(cal["model"].predict([raw_p])[0])
    except Exception:
        return float(raw_p)


# ─── Per-regime walk-forward（v105 Phase C2）──────────────────────


def per_regime_walk_forward_summary() -> dict:
    """各 regime 跑 TimeSeriesSplit walk-forward 統計（含 lockbox 20%）。

    回傳每個 regime 的 walk-forward AUC + lockbox AUC + sample count。
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score

    data = _build_per_regime_data()
    if not data:
        return {"status": "no_data"}

    out: dict[str, dict] = {}
    for regime, (raw_p, hit) in data.items():
        n = len(hit)
        if n < _MIN_SAMPLES:
            out[regime] = {"status": "skipped_low_samples", "n": n}
            continue
        if hit.nunique() < 2:
            out[regime] = {"status": "skipped_single_class", "n": n}
            continue

        # Lockbox 20% 最後段
        lockbox_n = max(int(n * 0.2), 5)
        train_p = raw_p.iloc[:-lockbox_n].values
        train_y = hit.iloc[:-lockbox_n].values
        lockbox_p = raw_p.iloc[-lockbox_n:].values
        lockbox_y = hit.iloc[-lockbox_n:].values

        # Walk-forward CV（用 raw_p 跟 hit 之間的 AUC，無模型）
        wf_aucs = []
        if len(train_y) >= 20 and len(set(train_y)) == 2:
            try:
                tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(train_y) // 15)))
                for _, val_idx in tscv.split(train_p):
                    val_y = train_y[val_idx]
                    val_p = train_p[val_idx]
                    if len(set(val_y)) == 2:
                        wf_aucs.append(float(roc_auc_score(val_y, val_p)))
            except Exception:
                pass

        # Lockbox AUC
        lockbox_auc = None
        if len(set(lockbox_y)) == 2:
            try:
                lockbox_auc = float(roc_auc_score(lockbox_y, lockbox_p))
            except Exception:
                pass

        out[regime] = {
            "status": "ok",
            "n_train": len(train_y),
            "n_lockbox": len(lockbox_y),
            "wf_auc_mean": round(float(np.mean(wf_aucs)), 4) if wf_aucs else None,
            "wf_auc_std": round(float(np.std(wf_aucs)), 4) if wf_aucs else None,
            "wf_n_folds": len(wf_aucs),
            "lockbox_auc": round(lockbox_auc, 4) if lockbox_auc is not None else None,
        }

    return out


if __name__ == "__main__":
    import json
    print("─── Per-regime isotonic 校準 ───")
    r1 = fit_per_regime_calibrators()
    print(json.dumps(r1, ensure_ascii=False, indent=2, default=str))

    print()
    print("─── Per-regime walk-forward ───")
    r2 = per_regime_walk_forward_summary()
    print(json.dumps(r2, ensure_ascii=False, indent=2, default=str))
