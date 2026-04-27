"""阿斯拉量化系統 — v101 Drift 偵測（Phase 2.5）

兩個獨立信號偵測「資料分布漂移」：

1. **Adversarial Validation**：
   訓練一個分類器分辨「樣本來自 train set 還是 production」。
   AUC > 0.65 = 顯著漂移（兩組樣本可被分辨 → 分布不同）。

2. **PSI（Population Stability Index）**：
   對每個特徵每週算 PSI（最近 30 天 vs 訓練集分布）。
   PSI 0.1-0.25 = 注意；PSI ≥ 0.25 = 重大漂移 → 強制重訓。

每週由 launchd 跑（在 retrain 之前 / quality gate 之前）。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

import lightgbm as lgb

from app.core.feature_extractor import FEATURE_COLUMNS
from app.core.prediction_tracker import prediction_tracker


def adversarial_validation(recent_days: int = 30, train_days_min: int = 60) -> dict:
    """訓練 / 推論資料分布是否一致？

    Returns:
        dict 含 auc、drift_detected（bool 或 "warning"）、action 建議
    """
    prediction_tracker._ensure_db()
    if not prediction_tracker._conn:
        return {"auc": None, "drift_detected": False, "action": "no_action", "reason": "DB 未初始化"}

    # 取最近 30 天 vs 訓練集（train_days_min 天前）
    now = datetime.now()
    recent_cutoff = (now - timedelta(days=recent_days)).isoformat()
    train_cutoff = (now - timedelta(days=train_days_min + recent_days)).isoformat()

    cursor = prediction_tracker._conn.execute(
        """SELECT pf.*, p.created_at FROM prediction_features pf
           JOIN predictions p ON pf.prediction_id = p.id
           WHERE p.status IN ('hit_target', 'hit_stop')
           ORDER BY p.created_at ASC"""
    )
    cols = [d[0] for d in cursor.description]
    rows = cursor.fetchall()
    if len(rows) < 30:
        return {"auc": None, "drift_detected": False, "action": "no_action", "reason": f"樣本不足 ({len(rows)} < 30)"}

    df = pd.DataFrame(rows, columns=cols)
    train_set = df[df["created_at"] < recent_cutoff].copy()
    recent_set = df[df["created_at"] >= recent_cutoff].copy()

    if len(train_set) < 20 or len(recent_set) < 5:
        return {
            "auc": None, "drift_detected": False, "action": "no_action",
            "reason": f"切分後樣本不足（train {len(train_set)}, recent {len(recent_set)}）",
        }

    # 標籤：0 = train, 1 = recent
    train_set["_is_recent"] = 0
    recent_set["_is_recent"] = 1
    combined = pd.concat([train_set, recent_set], ignore_index=True)

    X = combined[FEATURE_COLUMNS].copy()
    for col in FEATURE_COLUMNS:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0).astype(float)
    y = combined["_is_recent"].astype(int)

    # CV AUC（5-fold）
    try:
        clf = lgb.LGBMClassifier(
            n_estimators=50, max_depth=4, learning_rate=0.05,
            class_weight="balanced", random_state=42, verbose=-1,
        )
        n_splits = min(5, int(y.value_counts().min()))
        if n_splits < 2:
            return {"auc": None, "drift_detected": False, "action": "no_action", "reason": "類別不平衡"}
        scores = cross_val_score(clf, X, y, cv=n_splits, scoring="roc_auc", n_jobs=1)
        auc = float(np.mean(scores))
    except Exception as e:
        logger.warning(f"adversarial_validation 失敗：{e}")
        return {"auc": None, "drift_detected": False, "action": "no_action", "reason": str(e)}

    # 判定
    if auc > 0.65:
        action = "force_retrain"
        drift_detected = True
        msg = f"顯著漂移（AUC={auc:.3f} > 0.65）→ 強制重訓"
    elif auc > 0.55:
        action = "increase_retrain_freq"
        drift_detected = "warning"
        msg = f"輕微漂移（AUC={auc:.3f}）→ 建議提高重訓頻率"
    else:
        action = "no_action"
        drift_detected = False
        msg = f"分布穩定（AUC={auc:.3f}）"

    logger.info(f"[drift_monitor] {msg}")
    return {
        "auc": round(auc, 4),
        "drift_detected": drift_detected,
        "action": action,
        "message": msg,
        "n_train": len(train_set),
        "n_recent": len(recent_set),
    }


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index for single feature.

    PSI < 0.1：穩定
    0.1 ≤ PSI < 0.25：注意
    PSI ≥ 0.25：重大漂移
    """
    # 過濾 NaN
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    if len(reference) < 5 or len(current) < 5:
        return 0.0

    # 用 reference 的 quantile 分桶
    try:
        bin_edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
        bin_edges[0] -= 1e-9
        bin_edges[-1] += 1e-9
    except Exception:
        return 0.0

    # 補偏（避免 log(0)）
    eps = 1e-6
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)
    ref_pct = ref_counts / max(len(reference), 1) + eps
    cur_pct = cur_counts / max(len(current), 1) + eps

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def compute_psi_per_feature(recent_days: int = 30) -> dict:
    """對每個特徵算 PSI — 找出哪些特徵漂移最嚴重。"""
    prediction_tracker._ensure_db()
    if not prediction_tracker._conn:
        return {"error": "DB 未初始化"}

    now = datetime.now()
    recent_cutoff = (now - timedelta(days=recent_days)).isoformat()

    cursor = prediction_tracker._conn.execute(
        """SELECT pf.*, p.created_at FROM prediction_features pf
           JOIN predictions p ON pf.prediction_id = p.id
           WHERE p.status IN ('hit_target', 'hit_stop')"""
    )
    cols = [d[0] for d in cursor.description]
    rows = cursor.fetchall()

    df = pd.DataFrame(rows, columns=cols)
    train = df[df["created_at"] < recent_cutoff]
    recent = df[df["created_at"] >= recent_cutoff]

    if len(train) < 20 or len(recent) < 5:
        return {"error": "樣本不足", "n_train": len(train), "n_recent": len(recent)}

    psi_results = {}
    for col in FEATURE_COLUMNS:
        try:
            r = pd.to_numeric(train[col], errors="coerce").to_numpy()
            c = pd.to_numeric(recent[col], errors="coerce").to_numpy()
            psi = compute_psi(r, c)
            psi_results[col] = round(psi, 4)
        except Exception:
            psi_results[col] = None

    # 找最大漂移
    valid = [(k, v) for k, v in psi_results.items() if v is not None]
    valid.sort(key=lambda x: x[1], reverse=True)
    top_drift = valid[:5]

    max_psi = max((v for _, v in valid), default=0)
    if max_psi > 0.25:
        severity = "重大漂移"
    elif max_psi > 0.1:
        severity = "輕微漂移"
    else:
        severity = "穩定"

    return {
        "max_psi": max_psi,
        "severity": severity,
        "top_drift": top_drift,
        "all_psi": psi_results,
    }
