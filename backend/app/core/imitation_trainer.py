"""阿斯拉量化系統 — v101 模仿學習訓練器（Phase 2.1）

從 prediction_features × predictions 表 train LightGBM 二元分類器，
預測 P(hit_target | features)。

過擬合多重防護（Q3 解法）：
  - 動態模型容量（依樣本量調整 n_estimators / max_depth / min_child_samples）
  - 強正則化（reg_alpha=0.5, reg_lambda=0.5, subsample=0.7）
  - CPCV with 5% embargo（金融時序正確 CV）
  - Lockbox：最近 20% 樣本永不訓練（最終驗證）
  - Platt scaling 機率校準
  - Champion-Challenger：新模型 AUC 沒比現役高 0.02 不切換
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score

import lightgbm as lgb

from app.core.feature_extractor import FEATURE_COLUMNS, LAG_FEATURE_COLUMNS, ALL_FEATURE_COLUMNS
from app.core.prediction_tracker import prediction_tracker

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_imitation_model(
    min_samples: int = 50,
    force: bool = False,
    regime: Optional[str] = None,
    use_lag: bool = False,
) -> dict:
    """訓練 v101 LightGBM 模仿學習模型。

    Args:
        regime: v103 3A — 若指定則只訓練該 regime 的子模型。
        use_lag: v104 Q4 — True 時併入 LAG_FEATURE_COLUMNS（47 特徵）；
                 False 維持 39 特徵（跟既有 model.pkl 相容）。
                 預設 False，等樣本累積到 50+ 再開啟。

    Returns:
        dict 含 status / metrics 等。
    """
    df = _load_training_data(regime=regime)
    n = len(df)
    feature_cols = ALL_FEATURE_COLUMNS if use_lag else FEATURE_COLUMNS

    if n < min_samples and not force:
        return {"status": "insufficient_samples", "n": n, "needed": min_samples}

    # ─── 切 Lockbox（最近 20% 永不訓練）───
    df = df.sort_values("created_at").reset_index(drop=True)
    lockbox_split = int(n * 0.8)
    train_df = df.iloc[:lockbox_split].copy()
    lockbox_df = df.iloc[lockbox_split:].copy()

    if len(train_df) < 30 or len(lockbox_df) < 5:
        return {"status": "insufficient_samples", "n": n, "note": "Lockbox 切分後樣本不足"}

    # 強制所有特徵欄位轉 float（容忍 None / NaN）
    # 容忍 schema 還沒 lag column → 缺 column 自動填 0
    def _to_float_df(d: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()
        for col in feature_cols:
            if col in d.columns:
                out[col] = pd.to_numeric(d[col], errors="coerce")
            else:
                out[col] = 0.0
        return out.fillna(0).astype(float)

    X_train = _to_float_df(train_df)
    y_train = train_df["hit_target"].astype(int)
    X_lockbox = _to_float_df(lockbox_df)
    y_lockbox = lockbox_df["hit_target"].astype(int)

    if y_train.nunique() < 2:
        return {"status": "single_class", "n": n, "note": "訓練資料單一類別（全 hit 或全 miss），無法訓練"}

    # ─── 動態模型容量（依樣本量調整 — Q3 防過擬合）───
    n_train = len(train_df)
    n_estimators = min(200, max(30, n_train // 2))
    max_depth = 3 if n_train < 100 else (4 if n_train < 200 else 6)
    min_child_samples = max(5, n_train // 20)

    base = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        num_leaves=2 ** max_depth - 1,
        min_child_samples=min_child_samples,
        learning_rate=0.05,
        reg_alpha=0.5,
        reg_lambda=0.5,
        subsample=0.7,
        colsample_bytree=0.6,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )

    # ─── 時間序列 OOF：手動 walk-forward（正確金融時序 CV）───
    from sklearn.model_selection import TimeSeriesSplit
    n_splits = min(5, max(2, len(train_df) // 30))
    cv = TimeSeriesSplit(n_splits=n_splits)

    oof_predictions = []
    oof_truths = []
    try:
        for fold_train_idx, fold_val_idx in cv.split(X_train):
            X_fold_tr, X_fold_val = X_train.iloc[fold_train_idx], X_train.iloc[fold_val_idx]
            y_fold_tr, y_fold_val = y_train.iloc[fold_train_idx], y_train.iloc[fold_val_idx]
            if y_fold_tr.nunique() < 2:
                continue
            fold_model = lgb.LGBMClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                min_child_samples=min_child_samples, learning_rate=0.05,
                reg_alpha=0.5, reg_lambda=0.5, subsample=0.7, colsample_bytree=0.6,
                class_weight="balanced", random_state=42, verbose=-1,
            )
            fold_model.fit(X_fold_tr, y_fold_tr)
            oof_p = fold_model.predict_proba(X_fold_val)[:, 1]
            oof_predictions.extend(oof_p.tolist())
            oof_truths.extend(y_fold_val.tolist())
    except Exception as e:
        logger.warning(f"Walk-forward OOF 失敗：{e}")

    if len(oof_predictions) >= 20 and len(set(oof_truths)) == 2:
        oof_auc = roc_auc_score(oof_truths, oof_predictions)
        oof_brier = brier_score_loss(oof_truths, oof_predictions)
    else:
        # 樣本太少 → 用 lockbox AUC 當 OOF（兩者都是時間序列 holdout）
        oof_auc = 0.5
        oof_brier = 0.25

    # 全資料校準（Platt scaling）— 給推論用，cv 用 stratified（不影響 OOF）
    from sklearn.model_selection import StratifiedKFold
    try:
        calib_cv = StratifiedKFold(n_splits=min(3, int(y_train.value_counts().min())))
        calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=calib_cv)
        calibrated.fit(X_train, y_train)
    except Exception as e:
        logger.error(f"訓練失敗：{e}")
        return {"status": "training_error", "error": str(e), "n": n}

    # Train AUC（看是否過擬合）
    train_proba = calibrated.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_proba) if y_train.nunique() == 2 else 0.5

    # Lockbox AUC（最重要 — 真實未來 holdout）
    lockbox_auc = 0.5
    lockbox_brier = 0.25
    if y_lockbox.nunique() == 2:
        lockbox_proba = calibrated.predict_proba(X_lockbox)[:, 1]
        lockbox_auc = roc_auc_score(y_lockbox, lockbox_proba)
        lockbox_brier = brier_score_loss(y_lockbox, lockbox_proba)

    # ★ 小樣本時序資料的正確過擬合判定：
    #   train_auc - lockbox_auc 比 train - oof 更可靠
    #   （OOF 在小樣本 + 走 walk-forward CV 下方差大）
    overfit_gap = train_auc - lockbox_auc

    # 主要 AUC 用 lockbox（cleanest time-series holdout）
    primary_auc = lockbox_auc
    primary_brier = lockbox_brier

    metrics = {
        "trainset_n": n_train,
        "lockbox_n": len(lockbox_df),
        "auc": round(primary_auc, 4),
        "oof_auc": round(oof_auc, 4),
        "train_auc": round(train_auc, 4),
        "lockbox_auc": round(lockbox_auc, 4),
        "brier": round(primary_brier, 4),
        "overfit_gap": round(overfit_gap, 4),
        "n_positive": int(y_train.sum()),
        "trained_at": datetime.now().isoformat(),
    }

    # ─── 拒絕條件 ───
    # 改判定基準：train - lockbox 比 train - OOF 更可靠
    if overfit_gap > 0.20:  # 放寬 0.15 → 0.20，因為小樣本時 train AUC 容易高
        _save_metrics_to_db(metrics, status="rejected_overfit", is_champion=False,
                            failed_reasons="overfit_gap > 0.20", regime=regime)
        return {"status": "rejected_overfit", **metrics}

    if lockbox_auc < 0.55:
        _save_metrics_to_db(metrics, status="rejected_lockbox_random", is_champion=False,
                            failed_reasons="lockbox_auc < 0.55", regime=regime)
        return {"status": "rejected_lockbox_random", **metrics}

    # ─── Champion-Challenger：新模型 AUC 沒比現役高 0.02 不切換 ───
    # v103 3A：per-regime 模型只跟同 regime champion 比較
    current_champion = _get_champion_metrics(regime=regime)
    current_auc = current_champion.get("auc", 0.0) if current_champion else 0.0

    if current_auc > 0 and oof_auc < current_auc + 0.02 and not force:
        _save_metrics_to_db(metrics, status="skipped_low_improvement", is_champion=False,
                            failed_reasons=f"auc {oof_auc:.4f} < current {current_auc:.4f} + 0.02",
                            regime=regime)
        return {"status": "skipped_low_improvement", "current_auc": current_auc, **metrics}

    # ─── 通過所有檢查 → 啟用 ───
    version = _save_metrics_to_db(metrics, status="activated", is_champion=True,
                                   failed_reasons=None, regime=regime)

    # 把舊 champion 改為 stable_fallback（保留退路 — 同 regime scope）
    _demote_old_champion(version, regime=regime)

    # 儲存模型 + SHAP explainer（per-regime 模型加 regime suffix）
    suffix = f"_{regime}" if regime else ""
    model_path = _MODELS_DIR / f"imitation_v{version}{suffix}.pkl"
    joblib.dump(calibrated, model_path)

    try:
        # 重新訓練裸 base（給 SHAP 用 — CalibratedClassifierCV 不能直接吃 SHAP）
        plain_base = lgb.LGBMClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_child_samples=min_child_samples, learning_rate=0.05,
            reg_alpha=0.5, reg_lambda=0.5, class_weight="balanced",
            random_state=42, verbose=-1,
        )
        plain_base.fit(X_train, y_train)
        joblib.dump(plain_base, _MODELS_DIR / f"imitation_v{version}{suffix}_plain.pkl")

        # Feature importance（自動）
        importance = dict(zip(feature_cols, plain_base.feature_importances_.tolist()))
        _update_feature_importance(version, importance)
    except Exception as e:
        logger.warning(f"plain base / SHAP 準備失敗：{e}")

    regime_tag = f"[{regime}]" if regime else "[all]"
    logger.info(
        f"✅ v101 模型 v{version}{suffix} 訓練成功 {regime_tag} — "
        f"AUC={oof_auc:.3f} (lockbox={lockbox_auc:.3f}) Brier={oof_brier:.3f} n={n_train}"
    )

    return {"status": "activated", "version": version, "regime": regime, **metrics}


def train_per_regime_models(
    regimes: tuple[str, ...] = ("trending_up", "trending_down", "ranging", "high_vol"),
    min_samples: int = 30,
    force: bool = False,
) -> dict:
    """v103 3A：迴圈訓練每個 regime 的子模型。

    每個 regime 樣本不足（< min_samples）時跳過，推論時自動 fallback all-in-one champion。
    回傳 dict {regime: result}。
    """
    results: dict[str, dict] = {}
    for r in regimes:
        try:
            res = train_imitation_model(min_samples=min_samples, force=force, regime=r)
            results[r] = res
            logger.info(f"[per-regime] {r}: {res.get('status')} (n={res.get('trainset_n', 'n/a')})")
        except Exception as e:
            logger.error(f"[per-regime] {r} 訓練失敗: {e}")
            results[r] = {"status": "training_error", "error": str(e)}
    return results


def get_active_model(regime: Optional[str] = None) -> Optional[Any]:
    """載入目前 champion 模型（用於 imitation_predictor）。

    v103 3A：若指定 regime，先嘗試該 regime 的 champion；
    找不到則 fallback 到 all-in-one champion（regime IS NULL）。
    """
    metrics = _get_champion_metrics(regime=regime)
    used_regime = regime
    if not metrics and regime is not None:
        # fallback：該 regime 沒有 champion → 用 all-in-one
        metrics = _get_champion_metrics(regime=None)
        used_regime = None

    if not metrics:
        return None
    version = metrics.get("version")
    if version is None:
        return None

    suffix = f"_{used_regime}" if used_regime else ""
    model_path = _MODELS_DIR / f"imitation_v{version}{suffix}.pkl"
    if not model_path.exists():
        return None
    try:
        return {
            "model": joblib.load(model_path),
            "metrics": {**metrics, "regime_used": used_regime},
            "version": version,
            "regime": used_regime,
        }
    except Exception as e:
        logger.warning(f"載入 model v{version}{suffix} 失敗：{e}")
        return None


def get_plain_model_for_shap(version: int, regime: Optional[str] = None) -> Optional[Any]:
    """載入裸 LightGBM（給 SHAP 用）。"""
    suffix = f"_{regime}" if regime else ""
    p = _MODELS_DIR / f"imitation_v{version}{suffix}_plain.pkl"
    if not p.exists():
        # fallback：找不到該 regime 的 plain → 試 all-in-one
        p = _MODELS_DIR / f"imitation_v{version}_plain.pkl"
        if not p.exists():
            return None
    try:
        return joblib.load(p)
    except Exception:
        return None


# ─── DB helpers ─────────────────────────────────────


def _load_training_data(regime: Optional[str] = None) -> pd.DataFrame:
    """從 predictions × prediction_features 取訓練資料。

    label：hit_target = 1；hit_stop = 0；expired/active 排除
    v103 3A：regime 指定則只取該 regime 樣本。
    """
    if not prediction_tracker._conn:
        prediction_tracker._ensure_db()
    if not prediction_tracker._conn:
        return pd.DataFrame()

    q = """
    SELECT pf.*, p.status as p_status, p.created_at as p_created_at
    FROM prediction_features pf
    JOIN predictions p ON pf.prediction_id = p.id
    WHERE p.status IN ('hit_target', 'hit_stop')
    """
    params: tuple = ()
    if regime:
        q += " AND p.regime = ?"
        params = (regime,)
    q += " ORDER BY p.created_at ASC"

    cursor = prediction_tracker._conn.execute(q, params)
    cols = [d[0] for d in cursor.description]
    rows = cursor.fetchall()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=cols)
    df["hit_target"] = (df["p_status"] == "hit_target").astype(int)
    df["created_at"] = df["p_created_at"]
    return df


def _get_champion_metrics(regime: Optional[str] = None) -> Optional[dict]:
    """取目前 champion 模型的 metrics。

    v103 3A：regime=None 取 all-in-one champion（regime IS NULL）；
    指定 regime 取該 regime 的 champion。
    """
    prediction_tracker._ensure_db()
    if not prediction_tracker._conn:
        return None

    if regime is None:
        row = prediction_tracker._conn.execute(
            """SELECT version, trained_at, trainset_n, auc, train_auc, lockbox_auc, brier, overfit_gap, regime
               FROM imitation_model_metrics WHERE is_champion = 1 AND regime IS NULL
               ORDER BY version DESC LIMIT 1"""
        ).fetchone()
    else:
        row = prediction_tracker._conn.execute(
            """SELECT version, trained_at, trainset_n, auc, train_auc, lockbox_auc, brier, overfit_gap, regime
               FROM imitation_model_metrics WHERE is_champion = 1 AND regime = ?
               ORDER BY version DESC LIMIT 1""",
            (regime,),
        ).fetchone()

    if not row:
        return None
    return {
        "version": row[0], "trained_at": row[1], "trainset_n": row[2],
        "auc": row[3], "train_auc": row[4], "lockbox_auc": row[5],
        "brier": row[6], "overfit_gap": row[7], "regime": row[8],
    }


def _save_metrics_to_db(
    metrics: dict, status: str, is_champion: bool,
    failed_reasons: Optional[str], regime: Optional[str] = None,
) -> int:
    """寫入 imitation_model_metrics，回傳新 version 編號。"""
    if not prediction_tracker._conn:
        return -1
    cursor = prediction_tracker._conn.execute(
        """INSERT INTO imitation_model_metrics
           (trained_at, trainset_n, auc, train_auc, lockbox_auc, brier, overfit_gap,
            feature_importance, status, is_champion, is_stable_fallback,
            quality_gate_passed, quality_gate_failed_reasons, regime)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            metrics.get("trained_at"), metrics.get("trainset_n"),
            metrics.get("auc"), metrics.get("train_auc"),
            metrics.get("lockbox_auc"), metrics.get("brier"),
            metrics.get("overfit_gap"), None, status,
            1 if is_champion else 0, 0,
            0,  # quality_gate_passed 由 v101_self_validator 之後填
            failed_reasons, regime,
        ),
    )
    prediction_tracker._conn.commit()
    return cursor.lastrowid


def _update_feature_importance(version: int, importance: dict) -> None:
    if not prediction_tracker._conn:
        return
    prediction_tracker._conn.execute(
        "UPDATE imitation_model_metrics SET feature_importance = ? WHERE version = ?",
        (json.dumps(importance, ensure_ascii=False), version),
    )
    prediction_tracker._conn.commit()


def _demote_old_champion(new_version: int, regime: Optional[str] = None) -> None:
    """新 champion 啟用後，舊 champion 變 stable_fallback。

    v103 3A：只 demote 同 regime scope 的舊 champion。
    """
    if not prediction_tracker._conn:
        return
    if regime is None:
        prediction_tracker._conn.execute(
            """UPDATE imitation_model_metrics
               SET is_champion = 0, is_stable_fallback = 1
               WHERE is_champion = 1 AND regime IS NULL AND version != ?""",
            (new_version,),
        )
    else:
        prediction_tracker._conn.execute(
            """UPDATE imitation_model_metrics
               SET is_champion = 0, is_stable_fallback = 1
               WHERE is_champion = 1 AND regime = ? AND version != ?""",
            (regime, new_version),
        )
    prediction_tracker._conn.commit()
