"""阿斯拉量化系統 — v105 Phase B：因子權重學習器（資料驅動取代經驗值）。

從 199 筆 verified predictions 跑 logistic regression，學出 9 個 bias_score 分量
的最優權重，取代 v104.1 拍腦袋的 0.30/0.20/0.15 權重表。

也跑 PCA 對 39 ML 特徵正交化，給 v101 LightGBM 訓練可選使用。

執行：
    .venv/bin/python -m app.core.factor_weight_learner

存到：
- backend/data/db/bias_score_weights.json（logistic 權重）
- backend/models/factor_pca.pkl（PCA 模型）
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

_BACKEND_DIR = Path(__file__).resolve().parents[2]
_WEIGHTS_PATH = _BACKEND_DIR / "data" / "db" / "bias_score_weights.json"
_PCA_PATH = _BACKEND_DIR / "models" / "factor_pca.pkl"


# ─── Bias 分量名稱（順序對應 _compute_bias_score 的 contributions）──────
# 跟 regime_subtype.py 的分量保持一致
_BIAS_FEATURE_NAMES = [
    "ema60_slope_signed",       # +0.3 / -0.3 / 0
    "rsi_signed",               # +0.2 / -0.2 / 0
    "breadth_signed",           # 軟加 -0.20 ~ +0.20
    "funding_signed",           # +0.15 / -0.15 / 0
    "ls_ratio_signed",          # +0.15 / -0.15 / 0
    "rs_signed",                # +0.10 / -0.10 / 0
    "regime_market_signed",     # ±0.15 矩陣
    "divergence_signed",        # ±0.20 / ±0.10
]


def _build_training_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """從 prediction_features 表抽訓練資料。

    Returns:
        (X_bias_features, y, X_ml_features)
        - X_bias_features: 8 個 bias 分量（199 × 8）
        - y: hit_target=1 / hit_stop=0
        - X_ml_features: 39 個 ML 特徵（給 PCA 用）
    """
    from app.core.prediction_tracker import prediction_tracker
    prediction_tracker._ensure_db()
    if not prediction_tracker._conn:
        raise RuntimeError("DB 未初始化")

    rows = prediction_tracker._conn.execute(
        """SELECT pf.*, p.status, p.regime, p.regime_std, p.created_at
           FROM prediction_features pf
           JOIN predictions p ON pf.prediction_id = p.id
           WHERE p.status IN ('hit_target', 'hit_stop')"""
    ).fetchall()
    if not rows:
        raise RuntimeError("沒有 verified samples")

    cols = [d[0] for d in prediction_tracker._conn.execute(
        "SELECT * FROM prediction_features LIMIT 1"
    ).description] + ["status", "regime", "regime_std", "created_at"]

    df = pd.DataFrame(rows, columns=cols)
    y = (df["status"] == "hit_target").astype(int)

    # ─── 從 prediction_features 計算「bias 分量符號值」───
    # 這些值是「方向化分數」（正=偏多、負=偏空、0=中性），不是原始指標值
    # 同 _compute_bias_score 的邏輯（簡化版，用線性閾值）
    bias_x = pd.DataFrame()

    # 1. EMA60 斜率（從 close_return_20 / 20 換算近似）
    if "close_return_20" in df.columns:
        slope = pd.to_numeric(df["close_return_20"], errors="coerce") / 20.0  # 每根 K 線平均 %
        bias_x["ema60_slope_signed"] = np.where(
            slope > 0.05, 0.3, np.where(slope < -0.05, -0.3, 0.0)
        )
    else:
        bias_x["ema60_slope_signed"] = 0.0

    # 2. RSI
    rsi = pd.to_numeric(df["rsi_14"], errors="coerce")
    bias_x["rsi_signed"] = np.where(rsi >= 60, 0.2, np.where(rsi <= 40, -0.2, 0.0))

    # 3. breadth（軟加線性）
    breadth = pd.to_numeric(df["breadth_pct"], errors="coerce")
    def _breadth_sign(b):
        if pd.isna(b): return 0.0
        if b >= 65: return 0.20
        if b >= 55: return 0.10 + (b - 55) / 10 * 0.10
        if b <= 35: return -0.20
        if b <= 45: return -0.10 - (45 - b) / 10 * 0.10
        return 0.0
    bias_x["breadth_signed"] = breadth.apply(_breadth_sign)

    # 4-5: funding / LS — prediction_features 沒這欄，當 0
    bias_x["funding_signed"] = 0.0
    bias_x["ls_ratio_signed"] = 0.0

    # 6: RS（vs basket）
    rs = pd.to_numeric(df["rs_vs_basket"], errors="coerce")
    bias_x["rs_signed"] = np.where(rs > 1.2, 0.10, np.where(rs < 0.8, -0.10, 0.0))

    # 7: market_regime 矩陣 — 沒 symbol 對應分類，用 regime 簡化
    # trending_up regime 視為 +0.15，trending_down 視為 -0.15
    bias_x["regime_market_signed"] = df["regime_std"].map({
        "trending_up": 0.15, "trending_down": -0.15,
    }).fillna(0.0)

    # 8: divergence — prediction_features 沒這欄（v104.x 之後才有 RSI_Div）→ 0
    bias_x["divergence_signed"] = 0.0

    # ML 特徵（39+ 個）— 給 PCA 用，全部用數值欄位
    from app.core.feature_extractor import FEATURE_COLUMNS
    ml_x = df[[c for c in FEATURE_COLUMNS if c in df.columns]].copy()
    for c in ml_x.columns:
        ml_x[c] = pd.to_numeric(ml_x[c], errors="coerce")
    ml_x = ml_x.fillna(0)

    return bias_x.fillna(0), y, ml_x


def fit_bias_weights(min_samples: int = 50) -> dict:
    """跑 logistic regression 學 8 個 bias 分量的權重。

    Returns dict 含學到的權重 + metrics + lockbox AUC。
    """
    bias_x, y, _ = _build_training_data()
    n = len(y)
    if n < min_samples:
        return {"status": "insufficient_samples", "n": n}
    if y.nunique() < 2:
        return {"status": "single_class", "n": n}

    # ─── L2 正則化 logistic regression with CV ───
    # CV 自動選 C（正則化強度），lockbox 取最後 20% 不參與訓練
    lockbox_n = max(int(n * 0.2), 10)
    train_x = bias_x.iloc[:-lockbox_n]
    train_y = y.iloc[:-lockbox_n]
    lockbox_x = bias_x.iloc[-lockbox_n:]
    lockbox_y = y.iloc[-lockbox_n:]

    if train_y.nunique() < 2:
        return {"status": "single_class_train", "n_train": len(train_y)}

    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    lockbox_x_scaled = scaler.transform(lockbox_x)

    model = LogisticRegressionCV(
        Cs=[0.1, 0.5, 1.0, 5.0],
        cv=min(5, train_y.value_counts().min()) if train_y.value_counts().min() >= 2 else 2,
        penalty="l2",
        max_iter=1000,
        scoring="neg_log_loss",
        random_state=42,
    )
    try:
        model.fit(train_x_scaled, train_y)
    except Exception as e:
        return {"status": "training_error", "error": str(e), "n": n}

    # 學到的權重（在 scaled 空間，要反推回原空間）
    coefs = model.coef_[0]
    # scaled coefs / std → 原空間 coefs（更直觀）
    scaled_to_raw = coefs / scaler.scale_
    intercept = float(model.intercept_[0] - np.dot(coefs, scaler.mean_ / scaler.scale_))

    # Lockbox AUC
    from sklearn.metrics import roc_auc_score, brier_score_loss
    if lockbox_y.nunique() == 2:
        lockbox_proba = model.predict_proba(lockbox_x_scaled)[:, 1]
        lockbox_auc = float(roc_auc_score(lockbox_y, lockbox_proba))
        lockbox_brier = float(brier_score_loss(lockbox_y, lockbox_proba))
    else:
        lockbox_auc = 0.5
        lockbox_brier = 0.25

    weights_dict = {
        name: float(scaled_to_raw[i])
        for i, name in enumerate(bias_x.columns)
    }
    weights_payload = {
        "trained_at": datetime.now().isoformat(),
        "n_train": len(train_y),
        "n_lockbox": len(lockbox_y),
        "lockbox_auc": round(lockbox_auc, 4),
        "lockbox_brier": round(lockbox_brier, 4),
        "selected_C": float(model.C_[0]),
        "weights": weights_dict,
        "intercept": intercept,
        "feature_names": list(bias_x.columns),
    }

    # ─── Quality gate：lockbox AUC < 0.55 → reject，不啟用 ───
    if lockbox_auc < 0.55:
        # 仍存到 _rejected.json 給 debug 用，但 _WEIGHTS_PATH 不更新（生產用 fallback）
        rejected_path = _WEIGHTS_PATH.parent / "bias_score_weights_rejected.json"
        weights_payload["status"] = "rejected_low_auc"
        weights_payload["reason"] = f"lockbox_auc {lockbox_auc:.3f} < 0.55（資料品質不足，建議累積更多帶完整 features 的樣本後重訓）"
        rejected_path.write_text(json.dumps(weights_payload, ensure_ascii=False, indent=2))
        logger.warning(
            f"⚠️ bias_score 權重學習 lockbox AUC {lockbox_auc:.3f} 太低，"
            f"reject 不啟用（fallback 經驗值）。debug 存到 {rejected_path}"
        )
        return weights_payload

    # 存檔（通過 gate 才啟用）
    _WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _WEIGHTS_PATH.write_text(json.dumps(weights_payload, ensure_ascii=False, indent=2))
    logger.info(
        f"✅ bias_score 權重學習完成 — n={len(train_y)} lockbox_AUC={lockbox_auc:.3f} "
        f"weights stored to {_WEIGHTS_PATH}"
    )
    return {"status": "saved", **weights_payload}


def fit_pca_model(min_samples: int = 50, variance_threshold: float = 0.90) -> dict:
    """對 39 ML 特徵跑 PCA，抽 top N 解釋 90% 變異的主成分。"""
    _, _, ml_x = _build_training_data()
    n = len(ml_x)
    if n < min_samples:
        return {"status": "insufficient_samples", "n": n}

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(ml_x)

    pca = PCA(n_components=min(ml_x.shape[1], n - 1))
    pca.fit(x_scaled)

    # 找解釋 ≥ variance_threshold 變異的最少主成分數
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_components_chosen = int(np.argmax(cumsum >= variance_threshold) + 1)

    # 重新訓練固定 n_components
    pca_final = PCA(n_components=n_components_chosen)
    pca_final.fit(x_scaled)

    _PCA_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "scaler": scaler,
        "pca": pca_final,
        "feature_names": list(ml_x.columns),
        "n_components": n_components_chosen,
        "explained_variance_ratio": pca_final.explained_variance_ratio_.tolist(),
        "trained_at": datetime.now().isoformat(),
    }, _PCA_PATH)

    logger.info(
        f"✅ PCA 訓練完成 — 抽 {n_components_chosen} 個 PC（解釋 "
        f"{cumsum[n_components_chosen-1]:.1%} 變異）stored to {_PCA_PATH}"
    )
    return {
        "status": "saved",
        "n_samples": n,
        "n_components": n_components_chosen,
        "explained_variance": float(cumsum[n_components_chosen - 1]),
    }


def load_learned_weights() -> Optional[dict]:
    """讀 bias_score_weights.json，沒檔回 None。"""
    if not _WEIGHTS_PATH.exists():
        return None
    try:
        return json.loads(_WEIGHTS_PATH.read_text())
    except Exception:
        return None


if __name__ == "__main__":
    print("─── 訓練 bias_score 權重 ───")
    r1 = fit_bias_weights()
    print(json.dumps(r1, ensure_ascii=False, indent=2, default=str))

    print()
    print("─── 訓練 PCA ───")
    r2 = fit_pca_model()
    print(json.dumps(r2, ensure_ascii=False, indent=2, default=str))
