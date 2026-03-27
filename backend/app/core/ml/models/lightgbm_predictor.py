"""阿斯拉量化系統 — LightGBM 預測模型"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from app.core.ml.base import BasePredictor, TrainResult
from app.core.ml.registry import model_registry

_DEFAULT_CONFIG = {
    "n_estimators": 300,
    "max_depth": 5,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_samples": 20,
    "random_state": 42,
}


@model_registry.register(
    id="lightgbm",
    name="LightGBM",
    category="梯度提升樹",
    description="微軟開源的高效梯度提升框架，訓練快速，處理缺失值能力強，適合中高頻交易信號",
    requires=["lightgbm"],
    default_config=_DEFAULT_CONFIG,
    min_samples=300,
    supports_gpu=True,
    training_speed="fast",
)
class LightGBMPredictor(BasePredictor):

    def __init__(self):
        self._model = None
        self._feature_names: list[str] = []

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str],
        config: dict | None = None,
    ) -> TrainResult:
        import lightgbm as lgb
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
        )

        t0 = time.time()
        cfg = {**_DEFAULT_CONFIG, **(config or {})}
        self._feature_names = list(feature_names)

        pw = cfg.pop("pos_weight", None)
        self._model = lgb.LGBMClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            num_leaves=cfg["num_leaves"],
            learning_rate=cfg["learning_rate"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            reg_alpha=cfg["reg_alpha"],
            reg_lambda=cfg["reg_lambda"],
            min_child_samples=cfg["min_child_samples"],
            random_state=cfg["random_state"],
            scale_pos_weight=pw if pw and pw > 1.0 else 1.0,
            is_unbalance=True,
            verbose=-1,
            n_jobs=-1,
        )

        self._model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )

        y_pred = self._model.predict(X_val)
        y_prob = self._model.predict_proba(X_val)[:, 1]
        y_train_pred = self._model.predict(X_train)

        importance = dict(zip(
            self._feature_names,
            self._model.feature_importances_.astype(float),
        ))
        importance = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )

        elapsed_ms = int((time.time() - t0) * 1000)
        result = TrainResult(
            oos_accuracy=float(accuracy_score(y_val, y_pred)),
            oos_precision=float(precision_score(y_val, y_pred, zero_division=0)),
            oos_recall=float(recall_score(y_val, y_pred, zero_division=0)),
            oos_f1=float(f1_score(y_val, y_pred, zero_division=0)),
            oos_auc=float(roc_auc_score(y_val, y_prob)) if len(set(y_val)) > 1 else 0.5,
            train_accuracy=float(accuracy_score(y_train, y_train_pred)),
            feature_importance=importance,
            n_train_samples=len(y_train),
            n_oos_samples=len(y_val),
            train_time_ms=elapsed_ms,
        )

        logger.info(
            f"LightGBM 訓練完成: OOS acc={result.oos_accuracy:.3f} "
            f"AUC={result.oos_auc:.3f} ({elapsed_ms}ms)"
        )
        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("模型尚未訓練")
        return self._model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> dict[str, float]:
        if self._model is None:
            return {}
        importance = dict(zip(
            self._feature_names,
            self._model.feature_importances_.astype(float),
        ))
        total = sum(importance.values()) or 1.0
        return {
            k: round(v / total, 4)
            for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True)
        }

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("模型尚未訓練，無法儲存")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.booster_.save_model(str(path / "model.txt"))
        with open(path / "meta.json", "w") as f:
            json.dump({"feature_names": self._feature_names}, f)

    def load(self, path: Path) -> None:
        import lightgbm as lgb
        model_path = path / "model.txt"
        meta_path = path / "meta.json"
        if not model_path.exists():
            raise FileNotFoundError(f"找不到模型檔案: {model_path}")
        booster = lgb.Booster(model_file=str(model_path))
        self._model = _BoosterWrapper(booster)
        with open(meta_path) as f:
            meta = json.load(f)
        self._feature_names = meta["feature_names"]

    def is_fitted(self) -> bool:
        return self._model is not None


class _BoosterWrapper:
    """包裝 Booster 使其具備 predict_proba 和 feature_importances_ 介面"""

    def __init__(self, booster):
        self._booster = booster

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self._booster.predict(X)
        if raw.ndim == 1:
            return np.column_stack([1 - raw, raw])
        return raw

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.array(self._booster.feature_importance(importance_type="gain"),
                        dtype=float)
