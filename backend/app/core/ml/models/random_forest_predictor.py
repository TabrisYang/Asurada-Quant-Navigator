"""阿斯拉量化系統 — Random Forest 預測模型"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
from loguru import logger

from app.core.ml.base import BasePredictor, TrainResult
from app.core.ml.registry import model_registry

_DEFAULT_CONFIG = {
    "n_estimators": 500,
    "max_depth": 8,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "random_state": 42,
}


@model_registry.register(
    id="random_forest",
    name="Random Forest",
    category="集成學習",
    description="隨機森林，不易過擬合、穩定性高，數據量少時的保守選擇",
    requires=["sklearn"],
    default_config=_DEFAULT_CONFIG,
    min_samples=200,
    supports_gpu=False,
    training_speed="fast",
)
class RandomForestPredictor(BasePredictor):

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
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
        )

        t0 = time.time()
        cfg = {**_DEFAULT_CONFIG, **(config or {})}
        self._feature_names = list(feature_names)

        pw = cfg.pop("pos_weight", None)
        self._model = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_split=cfg["min_samples_split"],
            min_samples_leaf=cfg["min_samples_leaf"],
            max_features=cfg["max_features"],
            random_state=cfg["random_state"],
            class_weight="balanced" if pw and pw > 1.5 else None,
            n_jobs=-1,
        )

        self._model.fit(X_train, y_train)

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
            f"RandomForest 訓練完成: OOS acc={result.oos_accuracy:.3f} "
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
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(self._model, f)
        with open(path / "meta.json", "w") as f:
            json.dump({"feature_names": self._feature_names}, f)

    def load(self, path: Path) -> None:
        model_path = path / "model.pkl"
        meta_path = path / "meta.json"
        if not model_path.exists():
            raise FileNotFoundError(f"找不到模型檔案: {model_path}")
        with open(model_path, "rb") as f:
            self._model = pickle.load(f)
        with open(meta_path) as f:
            meta = json.load(f)
        self._feature_names = meta["feature_names"]

    def is_fitted(self) -> bool:
        return self._model is not None
