"""阿斯拉量化系統 — ML 模型抽象基類

所有 ML 模型都必須繼承 BasePredictor 並實作統一介面，
確保上層呼叫端完全不需要知道底層用的是什麼模型。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class TrainResult:
    """訓練結果"""
    oos_accuracy: float = 0.0
    oos_precision: float = 0.0
    oos_recall: float = 0.0
    oos_f1: float = 0.0
    oos_auc: float = 0.0
    train_accuracy: float = 0.0
    feature_importance: dict[str, float] = field(default_factory=dict)
    n_train_samples: int = 0
    n_oos_samples: int = 0
    train_time_ms: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def reliability_level(self) -> str:
        """模型可靠性分級：high / medium / low / unreliable

        - high:       OOS 準確率 > 58% 且樣本 ≥ 100
        - medium:     OOS 準確率 > 55% 且樣本 ≥ 50，或 F1 > 0.55
        - low:        OOS 準確率 > 55% 且樣本 ≥ 30（僅供參考，權重降低）
        - unreliable: 不達標（< 55% 視為無邊際，不納入預測）
        """
        acc = self.oos_accuracy
        f1 = self.oos_f1
        n = self.n_oos_samples

        if acc > 0.58 and n >= 100:
            return "high"
        if (acc > 0.55 and n >= 50) or (f1 > 0.55 and n >= 50):
            return "medium"
        if acc > 0.55 and n >= 30:
            return "low"
        return "unreliable"

    @property
    def is_reliable(self) -> bool:
        """向後相容：reliability_level 不為 unreliable 即視為可靠"""
        return self.reliability_level != "unreliable"

    def to_dict(self) -> dict:
        return {
            "oos_accuracy": round(self.oos_accuracy, 4),
            "oos_precision": round(self.oos_precision, 4),
            "oos_recall": round(self.oos_recall, 4),
            "oos_f1": round(self.oos_f1, 4),
            "oos_auc": round(self.oos_auc, 4),
            "train_accuracy": round(self.train_accuracy, 4),
            "feature_importance": {
                k: round(v, 4) for k, v in self.feature_importance.items()
            },
            "n_train_samples": self.n_train_samples,
            "n_oos_samples": self.n_oos_samples,
            "train_time_ms": self.train_time_ms,
            "is_reliable": self.is_reliable,
            "reliability_level": self.reliability_level,
        }


@dataclass
class PredictResult:
    """預測結果"""
    probability: float = 0.5
    direction: str = "neutral"      # long / short / neutral
    confidence: str = "low"         # high / medium / low
    top_features: list[dict] = field(default_factory=list)
    model_id: str = ""
    model_quality: Optional[dict] = None
    target_info: Optional[dict] = None  # {direction, threshold, lookback}

    def to_dict(self) -> dict:
        return {
            "probability": round(self.probability, 4),
            "direction": self.direction,
            "confidence": self.confidence,
            "top_features": self.top_features,
            "model_id": self.model_id,
            "model_quality": self.model_quality,
            "target_info": self.target_info,
        }


class BasePredictor(ABC):
    """所有 ML 模型的統一抽象介面"""

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str],
        config: dict | None = None,
    ) -> TrainResult:
        """訓練模型

        Args:
            X_train: 訓練特徵矩陣
            y_train: 訓練標籤 (0=跌, 1=漲)
            X_val: 驗證特徵矩陣
            y_val: 驗證標籤
            feature_names: 特徵名稱列表
            config: 模型特定配置（超參數等）

        Returns:
            TrainResult 包含 OOS 指標和特徵重要性
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """預測上漲機率

        Args:
            X: 特徵矩陣 (n_samples, n_features)

        Returns:
            上漲機率陣列 (n_samples,)，值域 [0, 1]
        """

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """取得特徵重要性排名

        Returns:
            {feature_name: importance_score}，已按重要性降序排列
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """序列化模型到檔案"""

    @abstractmethod
    def load(self, path: Path) -> None:
        """從檔案載入模型"""

    @abstractmethod
    def is_fitted(self) -> bool:
        """模型是否已完成訓練"""

    def predict_single(
        self, X: np.ndarray, feature_names: list[str], threshold: float = 0.6
    ) -> PredictResult:
        """預測單筆（最新一根 K 線），輸出結構化結果

        Args:
            X: 特徵向量 (1, n_features) 或 (n_features,)
            feature_names: 特徵名稱
            threshold: 信心門檻

        Returns:
            PredictResult
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        prob = float(self.predict(X)[0])
        importance = self.get_feature_importance()

        top_features = []
        for fname, score in list(importance.items())[:5]:
            idx = feature_names.index(fname) if fname in feature_names else -1
            val = float(X[0, idx]) if 0 <= idx < X.shape[1] else None
            top_features.append({
                "name": fname,
                "importance": round(score, 4),
                "value": round(val, 4) if val is not None else None,
            })

        if prob >= threshold:
            direction = "long"
        elif prob <= (1 - threshold):
            direction = "short"
        else:
            direction = "neutral"

        if abs(prob - 0.5) >= 0.2:
            confidence = "high"
        elif abs(prob - 0.5) >= 0.1:
            confidence = "medium"
        else:
            confidence = "low"

        return PredictResult(
            probability=prob,
            direction=direction,
            confidence=confidence,
            top_features=top_features,
        )
