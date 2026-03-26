"""阿斯拉量化系統 — ML 模型註冊中心

仿照 IndicatorRegistry 設計，用裝飾器自動註冊模型。
新增模型只需繼承 BasePredictor + @model_registry.register(...)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Type

from loguru import logger

from app.core.ml.base import BasePredictor


@dataclass
class ModelDefinition:
    """模型定義"""
    id: str
    name: str
    category: str               # 梯度提升樹 / 神經網路 / 其他
    description: str
    predictor_cls: Type[BasePredictor]
    requires: list[str] = field(default_factory=list)   # 依賴套件
    default_config: dict[str, Any] = field(default_factory=dict)
    min_samples: int = 300      # 最低訓練樣本數
    supports_gpu: bool = False
    training_speed: str = "fast"  # fast / medium / slow


class ModelRegistry:
    """ML 模型註冊中心"""

    def __init__(self):
        self._models: dict[str, ModelDefinition] = {}

    def register(
        self,
        id: str,
        name: str,
        category: str,
        description: str,
        requires: list[str] | None = None,
        default_config: dict[str, Any] | None = None,
        min_samples: int = 300,
        supports_gpu: bool = False,
        training_speed: str = "fast",
    ):
        """裝飾器：註冊 ML 模型類別"""
        def decorator(cls: Type[BasePredictor]):
            self._models[id] = ModelDefinition(
                id=id,
                name=name,
                category=category,
                description=description,
                predictor_cls=cls,
                requires=requires or [],
                default_config=default_config or {},
                min_samples=min_samples,
                supports_gpu=supports_gpu,
                training_speed=training_speed,
            )
            return cls
        return decorator

    def get(self, model_id: str) -> Optional[ModelDefinition]:
        return self._models.get(model_id)

    def list_all(self) -> list[ModelDefinition]:
        return list(self._models.values())

    def list_available(self) -> list[ModelDefinition]:
        """只回傳依賴已安裝的模型"""
        available = []
        for defn in self._models.values():
            if self._check_deps(defn.requires):
                available.append(defn)
        return available

    def create_predictor(self, model_id: str) -> Optional[BasePredictor]:
        """建立模型實例"""
        defn = self.get(model_id)
        if not defn:
            logger.warning(f"未知的模型 ID: {model_id}")
            return None
        if not self._check_deps(defn.requires):
            logger.warning(
                f"模型 {model_id} 依賴未安裝: {defn.requires}"
            )
            return None
        return defn.predictor_cls()

    def to_info_list(self) -> list[dict]:
        """轉換為前端用的模型資訊清單"""
        result = []
        for defn in self._models.values():
            installed = self._check_deps(defn.requires)
            result.append({
                "id": defn.id,
                "name": defn.name,
                "category": defn.category,
                "description": defn.description,
                "available": installed,
                "missing_deps": defn.requires if not installed else [],
                "min_samples": defn.min_samples,
                "supports_gpu": defn.supports_gpu,
                "training_speed": defn.training_speed,
                "default_config": defn.default_config,
            })
        return result

    @staticmethod
    def _check_deps(requires: list[str]) -> bool:
        """檢查依賴套件是否已安裝"""
        import importlib
        for pkg in requires:
            try:
                importlib.import_module(pkg)
            except ImportError:
                return False
        return True


# 全域單例
model_registry = ModelRegistry()
