"""阿斯拉量化系統 — 指標註冊中心

所有指標自動註冊，LLM 可查詢可用指標清單。
新增指標只需建立計算函式並用 @register_indicator 裝飾。
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import pandas as pd


@dataclass
class IndicatorDefinition:
    """指標定義"""
    id: str
    name: str
    category: str
    description: str
    parameters: dict[str, dict[str, Any]]  # {param_name: {default, min, max, type}}
    display_mode: str  # overlay / sub_chart / info_panel
    data_source: str  # ohlcv / external
    warmup_periods_func: Callable[[dict], int]  # 根據參數計算暖機期
    calculate_func: Callable[[pd.DataFrame, dict], dict[str, list]]
    pro_tip: str = ""


class IndicatorRegistry:
    """指標註冊中心"""

    def __init__(self):
        self._indicators: dict[str, IndicatorDefinition] = {}

    def register(
        self,
        id: str,
        name: str,
        category: str,
        description: str,
        parameters: dict[str, dict[str, Any]],
        display_mode: str,
        data_source: str = "ohlcv",
        warmup_func: Optional[Callable] = None,
        pro_tip: str = "",
    ):
        """裝飾器：註冊指標計算函式"""
        def decorator(func: Callable):
            self._indicators[id] = IndicatorDefinition(
                id=id,
                name=name,
                category=category,
                description=description,
                parameters=parameters,
                display_mode=display_mode,
                data_source=data_source,
                warmup_periods_func=warmup_func or (lambda p: 0),
                calculate_func=func,
                pro_tip=pro_tip,
            )
            return func
        return decorator

    def get(self, indicator_id: str) -> Optional[IndicatorDefinition]:
        """取得指標定義"""
        return self._indicators.get(indicator_id)

    def list_all(self) -> list[IndicatorDefinition]:
        """取得所有已註冊的指標"""
        return list(self._indicators.values())

    def list_by_category(self, category: str) -> list[IndicatorDefinition]:
        """依類別篩選指標"""
        return [ind for ind in self._indicators.values() if ind.category == category]

    def calculate(
        self,
        indicator_id: str,
        df: pd.DataFrame,
        params: Optional[dict] = None,
    ) -> Optional[dict[str, list]]:
        """計算指標（含空數據與數據量不足防護）"""
        indicator = self.get(indicator_id)
        if not indicator:
            return None

        if df is None or df.empty:
            return None

        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(set(df.columns)):
            return None

        # 合併預設參數
        merged_params = {}
        for key, config in indicator.parameters.items():
            merged_params[key] = params.get(key, config["default"]) if params else config["default"]

        warmup = indicator.warmup_periods_func(merged_params)
        if len(df) < max(warmup, 2):
            return None

        try:
            result = indicator.calculate_func(df, merged_params)
            # Warmup 保護：將暖機期間的值設為 None，避免產生不可靠信號
            if result and warmup > 0:
                for key, values in result.items():
                    if isinstance(values, list) and len(values) > warmup:
                        for i in range(warmup):
                            values[i] = None
            return result
        except Exception as e:
            from loguru import logger
            logger.warning(f"指標 {indicator_id} 計算失敗: {e}")
            return None

    def get_warmup_periods(self, indicator_id: str, params: Optional[dict] = None) -> int:
        """取得暖機期"""
        indicator = self.get(indicator_id)
        if not indicator:
            return 0
        merged_params = {}
        for key, config in indicator.parameters.items():
            merged_params[key] = params.get(key, config["default"]) if params else config["default"]
        return indicator.warmup_periods_func(merged_params)

    def to_info_list(self) -> list[dict]:
        """轉換為前端用的指標資訊清單"""
        result = []
        for ind in self._indicators.values():
            default_params = {k: v["default"] for k, v in ind.parameters.items()}
            result.append({
                "id": ind.id,
                "name": ind.name,
                "category": ind.category,
                "description": ind.description,
                "parameters": ind.parameters,
                "display_mode": ind.display_mode,
                "data_source": ind.data_source,
                "warmup_periods": ind.warmup_periods_func(default_params),
                "pro_tip": ind.pro_tip,
            })
        return result


# 全域單例
registry = IndicatorRegistry()
