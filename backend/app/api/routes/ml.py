"""阿斯拉量化系統 — ML 增強模組 API

端點：
  GET  /api/ml/models     — 列出可用模型
  GET  /api/ml/settings    — 讀取 ML 設定
  PUT  /api/ml/settings    — 更新 ML 設定
  GET  /api/ml/status      — 查詢模型狀態
  POST /api/ml/train       — 訓練模型
  POST /api/ml/predict     — 執行預測
  POST /api/ml/compare     — 多模型比較
  POST /api/ml/retrain     — 手動觸發再訓練
  GET  /api/ml/trained     — 列出所有已訓練模型
"""

from __future__ import annotations

import traceback

from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


def _ensure_models_registered():
    """確保所有 ML 模型已註冊（延遲匯入，避免啟動時載入重型依賴）"""
    from app.core.ml.registry import model_registry
    if len(model_registry.list_all()) == 0:
        import app.core.ml.models  # noqa: F401 — 觸發模型註冊


# ─── 請求模型 ──────────────────────────────────

class MLSettingsUpdate(BaseModel):
    enabled: Optional[str] = None
    model_id: Optional[str] = None
    feature_set: Optional[str] = None
    forward_period: Optional[int] = None
    threshold: Optional[float] = None
    show_explanation: Optional[bool] = None
    train_window: Optional[int] = None
    retrain_interval: Optional[int] = None
    walk_forward: Optional[bool] = None
    wf_windows: Optional[int] = None
    min_samples: Optional[int] = None
    consensus_mode: Optional[str] = None
    target_direction: Optional[str] = None
    target_threshold: Optional[float] = None
    lookback_window: Optional[int] = None


class TrainRequest(BaseModel):
    symbol: str
    timeframe: str = "4h"
    model_id: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class PredictRequest(BaseModel):
    symbol: str
    timeframe: str = "4h"
    model_id: Optional[str] = None
    consensus: bool = False


class CompareRequest(BaseModel):
    symbol: str
    timeframe: str = "4h"
    model_ids: Optional[list[str]] = None
    feature_set: str = "standard"
    forward_period: int = 5


# ─── 端點 ──────────────────────────────────────

@router.get("/models")
async def list_models():
    """列出所有已註冊的 ML 模型"""
    _ensure_models_registered()
    from app.core.ml.registry import model_registry
    return {
        "status": "success",
        "models": model_registry.to_info_list(),
    }


@router.get("/settings")
async def get_settings():
    """讀取使用者 ML 設定"""
    from app.core.ml.model_manager import model_manager
    return {
        "status": "success",
        "settings": model_manager.get_settings(),
    }


@router.put("/settings")
async def update_settings(body: MLSettingsUpdate):
    """更新使用者 ML 設定"""
    from app.core.ml.model_manager import model_manager

    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(400, "沒有提供任何要更新的設定")

    result = model_manager.update_settings(updates)
    return {"status": "success", "settings": result}


@router.get("/status")
async def get_model_status(
    symbol: str = Query(..., description="交易對"),
    timeframe: str = Query("4h", description="K 線級別"),
    model_id: Optional[str] = Query(None, description="模型 ID"),
):
    """查詢已訓練模型的狀態"""
    from app.core.ml.model_manager import model_manager

    status = model_manager.get_model_status(symbol, timeframe, model_id)
    if not status:
        return {"status": "no_model", "message": "尚未訓練模型"}

    return {
        "status": "success",
        "model_status": status,
    }


@router.post("/train")
async def train_model(body: TrainRequest):
    """訓練 ML 模型"""
    _ensure_models_registered()
    from app.core.ml.model_manager import model_manager
    from app.data.fetchers.crypto_engine import crypto_engine

    try:
        df = crypto_engine.load_local_data(
            body.symbol, body.timeframe, body.start_date, body.end_date,
        )
        if df.empty or len(df) < 100:
            return {
                "status": "error",
                "message": f"數據不足（{len(df)} 根），請先同步更多歷史數據",
            }

        result = model_manager.train_model(
            df, body.symbol, body.timeframe, body.model_id,
        )
        return result

    except Exception as e:
        logger.error(f"ML 訓練失敗: {e}\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e)}


@router.post("/predict")
async def predict(body: PredictRequest):
    """用已訓練模型預測（支援共識模式）

    - 指定 model_id → 僅用該模型預測
    - 未指定 model_id 且 consensus=True → 根據 consensus_mode 聚合多模型
    """
    _ensure_models_registered()
    from app.core.ml.model_manager import model_manager
    from app.data.fetchers.crypto_engine import crypto_engine

    try:
        df = crypto_engine.load_local_data(body.symbol, body.timeframe)
        if df.empty:
            return {"status": "error", "message": "無可用數據"}

        if body.model_id:
            result = model_manager.predict(df, body.symbol, body.timeframe, body.model_id)
        elif body.consensus:
            result = model_manager.predict_consensus(df, body.symbol, body.timeframe)
        else:
            result = model_manager.predict(df, body.symbol, body.timeframe)
        return result

    except Exception as e:
        logger.error(f"ML 預測失敗: {e}\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e)}


@router.post("/compare")
async def compare_models_endpoint(body: CompareRequest):
    """多模型比較"""
    _ensure_models_registered()
    from app.core.ml.comparison import compare_models
    from app.data.fetchers.crypto_engine import crypto_engine

    try:
        df = crypto_engine.load_local_data(body.symbol, body.timeframe)
        if df.empty or len(df) < 100:
            return {
                "status": "error",
                "message": f"數據不足（{len(df)} 根）",
            }

        from app.core.ml.model_manager import model_manager
        user_settings = model_manager.get_settings()
        result = compare_models(
            df, body.symbol, body.timeframe,
            model_ids=body.model_ids,
            feature_set=body.feature_set,
            forward_period=body.forward_period,
            target_direction=user_settings["target_direction"],
            target_threshold=user_settings["target_threshold"],
            lookback_window=user_settings["lookback_window"],
        )
        return result

    except Exception as e:
        logger.error(f"ML 模型比較失敗: {e}\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e)}


@router.post("/retrain")
async def retrain_model(body: TrainRequest):
    """手動觸發再訓練（邏輯同 train，但會清除舊模型計數器）"""
    return await train_model(body)


@router.get("/trained")
async def list_trained_models(
    symbol: Optional[str] = Query(None, description="按幣種篩選"),
):
    """列出所有已訓練的模型記錄"""
    from app.core.ml.model_manager import model_manager
    models = model_manager.list_trained_models(symbol)
    return {"status": "success", "models": models}


@router.get("/performance-history")
async def get_performance_history(
    symbol: Optional[str] = Query(None),
    model_id: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
):
    """取得模型效能歷史趨勢（各版本的 OOS accuracy）"""
    from app.core.ml.model_manager import model_manager
    history = model_manager.get_performance_history(symbol, model_id, limit)
    return {"status": "success", "history": history}


@router.get("/health")
async def check_model_health(
    symbol: str = Query(...),
    timeframe: str = Query("4h"),
    threshold: float = Query(0.55, ge=0.50, le=0.80),
):
    """模型健康度檢查（低於閾值觸發再訓練警示）"""
    from app.core.ml.model_manager import model_manager
    result = model_manager.check_model_health(symbol, timeframe, threshold)
    return {"status": "success", **result}
