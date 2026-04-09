"""阿斯拉量化系統 — 預警 API 路由"""

import json
from typing import Optional

from fastapi import APIRouter

from loguru import logger

router = APIRouter()


def _enrich_alert(alert: dict) -> dict:
    """從 trigger_conditions JSON 中解析 probability 欄位，附加到頂層。"""
    tc = alert.get("trigger_conditions")
    if tc and isinstance(tc, str):
        try:
            parsed = json.loads(tc)
            prob = parsed.get("probability")
            if prob:
                alert["evidence_summary"] = prob.get("evidence_summary")
                alert["probability_detail"] = prob.get("probability")
                alert["feature_attribution"] = prob.get("feature_attribution")
        except (json.JSONDecodeError, AttributeError):
            pass
    return alert


@router.get("/active")
async def get_active_alerts(symbol: Optional[str] = None):
    """取得進行中的預警（含機率和依據）。"""
    from app.core.auto_scanner import auto_scanner

    alerts = auto_scanner.get_active_alerts(symbol)
    return {"alerts": [_enrich_alert(a) for a in alerts], "count": len(alerts)}


@router.get("/history")
async def get_alert_history(symbol: Optional[str] = None, limit: int = 50):
    """取得歷史預警。"""
    from app.core.auto_scanner import auto_scanner

    alerts = auto_scanner.get_history(limit=limit, symbol=symbol)
    return {"alerts": alerts, "count": len(alerts)}


@router.post("/scan")
async def trigger_scan():
    """手動觸發全幣種掃描。"""
    from app.core.auto_scanner import auto_scanner

    results = auto_scanner.scan_all_symbols()
    auto_scanner.validate_expired_alerts()
    logger.info(f"[手動掃描] 完成，發現 {len(results)} 個預警")
    return {"alerts": results, "count": len(results)}


@router.post("/scan/{symbol}")
async def trigger_scan_symbol(symbol: str, timeframe: str = "4h"):
    """手動觸發單一幣種掃描。"""
    from app.core.auto_scanner import auto_scanner

    results = auto_scanner.scan_symbol(symbol, timeframe)
    return {"alerts": results, "count": len(results)}


@router.post("/dismiss/{alert_id}")
async def dismiss_alert(alert_id: int):
    """忽略某個預警。"""
    from app.core.auto_scanner import auto_scanner

    auto_scanner.dismiss_alert(alert_id)
    return {"status": "ok", "dismissed": alert_id}


@router.post("/check-data")
async def check_symbols_data(body: dict):
    """檢查多個幣種是否有足夠的本地數據可供掃描。"""
    from app.core.auto_scanner import auto_scanner

    symbols = body.get("symbols", [])
    timeframe = body.get("timeframe", "4h")
    if not isinstance(symbols, list):
        return {"status": "error", "message": "symbols 必須是陣列"}
    result = auto_scanner.check_symbols_data(symbols, timeframe)
    return {"status": "ok", "data_available": result}


@router.get("/probability/{symbol}")
async def get_movement_probability(
    symbol: str, timeframe: str = "4h", threshold: float = 3.0,
):
    """計算指定幣種未來 3~6 根 K 線內出現 ≥ threshold% 波動的機率（含歷史依據）。"""
    from app.core.auto_scanner import auto_scanner

    result = auto_scanner.estimate_probability(symbol, timeframe, threshold)
    return result


@router.post("/backtest")
async def backtest_scan(symbol: str, timeframe: str = "4h", timestamp: str = ""):
    """回測模式：在指定時間點掃描（驗證歷史事件）。"""
    from app.core.auto_scanner import auto_scanner

    if not timestamp:
        return {"error": "必須提供 timestamp 參數"}

    signals = auto_scanner.backtest_scan(symbol, timeframe, timestamp)
    return {"signals": signals, "count": len(signals), "timestamp": timestamp}
