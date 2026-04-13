"""阿斯拉量化系統 — 數據同步路由（完整版）

支援：日期範圍、細粒度進度追蹤、即時日誌、預估剩餘時間。
"""

import time
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks

from loguru import logger

from app.core.config.settings import settings
from app.data.fetchers.crypto_engine import crypto_engine
from app.data.fetchers.tw_stock_engine import tw_stock_engine
from app.models.schemas import DataSyncRequest, SyncTaskProgress
from app.utils.symbol import is_tw_stock, symbol_to_yf_ticker
from app.utils.timezone import taipei_now, parse_date_string

router = APIRouter()

# 同步任務存儲（記憶體內，重啟會清除）
_sync_tasks: dict[str, SyncTaskProgress] = {}


async def _run_sync(request: DataSyncRequest, task_id: str):
    """背景執行數據同步（細粒度追蹤版）"""

    task = _sync_tasks[task_id]
    task.status = "running"
    task.started_at = taipei_now().strftime("%Y-%m-%d %H:%M:%S")

    total = len(request.symbols) * len(request.timeframes)
    task.total_items = total
    done = 0
    start_time = time.time()

    # 解析日期範圍
    end_date: Optional[datetime] = None
    start_date: Optional[datetime] = None

    if request.end_date:
        try:
            end_date = parse_date_string(request.end_date, as_end_of_day=True)
        except ValueError:
            end_date = datetime.utcnow()
    else:
        end_date = datetime.utcnow()

    if request.start_date:
        try:
            start_date = parse_date_string(request.start_date)
        except ValueError:
            start_date = end_date - timedelta(days=request.days)
    else:
        # 用戶未指定起始日期 → 讓 engine 自行決定
        # 有本地數據 → 增量更新；沒有 → engine 預設從 2020 開始
        start_date = None

    try:
        for symbol in request.symbols:
            for tf in request.timeframes:
                item_label = f"{symbol} / {tf.value}"
                task.current_item = item_label
                task.logs.append(f"開始抓取: {item_label}")

                try:
                    def progress_cb(msg: str):
                        task.logs.append(msg)

                    if is_tw_stock(symbol):
                        await tw_stock_engine.fetch_ohlcv(
                            symbol=symbol,
                            timeframe=tf.value,
                            start_date=start_date,
                            end_date=end_date,
                            force_update=request.force_update,
                            progress_callback=progress_cb,
                        )
                    else:
                        await crypto_engine.fetch_ohlcv(
                            symbol=symbol,
                            timeframe=tf.value,
                            start_date=start_date,
                            end_date=end_date,
                            exchanges=request.exchanges,
                            force_update=request.force_update,
                            progress_callback=progress_cb,
                        )
                    task.logs.append(f"完成: {item_label}")

                    # ML: 遞增 bars 計數器
                    try:
                        from app.core.ml.model_manager import model_manager
                        model_manager.increment_bars(symbol, tf.value)
                    except Exception:
                        pass  # ML 模組未初始化時靜默忽略

                except Exception as e:
                    error_msg = f"{item_label}: {str(e)}"
                    task.errors.append(error_msg)
                    task.logs.append(f"錯誤: {error_msg}")

                done += 1
                task.completed_items = done
                task.progress = round(done / total * 100, 1)

                # 計算 ETA
                elapsed = time.time() - start_time
                if done > 0:
                    avg_time = elapsed / done
                    remaining = (total - done) * avg_time
                    task.eta_seconds = round(remaining, 1)

        # ML: 同步完成後驗證預測 + 檢查自動重訓
        retrain_count = 0
        validated_count = 0
        try:
            from app.core.ml.model_manager import model_manager
            for symbol in request.symbols:
                for tf in request.timeframes:
                    _engine = tw_stock_engine if is_tw_stock(symbol) else crypto_engine
                    df = _engine.load_local_data(symbol, tf.value)
                    if df is not None and len(df) > 0:
                        # 驗證過期的 ML 預測
                        validated_count += model_manager.validate_ml_predictions(
                            symbol, tf.value, df,
                        )
                        # 檢查自動重訓
                        results = model_manager.auto_retrain_if_needed(
                            symbol, tf.value, df,
                        )
                        retrain_count += sum(
                            1 for r in results if r.get("status") == "success"
                        )
        except Exception as e:
            logger.warning(f"ML 後處理失敗（不影響同步結果）: {e}")

        task.status = "completed"
        task.progress = 100
        task.eta_seconds = 0
        task.current_item = None
        if validated_count > 0:
            task.logs.append(f"已驗證 {validated_count} 筆 ML 預測")
        if retrain_count > 0:
            task.logs.append(f"已自動重訓 {retrain_count} 個模型")
        task.logs.append("所有同步任務已完成")

    except Exception as e:
        task.status = "failed"
        task.errors.append(str(e))
        task.logs.append(f"同步失敗: {str(e)}")

    finally:
        task.completed_at = taipei_now().strftime("%Y-%m-%d %H:%M:%S")
        await crypto_engine.close_all()


@router.post("/sync")
async def trigger_sync(request: DataSyncRequest, background_tasks: BackgroundTasks):
    """觸發數據同步（背景執行）

    支援：
    - 多個交易對 + 多個時間週期 批次同步
    - start_date / end_date 日期範圍指定
    - force_update 強制重新抓取
    - 至少 2 家交易所驗證
    """
    # 驗證至少 2 家交易所（台股不需要）
    has_crypto = any(not is_tw_stock(s) for s in request.symbols)
    if has_crypto and len(request.exchanges) < 2:
        raise HTTPException(
            status_code=400,
            detail="至少需要選擇 2 家交易所以啟用五源投票機制"
        )

    # 檢查是否有正在執行的任務
    running = [t for t in _sync_tasks.values() if t.status == "running"]
    if running:
        raise HTTPException(
            status_code=409,
            detail="已有同步任務正在執行中，請等待完成後再操作"
        )

    task_id = f"sync_{taipei_now().strftime('%Y%m%d_%H%M%S')}"

    # 清理已完成的舊任務（保留最近 20 個）
    _MAX_COMPLETED = 20
    completed = [k for k, v in _sync_tasks.items() if v.status in ("completed", "failed")]
    if len(completed) > _MAX_COMPLETED:
        for old_key in sorted(completed)[: len(completed) - _MAX_COMPLETED]:
            del _sync_tasks[old_key]

    # 初始化任務進度
    _sync_tasks[task_id] = SyncTaskProgress(
        task_id=task_id,
        status="pending",
        progress=0,
        total_items=len(request.symbols) * len(request.timeframes),
    )

    background_tasks.add_task(_run_sync, request, task_id)

    return {
        "status": "accepted",
        "task_id": task_id,
        "symbols": request.symbols,
        "timeframes": [tf.value for tf in request.timeframes],
        "exchanges": request.exchanges,
        "start_date": request.start_date,
        "end_date": request.end_date,
        "total_items": len(request.symbols) * len(request.timeframes),
        "message": f"同步任務已啟動，共 {len(request.symbols) * len(request.timeframes)} 組數據",
    }


@router.get("/sync-status")
async def get_sync_status():
    """取得總體同步狀態"""
    crypto_data = crypto_engine.list_available_data()
    tw_data = tw_stock_engine.list_available_data()
    available = crypto_data + tw_data
    metadata = crypto_engine.get_metadata()
    running = [t.model_dump() for t in _sync_tasks.values() if t.status == "running"]

    return {
        "available_data": available,
        "metadata": metadata,
        "active_tasks": running,
        "total_files": len(available),
    }


@router.get("/sync-task/{task_id}")
async def get_sync_task_status(task_id: str):
    """查詢特定同步任務的進度（含即時日誌）"""
    if task_id not in _sync_tasks:
        raise HTTPException(status_code=404, detail="找不到此任務")
    return _sync_tasks[task_id].model_dump()


@router.get("/sync-task/{task_id}/logs")
async def get_sync_task_logs(task_id: str, since: int = 0):
    """取得任務日誌（增量式，since 為已讀取的行數）"""
    if task_id not in _sync_tasks:
        raise HTTPException(status_code=404, detail="找不到此任務")
    task = _sync_tasks[task_id]
    return {
        "logs": task.logs[since:],
        "total": len(task.logs),
        "status": task.status,
    }


@router.get("/available-exchanges")
async def get_available_exchanges():
    """取得可選擇的交易所列表"""
    return {
        "exchanges": [
            {"id": "binance", "name": "Binance", "enabled_by_default": True, "description": "全球最大加密貨幣交易所"},
            {"id": "bybit", "name": "Bybit", "enabled_by_default": True, "description": "專業合約交易所"},
            {"id": "okx", "name": "OKX", "enabled_by_default": True, "description": "綜合性交易所"},
            {"id": "coinbase", "name": "Coinbase", "enabled_by_default": True, "description": "美國合規交易所 (使用 USD 對)"},
            {"id": "kraken", "name": "Kraken", "enabled_by_default": False, "description": "歐洲老牌交易所"},
        ]
    }


@router.get("/available-symbols")
async def get_available_symbols():
    """取得可選擇的交易對列表（含加密貨幣和台股）"""
    return {
        "symbols": settings.default_symbols,
        "tw_symbols": settings.default_tw_symbols,
        "custom_allowed": True,
    }


# 台股名稱快取（程序重啟清除）
_tw_name_cache: dict[str, str] = {
    "TWII": "加權指數",
    "TWOII": "櫃買指數",
}


@router.get("/tw-stock-name")
async def get_tw_stock_name(code: str):
    """用 yfinance 查詢台股中文名稱"""
    code = code.strip().upper()
    if code in _tw_name_cache:
        return {"code": code, "name": _tw_name_cache[code]}

    try:
        import yfinance as yf
        ticker = symbol_to_yf_ticker(f"{code}/TWD")
        info = yf.Ticker(ticker).info
        name = info.get("shortName") or info.get("longName") or ""
        # yfinance 的 shortName 可能是英文，嘗試取中文
        if not name or name.isascii():
            name = info.get("longName") or info.get("shortName") or ""
        _tw_name_cache[code] = name
        return {"code": code, "name": name}
    except Exception as e:
        logger.warning(f"查詢台股名稱失敗 {code}: {e}")
        return {"code": code, "name": ""}
