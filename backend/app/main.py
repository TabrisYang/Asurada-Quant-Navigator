"""阿斯拉量化系統 — FastAPI 主程式"""

import asyncio
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes import alerts, chat, chart, indicators, config, data_sync, export, factor_scan, learn, ml, predictions, scenario, smc, tw_scanner, system_health, positions, observability
from app.core.config.settings import settings
from app.core.usage_tracker import usage_tracker
from app.core.chat_history import chat_history
from app.core.analysis_cache import analysis_cache
from app.core.semantic_cache import semantic_cache
from app.core.knowledge_distiller import knowledge_distiller  # noqa: F401 — 確保初始化
from app.core.knowledge_fragments import fragment_store  # noqa: F401 — 確保初始化


def _migrate_data_layout():
    """
    一次性遷移：將舊版 data/ 扁平結構搬到新版子目錄結構。
      - *.db → data/db/
      - *.csv, sync_metadata.json → data/ohlcv/
    遷移完成後不會重複執行（因為來源檔案已移走）。
    """
    root = settings.data_path          # ./data
    db_dir = settings.db_path           # ./data/db
    ohlcv_dir = settings.ohlcv_path     # ./data/ohlcv

    moved = 0

    # 搬移 SQLite 資料庫
    for db_file in root.glob("*.db"):
        dest = db_dir / db_file.name
        if not dest.exists():
            shutil.move(str(db_file), str(dest))
            logger.info(f"   遷移 DB: {db_file.name} → data/db/")
            moved += 1

    # 搬移 CSV K 線數據
    for csv_file in root.glob("*.csv"):
        dest = ohlcv_dir / csv_file.name
        if not dest.exists():
            shutil.move(str(csv_file), str(dest))
            logger.info(f"   遷移 CSV: {csv_file.name} → data/ohlcv/")
            moved += 1

    # 搬移 sync_metadata.json
    meta_file = root / "sync_metadata.json"
    if meta_file.exists():
        dest = ohlcv_dir / "sync_metadata.json"
        if not dest.exists():
            shutil.move(str(meta_file), str(dest))
            logger.info("   遷移 sync_metadata.json → data/ohlcv/")
            moved += 1

    if moved:
        logger.info(f"✅ 資料遷移完成，共搬移 {moved} 個檔案")
    else:
        logger.debug("   資料目錄結構已是最新，無需遷移")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用程式生命週期管理"""
    logger.info("🚀 阿斯拉量化系統啟動中...")
    logger.info(f"   數據根目錄: {settings.data_path}")
    logger.info(f"   DB 目錄:    {settings.db_path}")
    logger.info(f"   OHLCV 目錄: {settings.ohlcv_path}")
    logger.info(f"   預設 LLM:   {settings.default_llm_provider}")
    logger.info(f"   除錯模式:   {settings.debug}")

    # 確保數據目錄存在
    settings.data_path.mkdir(parents=True, exist_ok=True)
    settings.db_path.mkdir(parents=True, exist_ok=True)
    settings.ohlcv_path.mkdir(parents=True, exist_ok=True)

    # 一次性遷移舊版扁平目錄
    _migrate_data_layout()

    # 啟動時清理過期記錄
    usage_tracker.cleanup_old_records()
    chat_history.cleanup_old_records()
    analysis_cache.cleanup()
    semantic_cache.cleanup()

    # v141.2：啟動時清 external_signals in-memory cache，避免「cache 卡 empty」
    # 30 分鐘 cache poisoning（某次採集全失敗 → 後續 30 分鐘都拿空）
    try:
        from app.core.external_signals import clear_cache as _clear_ext_cache
        _cleared = _clear_ext_cache()
        logger.info(f"[external_signals] 啟動清快取（{_cleared} 個 key）")
    except Exception as _e:
        logger.warning(f"[external_signals] 啟動清快取失敗（不影響）: {_e}")

    # 後台預載嵌入模型 + 種子知識（不阻塞啟動）
    import threading
    from app.core import embedding_service

    def _background_init():
        embedding_service.is_available()
        # 嵌入模型就緒後載入種子知識
        from app.core.seed_knowledge import initialize_seed_knowledge
        initialize_seed_knowledge()
        # 清理過期知識碎片
        fragment_store.cleanup_expired()
        # 預設分析策略（首次啟動時載入）
        from app.core.user_strategies import seed_default_strategies
        seed_default_strategies()
        # 初始化特徵分析（首次或過期時自動計算）
        try:
            from app.core.scanner_feature_profiler import scanner_feature_profiler
            scanner_feature_profiler.ensure_profiles()
        except Exception as e:
            logger.warning(f"[特徵分析] 背景初始化失敗: {e}")

        # v108 Phase 3：啟動時冷啟動 watcher sweep
        # （用既有本地資料補做離線期間的失效條件檢查，不強制重抓資料）
        try:
            from app.core.watcher import startup_sweep
            _sw = startup_sweep()
            logger.info(
                f"[watcher startup] symbols={_sw.get('total_symbols')} "
                f"checked={_sw.get('total_checked')} "
                f"invalidated={_sw.get('total_invalidated')} "
                f"shadow={_sw.get('shadow_only')}"
            )
        except Exception as e:
            logger.warning(f"[watcher startup] 啟動 sweep 失敗（不影響主流程）: {e}")

        # v109：事件日曆自動同步（只收 FOMC + NFP 等可驗證事件）
        # FOMC 用 HTML 結構爬蟲、NFP 用 first-Friday 規則計算；C 類（CPI/PPI/GDP）由 system prompt 提醒
        try:
            from app.core.event_calendar_sync import sync_verifiable_events
            _evt_sync = sync_verifiable_events()
            logger.info(
                f"[event_sync] status={_evt_sync.get('status')} "
                f"fomc={_evt_sync.get('fomc_status')} "
                f"nfp={_evt_sync.get('nfp_status')} "
                f"total={_evt_sync.get('n_total')}"
            )
        except Exception as e:
            logger.warning(f"[event_sync] 啟動同步失敗（不影響主流程）: {e}")

    threading.Thread(target=_background_init, daemon=True).start()

    # 背景排程：每 2 小時自動驗證過期預測
    async def _periodic_validation():
        from app.core.prediction_validator import validate_all_active
        while True:
            await asyncio.sleep(7200)  # 2 小時
            try:
                result = validate_all_active()
                if result.get("validated", 0) > 0:
                    logger.info(f"[排程驗證] {result['validated']} 筆預測已驗證")
            except Exception as e:
                logger.error(f"[排程驗證] 失敗: {e}")

    # 背景排程：每週自動覆盤
    async def _periodic_weekly_review():
        from datetime import datetime, timedelta
        from app.core.prediction_tracker import prediction_tracker

        # 啟動後等 60 秒再檢查，讓其他初始化先完成
        await asyncio.sleep(60)

        while True:
            try:
                last_review = prediction_tracker.get_last_review_time()
                now = datetime.now().astimezone()
                need_review = (last_review is None) or (now - last_review > timedelta(days=7))

                if need_review:
                    logger.info("[週覆盤] 開始自動產生每週覆盤報告...")
                    review_data = prediction_tracker.get_review_data(
                        start_date=(now - timedelta(days=7)).isoformat(),
                    )
                    if review_data["summary"]["total"] == 0:
                        logger.info("[週覆盤] 過去 7 天無已驗證預測，跳過")
                    else:
                        # 嘗試用 LLM 生成覆盤
                        report = await _run_auto_review(review_data)
                        prediction_tracker.save_review(
                            report=report,
                            summary=review_data["summary"],
                            is_auto=True,
                        )
                        # 覆盤後觸發自動調整
                        try:
                            from app.core.auto_adjuster import run_adjustment_cycle
                            run_adjustment_cycle()
                            logger.info("[週覆盤] 自動調整已觸發")
                        except Exception as e:
                            logger.warning(f"[週覆盤] 自動調整失敗: {e}")
            except Exception as e:
                logger.error(f"[週覆盤] 失敗: {e}")

            await asyncio.sleep(86400)  # 每 24 小時檢查一次

    async def _run_auto_review(review_data: dict) -> str:
        """用 LLM 生成覆盤報告，失敗時回退到模板。"""
        preds = review_data["predictions"]
        summary = review_data["summary"]

        preds_text = []
        for p in preds:
            outcome = f"{p['actual_outcome_pct']:.2f}%" if p.get("actual_outcome_pct") is not None else "N/A"
            preds_text.append(
                f"- {p['symbol']} {p['direction']} | 入場={p['entry_price']} "
                f"目標={p['target_price']} 止損={p['stop_price']} | "
                f"結果: {p['status']} ({outcome}) | "
                f"信心: {p['confidence']} | 指標: {p['indicators']} | regime: {p['regime']}"
            )

        prompt = f"""你是一位量化交易覆盤教練。請根據以下預測記錄，生成一份每週覆盤報告。

## 統計摘要
- 總預測數: {summary['total']}
- 命中目標: {summary['wins']} ({summary['win_rate']}%)
- 觸及止損: {summary['losses']}
- 到期未觸發: {summary['expired']}

## 預測明細
{chr(10).join(preds_text)}

## 請分析以下幾點：
1. **成功模式** — 命中目標的預測有什麼共同特徵？
2. **失敗模式** — 觸及止損的預測有什麼共同特徵？有哪些可以避免的錯誤？
3. **關鍵教訓** — 列出 3-5 條具體、可執行的改進建議
4. **下週建議** — 根據近期表現，建議調整什麼策略參數？

請用繁體中文回覆，格式清晰易讀。"""

        try:
            from app.core.llm.adapter import create_adapter
            adapter = create_adapter(provider=settings.default_llm_provider)
            llm_response = await adapter.chat(
                messages=[{"role": "user", "content": prompt}],
            )
            report = llm_response.message

            # 將教訓存入知識庫
            try:
                from app.api.routes.predictions import _save_lessons_from_review
                _save_lessons_from_review(report)
            except Exception:
                pass

            return report
        except Exception as e:
            logger.warning(f"[週覆盤] LLM 不可用，使用模板: {e}")
            from app.api.routes.predictions import _generate_template_report
            return _generate_template_report(review_data)

    async def _periodic_auto_scan():
        """定期自動掃描所有幣種，偵測異常前兆信號。間隔和幣種從用戶設定讀取。"""
        await asyncio.sleep(120)  # 啟動延遲，等數據引擎初始化
        while True:
            try:
                from app.core.auto_scanner import auto_scanner
                cfg = auto_scanner._load_scan_config()

                if cfg.get("scan_enabled", True):
                    # scan_all_symbols 是同步 CPU-bound（讀本地 CSV + 指標計算），
                    # 直接呼叫會卡住事件迴圈 → 用 to_thread 移出，避免每 4 小時凍結請求處理
                    results = await asyncio.to_thread(auto_scanner.scan_all_symbols)
                    await asyncio.to_thread(auto_scanner.validate_expired_alerts)
                    if results:
                        logger.info(f"[自動掃描] 發現 {len(results)} 個預警信號")
                    else:
                        logger.debug("[自動掃描] 本次未發現異常信號")
                else:
                    logger.debug("[自動掃描] 已停用，跳過本次掃描")

                interval_hours = cfg.get("scan_interval_hours", 4)
            except Exception as e:
                logger.error(f"[自動掃描] 失敗: {e}")
                interval_hours = 4
            await asyncio.sleep(interval_hours * 3600)

    validation_task = asyncio.create_task(_periodic_validation())
    review_task = asyncio.create_task(_periodic_weekly_review())
    scan_task = asyncio.create_task(_periodic_auto_scan())

    # 暴露 scan_task 供外部重啟（設定即時生效）
    app_state["scan_task"] = scan_task
    app_state["scan_factory"] = _periodic_auto_scan

    yield

    validation_task.cancel()
    review_task.cancel()
    scan_task.cancel()
    logger.info("👋 阿斯拉量化系統關閉")


# 全域狀態：供 API 路由存取 scan_task
app_state: dict = {}


app = FastAPI(
    title="阿斯拉量化系統",
    description="加密貨幣量化分析平台 — LLM 驅動的互動式 K 線圖技術分析",
    version="1.1.0",
    lifespan=lifespan,
)

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 註冊路由
app.include_router(chat.router, prefix="/api/chat", tags=["對話"])
app.include_router(chart.router, prefix="/api/chart", tags=["圖表"])
app.include_router(indicators.router, prefix="/api/indicators", tags=["指標"])
app.include_router(config.router, prefix="/api/config", tags=["設定"])
app.include_router(data_sync.router, prefix="/api/data", tags=["數據同步"])
app.include_router(export.router, prefix="/api/export", tags=["匯出"])
app.include_router(factor_scan.router, prefix="/api/factor-scan", tags=["因子掃描"])
app.include_router(learn.router, prefix="/api/learn", tags=["互動教學"])
app.include_router(ml.router, prefix="/api/ml", tags=["ML 增強"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["預測追蹤"])
app.include_router(scenario.router, prefix="/api/scenario", tags=["情境預測"])
app.include_router(smc.router, prefix="/api/smc", tags=["SMC 分析"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["預警"])
app.include_router(tw_scanner.router, prefix="/api/scanner", tags=["市場掃描"])
app.include_router(system_health.router, prefix="/api/system", tags=["系統健康"])
app.include_router(positions.router, tags=["持倉追蹤"])
app.include_router(observability.router, prefix="/api/system", tags=["可觀測性"])


@app.get("/api/health")
async def health_check():
    """健康檢查端點"""
    return {
        "status": "healthy",
        "system": "阿斯拉量化系統",
        "version": "1.1.0",
        "semantic_cache": semantic_cache.get_stats(),
        "knowledge_fragments": fragment_store.get_stats(),
    }
