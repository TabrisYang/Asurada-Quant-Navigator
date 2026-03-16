"""阿斯拉量化系統 — FastAPI 主程式"""

import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes import chat, chart, indicators, config, data_sync, export, factor_scan
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

    threading.Thread(target=_background_init, daemon=True).start()

    yield

    logger.info("👋 阿斯拉量化系統關閉")


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
