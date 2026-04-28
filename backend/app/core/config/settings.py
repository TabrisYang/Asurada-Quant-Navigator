"""阿斯拉量化系統 — 全域設定"""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """應用程式全域設定，支援 .env 檔案覆蓋"""

    # 伺服器
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # 資料路徑
    data_dir: str = "./data"

    # 資料庫（預設 SQLite；生產環境可設為 PostgreSQL URL）
    # 例如: postgresql://user:pass@localhost:5432/asura
    database_url: Optional[str] = None

    # LLM 預設供應商
    default_llm_provider: str = "openai"

    # LLM 生成參數
    llm_temperature: float = 0.3
    llm_max_tokens: int = 8192
    llm_timeout: int = 60
    llm_stream_timeout: int = 120

    # 日誌
    log_level: str = "INFO"

    # CORS（支援環境變數 CORS_ORIGINS 覆蓋，逗號分隔）
    cors_origins: list[str] = [
        o.strip()
        for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    ]

    # 數據抓取
    default_exchanges: list[str] = ["binance", "bybit", "okx", "coinbase"]
    default_symbols: list[str] = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT",
        "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", "MATIC/USDT",
    ]
    default_tw_symbols: list[str] = [
        "TWII/TWD",  # 加權指數
        "2330/TWD", "2317/TWD", "2454/TWD", "2412/TWD", "3008/TWD",
        "2881/TWD", "2882/TWD", "1301/TWD", "2308/TWD", "2303/TWD",
    ]
    default_timeframes: list[str] = ["15m", "1h", "4h", "1d", "1w"]
    default_fetch_days: int = 90
    anomaly_threshold: float = 0.005  # 0.5%

    # 每交易所個別 Rate Limit（秒）
    exchange_rate_limits: dict[str, float] = {
        "binance": 0.2,
        "bybit": 0.2,
        "okx": 0.2,
        "coinbase": 0.3,
        "kraken": 0.2,
    }
    exchange_switch_delay: float = 0.3

    # 重試配置
    max_retries: int = 3
    retry_delays: list[float] = [1.0, 2.0, 5.0]
    api_timeout: int = 30

    # 每次請求最大 K 線數
    max_bars_per_request: int = 1000

    # 時區
    display_timezone: str = "Asia/Taipei"

    # ═══════════════════════════════════════════════════════════
    # v101 模仿學習 + Quality Gate（預設全部 OFF / SHADOW，user 體驗 = v100）
    # ═══════════════════════════════════════════════════════════
    imitation_learning_enabled: bool = False     # 主開關（user 端注入）
    imitation_blend_enabled: bool = False         # 動態 blend（rule × ML 權重）
    # ★ v101 修復：預設 SHADOW=False — chat 流程完全不呼叫 lightgbm/shap，
    #   避免 native lib（numpy/pandas_ta/statsmodels/sklearn × lightgbm/shap）
    #   在 macOS Python 3.12 上競爭資源造成 segfault
    #   → 保留訓練（launchd 獨立進程）+ 特徵記錄（純 SQL，無 native）
    #   → 未來 v102 加 subprocess 隔離後可再啟用 SHADOW
    imitation_shadow_mode: bool = False           # 預設 OFF（避免 chat 流程觸發 native 衝突）
    imitation_canary_pct: int = 0                 # Canary 流量 %（0/1/10/25/50/100）
    quality_gate_enabled: bool = True             # Quality Gate 7 硬閾值（永遠開）
    adversarial_val_enabled: bool = False         # Drift 偵測
    champion_challenger_enabled: bool = False     # 自動模型切換
    auto_rollback_enabled: bool = True            # 表現變差自動關閉（永遠開）
    feature_recording_enabled: bool = True        # Phase 2.0 特徵快照記錄（純 SQL 安全）

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @property
    def data_path(self) -> Path:
        """返回數據根目錄的 Path 物件"""
        path = Path(self.data_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def use_postgres(self) -> bool:
        """是否使用 PostgreSQL"""
        return bool(self.database_url and self.database_url.startswith("postgresql"))

    @property
    def db_path(self) -> Path:
        """返回 SQLite 資料庫專用目錄（與 K 線數據分離，避免誤刪）"""
        path = Path(self.data_dir) / "db"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def ohlcv_path(self) -> Path:
        """返回 K 線 CSV 數據專用目錄"""
        path = Path(self.data_dir) / "ohlcv"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_exchange_delay(self, exchange_id: str) -> float:
        """取得特定交易所的請求延遲"""
        return self.exchange_rate_limits.get(exchange_id, 0.25)


# 全域設定單例
settings = Settings()
