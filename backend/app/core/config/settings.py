"""阿斯拉量化系統 — 全域設定"""

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

    # LLM 預設供應商
    default_llm_provider: str = "openai"

    # LLM 生成參數
    llm_temperature: float = 0.3
    llm_max_tokens: int = 8192
    llm_timeout: int = 60
    llm_stream_timeout: int = 120

    # 日誌
    log_level: str = "INFO"

    # CORS
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # 數據抓取
    default_exchanges: list[str] = ["binance", "bybit", "okx", "coinbase"]
    default_symbols: list[str] = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT",
        "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", "MATIC/USDT",
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
