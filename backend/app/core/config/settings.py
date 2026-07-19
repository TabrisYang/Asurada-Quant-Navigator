"""阿斯拉量化系統 — 全域設定"""

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """應用程式全域設定，支援 .env 檔案覆蓋"""

    # 伺服器
    # v155 安全預設：只綁本機（原 0.0.0.0 讓區網任何裝置可打無認證的 API）。
    # 刻意要區網存取時在 .env 設 HOST=0.0.0.0，並強烈建議同時設 API_TOKEN。
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False

    # v155 選配 API 認證：非空時所有 API（除 /api/health 與 CORS preflight）
    # 需帶 X-API-Token header。前端在設定面板輸入同一組 token 即自動附帶。
    api_token: str = ""

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
    # 無本地檔且呼叫端未指定 start_date 時，加密貨幣回溯的起點。
    # 之前誤用 default_fetch_days(90)當起點 → 新標的只抓 90 天（見 crypto_engine.fetch_ohlcv）。
    # binance 現貨可回溯至 2017，取 2018 保守起點以確保跨牛熊完整歷史。
    default_history_start: str = "2018-01-01"
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
    # ★ v102：subprocess 隔離 — chat 流程透過 ml_client.predict_via_subprocess()
    #   呼叫獨立 Python process 跑 lightgbm/shap，主進程永不載 ML lib
    #   → 不再 segfault，可放心開 SHADOW 模式（驗證 subprocess 流程穩定）
    #   → user 仍看 v100 體驗（imitation_learning_enabled 預設 False）
    imitation_shadow_mode: bool = True            # subprocess 安全，恢復預設 True
    imitation_canary_pct: int = 0                 # Canary 流量 %（0/1/10/25/50/100）
    quality_gate_enabled: bool = True             # Quality Gate 7 硬閾值（永遠開）
    adversarial_val_enabled: bool = False         # Drift 偵測
    champion_challenger_enabled: bool = False     # 自動模型切換
    auto_rollback_enabled: bool = True            # 表現變差自動關閉（永遠開）
    feature_recording_enabled: bool = True        # Phase 2.0 特徵快照記錄（純 SQL 安全）

    # v104：自適應 ATR 倍數（timeframe + regime + 信心三維度）
    # 預設 True；若實測新倍數比舊差，設 False 即一鍵回退到 1.5/2.0 固定值
    adaptive_atr_mults_enabled: bool = True

    # v104.1：bias_score 擴展訊號維度（5 分量 → 9 分量）
    # 預設 True；False 時走原 5 分量行為（緊急回滾用）
    bias_score_extended_dimensions: bool = True

    # v105 Phase B：bias_score 權重資料驅動（取代 v104.1 經驗值）
    # 預設 True；若 learned weights 存在（通過 lockbox AUC ≥ 0.55 quality gate）才會生效；
    # 沒檔自動 fallback 經驗值。設 False 強制走經驗值。
    bias_score_data_driven_weights: bool = True

    # v124：機率三聯顯示（baseline / TA 條件化 / track record + Wilson CI）
    # 預設 True；若 shadow mode PF/勝率劣化 >10% 設 False 即可一鍵關閉，
    # 不需 revert PR。chat.py 注入點與 function_defs.py 規則段都會檢查此 flag。
    probability_triplet_enabled: bool = True

    # v136：完整布林通道策略系統（3 核心策略 + regime 切換 + entry/exit/stop）
    # 預設 True；若想暫時關閉，設 False 即可，auto_scanner 會 fallback 原行為。
    bollinger_signals_enabled: bool = True
    # 布林策略 threshold 微調（可從 .env 覆蓋）
    bollinger_squeeze_min_duration: int = 5
    bollinger_breakout_bandwidth_roc: float = 5.0
    bollinger_walk_band_min_touches: int = 3
    bollinger_walk_band_min_adx: float = 25.0
    bollinger_upper_band_threshold: float = 0.9
    bollinger_lower_band_threshold: float = 0.1

    # v132：完整分析「編排管線」— 把 seg2 monolith（13 段一次出）拆成
    # 5 個維度 focused call + 1 個 synthesis call，每維度品質對齊「單獨問」。
    # 預設 False（灰度）；確認 shadow mode 不劣化後再切 True。
    comprehensive_pipeline_enabled: bool = False

    # ── ChatGPT 訂閱制（Codex CLI）：執行檔路徑覆寫（空 = 自動尋找）──
    codex_cli_path: str = ""

    # ── 回測：台股現股賣出證交稅 0.3%（backtest/engine.py，僅 /TWD 標的觸發）──
    tw_tax_enabled: bool = True

    # ── LLM 覆核層：回答完成後用低一階模型交叉檢查數據/邏輯（core/llm/verifier.py）──
    verify_enabled: bool = True
    verify_model_override: str = ""  # 空 = 自動降家族（Opus→Sonnet→Haiku）；填模型 ID 覆寫
    verify_timeout_sec: int = 120    # subscription CLI 首 token 內建 120s fail-fast，外層不能更短
    verify_min_text_len: int = 500   # 短於此的回答不覆核

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
