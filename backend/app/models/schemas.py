"""阿斯拉量化系統 — Pydantic 資料模型"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ========== 列舉型別 ==========

class Timeframe(str, Enum):
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


class IndicatorAction(str, Enum):
    ADD = "add"
    REMOVE = "remove"
    UPDATE = "update"


class DisplayMode(str, Enum):
    OVERLAY = "overlay"
    SUB_CHART = "sub_chart"
    INFO_PANEL = "info_panel"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    CLAUDE = "claude"
    OLLAMA = "ollama"


class ConditionOperator(str, Enum):
    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="
    EQ = "=="
    CROSS_ABOVE = "cross_above"
    CROSS_BELOW = "cross_below"
    BETWEEN = "between"


# ========== 請求模型 ==========

class ChatMessageItem(BaseModel):
    """對話歷史中的單條訊息"""
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    """LLM 對話請求"""
    message: str = Field(..., max_length=10000)
    messages: list[ChatMessageItem] = Field(default_factory=list, max_length=50)
    conversation_id: Optional[str] = None
    chart_state: Optional[dict[str, Any]] = None
    llm_provider: Optional[LLMProvider] = None
    session_id: Optional[str] = None       # 優先：使用 session token（安全）
    api_key: Optional[str] = None          # 向下相容：直接傳 key（不建議）
    mode: Optional[str] = None             # "quant_research" = 強制量化回測分析


class ChartDataRequest(BaseModel):
    """K 線數據請求"""
    symbol: str = "BTC/USDT"
    timeframe: Timeframe = Timeframe.D1
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class IndicatorRequest(BaseModel):
    """指標操作請求"""
    action: IndicatorAction
    indicator_type: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    display_mode: Optional[DisplayMode] = None


class ConditionItem(BaseModel):
    """單一條件"""
    indicator: str
    operator: ConditionOperator
    value: Optional[float] = None
    value2: Optional[float] = None  # for BETWEEN
    parameters: dict[str, Any] = Field(default_factory=dict)


class ConditionSearchRequest(BaseModel):
    """條件搜尋請求"""
    symbol: str = "BTC/USDT"
    timeframe: Timeframe = Timeframe.D1
    conditions: list[ConditionItem]
    logical_operator: str = "AND"
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class BacktestRequest(BaseModel):
    """回測請求"""
    symbol: str = "BTC/USDT"
    timeframe: Timeframe = Timeframe.D1
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    entry_conditions: list[ConditionItem]
    exit_conditions: list[ConditionItem]
    initial_capital: float = 10000.0
    position_size: float = 1.0  # 1.0 = 100%


class LLMConfigRequest(BaseModel):
    """LLM 設定請求"""
    provider: LLMProvider
    api_key: Optional[str] = None          # 初次設定時傳入
    session_id: Optional[str] = None       # 已有 session 時使用
    model_name: Optional[str] = None
    base_url: Optional[str] = None  # for Ollama


class DataSyncRequest(BaseModel):
    """數據同步請求（完整版：支援日期範圍）"""
    symbols: list[str] = Field(default_factory=lambda: ["BTC/USDT"])
    timeframes: list[Timeframe] = Field(default_factory=lambda: [Timeframe.D1])
    exchanges: list[str] = Field(
        default_factory=lambda: ["binance", "bybit", "okx", "coinbase"]
    )
    start_date: Optional[str] = None  # YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS
    end_date: Optional[str] = None    # YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS
    days: int = 90  # 當 start_date 為空時使用
    force_update: bool = False


# ========== 回應模型 ==========

class OHLCVData(BaseModel):
    """單根 K 線數據"""
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    final_price: Optional[float] = None
    anomaly_detected: Optional[bool] = None


class IndicatorData(BaseModel):
    """指標計算結果"""
    name: str
    display_mode: DisplayMode
    data: dict[str, list[Optional[float]]]  # {series_name: [values]}
    parameters: dict[str, Any]


class MatchedPeriod(BaseModel):
    """條件匹配的時間段"""
    start: str
    end: str
    values: dict[str, float] = Field(default_factory=dict)


class ChartDataResponse(BaseModel):
    """K 線數據回應"""
    symbol: str
    timeframe: str
    ohlcv: list[OHLCVData]
    indicators: list[IndicatorData] = Field(default_factory=list)


class ConditionSearchResponse(BaseModel):
    """條件搜尋回應"""
    symbol: str
    timeframe: str
    matched_periods: list[MatchedPeriod]
    total_matches: int
    summary: str


class BacktestResult(BaseModel):
    """回測結果"""
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    equity_curve: list[dict[str, Any]]
    trades: list[dict[str, Any]]


class TokenUsageResponse(BaseModel):
    """Token 用量（回應給前端）"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    provider: str = ""
    estimated_cost_usd: float = 0.0


class ChatResponse(BaseModel):
    """LLM 對話回應"""
    message: str
    function_calls: list[dict[str, Any]] = Field(default_factory=list)
    chart_updates: Optional[dict[str, Any]] = None
    conversation_id: str
    usage: Optional[TokenUsageResponse] = None


class IndicatorInfo(BaseModel):
    """指標資訊（給前端顯示用）"""
    id: str
    name: str
    category: str
    description: str
    parameters: dict[str, dict[str, Any]]
    display_mode: DisplayMode
    data_source: str
    warmup_periods: int
    pro_tip: str


class IndicatorListResponse(BaseModel):
    """可用指標清單回應"""
    indicators: list[IndicatorInfo]
    total: int


class SyncTaskProgress(BaseModel):
    """同步任務進度（細粒度）"""
    task_id: str
    status: str  # "pending" | "running" | "completed" | "failed"
    progress: float  # 0-100
    current_item: Optional[str] = None  # 目前處理的 symbol/timeframe
    total_items: int = 0
    completed_items: int = 0
    eta_seconds: Optional[float] = None
    logs: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class SyncStatus(BaseModel):
    """同步狀態"""
    symbol: str
    timeframe: str
    last_sync: Optional[str] = None
    total_records: int = 0
    status: str = "not_synced"


class ErrorResponse(BaseModel):
    """錯誤回應"""
    error: str
    detail: Optional[str] = None
    code: str = "UNKNOWN_ERROR"
