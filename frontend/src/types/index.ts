/** 阿斯拉量化系統 — TypeScript 型別定義 */

// ========== 列舉 ==========

export type Timeframe = '15m' | '1h' | '4h' | '1d' | '1w';

export type DisplayMode = 'overlay' | 'sub_chart' | 'info_panel';

export type LLMProvider = 'openai' | 'gemini' | 'claude' | 'claude_subscription' | 'ollama';

export type IndicatorAction = 'add' | 'remove' | 'update';

// ========== K 線數據 ==========

export interface OHLCVData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  final_price?: number;
  anomaly_detected?: boolean;
}

// ========== 指標 ==========

export interface IndicatorParam {
  default: number;
  min: number;
  max: number;
  type: 'int' | 'float';
  step?: number;
}

export interface IndicatorInfo {
  id: string;
  name: string;
  category: string;
  description: string;
  parameters: Record<string, IndicatorParam>;
  display_mode: DisplayMode;
  data_source: 'ohlcv' | 'external';
  warmup_periods: number;
  pro_tip: string;
}

export interface ActiveIndicator {
  id: string;
  indicator_type: string;
  parameters: Record<string, number>;
  display_mode: DisplayMode;
  visible: boolean;
  data?: Record<string, (number | null)[]>;
}

// ========== 條件 ==========

export interface ConditionItem {
  indicator: string;
  operator: '>' | '<' | '>=' | '<=' | '==' | 'cross_above' | 'cross_below' | 'between';
  value?: number;
  value2?: number;
  parameters?: Record<string, number>;
}

export interface MatchedPeriod {
  start: string;
  end: string;
  values: Record<string, number>;
}

// ========== 圖表狀態 ==========

export interface ChartState {
  symbol: string;
  timeframe: Timeframe;
  startDate: string | null;
  endDate: string | null;
  ohlcvData: OHLCVData[];
  activeIndicators: ActiveIndicator[];
  annotations: Annotation[];
}

export interface Annotation {
  id: string;
  type: 'highlight_range' | 'vertical_line' | 'horizontal_line' | 'text_label' | 'trend_line';
  startTime?: string;
  endTime?: string;
  price?: number;
  endPrice?: number;
  text?: string;
  color?: string;
  lineWidth?: number;
  lineStyle?: number; // 0=Solid, 1=Dotted, 2=Dashed, 3=LargeDashed
  groupId?: string;
  groupName?: string;
  visible?: boolean;
}

// ========== Token 用量 ==========

export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  model: string;
  provider: string;
  estimated_cost_usd: number;
}

// ========== 對話 ==========

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  functionCalls?: FunctionCall[];
  usage?: TokenUsage;
  isStreaming?: boolean;    // 正在串流中
  isThinking?: boolean;     // LLM 正在思考
  statusText?: string;      // 進度狀態文字（取代固定的「思考中」）
  progress?: {              // 分析進度
    percentage: number;
    completed: number;
    total: number;
    current_task: string;
  };
}

export interface FunctionCall {
  name: string;
  arguments: Record<string, unknown>;
  result?: unknown;
}

// ========== LLM 設定 ==========

export interface LLMConfig {
  provider: LLMProvider;
  sessionId?: string;    // 後端 session token（安全，不含明文 Key）
  modelName?: string;
  baseUrl?: string;
}

// ========== 回測 ==========

export interface BacktestResult {
  total_trades: number;
  win_rate: number;
  profit_factor: number;
  sharpe_ratio: number;
  max_drawdown: number;
  total_return: number;
  equity_curve: Array<{ date: string; value: number }>;
  trades: Array<{
    entry_date: string;
    exit_date: string;
    entry_price: number;
    exit_price: number;
    pnl: number;
    type: 'long' | 'short';
  }>;
}
