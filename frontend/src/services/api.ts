/** 阿斯拉量化系統 — API 服務層（完整版 + Streaming） */

import axios from 'axios';
import type { Timeframe, LLMProvider, ConditionItem, TokenUsage } from '../types';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
  timeout: Number(import.meta.env.VITE_API_TIMEOUT) || 60000,
  headers: { 'Content-Type': 'application/json' },
});

// 統一錯誤攔截：所有 API 錯誤不再拋出未處理異常
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const status = error?.response?.status;
    const msg = error?.response?.data?.detail
      || error?.response?.data?.message
      || error?.message
      || '未知錯誤';

    if (status === 401 || status === 403) {
      console.warn(`[API] 認證失敗 (${status}): ${msg}`);
    } else if (status === 429) {
      console.warn(`[API] 請求過於頻繁: ${msg}`);
    } else if (status && status >= 500) {
      console.error(`[API] 伺服器錯誤 (${status}): ${msg}`);
    } else if (error.code === 'ECONNABORTED') {
      console.error('[API] 請求超時');
    } else {
      console.error(`[API] 請求失敗: ${msg}`);
    }

    return Promise.reject(error);
  },
);

// ===== 圖表 API =====

export async function fetchChartData(
  symbol: string,
  timeframe: Timeframe,
  startDate?: string,
  endDate?: string
) {
  const params: Record<string, string> = { symbol, timeframe };
  if (startDate) params.start_date = startDate;
  if (endDate) params.end_date = endDate;
  const res = await api.get('/chart/data', { params });
  return res.data;
}

// ===== 指標 API =====

export async function fetchIndicatorList() {
  const res = await api.get('/indicators/list');
  return res.data;
}

export async function calculateIndicator(
  indicatorType: string,
  action: string,
  parameters: Record<string, unknown>
) {
  const res = await api.post('/indicators/calculate', {
    action,
    indicator_type: indicatorType,
    parameters,
  });
  return res.data;
}

export async function searchConditions(
  symbol: string,
  timeframe: Timeframe,
  conditions: ConditionItem[],
  logicalOperator: string = 'AND',
  startDate?: string,
  endDate?: string
) {
  const res = await api.post('/indicators/search', {
    symbol,
    timeframe,
    conditions,
    logical_operator: logicalOperator,
    start_date: startDate,
    end_date: endDate,
  });
  return res.data;
}

// ===== 對話 API =====

export async function sendChatMessage(
  message: string,
  conversationId?: string,
  chartState?: Record<string, unknown>,
  llmProvider?: LLMProvider,
  sessionId?: string
) {
  const res = await api.post('/chat/', {
    message,
    conversation_id: conversationId,
    chart_state: chartState,
    llm_provider: llmProvider,
    session_id: sessionId,
  });
  return res.data;
}

// ===== 對話 Streaming API =====

export interface StreamCallbacks {
  onThinking?: () => void;
  onStatus?: (message: string) => void;
  onToken?: (content: string) => void;
  onFunctionCalls?: (calls: Array<{ name: string; arguments: Record<string, unknown> }>) => void;
  onChartUpdates?: (updates: Record<string, unknown>) => void;
  onUsage?: (usage: TokenUsage) => void;
  onError?: (error: string) => void;
  onDone?: (conversationId?: string, hints?: Record<string, unknown>) => void;
}

export async function streamChatMessage(
  message: string,
  callbacks: StreamCallbacks,
  conversationId?: string,
  chartState?: Record<string, unknown>,
  llmProvider?: LLMProvider,
  sessionId?: string,
  chatHistory?: Array<{ role: string; content: string }>,
  mode?: string,
): Promise<void> {
  const body: Record<string, unknown> = {
    message,
    messages: chatHistory || [],
    conversation_id: conversationId,
    chart_state: chartState,
    llm_provider: llmProvider,
    session_id: sessionId,
  };
  if (mode) {
    body.mode = mode;
  }

  let doneEmitted = false;
  const wrappedCallbacks: StreamCallbacks = {
    ...callbacks,
    onDone: (convId, hints) => {
      doneEmitted = true;
      callbacks.onDone?.(convId, hints);
    },
  };

  try {
    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errText = await response.text();
      wrappedCallbacks.onError?.(`伺服器錯誤 (${response.status}): ${errText}`);
      if (!doneEmitted) { doneEmitted = true; callbacks.onDone?.(); }
      return;
    }

    const reader = response.body?.getReader();
    if (!reader) {
      wrappedCallbacks.onError?.('瀏覽器不支援 Streaming');
      if (!doneEmitted) { doneEmitted = true; callbacks.onDone?.(); }
      return;
    }

    const decoder = new TextDecoder();
    let buffer = '';
    let currentEvent = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('event: ')) {
          currentEvent = line.slice(7).trim();
        } else if (line.startsWith('data: ')) {
          const dataStr = line.slice(6);
          try {
            const data = JSON.parse(dataStr);
            _handleSSEEvent(currentEvent, data, wrappedCallbacks);
          } catch {
            // ignore
          }
          currentEvent = '';
        }
      }
    }

    // 處理 buffer 中殘餘的事件
    if (buffer.trim()) {
      const remaining = buffer.split('\n');
      for (const line of remaining) {
        if (line.startsWith('event: ')) {
          currentEvent = line.slice(7).trim();
        } else if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            _handleSSEEvent(currentEvent, data, wrappedCallbacks);
          } catch {
            // ignore
          }
          currentEvent = '';
        }
      }
    }
  } catch (err: unknown) {
    callbacks.onError?.((err as Error)?.message || '網路連線失敗');
  } finally {
    if (!doneEmitted) {
      callbacks.onDone?.();
    }
  }
}

function _handleSSEEvent(
  event: string,
  data: Record<string, unknown>,
  callbacks: StreamCallbacks,
) {
  switch (event) {
    case 'thinking':
      callbacks.onThinking?.();
      break;
    case 'status':
      if (typeof data.message === 'string') {
        callbacks.onStatus?.(data.message);
      }
      break;
    case 'token':
      if (typeof data.content === 'string') {
        callbacks.onToken?.(data.content);
      }
      break;
    case 'function':
      if (Array.isArray(data.function_calls)) {
        callbacks.onFunctionCalls?.(data.function_calls as any);
      }
      break;
    case 'chart':
      if (data.chart_updates) {
        callbacks.onChartUpdates?.(data.chart_updates as Record<string, unknown>);
      }
      break;
    case 'usage':
      if (data.usage) {
        callbacks.onUsage?.(data.usage as TokenUsage);
      }
      break;
    case 'error':
      callbacks.onError?.(data.error as string || '未知錯誤');
      break;
    case 'done':
      callbacks.onDone?.(data.conversation_id as string | undefined, data);
      break;
    default:
      break;
  }
}

// ===== 設定 API =====

export async function configureLLM(
  provider: LLMProvider,
  apiKey?: string,
  modelName?: string,
  baseUrl?: string
) {
  // 將 API Key 傳送到後端加密儲存，回傳 session_id
  const res = await api.post('/config/llm', {
    provider,
    api_key: apiKey,
    model_name: modelName,
    base_url: baseUrl,
  });
  return res.data; // { status, provider, session_id, message }
}

export async function testLLMConnection(
  provider: LLMProvider,
  sessionId?: string,
  apiKey?: string,
  baseUrl?: string,
  modelName?: string,
) {
  const res = await api.post('/config/llm/test', {
    provider,
    session_id: sessionId,
    api_key: apiKey,
    base_url: baseUrl,
    model_name: modelName,
  });
  return res.data;
}

export async function fetchLLMProviders() {
  const res = await api.get('/config/llm/providers');
  return res.data;
}

export async function discoverModels(
  provider: LLMProvider,
  sessionId?: string,
  apiKey?: string,
  baseUrl?: string
) {
  const res = await api.post('/config/llm/models', {
    provider,
    session_id: sessionId,
    api_key: apiKey,
    base_url: baseUrl,
  });
  return res.data; // { status, provider, models: [{id, name, description}], total }
}

// ===== 知識蒸餾 API =====

export async function fetchDistillStatus() {
  const res = await api.get('/chat/distill/status');
  return res.data;
}

export async function previewDistill(
  provider: LLMProvider,
  sessionId?: string,
) {
  const res = await api.post('/chat/distill/preview', {
    message: '_distill_',
    provider,
    session_id: sessionId,
  }, { timeout: 120000 }); // 蒸餾可能需要較長時間
  return res.data;
}

export async function confirmDistill(
  previews: Array<Record<string, unknown>>,
  profile: string,
  totalTokens: number,
  provider: LLMProvider,
  sessionId?: string,
) {
  const res = await api.post('/chat/distill/confirm', {
    message: '_confirm_distill_',
    provider,
    session_id: sessionId,
    chart_state: {
      previews,
      profile,
      total_tokens_used: totalTokens,
    },
  });
  return res.data;
}

export async function fetchDistilledKnowledge() {
  const res = await api.get('/chat/distill/knowledge');
  return res.data;
}

// ===== 對話歷史 API =====

export async function fetchChatHistory(limit: number = 20) {
  const res = await api.get('/chat/history', { params: { limit } });
  return res.data;
}

export async function fetchConversationMessages(conversationId: string) {
  const res = await api.get(`/chat/history/${conversationId}`);
  return res.data;
}

export async function deleteConversation(conversationId: string) {
  const res = await api.delete(`/chat/history/${conversationId}`);
  return res.data;
}

// ===== Token 用量查詢 API =====

export async function fetchUsageSummary(
  provider: LLMProvider,
  sessionId?: string,
) {
  const res = await api.post('/config/usage/summary', {
    provider,
    session_id: sessionId,
  });
  return res.data;
}

export async function fetchUsageDaily(
  provider: LLMProvider,
  sessionId?: string,
) {
  const res = await api.post('/config/usage/daily', {
    provider,
    session_id: sessionId,
  });
  return res.data;
}

// ===== 數據同步 API（完整版）=====

export interface SyncRequestParams {
  symbols: string[];
  timeframes: string[];
  exchanges: string[];
  start_date?: string;
  end_date?: string;
  days?: number;
  force_update?: boolean;
}

export async function triggerDataSync(params: SyncRequestParams) {
  const res = await api.post('/data/sync', params);
  return res.data;
}

export async function fetchSyncStatus() {
  const res = await api.get('/data/sync-status');
  return res.data;
}

export async function fetchSyncTaskProgress(taskId: string) {
  const res = await api.get(`/data/sync-task/${taskId}`);
  return res.data;
}

export async function fetchSyncTaskLogs(taskId: string, since: number = 0) {
  const res = await api.get(`/data/sync-task/${taskId}/logs`, { params: { since } });
  return res.data;
}

export async function fetchAvailableExchanges() {
  const res = await api.get('/data/available-exchanges');
  return res.data;
}

export async function fetchAvailableSymbols() {
  const res = await api.get('/data/available-symbols');
  return res.data;
}

// ===== 知識匯出 =====

export async function exportKnowledgePDF(
  includeFragments = true,
  includeCache = false,
  includeHistory = true,
  sessionId?: string,
) {
  const res = await api.get('/export/knowledge-pdf', {
    params: {
      include_fragments: includeFragments,
      include_cache: includeCache,
      include_history: includeHistory,
      session_id: sessionId || undefined,
    },
    responseType: 'blob',
    timeout: 300000,
  });
  const blob = new Blob([res.data], { type: 'application/pdf' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  const now = new Date().toISOString().slice(0, 10);
  a.download = `阿斯拉分析報告_${now}.pdf`;
  a.click();
  URL.revokeObjectURL(url);
}

// ===== 使用者自訂分析策略 =====

export interface UserStrategy {
  id: string;
  title: string;
  content: string;
  enabled: boolean;
  created_at: string;
  updated_at: string;
}

export async function fetchStrategies(): Promise<UserStrategy[]> {
  const res = await api.get('/config/strategies');
  return res.data?.strategies || [];
}

export async function addStrategy(title: string, content: string, enabled = true): Promise<UserStrategy> {
  const res = await api.post('/config/strategies', { title, content, enabled });
  return res.data?.strategy;
}

export async function updateStrategy(
  id: string, updates: { title?: string; content?: string; enabled?: boolean }
): Promise<UserStrategy> {
  const res = await api.put(`/config/strategies/${id}`, updates);
  return res.data?.strategy;
}

export async function deleteStrategy(id: string): Promise<void> {
  await api.delete(`/config/strategies/${id}`);
}

// ===== 因子掃描 =====

export interface FactorScanRequest {
  symbol: string;
  timeframe: string;
  forward_period?: number;
  top_n?: number;
}

export interface FactorScanItem {
  factor: string;
  ic_recent: number;
  ic_full: number;
  decay_trend: string;
  decay_curve: (number | null)[];
  half_life: number | null;
  confidence: string;
  status_label: string;
  stars: number;
  samples: number;
}

export interface FactorScanResult {
  status: string;
  message?: string;
  symbol?: string;
  timeframe?: string;
  total_bars?: number;
  recent_bars?: number;
  forward_period?: number;
  regime?: { label: string; adx: number | null; atr: number | null };
  total_factors_scanned?: number;
  positive_top?: FactorScanItem[];
  negative_top?: FactorScanItem[];
  combo_top?: {
    positive_combos?: { factor_a: string; factor_b: string; combo_ic: number; combo_abs_ic: number }[];
    negative_combos?: { factor_a: string; factor_b: string; combo_ic: number; combo_abs_ic: number }[];
    hedge_combos?: { factor_a: string; factor_b: string; combo_ic: number; combo_abs_ic: number }[];
  };
  quantile_analysis?: Record<string, {
    factor: string;
    quantile_returns_pct: (number | null)[];
    quantile_ranges?: { low: number; high: number; label: string }[];
    is_monotonic: boolean;
    spread_pct: number;
    best_quantile?: { index: number; range: { low: number; high: number }; return_pct: number; entry_suggestion: string } | null;
  }>;
  high_correlation_warnings?: { factor_a: string; factor_b: string; correlation: number }[];
  effective_count?: number;
}

export async function runFactorScan(req: FactorScanRequest): Promise<FactorScanResult> {
  try {
    const res = await api.post('/factor-scan/scan', req);
    return res.data;
  } catch (err: unknown) {
    return { status: 'error', message: (err as Error)?.message || '因子掃描失敗' };
  }
}

// ===== 健康檢查 =====

export async function healthCheck() {
  const res = await api.get('/health');
  return res.data;
}

export default api;
