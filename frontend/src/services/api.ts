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

/** 從 Axios 錯誤中提取後端回傳的訊息，優先取 response.data 中的 message/detail */
function extractErrorMessage(err: unknown, fallback: string): string {
  const ax = err as Record<string, any>;
  return ax?.response?.data?.message
    || ax?.response?.data?.detail
    || ax?.message
    || fallback;
}

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

export interface ProgressInfo {
  percentage: number;
  completed: number;
  total: number;
  current_task: string;
  message: string;
}

export interface StreamCallbacks {
  onThinking?: () => void;
  onStatus?: (message: string) => void;
  onProgress?: (progress: ProgressInfo) => void;
  onToken?: (content: string) => void;
  onFunctionCalls?: (calls: Array<{ name: string; arguments: Record<string, unknown> }>) => void;
  onChartUpdates?: (updates: Record<string, unknown>) => void;
  onUsage?: (usage: TokenUsage) => void;
  onError?: (error: string) => void;
  onDone?: (conversationId?: string, hints?: Record<string, unknown>) => void;
  // v100：結論卡「📈 系統參考」由系統替換為實際命中率
  onAccuracyInject?: (data: { old_pattern: string; new_text: string }) => void;
}

/** SSE 串流回傳值：promise 等待完成，abort 可主動斷開連線 */
export interface StreamHandle {
  promise: Promise<void>;
  abort: () => void;
}

export function streamChatMessage(
  message: string,
  callbacks: StreamCallbacks,
  conversationId?: string,
  chartState?: Record<string, unknown>,
  llmProvider?: LLMProvider,
  sessionId?: string,
  chatHistory?: Array<{ role: string; content: string }>,
  mode?: string,
  chartScreenshot?: string,
): StreamHandle {
  const controller = new AbortController();

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
  if (chartScreenshot) {
    body.chart_screenshot = chartScreenshot;
  }

  let doneEmitted = false;
  const wrappedCallbacks: StreamCallbacks = {
    ...callbacks,
    onDone: (convId, hints) => {
      doneEmitted = true;
      callbacks.onDone?.(convId, hints);
    },
  };

  const MAX_RETRIES = 2;
  const RETRY_DELAYS = [2000, 5000];

  const promise = (async () => {
    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      // 已被外部 abort → 立即停止
      if (controller.signal.aborted) break;

      // ★ fetch() headers 30 秒保護：若 browser 連線池 / 後端佔線導致卡住，直接 abort
      const HEADERS_TIMEOUT_MS = 30_000;
      let headersTimer: ReturnType<typeof setTimeout> | null = setTimeout(() => {
        controller.abort();
      }, HEADERS_TIMEOUT_MS);
      try {
        const response = await fetch('/api/chat/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
          signal: controller.signal,
        });
        clearTimeout(headersTimer); headersTimer = null;

        if (!response.ok) {
          const errText = await response.text();
          if (response.status >= 400 && response.status < 500) {
            wrappedCallbacks.onError?.(`伺服器錯誤 (${response.status}): ${errText}`);
            if (!doneEmitted) { doneEmitted = true; callbacks.onDone?.(); }
            return;
          }
          if (attempt < MAX_RETRIES) {
            wrappedCallbacks.onStatus?.(`連線異常，${RETRY_DELAYS[attempt] / 1000}s 後重試...`);
            await new Promise(r => setTimeout(r, RETRY_DELAYS[attempt]));
            continue;
          }
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
        const STREAM_TIMEOUT_MS = 300_000;

        try {
          while (true) {
            if (controller.signal.aborted) break;

            const readPromise = reader.read();
            let timeoutId: ReturnType<typeof setTimeout>;
            const timeoutPromise = new Promise<{ done: true; value: undefined }>((resolve) => {
              timeoutId = setTimeout(() => resolve({ done: true, value: undefined }), STREAM_TIMEOUT_MS);
            });
            const { done, value } = await Promise.race([readPromise, timeoutPromise]);
            clearTimeout(timeoutId!);
            if (done) {
              if (!value) {
                // 超時：強制斷開 HTTP 連線，通知後端停止處理
                controller.abort();
                wrappedCallbacks.onError?.('分析連線無回應超過 300 秒，已自動斷開（如分析仍在進行請查看後端 log）');
              }
              break;
            }

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
        } finally {
          // 確保 reader 關閉，斷開 HTTP 連線，通知後端停止
          reader.cancel().catch(() => {});
        }

        break; // 成功完成

      } catch (err: unknown) {
        // AbortError 代表主動終止，不重試
        if ((err as Error)?.name === 'AbortError') break;

        if (attempt < MAX_RETRIES && !doneEmitted) {
          wrappedCallbacks.onStatus?.(`網路中斷，${RETRY_DELAYS[attempt] / 1000}s 後重試...`);
          await new Promise(r => setTimeout(r, RETRY_DELAYS[attempt]));
          continue;
        }
        callbacks.onError?.((err as Error)?.message || '網路連線失敗');
      } finally {
        if (headersTimer) { clearTimeout(headersTimer); headersTimer = null; }
      }
    }

    if (!doneEmitted) {
      callbacks.onDone?.();
    }
  })();

  return {
    promise,
    abort: () => controller.abort(),
  };
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
    case 'progress':
      callbacks.onProgress?.(data as unknown as ProgressInfo);
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
    case 'accuracy_inject':
      callbacks.onAccuracyInject?.(data as unknown as { old_pattern: string; new_text: string });
      break;
    default:
      break;
  }
}

// ===== 台股掃描 API =====

export interface TwScanResult {
  code: string;
  name: string;
  market: string;         // "listed" | "otc"
  industry: string;
  price: number;
  price_date: string;     // YYYY-MM-DD
  bb_width_pctile: number;
  bb_width: number;
  volume_5d_avg: number;  // 張
  change_20d: number;     // %
}

export interface TwScanProgress {
  current: number;
  total: number;
  found: number;
  fail: number;
  eta_sec: number;
}

export interface TwScanFailure {
  code: string;
  name: string;
  market: string;      // "listed" | "otc"
  industry: string;
  reason: string;
}

export interface TwScanDone {
  total_scanned: number;
  total_found: number;
  total_fail: number;
  duration_sec: number;
  failures?: TwScanFailure[];
}

export interface TwScanCallbacks {
  onProgress?: (p: TwScanProgress) => void;
  onResult?: (r: TwScanResult) => void;
  onFailure?: (f: TwScanFailure) => void;
  onWarning?: (message: string) => void;
  onDone?: (summary: TwScanDone) => void;
  onError?: (error: string) => void;
}

export interface TwScanRequest {
  timeframe?: string;
  pctile_threshold?: number;
  markets?: string[];
  // 進階過濾
  min_volume?: number;
  require_healthy_trend?: boolean;
  max_adx?: number | null;
  persistence_bars?: number;
  min_abs_bb_width?: number;
  history_days?: number;
}

export function streamTwScan(
  request: TwScanRequest,
  callbacks: TwScanCallbacks,
): StreamHandle {
  const controller = new AbortController();

  const promise = (async () => {
    try {
      const response = await fetch('/api/scanner/tw-bb-width', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
        signal: controller.signal,
      });
      if (!response.ok) {
        const errText = await response.text();
        callbacks.onError?.(`伺服器錯誤 (${response.status}): ${errText}`);
        return;
      }
      const reader = response.body?.getReader();
      if (!reader) {
        callbacks.onError?.('瀏覽器不支援 Streaming');
        return;
      }
      const decoder = new TextDecoder();
      let buffer = '';
      let currentEvent = '';
      try {
        while (true) {
          if (controller.signal.aborted) break;
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';
          for (const line of lines) {
            if (line.startsWith('event: ')) {
              currentEvent = line.slice(7).trim();
            } else if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                _dispatchTwScanEvent(currentEvent, data, callbacks);
              } catch { /* ignore */ }
              currentEvent = '';
            }
          }
        }
      } finally {
        reader.cancel().catch(() => {});
      }
    } catch (err: unknown) {
      if ((err as Error)?.name === 'AbortError') return;
      callbacks.onError?.((err as Error)?.message || '網路連線失敗');
    }
  })();

  return { promise, abort: () => controller.abort() };
}

function _dispatchTwScanEvent(
  event: string,
  data: Record<string, unknown>,
  callbacks: TwScanCallbacks,
) {
  switch (event) {
    case 'progress':
      callbacks.onProgress?.(data as unknown as TwScanProgress);
      break;
    case 'result':
      callbacks.onResult?.(data as unknown as TwScanResult);
      break;
    case 'failure':
      callbacks.onFailure?.(data as unknown as TwScanFailure);
      break;
    case 'warning':
      callbacks.onWarning?.((data.message as string) || '');
      break;
    case 'done':
      callbacks.onDone?.(data as unknown as TwScanDone);
      break;
    case 'error':
      callbacks.onError?.((data.error as string) || '未知錯誤');
      break;
    default:
      break;
  }
}

// ===== 台股掃描歷史 API（Phase 4） =====

export interface TwScanSummary {
  scan_id: string;
  scanned_at: string;
  timeframe: string;
  params: Record<string, unknown>;
  total_scanned: number;
  total_found: number;
  total_fail: number;
  duration_sec: number;
}

export async function listTwScanHistory(limit = 20): Promise<TwScanSummary[]> {
  const res = await api.get(`/scanner/tw-bb-width/history?limit=${limit}`);
  return res.data.scans;
}

export async function getTwScanResult(scanId: string) {
  const res = await api.get(`/scanner/tw-bb-width/history/${scanId}`);
  return res.data as {
    scan_id: string;
    scanned_at: string;
    timeframe: string;
    params: Record<string, unknown>;
    results: TwScanResult[];
    total_scanned: number;
    total_found: number;
    total_fail: number;
    duration_sec: number;
    failures: TwScanFailure[];
  };
}

export async function deleteTwScan(scanId: string) {
  await api.delete(`/scanner/tw-bb-width/history/${scanId}`);
}

export interface TwScanRevisitItem extends TwScanResult {
  scan_price: number;
  scan_date: string;
  current_price: number;
  current_date: string;
  return_pct: number;
}

export async function revisitTwScan(scanId: string): Promise<{
  scan_id: string;
  scanned_at: string;
  total_original: number;
  total_revisited: number;
  results: TwScanRevisitItem[];
}> {
  const res = await api.get(`/scanner/tw-bb-width/history/${scanId}/revisit`, { timeout: 300000 });
  return res.data;
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

/**
 * SSE 串流版蒸餾預覽（v100）。
 *
 * 蒸餾需要 N 個 symbol × 1 次 LLM 呼叫 + 1 次 user profile，
 * N 個 symbol 多時總時長可達 3-7 分鐘，遠超 axios 預設 timeout。
 * 改 SSE 後靠 stream 事件持續維持連線，不再被 timeout 中斷。
 *
 * 回呼參數：
 *   onProgress({current, total, message}): 每蒸餾完一個 symbol 觸發
 *   onPreviewItem(item): 每個 symbol 完成蒸餾就推一筆
 *   onError(message): 後端錯誤
 *   onDone(payload): 全部完成，含 previews / profile_preview / total_tokens_used
 *
 * 回傳 abort 函式：呼叫即可主動斷開（並不會影響後端正在跑的 LLM 呼叫）。
 */
export function streamDistillPreview(
  provider: LLMProvider,
  sessionId: string | undefined,
  callbacks: {
    onStatus?: (msg: string, total: number) => void;
    onProgress?: (current: number, total: number, message: string, currentSymbol?: string) => void;
    onPreviewItem?: (item: Record<string, unknown>) => void;
    onError?: (message: string) => void;
    onDone?: (payload: {
      previews: Array<Record<string, unknown>>;
      profile_preview: string | null;
      total_tokens_used: number;
    }) => void;
  },
): () => void {
  const controller = new AbortController();

  (async () => {
    try {
      const response = await fetch('/api/chat/distill/preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: '_distill_',
          provider,
          session_id: sessionId,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        callbacks.onError?.(`伺服器錯誤 (${response.status})`);
        return;
      }

      const reader = response.body?.getReader();
      if (!reader) {
        callbacks.onError?.('瀏覽器不支援 streaming');
        return;
      }
      const decoder = new TextDecoder();
      let buffer = '';
      let currentEvent = '';

      while (true) {
        if (controller.signal.aborted) break;
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
              if (currentEvent === 'status') {
                callbacks.onStatus?.(data.message ?? '', data.total ?? 0);
              } else if (currentEvent === 'progress') {
                callbacks.onProgress?.(data.current ?? 0, data.total ?? 0, data.message ?? '', data.current_symbol);
              } else if (currentEvent === 'preview_item') {
                callbacks.onPreviewItem?.(data);
              } else if (currentEvent === 'error') {
                callbacks.onError?.(data.message ?? '未知錯誤');
              } else if (currentEvent === 'done') {
                if (data.status === 'ok') {
                  callbacks.onDone?.({
                    previews: data.previews ?? [],
                    profile_preview: data.profile_preview ?? null,
                    total_tokens_used: data.total_tokens_used ?? 0,
                  });
                }
                return;
              }
            } catch (e) {
              // 解析失敗 → 略過該行
            }
          }
        }
      }
    } catch (e) {
      if (controller.signal.aborted) return;  // 主動取消不算錯誤
      const msg = e instanceof Error ? e.message : String(e);
      callbacks.onError?.(`連線中斷：${msg}`);
    }
  })();

  return () => controller.abort();
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

// ===== 知識碎片 API =====

export interface FragmentItem {
  id: number;
  content: string;
  type: string;
  symbol: string;
  source_question: string;
  hit_count: number;
  quality_score: number;
  age_days: number;
  created_at: string;
  is_seed: boolean;
}

export interface FragmentListResult {
  status: string;
  fragments: FragmentItem[];
  total: number;
  symbols: string[];
  types: string[];
}

export async function fetchFragments(params?: {
  symbol?: string;
  fragment_type?: string;
  sort_by?: string;
  limit?: number;
  offset?: number;
}): Promise<FragmentListResult> {
  const res = await api.get('/chat/fragments', { params });
  return res.data;
}

export async function deleteFragment(fragmentId: number) {
  const res = await api.delete(`/chat/fragments/${fragmentId}`);
  return res.data;
}

export async function addUserNote(note: string, symbol?: string) {
  const res = await api.post('/chat/fragments/note', { message: note, symbol });
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
  p_value_recent?: number | null;
  p_value_full?: number | null;
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
  strategy_tiers?: {
    strict?: StrategyTier;
    moderate?: StrategyTier;
    loose?: StrategyTier;
  };
  effective_count?: number;
  ic_threshold_used?: number;
  p_value_cutoff?: number;
  oos_split_ratio?: string;
  dedup_threshold?: number;
  scan_warnings?: string[];
}

export interface StrategyTierCondition {
  factor: string;
  description: string;
  abs_ic: number;
  return_pct: number;
}

export interface StrategyTier {
  conditions: StrategyTierCondition[];
  trigger_count: number;
  factor_count: number;
  source?: string;
}

export async function runFactorScan(req: FactorScanRequest): Promise<FactorScanResult> {
  try {
    const res = await api.post('/factor-scan/scan', req, { timeout: 180_000 });
    return res.data;
  } catch (err: unknown) {
    return { status: 'error', message: extractErrorMessage(err, '因子掃描失敗') };
  }
}

// ===== 觸發次數預覽 =====

export interface TriggerCondition {
  factor: string;
  operator: string;
  value: number;
  value2?: number;
  enabled: boolean;
}

export interface TriggerPreviewResult {
  status: string;
  message?: string;
  trigger_count: number;
  total_bars: number;
  trigger_pct: number;
  conditions_used: number;
  last_trigger_time?: string;
}

export async function triggerPreview(
  symbol: string, timeframe: string, conditions: TriggerCondition[],
): Promise<TriggerPreviewResult> {
  try {
    const res = await api.post('/factor-scan/trigger-preview', { symbol, timeframe, conditions });
    return res.data;
  } catch (err: unknown) {
    return { status: 'error', message: extractErrorMessage(err, '預覽失敗'),
             trigger_count: 0, total_bars: 0, trigger_pct: 0, conditions_used: 0 };
  }
}

// ===== ML 增強 =====

export interface MLModel {
  id: string;
  name: string;
  category: string;
  description: string;
  available: boolean;
  missing_deps: string[];
  min_samples: number;
  supports_gpu: boolean;
  training_speed: string;
  default_config: Record<string, any>;
}

export interface MLSettings {
  enabled: string;
  model_id: string;
  feature_set: string;
  forward_period: number;
  threshold: number;
  show_explanation: boolean;
  train_window: number;
  retrain_interval: number;
  walk_forward: boolean;
  wf_windows: number;
  min_samples: number;
  consensus_mode: string;
  target_direction: string;
  target_threshold: number;
  lookback_window: number;
  symbol?: string;
  _warnings?: string[];
  _per_symbol?: boolean;
}

export async function fetchMLModels(): Promise<MLModel[]> {
  const res = await api.get('/ml/models');
  return res.data.models || [];
}

export async function fetchMLSettings(): Promise<MLSettings> {
  const res = await api.get('/ml/settings');
  return res.data.settings;
}

export async function updateMLSettings(updates: Partial<MLSettings>): Promise<MLSettings> {
  const res = await api.put('/ml/settings', updates);
  return res.data.settings;
}

export async function fetchMLStatus(symbol: string, timeframe: string = '4h', modelId?: string) {
  const params: Record<string, string> = { symbol, timeframe };
  if (modelId) params.model_id = modelId;
  const res = await api.get('/ml/status', { params });
  return res.data;
}

export async function trainMLModel(params: {
  symbol: string;
  timeframe?: string;
  model_id?: string;
  start_date?: string;
  end_date?: string;
}) {
  const res = await api.post('/ml/train', params);
  return res.data;
}

export async function predictML(params: {
  symbol: string;
  timeframe?: string;
  model_id?: string;
  consensus?: boolean;
}) {
  const res = await api.post('/ml/predict', params);
  return res.data;
}

export async function compareMLModels(params: {
  symbol: string;
  timeframe?: string;
  model_ids?: string[];
  feature_set?: string;
  forward_period?: number;
}) {
  const res = await api.post('/ml/compare', params);
  return res.data;
}

export async function retrainMLModel(params: {
  symbol: string;
  timeframe?: string;
  model_id?: string;
}) {
  const res = await api.post('/ml/retrain', params);
  return res.data;
}

export async function fetchTrainedModels(symbol?: string) {
  const params: Record<string, string> = {};
  if (symbol) params.symbol = symbol;
  const res = await api.get('/ml/trained', { params });
  return res.data;
}

// ===== ML 監控 API =====

export async function fetchMLPerformanceHistory(symbol?: string, modelId?: string) {
  const params: Record<string, string> = {};
  if (symbol) params.symbol = symbol;
  if (modelId) params.model_id = modelId;
  const res = await api.get('/ml/performance-history', { params });
  return res.data;
}

export async function checkMLHealth(symbol: string, timeframe: string = '4h') {
  const res = await api.get('/ml/health', { params: { symbol, timeframe } });
  return res.data;
}

export async function fetchMLPredictionAccuracy(symbol?: string) {
  const params: Record<string, string> = {};
  if (symbol) params.symbol = symbol;
  const res = await api.get('/ml/prediction-accuracy', { params });
  return res.data;
}

// ===== 預測追蹤 API =====

export async function fetchPredictionStats(symbol?: string, days?: number) {
  const params: Record<string, string> = {};
  if (symbol) params.symbol = symbol;
  if (days) params.days = String(days);
  const res = await api.get('/predictions/stats', { params });
  return res.data;
}

export async function fetchActivePredictions(symbol?: string) {
  const params: Record<string, string> = {};
  if (symbol) params.symbol = symbol;
  const res = await api.get('/predictions/active', { params });
  return res.data;
}

export async function fetchPredictionHistory(symbol?: string, limit?: number) {
  const params: Record<string, string> = {};
  if (symbol) params.symbol = symbol;
  if (limit) params.limit = String(limit);
  const res = await api.get('/predictions/history', { params });
  return res.data;
}

export async function updatePredictionNote(predId: number, note: string) {
  const res = await api.put(`/predictions/${predId}/note`, { note });
  return res.data;
}

export async function generateReview(
  startDate?: string, endDate?: string, symbol?: string, sessionId?: string,
) {
  const body: Record<string, string> = {};
  if (startDate) body.start_date = startDate;
  if (endDate) body.end_date = endDate;
  if (symbol) body.symbol = symbol;
  if (sessionId) body.session_id = sessionId;
  const res = await api.post('/predictions/review', body, { timeout: 120000 });
  return res.data;
}

export async function fetchReviewHistory(limit: number = 10) {
  const res = await api.get('/predictions/reviews', { params: { limit: String(limit) } });
  return res.data;
}

export async function clearPredictions(symbol?: string) {
  const params: Record<string, string> = {};
  if (symbol) params.symbol = symbol;
  const res = await api.delete('/predictions/clear', { params });
  return res.data;
}

// ===== 自動調整 =====

export async function fetchAdjustments(symbol?: string) {
  const params: Record<string, string> = {};
  if (symbol) params.symbol = symbol;
  const res = await api.get('/predictions/adjustments', { params });
  return res.data;
}

export async function overrideAdjustment(adjId: number, value?: number) {
  const res = await api.put(`/predictions/adjustments/${adjId}/override`, { value });
  return res.data;
}

export async function recalculateAdjustments(symbol?: string) {
  const params: Record<string, string> = {};
  if (symbol) params.symbol = symbol;
  const res = await api.post('/predictions/adjustments/recalculate', null, { params });
  return res.data;
}

// ===== 情境預測 =====

export async function fetchScenarios(symbol: string, timeframe: string = '1d', forwardBars: number = 5) {
  const res = await api.post('/scenario/predict', {
    symbol,
    timeframe,
    forward_bars: forwardBars,
  });
  return res.data;
}

export async function fetchLatestScenarios(symbol: string, timeframe: string = '1d') {
  const res = await api.get(`/scenario/latest/${encodeURIComponent(symbol)}`, {
    params: { timeframe },
  });
  return res.data;
}

// ===== SMC 訂單流分析 =====

export async function fetchSMCAnalysis(symbol: string, timeframe: string = '1d', lookback: number = 120) {
  const res = await api.post('/smc/analyze', { symbol, timeframe, lookback });
  return res.data;
}

export async function fetchLatestSMC(symbol: string, timeframe: string = '1d') {
  const res = await api.get(`/smc/latest/${encodeURIComponent(symbol)}`, {
    params: { timeframe },
  });
  return res.data;
}

export function exportSMCCsv(symbol: string, timeframe: string = '1d') {
  const encodedSymbol = encodeURIComponent(symbol);
  const baseUrl = api.defaults.baseURL || '/api';
  window.open(`${baseUrl}/smc/export-csv/${encodedSymbol}?timeframe=${timeframe}`, '_blank');
}

// ===== 系統通用設定 =====

export async function fetchSystemSettings(): Promise<Record<string, unknown>> {
  const res = await api.get('/config/system-settings');
  return res.data.settings;
}

export async function updateSystemSettings(updates: Record<string, unknown>): Promise<Record<string, unknown>> {
  const res = await api.put('/config/system-settings', updates);
  return res.data.settings;
}

// ===== Session 快取管理 =====

export async function clearSessionCache(): Promise<string> {
  const res = await api.post('/config/clear-session-cache');
  return res.data.message;
}

// ===== 預警系統 =====

export interface Alert {
  id: number;
  symbol: string;
  timeframe: string;
  alert_type: string;
  direction: string;
  confidence: string;
  trigger_conditions?: string;
  signal_score: number;
  created_at: string;
  expires_at: string;
  status: string;
  outcome_pct?: number;
  move_probability?: number;
  evidence_summary?: string;
  probability_detail?: Record<string, { up_pct: number; down_pct: number; any_move_pct: number }>;
  feature_attribution?: Array<{ feature: string; presence_in_hits: number; lift: number }>;
}

export async function fetchActiveAlerts(symbol?: string) {
  const params = symbol ? { symbol } : {};
  const res = await api.get('/alerts/active', { params });
  return res.data;
}

export async function fetchAlertHistory(symbol?: string, limit = 50) {
  const params: Record<string, unknown> = { limit };
  if (symbol) params.symbol = symbol;
  const res = await api.get('/alerts/history', { params });
  return res.data;
}

export async function triggerManualScan() {
  const res = await api.post('/alerts/scan', {}, { timeout: 120_000 });
  return res.data;
}

export async function dismissAlert(alertId: number) {
  const res = await api.post(`/alerts/dismiss/${alertId}`);
  return res.data;
}

export async function fetchMovementProbability(symbol: string, timeframe = '4h', threshold = 3.0) {
  const res = await api.get(`/alerts/probability/${symbol}`, { params: { timeframe, threshold } });
  return res.data;
}

export async function checkSymbolsData(symbols: string[], timeframe = '4h') {
  const res = await api.post('/alerts/check-data', { symbols, timeframe });
  return res.data.data_available as Record<string, boolean>;
}

// ===== 掃描器校準 =====

export async function fetchScannerCalibration() {
  const res = await api.get('/config/scanner-calibration');
  return res.data.calibration;
}

export async function resetScannerCalibration() {
  const res = await api.post('/config/scanner-calibration/reset');
  return res.data;
}

// ===== 全量歷史特徵分析 =====

export async function fetchFeatureProfiles() {
  const res = await api.get('/config/feature-profiles');
  return res.data.profiles;
}

export async function recomputeFeatureProfiles() {
  const res = await api.post('/config/feature-profiles/recompute');
  return res.data.result;
}

// ===== 健康檢查 =====

export async function healthCheck() {
  const res = await api.get('/health');
  return res.data;
}

// ── 台股名稱動態查詢（帶快取） ──

const _twNameCache: Record<string, string> = {
  'TWII': '加權指數',
  'TWOII': '櫃買指數',
};
const _twNamePending: Record<string, Promise<string>> = {};

export async function fetchTwStockName(code: string): Promise<string> {
  code = code.toUpperCase();
  if (_twNameCache[code] !== undefined) return _twNameCache[code];
  // 避免同一代碼重複請求
  if (_twNamePending[code] !== undefined) return _twNamePending[code];
  _twNamePending[code] = api.get('/tw-stock-name', { params: { code } })
    .then((res) => {
      const name = res.data?.name || '';
      _twNameCache[code] = name;
      delete _twNamePending[code];
      return name;
    })
    .catch(() => {
      delete _twNamePending[code];
      return '';
    });
  return _twNamePending[code];
}

export function getTwStockNameSync(code: string): string | undefined {
  return _twNameCache[code.toUpperCase()];
}

export function setTwStockNameCache(code: string, name: string) {
  _twNameCache[code.toUpperCase()] = name;
}

/** 搜尋台股：支援代碼或中文名稱 */
export async function searchTwStock(query: string): Promise<{ code: string; name: string; symbol: string }[]> {
  try {
    const res = await api.get('/chart/tw-stock-search', { params: { q: query } });
    return res.data?.results || [];
  } catch {
    return [];
  }
}

/** 取得有下載數據的標的清單（去重，過濾掉數據不足的） */
export async function fetchDownloadedSymbols(): Promise<{ symbol: string; records: number }[]> {
  try {
    const res = await api.get('/chart/available/list');
    const items: { symbol: string; records: number }[] = res.data?.data || [];
    // 去重：同一 symbol 不同 timeframe 合併，取最大 records
    const map = new Map<string, number>();
    for (const item of items) {
      const existing = map.get(item.symbol) || 0;
      map.set(item.symbol, Math.max(existing, item.records));
    }
    // 過濾：至少 10 根 K 線才算有效數據
    return Array.from(map.entries())
      .filter(([, records]) => records >= 10)
      .map(([symbol, records]) => ({ symbol, records }));
  } catch {
    return [];
  }
}

export default api;
