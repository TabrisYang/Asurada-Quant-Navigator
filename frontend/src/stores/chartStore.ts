/** 阿斯拉量化系統 — 圖表集中式狀態管理 (Zustand) — 完整版 */

import { create } from 'zustand';
import { loadPersistedSession } from '../services/session';
import type {
  Timeframe,
  OHLCVData,
  ActiveIndicator,
  Annotation,
  ChatMessage,
  LLMConfig,
} from '../types';
import { fetchChartData, calculateIndicator } from '../services/api';

interface ChartStore {
  // ===== 圖表狀態 =====
  symbol: string;
  timeframe: Timeframe;
  startDate: string | null;
  endDate: string | null;
  ohlcvData: OHLCVData[];
  loading: boolean;
  dataError: string | null;

  // ===== 指標狀態 =====
  activeIndicators: ActiveIndicator[];

  // ===== 標記狀態 =====
  annotations: Annotation[];

  // ===== 對話狀態 =====
  messages: ChatMessage[];
  conversationId: string | null;
  chatLoading: boolean;

  // ===== LLM 設定 =====
  llmConfig: LLMConfig;

  // ===== 同步面板 =====
  showSyncPanel: boolean;

  // ===== 圖表操作 =====
  setSymbol: (symbol: string) => void;
  setTimeframe: (timeframe: Timeframe) => void;
  setDateRange: (start: string | null, end: string | null) => void;
  setOHLCVData: (data: OHLCVData[]) => void;
  setLoading: (loading: boolean) => void;

  // ===== 數據載入（核心）=====
  loadChartData: (symbol?: string, timeframe?: Timeframe) => Promise<void>;

  // ===== 指標操作 =====
  addIndicator: (indicator: ActiveIndicator) => void;
  removeIndicator: (id: string) => void;
  updateIndicatorParams: (id: string, params: Record<string, number>) => void;
  toggleIndicatorVisibility: (id: string) => void;
  updateIndicatorData: (id: string, data: Record<string, (number | null)[]>) => void;
  clearIndicators: () => void;
  loadIndicatorData: (indicator: ActiveIndicator) => Promise<void>;
  reloadAllIndicators: () => Promise<void>;

  // ===== 標記操作 =====
  addAnnotation: (annotation: Annotation) => void;
  removeAnnotation: (id: string) => void;
  clearAnnotations: () => void;
  toggleAnnotationGroup: (groupId: string) => void;
  removeAnnotationGroup: (groupId: string) => void;
  getAnnotationGroups: () => { groupId: string; groupName: string; visible: boolean; count: number; color: string }[];

  // ===== 對話操作 =====
  addMessage: (message: ChatMessage) => void;
  updateMessage: (id: string, updates: Partial<ChatMessage>) => void;
  setChatLoading: (loading: boolean) => void;
  clearMessages: () => void;

  // ===== LLM 設定操作 =====
  setLLMConfig: (config: Partial<LLMConfig>) => void;

  // ===== 同步面板 =====
  setShowSyncPanel: (show: boolean) => void;

  // ===== 外部注入聊天訊息（因子掃描→回測）=====
  pendingChatMessage: string | null;
  setPendingChatMessage: (msg: string | null) => void;

  // ===== 因子掃描結果（共享給助手）=====
  lastFactorScan: { symbol: string; timeframe: string; timestamp: number; summary: string } | null;
  setLastFactorScan: (scan: { symbol: string; timeframe: string; timestamp: number; summary: string } | null) => void;

  // ===== 圖表截圖（給 LLM 視覺分析用）=====
  captureScreenshot: (() => Promise<string | null>) | null;
  setCaptureScreenshot: (fn: (() => Promise<string | null>) | null) => void;

  // ===== 取得圖表狀態摘要（給 LLM 用）=====
  getChartStateSummary: () => Record<string, unknown>;
}

// 從 localStorage 載入持久化的 LLM session（含模型名稱）
function _loadInitialLLMConfig(): LLMConfig {
  const saved = loadPersistedSession();
  if (saved) {
    return {
      provider: saved.provider as LLMConfig['provider'],
      sessionId: saved.sessionId,
      modelName: saved.modelName,
    };
  }
  return { provider: 'openai' };
}

export const useChartStore = create<ChartStore>((set, get) => ({
  // ===== 初始狀態 =====
  symbol: 'BTC/USDT',
  timeframe: '1d',
  startDate: null,
  endDate: null,
  ohlcvData: [],
  loading: false,
  dataError: null,
  activeIndicators: [],
  annotations: [],
  messages: [],
  conversationId: null,
  chatLoading: false,
  llmConfig: _loadInitialLLMConfig(),
  showSyncPanel: false,
  pendingChatMessage: null,
  setPendingChatMessage: (msg) => set({ pendingChatMessage: msg }),
  lastFactorScan: null,
  setLastFactorScan: (scan) => set({ lastFactorScan: scan }),
  captureScreenshot: null,
  setCaptureScreenshot: (fn) => set({ captureScreenshot: fn }),

  // ===== 圖表操作 =====
  setSymbol: (symbol) => set((state) => {
    if (symbol === state.symbol) return { symbol };
    const divider: ChatMessage = {
      id: `divider-${Date.now()}`,
      role: 'system',
      content: `__SYMBOL_SWITCH__${symbol}`,
      timestamp: new Date().toISOString(),
    };
    return {
      symbol,
      messages: [...state.messages, divider],
      conversationId: null,
      lastFactorScan: null,
    };
  }),
  setTimeframe: (timeframe) => set({ timeframe, lastFactorScan: null }),
  setDateRange: (start, end) => set({ startDate: start, endDate: end }),
  setOHLCVData: (data) => set({ ohlcvData: data }),
  setLoading: (loading) => set({ loading }),

  // ===== 數據載入（核心邏輯）=====
  loadChartData: async (sym?: string, tf?: Timeframe) => {
    const state = get();
    const symbol = sym || state.symbol;
    const timeframe = tf || state.timeframe;

    set({ loading: true, dataError: null });

    try {
      const res = await fetchChartData(
        symbol,
        timeframe,
        state.startDate || undefined,
        state.endDate || undefined
      );

      const ohlcv: OHLCVData[] = res.ohlcv || [];
      set({ ohlcvData: ohlcv, loading: false });

      // 數據更新後，重新計算所有啟用的指標
      const indicators = get().activeIndicators;
      if (indicators.length > 0 && ohlcv.length > 0) {
        // 非阻塞：在背景中重新計算
        setTimeout(() => get().reloadAllIndicators(), 100);
      }
    } catch (err: unknown) {
      const e = err as { response?: { data?: { detail?: string } }; message?: string };
      const errorMsg = e?.response?.data?.detail || e?.message || '載入數據失敗';
      set({ loading: false, dataError: errorMsg, ohlcvData: [] });
    }
  },

  // ===== 指標操作（雙軌同步核心）=====
  addIndicator: (indicator) =>
    set((state) => ({
      activeIndicators: [...state.activeIndicators.filter((i) => i.id !== indicator.id), indicator],
    })),

  removeIndicator: (id) =>
    set((state) => ({
      activeIndicators: state.activeIndicators.filter((i) => i.id !== id),
    })),

  updateIndicatorParams: (id, params) =>
    set((state) => ({
      activeIndicators: state.activeIndicators.map((i) =>
        i.id === id ? { ...i, parameters: { ...i.parameters, ...params } } : i
      ),
    })),

  toggleIndicatorVisibility: (id) =>
    set((state) => ({
      activeIndicators: state.activeIndicators.map((i) =>
        i.id === id ? { ...i, visible: !i.visible } : i
      ),
    })),

  updateIndicatorData: (id, data) =>
    set((state) => ({
      activeIndicators: state.activeIndicators.map((i) =>
        i.id === id ? { ...i, data } : i
      ),
    })),

  clearIndicators: () => set({ activeIndicators: [] }),

  // ===== 載入指標數據（呼叫後端計算）=====
  loadIndicatorData: async (indicator: ActiveIndicator) => {
    const state = get();
    try {
      const res = await calculateIndicator(
        indicator.indicator_type,
        'add',
        {
          ...indicator.parameters,
          symbol: state.symbol,
          timeframe: state.timeframe,
          start_date: state.startDate || undefined,
          end_date: state.endDate || undefined,
        }
      );
      // res = { name, display_mode, data: { series_name: [values] }, parameters }
      if (res?.data) {
        get().updateIndicatorData(indicator.id, res.data);
      }
    } catch (err: unknown) {
      console.error(`載入指標 ${indicator.indicator_type} 失敗:`, (err as Error)?.message);
    }
  },

  reloadAllIndicators: async () => {
    const state = get();
    const promises = state.activeIndicators.map((ind) =>
      get().loadIndicatorData(ind)
    );
    await Promise.all(promises);
  },

  // ===== 標記操作 =====
  addAnnotation: (annotation) =>
    set((state) => ({
      annotations: [...state.annotations, { ...annotation, visible: annotation.visible !== false }],
    })),

  removeAnnotation: (id) =>
    set((state) => ({
      annotations: state.annotations.filter((a) => a.id !== id),
    })),

  clearAnnotations: () => set({ annotations: [] }),

  toggleAnnotationGroup: (groupId) =>
    set((state) => {
      const group = state.annotations.filter((a) => a.groupId === groupId);
      const allVisible = group.every((a) => a.visible !== false);
      return {
        annotations: state.annotations.map((a) =>
          a.groupId === groupId ? { ...a, visible: !allVisible } : a
        ),
      };
    }),

  removeAnnotationGroup: (groupId) =>
    set((state) => ({
      annotations: state.annotations.filter((a) => a.groupId !== groupId),
    })),

  getAnnotationGroups: () => {
    const anns = get().annotations;
    const map = new Map<string, { groupName: string; visible: boolean; count: number; color: string }>();
    for (const a of anns) {
      const gid = a.groupId || '__ungrouped__';
      const existing = map.get(gid);
      if (existing) {
        existing.count++;
        if (a.visible === false) existing.visible = false;
      } else {
        map.set(gid, {
          groupName: a.groupName || '未分組標記',
          visible: a.visible !== false,
          count: 1,
          color: a.color || '#58a6ff',
        });
      }
    }
    return Array.from(map.entries()).map(([groupId, v]) => ({ groupId, ...v }));
  },

  // ===== 對話操作 =====
  addMessage: (message) =>
    set((state) => ({ messages: [...state.messages, message] })),

  updateMessage: (id, updates) =>
    set((state) => ({
      messages: state.messages.map((msg) =>
        msg.id === id ? { ...msg, ...updates } : msg,
      ),
    })),

  setChatLoading: (loading) => set({ chatLoading: loading }),

  clearMessages: () => set({ messages: [], conversationId: null }),

  // ===== LLM 設定 =====
  setLLMConfig: (config) =>
    set((state) => ({ llmConfig: { ...state.llmConfig, ...config } })),

  // ===== 同步面板 =====
  setShowSyncPanel: (show) => set({ showSyncPanel: show }),

  // ===== 圖表狀態摘要（含實際價格數據，給 LLM 分析用）=====
  getChartStateSummary: () => {
    const state = get();
    const data = state.ohlcvData;
    const n = data.length;

    // 基礎 metadata
    const summary: Record<string, unknown> = {
      symbol: state.symbol,
      timeframe: state.timeframe,
      startDate: state.startDate,
      endDate: state.endDate,
      dataPoints: n,
      activeIndicators: state.activeIndicators.map((i) => ({
        type: i.indicator_type,
        params: i.parameters,
        visible: i.visible,
      })),
      annotationCount: state.annotations.length,
    };

    // 加入實際價格數據摘要（讓 LLM 能直接分析）
    if (n > 0) {
      const closes = data.map((d) => d.close);
      const highs = data.map((d) => d.high);
      const lows = data.map((d) => d.low);
      const volumes = data.map((d) => d.volume);

      const currentPrice = closes[n - 1];
      const firstPrice = closes[0];
      const priceChangePercent = ((currentPrice - firstPrice) / firstPrice * 100).toFixed(2);

      summary.priceOverview = {
        currentPrice,
        lastClose: currentPrice,
        periodHigh: Math.max(...highs),
        periodLow: Math.min(...lows),
        firstTimestamp: data[0].timestamp,
        lastTimestamp: data[n - 1].timestamp,
        priceChangePercent: `${priceChangePercent}%`,
        avgVolume: Math.round(volumes.reduce((a, b) => a + b, 0) / n),
      };

      // 根據時間級別動態決定傳送 K 線數量
      const candleCountMap: Record<string, number> = {
        '15m': 20, '1h': 30, '4h': 40, '1d': 50, '1w': 50,
      };
      const recentCount = Math.min(candleCountMap[state.timeframe] ?? 30, n);
      summary.recentCandles = data.slice(-recentCount).map((d) => ({
        t: d.timestamp.slice(0, 16),
        c: +d.close.toFixed(4),
        h: +d.high.toFixed(4),
        l: +d.low.toFixed(4),
        v: Math.round(d.volume),
      }));

      // ★ 注入已啟用指標的實際計算值（最近 5 根 + 方向標籤）
      const RECENT_N = 5;
      const indicatorValues: Record<string, { values: (number | null)[]; trend: string }> = {};

      for (const ind of state.activeIndicators) {
        if (!ind.visible || !ind.data) continue;
        for (const [seriesName, seriesData] of Object.entries(ind.data)) {
          if (!seriesData || seriesData.length === 0) continue;
          const tail = seriesData.slice(-RECENT_N);
          const rounded = tail.map((v) => (v != null ? +v.toFixed(4) : null));

          const validVals = rounded.filter((v): v is number => v != null);
          let trend = '→';
          if (validVals.length >= 2) {
            const first = validVals[0];
            const last = validVals[validVals.length - 1];
            const diff = last - first;
            const threshold = Math.abs(first) * 0.02 || 0.001;
            if (diff > threshold) trend = '↑';
            else if (diff < -threshold) trend = '↓';
          }

          const key = seriesName === 'value' || seriesName === ind.indicator_type
            ? ind.indicator_type
            : `${ind.indicator_type}_${seriesName}`;
          indicatorValues[key] = { values: rounded, trend };
        }
      }

      if (Object.keys(indicatorValues).length > 0) {
        summary.indicatorValues = indicatorValues;
      }
    }

    // ★ 注入因子掃描結果摘要（同標的+同級別時才注入）
    const scan = state.lastFactorScan;
    if (scan && scan.symbol === state.symbol && scan.timeframe === state.timeframe) {
      const ageMin = Math.round((Date.now() - scan.timestamp) / 60000);
      if (ageMin < 60) {
        summary.factorScanSummary = `[因子掃描結果 ${ageMin}分鐘前] ${scan.summary}`;
      }
    }

    return summary;
  },
}));
