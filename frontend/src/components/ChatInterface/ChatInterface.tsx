/** 阿斯拉量化系統 — 對話介面組件（Streaming + Token Usage + 累計用量） */

import { useState, useRef, useEffect, useCallback } from 'react';
import { useChartStore } from '../../stores/chartStore';
import { streamChatMessage, fetchUsageSummary, fetchChatHistory, fetchConversationMessages, fetchDistillStatus, previewDistill, confirmDistill, fetchFragments, deleteFragment, addUserNote } from '../../services/api';
import { toast } from '../Toast';
import type { FragmentItem } from '../../services/api';
import type { ChatMessage, TokenUsage, LLMProvider, Timeframe, Annotation } from '../../types';

const QUICK_QUESTIONS_KEY = 'asla_quick_questions';
const DEFAULT_QUICK_QUESTIONS = [
  '顯示 BTC 週線的布林通道',
  'RSI 低於 30 的時間段有哪些？',
  '幫我看 ETH 的 MACD 黃金交叉',
  '比較 RSI 和 MACD 策略的績效',
];

const ANALYSIS_CATEGORIES = [
  {
    tab: '基礎分析',
    allPrompt: '對 {symbol} 進行完整基礎分析（市場環境判定 + 七維度技術分析 + SMC 智慧資金分析 + 情境預測）',
    items: [
      { label: '市場環境判定', prompt: '分析 {symbol} 的市場環境（Regime + 結構 + 多時框對齊）' },
      { label: '七維度技術分析', prompt: '對 {symbol} 進行完整的七維度技術分析' },
      { label: 'SMC 智慧資金分析', prompt: '分析 {symbol} 的 SMC 訂單流結構' },
      { label: '情境預測', prompt: '預測 {symbol} 未來三種可能情境' },
    ],
  },
  {
    tab: '因子驗證',
    mode: 'factor_validation',
    allPrompt: '對 {symbol} 進行因子驗證（因子 IC 排名 + 組合 IC + Bucket 評分 + 條件機率）',
    items: [
      { label: '因子 IC 排名', prompt: '分析 {symbol} 哪些因子最有效、IC 排名如何' },
      { label: '組合因子分析', prompt: '分析 {symbol} 的雙因子組合 IC，哪些因子搭配效果最好' },
      { label: '條件機率掃描', prompt: '掃描 {symbol} 各指標的條件機率，找出最佳進場區間' },
      { label: '因子群評分', prompt: '對 {symbol} 進行趨勢/動量/波動/量能/結構五群 Bucket 評分' },
    ],
  },
  {
    tab: '策略回測',
    mode: 'strategy_backtest',
    allPrompt: '對 {symbol} 進行策略回測驗證（回測 + Monte Carlo + Walk Forward + CPCV 交叉驗證）',
    items: [
      { label: '多策略比較', prompt: '回測比較 {symbol} 的多種交易策略績效' },
      { label: 'MC + WF 驗證', prompt: '對 {symbol} 進行 Monte Carlo 穩定性測試和 Walk Forward 過擬合檢測' },
      { label: 'CPCV 嚴格驗證', prompt: '對 {symbol} 進行 CPCV 組合淨化交叉驗證（最嚴格的過擬合檢測）' },
      { label: '指標參數校準', prompt: '校準 {symbol} 的最佳指標參數' },
    ],
  },
  {
    tab: '市場體制',
    mode: 'regime_analysis',
    allPrompt: '對 {symbol} 進行市場體制分析（GMM Regime + GARCH 波動率 + HMM 狀態轉移）',
    items: [
      { label: 'GMM Regime 分類', prompt: '用 GMM 分析 {symbol} 當前處於哪種市場體制（強多/弱多/盤整/強空）' },
      { label: 'GARCH 波動率', prompt: '用 GARCH 預測 {symbol} 未來波動率是擴張還是收斂' },
      { label: 'HMM 狀態轉移', prompt: '用 HMM 分析 {symbol} 當前 regime 還會持續多久' },
      { label: '事件型態分析', prompt: '分析 {symbol} 歷史上大漲大跌前的共通特徵' },
    ],
  },
  {
    tab: '完整分析',
    allPrompt: '對 {symbol} 進行完整量化研究 + 因子驗證與監控 + 完整分析三階段',
    items: [
      { label: '完整量化研究', prompt: '對 {symbol} 進行完整量化研究（因子IC + Monte Carlo + Walk Forward + CPCV + GMM/GARCH/HMM）' },
      { label: '因子驗證與監控', prompt: '驗證 {symbol} 當前哪些因子有效、哪些已衰退' },
      { label: '完整分析三階段', prompt: '完整分析一 {symbol}' },
    ],
  },
];

function loadQuickQuestions(): string[] {
  try {
    const raw = localStorage.getItem(QUICK_QUESTIONS_KEY);
    if (raw) {
      const arr = JSON.parse(raw);
      if (Array.isArray(arr) && arr.length > 0) return arr;
    }
  } catch { /* ignore */ }
  return [...DEFAULT_QUICK_QUESTIONS];
}

function saveQuickQuestions(questions: string[]) {
  localStorage.setItem(QUICK_QUESTIONS_KEY, JSON.stringify(questions));
}

interface UsageSummaryData {
  today: { total_tokens: number; estimated_cost_usd: number; request_count: number };
  month: { total_tokens: number; estimated_cost_usd: number; request_count: number };
  all_time: { total_tokens: number; estimated_cost_usd: number; request_count: number };
  note: string;
}

/** 即時耗時顯示：掛載後自動累加秒數 */
function ElapsedTimer({ prefix }: { prefix?: string }) {
  const [elapsed, setElapsed] = useState(0);
  const startRef = useRef(Date.now());
  useEffect(() => {
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - startRef.current) / 1000)), 1000);
    return () => clearInterval(id);
  }, []);
  if (elapsed < 2) return null;
  const min = Math.floor(elapsed / 60);
  const sec = elapsed % 60;
  const timeStr = min > 0 ? `${min}分${sec}秒` : `${sec}秒`;
  return <span className="ml-1 opacity-70">{prefix}{timeStr}</span>;
}

export default function ChatInterface() {
  const [input, setInput] = useState('');
  const [showUsagePanel, setShowUsagePanel] = useState(false);
  const [showHistoryPanel, setShowHistoryPanel] = useState(false);
  const [usageSummary, setUsageSummary] = useState<UsageSummaryData | null>(null);
  const [usageLoading, setUsageLoading] = useState(false);
  const [historyList, setHistoryList] = useState<Array<{id: string; title: string; updated_at: string; message_count: number}>>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [showDistillPanel, setShowDistillPanel] = useState(false);
  const [distillStatus, setDistillStatus] = useState<Record<string, unknown> | null>(null);
  const [distillPreviews, setDistillPreviews] = useState<Array<Record<string, unknown>> | null>(null);
  const [distillProfile, setDistillProfile] = useState<string>('');
  const [distillTokens, setDistillTokens] = useState(0);
  const [distillLoading, setDistillLoading] = useState(false);
  const [distillStep, setDistillStep] = useState<'status' | 'preview' | 'done'>('status');
  const [distillHint, setDistillHint] = useState(false);
  const [quickQuestions, setQuickQuestions] = useState<string[]>(loadQuickQuestions);
  const [editingQuick, setEditingQuick] = useState(false);
  const [activeAnalysisTab, setActiveAnalysisTab] = useState(0);
  const [analysisBarOpen, setAnalysisBarOpen] = useState(true);
  const [showKnowledgePanel, setShowKnowledgePanel] = useState(false);
  const [knowledgeFragments, setKnowledgeFragments] = useState<FragmentItem[]>([]);
  const [knowledgeTotal, setKnowledgeTotal] = useState(0);
  const [knowledgeSymbols, setKnowledgeSymbols] = useState<string[]>([]);
  const [knowledgeTypes, setKnowledgeTypes] = useState<string[]>([]);
  const [knowledgeLoading, setKnowledgeLoading] = useState(false);
  const [editDraft, setEditDraft] = useState<string[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const isComposingRef = useRef(false);
  const streamingMsgIdRef = useRef<string | null>(null);
  const messages = useChartStore((s) => s.messages);
  const addMessage = useChartStore((s) => s.addMessage);
  const updateMessage = useChartStore((s) => s.updateMessage);
  const chatLoading = useChartStore((s) => s.chatLoading);
  const setChatLoading = useChartStore((s) => s.setChatLoading);
  const llmConfig = useChartStore((s) => s.llmConfig);
  const getChartStateSummary = useChartStore((s) => s.getChartStateSummary);
  const conversationId = useChartStore((s) => s.conversationId);
  const pendingChatMessage = useChartStore((s) => s.pendingChatMessage);
  const setPendingChatMessage = useChartStore((s) => s.setPendingChatMessage);

  // 外部注入訊息（例如因子掃描→一鍵回測）
  useEffect(() => {
    if (pendingChatMessage && !chatLoading) {
      setInput(pendingChatMessage);
      setPendingChatMessage(null);
      setTimeout(() => {
        const btn = document.getElementById('chat-send-btn');
        btn?.click();
      }, 100);
    }
  }, [pendingChatMessage, chatLoading, setPendingChatMessage]);

  // 載入歷史對話列表
  const loadHistory = useCallback(async () => {
    setHistoryLoading(true);
    try {
      const data = await fetchChatHistory(15);
      setHistoryList(data.conversations || []);
    } catch {
      toast('載入對話歷史失敗', 'warning');
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  // 載入知識碎片
  const loadKnowledge = useCallback(async (params?: {
    symbol?: string; fragment_type?: string; sort_by?: string;
  }) => {
    setKnowledgeLoading(true);
    try {
      const data = await fetchFragments({ limit: 200, ...params });
      setKnowledgeFragments(data.fragments || []);
      setKnowledgeTotal(data.total || 0);
      setKnowledgeSymbols(data.symbols || []);
      setKnowledgeTypes(data.types || []);
    } catch { toast('載入知識碎片失敗', 'warning'); }
    finally { setKnowledgeLoading(false); }
  }, []);

  const handleDeleteFragment = useCallback(async (id: number) => {
    try {
      await deleteFragment(id);
      setKnowledgeFragments(prev => prev.filter(f => f.id !== id));
      setKnowledgeTotal(prev => Math.max(0, prev - 1));
    } catch { toast('刪除知識碎片失敗', 'error'); }
  }, []);

  // 恢復歷史對話
  const restoreConversation = useCallback(async (convId: string) => {
    try {
      const data = await fetchConversationMessages(convId);
      if (data.messages && data.messages.length > 0) {
        // 清除當前對話，載入歷史
        useChartStore.getState().clearMessages();
        const store = useChartStore.getState();
        for (const msg of data.messages) {
          store.addMessage({
            id: `${msg.role}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
            role: msg.role as 'user' | 'assistant',
            content: msg.content,
            timestamp: msg.timestamp,
            usage: msg.usage,
          });
        }
        // 設定 conversation_id（讓後續對話延續）
        useChartStore.setState({ conversationId: convId });
        setShowHistoryPanel(false);
      }
    } catch {
      toast('恢復對話失敗', 'error');
    }
  }, []);

  // 載入蒸餾狀態
  const loadDistillStatus = useCallback(async () => {
    setDistillLoading(true);
    try {
      const data = await fetchDistillStatus();
      setDistillStatus(data);
      setDistillStep('status');
    } catch { /* 靜默 */ }
    finally { setDistillLoading(false); }
  }, []);

  // 執行蒸餾預覽
  const runDistillPreview = useCallback(async () => {
    if (!llmConfig.sessionId) return;
    setDistillLoading(true);
    setDistillStep('preview');
    try {
      const data = await previewDistill(
        llmConfig.provider as LLMProvider,
        llmConfig.sessionId,
      );
      if (data.status === 'ok') {
        setDistillPreviews(data.previews || []);
        setDistillProfile(data.profile_preview || '');
        setDistillTokens(data.total_tokens_used || 0);
      } else {
        setDistillPreviews(null);
        setDistillStep('status');
      }
    } catch { setDistillStep('status'); }
    finally { setDistillLoading(false); }
  }, [llmConfig.sessionId, llmConfig.provider]);

  // 確認蒸餾
  const runDistillConfirm = useCallback(async () => {
    if (!distillPreviews || !llmConfig.sessionId) return;
    setDistillLoading(true);
    try {
      await confirmDistill(
        distillPreviews,
        distillProfile,
        distillTokens,
        llmConfig.provider as LLMProvider,
        llmConfig.sessionId,
      );
      setDistillStep('done');
    } catch { /* 靜默 */ }
    finally { setDistillLoading(false); }
  }, [distillPreviews, distillProfile, distillTokens, llmConfig]);

  // 載入用量摘要
  const loadUsageSummary = useCallback(async () => {
    if (!llmConfig.sessionId) return;
    setUsageLoading(true);
    try {
      const data = await fetchUsageSummary(
        llmConfig.provider as LLMProvider,
        llmConfig.sessionId,
      );
      setUsageSummary(data);
    } catch {
      // 靜默失敗，不影響主流程
    } finally {
      setUsageLoading(false);
    }
  }, [llmConfig.sessionId, llmConfig.provider]);

  // 自動捲動到最新訊息
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = useCallback(async (sendMode?: string) => {
    const trimmed = input.trim();
    if (!trimmed || chatLoading) return;

    // 新增使用者訊息
    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: trimmed,
      timestamp: new Date().toISOString(),
    };
    addMessage(userMsg);
    setInput('');
    setChatLoading(true);

    // 建立空的助手訊息（等待串流填入）
    const assistantMsgId = `assistant-${Date.now()}`;
    streamingMsgIdRef.current = assistantMsgId;
    const assistantMsg: ChatMessage = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString(),
      isThinking: true,
      isStreaming: true,
    };
    addMessage(assistantMsg);

    let accumulatedText = '';
    let doneReceived = false;

    // ★ 建立對話歷史：只送出「最後一次切換標的」之後的對話，避免不同標的混淆
    const currentMessages = useChartStore.getState().messages;
    const lastDividerIdx = currentMessages.reduce(
      (acc, m, i) => (m.content?.startsWith('__SYMBOL_SWITCH__') ? i : acc), -1
    );
    const relevantMessages = lastDividerIdx >= 0
      ? currentMessages.slice(lastDividerIdx + 1)
      : currentMessages;
    const chatHistory = relevantMessages
      .filter((m) => (m.role === 'user' || m.role === 'assistant') && m.content.trim() && m.id !== assistantMsgId)
      .slice(-20)
      .map((m) => ({ role: m.role, content: m.content }));

    // 截取圖表畫面（供 LLM 視覺分析）
    let screenshot: string | undefined;
    const captureFn = useChartStore.getState().captureScreenshot;
    if (captureFn) {
      try {
        const img = await captureFn();
        if (img) screenshot = img;
      } catch { /* 截圖失敗不阻塞發送 */ }
    }

    await streamChatMessage(
      trimmed,
      {
        onThinking: () => {
          updateMessage(assistantMsgId, { isThinking: true, isStreaming: true, statusText: '思考中' });
        },

        onStatus: (statusMessage: string) => {
          updateMessage(assistantMsgId, { isStreaming: true, statusText: statusMessage });
        },

        onProgress: (prog) => {
          updateMessage(assistantMsgId, {
            isStreaming: true,
            statusText: prog.message,
            progress: {
              percentage: prog.percentage,
              completed: prog.completed,
              total: prog.total,
              current_task: prog.current_task,
            },
          });
        },

        onToken: (content: string) => {
          accumulatedText += content;
          updateMessage(assistantMsgId, {
            content: accumulatedText,
            isThinking: false,
            isStreaming: true,
            statusText: undefined,
            progress: undefined,
          });
        },

        onFunctionCalls: (calls) => {
          updateMessage(assistantMsgId, {
            functionCalls: calls.map((c) => ({
              name: c.name,
              arguments: c.arguments,
            })),
          });
        },

        onChartUpdates: (updates) => {
          // 使用 setTimeout 將 store 更新排到 SSE 事件處理之外，
          // 避免在 streaming 回調中直接觸發大量 React 重新渲染導致 chart 崩潰
          setTimeout(() => {
            try {
              const store = useChartStore.getState();

              // 1. 切換幣種 / 時間週期 / 日期範圍
              //    只更新 state，App.tsx 的 useEffect 會統一觸發 loadChartData
              const newSymbol = updates.symbol as string | undefined;
              const newTimeframe = updates.timeframe as string | undefined;

              if (newSymbol && newSymbol !== store.symbol) store.setSymbol(newSymbol);
              if (newTimeframe && newTimeframe !== store.timeframe) store.setTimeframe(newTimeframe as Timeframe);
              // ★ 不再從 LLM chart_updates 套用日期範圍，避免 LLM 的查詢範圍
              //   覆蓋圖表顯示，導致只顯示部分數據。日期範圍僅由使用者手動操作。

              // 2. 指標操作（add / remove / update）
              const indicatorActions = updates.indicator_actions as Array<{
                action: string;
                indicator_id: string;
                indicator_name?: string;
                parameters?: Record<string, number>;
                display_mode?: string;
              }> | undefined;

              if (indicatorActions) {
                for (const ia of indicatorActions) {
                  if (ia.action === 'remove') {
                    store.removeIndicator(ia.indicator_id);
                  } else if (ia.action === 'add' || ia.action === 'update') {
                    const existing = store.activeIndicators.find(i => i.id === ia.indicator_id);
                    if (existing && ia.action === 'update' && ia.parameters) {
                      store.updateIndicatorParams(ia.indicator_id, ia.parameters);
                    } else {
                      const newInd = {
                        id: ia.indicator_id,
                        indicator_type: ia.indicator_id.toUpperCase(),
                        parameters: ia.parameters || {},
                        display_mode: (ia.display_mode || 'sub_chart') as 'overlay' | 'sub_chart' | 'info_panel',
                        visible: true,
                      };
                      store.addIndicator(newInd);
                    }
                    // 指標 data 載入由 App.tsx 的 useEffect 自動處理
                  }
                }
              }

              // 3. 圖表標記（annotations）
              const annotations = updates.annotations as Array<{
                type: string;
                time?: string;
                startTime?: string;
                endTime?: string;
                price?: number;
                endPrice?: number;
                text?: string;
                color?: string;
                lineWidth?: number;
                lineStyle?: number;
                groupId?: string;
                groupName?: string;
              }> | undefined;

              if (annotations && annotations.length > 0) {
                for (const ann of annotations) {
                  const annotation: Annotation = {
                    id: `ann-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
                    type: ann.type as Annotation['type'],
                    startTime: ann.startTime || ann.time,
                    endTime: ann.endTime,
                    price: ann.price,
                    endPrice: ann.endPrice,
                    text: ann.text,
                    color: ann.color || '#58a6ff',
                    lineWidth: ann.lineWidth,
                    lineStyle: ann.lineStyle,
                    groupId: ann.groupId,
                    groupName: ann.groupName,
                  };
                  store.addAnnotation(annotation);
                }
                const groupNames = [...new Set(annotations.map(a => a.groupName).filter(Boolean))];
                const label = groupNames.length > 0 ? groupNames.join('、') : `${annotations.length} 筆標記`;
                updateMessage(assistantMsgId, {
                  statusText: `已繪製：${label}`,
                });
              }
            } catch (err) {
              console.error('[onChartUpdates] 處理圖表更新時發生錯誤:', err);
            }
          }, 0);
        },

        onUsage: (usage: TokenUsage) => {
          updateMessage(assistantMsgId, { usage });
        },

        onError: (error: string) => {
          if (!accumulatedText) {
            // 如果還沒有任何文字，直接顯示錯誤
            updateMessage(assistantMsgId, {
              content: `錯誤：${error}`,
              role: 'system',
              isThinking: false,
              isStreaming: false,
            });
          } else {
            // 已有文字，追加錯誤
            accumulatedText += `\n\n⚠️ ${error}`;
            updateMessage(assistantMsgId, {
              content: accumulatedText,
              isStreaming: false,
            });
          }
        },

        onDone: (_convId?: string, hints?: Record<string, unknown>) => {
          doneReceived = true;
          updateMessage(assistantMsgId, {
            isThinking: false,
            isStreaming: false,
            statusText: undefined,
          });
          setChatLoading(false);
          streamingMsgIdRef.current = null;
          if (hints?.distill_hint) {
            setDistillHint(true);
          }
        },
      },
      conversationId || undefined,
      getChartStateSummary(),
      llmConfig.provider,
      llmConfig.sessionId,
      chatHistory,
      sendMode,
      screenshot,
    );

    // ★ 安全兜底：如果串流結束但沒收到 done 事件（連線中斷等），也要結束載入狀態
    if (!doneReceived) {
      updateMessage(assistantMsgId, {
        isThinking: false,
        isStreaming: false,
        statusText: undefined,
      });
      streamingMsgIdRef.current = null;
    }
    setChatLoading(false);
  }, [input, chatLoading, addMessage, updateMessage, setChatLoading, llmConfig, getChartStateSummary, conversationId]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.nativeEvent.isComposing || isComposingRef.current) return;
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* 標題列 */}
      <div
        className="px-4 py-3 border-b flex items-center justify-between"
        style={{ borderColor: 'var(--border-color)' }}
      >
        <h2 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
          AI 分析助手
        </h2>
        <div className="flex items-center gap-3">
          <button
            onClick={() => {
              setShowDistillPanel(!showDistillPanel);
              setShowHistoryPanel(false);
              setShowUsagePanel(false);
              setShowKnowledgePanel(false);
              if (!showDistillPanel) loadDistillStatus();
            }}
            className="text-xs cursor-pointer hover:opacity-80"
            style={{ color: 'var(--accent-blue)' }}
            title="整理知識（蒸餾歷史對話為精華）"
          >
            🧠 整理
          </button>
          <button
            onClick={() => {
              setShowHistoryPanel(!showHistoryPanel);
              setShowUsagePanel(false);
              setShowDistillPanel(false);
              setShowKnowledgePanel(false);
              if (!showHistoryPanel) loadHistory();
            }}
            className="text-xs cursor-pointer hover:opacity-80"
            style={{ color: 'var(--accent-blue)' }}
            title="歷史對話記錄"
          >
            📋 歷史
          </button>
          <button
            onClick={() => {
              setShowKnowledgePanel(!showKnowledgePanel);
              setShowUsagePanel(false);
              setShowHistoryPanel(false);
              setShowDistillPanel(false);
              if (!showKnowledgePanel) loadKnowledge();
            }}
            className="text-xs cursor-pointer hover:opacity-80"
            style={{ color: 'var(--accent-blue)' }}
            title="瀏覽系統累積的知識碎片"
          >
            📚 知識庫
          </button>
          <button
            onClick={() => {
              setShowUsagePanel(!showUsagePanel);
              setShowHistoryPanel(false);
              setShowDistillPanel(false);
              setShowKnowledgePanel(false);
              if (!showUsagePanel) loadUsageSummary();
            }}
            className="text-xs cursor-pointer hover:opacity-80"
            style={{ color: 'var(--accent-blue)' }}
            title="查看 Token 用量統計"
          >
            📊 用量
          </button>
          <button
            onClick={() => useChartStore.getState().clearMessages()}
            className="text-xs cursor-pointer hover:opacity-80"
            style={{ color: 'var(--text-secondary)' }}
          >
            清除對話
          </button>
        </div>
      </div>

      {/* 知識蒸餾面板 */}
      {showDistillPanel && (
        <DistillPanel
          status={distillStatus}
          previews={distillPreviews}
          profile={distillProfile}
          step={distillStep}
          loading={distillLoading}
          tokensUsed={distillTokens}
          onPreview={runDistillPreview}
          onConfirm={runDistillConfirm}
          onClose={() => { setShowDistillPanel(false); setDistillStep('status'); setDistillPreviews(null); }}
        />
      )}

      {/* 歷史對話面板 */}
      {showHistoryPanel && (
        <HistoryPanel
          conversations={historyList}
          loading={historyLoading}
          onSelect={restoreConversation}
          onRefresh={loadHistory}
        />
      )}

      {/* 知識碎片瀏覽器 */}
      {showKnowledgePanel && (
        <KnowledgePanel
          fragments={knowledgeFragments}
          total={knowledgeTotal}
          symbols={knowledgeSymbols}
          types={knowledgeTypes}
          loading={knowledgeLoading}
          onRefresh={loadKnowledge}
          onDelete={handleDeleteFragment}
        />
      )}

      {/* Token 用量統計面板 */}
      {showUsagePanel && (
        <UsageSummaryPanel
          summary={usageSummary}
          loading={usageLoading}
          onRefresh={loadUsageSummary}
        />
      )}

      {/* 分析功能快捷面板（固定在訊息列表上方） */}
      <div style={{ borderBottom: '1px solid var(--border-primary)', flexShrink: 0 }}>
        <div
          className="flex items-center justify-between px-3 py-1 cursor-pointer"
          style={{ background: 'var(--bg-secondary)' }}
          onClick={() => setAnalysisBarOpen((v) => !v)}
        >
          <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
            {analysisBarOpen ? '▼' : '▶'} 分析功能
          </span>
          <div className="flex gap-1">
            {ANALYSIS_CATEGORIES.map((cat, i) => (
              <button
                key={i}
                onClick={(e) => { e.stopPropagation(); setActiveAnalysisTab(i); if (!analysisBarOpen) setAnalysisBarOpen(true); }}
                className="px-2 py-0.5 rounded text-xs cursor-pointer transition-colors"
                style={{
                  background: activeAnalysisTab === i ? 'var(--accent-blue)' : 'transparent',
                  color: activeAnalysisTab === i ? '#fff' : 'var(--text-secondary)',
                }}
              >
                {cat.tab}
              </button>
            ))}
          </div>
        </div>
        {analysisBarOpen && (
          <div className="flex flex-wrap gap-1.5 px-3 py-2" style={{ background: 'var(--bg-secondary)' }}>
            <button
              onClick={() => {
                const sym = useChartStore.getState().symbol || 'BTC/USDT';
                const cat = ANALYSIS_CATEGORIES[activeAnalysisTab];
                const prompt = cat.allPrompt.replace('{symbol}', sym);
                setInput(prompt);
                // 有指定 mode 的分析類型，直接帶 mode 發送
                const catMode = (cat as any).mode;
                if (catMode) {
                  setTimeout(() => handleSend(catMode), 50);
                }
              }}
              className="px-2.5 py-1 rounded text-xs cursor-pointer transition-colors font-medium"
              style={{ background: 'var(--accent-blue)', color: '#fff' }}
            >
              全部分析
            </button>
            {ANALYSIS_CATEGORIES[activeAnalysisTab].items.map((item, i) => (
              <button
                key={i}
                onClick={() => {
                  const sym = useChartStore.getState().symbol || 'BTC/USDT';
                  setInput(item.prompt.replace('{symbol}', sym));
                }}
                className="px-2.5 py-1 rounded text-xs cursor-pointer transition-colors"
                style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
              >
                {item.label}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* 訊息列表 */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.length === 0 && (
          <div className="text-center py-8" style={{ color: 'var(--text-secondary)' }}>
            <p className="text-sm mb-4">歡迎使用阿斯拉量化系統</p>

            {!editingQuick ? (
              <>
                <div className="flex items-center justify-center gap-2 mb-2">
                  <p className="text-xs">你可以嘗試以下問題：</p>
                  <button
                    onClick={() => { setEditDraft([...quickQuestions]); setEditingQuick(true); }}
                    className="text-xs cursor-pointer hover:opacity-80 px-1.5 py-0.5 rounded"
                    style={{ color: 'var(--accent-blue)', background: 'var(--bg-tertiary)' }}
                    title="自訂快捷問題"
                  >
                    ✏️ 編輯
                  </button>
                </div>
                <div className="space-y-2">
                  {quickQuestions.map((q, i) => (
                    <button
                      key={i}
                      onClick={() => setInput(q)}
                      className="block w-full text-left px-3 py-2 rounded text-xs cursor-pointer transition-colors"
                      style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
                    >
                      {q}
                    </button>
                  ))}
                </div>

              </>
            ) : (
              <div className="text-left mx-auto" style={{ maxWidth: '400px' }}>
                <p className="text-xs mb-2 text-center">自訂快捷問題（可新增、修改、刪除）</p>
                <div className="space-y-2">
                  {editDraft.map((q, i) => (
                    <div key={i} className="flex items-center gap-1">
                      <input
                        type="text"
                        value={q}
                        onChange={(e) => {
                          const next = [...editDraft];
                          next[i] = e.target.value;
                          setEditDraft(next);
                        }}
                        className="flex-1 px-2 py-1.5 rounded text-xs outline-none"
                        style={{
                          background: 'var(--bg-primary)',
                          color: 'var(--text-primary)',
                          border: '1px solid var(--border-primary)',
                        }}
                      />
                      <button
                        onClick={() => setEditDraft(editDraft.filter((_, j) => j !== i))}
                        className="text-xs cursor-pointer hover:opacity-80 px-1.5 py-1 rounded flex-shrink-0"
                        style={{ color: '#f85149', background: 'rgba(248,81,73,0.12)' }}
                        title="刪除此項"
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                </div>
                <button
                  onClick={() => setEditDraft([...editDraft, ''])}
                  className="w-full mt-2 text-xs cursor-pointer hover:opacity-80 px-3 py-1.5 rounded"
                  style={{ color: 'var(--accent-blue)', background: 'var(--bg-tertiary)', border: '1px dashed var(--border-primary)' }}
                >
                  + 新增問題
                </button>
                <div className="flex gap-2 mt-3 justify-center">
                  <button
                    onClick={() => {
                      const cleaned = editDraft.map((s) => s.trim()).filter(Boolean);
                      if (cleaned.length === 0) return;
                      setQuickQuestions(cleaned);
                      saveQuickQuestions(cleaned);
                      setEditingQuick(false);
                    }}
                    className="text-xs cursor-pointer hover:opacity-80 px-4 py-1.5 rounded"
                    style={{ background: 'var(--accent-blue)', color: '#fff' }}
                  >
                    儲存
                  </button>
                  <button
                    onClick={() => setEditingQuick(false)}
                    className="text-xs cursor-pointer hover:opacity-80 px-4 py-1.5 rounded"
                    style={{ background: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}
                  >
                    取消
                  </button>
                  <button
                    onClick={() => setEditDraft([...DEFAULT_QUICK_QUESTIONS])}
                    className="text-xs cursor-pointer hover:opacity-80 px-4 py-1.5 rounded"
                    style={{ background: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}
                    title="恢復為預設問題"
                  >
                    重置
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {messages.map((msg) => {
          if (msg.content?.startsWith('__SYMBOL_SWITCH__')) {
            const newSymbol = msg.content.replace('__SYMBOL_SWITCH__', '');
            return (
              <div key={msg.id} className="flex items-center gap-2 my-3 px-2">
                <div className="flex-1 h-px" style={{ background: 'var(--border-primary)' }} />
                <span className="text-xs px-2 py-0.5 rounded whitespace-nowrap" style={{ color: 'var(--text-secondary)', background: 'var(--bg-tertiary)' }}>
                  已切換到 {newSymbol}
                </span>
                <div className="flex-1 h-px" style={{ background: 'var(--border-primary)' }} />
              </div>
            );
          }
          return (
          <div key={msg.id}>
            <div className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div
                className="max-w-[85%] px-3 py-2 rounded-lg text-sm whitespace-pre-wrap"
                style={{
                  background:
                    msg.role === 'user'
                      ? 'var(--accent-blue)'
                      : msg.role === 'system'
                      ? 'rgba(248, 81, 73, 0.15)'
                      : 'var(--bg-tertiary)',
                  color:
                    msg.role === 'user'
                      ? '#fff'
                      : msg.role === 'system'
                      ? 'var(--accent-red)'
                      : 'var(--text-primary)',
                }}
              >
                {/* 思考中動畫（尚無內容） */}
                {msg.isThinking && !msg.content && (
                  <span className="flex items-center gap-1.5">
                    <span className="flex gap-0.5">
                      <span className="w-1.5 h-1.5 rounded-full animate-bounce" style={{ background: 'var(--text-secondary)', animationDelay: '0ms' }} />
                      <span className="w-1.5 h-1.5 rounded-full animate-bounce" style={{ background: 'var(--text-secondary)', animationDelay: '150ms' }} />
                      <span className="w-1.5 h-1.5 rounded-full animate-bounce" style={{ background: 'var(--text-secondary)', animationDelay: '300ms' }} />
                    </span>
                    <span style={{ color: 'var(--text-secondary)', transition: 'opacity 0.3s ease' }}>
                      {msg.statusText || '思考中'}
                      <ElapsedTimer prefix="  已等待 " />
                    </span>
                  </span>
                )}

                {/* 訊息內容 */}
                {msg.content}

                {/* 串流中的游標（正在輸出文字時） */}
                {msg.isStreaming && msg.content && !msg.statusText && (
                  <span className="inline-block w-0.5 h-4 ml-0.5 animate-pulse" style={{ background: 'var(--accent-blue)' }} />
                )}

                {/* 進度狀態列：有內容且仍在串流、有 statusText 時顯示 */}
                {msg.isStreaming && msg.content && msg.statusText && (
                  <div
                    className="mt-2 px-2.5 py-1.5 rounded"
                    style={{
                      background: 'rgba(88, 166, 255, 0.08)',
                      border: '1px solid rgba(88, 166, 255, 0.2)',
                    }}
                  >
                    <div className="flex items-center gap-2">
                      <span className="flex gap-0.5 flex-shrink-0">
                        <span className="w-1 h-1 rounded-full animate-bounce" style={{ background: 'var(--accent-blue)', animationDelay: '0ms' }} />
                        <span className="w-1 h-1 rounded-full animate-bounce" style={{ background: 'var(--accent-blue)', animationDelay: '150ms' }} />
                        <span className="w-1 h-1 rounded-full animate-bounce" style={{ background: 'var(--accent-blue)', animationDelay: '300ms' }} />
                      </span>
                      <span className="text-xs flex-1" style={{ color: 'var(--accent-blue)' }}>
                        {msg.statusText}
                        <ElapsedTimer />
                      </span>
                      {msg.progress && (
                        <span className="text-xs font-mono font-semibold" style={{ color: 'var(--accent-blue)' }}>
                          {msg.progress.percentage}%
                        </span>
                      )}
                    </div>
                    {/* 進度條 */}
                    {msg.progress && msg.progress.total > 0 && (
                      <div
                        className="mt-1.5 rounded-full overflow-hidden"
                        style={{ height: 4, background: 'rgba(88, 166, 255, 0.15)' }}
                      >
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${msg.progress.percentage}%`,
                            background: 'var(--accent-blue)',
                            transition: 'width 0.5s ease-out',
                          }}
                        />
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Token 用量（助手訊息下方顯示） */}
            {msg.role === 'assistant' && msg.usage && !msg.isStreaming && (
              <TokenUsageBadge usage={msg.usage} />
            )}
          </div>
        );
        })}

        <div ref={messagesEndRef} />
      </div>

      {/* 蒸餾提醒條 */}
      {distillHint && (
        <div
          className="px-4 py-2 flex items-center justify-between text-xs"
          style={{ background: 'rgba(56,139,253,0.08)', borderTop: '1px solid var(--accent-blue)' }}
        >
          <span style={{ color: 'var(--accent-blue)' }}>
            💡 已累積超過 30 天對話，建議點擊「🧠 整理」來壓縮知識，讓 AI 更聰明
          </span>
          <button
            onClick={() => setDistillHint(false)}
            className="cursor-pointer hover:opacity-60"
            style={{ color: 'var(--text-secondary)' }}
          >
            ✕
          </button>
        </div>
      )}

      {/* 輸入區域 */}
      <div
        className="p-3 border-t"
        style={{ borderColor: 'var(--border-color)' }}
      >
        <div
          className="flex items-end gap-2 rounded-lg p-2"
          style={{ background: 'var(--bg-tertiary)' }}
        >
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            onCompositionStart={() => { isComposingRef.current = true; }}
            onCompositionEnd={() => { isComposingRef.current = false; }}
            placeholder="輸入你的分析需求..."
            rows={1}
            className="flex-1 resize-none border-none outline-none text-sm bg-transparent"
            style={{ color: 'var(--text-primary)', minHeight: '36px', maxHeight: '120px' }}
          />
          <button
            id="chat-send-btn"
            onClick={() => handleSend()}
            disabled={chatLoading || !input.trim()}
            className="px-4 py-1.5 rounded text-sm font-medium cursor-pointer transition-opacity disabled:opacity-40"
            style={{
              background: 'var(--accent-blue)',
              color: '#fff',
            }}
          >
            發送
          </button>
          <button
            onClick={() => handleSend('quant_research')}
            disabled={chatLoading || !input.trim()}
            className="px-3 py-1.5 rounded text-sm font-medium cursor-pointer transition-opacity disabled:opacity-40"
            title="將你的策略描述送出進行量化回測驗證"
            style={{
              background: 'var(--accent-green, #22c55e)',
              color: '#fff',
            }}
          >
            📊 回測分析
          </button>
          <button
            onClick={() => handleSend('calibrate')}
            disabled={chatLoading}
            className="px-3 py-1.5 rounded text-sm font-medium cursor-pointer transition-opacity disabled:opacity-40"
            title="掃描歷史數據，校準此標的的最佳指標參數"
            style={{
              background: 'var(--accent-orange, #f0883e)',
              color: '#fff',
            }}
          >
            🎯 校準指標
          </button>
        </div>
        <p className="text-xs mt-1 px-1" style={{ color: 'var(--text-secondary)' }}>
          Enter 發送，Shift+Enter 換行｜「回測分析」量化驗證策略｜「校準指標」找出最佳參數
        </p>
      </div>
    </div>
  );
}


// ─── 知識蒸餾面板 ─────────────────────────────

function DistillPanel({
  status,
  previews,
  profile,
  step,
  loading,
  tokensUsed,
  onPreview,
  onConfirm,
  onClose,
}: {
  status: Record<string, unknown> | null;
  previews: Array<Record<string, unknown>> | null;
  profile: string;
  step: 'status' | 'preview' | 'done';
  loading: boolean;
  tokensUsed: number;
  onPreview: () => void;
  onConfirm: () => void;
  onClose: () => void;
}) {
  const formatNum = (n: unknown) => {
    const num = Number(n) || 0;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return String(num);
  };

  return (
    <div
      className="border-b overflow-y-auto"
      style={{ borderColor: 'var(--border-color)', background: 'var(--bg-secondary)', maxHeight: '320px' }}
    >
      <div className="px-4 py-2 flex items-center justify-between sticky top-0" style={{ background: 'var(--bg-secondary)' }}>
        <span className="text-xs font-medium" style={{ color: 'var(--text-primary)' }}>
          🧠 知識整理（蒸餾）
        </span>
        <button onClick={onClose} className="text-xs cursor-pointer hover:opacity-80" style={{ color: 'var(--text-secondary)' }}>
          ✕ 關閉
        </button>
      </div>

      <div className="px-4 pb-3 text-xs">
        {/* 步驟一：狀態 */}
        {step === 'status' && (
          <>
            {!status ? (
              <p style={{ color: 'var(--text-secondary)' }}>{loading ? '正在載入...' : '無法取得蒸餾狀態'}</p>
            ) : (
              <>
                <div className="grid grid-cols-2 gap-2 mb-2">
                  <div className="p-2 rounded" style={{ background: 'var(--bg-tertiary)' }}>
                    <div style={{ color: 'var(--text-secondary)' }}>待整理訊息</div>
                    <div className="font-mono font-medium" style={{ color: 'var(--text-primary)' }}>
                      {formatNum(status.undistilled_messages)} 則
                    </div>
                  </div>
                  <div className="p-2 rounded" style={{ background: 'var(--bg-tertiary)' }}>
                    <div style={{ color: 'var(--text-secondary)' }}>累積天數</div>
                    <div className="font-mono font-medium" style={{ color: 'var(--text-primary)' }}>
                      {String(status.undistilled_days || 0)} 天
                    </div>
                  </div>
                  <div className="p-2 rounded" style={{ background: 'var(--bg-tertiary)' }}>
                    <div style={{ color: 'var(--text-secondary)' }}>預估消耗</div>
                    <div className="font-mono font-medium" style={{ color: 'var(--text-primary)' }}>
                      ~{formatNum(status.estimated_tokens)} tokens
                    </div>
                  </div>
                  <div className="p-2 rounded" style={{ background: 'var(--bg-tertiary)' }}>
                    <div style={{ color: 'var(--text-secondary)' }}>已有知識</div>
                    <div className="font-mono font-medium" style={{ color: 'var(--text-primary)' }}>
                      {String(status.existing_knowledge || 0)} 筆
                    </div>
                  </div>
                </div>

                {status.should_distill ? (
                  <div className="p-2 rounded mb-2" style={{ background: 'rgba(56,139,253,0.1)', border: '1px solid var(--accent-blue)' }}>
                    <p style={{ color: 'var(--accent-blue)' }}>
                      已累積超過 30 天的對話，建議進行知識整理以節省儲存空間並提升 AI 回答品質。
                    </p>
                  </div>
                ) : (
                  <p className="mb-2" style={{ color: 'var(--text-secondary)' }}>
                    累積未滿 30 天，但你仍可手動執行整理。
                  </p>
                )}

                <button
                  onClick={onPreview}
                  disabled={loading || Number(status.undistilled_messages) < 4}
                  className="w-full py-2 rounded text-xs font-medium cursor-pointer transition-opacity disabled:opacity-40"
                  style={{ background: 'var(--accent-blue)', color: '#fff' }}
                >
                  {loading ? '正在分析中...' : '開始整理（先預覽）'}
                </button>

                {Number(status.undistilled_messages) < 4 && (
                  <p className="mt-1" style={{ color: 'var(--text-secondary)', fontSize: '10px' }}>
                    訊息數不足，至少需要 4 則以上的對話才能整理
                  </p>
                )}
              </>
            )}
          </>
        )}

        {/* 步驟二：預覽 */}
        {step === 'preview' && (
          <>
            {loading ? (
              <div className="py-4 text-center" style={{ color: 'var(--text-secondary)' }}>
                <p className="mb-1">🧠 AI 正在整理你的歷史對話...</p>
                <p style={{ fontSize: '10px' }}>這可能需要 30-60 秒，請耐心等候</p>
              </div>
            ) : previews && previews.length > 0 ? (
              <>
                <p className="mb-2 font-medium" style={{ color: 'var(--text-primary)' }}>
                  預覽結果（請確認後再儲存）
                </p>

                {previews.map((p, i) => (
                  <div key={i} className="mb-2 p-2 rounded" style={{ background: 'var(--bg-tertiary)' }}>
                    <div className="flex justify-between mb-1">
                      <span className="font-medium" style={{ color: 'var(--accent-blue)' }}>
                        {String(p.symbol === '_general' ? '一般問題' : p.symbol)}
                      </span>
                      <span style={{ color: 'var(--text-secondary)', fontSize: '10px' }}>
                        {String(p.source_count || 0)} 組 Q&A → {formatNum(p.distilled_chars)} 字
                      </span>
                    </div>
                    <p className="whitespace-pre-wrap" style={{ color: 'var(--text-primary)', fontSize: '11px', maxHeight: '120px', overflow: 'auto' }}>
                      {String(p.summary || p.error || '無內容')}
                    </p>
                  </div>
                ))}

                {profile && (
                  <div className="mb-2 p-2 rounded" style={{ background: 'var(--bg-tertiary)' }}>
                    <span className="font-medium" style={{ color: 'var(--accent-blue)' }}>
                      你的分析風格
                    </span>
                    <p className="mt-1 whitespace-pre-wrap" style={{ color: 'var(--text-primary)', fontSize: '11px' }}>
                      {profile}
                    </p>
                  </div>
                )}

                <p className="mb-2" style={{ color: 'var(--text-secondary)', fontSize: '10px' }}>
                  本次整理消耗 {formatNum(tokensUsed)} tokens。確認儲存後，AI 未來將自動參考這些知識。
                </p>

                <div className="flex gap-2">
                  <button
                    onClick={onConfirm}
                    disabled={loading}
                    className="flex-1 py-2 rounded text-xs font-medium cursor-pointer disabled:opacity-40"
                    style={{ background: 'var(--accent-green)', color: '#fff' }}
                  >
                    確認儲存
                  </button>
                  <button
                    onClick={onClose}
                    className="flex-1 py-2 rounded text-xs cursor-pointer"
                    style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
                  >
                    取消
                  </button>
                </div>
              </>
            ) : (
              <p style={{ color: 'var(--text-secondary)' }}>沒有足夠的對話內容可以整理。</p>
            )}
          </>
        )}

        {/* 步驟三：完成 */}
        {step === 'done' && (
          <div className="py-3 text-center">
            <p className="mb-2 font-medium" style={{ color: 'var(--accent-green)' }}>
              ✅ 知識整理完成！
            </p>
            <p style={{ color: 'var(--text-secondary)' }}>
              AI 未來回答時將自動參考這些蒸餾後的知識，系統會越用越聰明。
            </p>
            <button
              onClick={onClose}
              className="mt-2 px-4 py-1.5 rounded text-xs cursor-pointer"
              style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
            >
              關閉
            </button>
          </div>
        )}
      </div>
    </div>
  );
}


// ─── Token 用量顯示組件 ──────────────────────────

// ─── 歷史對話面板 ─────────────────────────────

function HistoryPanel({
  conversations,
  loading,
  onSelect,
  onRefresh,
}: {
  conversations: Array<{ id: string; title: string; updated_at: string; message_count: number }>;
  loading: boolean;
  onSelect: (id: string) => void;
  onRefresh: () => void;
}) {
  const formatDate = (iso: string) => {
    try {
      const d = new Date(iso + 'Z');
      return d.toLocaleDateString('zh-TW', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
    } catch {
      return iso.slice(0, 10);
    }
  };

  return (
    <div
      className="border-b overflow-y-auto"
      style={{ borderColor: 'var(--border-color)', background: 'var(--bg-secondary)', maxHeight: '200px' }}
    >
      <div className="px-4 py-2 flex items-center justify-between sticky top-0" style={{ background: 'var(--bg-secondary)' }}>
        <span className="text-xs font-medium" style={{ color: 'var(--text-primary)' }}>
          歷史對話
        </span>
        <button
          onClick={onRefresh}
          disabled={loading}
          className="text-xs cursor-pointer hover:opacity-80 disabled:opacity-40"
          style={{ color: 'var(--accent-blue)' }}
        >
          {loading ? '載入中...' : '🔄'}
        </button>
      </div>

      {conversations.length === 0 ? (
        <p className="px-4 py-3 text-xs" style={{ color: 'var(--text-secondary)' }}>
          {loading ? '正在載入...' : '尚無歷史對話'}
        </p>
      ) : (
        <div className="px-2 pb-2">
          {conversations.map((conv) => (
            <button
              key={conv.id}
              onClick={() => onSelect(conv.id)}
              className="w-full text-left px-3 py-2 rounded text-xs cursor-pointer transition-colors mb-0.5 hover:opacity-80"
              style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
            >
              <div className="flex justify-between items-center">
                <span className="truncate flex-1 mr-2">{conv.title}</span>
                <span className="shrink-0" style={{ color: 'var(--text-secondary)', fontSize: '10px' }}>
                  {formatDate(conv.updated_at)}
                </span>
              </div>
              <div style={{ color: 'var(--text-secondary)', fontSize: '10px' }}>
                {conv.message_count} 則訊息
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}


// ─── 知識碎片瀏覽器面板 ─────────────────────────

const _TYPE_LABELS: Record<string, { label: string; icon: string; color: string }> = {
  strategy:           { label: '策略',     icon: '🎯', color: '#a855f7' },
  support_resistance: { label: '支撐壓力', icon: '🏷',  color: '#3b82f6' },
  lesson:             { label: '經驗教訓', icon: '💡', color: '#eab308' },
  factor_update:      { label: '因子更新', icon: '📊', color: '#22c55e' },
  invalidation:       { label: '失效條件', icon: '⚠️',  color: '#ef4444' },
  trend:              { label: '趨勢',     icon: '📈', color: '#06b6d4' },
  next_validation:    { label: '驗證待辦', icon: '🔍', color: '#f97316' },
  pattern:            { label: '型態',     icon: '📐', color: '#c084fc' },
  regime_tag:         { label: '市場環境', icon: '🌐', color: '#a3a3a3' },
  scores:             { label: '評分',     icon: '📝', color: '#737373' },
  indicator:          { label: '指標',     icon: '📉', color: '#2563eb' },
  volume:             { label: '量能',     icon: '📊', color: '#15803d' },
  sentiment:          { label: '情緒',     icon: '💭', color: '#facc15' },
  general:            { label: '一般',     icon: '📄', color: '#9ca3af' },
  user_note:          { label: '使用者筆記', icon: '✏️', color: '#79c0ff' },
};

function KnowledgePanel({
  fragments,
  total,
  symbols,
  types,
  loading,
  onRefresh,
  onDelete,
}: {
  fragments: FragmentItem[];
  total: number;
  symbols: string[];
  types: string[];
  loading: boolean;
  onRefresh: (params?: { symbol?: string; fragment_type?: string; sort_by?: string }) => void;
  onDelete: (id: number) => void;
}) {
  const [filterSymbol, setFilterSymbol] = useState('');
  const [filterType, setFilterType] = useState('');
  const [sortBy, setSortBy] = useState('hit_count');
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [panelHeight, setPanelHeight] = useState(280);
  const dragging = useRef(false);
  const dragStartY = useRef(0);
  const dragStartH = useRef(0);

  const applyFilters = (sym?: string, typ?: string, sort?: string) => {
    const s = sym ?? filterSymbol;
    const t = typ ?? filterType;
    const sb = sort ?? sortBy;
    onRefresh({ symbol: s || undefined, fragment_type: t || undefined, sort_by: sb });
  };

  const onDragStart = (e: React.MouseEvent) => {
    e.preventDefault();
    dragging.current = true;
    dragStartY.current = e.clientY;
    dragStartH.current = panelHeight;

    const onMove = (ev: MouseEvent) => {
      if (!dragging.current) return;
      const delta = ev.clientY - dragStartY.current;
      setPanelHeight(Math.max(120, Math.min(600, dragStartH.current + delta)));
    };
    const onUp = () => {
      dragging.current = false;
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  };

  return (
    <div
      className="border-b flex flex-col"
      style={{ borderColor: 'var(--border-color)', background: 'var(--bg-secondary)' }}
    >
      {/* 頂部 */}
      <div className="px-4 py-2 flex items-center justify-between shrink-0">
        <span className="text-xs font-medium" style={{ color: 'var(--text-primary)' }}>
          📚 知識庫 ({total} 筆)
        </span>
        <button
          onClick={() => applyFilters()}
          disabled={loading}
          className="text-xs cursor-pointer hover:opacity-80 disabled:opacity-40"
          style={{ color: 'var(--accent-blue)' }}
        >
          {loading ? '載入中...' : '🔄'}
        </button>
      </div>

      {/* 篩選列 */}
      <div className="px-4 pb-2 flex items-center gap-2 flex-wrap shrink-0">
        <select
          value={filterSymbol}
          onChange={e => { setFilterSymbol(e.target.value); applyFilters(e.target.value, undefined, undefined); }}
          className="text-xs px-2 py-1 rounded border cursor-pointer"
          style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)', borderColor: 'var(--border-color)' }}
        >
          <option value="">全部標的</option>
          {symbols.map(s => <option key={s} value={s === '通用' ? '' : s}>{s}</option>)}
        </select>

        <select
          value={filterType}
          onChange={e => { setFilterType(e.target.value); applyFilters(undefined, e.target.value, undefined); }}
          className="text-xs px-2 py-1 rounded border cursor-pointer"
          style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)', borderColor: 'var(--border-color)' }}
        >
          <option value="">全部類型</option>
          {types.map(t => {
            const info = _TYPE_LABELS[t];
            return <option key={t} value={t}>{info ? `${info.icon} ${info.label}` : t}</option>;
          })}
        </select>

        <select
          value={sortBy}
          onChange={e => { setSortBy(e.target.value); applyFilters(undefined, undefined, e.target.value); }}
          className="text-xs px-2 py-1 rounded border cursor-pointer"
          style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)', borderColor: 'var(--border-color)' }}
        >
          <option value="hit_count">命中最多</option>
          <option value="created_at">最新</option>
          <option value="type">按類型</option>
        </select>
      </div>

      {/* 碎片列表 */}
      <div className="overflow-y-auto px-3 pb-1" style={{ maxHeight: `${panelHeight}px` }}>
        {fragments.length === 0 ? (
          <p className="px-2 py-3 text-xs text-center" style={{ color: 'var(--text-secondary)' }}>
            {loading ? '正在載入...' : '尚無知識碎片（與 AI 對話後會自動累積）'}
          </p>
        ) : (
          fragments.map(frag => {
            const info = _TYPE_LABELS[frag.type] || _TYPE_LABELS.general;
            const isExpanded = expandedId === frag.id;
            return (
              <div
                key={frag.id}
                className="mb-1.5 rounded transition-colors"
                style={{ background: 'var(--bg-tertiary)' }}
              >
                {/* 主行 */}
                <div
                  className="flex items-start gap-2 px-3 py-2 cursor-pointer hover:opacity-90"
                  onClick={() => setExpandedId(isExpanded ? null : frag.id)}
                >
                  {/* 類型標籤 */}
                  <span
                    className="shrink-0 text-xs px-1.5 py-0.5 rounded font-medium"
                    style={{ background: `${info.color}22`, color: info.color, whiteSpace: 'nowrap' }}
                  >
                    {info.icon} {info.label}
                  </span>
                  {/* 內容 */}
                  <div className="flex-1 min-w-0">
                    <p className="text-xs leading-relaxed" style={{
                      color: 'var(--text-primary)',
                      display: '-webkit-box',
                      WebkitLineClamp: isExpanded ? 999 : 2,
                      WebkitBoxOrient: 'vertical',
                      overflow: 'hidden',
                    }}>
                      {frag.content}
                    </p>
                  </div>
                  {/* 指標 */}
                  <div className="shrink-0 flex items-center gap-2">
                    {frag.hit_count > 0 && (
                      <span className="text-xs" style={{ color: 'var(--accent-green)', fontSize: '10px' }}
                            title="被引用次數">
                        ×{frag.hit_count}
                      </span>
                    )}
                    <button
                      onClick={e => { e.stopPropagation(); if (confirm('確定刪除此知識碎片？')) onDelete(frag.id); }}
                      className="text-xs cursor-pointer hover:opacity-80"
                      style={{ color: 'var(--text-secondary)', fontSize: '10px' }}
                      title="刪除此碎片"
                    >
                      🗑
                    </button>
                  </div>
                </div>

                {/* 展開區域：來源追溯 */}
                {isExpanded && (
                  <div className="px-3 pb-2 pt-0" style={{ borderTop: '1px dashed var(--border-color)' }}>
                    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs mt-1.5" style={{ color: 'var(--text-secondary)', fontSize: '10px' }}>
                      <span>📍 {frag.symbol}</span>
                      <span>📅 {frag.created_at}</span>
                      <span>⏱ {frag.age_days < 1 ? '今天' : `${Math.floor(frag.age_days)} 天前`}</span>
                      <span>⭐ 品質 {(frag.quality_score * 100).toFixed(0)}%</span>
                      <span>🔄 引用 {frag.hit_count} 次</span>
                      {frag.is_seed && <span style={{ color: 'var(--accent-orange, #f0883e)' }}>🌱 種子碎片</span>}
                    </div>
                    {frag.source_question && (
                      <div className="mt-1.5 p-1.5 rounded text-xs" style={{ background: 'var(--bg-secondary)', color: 'var(--text-secondary)' }}>
                        <span style={{ opacity: 0.6 }}>來源提問：</span>{frag.source_question}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* 拖曳條 */}
      {/* 筆記輸入區 */}
      <NoteInput onRefresh={() => applyFilters()} />

      <div
        onMouseDown={onDragStart}
        className="shrink-0 flex items-center justify-center cursor-row-resize select-none"
        style={{ height: '8px', background: 'var(--border-color)', opacity: 0.5 }}
        title="拖曳調整面板高度"
      >
        <div style={{ width: 30, height: 3, borderRadius: 2, background: 'var(--text-secondary)', opacity: 0.5 }} />
      </div>
    </div>
  );
}

function NoteInput({ onRefresh }: { onRefresh: () => void }) {
  const [text, setText] = useState('');
  const [saving, setSaving] = useState(false);
  const symbol = useChartStore(s => s.symbol);

  const handleSave = async () => {
    if (text.trim().length < 5) return;
    setSaving(true);
    try {
      await addUserNote(text.trim(), symbol);
      toast('筆記已儲存到知識庫', 'success');
      setText('');
      onRefresh();
    } catch {
      toast('筆記儲存失敗', 'error');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="px-3 py-2 shrink-0 border-t" style={{ borderColor: 'var(--border-color)' }}>
      <div className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>
        ✏️ 添加學習筆記（會自動成為 AI 助手的參考知識）
      </div>
      <div className="flex gap-1.5">
        <input
          type="text"
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && !e.nativeEvent.isComposing) handleSave(); }}
          placeholder="記錄你的分析心得、策略觀察..."
          className="flex-1 px-2 py-1 rounded text-xs border"
          style={{
            background: 'var(--bg-primary)',
            color: 'var(--text-primary)',
            borderColor: 'var(--border-color)',
          }}
          maxLength={2000}
        />
        <button
          onClick={handleSave}
          disabled={saving || text.trim().length < 5}
          className="px-3 py-1 rounded text-xs cursor-pointer disabled:opacity-40 hover:opacity-80"
          style={{ background: 'var(--accent-blue)', color: '#fff' }}
        >
          {saving ? '...' : '儲存'}
        </button>
      </div>
    </div>
  );
}


// ─── 累計用量統計面板 ─────────────────────────

function UsageSummaryPanel({
  summary,
  loading,
  onRefresh,
}: {
  summary: UsageSummaryData | null;
  loading: boolean;
  onRefresh: () => void;
}) {
  const formatTokens = (n: number) => {
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
    if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
    return String(n);
  };

  const formatCost = (usd: number) => {
    if (usd === 0) return '免費';
    if (usd < 0.01) return `$${usd.toFixed(6)}`;
    return `$${usd.toFixed(4)}`;
  };

  return (
    <div
      className="px-4 py-3 border-b text-xs"
      style={{ borderColor: 'var(--border-color)', background: 'var(--bg-secondary)' }}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="font-medium" style={{ color: 'var(--text-primary)' }}>
          Token 用量統計
        </span>
        <button
          onClick={onRefresh}
          disabled={loading}
          className="cursor-pointer hover:opacity-80 disabled:opacity-40"
          style={{ color: 'var(--accent-blue)' }}
        >
          {loading ? '載入中...' : '🔄 更新'}
        </button>
      </div>

      {!summary ? (
        <p style={{ color: 'var(--text-secondary)' }}>
          {loading ? '正在載入...' : '尚無用量數據（需先設定 API Key）'}
        </p>
      ) : (
        <>
          <div className="grid grid-cols-3 gap-2 mb-1.5">
            <div className="text-center p-1.5 rounded" style={{ background: 'var(--bg-tertiary)' }}>
              <div style={{ color: 'var(--text-secondary)' }}>今日</div>
              <div className="font-mono font-medium" style={{ color: 'var(--text-primary)' }}>
                {formatTokens(summary.today.total_tokens)}
              </div>
              <div style={{ color: 'var(--accent-green)' }}>
                {formatCost(summary.today.estimated_cost_usd)}
              </div>
            </div>
            <div className="text-center p-1.5 rounded" style={{ background: 'var(--bg-tertiary)' }}>
              <div style={{ color: 'var(--text-secondary)' }}>本月</div>
              <div className="font-mono font-medium" style={{ color: 'var(--text-primary)' }}>
                {formatTokens(summary.month.total_tokens)}
              </div>
              <div style={{ color: 'var(--accent-green)' }}>
                {formatCost(summary.month.estimated_cost_usd)}
              </div>
            </div>
            <div className="text-center p-1.5 rounded" style={{ background: 'var(--bg-tertiary)' }}>
              <div style={{ color: 'var(--text-secondary)' }}>累計</div>
              <div className="font-mono font-medium" style={{ color: 'var(--text-primary)' }}>
                {formatTokens(summary.all_time.total_tokens)}
              </div>
              <div style={{ color: 'var(--accent-green)' }}>
                {formatCost(summary.all_time.estimated_cost_usd)}
              </div>
            </div>
          </div>
          <p style={{ color: 'var(--text-secondary)', opacity: 0.6, fontSize: '10px' }}>
            ⚠️ {summary.note}
          </p>
        </>
      )}
    </div>
  );
}


// ─── Token 用量顯示組件（單條訊息）──────────────

function TokenUsageBadge({ usage }: { usage: TokenUsage }) {
  const cost = usage.estimated_cost_usd;
  const isFree = cost === 0;
  // 用前端加總避免 Google API total_token_count 包含 system instruction 導致數字不一致
  const displayTotal = usage.prompt_tokens + usage.completion_tokens;

  return (
    <div
      className="flex items-center gap-2 mt-1 ml-1 text-xs"
      style={{ color: 'var(--text-secondary)', opacity: 0.7 }}
    >
      <span title="模型">
        {usage.model || usage.provider}
      </span>
      <span title="Token 用量（輸入 + 輸出）" className="flex items-center gap-0.5">
        <svg width="10" height="10" viewBox="0 0 16 16" fill="currentColor" opacity="0.5">
          <path d="M8 0a8 8 0 110 16A8 8 0 018 0zm0 2a6 6 0 100 12A6 6 0 008 2zm0 1a1 1 0 011 1v4.586l2.707 2.707a1 1 0 01-1.414 1.414l-3-3A1 1 0 017 9V4a1 1 0 011-1z"/>
        </svg>
        {usage.prompt_tokens} + {usage.completion_tokens} = {displayTotal} tokens
      </span>
      <span title="估計費用">
        {isFree ? (
          <span style={{ color: 'var(--accent-green)' }}>免費</span>
        ) : (
          <span>${cost < 0.01 ? cost.toFixed(6) : cost.toFixed(4)}</span>
        )}
      </span>
    </div>
  );
}
