/** 阿斯拉量化系統 — 對話介面組件（Streaming + Token Usage + 累計用量） */

import { useState, useRef, useEffect, useCallback } from 'react';
import { useChartStore } from '../../stores/chartStore';
import { streamChatMessage, fetchUsageSummary, fetchChatHistory, fetchConversationMessages, fetchDistillStatus, streamDistillPreview, confirmDistill, fetchFragments, deleteFragment, addUserNote } from '../../services/api';
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

// v133：UI 重組為 7 tabs，每 item 綁 mode 確保 LLM 強制 call 對應 function。
// 新增 Tab 2「市場結構與情境」收納 SMC / 三情境 / 條件機率 / 事件型態 4 個專屬入口。
// 原「市場體制」tab 移除，GMM/HMM 上移 Tab 1 基礎分析。
// Tab 7「全部分析」改名「完整技術分析」對齊 v132 編排管線實際涵蓋範圍，
// 並新增「一鍵連跑」sequence item 補足基本面/族群/事件型態。
// v144：分析功能按鈕重構 — 改成「報告維度」直達按鈕。
// 每顆點了直接出該維度報告：mode → 帶 mode 自動送；autoSend → 聚焦 prompt 自動送（不帶 mode prefix）；
// sequence → 一鍵連跑。移除原 7 tab 結構 + 綜合段落按鈕（跨維度結論/正式結論卡/策略風險/摘要表 — 單獨跑會空殼）。
interface AnalysisAction {
  label: string;
  prompt: string;
  mode?: string;       // 帶 mode → 後端注入該模式 prefix，自動送出
  autoSend?: boolean;  // 無 mode 但自動送（聚焦 prompt 讓 LLM 只出該段，不被 mode prefix 蓋過）
  sequence?: string[]; // 一鍵連跑：依序送多條
}

const ANALYSIS_ACTIONS: AnalysisAction[] = [
  { label: '市場環境維度', prompt: '分析 {symbol} 的市場環境維度（GMM/HMM/GARCH 三模型 + 情境概率 + 因子 Bucket + regime 相容性）', mode: 'regime_analysis' },
  { label: '協議基本面概覽', prompt: '輸出 {symbol} 的協議基本面概覽（代幣經濟學 + 生態活躍度 + 技術路線圖含非即時免責），不要其他段落', mode: 'crypto_fundamental' },
  { label: '基本面 vs 技術面交叉驗證', prompt: '對 {symbol} 進行加密基本面 vs 技術面交叉驗證，不要其他段落', mode: 'crypto_fundamental' },
  { label: '結構分析維度', prompt: '對 {symbol} 進行 SMC 結構分析（BOS / CHoCH / Order Block / FVG / 流動性掃蕩 / Premium-Discount）', mode: 'smc_structure' },
  { label: '動能特徵維度', prompt: '對 {symbol} 進行完整動能分析（多週期動量 + 加速度 + 相對強弱 + 反轉偵測）', mode: 'momentum_analysis' },
  { label: '多策略回測維度分析', prompt: '對 {symbol} 進行多策略回測（回測 + Monte Carlo + Walk Forward + CPCV 交叉驗證）', mode: 'strategy_backtest' },
  { label: '量化研究維度', prompt: '對 {symbol} 進行完整量化研究（因子 IC + Decay + Walk Forward + Monte Carlo + 倉位）', mode: 'quant_research' },
  { label: '基本面分析', prompt: '對 {symbol} 進行完整基本面分析', mode: 'fundamental_analysis' },
  { label: '族群分析', prompt: '分析 {symbol} 所屬族群的技術面、Breadth 與族群內個股相對強弱排名', mode: 'sector_analysis' },
  { label: '校準的最佳指標參數', prompt: '校準 {symbol} 的最佳指標參數', mode: 'calibrate' },
  { label: '規劃的分批進場價位', prompt: '規劃 {symbol} 的分批進場價位（依當前 regime 自動選配比策略）', mode: 'compute_laddered' },
  // 一鍵連跑：sequence 依標的型別動態組（見 buildRunAllSequence）。
  // 此處 sequence 僅作「這是 sequence 按鈕」的判定旗標（內容會被動態版本取代）。
  { label: '一鍵連跑：技術 + 基本面 + 事件 + 校準', prompt: '對 {symbol} 進行全部分析', sequence: ['對 {symbol} 進行全部分析（v132 5 維度技術面編排）'] },
];

// v149：一鍵連跑改成 symbol-aware + 去冗餘。
// - 去冗餘：移除「分批進場」步驟（步驟1 comprehensive 已強制 compute_laddered_entries
//   並在 seg1 輸出分批表），避免重複多跑一通重 LLM 呼叫。
// - Crypto：以「協議基本面概覽」取代台股月營收基本面；跳過台股族群（crypto 無此概念）。
// - 回傳含 {symbol} 模板，由呼叫端做 replace。
function buildRunAllSequence(sym: string): string[] {
  const isTW = sym.toUpperCase().endsWith('/TWD');
  if (isTW) {
    return [
      '對 {symbol} 進行全部分析（v132 5 維度技術面編排）',
      '對 {symbol} 進行基本面分析',
      '分析 {symbol} 所屬族群的技術面與相對強弱',
      '分析 {symbol} 歷史大漲大跌前的共通特徵',
      '校準 {symbol} 的最佳指標參數',
    ];
  }
  return [
    '對 {symbol} 進行全部分析（v132 5 維度技術面編排）',
    '輸出 {symbol} 的協議基本面概覽（代幣經濟學 + 生態活躍度 + 技術路線圖含非即時免責）',
    '分析 {symbol} 歷史大漲大跌前的共通特徵',
    '校準 {symbol} 的最佳指標參數',
  ];
}

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
  // ★ v100：SSE 蒸餾進度 + 錯誤顯示
  const [distillProgress, setDistillProgress] = useState<{ current: number; total: number; message: string } | null>(null);
  const [distillError, setDistillError] = useState<string | null>(null);
  const distillAbortRef = useRef<(() => void) | null>(null);
  const [quickQuestions, setQuickQuestions] = useState<string[]>(loadQuickQuestions);
  const [editingQuick, setEditingQuick] = useState(false);
  const [analysisBarOpen, setAnalysisBarOpen] = useState(true);
  // v146：多選打包 — 已勾選的維度按鈕索引
  const [selectedActions, setSelectedActions] = useState<Set<number>>(new Set());
  const [showKnowledgePanel, setShowKnowledgePanel] = useState(false);
  const [knowledgeFragments, setKnowledgeFragments] = useState<FragmentItem[]>([]);
  const [knowledgeTotal, setKnowledgeTotal] = useState(0);
  const [knowledgeSymbols, setKnowledgeSymbols] = useState<string[]>([]);
  const [knowledgeTypes, setKnowledgeTypes] = useState<string[]>([]);
  const [knowledgeLoading, setKnowledgeLoading] = useState(false);
  const [editDraft, setEditDraft] = useState<string[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const userScrolledUpRef = useRef(false);
  const isComposingRef = useRef(false);
  const streamingMsgIdRef = useRef<string | null>(null);
  const abortStreamRef = useRef<(() => void) | null>(null);
  // v139：sequence 標記讓後端走精簡 chart_state 省 token
  const messageQueueRef = useRef<{ text: string; mode?: string; isSequenceFollow?: boolean }[]>([]);
  const [queueLength, setQueueLength] = useState(0);
  const MAX_QUEUE = 3;
  const messages = useChartStore((s) => s.messages);
  const addMessage = useChartStore((s) => s.addMessage);
  const updateMessage = useChartStore((s) => s.updateMessage);
  const chatLoading = useChartStore((s) => s.chatLoading);
  const setChatLoading = useChartStore((s) => s.setChatLoading);
  const llmConfig = useChartStore((s) => s.llmConfig);
  // getChartStateSummary 和 conversationId 在 _executeSend 中從 store 即時讀取
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

  // ★ v100：執行蒸餾預覽 — 改用 SSE 串流（不再受 axios timeout 限制）
  const runDistillPreview = useCallback(() => {
    if (!llmConfig.sessionId) return;
    setDistillLoading(true);
    setDistillStep('preview');
    setDistillError(null);
    setDistillProgress(null);
    // 串流期間累積 previews，避免最終事件遺失時前端空空
    const accumPreviews: Array<Record<string, unknown>> = [];

    distillAbortRef.current = streamDistillPreview(
      llmConfig.provider as LLMProvider,
      llmConfig.sessionId,
      {
        onStatus: (msg, total) => setDistillProgress({ current: 0, total, message: msg }),
        onProgress: (current, total, message) => setDistillProgress({ current, total, message }),
        onPreviewItem: (item) => {
          accumPreviews.push(item);
          // 邊收邊推到畫面（使用者可以看到一個個冒出來）
          setDistillPreviews([...accumPreviews]);
        },
        onError: (msg) => {
          setDistillError(msg);
          setDistillLoading(false);
          // 不退回 status — 留在 preview 步驟讓使用者看到錯誤
        },
        onDone: (payload) => {
          setDistillPreviews(payload.previews);
          setDistillProfile(payload.profile_preview || '');
          setDistillTokens(payload.total_tokens_used || 0);
          setDistillProgress(null);
          setDistillLoading(false);
        },
      },
    );
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

  // 智慧捲動：只有用戶在底部附近才自動捲動，用戶往上看歷史時不打擾
  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return;

    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = container;
      // 距離底部 < 150px 視為「在底部」
      userScrolledUpRef.current = scrollHeight - scrollTop - clientHeight > 150;
    };
    container.addEventListener('scroll', handleScroll);
    return () => container.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    if (!userScrolledUpRef.current) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  const handleSend = useCallback(async (sendMode?: string, explicitText?: string) => {
    // v146.1：explicitText 讓批次送出直接帶文字（不靠 stale input state closure）
    const trimmed = (explicitText ?? input).trim();
    if (!trimmed) return;

    const currentlyLoading = useChartStore.getState().chatLoading;

    // 如果正在分析中 → 加入佇列
    if (currentlyLoading) {
      if (messageQueueRef.current.length >= MAX_QUEUE) {
        const warnMsg: ChatMessage = {
          id: `system-${Date.now()}`,
          role: 'system',
          content: `⚠️ 排隊已滿（最多 ${MAX_QUEUE} 個），請等待目前分析完成。`,
          timestamp: new Date().toISOString(),
        };
        addMessage(warnMsg);
        return;
      }
      messageQueueRef.current.push({ text: trimmed, mode: sendMode });
      setQueueLength(messageQueueRef.current.length);
      const queueMsg: ChatMessage = {
        id: `user-${Date.now()}`,
        role: 'user',
        content: trimmed,
        timestamp: new Date().toISOString(),
      };
      addMessage(queueMsg);
      const queueNotice: ChatMessage = {
        id: `system-queue-${Date.now()}`,
        role: 'system',
        content: `⏳ 已加入排隊（第 ${messageQueueRef.current.length} 個），將在目前分析完成後自動執行。`,
        timestamp: new Date().toISOString(),
      };
      addMessage(queueNotice);
      setInput('');
      return;
    }

    setInput('');
    await _executeSend(trimmed, sendMode, true);
  }, [input, addMessage]);

  // 佇列處理（由 _executeSend 結束時直接呼叫，不依賴 useEffect）
  const _processQueue = async () => {
    if (messageQueueRef.current.length === 0) return;
    await new Promise(resolve => setTimeout(resolve, 1500));

    // ★ 兜底：主動清理前一任務可能殘留的狀態
    if (abortStreamRef.current) {
      try { abortStreamRef.current(); } catch {}
      abortStreamRef.current = null;
    }
    if (useChartStore.getState().chatLoading) {
      setChatLoading(false);
      await new Promise(r => setTimeout(r, 50));  // 讓 React flush state
    }

    if (messageQueueRef.current.length > 0 && !abortRef.current) {
      const next = messageQueueRef.current.shift()!;
      setQueueLength(messageQueueRef.current.length);
      const startNotice: ChatMessage = {
        id: `system-queue-start-${Date.now()}`,
        role: 'system',
        content: `▶️ 排隊任務開始執行：${next.text.slice(0, 50)}${next.text.length > 50 ? '...' : ''}`,
        timestamp: new Date().toISOString(),
      };
      addMessage(startNotice);
      // v139：傳遞 isSequenceFollow flag 給 _executeSend，讓後端走精簡 chart_state
      await _executeSend(next.text, next.mode, false, next.isSequenceFollow ?? false);
    }
  };

  // 終止執行（停止當前分析 + 清空佇列 + 真正斷開 HTTP 連線）
  const abortRef = useRef(false);
  const handleAbort = () => {
    abortRef.current = true;
    messageQueueRef.current = [];
    setQueueLength(0);

    // 真正斷開 SSE 連線（通知後端停止處理）
    abortStreamRef.current?.();
    abortStreamRef.current = null;

    setChatLoading(false);
    if (streamingMsgIdRef.current) {
      const currentMsg = useChartStore.getState().messages.find(m => m.id === streamingMsgIdRef.current);
      updateMessage(streamingMsgIdRef.current, {
        isThinking: false,
        isStreaming: false,
        statusText: undefined,
        content: (currentMsg?.content || '') + '\n\n⛔ 已終止',
      });
      streamingMsgIdRef.current = null;
    }
    const abortMsg: ChatMessage = {
      id: `system-abort-${Date.now()}`,
      role: 'system',
      content: '⛔ 已終止分析，可以繼續輸入新的問題。',
      timestamp: new Date().toISOString(),
    };
    addMessage(abortMsg);
    setTimeout(() => { abortRef.current = false; }, 500);
  };

  // v110：監聽 TopBar 廣播的「強制中斷」事件（streaming 中切換 symbol/timeframe/日期時觸發）
  // 走既有 handleAbort 路徑保證所有清理（佇列、SSE、UI 狀態、終止訊息）一致
  useEffect(() => {
    const onForceAbort = () => {
      if (useChartStore.getState().chatLoading) {
        handleAbort();
      }
    };
    window.addEventListener('force-abort-streaming', onForceAbort);
    return () => window.removeEventListener('force-abort-streaming', onForceAbort);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // P3：visibilitychange 保護
  // 切到瀏覽器分頁不主動 abort（既有行為）。切回 visible 時若仍在 streaming，
  // 在當前 streaming bubble 顯示「分析仍在進行中」狀態，避免使用者誤以為系統卡死。
  useEffect(() => {
    const onVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        const store = useChartStore.getState();
        if (store.chatLoading && streamingMsgIdRef.current) {
          const currentMsg = store.messages.find(m => m.id === streamingMsgIdRef.current);
          if (currentMsg?.isStreaming) {
            store.updateMessage(streamingMsgIdRef.current, {
              statusText: currentMsg.statusText || '分析仍在進行中（您剛切回此分頁）...',
            });
          }
        }
      }
    };
    document.addEventListener('visibilitychange', onVisibilityChange);
    return () => document.removeEventListener('visibilitychange', onVisibilityChange);
  }, []);

  // 普通函式（非 useCallback）— 每次呼叫時讀取最新的 store 狀態
  // v139：isSequenceFollow=true 標明為一鍵連跑後續訊息，後端走精簡 chart_state 省 token
  const _executeSend = async (trimmed: string, sendMode?: string, addUserMsg: boolean = true, isSequenceFollow: boolean = false) => {
    const store = useChartStore.getState();
    const currentLlmConfig = store.llmConfig;
    const currentConversationId = store.conversationId;

    if (addUserMsg) {
      const userMsg: ChatMessage = {
        id: `user-${Date.now()}`,
        role: 'user',
        content: trimmed,
        timestamp: new Date().toISOString(),
      };
      addMessage(userMsg);
    }
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
    // v125-A → v132: 多段輸出狀態。seg1（30 秒卡）固定用 assistantMsgId / accumulatedText；
    // segment ≥ 2 用 map（舊 monolith 只有 seg2；新編排管線有 2-6 五維度 + 7 synthesis）。
    const segBubbles = new Map<number, string>();  // segment 號 → bubble message id
    const segTexts = new Map<number, string>();    // segment 號 → 累積文字
    let activeSegment = 1;  // 由 onSegmentComplete 切換、由 segment-aware onToken 維護
    let warnedSegIncomplete = false;  // 後端已送 warning event → onDone 不重複偵測

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
    if (captureFn && !abortRef.current) {
      try {
        // ★ 5 秒超時：html2canvas 已知在某些圖表狀態下會 hang，避免整個任務卡住
        const img = await Promise.race([
          captureFn(),
          new Promise<null>((resolve) => setTimeout(() => resolve(null), 5000)),
        ]);
        if (img) screenshot = img;
      } catch { /* 截圖失敗不阻塞發送 */ }
    }

    // 被終止 → 不再送出請求
    if (abortRef.current) {
      setChatLoading(false);
      return;
    }

    const { promise: streamPromise, abort: streamAbort } = streamChatMessage(
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

        onToken: (content: string, segment?: number) => {
          // v132: 依後端 segment 欄位路由 token 到正確的 message bubble
          // - segment ≥ 2 且該 bubble 已建立 → 寫進 segBubbles 對應 bubble
          // - 其餘（segment 1 / undefined / bubble 未建立）→ 寫進 seg1（assistantMsgId）
          const seg = segment ?? activeSegment;
          const segMsgId = seg >= 2 ? segBubbles.get(seg) : undefined;
          if (segMsgId) {
            const next = (segTexts.get(seg) ?? '') + content;
            segTexts.set(seg, next);
            updateMessage(segMsgId, {
              content: next,
              isThinking: false,
              isStreaming: true,
              statusText: undefined,
              progress: undefined,
            });
          } else {
            accumulatedText += content;
            updateMessage(assistantMsgId, {
              content: accumulatedText,
              isThinking: false,
              isStreaming: true,
              statusText: undefined,
              progress: undefined,
            });
          }
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
                // 新一輪分析開始畫圖 → 自動隱藏先前輪次的標註（圖層面板可恢復）
                // 同一輪內多批標註共用 turnId，重複呼叫無副作用
                store.startAnnotationTurn(assistantMsgId);
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
                    turnId: assistantMsgId,
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
            updateMessage(assistantMsgId, {
              content: `錯誤：${error}`,
              role: 'system',
              isThinking: false,
              isStreaming: false,
            });
          } else {
            accumulatedText += `\n\n⚠️ ${error}`;
            updateMessage(assistantMsgId, {
              content: accumulatedText,
              isStreaming: false,
            });
          }
        },

        onDone: (_convId?: string, hints?: Record<string, unknown>) => {
          doneReceived = true;
          // v132：收尾 seg1 + 所有 segment ≥ 2 的 bubble
          updateMessage(assistantMsgId, {
            isThinking: false,
            isStreaming: false,
            statusText: undefined,
          });
          // 舊 monolith 路徑（只有 seg2 一個額外 bubble）→ 沿用 4000 字啟發式完整性檢測。
          // 新編排管線（segBubbles 有 2-6 維度 + 7 synthesis）→ 後端已逐段 emit warning，
          // 各維度本來就只有 ~800-1500 字，不適用 4000 字門檻，故僅做收尾。
          const isMonolithSeg2 = segBubbles.size === 1 && segBubbles.has(2);
          if (isMonolithSeg2 && !warnedSegIncomplete) {
            const seg2MsgId = segBubbles.get(2)!;
            const seg2Text = segTexts.get(2) ?? '';
            const hasSeg2Keyword = /#1|#2|📊\s*市場環境|🏛\s*結構|⚡\s*動能/.test(seg2Text);
            const isShort = seg2Text.length < 4000;
            if (isShort || !hasSeg2Keyword) {
              const note = (
                `\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n` +
                `⚠️ **第 2 段（完整分析）可能不完整**\n` +
                `字數 ${seg2Text.length}（預期 ≥ 4000）${!hasSeg2Keyword ? '、未偵測到「📊 市場環境」等關鍵段落' : ''}\n` +
                `streaming 可能中途中斷。可重新對相同標的送出「全部分析」重生第 2 段。\n` +
                `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`
              );
              segTexts.set(2, seg2Text + note);
              updateMessage(seg2MsgId, {
                content: seg2Text + note,
                segmentIncomplete: true,
              });
            }
          }
          // 收尾所有 segment ≥ 2 的 bubble
          for (const segMsgId of segBubbles.values()) {
            updateMessage(segMsgId, {
              isThinking: false,
              isStreaming: false,
              statusText: undefined,
            });
          }
          setChatLoading(false);
          streamingMsgIdRef.current = null;
          if (hints?.distill_hint) {
            setDistillHint(true);
          }
        },
        // v100：結論卡「📈 系統參考」由系統替換為實際命中率
        onAccuracyInject: (data: { old_pattern: string; new_text: string }) => {
          const currentMsg = useChartStore.getState().messages.find(m => m.id === assistantMsgId);
          if (!currentMsg) return;
          const replaced = String(currentMsg.content || '').replace(
            /📈\s*系統參考：[^\n]*/,
            data.new_text,
          );
          updateMessage(assistantMsgId, { content: replaced });
        },
        // v104 Q3：LLM 數值編造偵測 — 有 mismatch 時附加紅色警示區塊到訊息底部
        onFactCheck: (data) => {
          if (!data.mismatches || data.mismatches.length === 0) return;
          const currentMsg = useChartStore.getState().messages.find(m => m.id === assistantMsgId);
          if (!currentMsg) return;
          if (String(currentMsg.content || '').includes('🚨 **系統數值核對**')) return;

          const lines = data.mismatches.slice(0, 5).map((m) => {
            const label = m.name ? `${m.type}/${m.name}` : m.type;
            return `  - **${label}**：報告寫 \`${m.claimed}\`，實際值 \`${m.actual}\`\n    片段：${m.snippet}`;
          });
          const note = `\n\n---\n🚨 **系統數值核對**：報告中 ${data.mismatches.length} 處數字跟即時資料不一致，請以下列實際值為準：\n${lines.join('\n')}`;
          updateMessage(assistantMsgId, { content: String(currentMsg.content || '') + note });
        },
        // v103 Phase 2B：refined SHAP（基於真實 entry/target/stop）
        onShapRefine: (data) => {
          if (!data.top_features?.length) return;
          const currentMsg = useChartStore.getState().messages.find(m => m.id === assistantMsgId);
          if (!currentMsg) return;
          // 避免重複附加
          if (String(currentMsg.content || '').includes('🎯 **SHAP（基於真實進場價）**')) return;

          const topLine = data.top_features
            .slice(0, 3)
            .map(f => `${f.name}=${f.value != null ? f.value.toFixed(2) : '?'}（推${f.direction === 'long' ? '多' : '空'}）`)
            .join('、');
          const parts = [`🎯 **SHAP（基於真實進場價）**：${topLine}`];
          if (data.p_hit_target != null) parts.push(`命中機率 ${(data.p_hit_target * 100).toFixed(0)}%`);
          if (data.model_regime) parts.push(`模型 regime：${data.model_regime}`);
          const refinedNote = '\n\n' + parts.join('｜');

          updateMessage(assistantMsgId, { content: String(currentMsg.content || '') + refinedNote });
        },
        // v107.1：機械審查徽章（純 Python，零成本）
        onAudit: (data) => {
          if (!data || data.n_checks === 0) return;
          const currentMsg = useChartStore.getState().messages.find(m => m.id === assistantMsgId);
          if (!currentMsg) return;
          if (String(currentMsg.content || '').includes('🔍 **機械審查**')) return;

          const lines = [`🔍 **機械審查**：${data.summary}`];
          if (!data.passed && data.issues.length > 0) {
            lines.push('');
            for (const issue of data.issues.slice(0, 5)) {
              lines.push(`  - ${issue}`);
            }
          }
          const note = '\n\n---\n' + lines.join('\n');
          updateMessage(assistantMsgId, { content: String(currentMsg.content || '') + note });
        },

        // AI 覆核：回答完成後低一階模型交叉檢查（issues 明細由後端以 token 串流附加）
        onVerify: (data) => {
          const seg = activeSegment;
          const msgId = seg === 1 ? assistantMsgId : segBubbles.get(seg);
          if (!msgId) return;
          if (data.status === 'checking') {
            updateMessage(msgId, { statusText: '🔎 AI 覆核中（低一階模型交叉檢查數據/邏輯）...' });
            return;
          }
          if (data.status === 'pass') {
            const marker = '🔎 **AI 覆核**';
            const note = `\n\n---\n${marker}：✅ 通過（${data.model ?? '低階模型'} 交叉檢查，未發現數據/邏輯錯誤）`;
            if (seg === 1) {
              if (!accumulatedText.includes(marker)) {
                accumulatedText += note;
                updateMessage(msgId, { content: accumulatedText, statusText: undefined });
                return;
              }
            } else {
              const cur = segTexts.get(seg) ?? '';
              if (!cur.includes(marker)) {
                const next = cur + note;
                segTexts.set(seg, next);
                updateMessage(msgId, { content: next, statusText: undefined });
                return;
              }
            }
          }
          // issues / skipped / 已附加過 → 只清 statusText
          updateMessage(msgId, { statusText: undefined });
        },

        // v125-A → v132：某段完成 → 結束該段 bubble、新建下一段 bubble
        // 舊 monolith：seg1 → seg2 兩段。新編排管線：seg1 → 5 維度 → synthesis 共 7 段。
        // 每段獨立 message，後段 streaming 中斷不污染前段。
        onSegmentComplete: (data) => {
          const doneSeg = data.segment;
          const nextSeg = data.next_segment;
          const nextLabel = data.next_label || `第 ${nextSeg} 段`;
          // 1. 結束已完成段的 bubble（seg1 用 assistantMsgId，其餘查 segBubbles）
          const doneMsgId = doneSeg === 1 ? assistantMsgId : segBubbles.get(doneSeg);
          if (doneMsgId) {
            const doneText = doneSeg === 1 ? accumulatedText : (segTexts.get(doneSeg) ?? '');
            updateMessage(doneMsgId, {
              content: doneText,
              isStreaming: false,
              segment: doneSeg,
              segmentComplete: true,
              statusText: undefined,
            });
          }
          // 2. 新建下一段 bubble
          const newId = `assistant-seg${nextSeg}-${Date.now()}`;
          segBubbles.set(nextSeg, newId);
          segTexts.set(nextSeg, '');
          activeSegment = nextSeg;
          streamingMsgIdRef.current = newId;
          const nextInitMsg: ChatMessage = {
            id: newId,
            role: 'assistant',
            content: '',
            timestamp: new Date().toISOString(),
            isThinking: false,
            isStreaming: true,
            segment: nextSeg,
            statusText: `${nextLabel}生成中...`,
          };
          addMessage(nextInitMsg);
        },

        // v125-B → v132：後端 warning event（維度不完整 / synthesis 不完整 / 字數不足等）
        // 一律附加到當前 active 段的 bubble；標記 warnedSegIncomplete 讓 onDone 不重複偵測
        onWarning: (data) => {
          // AI 覆核的問題明細已由後端 token 串流附加到內容，不再疊 warning 橫幅、
          // 也不該把該段標成 incomplete（覆核有問題 ≠ 生成不完整）
          if (data.type === 'verify_issues') return;
          warnedSegIncomplete = true;
          const seg = activeSegment;
          const msgId = seg === 1 ? assistantMsgId : segBubbles.get(seg);
          if (!msgId) return;
          const note = `\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n⚠️ **${data.message}**\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`;
          if (seg === 1) {
            accumulatedText += note;
            updateMessage(msgId, { content: accumulatedText, segmentIncomplete: true });
          } else {
            const next = (segTexts.get(seg) ?? '') + note;
            segTexts.set(seg, next);
            updateMessage(msgId, { content: next, segmentIncomplete: true });
          }
        },
      },
      currentConversationId || undefined,
      store.getChartStateSummary(),
      currentLlmConfig.provider,
      currentLlmConfig.sessionId,
      chatHistory,
      sendMode,
      screenshot,
      isSequenceFollow,  // v139 token 優化：sequence 後續訊息走精簡 chart_state
    );

    // 保存 abort 函式，讓 handleAbort 可以真正斷開連線
    abortStreamRef.current = streamAbort;
    // v122 fix: 「全部分析」實際需 15-20 分鐘（round1 LLM 8m + function calls 2m
    // + round2 兩段 5-10m）。舊版 630s 永遠不夠，導致用戶看到「stream 莫名斷
    // 在 10 分 30 秒」的 bug（前端 HARD_TIMEOUT 觸發 → controller.abort → 後端
    // 收到 cancel → client_disconnect log）。新值 30 分鐘給「全部分析」充分
    // 空間；真正 idle 仍由 api.ts STREAM_TIMEOUT_MS (600s) 觸發、不會無限等。
    const HARD_TIMEOUT_MS = 1800_000;
    try {
      await Promise.race([
        streamPromise,
        new Promise((_, reject) => setTimeout(
          () => reject(new Error('hard_timeout_streamPromise')), HARD_TIMEOUT_MS
        )),
      ]);
    } catch (e) {
      // hard_timeout 觸發 → 確保佇列不卡（finally 會跑 setChatLoading(false)）
      try { streamAbort(); } catch {}
    }
    abortStreamRef.current = null;

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

    // 處理佇列中的下一個任務
    if (!abortRef.current) {
      await _processQueue();
    }
  };

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
          progress={distillProgress}
          error={distillError}
          onPreview={runDistillPreview}
          onConfirm={runDistillConfirm}
          onCancel={() => {
            distillAbortRef.current?.();
            distillAbortRef.current = null;
            setDistillLoading(false);
            setDistillStep('status');
            setDistillError(null);
            setDistillProgress(null);
            setDistillPreviews(null);
          }}
          onClose={() => {
            distillAbortRef.current?.();
            distillAbortRef.current = null;
            setShowDistillPanel(false);
            setDistillStep('status');
            setDistillPreviews(null);
            setDistillError(null);
            setDistillProgress(null);
          }}
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
        </div>
        {analysisBarOpen && (
          <div className="flex flex-wrap gap-1.5 px-3 py-2" style={{ background: 'var(--bg-secondary)' }}>
            {ANALYSIS_ACTIONS.map((item, i) => {
              // v146：一鍵連跑（有 sequence）維持立即送；其餘 11 維度改勾選式打包
              const isSequence = !!(item.sequence && item.sequence.length > 0);
              const isSelected = selectedActions.has(i);
              return (
                <button
                  key={i}
                  onClick={() => {
                    const sym = useChartStore.getState().symbol || 'BTC/USDT';
                    if (isSequence) {
                      const seq = buildRunAllSequence(sym);
                      setInput(seq[0].replace('{symbol}', sym));
                      setTimeout(() => handleSend(), 50);
                      for (let k = 1; k < seq.length; k++) {
                        messageQueueRef.current.push({ text: seq[k].replace('{symbol}', sym), isSequenceFollow: true });
                      }
                      setQueueLength(messageQueueRef.current.length);
                    } else {
                      // 維度按鈕：toggle 勾選（不立即送）
                      setSelectedActions(prev => {
                        const next = new Set(prev);
                        if (next.has(i)) next.delete(i); else next.add(i);
                        return next;
                      });
                    }
                  }}
                  className="px-2.5 py-1 rounded text-xs cursor-pointer transition-colors"
                  style={
                    isSequence
                      ? { background: 'var(--accent-blue)', color: '#fff', fontWeight: 500 }
                      : isSelected
                        ? { background: 'var(--accent-blue)', color: '#fff', border: '1px solid var(--accent-blue)' }
                        : { background: 'var(--bg-tertiary)', color: 'var(--text-primary)', border: '1px solid transparent' }
                  }
                >
                  {!isSequence && isSelected ? '✓ ' : ''}{item.label}
                </button>
              );
            })}
            {/* v146：送出選取 + 清除（依勾選順序打包，各帶自己的 mode 接力送出）*/}
            {selectedActions.size > 0 && (
              <>
                <button
                  onClick={() => {
                    const sym = useChartStore.getState().symbol || 'BTC/USDT';
                    const idxs = Array.from(selectedActions).sort((a, b) => a - b);
                    const ts = Date.now();
                    idxs.forEach((idx, order) => {
                      const it = ANALYSIS_ACTIONS[idx];
                      const text = it.prompt.replace('{symbol}', sym);
                      if (order === 0) {
                        // 第 1 個立即送（explicitText 不靠 stale input；_executeSend 會加 user 泡泡）
                        setInput('');
                        const m = it.mode;
                        setTimeout(() => handleSend(m, text), 50);
                      } else {
                        // 其餘：立即加唯一 id 的 user 泡泡（全部可見）+ 排隊接力
                        addMessage({
                          id: `user-batch-${ts}-${idx}`,
                          role: 'user',
                          content: text,
                          timestamp: new Date().toISOString(),
                        });
                        messageQueueRef.current.push({ text, mode: it.mode, isSequenceFollow: true });
                      }
                    });
                    setQueueLength(messageQueueRef.current.length);
                    setSelectedActions(new Set());
                  }}
                  className="px-2.5 py-1 rounded text-xs cursor-pointer font-medium"
                  style={{ background: 'var(--accent-green, #3fb950)', color: '#fff' }}
                >
                  ▶ 送出選取 ({selectedActions.size})
                </button>
                <button
                  onClick={() => setSelectedActions(new Set())}
                  className="px-2 py-1 rounded text-xs cursor-pointer"
                  style={{ background: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}
                >
                  清除
                </button>
              </>
            )}
          </div>
        )}
      </div>

      {/* 訊息列表 */}
      <div ref={messagesContainerRef} className="flex-1 overflow-y-auto p-4 space-y-3">
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
            disabled={!input.trim()}
            className="px-4 py-1.5 rounded text-sm font-medium cursor-pointer transition-opacity disabled:opacity-40"
            style={{
              background: chatLoading ? 'var(--accent-orange, #f59e0b)' : 'var(--accent-blue)',
              color: '#fff',
            }}
          >
            {chatLoading ? `排隊${queueLength > 0 ? `(${queueLength})` : ''}` : '發送'}
          </button>
          {chatLoading && (
            <button
              onClick={handleAbort}
              className="px-3 py-1.5 rounded text-xs cursor-pointer font-medium"
              style={{ background: '#dc2626', color: '#fff' }}
              title="終止分析"
            >
              ⛔ 終止
            </button>
          )}
          {queueLength > 0 && !chatLoading && (
            <button
              onClick={() => { messageQueueRef.current = []; setQueueLength(0); }}
              className="px-2 py-1.5 rounded text-xs cursor-pointer"
              style={{ background: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}
              title="清除排隊"
            >
              ✕
            </button>
          )}
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
  progress,
  error,
  onPreview,
  onConfirm,
  onCancel,
  onClose,
}: {
  status: Record<string, unknown> | null;
  previews: Array<Record<string, unknown>> | null;
  profile: string;
  step: 'status' | 'preview' | 'done';
  loading: boolean;
  tokensUsed: number;
  progress: { current: number; total: number; message: string } | null;
  error: string | null;
  onPreview: () => void;
  onConfirm: () => void;
  onCancel: () => void;
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
            {/* ★ v100：錯誤訊息（非靜默退回，給重試按鈕）*/}
            {error && (
              <div className="mb-2 p-2 rounded" style={{ background: 'rgba(248,81,73,0.1)', border: '1px solid var(--accent-red, #f85149)' }}>
                <p className="font-medium mb-1" style={{ color: 'var(--accent-red, #f85149)' }}>
                  ⚠️ 蒸餾失敗
                </p>
                <p style={{ color: 'var(--text-primary)', fontSize: '11px' }}>{error}</p>
                <div className="flex gap-2 mt-2">
                  <button onClick={onPreview} className="flex-1 py-1.5 rounded text-xs cursor-pointer"
                    style={{ background: 'var(--accent-blue)', color: '#fff' }}>重試</button>
                  <button onClick={onCancel} className="flex-1 py-1.5 rounded text-xs cursor-pointer"
                    style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}>返回</button>
                </div>
              </div>
            )}

            {loading ? (
              <div className="py-4 text-center" style={{ color: 'var(--text-secondary)' }}>
                <p className="mb-1">🧠 AI 正在整理你的歷史對話...</p>
                {/* ★ v100：SSE 進度條 */}
                {progress && progress.total > 0 ? (
                  <>
                    <div className="my-2 mx-2" style={{ background: 'var(--bg-tertiary)', height: '6px', borderRadius: '3px', overflow: 'hidden' }}>
                      <div style={{
                        background: 'var(--accent-blue)', height: '100%',
                        width: `${Math.round((progress.current / progress.total) * 100)}%`,
                        transition: 'width 0.3s ease',
                      }} />
                    </div>
                    <p className="font-medium" style={{ color: 'var(--accent-blue)', fontSize: '11px' }}>
                      {progress.message || `${progress.current} / ${progress.total}`}
                    </p>
                  </>
                ) : (
                  <p style={{ fontSize: '10px' }}>這可能需要 1-7 分鐘（依未蒸餾 symbol 數量），請耐心等候</p>
                )}
                {/* 邊收邊顯示已完成的預覽（讓使用者看到一個個冒出來，不會以為當機）*/}
                {previews && previews.length > 0 && (
                  <p className="mt-2" style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>
                    已完成 {previews.length} 個 symbol
                  </p>
                )}
                <button
                  onClick={onCancel}
                  className="mt-3 px-3 py-1 rounded text-xs cursor-pointer"
                  style={{ background: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}
                >
                  取消蒸餾
                </button>
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
