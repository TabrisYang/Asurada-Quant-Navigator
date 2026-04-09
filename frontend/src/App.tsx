/** 阿斯拉量化系統 — 主應用組件 */

import { Component, useState, useEffect, useRef, useCallback, lazy, Suspense } from 'react';
import type { ReactNode, ErrorInfo } from 'react';
import TopBar from './components/TopBar';
import OnboardingGuide, { useOnboarding } from './components/OnboardingGuide';
import { ToastContainer } from './components/Toast';
import { useChartStore } from './stores/chartStore';
import { calculateIndicator } from './services/api';

// Lazy load 重量級元件（Code Splitting）
const AlertPanel = lazy(() => import('./components/AlertPanel/AlertPanel'));
const ChartView = lazy(() => import('./components/ChartView/ChartView'));
const ChatInterface = lazy(() => import('./components/ChatInterface/ChatInterface'));
const IndicatorPanel = lazy(() => import('./components/IndicatorPanel/IndicatorPanel'));
const SettingsPanel = lazy(() => import('./components/SettingsPanel/SettingsPanel'));
const SyncPanel = lazy(() => import('./components/SyncPanel/SyncPanel'));

// 共用載入骨架
function LoadingSkeleton({ name }: { name: string }) {
  return (
    <div className="flex items-center justify-center h-full" style={{ background: 'var(--bg-secondary)' }}>
      <span style={{ color: 'var(--text-secondary)', fontSize: '12px' }}>
        {name} 載入中...
      </span>
    </div>
  );
}

// ─── ErrorBoundary：捕獲子組件錯誤，避免整頁白屏/黑屏 ───
interface EBProps { children: ReactNode; name: string }
interface EBState { hasError: boolean; error: Error | null }

class ErrorBoundary extends Component<EBProps, EBState> {
  state: EBState = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(`[ErrorBoundary:${this.props.name}]`, error, info.componentStack);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex items-center justify-center h-full p-4" style={{ background: '#0d1117' }}>
          <div className="text-center" style={{ color: '#8b949e' }}>
            <p className="text-sm mb-2" style={{ color: '#f85149' }}>
              {this.props.name} 發生錯誤
            </p>
            <p className="text-xs mb-3" style={{ maxWidth: '280px', wordBreak: 'break-word' }}>
              {this.state.error?.message || '未知錯誤'}
            </p>
            <button
              onClick={() => this.setState({ hasError: false, error: null })}
              className="px-3 py-1 rounded text-xs cursor-pointer"
              style={{ background: '#58a6ff', color: '#fff' }}
            >
              重試
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

function App() {
  const [showSettings, setShowSettings] = useState(false);
  const [chatExpanded, setChatExpanded] = useState(true);
  const [panelExpanded, setPanelExpanded] = useState(true);

  // 對話面板寬度拖曳調整
  const [chatWidth, setChatWidth] = useState(() => {
    const saved = localStorage.getItem('asura_chat_width');
    return saved ? Math.max(320, Math.min(700, Number(saved))) : 380;
  });
  const chatResizeRef = useRef({ dragging: false, startX: 0, startWidth: 0 });

  const handleChatResizeMove = useCallback((e: MouseEvent) => {
    const ref = chatResizeRef.current;
    if (!ref.dragging) return;
    const delta = ref.startX - e.clientX;
    const newWidth = Math.max(320, Math.min(700, ref.startWidth + delta));
    setChatWidth(newWidth);
  }, []);

  const handleChatResizeEnd = useCallback(() => {
    chatResizeRef.current.dragging = false;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    document.removeEventListener('mousemove', handleChatResizeMove);
    document.removeEventListener('mouseup', handleChatResizeEnd);
    setChatWidth((w) => { localStorage.setItem('asura_chat_width', String(w)); return w; });
  }, [handleChatResizeMove]);

  const handleChatResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    chatResizeRef.current = { dragging: true, startX: e.clientX, startWidth: chatWidth };
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    document.addEventListener('mousemove', handleChatResizeMove);
    document.addEventListener('mouseup', handleChatResizeEnd);
  }, [chatWidth, handleChatResizeMove, handleChatResizeEnd]);
  const { visible: showOnboarding, dismiss: dismissOnboarding } = useOnboarding();

  const symbol = useChartStore((s) => s.symbol);
  const timeframe = useChartStore((s) => s.timeframe);
  const startDate = useChartStore((s) => s.startDate);
  const endDate = useChartStore((s) => s.endDate);
  const showSyncPanel = useChartStore((s) => s.showSyncPanel);
  const loadChartData = useChartStore((s) => s.loadChartData);

  const activeIndicators = useChartStore((s) => s.activeIndicators);
  const ohlcvData = useChartStore((s) => s.ohlcvData);
  const updateIndicatorData = useChartStore((s) => s.updateIndicatorData);

  // 自動載入數據：當 symbol / timeframe / 日期範圍 變化時觸發
  useEffect(() => {
    loadChartData(symbol, timeframe);
  }, [symbol, timeframe, startDate, endDate, loadChartData]);

  // ===== 自動載入指標數據 =====
  // 監控 activeIndicators，任何缺少 data 的指標自動呼叫後端 API 計算
  const loadingIndicatorsRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (ohlcvData.length === 0) return;

    const needLoad = activeIndicators.filter(
      (ind) => !ind.data && !loadingIndicatorsRef.current.has(ind.id)
    );

    if (needLoad.length === 0) return;

    for (const ind of needLoad) {
      loadingIndicatorsRef.current.add(ind.id);

      calculateIndicator(ind.indicator_type, 'add', {
        ...ind.parameters,
        symbol,
        timeframe,
        start_date: startDate || undefined,
        end_date: endDate || undefined,
      })
        .then((res) => {
          if (res?.data) {
            updateIndicatorData(ind.id, res.data);
          }
        })
        .catch((err) => {
          console.error(`指標載入失敗: ${ind.indicator_type}`, err?.message);
        })
        .finally(() => {
          loadingIndicatorsRef.current.delete(ind.id);
        });
    }
  }, [activeIndicators, ohlcvData.length, symbol, timeframe, startDate, endDate, updateIndicatorData]);

  return (
    <div className="flex flex-col h-screen" style={{ background: 'var(--bg-primary)' }}>
      {/* 頂部工具列 */}
      <ErrorBoundary name="頂部工具列">
        <div data-guide="topbar">
          <TopBar onSettingsClick={() => setShowSettings(true)} />
        </div>
      </ErrorBoundary>

      {/* 主要內容區 */}
      <div className="flex flex-1 overflow-hidden">
        {/* 左側：指標面板 */}
        {panelExpanded && (
          <div
            data-guide="indicator-panel"
            className="flex flex-col border-r overflow-y-auto"
            style={{
              width: '280px',
              minWidth: '280px',
              borderColor: 'var(--border-color)',
              background: 'var(--bg-secondary)',
            }}
          >
            <ErrorBoundary name="指標面板">
              <Suspense fallback={<LoadingSkeleton name="指標面板" />}>
                <IndicatorPanel />
              </Suspense>
            </ErrorBoundary>
          </div>
        )}

        {/* 中間：預警 + K 線圖表 */}
        <div data-guide="chart" className="flex-1 flex flex-col overflow-hidden">
          <ErrorBoundary name="預警面板">
            <Suspense fallback={null}>
              <AlertPanel />
            </Suspense>
          </ErrorBoundary>
          <ErrorBoundary name="K 線圖表">
            <Suspense fallback={<LoadingSkeleton name="K 線圖表" />}>
              <ChartView />
            </Suspense>
          </ErrorBoundary>
        </div>

        {/* 右側：對話介面（可拖曳左邊框調整寬度） */}
        {chatExpanded && (
          <div className="relative flex" style={{ width: chatWidth, minWidth: 320 }}>
            {/* 拖曳把手 */}
            <div
              onMouseDown={handleChatResizeStart}
              className="absolute left-0 top-0 bottom-0 z-10"
              style={{ width: 5, cursor: 'col-resize' }}
            />
            <div
              data-guide="chat"
              className="flex flex-col border-l overflow-hidden flex-1"
              style={{
                borderColor: 'var(--border-color)',
                background: 'var(--bg-secondary)',
              }}
            >
              <ErrorBoundary name="AI 助手">
                <Suspense fallback={<LoadingSkeleton name="AI 助手" />}>
                  <ChatInterface />
                </Suspense>
              </ErrorBoundary>
            </div>
          </div>
        )}
      </div>

      {/* 底部狀態列 */}
      <div
        className="flex items-center justify-between px-4 py-1 text-xs border-t"
        style={{
          borderColor: 'var(--border-color)',
          color: 'var(--text-secondary)',
          background: 'var(--bg-secondary)',
        }}
      >
        <div className="flex items-center gap-4">
          <span>{symbol} · {timeframe}</span>
          <span className="flex items-center gap-1">
            <span
              className="inline-block w-2 h-2 rounded-full"
              style={{ background: 'var(--accent-green)' }}
            />
            系統就緒
          </span>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => setPanelExpanded(!panelExpanded)}
            className="hover:opacity-80 cursor-pointer"
          >
            {panelExpanded ? '隱藏指標面板' : '顯示指標面板'}
          </button>
          <button
            onClick={() => setChatExpanded(!chatExpanded)}
            className="hover:opacity-80 cursor-pointer"
          >
            {chatExpanded ? '隱藏對話' : '顯示對話'}
          </button>
        </div>
      </div>

      {/* 設定面板（Modal） */}
      {showSettings && (
        <ErrorBoundary name="設定面板">
          <Suspense fallback={<LoadingSkeleton name="設定面板" />}>
            <SettingsPanel onClose={() => setShowSettings(false)} />
          </Suspense>
        </ErrorBoundary>
      )}

      {/* 數據同步面板（Modal） */}
      {showSyncPanel && (
        <ErrorBoundary name="數據同步">
          <Suspense fallback={<LoadingSkeleton name="數據同步" />}>
            <SyncPanel />
          </Suspense>
        </ErrorBoundary>
      )}

      {/* 新手引導 */}
      {showOnboarding && <OnboardingGuide onDismiss={dismissOnboarding} />}

      {/* 全域通知 */}
      <ToastContainer />
    </div>
  );
}

export default App;
