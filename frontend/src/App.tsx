/** 阿斯拉量化系統 — 主應用組件 */

import { Component, useState, useEffect, useRef } from 'react';
import type { ReactNode, ErrorInfo } from 'react';
import ChartView from './components/ChartView/ChartView';
import ChatInterface from './components/ChatInterface/ChatInterface';
import IndicatorPanel from './components/IndicatorPanel/IndicatorPanel';
import SettingsPanel from './components/SettingsPanel/SettingsPanel';
import SyncPanel from './components/SyncPanel/SyncPanel';
import TopBar from './components/TopBar';
import OnboardingGuide, { useOnboarding } from './components/OnboardingGuide';
import { ToastContainer } from './components/Toast';
import { useChartStore } from './stores/chartStore';
import { calculateIndicator } from './services/api';

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
              <IndicatorPanel />
            </ErrorBoundary>
          </div>
        )}

        {/* 中間：K 線圖表 */}
        <div data-guide="chart" className="flex-1 flex flex-col overflow-hidden">
          <ErrorBoundary name="K 線圖表">
            <ChartView />
          </ErrorBoundary>
        </div>

        {/* 右側：對話介面 */}
        {chatExpanded && (
          <div
            data-guide="chat"
            className="flex flex-col border-l overflow-hidden"
            style={{
              width: '380px',
              minWidth: '380px',
              borderColor: 'var(--border-color)',
              background: 'var(--bg-secondary)',
            }}
          >
            <ErrorBoundary name="AI 助手">
              <ChatInterface />
            </ErrorBoundary>
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
          <SettingsPanel onClose={() => setShowSettings(false)} />
        </ErrorBoundary>
      )}

      {/* 數據同步面板（Modal） */}
      {showSyncPanel && (
        <ErrorBoundary name="數據同步">
          <SyncPanel />
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
