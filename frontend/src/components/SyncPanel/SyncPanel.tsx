/** 阿斯拉量化系統 — 數據同步面板
 *
 * 完整功能：
 * - 交易對多選（預設列表 + 自訂輸入）
 * - 時間週期多選 (15m, 1h, 4h, 1d, 1w)
 * - 日期範圍選擇（起始 ~ 結束）
 * - 交易所多選（至少 2 家）
 * - 強制更新開關
 * - 即時進度條 + 當前任務 + ETA
 * - 狀態表格 + 即時日誌
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { useChartStore } from '../../stores/chartStore';
import { toast } from '../Toast';
import {
  triggerDataSync,
  fetchSyncTaskProgress,
  fetchAvailableExchanges,
  type SyncRequestParams,
} from '../../services/api';

// 預設交易對
const DEFAULT_SYMBOLS = [
  'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT',
  'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT', 'MATIC/USDT',
];

// 時間週期
const ALL_TIMEFRAMES = [
  { value: '15m', label: '15 分鐘' },
  { value: '1h', label: '1 小時' },
  { value: '4h', label: '4 小時' },
  { value: '1d', label: '1 天' },
  { value: '1w', label: '1 週' },
];

interface ExchangeInfo {
  id: string;
  name: string;
  enabled_by_default: boolean;
  description: string;
}

interface TaskProgress {
  task_id: string;
  status: string;
  progress: number;
  current_item: string | null;
  total_items: number;
  completed_items: number;
  eta_seconds: number | null;
  logs: string[];
  errors: string[];
  started_at: string | null;
  completed_at: string | null;
}

export default function SyncPanel() {
  const setShowSyncPanel = useChartStore((s) => s.setShowSyncPanel);
  const loadChartData = useChartStore((s) => s.loadChartData);

  // ===== 表單狀態 =====
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>(['BTC/USDT']);
  const [selectedTimeframes, setSelectedTimeframes] = useState<string[]>(['1d']);
  const [exchanges, setExchanges] = useState<ExchangeInfo[]>([]);
  const [selectedExchanges, setSelectedExchanges] = useState<string[]>([
    'binance', 'bybit', 'okx', 'coinbase',
  ]);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [forceUpdate, setForceUpdate] = useState(false);
  const [customSymbol, setCustomSymbol] = useState('');

  // ===== 進度狀態 =====
  const [syncing, setSyncing] = useState(false);
  const [_taskId, setTaskId] = useState<string | null>(null);
  const [progress, setProgress] = useState<TaskProgress | null>(null);
  const [error, setError] = useState<string | null>(null);

  const logsEndRef = useRef<HTMLDivElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // 載入交易所清單
  useEffect(() => {
    fetchAvailableExchanges()
      .then((res) => setExchanges(res.exchanges || []))
      .catch(() => {
        // 用預設值
        setExchanges([
          { id: 'binance', name: 'Binance', enabled_by_default: true, description: '全球最大' },
          { id: 'bybit', name: 'Bybit', enabled_by_default: true, description: '合約交易所' },
          { id: 'okx', name: 'OKX', enabled_by_default: true, description: '綜合性' },
          { id: 'coinbase', name: 'Coinbase', enabled_by_default: true, description: 'USD 對' },
          { id: 'kraken', name: 'Kraken', enabled_by_default: false, description: '歐洲老牌' },
        ]);
      });

    // 設定預設日期範圍（最近 90 天），含時分
    const today = new Date();
    const past = new Date(today);
    past.setDate(past.getDate() - 90);
    const toLocalISO = (d: Date) => {
      const pad = (n: number) => String(n).padStart(2, '0');
      return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
    };
    setEndDate(toLocalISO(today));
    setStartDate(toLocalISO(past));

    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  // 日誌自動捲動
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [progress?.logs.length]);

  // ===== 輪詢進度 =====
  const startPolling = useCallback(
    (id: string) => {
      if (pollRef.current) clearInterval(pollRef.current);

      pollRef.current = setInterval(async () => {
        try {
          const data: TaskProgress = await fetchSyncTaskProgress(id);
          setProgress(data);

          if (data.status === 'completed' || data.status === 'failed') {
            if (pollRef.current) clearInterval(pollRef.current);
            pollRef.current = null;
            setSyncing(false);

            if (data.status === 'completed') {
              loadChartData();
              toast(`數據同步完成（${data.completed_items}/${data.total_items}）`, 'success');
            } else {
              toast('數據同步失敗，請檢查日誌', 'error');
            }
          }
        } catch {
          // 忽略輪詢錯誤
        }
      }, 1500);
    },
    [loadChartData]
  );

  // ===== 開始同步 =====
  const handleSync = async () => {
    setError(null);

    if (selectedSymbols.length === 0) {
      setError('請至少選擇一個交易對');
      return;
    }
    if (selectedTimeframes.length === 0) {
      setError('請至少選擇一個時間週期');
      return;
    }
    if (selectedExchanges.length < 2) {
      setError('至少需要選擇 2 家交易所以啟用五源投票機制');
      return;
    }

    setSyncing(true);
    setProgress(null);

    try {
      const params: SyncRequestParams = {
        symbols: selectedSymbols,
        timeframes: selectedTimeframes,
        exchanges: selectedExchanges,
        force_update: forceUpdate,
      };
      if (startDate) params.start_date = startDate;
      if (endDate) params.end_date = endDate;

      const res = await triggerDataSync(params);
      setTaskId(res.task_id);
      startPolling(res.task_id);
    } catch (err: any) {
      setSyncing(false);
      const msg = err?.response?.data?.detail || err?.message || '同步啟動失敗';
      setError(msg);
      toast(msg, 'error');
    }
  };

  // ===== 交易對切換 =====
  const toggleSymbol = (sym: string) => {
    setSelectedSymbols((prev) =>
      prev.includes(sym) ? prev.filter((s) => s !== sym) : [...prev, sym]
    );
  };

  // 自訂交易對
  const addCustomSymbol = () => {
    const sym = customSymbol.trim().toUpperCase();
    if (!sym) return;
    // 自動加斜線
    let normalized = sym;
    if (!sym.includes('/')) {
      if (sym.endsWith('USDT')) normalized = sym.slice(0, -4) + '/USDT';
      else if (sym.endsWith('USD')) normalized = sym.slice(0, -3) + '/USD';
      else normalized = sym + '/USDT';
    }
    if (!selectedSymbols.includes(normalized)) {
      setSelectedSymbols((prev) => [...prev, normalized]);
    }
    setCustomSymbol('');
  };

  // ===== 時間週期切換 =====
  const toggleTimeframe = (tf: string) => {
    setSelectedTimeframes((prev) =>
      prev.includes(tf) ? prev.filter((t) => t !== tf) : [...prev, tf]
    );
  };

  // ===== 交易所切換 =====
  const toggleExchange = (id: string) => {
    setSelectedExchanges((prev) =>
      prev.includes(id) ? prev.filter((e) => e !== id) : [...prev, id]
    );
  };

  // ===== 格式化 ETA =====
  const formatETA = (seconds: number | null): string => {
    if (seconds === null || seconds <= 0) return '--';
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return `${m}m ${s}s`;
  };

  // ===== 快速選擇 =====
  const selectAllSymbols = () => setSelectedSymbols([...DEFAULT_SYMBOLS]);
  const clearAllSymbols = () => setSelectedSymbols([]);
  const selectAllTimeframes = () => setSelectedTimeframes(ALL_TIMEFRAMES.map((t) => t.value));

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ background: 'rgba(0,0,0,0.6)' }}
      onClick={() => !syncing && setShowSyncPanel(false)}
    >
      <div
        className="rounded-xl shadow-2xl flex flex-col"
        style={{
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-color)',
          width: '720px',
          maxHeight: '90vh',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* ===== 標題列 ===== */}
        <div
          className="flex items-center justify-between px-6 py-4 border-b"
          style={{ borderColor: 'var(--border-color)' }}
        >
          <h2 className="text-lg font-bold" style={{ color: 'var(--text-primary)' }}>
            數據同步
          </h2>
          {!syncing && (
            <button
              onClick={() => setShowSyncPanel(false)}
              className="text-lg cursor-pointer hover:opacity-80"
              style={{ color: 'var(--text-secondary)' }}
            >
              ✕
            </button>
          )}
        </div>

        {/* ===== 可捲動的內容 ===== */}
        <div className="flex-1 overflow-y-auto p-6 space-y-5">
          {/* === 交易對選擇 === */}
          <div>
            <h3 className="text-sm font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
              交易對
              <span className="text-xs font-normal ml-2" style={{ color: 'var(--text-secondary)' }}>
                已選 {selectedSymbols.length} 個
              </span>
            </h3>
            <div className="flex flex-wrap gap-1.5 mb-2">
              {DEFAULT_SYMBOLS.map((sym) => (
                <button
                  key={sym}
                  onClick={() => toggleSymbol(sym)}
                  disabled={syncing}
                  className="px-2.5 py-1 rounded text-xs cursor-pointer transition-colors"
                  style={{
                    background: selectedSymbols.includes(sym)
                      ? 'var(--accent-blue)'
                      : 'var(--bg-tertiary)',
                    color: selectedSymbols.includes(sym) ? '#fff' : 'var(--text-secondary)',
                    opacity: syncing ? 0.5 : 1,
                  }}
                >
                  {sym}
                </button>
              ))}
              {/* 已選但不在預設列表的 */}
              {selectedSymbols
                .filter((s) => !DEFAULT_SYMBOLS.includes(s))
                .map((sym) => (
                  <button
                    key={sym}
                    onClick={() => toggleSymbol(sym)}
                    disabled={syncing}
                    className="px-2.5 py-1 rounded text-xs cursor-pointer"
                    style={{ background: 'var(--accent-purple)', color: '#fff' }}
                  >
                    {sym} ✕
                  </button>
                ))}
            </div>
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={customSymbol}
                onChange={(e) => setCustomSymbol(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && addCustomSymbol()}
                placeholder="自訂交易對，如 PEPE/USDT"
                disabled={syncing}
                className="flex-1 px-3 py-1.5 rounded text-xs border-none outline-none"
                style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
              />
              <button
                onClick={addCustomSymbol}
                disabled={syncing}
                className="px-3 py-1.5 rounded text-xs cursor-pointer"
                style={{ background: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}
              >
                新增
              </button>
              <button
                onClick={selectAllSymbols}
                disabled={syncing}
                className="px-2 py-1.5 rounded text-xs cursor-pointer"
                style={{ color: 'var(--accent-blue)' }}
              >
                全選
              </button>
              <button
                onClick={clearAllSymbols}
                disabled={syncing}
                className="px-2 py-1.5 rounded text-xs cursor-pointer"
                style={{ color: 'var(--text-secondary)' }}
              >
                清除
              </button>
            </div>
          </div>

          {/* === 時間週期 === */}
          <div>
            <h3 className="text-sm font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
              時間週期
              <button
                onClick={selectAllTimeframes}
                disabled={syncing}
                className="text-xs font-normal ml-2 cursor-pointer"
                style={{ color: 'var(--accent-blue)' }}
              >
                全選
              </button>
            </h3>
            <div className="flex gap-2">
              {ALL_TIMEFRAMES.map((tf) => (
                <button
                  key={tf.value}
                  onClick={() => toggleTimeframe(tf.value)}
                  disabled={syncing}
                  className="px-4 py-2 rounded text-sm cursor-pointer transition-colors"
                  style={{
                    background: selectedTimeframes.includes(tf.value)
                      ? 'var(--accent-blue)'
                      : 'var(--bg-tertiary)',
                    color: selectedTimeframes.includes(tf.value)
                      ? '#fff'
                      : 'var(--text-secondary)',
                    opacity: syncing ? 0.5 : 1,
                  }}
                >
                  <div className="font-medium">{tf.value}</div>
                  <div className="text-xs opacity-80">{tf.label}</div>
                </button>
              ))}
            </div>
          </div>

          {/* === 日期範圍 === */}
          <div>
            <h3 className="text-sm font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
              日期範圍
              <span className="text-xs font-normal ml-2" style={{ color: 'var(--text-secondary)' }}>
                可精確到小時分鐘
              </span>
            </h3>
            <div className="flex items-center gap-3 flex-wrap">
              <input
                type="datetime-local"
                lang="zh-TW"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                disabled={syncing}
                className="px-3 py-2 rounded text-sm border-none outline-none"
                style={{
                  background: 'var(--bg-tertiary)',
                  color: 'var(--text-primary)',
                  colorScheme: 'dark',
                }}
              />
              <span style={{ color: 'var(--text-secondary)' }}>~</span>
              <input
                type="datetime-local"
                lang="zh-TW"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                disabled={syncing}
                className="px-3 py-2 rounded text-sm border-none outline-none"
                style={{
                  background: 'var(--bg-tertiary)',
                  color: 'var(--text-primary)',
                  colorScheme: 'dark',
                }}
              />
              {/* 快速選擇 */}
              <div className="flex gap-1 ml-1">
                {[30, 90, 180, 365].map((d) => {
                  const toLocalISO = (dt: Date) => {
                    const pad = (n: number) => String(n).padStart(2, '0');
                    return `${dt.getFullYear()}-${pad(dt.getMonth() + 1)}-${pad(dt.getDate())}T${pad(dt.getHours())}:${pad(dt.getMinutes())}`;
                  };
                  return (
                    <button
                      key={d}
                      disabled={syncing}
                      onClick={() => {
                        const today = new Date();
                        const past = new Date(today);
                        past.setDate(past.getDate() - d);
                        setEndDate(toLocalISO(today));
                        setStartDate(toLocalISO(past));
                      }}
                      className="px-2 py-1 rounded text-xs cursor-pointer"
                      style={{ background: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}
                    >
                      {d}天
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          {/* === 交易所選擇 === */}
          <div>
            <h3 className="text-sm font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
              交易所
              <span className="text-xs font-normal ml-2" style={{ color: 'var(--text-secondary)' }}>
                至少選擇 2 家
              </span>
            </h3>
            <div className="grid grid-cols-2 gap-2">
              {exchanges.map((ex) => (
                <button
                  key={ex.id}
                  onClick={() => toggleExchange(ex.id)}
                  disabled={syncing}
                  className="flex items-center gap-2 px-3 py-2 rounded text-left cursor-pointer transition-colors"
                  style={{
                    background: selectedExchanges.includes(ex.id)
                      ? 'rgba(88, 166, 255, 0.1)'
                      : 'var(--bg-tertiary)',
                    border: selectedExchanges.includes(ex.id)
                      ? '1px solid var(--accent-blue)'
                      : '1px solid transparent',
                    opacity: syncing ? 0.5 : 1,
                  }}
                >
                  <div
                    className="w-3 h-3 rounded-sm border flex items-center justify-center"
                    style={{
                      borderColor: selectedExchanges.includes(ex.id)
                        ? 'var(--accent-blue)'
                        : 'var(--border-color)',
                      background: selectedExchanges.includes(ex.id)
                        ? 'var(--accent-blue)'
                        : 'transparent',
                    }}
                  >
                    {selectedExchanges.includes(ex.id) && (
                      <span className="text-white text-xs">✓</span>
                    )}
                  </div>
                  <div>
                    <span className="text-sm" style={{ color: 'var(--text-primary)' }}>
                      {ex.name}
                    </span>
                    <span className="text-xs ml-2" style={{ color: 'var(--text-secondary)' }}>
                      {ex.description}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* === 進階選項 === */}
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={forceUpdate}
                onChange={(e) => setForceUpdate(e.target.checked)}
                disabled={syncing}
                className="accent-blue-500"
              />
              <span className="text-sm" style={{ color: 'var(--text-primary)' }}>
                強制更新（忽略本地數據，重新抓取全部）
              </span>
            </label>
          </div>

          {/* === 同步摘要 === */}
          <div
            className="p-3 rounded-lg text-xs"
            style={{ background: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}
          >
            將同步 <b style={{ color: 'var(--text-primary)' }}>{selectedSymbols.length}</b> 個交易對
            × <b style={{ color: 'var(--text-primary)' }}>{selectedTimeframes.length}</b> 個週期
            = <b style={{ color: 'var(--accent-blue)' }}>{selectedSymbols.length * selectedTimeframes.length}</b> 組數據，
            使用 <b style={{ color: 'var(--text-primary)' }}>{selectedExchanges.length}</b> 家交易所
            {startDate && endDate && (
              <span>
                ，日期範圍：{startDate.replace('T', ' ')} ~ {endDate.replace('T', ' ')}
              </span>
            )}
          </div>

          {/* === 進度顯示 === */}
          {progress && (
            <div className="space-y-3">
              {/* 進度條 */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                    {progress.status === 'completed'
                      ? '同步完成'
                      : progress.status === 'failed'
                      ? '同步失敗'
                      : progress.current_item || '準備中...'}
                  </span>
                  <span className="text-xs" style={{ color: 'var(--text-primary)' }}>
                    {progress.completed_items}/{progress.total_items} ({Math.round(progress.progress)}%)
                  </span>
                </div>
                <div
                  className="w-full h-2 rounded-full overflow-hidden"
                  style={{ background: 'var(--bg-tertiary)' }}
                >
                  <div
                    className="h-full rounded-full transition-all duration-300"
                    style={{
                      width: `${progress.progress}%`,
                      background:
                        progress.status === 'completed'
                          ? 'var(--accent-green)'
                          : progress.status === 'failed'
                          ? 'var(--accent-red)'
                          : 'var(--accent-blue)',
                    }}
                  />
                </div>
                {progress.eta_seconds !== null && progress.status === 'running' && (
                  <div className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>
                    預估剩餘時間：{formatETA(progress.eta_seconds)}
                  </div>
                )}
              </div>

              {/* 錯誤訊息 */}
              {progress.errors.length > 0 && (
                <div
                  className="p-2 rounded text-xs space-y-1"
                  style={{ background: 'rgba(248, 81, 73, 0.1)' }}
                >
                  {progress.errors.map((err, i) => (
                    <div key={i} style={{ color: 'var(--accent-red)' }}>
                      {err}
                    </div>
                  ))}
                </div>
              )}

              {/* 即時日誌 */}
              <div>
                <h4 className="text-xs font-medium mb-1" style={{ color: 'var(--text-secondary)' }}>
                  同步日誌
                </h4>
                <div
                  className="rounded p-2 max-h-40 overflow-y-auto text-xs font-mono space-y-0.5"
                  style={{ background: 'var(--bg-primary)', color: 'var(--text-secondary)' }}
                >
                  {progress.logs.map((log, i) => (
                    <div key={i} className={log.includes('錯誤') ? 'text-red-400' : ''}>
                      {log}
                    </div>
                  ))}
                  <div ref={logsEndRef} />
                </div>
              </div>
            </div>
          )}

          {/* === 錯誤提示 === */}
          {error && (
            <div
              className="p-3 rounded text-sm"
              style={{ background: 'rgba(248, 81, 73, 0.1)', color: 'var(--accent-red)' }}
            >
              {error}
            </div>
          )}
        </div>

        {/* ===== 底部按鈕列 ===== */}
        <div
          className="flex items-center justify-end gap-3 px-6 py-4 border-t"
          style={{ borderColor: 'var(--border-color)' }}
        >
          {!syncing && (
            <button
              onClick={() => setShowSyncPanel(false)}
              className="px-4 py-2 rounded-lg text-sm cursor-pointer"
              style={{ background: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}
            >
              關閉
            </button>
          )}
          <button
            onClick={handleSync}
            disabled={syncing}
            className="px-6 py-2 rounded-lg text-sm font-medium cursor-pointer transition-opacity disabled:opacity-50"
            style={{ background: 'var(--accent-blue)', color: '#fff' }}
          >
            {syncing ? '同步中...' : '開始同步'}
          </button>
        </div>
      </div>
    </div>
  );
}
