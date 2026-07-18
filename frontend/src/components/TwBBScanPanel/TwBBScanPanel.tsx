/** 阿斯拉量化系統 — 台股 BB Width 壓縮掃描器面板
 *
 * 一鍵掃描全部上市櫃 ~1900 檔，找出 BB Width 百分位低於門檻（布林通道壓縮）的個股。
 *   - 顯示代號 / 名稱 / 最新價 + 日期 / 產業 / BB Width 百分位 / 20 日漲跌
 *   - 支援即時進度 + 取消
 *   - 點「分析」將標的注入主圖表 + 對 AI 送完整分析請求
 *   - CSV 匯出
 */

import { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import { useChartStore } from '../../stores/chartStore';
import { addSymbolsToList } from '../TopBar';
import {
  streamTwScan,
  listTwScanHistory,
  getTwScanResult,
  revisitTwScan,
  deleteTwScan,
  type TwScanResult,
  type TwScanProgress,
  type TwScanDone,
  type TwScanFailure,
  type TwScanSummary,
  type TwScanRevisitItem,
} from '../../services/api';
import { toast } from '../Toast';
import { PCTILE_OPTIONS, Th, FailureList } from './shared';
import { TrackView } from './TrackView';
import { HistoryView } from './HistoryView';

type SortKey = 'bb_width_pctile' | 'change_20d' | 'volume_5d_avg' | 'code';
type PanelTab = 'scan' | 'history' | 'track';

const PANEL_SIZE_STORAGE_KEY = 'asura_twscan_panel_size';
const PANEL_MIN_WIDTH = 600;
const PANEL_MIN_HEIGHT = 400;
const PANEL_DEFAULT_WIDTH = 1000;

type ResizeDirection = 'n' | 's' | 'e' | 'w' | 'se';
interface PanelState { width: number; height: number; left: number; top: number; }

function _defaultPanelState(): PanelState {
  const w = Math.min(PANEL_DEFAULT_WIDTH, window.innerWidth);
  const h = Math.floor(window.innerHeight * 0.9);
  return {
    width: w,
    height: h,
    left: Math.max(0, Math.floor((window.innerWidth - w) / 2)),
    top: Math.max(0, Math.floor(window.innerHeight * 0.05)),
  };
}

function _clampPanelState(s: PanelState): PanelState {
  const w = Math.max(PANEL_MIN_WIDTH, Math.min(window.innerWidth, s.width));
  const h = Math.max(PANEL_MIN_HEIGHT, Math.min(window.innerHeight, s.height));
  const left = Math.max(0, Math.min(window.innerWidth - w, s.left));
  const top = Math.max(0, Math.min(window.innerHeight - h, s.top));
  return { width: w, height: h, left, top };
}

function _loadPanelState(): PanelState {
  try {
    const raw = localStorage.getItem(PANEL_SIZE_STORAGE_KEY);
    if (raw) {
      const p = JSON.parse(raw);
      if (typeof p?.width === 'number' && typeof p?.height === 'number') {
        const def = _defaultPanelState();
        return _clampPanelState({
          width: p.width,
          height: p.height,
          left: typeof p.left === 'number' ? p.left : def.left,
          top: typeof p.top === 'number' ? p.top : def.top,
        });
      }
    }
  } catch {
    /* ignore parse errors */
  }
  return _defaultPanelState();
}

export default function TwBBScanPanel() {
  const show = useChartStore((s) => s.showTwScanPanel);
  const setShow = useChartStore((s) => s.setShowTwScanPanel);
  const setSymbol = useChartStore((s) => s.setSymbol);
  const setTimeframe = useChartStore((s) => s.setTimeframe);
  const setPendingChatMessage = useChartStore((s) => s.setPendingChatMessage);

  // 視窗大小與位置（可拖四邊 + 右下角、localStorage 持久化）
  const [panelState, setPanelState] = useState<PanelState>(() => _loadPanelState());
  const resizeRef = useRef({
    active: false,
    direction: 'se' as ResizeDirection,
    startX: 0, startY: 0,
    startW: 0, startH: 0, startLeft: 0, startTop: 0,
  });

  const handleResizeMove = useCallback((e: MouseEvent) => {
    const ref = resizeRef.current;
    if (!ref.active) return;
    const dx = e.clientX - ref.startX;
    const dy = e.clientY - ref.startY;

    let { startW: w, startH: h, startLeft: left, startTop: top } = ref;
    const d = ref.direction;
    if (d === 'e' || d === 'se') w = ref.startW + dx;
    if (d === 's' || d === 'se') h = ref.startH + dy;
    if (d === 'w') { left = ref.startLeft + dx; w = ref.startW - dx; }
    if (d === 'n') { top = ref.startTop + dy; h = ref.startH - dy; }

    // 左/上拉到 min 時：固定 left/top 不再變動，避免 modal 反向偏移
    if (d === 'w' && w < PANEL_MIN_WIDTH) {
      left = ref.startLeft + (ref.startW - PANEL_MIN_WIDTH);
      w = PANEL_MIN_WIDTH;
    }
    if (d === 'n' && h < PANEL_MIN_HEIGHT) {
      top = ref.startTop + (ref.startH - PANEL_MIN_HEIGHT);
      h = PANEL_MIN_HEIGHT;
    }

    setPanelState(_clampPanelState({ width: w, height: h, left, top }));
  }, []);

  const handleResizeEnd = useCallback(() => {
    resizeRef.current.active = false;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    document.removeEventListener('mousemove', handleResizeMove);
    document.removeEventListener('mouseup', handleResizeEnd);
    setPanelState((s) => {
      try { localStorage.setItem(PANEL_SIZE_STORAGE_KEY, JSON.stringify(s)); } catch { /* ignore */ }
      return s;
    });
  }, [handleResizeMove]);

  const startResize = useCallback((direction: ResizeDirection, cursor: string) =>
    (e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      resizeRef.current = {
        active: true,
        direction,
        startX: e.clientX,
        startY: e.clientY,
        startW: panelState.width,
        startH: panelState.height,
        startLeft: panelState.left,
        startTop: panelState.top,
      };
      document.body.style.cursor = cursor;
      document.body.style.userSelect = 'none';
      document.addEventListener('mousemove', handleResizeMove);
      document.addEventListener('mouseup', handleResizeEnd);
    },
  [panelState, handleResizeMove, handleResizeEnd]);

  // 分頁
  const [tab, setTab] = useState<PanelTab>('scan');

  // 歷史相關狀態
  const [history, setHistory] = useState<TwScanSummary[]>([]);
  const [viewingScan, setViewingScan] = useState<{
    summary: TwScanSummary;
    results: TwScanResult[];
    failures: TwScanFailure[];
  } | null>(null);
  const [revisitLoading, setRevisitLoading] = useState(false);
  const [revisitData, setRevisitData] = useState<TwScanRevisitItem[] | null>(null);

  // 參數
  const [pctile, setPctile] = useState(20);
  const [markets, setMarkets] = useState<string[]>(['listed', 'otc']);
  // 進階過濾
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [minVolume, setMinVolume] = useState(200);            // 張
  const [requireHealthyTrend, setRequireHealthyTrend] = useState(true);
  const [maxAdxEnabled, setMaxAdxEnabled] = useState(false);  // BB 已含蓋盤整特徵，預設關閉
  const [persistenceBars, setPersistenceBars] = useState(3);  // 最近 N 根都壓縮
  const [minAbsBbWidth, setMinAbsBbWidth] = useState(3.0);    // 絕對 BB Width 下限 %
  const [historyDays, setHistoryDays] = useState(400);        // 抓取歷史天數（日曆日）

  // 掃描資料區間：live preview（依當前 historyDays） + 鎖定值（掃描開始時捕獲）
  const livePreviewRange = useMemo(() => {
    const end = new Date();
    const start = new Date(end.getTime() - historyDays * 86400000);
    const fmt = (d: Date) => d.toISOString().slice(0, 10);
    return { start: fmt(start), end: fmt(end), days: historyDays };
  }, [historyDays]);
  const [scanRange, setScanRange] = useState<{ start: string; end: string; days: number } | null>(null);

  // 掃描狀態
  const [scanning, setScanning] = useState(false);
  const [progress, setProgress] = useState<TwScanProgress | null>(null);
  const [results, setResults] = useState<TwScanResult[]>([]);
  const [failures, setFailures] = useState<TwScanFailure[]>([]);
  const [showFailures, setShowFailures] = useState(false);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [summary, setSummary] = useState<TwScanDone | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>('bb_width_pctile');

  const abortRef = useRef<(() => void) | null>(null);

  // 載入歷史列表（切到歷史 tab 時）
  useEffect(() => {
    if (tab !== 'history' || viewingScan) return;
    listTwScanHistory(20).then(setHistory).catch((e) => {
      console.error('載入歷史失敗', e);
    });
  }, [tab, viewingScan]);

  const handleOpenHistory = useCallback(async (s: TwScanSummary) => {
    try {
      const data = await getTwScanResult(s.scan_id);
      setViewingScan({
        summary: s,
        results: data.results,
        failures: data.failures ?? [],
      });
      setRevisitData(null);
    } catch (e) {
      toast('載入歷史結果失敗', 'error');
    }
  }, []);

  const handleRevisit = useCallback(async () => {
    if (!viewingScan) return;
    setRevisitLoading(true);
    try {
      const data = await revisitTwScan(viewingScan.summary.scan_id);
      setRevisitData(data.results);
      toast(`已取得 ${data.total_revisited}/${data.total_original} 檔當前價格`, 'success');
    } catch (e) {
      toast('回看失敗（可能需要抓取大量資料）', 'error');
    } finally {
      setRevisitLoading(false);
    }
  }, [viewingScan]);

  const handleDeleteHistory = useCallback(async (scanId: string) => {
    if (!confirm('確認要刪除這次掃描紀錄？')) return;
    try {
      await deleteTwScan(scanId);
      setHistory((prev) => prev.filter((s) => s.scan_id !== scanId));
      toast('已刪除', 'success');
    } catch {
      toast('刪除失敗', 'error');
    }
  }, []);

  const toggleMarket = (m: string) => {
    setMarkets((prev) => prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m]);
  };

  const handleStart = useCallback(() => {
    if (markets.length === 0) {
      toast('請至少選擇一個市場（上市或上櫃）', 'warning');
      return;
    }
    setScanning(true);
    setProgress(null);
    setResults([]);
    setFailures([]);
    setShowFailures(false);
    setWarnings([]);
    setSummary(null);
    setScanRange(livePreviewRange);  // 鎖定本次掃描的資料區間，避免之後改參數時被覆蓋

    // 盤中警示
    const now = new Date();
    const hhmm = now.getHours() * 100 + now.getMinutes();
    if (hhmm < 1400) {
      setWarnings(['⚠️ 當日台股尚未收盤（14:00 前），最新一根 K 線不完整，掃描結果可能不準']);
    }

    const handle = streamTwScan(
      {
        timeframe: '1d',
        pctile_threshold: pctile,
        markets,
        min_volume: minVolume,
        require_healthy_trend: requireHealthyTrend,
        max_adx: maxAdxEnabled ? 25 : null,
        persistence_bars: persistenceBars,
        min_abs_bb_width: minAbsBbWidth,
        history_days: historyDays,
      },
      {
        onProgress: (p) => setProgress(p),
        onResult: (r) => setResults((prev) => [...prev, r]),
        onFailure: (f) => setFailures((prev) => [...prev, f]),
        onWarning: (msg) => setWarnings((prev) => [...prev, msg]),
        onDone: (s) => {
          setSummary(s);
          setScanning(false);
          abortRef.current = null;
          toast(`掃描完成：找到 ${s.total_found} 檔`, 'success');
        },
        onError: (e) => {
          toast(`掃描錯誤: ${e}`, 'error');
          setScanning(false);
          abortRef.current = null;
        },
      },
    );
    abortRef.current = handle.abort;
  }, [pctile, markets, minVolume, requireHealthyTrend, maxAdxEnabled, persistenceBars, minAbsBbWidth, historyDays, livePreviewRange]);

  const handleCancel = useCallback(() => {
    abortRef.current?.();
    abortRef.current = null;
    setScanning(false);
    toast('已取消掃描', 'info');
  }, []);

  const handleClose = useCallback(() => {
    if (scanning) {
      abortRef.current?.();
      abortRef.current = null;
      setScanning(false);
    }
    setShow(false);
  }, [scanning, setShow]);

  const handleAnalyze = useCallback((r: TwScanResult) => {
    const sym = `${r.code}/TWD`;
    addSymbolsToList([sym]);
    window.dispatchEvent(new CustomEvent('symbols-updated'));
    setSymbol(sym);
    setTimeframe('1d');
    // v137：若有 Bollinger 訊號，prompt 帶 context 讓 LLM 從這個訊號開始分析
    const bs = r.bollinger_signal;
    const bbContext = bs?.label
      ? `（當前狀態：${bs.emoji} ${bs.label}，策略 ${bs.strategy}、regime ${bs.regime_used}，BB 百分位 ${r.bb_width_pctile.toFixed(1)}%）`
      : `（BB 百分位 ${r.bb_width_pctile.toFixed(1)}%）`;
    setPendingChatMessage(
      `對 ${sym} 進行完整量化研究 + 因子驗證與監控 + 完整分析三階段 ${bbContext}`
    );
    setShow(false);
    toast(`切換到 ${r.code} ${r.name}，AI 分析已送出`, 'success');
  }, [setSymbol, setTimeframe, setPendingChatMessage, setShow]);

  const handleExportCsv = useCallback(() => {
    if (results.length === 0) return;
    const headers = ['代號', '名稱', '市場', '產業', '價格', '日期', 'BB百分位(%)', 'BB寬度(%)', '5日均量(張)', '20日漲跌(%)'];
    const rows = sorted.map((r) => [
      r.code, r.name, r.market === 'listed' ? '上市' : '上櫃',
      r.industry, r.price, r.price_date,
      r.bb_width_pctile, r.bb_width, r.volume_5d_avg, r.change_20d,
    ].join(','));
    const csv = '﻿' + [headers.join(','), ...rows].join('\n');  // BOM for Excel
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `tw_bb_scan_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [results]);

  // 排序後的結果
  const sorted = [...results].sort((a, b) => {
    if (sortKey === 'bb_width_pctile') return a.bb_width_pctile - b.bb_width_pctile;
    if (sortKey === 'change_20d') return b.change_20d - a.change_20d;
    if (sortKey === 'volume_5d_avg') return b.volume_5d_avg - a.volume_5d_avg;
    return a.code.localeCompare(b.code);
  });

  if (!show) return null;

  const progressPct = progress ? Math.floor((progress.current / progress.total) * 100) : 0;

  return (
    <div
      className="fixed inset-0 z-50"
      style={{ background: 'rgba(0,0,0,0.6)' }}
      onClick={() => !scanning && handleClose()}
    >
      <div
        className="rounded-xl shadow-2xl flex flex-col"
        style={{
          background: 'var(--bg-secondary)',
          border: '1px solid var(--border-color)',
          position: 'absolute',
          left: `${panelState.left}px`,
          top: `${panelState.top}px`,
          width: `${panelState.width}px`,
          height: `${panelState.height}px`,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* 四邊 + 右下角 resize handles（localStorage 持久化大小與位置） */}
        <div
          onMouseDown={startResize('n', 'ns-resize')}
          title="往上/下拖拉以改變高度"
          style={{ position: 'absolute', top: 0, left: 8, right: 8, height: 4, cursor: 'ns-resize', zIndex: 20 }}
        />
        <div
          onMouseDown={startResize('s', 'ns-resize')}
          title="往上/下拖拉以改變高度"
          style={{ position: 'absolute', bottom: 0, left: 8, right: 8, height: 4, cursor: 'ns-resize', zIndex: 20 }}
        />
        <div
          onMouseDown={startResize('w', 'ew-resize')}
          title="往左/右拖拉以改變寬度"
          style={{ position: 'absolute', left: 0, top: 8, bottom: 8, width: 4, cursor: 'ew-resize', zIndex: 20 }}
        />
        <div
          onMouseDown={startResize('e', 'ew-resize')}
          title="往左/右拖拉以改變寬度"
          style={{ position: 'absolute', right: 0, top: 8, bottom: 8, width: 4, cursor: 'ew-resize', zIndex: 20 }}
        />
        <div
          onMouseDown={startResize('se', 'nwse-resize')}
          title="拖拉以同時調整寬高"
          style={{
            position: 'absolute',
            right: 0,
            bottom: 0,
            width: 16,
            height: 16,
            cursor: 'nwse-resize',
            zIndex: 21,
            background:
              'linear-gradient(135deg, transparent 0%, transparent 45%, var(--border-color) 45%, var(--border-color) 55%, transparent 55%, transparent 70%, var(--border-color) 70%, var(--border-color) 80%, transparent 80%)',
          }}
        />
        {/* ===== 標題列 + Tab 切換 ===== */}
        <div
          className="flex items-center justify-between px-6 py-3 border-b shrink-0"
          style={{ borderColor: 'var(--border-color)' }}
        >
          <div className="flex items-center gap-6">
            <h2 className="text-lg font-bold" style={{ color: 'var(--text-primary)' }}>
              🔍 台股掃描
            </h2>
            <div className="flex gap-1">
              <TabButton label="本次掃描" active={tab === 'scan'} onClick={() => { setTab('scan'); setViewingScan(null); }} />
              <TabButton label="📜 歷史" active={tab === 'history'} onClick={() => setTab('history')} />
              <TabButton label="📊 跨日追蹤" active={tab === 'track'} onClick={() => setTab('track')} />
            </div>
          </div>
          <button
            onClick={handleClose}
            className="text-lg cursor-pointer hover:opacity-80"
            style={{ color: 'var(--text-secondary)' }}
          >
            ✕
          </button>
        </div>

        {tab === 'history' && (
          <HistoryView
            history={history}
            viewing={viewingScan}
            onOpen={handleOpenHistory}
            onBack={() => { setViewingScan(null); setRevisitData(null); }}
            onDelete={handleDeleteHistory}
            onRevisit={handleRevisit}
            revisitLoading={revisitLoading}
            revisitData={revisitData}
            onAnalyze={handleAnalyze}
          />
        )}

        {tab === 'track' && <TrackView pctileThreshold={pctile} />}

        {tab === 'scan' && <>
        {/* ===== 控制區 ===== */}
        <div className="px-6 py-4 space-y-3 border-b" style={{ borderColor: 'var(--border-color)' }}>
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-2">
              <label className="text-sm" style={{ color: 'var(--text-secondary)' }}>壓縮強度</label>
              <select
                value={pctile}
                onChange={(e) => setPctile(Number(e.target.value))}
                disabled={scanning}
                className="px-3 py-1 rounded text-sm border-none outline-none cursor-pointer"
                style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
              >
                {PCTILE_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
            </div>

            <div className="flex items-center gap-3">
              <label className="text-sm" style={{ color: 'var(--text-secondary)' }}>市場</label>
              <label className="flex items-center gap-1 text-sm cursor-pointer" style={{ color: 'var(--text-primary)' }}>
                <input type="checkbox" checked={markets.includes('listed')}
                  onChange={() => toggleMarket('listed')} disabled={scanning} />
                上市
              </label>
              <label className="flex items-center gap-1 text-sm cursor-pointer" style={{ color: 'var(--text-primary)' }}>
                <input type="checkbox" checked={markets.includes('otc')}
                  onChange={() => toggleMarket('otc')} disabled={scanning} />
                上櫃
              </label>
            </div>

            <div className="ml-auto flex gap-2">
              {!scanning ? (
                <button
                  onClick={handleStart}
                  className="px-4 py-1.5 rounded text-sm font-medium cursor-pointer hover:opacity-90"
                  style={{ background: 'var(--accent-blue)', color: '#fff' }}
                >
                  🔍 開始掃描
                </button>
              ) : (
                <button
                  onClick={handleCancel}
                  className="px-4 py-1.5 rounded text-sm font-medium cursor-pointer hover:opacity-90"
                  style={{ background: '#dc2626', color: '#fff' }}
                >
                  ⛔ 取消
                </button>
              )}
              {results.length > 0 && !scanning && (
                <button
                  onClick={handleExportCsv}
                  className="px-4 py-1.5 rounded text-sm cursor-pointer hover:opacity-90"
                  style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
                >
                  📥 匯出 CSV
                </button>
              )}
            </div>
          </div>

          {/* 進階過濾（可折疊） */}
          <div>
            <button
              onClick={() => setShowAdvanced((v) => !v)}
              disabled={scanning}
              className="text-xs cursor-pointer hover:opacity-80"
              style={{ color: 'var(--text-secondary)' }}
            >
              {showAdvanced ? '▼' : '▶'} 進階過濾（減少誤報）
            </button>
            {showAdvanced && (
              <div className="mt-2 p-3 rounded space-y-2" style={{ background: 'var(--bg-tertiary)' }}>
                <div className="flex items-center gap-2 text-sm">
                  <label style={{ color: 'var(--text-primary)' }}>
                    壓縮持續性：最近
                  </label>
                  <input
                    type="number"
                    min={1}
                    max={10}
                    value={persistenceBars}
                    onChange={(e) => setPersistenceBars(Math.max(1, Math.min(10, Number(e.target.value) || 1)))}
                    disabled={scanning}
                    className="w-16 px-2 py-0.5 rounded border-none outline-none text-right"
                    style={{ background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}
                  />
                  <span style={{ color: 'var(--text-secondary)' }}>根 K 線都要壓縮（1 = 只看最新一根）</span>
                </div>
                <div className="flex items-center gap-2 text-sm">
                  <label style={{ color: 'var(--text-primary)' }}>
                    絕對 BB Width &ge;
                  </label>
                  <input
                    type="number"
                    step={0.5}
                    min={0}
                    value={minAbsBbWidth}
                    onChange={(e) => setMinAbsBbWidth(Math.max(0, Number(e.target.value) || 0))}
                    disabled={scanning}
                    className="w-20 px-2 py-0.5 rounded border-none outline-none text-right"
                    style={{ background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}
                  />
                  <span style={{ color: 'var(--text-secondary)' }}>%（排除常年低波動 ETF/控股；0 = 不過濾）</span>
                </div>
                <div className="flex items-center gap-2 text-sm">
                  <label style={{ color: 'var(--text-primary)' }}>
                    抓取歷史天數
                  </label>
                  <input
                    type="number"
                    min={220}
                    max={3000}
                    step={10}
                    value={historyDays}
                    onChange={(e) => setHistoryDays(Math.max(220, Math.min(3000, Number(e.target.value) || 220)))}
                    disabled={scanning}
                    className="w-24 px-2 py-0.5 rounded border-none outline-none text-right"
                    style={{ background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}
                  />
                  <span style={{ color: 'var(--text-secondary)' }}>天（最少 220，≈ 150 根交易日；新股通常沒這麼多歷史）</span>
                </div>
                <div className="flex items-center gap-2 text-sm">
                  <label style={{ color: 'var(--text-primary)' }}>
                    5 日均量 &ge;
                  </label>
                  <input
                    type="number"
                    value={minVolume}
                    onChange={(e) => setMinVolume(Math.max(0, Number(e.target.value) || 0))}
                    disabled={scanning}
                    className="w-20 px-2 py-0.5 rounded border-none outline-none text-right"
                    style={{ background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}
                  />
                  <span style={{ color: 'var(--text-secondary)' }}>張（排除冷門股）</span>
                </div>
                <label className="flex items-center gap-2 text-sm cursor-pointer" style={{ color: 'var(--text-primary)' }}>
                  <input type="checkbox" checked={requireHealthyTrend}
                    onChange={(e) => setRequireHealthyTrend(e.target.checked)} disabled={scanning} />
                  趨勢健康（MA60 斜率 &gt; 0 或 收盤 &gt; MA20 &gt; MA60）
                </label>
                <label className="flex items-center gap-2 text-sm cursor-pointer" style={{ color: 'var(--text-primary)' }}>
                  <input type="checkbox" checked={maxAdxEnabled}
                    onChange={(e) => setMaxAdxEnabled(e.target.checked)} disabled={scanning} />
                  僅保留 ADX &lt; 25 的個股（與 BB 壓縮重疊，預設關閉）
                </label>
              </div>
            )}
          </div>

          {/* 資料區間提示（live preview，掃描中改成顯示鎖定值） */}
          {!summary && (
            <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>
              📅 資料區間：<b style={{ color: 'var(--text-primary)' }}>
                {(scanning && scanRange ? scanRange : livePreviewRange).start} ~ {(scanning && scanRange ? scanRange : livePreviewRange).end}
              </b>
              （{(scanning && scanRange ? scanRange : livePreviewRange).days} 天，判斷最新一根 K 線的壓縮程度）
            </div>
          )}

          {/* 進度條 */}
          {progress && (
            <div>
              <div className="flex justify-between text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>
                <span>
                  進度：{progress.current} / {progress.total}
                  （找到 {progress.found}，失敗 {progress.fail}）
                </span>
                <span>
                  {scanning ? `預計剩餘 ${Math.floor(progress.eta_sec / 60)} 分 ${progress.eta_sec % 60} 秒` : '已完成'}
                </span>
              </div>
              <div className="rounded-full overflow-hidden" style={{ height: 6, background: 'rgba(88,166,255,0.15)' }}>
                <div
                  className="h-full rounded-full transition-all duration-300"
                  style={{ width: `${progressPct}%`, background: 'var(--accent-blue)' }}
                />
              </div>
            </div>
          )}

          {/* 警告 */}
          {warnings.length > 0 && (
            <div className="space-y-1">
              {warnings.map((w, i) => (
                <div key={i} className="text-xs px-3 py-1.5 rounded"
                  style={{ background: 'rgba(245,158,11,0.15)', color: 'var(--accent-orange, #f59e0b)' }}>
                  {w}
                </div>
              ))}
            </div>
          )}

          {/* 摘要 */}
          {summary && (
            <div className="text-sm space-y-0.5" style={{ color: 'var(--text-secondary)' }}>
              <div>
                ✅ 掃描完成：掃 {summary.total_scanned} 檔，找到 <b style={{ color: 'var(--accent-blue)' }}>{summary.total_found}</b> 檔，失敗 {summary.total_fail}，耗時 {summary.duration_sec}s
              </div>
              {scanRange && (
                <div className="text-xs">
                  📅 資料區間：{scanRange.start} ~ {scanRange.end}（{scanRange.days} 天）
                </div>
              )}
            </div>
          )}

          {/* 失敗清單（可展開） */}
          {failures.length > 0 && (
            <FailureList
              failures={failures}
              expanded={showFailures}
              onToggle={() => setShowFailures((v) => !v)}
            />
          )}
        </div>

        {/* ===== 結果表格 ===== */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          {results.length === 0 ? (
            <div className="text-center py-12" style={{ color: 'var(--text-secondary)' }}>
              {scanning ? '掃描中…找到的標的會即時顯示在這裡' : '尚未掃描，按「🔍 開始掃描」開始'}
            </div>
          ) : (
            <table className="w-full text-sm">
              <thead style={{ color: 'var(--text-secondary)' }}>
                <tr className="border-b" style={{ borderColor: 'var(--border-color)' }}>
                  <Th label="代號" active={sortKey === 'code'} onClick={() => setSortKey('code')} />
                  <th className="text-left py-2 px-2">名稱</th>
                  <th className="text-left py-2 px-2">市場</th>
                  <th className="text-left py-2 px-2">產業</th>
                  <th className="text-right py-2 px-2">價格（日期）</th>
                  <Th label="BB百分位" active={sortKey === 'bb_width_pctile'} onClick={() => setSortKey('bb_width_pctile')} right />
                  {/* v137：Bollinger 訊號欄位 */}
                  <th className="text-left py-2 px-2" title="完整布林通道訊號 (Squeeze / 突破 / Walking / 反轉)">Bollinger</th>
                  <Th label="20日漲跌" active={sortKey === 'change_20d'} onClick={() => setSortKey('change_20d')} right />
                  <Th label="5日均量" active={sortKey === 'volume_5d_avg'} onClick={() => setSortKey('volume_5d_avg')} right />
                  <th className="text-center py-2 px-2">操作</th>
                </tr>
              </thead>
              <tbody style={{ color: 'var(--text-primary)' }}>
                {sorted.map((r) => (
                  <tr key={r.code} className="border-b hover:brightness-110" style={{ borderColor: 'var(--border-color)' }}>
                    <td className="py-1.5 px-2 font-mono" style={{ color: 'var(--accent-blue)' }}>{r.code}</td>
                    <td className="py-1.5 px-2">{r.name}</td>
                    <td className="py-1.5 px-2 text-xs">{r.market === 'listed' ? '上市' : '上櫃'}</td>
                    <td className="py-1.5 px-2 text-xs">{r.industry}</td>
                    <td className="py-1.5 px-2 text-right">
                      {r.price.toFixed(2)} <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>({r.price_date})</span>
                    </td>
                    <td className="py-1.5 px-2 text-right font-semibold"
                      style={{ color: r.bb_width_pctile < 5 ? '#dc2626' : r.bb_width_pctile < 10 ? '#f59e0b' : 'var(--text-primary)' }}>
                      {r.bb_width_pctile.toFixed(1)}%
                    </td>
                    {/* v137：Bollinger 訊號 cell */}
                    <td className="py-1.5 px-2 text-xs">
                      {r.bollinger_signal?.label ? (
                        <span
                          style={{
                            background: 'rgba(63,185,80,0.15)',
                            color: '#3fb950',
                            borderRadius: '3px',
                            padding: '1px 6px',
                            fontWeight: 600,
                            whiteSpace: 'nowrap',
                          }}
                          title={`策略: ${r.bollinger_signal.strategy} | regime: ${r.bollinger_signal.regime_used}${
                            r.bollinger_signal.entry_exit?.stop
                              ? ` | SL: ${r.bollinger_signal.entry_exit.stop.toFixed(2)} TP1: ${r.bollinger_signal.entry_exit.target_1?.toFixed(2)} (RR: ${r.bollinger_signal.entry_exit.rr_1})`
                              : ''
                          }`}
                        >
                          {r.bollinger_signal.emoji} {r.bollinger_signal.label}
                        </span>
                      ) : (
                        <span style={{ color: 'var(--text-secondary)' }}>—</span>
                      )}
                    </td>
                    <td className="py-1.5 px-2 text-right"
                      style={{ color: r.change_20d > 0 ? 'var(--accent-green, #10b981)' : r.change_20d < 0 ? '#dc2626' : 'var(--text-primary)' }}>
                      {r.change_20d > 0 ? '+' : ''}{r.change_20d.toFixed(1)}%
                    </td>
                    <td className="py-1.5 px-2 text-right text-xs" style={{ color: 'var(--text-secondary)' }}>
                      {r.volume_5d_avg.toLocaleString()}
                    </td>
                    <td className="py-1.5 px-2 text-center">
                      <button
                        onClick={() => handleAnalyze(r)}
                        className="px-2 py-0.5 rounded text-xs cursor-pointer hover:opacity-80"
                        style={{ background: 'var(--accent-blue)', color: '#fff' }}
                      >
                        🔍 分析
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
        </>}
      </div>
    </div>
  );
}

function TabButton({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="px-3 py-1 text-sm cursor-pointer transition-colors"
      style={{
        color: active ? 'var(--accent-blue)' : 'var(--text-secondary)',
        borderBottom: active ? '2px solid var(--accent-blue)' : '2px solid transparent',
      }}
    >
      {label}
    </button>
  );
}
