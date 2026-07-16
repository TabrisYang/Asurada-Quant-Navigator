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
  streamTwScanRange,
  downloadTwScanRangeCSV,
  exportTrackToGoogleSheet,
  unlockGoogleSheets,
  listTwScanHistory,
  getTwScanResult,
  revisitTwScan,
  deleteTwScan,
  fetchTwEpsBatch,
  type TwScanResult,
  type TwScanProgress,
  type TwScanDone,
  type TwScanFailure,
  type TwScanSummary,
  type TwScanRevisitItem,
  type TwTrackRangeResult,
  type TwTrackProgress,
  type TwTrackSymbol,
  type TwDailyFeatures,
  type TwEpsEntry,
} from '../../services/api';
import { toast } from '../Toast';
import { GoogleSheetSetupWizard } from './GoogleSheetSetupWizard';

type SortKey = 'bb_width_pctile' | 'change_20d' | 'volume_5d_avg' | 'code';
type PanelTab = 'scan' | 'history' | 'track';
type TrackMetric = 'bb_pctile' | 'bb_width' | 'close' | 'change_20d' | 'vol_5d';
type TrackScope = 'recent_scan' | 'full_market';

const PCTILE_OPTIONS = [
  { value: 10, label: '強壓縮 (<10%)' },
  { value: 15, label: '中度壓縮 (<15%)' },
  { value: 20, label: '寬鬆 (<20%)' },
  { value: 25, label: '極寬鬆 (<25%)' },
];

// 視窗尺寸與位置限制 — localStorage 同時儲存 size + pos
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

function Th({ label, active, onClick, right = false }: {
  label: string; active: boolean; onClick: () => void; right?: boolean;
}) {
  return (
    <th
      onClick={onClick}
      className={`py-2 px-2 cursor-pointer select-none hover:opacity-80 ${right ? 'text-right' : 'text-left'}`}
      style={{ color: active ? 'var(--accent-blue)' : undefined }}
    >
      {label} {active ? '▼' : ''}
    </th>
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

// ─── 跨日追蹤視圖 ────────────────────────────

const METRIC_OPTIONS: { value: TrackMetric; label: string }[] = [
  { value: 'bb_pctile', label: 'BB 百分位' },
  { value: 'bb_width', label: '帶寬 %' },
  { value: 'close', label: '收盤價' },
  { value: 'change_20d', label: '20 日漲跌幅' },
  { value: 'vol_5d', label: '5 日均量（張）' },
];

// 帶寬上限選項（跨日追蹤篩選；0 = 不過濾）— 注意與掃描端 min_abs_bb_width（下限）語意相反
const BB_WIDTH_MAX_OPTIONS = [
  { value: 0, label: '不限' },
  { value: 10, label: '< 10%' },
  { value: 15, label: '< 15%' },
  { value: 20, label: '< 20%' },
  { value: 25, label: '< 25%' },
];

const TRACK_FILTER_STORAGE_KEY = 'tw_track_filters';

interface TrackFilters {
  pctile: number;
  maxBbWidth: number;
  maxClose: number;
  minVol5d: number;
}

function _loadTrackFilters(fallbackPctile: number): TrackFilters {
  const defaults: TrackFilters = { pctile: fallbackPctile, maxBbWidth: 0, maxClose: 0, minVol5d: 0 };
  try {
    const raw = localStorage.getItem(TRACK_FILTER_STORAGE_KEY);
    if (!raw) return defaults;
    const parsed = JSON.parse(raw);
    return {
      pctile: Number(parsed.pctile) || fallbackPctile,
      maxBbWidth: Number(parsed.maxBbWidth) || 0,
      maxClose: Number(parsed.maxClose) || 0,
      minVol5d: Number(parsed.minVol5d) || 0,
    };
  } catch {
    return defaults;
  }
}

function _todayISO(): string {
  return new Date().toISOString().slice(0, 10);
}
function _daysAgoISO(days: number): string {
  const d = new Date();
  d.setDate(d.getDate() - days);
  return d.toISOString().slice(0, 10);
}

function _formatMetric(metric: TrackMetric, feats: TwDailyFeatures | null | undefined): string {
  if (!feats) return '—';
  const val = feats[metric];
  if (val === null || val === undefined) return '—';
  if (metric === 'vol_5d') return Number(val).toLocaleString();
  if (metric === 'change_20d') {
    const n = Number(val);
    return `${n > 0 ? '+' : ''}${n.toFixed(2)}%`;
  }
  if (metric === 'close') return Number(val).toFixed(2);
  if (metric === 'bb_width') return `${Number(val).toFixed(2)}%`;
  return `${Number(val).toFixed(1)}%`;
}

function _metricColor(metric: TrackMetric, feats: TwDailyFeatures | null | undefined): string | undefined {
  if (!feats) return undefined;
  if (metric === 'bb_pctile' && feats.bb_pctile !== null && feats.bb_pctile !== undefined) {
    if (feats.bb_pctile < 5) return '#dc2626';
    if (feats.bb_pctile < 10) return '#f59e0b';
  }
  if (metric === 'change_20d' && feats.change_20d !== null && feats.change_20d !== undefined) {
    return feats.change_20d > 0 ? 'var(--accent-green, #10b981)' : feats.change_20d < 0 ? '#dc2626' : undefined;
  }
  return undefined;
}

function Sparkline({ symbol, metric, scanDates }: {
  symbol: TwTrackSymbol;
  metric: TrackMetric;
  scanDates: string[];
}) {
  const values: (number | null)[] = scanDates.map((d) => {
    const f = symbol.daily_features[d];
    if (!f) return null;
    const v = f[metric];
    return v === null || v === undefined ? null : Number(v);
  });
  const numeric = values.filter((v): v is number => v !== null);
  if (numeric.length < 2) return <span style={{ color: 'var(--text-secondary)' }}>—</span>;
  const min = Math.min(...numeric);
  const max = Math.max(...numeric);
  const range = max - min || 1;
  const w = 80;
  const h = 24;
  const step = values.length > 1 ? w / (values.length - 1) : 0;

  let path = '';
  values.forEach((v, i) => {
    if (v === null) return;
    const x = i * step;
    const y = h - ((v - min) / range) * h;
    path += (path === '' ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1) + ' ';
  });

  const trend = numeric[numeric.length - 1] - numeric[0];
  const color = trend > 0 ? 'var(--accent-green, #10b981)' : trend < 0 ? '#dc2626' : 'var(--text-secondary)';
  return (
    <svg width={w} height={h} style={{ display: 'block' }}>
      <path d={path} fill="none" stroke={color} strokeWidth="1.2" />
    </svg>
  );
}

// 預設回推天數：含頭含尾總共會顯示 10 個日期欄位
const DEFAULT_TRACK_LOOKBACK_DAYS = 9;

/** 跨日追蹤表格 → 匯出用 headers + rows。
 *  與畫面一致：含 EPS/本益比三欄、排除已隱藏標的。CSV 與 Google Sheet 匯出共用。 */
function buildTrackExportRows(
  result: TwTrackRangeResult,
  removedCodes: Set<string>,
  epsMap: Record<string, TwEpsEntry>,
): { headers: string[]; rows: (string | number | null)[][] } {
  const headers = ['代號', '名稱', '市場', '產業', '首次符合', '符合天數', '最新價', '最新價日期', '上年度 EPS', '最新季 EPS', '本益比'];
  for (const d of result.scan_dates) {
    headers.push(`${d}_BB%`, `${d}_帶寬%`, `${d}_收盤`, `${d}_20日漲跌%`, `${d}_5日均量`, `${d}_突破`);
  }
  const rows: (string | number | null)[][] = [];
  for (const s of result.symbols) {
    if (removedCodes.has(s.code)) continue;
    const eps = epsMap[s.code];
    const ok = eps?.quality === 'ok';
    const row: (string | number | null)[] = [
      s.code, s.name, s.market, s.industry, s.first_match_date, s.match_count,
      s.latest_close ?? null,
      s.latest_close_date ?? null,
      (ok ? eps?.annual_prev : null) ?? null,
      (ok ? eps?.quarter_latest : null) ?? null,
      (ok ? eps?.pe_ttm : null) ?? null,
    ];
    for (const d of result.scan_dates) {
      const f = s.daily_features[d];
      row.push(
        f?.bb_pctile ?? null, f?.bb_width ?? null, f?.close ?? null,
        f?.change_20d ?? null, f?.vol_5d ?? null,
        f?.breakout === 'up' ? '↑' : f?.breakout === 'down' ? '↓' : null,
      );
    }
    rows.push(row);
  }
  return { headers, rows };
}

/** 'var(--accent-green, #10b981)' / '#dc2626' → hex（Google Sheets setFontColors 用） */
function _cssColorToHex(c: string | undefined): string | null {
  if (!c) return null;
  const m = c.match(/#[0-9a-fA-F]{3,8}/);
  return m ? m[0] : null;
}

/** 匯出註記：把當時的篩選條件組成一行文字（CSV / Google Sheet 共用） */
function buildTrackFilterNote(
  filters: TrackFilters,
  startDate: string,
  endDate: string,
  scope: TrackScope,
  removedCount: number,
  result?: TwTrackRangeResult | null,
): string {
  const parts = [`BB% < ${filters.pctile}%`];
  if (filters.maxBbWidth > 0) parts.push(`帶寬 < ${filters.maxBbWidth}%`);
  if (filters.maxClose > 0) parts.push(`收盤價 ≤ ${filters.maxClose}`);
  if (filters.minVol5d > 0) parts.push(`5日均量 ≥ ${filters.minVol5d}張`);
  parts.push(`區間 ${startDate}~${endDate}`);
  parts.push(scope === 'full_market' ? '標的池：全市場' : '標的池：最近30天掃描聯集');
  if (removedCount > 0) parts.push(`已手動隱藏 ${removedCount} 檔`);
  if (result?.generated_at) parts.push(`抓取於 ${result.generated_at}`);
  if (result?.data_check && !result.data_check.passed) {
    parts.push(`⚠️ 數據檢核 ${result.data_check.n_issues} 項異常`);
  }
  return parts.join('；');
}

/** Google Sheet 走勢欄：SPARKLINE 公式（顏色沿用畫面首尾比較邏輯，漲綠跌紅平灰） */
function _buildSparklineFormula(
  s: TwTrackSymbol,
  scanDates: string[],
  metric: TrackMetric,
): string {
  const vals = scanDates
    .map((d) => s.daily_features[d]?.[metric])
    .filter((v): v is number => v !== null && v !== undefined)
    .map((v) => Number(v));
  if (vals.length < 2) return '';
  const trend = vals[vals.length - 1] - vals[0];
  const color = trend > 0 ? '#10b981' : trend < 0 ? '#dc2626' : '#8b949e';
  return `=SPARKLINE({${vals.join(',')}},{"charttype","line";"color","${color}"})`;
}

/** Google Sheet 匯出：照畫面顯示 — 每日欄為所選指標的「✅ 0.4%」文字＋字體色矩陣 */
function buildTrackSheetExport(
  result: TwTrackRangeResult,
  removedCodes: Set<string>,
  epsMap: Record<string, TwEpsEntry>,
  metric: TrackMetric,
  filterNote?: string,
): { headers: string[]; rows: (string | number | null)[][]; colors: (string | null)[][] } {
  const headers = [
    '代號', '名稱', '產業', '首次符合', '符合天數',
    `最新價${result.generated_at ? `（${result.generated_at} 抓取）` : ''}`, '走勢',
    '上年度 EPS', '最新季 EPS', '本益比',
    ...result.scan_dates.map((d) => d.slice(5)),
  ];
  const rows: (string | number | null)[][] = [];
  const colors: (string | null)[][] = [];
  for (const s of result.symbols) {
    if (removedCodes.has(s.code)) continue;
    const eps = epsMap[s.code];
    const ok = eps?.quality === 'ok';
    const row: (string | number | null)[] = [
      s.code, s.name, s.industry, s.first_match_date, s.match_count,
      s.latest_close != null
        ? `${s.latest_close}${s.latest_close_date ? ` (${s.latest_close_date.slice(5)})` : ''}`
        : null,
      _buildSparklineFormula(s, result.scan_dates, metric),
      (ok ? eps?.annual_prev : null) ?? null,
      (ok ? eps?.quarter_latest : null) ?? null,
      (ok ? eps?.pe_ttm : null) ?? null,
    ];
    const colorRow: (string | null)[] = new Array(row.length).fill(null);
    for (const d of result.scan_dates) {
      const f = s.daily_features[d];
      const text = _formatMetric(metric, f);
      const mark = f?.breakout === 'up' ? '🚀 ' : f?.breakout === 'down' ? '🔻 ' : '';
      row.push(`${mark}${f?.matched ? '✅ ' : ''}${text}`);
      // 突破日以綠/紅覆蓋指標本身的顏色
      const breakColor = f?.breakout === 'up' ? '#10b981' : f?.breakout === 'down' ? '#dc2626' : null;
      colorRow.push(breakColor ?? _cssColorToHex(_metricColor(metric, f)));
    }
    rows.push(row);
    colors.push(colorRow);
  }
  if (filterNote) {
    rows.push([`篩選條件：${filterNote}`, ...new Array(headers.length - 1).fill('')]);
    colors.push(new Array(headers.length).fill(null));
  }
  return { headers, rows, colors };
}

type TrackSortKey = 'code' | 'first_match_date' | 'match_count' | 'latest_close';

function TrackView({ pctileThreshold }: { pctileThreshold: number }) {
  const [startDate, setStartDate] = useState(() => _daysAgoISO(DEFAULT_TRACK_LOOKBACK_DAYS));
  const [endDate, setEndDate] = useState(() => _todayISO());
  const [scope, setScope] = useState<TrackScope>('recent_scan');
  const [metric, setMetric] = useState<TrackMetric>('bb_pctile');
  // 篩選條件（localStorage 記憶）：門檻預設繼承掃描分頁，其餘 0 = 不過濾
  const [filters, setFilters] = useState<TrackFilters>(() => _loadTrackFilters(pctileThreshold));
  const [showFilters, setShowFilters] = useState(false);
  // 排序（null = 後端預設序：首次符合升序、符合天數降序）
  const [sort, setSort] = useState<{ key: TrackSortKey; dir: 'asc' | 'desc' } | null>(null);
  // 數據檢核異常清單展開
  const [showDataCheck, setShowDataCheck] = useState(false);

  useEffect(() => {
    try { localStorage.setItem(TRACK_FILTER_STORAGE_KEY, JSON.stringify(filters)); } catch { /* ignore */ }
  }, [filters]);

  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<TwTrackProgress | null>(null);
  const [result, setResult] = useState<TwTrackRangeResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [csvLoading, setCsvLoading] = useState(false);
  // Google Sheet 匯出（Apps Script webhook；未設定時彈精靈，完成後自動續匯出）
  const [sheetExporting, setSheetExporting] = useState(false);
  const [lastSheetUrl, setLastSheetUrl] = useState<string | null>(null);
  const [wizardOpen, setWizardOpen] = useState(false);
  const pendingExportRef = useRef<Parameters<typeof exportTrackToGoogleSheet>[0] | null>(null);
  // v127: 用戶可點 ✕ 暫時隱藏不想追蹤的標的（重新執行追蹤會復原）
  const [removedCodes, setRemovedCodes] = useState<Set<string>>(new Set());
  // EPS 補抓：result 來後非同步補（不阻塞主流程，後端有 24h cache）
  const [epsMap, setEpsMap] = useState<Record<string, TwEpsEntry>>({});
  const [epsLoading, setEpsLoading] = useState(false);

  const abortRef = useRef<(() => void) | null>(null);

  // 點代號 → 切主圖表 + 送 AI 分析（同「本次掃描」分頁的分析按鈕行為）
  const setSymbol = useChartStore((s) => s.setSymbol);
  const setTimeframe = useChartStore((s) => s.setTimeframe);
  const setPendingChatMessage = useChartStore((s) => s.setPendingChatMessage);
  const setShowPanel = useChartStore((s) => s.setShowTwScanPanel);

  const handleAnalyzeSymbol = useCallback((s: TwTrackSymbol) => {
    const sym = `${s.code}/TWD`;
    addSymbolsToList([sym]);
    window.dispatchEvent(new CustomEvent('symbols-updated'));
    setSymbol(sym);
    setTimeframe('1d');
    const hasBreakUp = Object.values(s.daily_features).some((f) => f?.breakout === 'up');
    const ctx = `（跨日追蹤：${s.first_match_date} 首次進入布林壓縮，區間內 ${s.match_count} 日符合 BB% < ${filters.pctile}%${hasBreakUp ? '，且已出現壓縮後突破上軌' : ''}）`;
    setPendingChatMessage(`對 ${sym} 進行完整量化研究 + 因子驗證與監控 + 完整分析三階段 ${ctx}`);
    setShowPanel(false);
    toast(`切換到 ${s.code} ${s.name}，AI 分析已送出`, 'success');
  }, [filters.pctile, setSymbol, setTimeframe, setPendingChatMessage, setShowPanel]);

  // result 變動 → 重抓 EPS（後端 cache 命中時近乎即時）
  useEffect(() => {
    if (!result || result.symbols.length === 0) {
      setEpsMap({});
      return;
    }
    let cancelled = false;
    setEpsLoading(true);
    fetchTwEpsBatch(result.symbols.map((s) => s.code))
      .then((r) => { if (!cancelled) setEpsMap(r.data || {}); })
      .catch((e) => { if (!cancelled) console.warn('[TrackView] EPS 抓取失敗', e?.message); })
      .finally(() => { if (!cancelled) setEpsLoading(false); });
    return () => { cancelled = true; };
  }, [result]);

  // v127: 過濾掉用戶已刪除的標的（＋可選排序；null = 後端預設序）
  const visibleSymbols = useMemo(() => {
    let list = result ? result.symbols.filter((s) => !removedCodes.has(s.code)) : [];
    if (sort) {
      list = [...list].sort((a, b) => {
        let cmp: number;
        if (sort.key === 'match_count') cmp = a.match_count - b.match_count;
        else if (sort.key === 'first_match_date') cmp = a.first_match_date.localeCompare(b.first_match_date);
        else if (sort.key === 'latest_close') cmp = (a.latest_close ?? 0) - (b.latest_close ?? 0);
        else cmp = a.code.localeCompare(b.code);
        return sort.dir === 'asc' ? cmp : -cmp;
      });
    }
    return list;
  }, [result, removedCodes, sort]);

  // 點欄位標題排序：無 → 升冪 → 降冪 → 無
  const handleSortClick = useCallback((key: TrackSortKey) => {
    setSort((prev) => {
      if (prev?.key !== key) return { key, dir: 'asc' };
      if (prev.dir === 'asc') return { key, dir: 'desc' };
      return null;
    });
  }, []);

  const sortIndicator = (key: TrackSortKey) =>
    sort?.key === key ? (sort.dir === 'asc' ? ' ▲' : ' ▼') : '';

  const handleRemoveSymbol = useCallback((code: string) => {
    setRemovedCodes((prev) => {
      const next = new Set(prev);
      next.add(code);
      return next;
    });
  }, []);

  const handleRestoreAll = useCallback(() => {
    setRemovedCodes(new Set());
    toast('已還原所有隱藏的標的', 'info');
  }, []);

  const handleStart = useCallback(() => {
    if (!startDate || !endDate) { toast('請選擇日期範圍', 'warning'); return; }
    if (startDate > endDate) { toast('起始日期不能晚於結束日期', 'warning'); return; }

    if (scope === 'full_market') {
      const ok = confirm('全市場追蹤約 1938 檔，預估耗時 1-2 小時。\n確定執行嗎？');
      if (!ok) return;
    }

    setRunning(true);
    setProgress(null);
    setResult(null);
    setError(null);
    setRemovedCodes(new Set());  // v127: 新追蹤、清空隱藏清單

    const handle = streamTwScanRange(
      {
        start_date: startDate, end_date: endDate, scope,
        pctile_threshold: filters.pctile,
        max_abs_bb_width: filters.maxBbWidth,
        max_close: filters.maxClose,
        min_vol_5d: filters.minVol5d,
      },
      {
        onProgress: (p) => setProgress(p),
        onResult: (r) => {
          setResult(r);
          setRunning(false);
          abortRef.current = null;
          toast(`追蹤完成：${r.total_matched}/${r.total_scanned} 檔符合，耗時 ${r.duration_sec}s`, 'success');
        },
        onError: (e) => {
          setError(e);
          setRunning(false);
          abortRef.current = null;
          toast(`追蹤錯誤：${e}`, 'error');
        },
      },
    );
    abortRef.current = handle.abort;
  }, [startDate, endDate, scope, filters]);

  const handleCancel = useCallback(() => {
    abortRef.current?.();
    abortRef.current = null;
    setRunning(false);
    toast('已取消追蹤', 'info');
  }, []);

  const handleExportCsv = useCallback(async () => {
    if (!startDate || !endDate) return;
    // v127: 若用戶有隱藏標的、改用前端組 CSV（後端重跑會包含已隱藏）
    if (result && removedCodes.size > 0) {
      const { headers, rows } = buildTrackExportRows(result, removedCodes, epsMap);
      const note = buildTrackFilterNote(filters, startDate, endDate, scope, removedCodes.size, result);
      const lines = [headers.join(','), ...rows.map((r) => r.map((v) => v ?? '').join(',')), '', `篩選條件,${note}`];
      const csv = '﻿' + lines.join('\n');
      const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `tw_bb_track_${startDate}_to_${endDate}_${scope}_filtered.csv`;
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      URL.revokeObjectURL(url);
      toast(`CSV 下載完成（已排除 ${removedCodes.size} 檔隱藏標的）`, 'success');
      return;
    }
    // 無隱藏 → 用後端原始邏輯下載（重跑、體積較小）
    setCsvLoading(true);
    try {
      await downloadTwScanRangeCSV({
        start_date: startDate, end_date: endDate, scope,
        pctile_threshold: filters.pctile,
        max_abs_bb_width: filters.maxBbWidth,
        max_close: filters.maxClose,
        min_vol_5d: filters.minVol5d,
      });
      toast('CSV 下載完成', 'success');
    } catch (e) {
      toast(`匯出失敗：${(e as Error).message}`, 'error');
    } finally {
      setCsvLoading(false);
    }
  }, [startDate, endDate, scope, filters, result, removedCodes, epsMap]);

  const doSheetExport = useCallback(async (payload: Parameters<typeof exportTrackToGoogleSheet>[0]) => {
    const detailOf = (e: unknown) =>
      (e as { response?: { data?: { detail?: { code?: string; message?: string } | string } } })
        ?.response?.data?.detail;
    const msgOf = (e: unknown) => {
      const d = detailOf(e);
      return typeof d === 'string' ? d : d?.message || (e as Error).message;
    };
    setSheetExporting(true);
    try {
      // 最多兩輪：第一輪撞到「後端重啟需重輸密碼」→ 解鎖後第二輪重試
      for (let attempt = 0; attempt < 2; attempt++) {
        try {
          const r = await exportTrackToGoogleSheet(payload);
          setLastSheetUrl(r.sheet_url);
          toast(`已寫入分頁「${r.sheet_title}」（${payload.rows.length} 檔）`, 'success');
          return;
        } catch (e) {
          const detail = detailOf(e);
          const code = typeof detail === 'object' ? detail?.code : undefined;
          if (code === 'gsheet_not_configured') {
            // 首次使用：開設定精靈，完成後自動續做這筆匯出
            pendingExportRef.current = payload;
            setWizardOpen(true);
            return;
          }
          if (code === 'gsheet_password_required' && attempt === 0) {
            const pw = window.prompt('後端重啟過，請重新輸入匯出密碼（僅存記憶體、不寫入磁碟）：');
            if (!pw) { toast('已取消匯出', 'info'); return; }
            try {
              await unlockGoogleSheets(pw);
              continue;  // 解鎖成功 → 重試匯出
            } catch (e2) {
              toast(`密碼驗證失敗：${msgOf(e2)}`, 'error');
              return;
            }
          }
          toast(`匯出失敗：${msgOf(e)}`, 'error');
          return;
        }
      }
    } finally {
      setSheetExporting(false);
    }
  }, []);

  const handleExportGoogleSheet = useCallback(() => {
    if (!result) return;
    const note = buildTrackFilterNote(filters, startDate, endDate, scope, removedCodes.size, result);
    const { headers, rows, colors } = buildTrackSheetExport(result, removedCodes, epsMap, metric, note);
    if (rows.length === 0) { toast('沒有可匯出的標的', 'warning'); return; }
    const url = window.prompt(
      '貼上目標 Google 試算表網址（會在其中自動新增一個分頁）：',
      localStorage.getItem('tw_track_gsheet_url') ?? '',
    );
    if (!url?.trim()) return;
    localStorage.setItem('tw_track_gsheet_url', url.trim());
    const metricLabel = METRIC_OPTIONS.find((o) => o.value === metric)?.label ?? metric;
    void doSheetExport({
      spreadsheet_url: url.trim(),
      sheet_title: `跨日追蹤 ${startDate}~${endDate}（${metricLabel}）`,
      headers,
      rows,
      colors,
    });
  }, [result, removedCodes, epsMap, metric, filters, scope, startDate, endDate, doSheetExport]);

  const handleWizardComplete = useCallback(() => {
    setWizardOpen(false);
    const pending = pendingExportRef.current;
    pendingExportRef.current = null;
    if (pending) void doSheetExport(pending);
  }, [doSheetExport]);

  const progressPct = progress ? Math.floor((progress.current / progress.total) * 100) : 0;

  return (
    <>
      <div className="px-6 py-4 space-y-3 border-b shrink-0" style={{ borderColor: 'var(--border-color)' }}>
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <label className="text-sm" style={{ color: 'var(--text-secondary)' }}>標的池</label>
            <select
              value={scope}
              onChange={(e) => setScope(e.target.value as TrackScope)}
              disabled={running}
              className="px-3 py-1 rounded text-sm border-none outline-none cursor-pointer"
              style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
            >
              <option value="recent_scan">最近 30 天掃描聯集（~30-70 檔，~30 秒）</option>
              <option value="full_market">全市場 1938 檔（~2 小時）</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            <label className="text-sm" style={{ color: 'var(--text-secondary)' }}>顯示指標</label>
            <select
              value={metric}
              onChange={(e) => setMetric(e.target.value as TrackMetric)}
              className="px-3 py-1 rounded text-sm border-none outline-none cursor-pointer"
              style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
            >
              {METRIC_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-2">
            <label className="text-sm" style={{ color: 'var(--text-secondary)' }}>壓縮強度</label>
            <select
              value={filters.pctile}
              onChange={(e) => setFilters((f) => ({ ...f, pctile: Number(e.target.value) }))}
              disabled={running}
              className="px-3 py-1 rounded text-sm border-none outline-none cursor-pointer"
              style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
            >
              {PCTILE_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>

          <button
            onClick={() => setShowFilters((v) => !v)}
            className="px-3 py-1 rounded text-sm cursor-pointer hover:opacity-90"
            style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
          >
            {showFilters ? '▼' : '▶'} 進階篩選
            {(filters.maxBbWidth > 0 || filters.maxClose > 0 || filters.minVol5d > 0) && (
              <span style={{ color: 'var(--accent-blue)' }}> ●</span>
            )}
          </button>

          <div className="flex items-center gap-2">
            <label className="text-sm" style={{ color: 'var(--text-secondary)' }}>日期範圍</label>
            <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} disabled={running}
              className="px-2 py-1 rounded text-sm border-none outline-none"
              style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }} />
            <span style={{ color: 'var(--text-secondary)' }}>~</span>
            <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} disabled={running}
              className="px-2 py-1 rounded text-sm border-none outline-none"
              style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }} />
          </div>

          <div className="ml-auto flex gap-2">
            {!running ? (
              <button
                onClick={handleStart}
                className="px-4 py-1.5 rounded text-sm font-medium cursor-pointer hover:opacity-90"
                style={{ background: 'var(--accent-blue)', color: '#fff' }}
              >
                ▶ 執行追蹤
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
            <button
              onClick={handleExportCsv}
              disabled={running || csvLoading}
              className="px-4 py-1.5 rounded text-sm cursor-pointer hover:opacity-90"
              style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)', opacity: (running || csvLoading) ? 0.6 : 1 }}
            >
              {csvLoading ? '匯出中…' : '📥 匯出 CSV'}
            </button>
            <button
              onClick={handleExportGoogleSheet}
              disabled={running || sheetExporting || !result}
              className="px-4 py-1.5 rounded text-sm cursor-pointer hover:opacity-90"
              style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)', opacity: (running || sheetExporting || !result) ? 0.6 : 1 }}
              title="把目前表格（含 EPS 欄、排除已隱藏標的）匯出到 Google 試算表的新分頁"
            >
              {sheetExporting ? '匯出中…' : '📤 匯出 Google Sheet'}
            </button>
          </div>
        </div>

        {showFilters && (
          <div className="rounded px-4 py-3 space-y-2 text-sm" style={{ background: 'var(--bg-tertiary)' }}>
            <div className="flex items-center gap-2">
              <label className="w-32 shrink-0" style={{ color: 'var(--text-secondary)' }}>帶寬上限</label>
              <select
                value={filters.maxBbWidth}
                onChange={(e) => setFilters((f) => ({ ...f, maxBbWidth: Number(e.target.value) }))}
                disabled={running}
                className="px-2 py-1 rounded border-none outline-none cursor-pointer"
                style={{ background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}
              >
                {BB_WIDTH_MAX_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
              <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                ✅ 需同時滿足帶寬（(2×std/SMA)×100）低於此值
              </span>
            </div>
            <div className="flex items-center gap-2">
              <label className="w-32 shrink-0" style={{ color: 'var(--text-secondary)' }}>收盤價上限</label>
              <input
                type="number" min={0} step={10}
                value={filters.maxClose || ''}
                placeholder="不過濾"
                onChange={(e) => setFilters((f) => ({ ...f, maxClose: Number(e.target.value) || 0 }))}
                disabled={running}
                className="w-28 px-2 py-1 rounded border-none outline-none"
                style={{ background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}
              />
              <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                元；排除高價股（用最新收盤價判定整檔）
              </span>
            </div>
            <div className="flex items-center gap-2">
              <label className="w-32 shrink-0" style={{ color: 'var(--text-secondary)' }}>5 日均量下限</label>
              <input
                type="number" min={0} step={100}
                value={filters.minVol5d || ''}
                placeholder="不過濾"
                onChange={(e) => setFilters((f) => ({ ...f, minVol5d: Number(e.target.value) || 0 }))}
                disabled={running}
                className="w-28 px-2 py-1 rounded border-none outline-none"
                style={{ background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}
              />
              <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                張；排除流動性不足（用最新 5 日均量判定整檔）
              </span>
            </div>
            <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>
              條件變更後請重新按「▶ 執行追蹤」生效（本地計算、約 1 秒內）
            </div>
          </div>
        )}

        <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>
          ✅ 表示該日 BB% &lt; {filters.pctile}%
          {filters.maxBbWidth > 0 && <> 且帶寬 &lt; {filters.maxBbWidth}%</>}
          （符合「壓縮強度」條件）；🚀/🔻 = 壓縮後首次突破上軌/跌破下軌；點代號可直接切圖表＋AI 分析
        </div>

        {progress && running && (
          <div>
            <div className="flex justify-between text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>
              <span>進度：{progress.current} / {progress.total}（{progress.phase}）</span>
              <span>{progressPct}%</span>
            </div>
            <div className="rounded-full overflow-hidden" style={{ height: 6, background: 'rgba(88,166,255,0.15)' }}>
              <div className="h-full rounded-full transition-all duration-300" style={{ width: `${progressPct}%`, background: 'var(--accent-blue)' }} />
            </div>
          </div>
        )}

        {error && (
          <div className="text-xs px-3 py-1.5 rounded"
            style={{ background: 'rgba(220,38,38,0.15)', color: '#dc2626' }}>
            {error}
          </div>
        )}

        {result && !running && (
          <div className="text-sm flex items-center gap-2 flex-wrap" style={{ color: 'var(--text-secondary)' }}>
            <span>
              ✅ 追蹤完成：掃 <b style={{ color: 'var(--text-primary)' }}>{result.total_scanned}</b> 檔，
              符合 <b style={{ color: 'var(--accent-blue)' }}>{result.total_matched}</b> 檔，
              日期數 <b style={{ color: 'var(--text-primary)' }}>{result.scan_dates.length}</b>，
              耗時 {result.duration_sec}s
            </span>
            {removedCodes.size > 0 && (
              <>
                <span style={{ color: 'var(--accent-orange, #f59e0b)' }}>
                  ｜已隱藏 <b>{removedCodes.size}</b> 檔，顯示 <b>{visibleSymbols.length}</b> 檔
                </span>
                <button
                  onClick={handleRestoreAll}
                  className="px-2 py-0.5 rounded text-xs cursor-pointer hover:opacity-90"
                  style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
                  title="把所有隱藏的標的還原回來"
                >
                  🔄 還原全部
                </button>
              </>
            )}
            {lastSheetUrl && (
              <a
                href={lastSheetUrl} target="_blank" rel="noreferrer"
                className="text-xs hover:underline"
                style={{ color: 'var(--accent-blue)' }}
              >
                ↗ 開啟 Google Sheet
              </a>
            )}
            {result.data_check && (
              result.data_check.passed ? (
                <span className="text-xs" style={{ color: 'var(--accent-green, #10b981)' }}
                  title={`已檢查 ${result.data_check.n_checks} 項：數值範圍、單日跳動（台股 ±10% 漲跌停）、缺漏率、資料時效`}>
                  🧪 數據檢核：✅ 通過（{result.data_check.n_checks} 項）
                </span>
              ) : (
                <button
                  onClick={() => setShowDataCheck((v) => !v)}
                  className="text-xs px-2 py-0.5 rounded cursor-pointer hover:opacity-90"
                  style={{ background: 'rgba(245,158,11,0.15)', color: 'var(--accent-orange, #f59e0b)' }}
                  title="點擊展開/收合異常明細"
                >
                  🧪 數據檢核：⚠️ {result.data_check.n_issues} 項異常 {showDataCheck ? '▲' : '▼'}
                </button>
              )
            )}
          </div>
        )}

        {result && !running && showDataCheck && result.data_check && !result.data_check.passed && (
          <div className="text-xs px-3 py-2 rounded space-y-0.5"
            style={{ background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.25)', color: 'var(--text-primary)' }}>
            {result.data_check.issues.map((issue, i) => (
              <div key={i}>• {issue}</div>
            ))}
            <div style={{ color: 'var(--text-secondary)' }}>
              異常標的的數值請以官方來源複核；「單日跳動超過漲跌停」多為除權息未還原，該檔的 BB%/漲跌幅可能失真
            </div>
          </div>
        )}
      </div>

      <div className="min-h-0 min-w-0 overflow-auto px-6 py-4">
        {!result ? (
          <div className="text-center py-12" style={{ color: 'var(--text-secondary)' }}>
            {running ? '追蹤中… 結果會在完成後一次顯示' : '尚未追蹤。選好日期範圍 + 標的池後按「執行追蹤」'}
          </div>
        ) : visibleSymbols.length === 0 ? (
          <div className="text-center py-12" style={{ color: 'var(--text-secondary)' }}>
            {removedCodes.size > 0
              ? `所有 ${removedCodes.size} 檔標的都被隱藏了，點上方「🔄 還原全部」復原`
              : `此區間內無任何標的符合 BB% < ${filters.pctile}%${filters.maxBbWidth > 0 ? `（且帶寬 < ${filters.maxBbWidth}%）` : ''} 條件`}
          </div>
        ) : (
          <table className="text-sm tw-track-table" style={{ minWidth: '100%' }}>
            <thead style={{ color: 'var(--text-secondary)' }}>
              <tr className="border-b" style={{ borderColor: 'var(--border-color)' }}>
                {/* v127: 操作欄（X 刪除按鈕）。凍結欄寬度固定，left 由寬度推導（32/60/110） */}
                <th className="text-center py-2 px-1 sticky left-0 top-0 z-30" style={{ background: 'var(--bg-secondary)', width: '32px', minWidth: '32px', maxWidth: '32px' }}></th>
                <th
                  className="text-left py-2 px-2 sticky top-0 z-30 cursor-pointer hover:opacity-80 select-none"
                  style={{ background: 'var(--bg-secondary)', left: '32px', width: '60px', minWidth: '60px', maxWidth: '60px' }}
                  onClick={() => handleSortClick('code')}
                  title="點擊依代號排序"
                >代號{sortIndicator('code')}</th>
                <th className="text-left py-2 px-2 sticky top-0 z-30" style={{ background: 'var(--bg-secondary)', left: '92px', width: '110px', minWidth: '110px', maxWidth: '110px' }}>名稱</th>
                <th className="text-left py-2 px-2 whitespace-nowrap sticky top-0 z-20" style={{ background: 'var(--bg-secondary)' }}>產業</th>
                <th
                  className="text-left py-2 px-2 cursor-pointer hover:opacity-80 select-none sticky top-0 z-20"
                  style={{ background: 'var(--bg-secondary)' }}
                  onClick={() => handleSortClick('first_match_date')}
                  title="點擊依首次符合日排序"
                >首次符合{sortIndicator('first_match_date')}</th>
                <th
                  className="text-center py-2 px-2 cursor-pointer hover:opacity-80 select-none sticky top-0 z-20"
                  style={{ background: 'var(--bg-secondary)' }}
                  onClick={() => handleSortClick('match_count')}
                  title="點擊依符合天數排序"
                >符合天數{sortIndicator('match_count')}</th>
                <th
                  className="text-right py-2 px-2 whitespace-nowrap cursor-pointer hover:opacity-80 select-none sticky top-0 z-20"
                  style={{ background: 'var(--bg-secondary)' }}
                  onClick={() => handleSortClick('latest_close')}
                  title="範圍內最新一根日 K 的收盤價（盤中通常是前一交易日）。點擊可排序"
                >
                  最新價{sortIndicator('latest_close')}
                  {result.generated_at && (
                    <div className="text-[10px] font-normal" style={{ color: 'var(--text-secondary)' }}>
                      {result.generated_at} 抓取
                    </div>
                  )}
                </th>
                <th className="text-center py-2 px-2 sticky top-0 z-20" style={{ background: 'var(--bg-secondary)' }}>走勢</th>
                <th
                  className="text-right py-2 px-2 whitespace-nowrap sticky top-0 z-20"
                  style={{ background: 'var(--bg-secondary)' }}
                  title="最近一份完整年報的 Basic EPS（4 季加總、單位 NTD）。主源 FinMind v4 即時季報，失敗才走 yfinance；hover 個格可看具體年份、來源與抓取時間"
                >
                  上年度 EPS{epsLoading && <span style={{ marginLeft: 4, color: 'var(--text-secondary)' }}>…</span>}
                </th>
                <th
                  className="text-right py-2 px-2 whitespace-nowrap sticky top-0 z-20"
                  style={{ background: 'var(--bg-secondary)' }}
                  title="最新一季 Basic EPS（單季實際值，非 TTM 累計）。主源 FinMind"
                >
                  最新季 EPS
                </th>
                <th
                  className="text-right py-2 px-2 whitespace-nowrap sticky top-0 z-20"
                  style={{ background: 'var(--bg-secondary)' }}
                  title="當前本益比（PER）。主源 FinMind 即時 PER，失敗 fallback yfinance trailingPE"
                >
                  本益比
                </th>
                {result.scan_dates.map((d) => (
                  <th key={d} className="text-right py-2 px-2 whitespace-nowrap sticky top-0 z-20" style={{ background: 'var(--bg-secondary)' }}>{d.slice(5)}</th>
                ))}
              </tr>
            </thead>
            <tbody style={{ color: 'var(--text-primary)' }}>
              {visibleSymbols.map((s) => (
                <tr key={s.code} className="border-b hover:brightness-110 group" style={{ borderColor: 'var(--border-color)' }}>
                  {/* v127: X 刪除按鈕（hover 整行才明顯） */}
                  <td className="py-1.5 px-1 text-center sticky left-0 z-10" style={{ background: 'var(--bg-secondary)', width: '32px', minWidth: '32px', maxWidth: '32px' }}>
                    <button
                      onClick={() => handleRemoveSymbol(s.code)}
                      className="px-1.5 py-0.5 rounded text-xs cursor-pointer opacity-30 hover:opacity-100 hover:bg-red-500/20"
                      style={{ color: '#dc2626' }}
                      title={`隱藏 ${s.code} ${s.name}（暫時、重跑復原）`}
                    >
                      ✕
                    </button>
                  </td>
                  <td
                    className="py-1.5 px-2 font-mono sticky z-10 cursor-pointer hover:underline"
                    style={{ color: 'var(--accent-blue)', background: 'var(--bg-secondary)', left: '32px', width: '60px', minWidth: '60px', maxWidth: '60px' }}
                    onClick={() => handleAnalyzeSymbol(s)}
                    title={`點擊切換主圖表到 ${s.code} 並送出 AI 完整分析`}
                  >{s.code}</td>
                  <td
                    className="py-1.5 px-2 sticky z-10 whitespace-nowrap"
                    style={{ background: 'var(--bg-secondary)', left: '92px', width: '110px', minWidth: '110px', maxWidth: '110px', overflow: 'hidden', textOverflow: 'ellipsis' }}
                    title={s.name}
                  >{s.name}</td>
                  <td className="py-1.5 px-2 whitespace-nowrap text-xs" style={{ color: 'var(--text-secondary)' }}>{s.industry || '—'}</td>
                  <td className="py-1.5 px-2 whitespace-nowrap text-xs">
                    {s.first_match_date}
                    {s.first_match_date === result.scan_dates[0] && <span title="範圍內首日就符合"> ✨</span>}
                  </td>
                  <td className="py-1.5 px-2 text-center">{s.match_count}</td>
                  <td className="py-1.5 px-2 text-right whitespace-nowrap">
                    {s.latest_close != null ? s.latest_close.toFixed(2) : '—'}
                    {s.latest_close_date && (
                      <span className="text-xs" style={{ color: 'var(--text-secondary)' }}> ({s.latest_close_date.slice(5)})</span>
                    )}
                  </td>
                  <td className="py-1.5 px-2 text-center">
                    <Sparkline symbol={s} metric={metric} scanDates={result.scan_dates} />
                  </td>
                  {(() => {
                    const eps = epsMap[s.code];
                    const ok = eps?.quality === 'ok';
                    const annual = ok ? eps?.annual_prev : null;
                    const quarter = ok ? eps?.quarter_latest : null;
                    const pe = eps?.pe_ttm;
                    const sourceTag = eps?.data_source && eps.data_source !== 'none'
                      ? eps.data_source.toUpperCase() : '';
                    const asOf = eps?.as_of ? ` · 抓取於 ${eps.as_of.slice(0, 10)}` : '';
                    // EPS 抓取失敗時的 tooltip（後端失敗只快取 10 分鐘，會自動重試）
                    const missTip = 'EPS 暫時抓取失敗（來源額度用盡或無資料），約 10 分鐘後自動重試';
                    const annualTip = eps?.annual_prev_label
                      ? `${eps.annual_prev_label} 年報${sourceTag ? ' · ' + sourceTag : ''}${asOf}`
                      : missTip;
                    const quarterTip = eps?.quarter_latest_label
                      ? `${eps.quarter_latest_label} 季報${sourceTag ? ' · ' + sourceTag : ''}${asOf}`
                      : missTip;
                    return (
                      <>
                        <td className="py-1.5 px-2 text-right whitespace-nowrap text-xs" title={annualTip}>
                          {typeof annual === 'number' ? annual.toFixed(2) : '—'}
                        </td>
                        <td className="py-1.5 px-2 text-right whitespace-nowrap text-xs" title={quarterTip}>
                          {typeof quarter === 'number' ? quarter.toFixed(2) : '—'}
                        </td>
                        <td className="py-1.5 px-2 text-right whitespace-nowrap text-xs"
                          title={typeof pe === 'number' ? 'TTM 本益比（yfinance）' : missTip}>
                          {typeof pe === 'number' ? pe.toFixed(2) : '—'}
                        </td>
                      </>
                    );
                  })()}
                  {result.scan_dates.map((d) => {
                    const feats = s.daily_features[d];
                    const text = _formatMetric(metric, feats);
                    const color = _metricColor(metric, feats);
                    return (
                      <td key={d} className="py-1.5 px-2 text-right whitespace-nowrap" style={{ color }}>
                        {feats?.breakout === 'up' && <span title="壓縮後首次突破上軌">🚀</span>}
                        {feats?.breakout === 'down' && <span title="壓縮後首次跌破下軌">🔻</span>}
                        {feats?.matched && <span style={{ color: 'var(--accent-green, #10b981)' }}>✅</span>}
                        {(feats?.matched || feats?.breakout) ? ' ' : ''}{text}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {wizardOpen && (
        <GoogleSheetSetupWizard
          onComplete={handleWizardComplete}
          onClose={() => { setWizardOpen(false); pendingExportRef.current = null; }}
        />
      )}
    </>
  );
}

function FailureList({ failures, expanded, onToggle }: {
  failures: TwScanFailure[];
  expanded: boolean;
  onToggle: () => void;
}) {
  return (
    <div
      className="rounded text-xs"
      style={{ background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.25)' }}
    >
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between px-3 py-2 cursor-pointer hover:opacity-80 text-left"
        style={{ color: 'var(--accent-orange, #f59e0b)' }}
      >
        <span>{expanded ? '▼' : '▶'}  失敗 {failures.length} 檔（點擊{expanded ? '收合' : '展開'}）</span>
        <span style={{ color: 'var(--text-secondary)' }}>資料不足 / yfinance 抓取失敗</span>
      </button>
      {expanded && (
        <div className="max-h-56 overflow-y-auto px-3 pb-3">
          <table className="w-full">
            <thead style={{ color: 'var(--text-secondary)' }}>
              <tr className="border-b" style={{ borderColor: 'var(--border-color)' }}>
                <th className="text-left py-1">代號</th>
                <th className="text-left py-1">名稱</th>
                <th className="text-left py-1">市場</th>
                <th className="text-left py-1">產業</th>
                <th className="text-left py-1">失敗原因</th>
              </tr>
            </thead>
            <tbody style={{ color: 'var(--text-primary)' }}>
              {failures.map((f) => (
                <tr key={f.code} className="border-b" style={{ borderColor: 'var(--border-color)' }}>
                  <td className="py-1 font-mono" style={{ color: 'var(--accent-blue)' }}>{f.code}</td>
                  <td className="py-1">{f.name}</td>
                  <td className="py-1">{f.market === 'listed' ? '上市' : '上櫃'}</td>
                  <td className="py-1">{f.industry || '—'}</td>
                  <td className="py-1" style={{ color: 'var(--text-secondary)' }}>{f.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ─── 歷史視圖（Phase 4）────────────────────────────

interface HistoryViewProps {
  history: TwScanSummary[];
  viewing: { summary: TwScanSummary; results: TwScanResult[]; failures: TwScanFailure[] } | null;
  onOpen: (s: TwScanSummary) => void;
  onBack: () => void;
  onDelete: (scanId: string) => void;
  onRevisit: () => void;
  revisitLoading: boolean;
  revisitData: TwScanRevisitItem[] | null;
  onAnalyze: (r: TwScanResult) => void;
}

function HistoryView({ history, viewing, onOpen, onBack, onDelete, onRevisit, revisitLoading, revisitData, onAnalyze }: HistoryViewProps) {
  const [showHistoryFailures, setShowHistoryFailures] = useState(false);

  if (!viewing) {
    return (
      <div className="flex-1 overflow-y-auto px-6 py-4">
        {history.length === 0 ? (
          <div className="text-center py-12" style={{ color: 'var(--text-secondary)' }}>
            尚無掃描歷史。執行一次掃描後就會出現在這裡。
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead style={{ color: 'var(--text-secondary)' }}>
              <tr className="border-b" style={{ borderColor: 'var(--border-color)' }}>
                <th className="text-left py-2 px-2">時間</th>
                <th className="text-left py-2 px-2">門檻</th>
                <th className="text-right py-2 px-2">掃描 / 找到</th>
                <th className="text-right py-2 px-2">耗時</th>
                <th className="text-center py-2 px-2">操作</th>
              </tr>
            </thead>
            <tbody style={{ color: 'var(--text-primary)' }}>
              {history.map((s) => (
                <tr key={s.scan_id} className="border-b hover:brightness-110" style={{ borderColor: 'var(--border-color)' }}>
                  <td className="py-1.5 px-2 font-mono text-xs">{s.scanned_at.replace('T', ' ')}</td>
                  <td className="py-1.5 px-2 text-xs">
                    BB&lt;{(s.params.pctile_threshold as number) ?? '?'}%，{s.timeframe}
                  </td>
                  <td className="py-1.5 px-2 text-right">
                    {s.total_scanned} / <b style={{ color: 'var(--accent-blue)' }}>{s.total_found}</b>
                  </td>
                  <td className="py-1.5 px-2 text-right text-xs">{s.duration_sec.toFixed(1)}s</td>
                  <td className="py-1.5 px-2 text-center">
                    <button onClick={() => onOpen(s)}
                      className="px-2 py-0.5 rounded text-xs cursor-pointer hover:opacity-80 mr-1"
                      style={{ background: 'var(--accent-blue)', color: '#fff' }}>
                      檢視
                    </button>
                    <button onClick={() => onDelete(s.scan_id)}
                      className="px-2 py-0.5 rounded text-xs cursor-pointer hover:opacity-80"
                      style={{ background: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}>
                      刪除
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    );
  }

  // 檢視某次歷史
  return (
    <>
      <div className="px-6 py-3 border-b flex items-center justify-between" style={{ borderColor: 'var(--border-color)' }}>
        <div>
          <button onClick={onBack} className="text-sm cursor-pointer hover:opacity-80" style={{ color: 'var(--text-secondary)' }}>
            ← 返回歷史列表
          </button>
          <div className="text-sm mt-1" style={{ color: 'var(--text-primary)' }}>
            掃描時間：<b>{viewing.summary.scanned_at.replace('T', ' ')}</b>　|　找到 {viewing.summary.total_found} 檔
          </div>
        </div>
        <button onClick={onRevisit} disabled={revisitLoading}
          className="px-3 py-1.5 rounded text-sm cursor-pointer hover:opacity-90"
          style={{ background: 'var(--accent-green, #10b981)', color: '#fff', opacity: revisitLoading ? 0.6 : 1 }}>
          {revisitLoading ? '取當前價中…' : '📊 回看：後續漲跌幅'}
        </button>
      </div>
      {viewing.failures.length > 0 && (
        <div className="px-6 pt-3">
          <FailureList
            failures={viewing.failures}
            expanded={showHistoryFailures}
            onToggle={() => setShowHistoryFailures((v) => !v)}
          />
        </div>
      )}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        <table className="w-full text-sm">
          <thead style={{ color: 'var(--text-secondary)' }}>
            <tr className="border-b" style={{ borderColor: 'var(--border-color)' }}>
              <th className="text-left py-2 px-2">代號</th>
              <th className="text-left py-2 px-2">名稱</th>
              <th className="text-left py-2 px-2">產業</th>
              <th className="text-right py-2 px-2">當時價格</th>
              {revisitData && <th className="text-right py-2 px-2">現在價格</th>}
              {revisitData && <th className="text-right py-2 px-2">後續漲跌</th>}
              <th className="text-right py-2 px-2">BB百分位</th>
              <th className="text-center py-2 px-2">操作</th>
            </tr>
          </thead>
          <tbody style={{ color: 'var(--text-primary)' }}>
            {(revisitData ?? viewing.results).map((r) => {
              const rv = revisitData ? (r as TwScanRevisitItem) : null;
              return (
                <tr key={r.code} className="border-b hover:brightness-110" style={{ borderColor: 'var(--border-color)' }}>
                  <td className="py-1.5 px-2 font-mono" style={{ color: 'var(--accent-blue)' }}>{r.code}</td>
                  <td className="py-1.5 px-2">{r.name}</td>
                  <td className="py-1.5 px-2 text-xs">{r.industry}</td>
                  <td className="py-1.5 px-2 text-right">
                    {r.price.toFixed(2)} <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>({r.price_date})</span>
                  </td>
                  {rv && (
                    <td className="py-1.5 px-2 text-right">
                      {rv.current_price.toFixed(2)} <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>({rv.current_date})</span>
                    </td>
                  )}
                  {rv && (
                    <td className="py-1.5 px-2 text-right font-semibold"
                      style={{ color: rv.return_pct > 0 ? 'var(--accent-green, #10b981)' : rv.return_pct < 0 ? '#dc2626' : 'var(--text-primary)' }}>
                      {rv.return_pct > 0 ? '+' : ''}{rv.return_pct.toFixed(1)}%
                    </td>
                  )}
                  <td className="py-1.5 px-2 text-right">{r.bb_width_pctile.toFixed(1)}%</td>
                  <td className="py-1.5 px-2 text-center">
                    <button onClick={() => onAnalyze(r)}
                      className="px-2 py-0.5 rounded text-xs cursor-pointer hover:opacity-80"
                      style={{ background: 'var(--accent-blue)', color: '#fff' }}>
                      🔍 分析
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}
