/** 台股掃描 — 跨日追蹤分頁（v154 由 TwBBScanPanel.tsx 純搬家拆分，邏輯零改動） */

import { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import { useChartStore } from '../../stores/chartStore';
import { addSymbolsToList } from '../TopBar';
import {
  streamTwScanRange,
  downloadTwScanRangeCSV,
  exportTrackToGoogleSheet,
  unlockGoogleSheets,
  fetchTwEpsBatch,
  fetchTwTrackHistory,
  fetchTwTrackHistoryItem,
  type TwTrackHistoryItem,
  type TwTrackRangeResult,
  type TwTrackProgress,
  type TwTrackSymbol,
  type TwDailyFeatures,
  type TwEpsEntry,
} from '../../services/api';
import { toast } from '../Toast';
import { GoogleSheetSetupWizard } from './GoogleSheetSetupWizard';
import { PCTILE_OPTIONS } from './shared';

type TrackMetric = 'bb_pctile' | 'bb_width' | 'close' | 'change_20d' | 'vol_5d';
type TrackScope = 'recent_scan' | 'full_market';

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

export function TrackView({ pctileThreshold }: { pctileThreshold: number }) {
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

  // v155：追蹤歷史（結果落地）— 開面板自動載入上次結果、下拉可回顧
  const [trackHistory, setTrackHistory] = useState<TwTrackHistoryItem[]>([]);
  const [historicalAt, setHistoricalAt] = useState<string | null>(null);
  const historyLoadedRef = useRef(false);

  const loadHistoryItem = useCallback(async (trackId: string) => {
    try {
      const item = await fetchTwTrackHistoryItem(trackId);
      setResult(item.result);
      setHistoricalAt(item.result.generated_at || '（時間不明）');
      setRemovedCodes(new Set());
      setError(null);
    } catch (e) {
      toast(`載入歷史追蹤失敗：${(e as Error).message}`, 'error');
    }
  }, []);

  useEffect(() => {
    if (historyLoadedRef.current) return;
    historyLoadedRef.current = true;
    fetchTwTrackHistory(10)
      .then((r) => {
        setTrackHistory(r.items || []);
        // 面板剛開（無結果）→ 自動載入最近一筆
        if (r.items?.length) void loadHistoryItem(r.items[0].track_id);
      })
      .catch(() => { /* 無歷史或後端未升級，靜默 */ });
  }, [loadHistoryItem]);

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

    // 當次篩選條件（0 = 未啟用的省略）
    const condParts = [`BB% < ${filters.pctile}%`];
    if (filters.maxBbWidth > 0) condParts.push(`帶寬 < ${filters.maxBbWidth}%`);
    if (filters.maxClose > 0) condParts.push(`收盤 ≤ ${filters.maxClose} 元`);
    if (filters.minVol5d > 0) condParts.push(`5日均量 ≥ ${filters.minVol5d} 張`);

    // 突破現況：取區間內最後一個突破日
    let breakLine = '';
    for (const [d, f] of Object.entries(s.daily_features)) {
      if (f?.breakout === 'up') breakLine = `；已於 ${d.slice(5)} 壓縮後突破上軌`;
      else if (f?.breakout === 'down') breakLine = `；已於 ${d.slice(5)} 壓縮後跌破下軌`;
    }

    const latestLine = s.latest_close != null
      ? `最新收盤 ${s.latest_close}${s.latest_close_date ? `（${s.latest_close_date}）` : ''}`
      : '最新收盤 —';

    setPendingChatMessage(
      `對 ${sym} 進行「布林壓縮條件歷史回測報告」（口徑已明確，直接執行、不需參數確認）：\n\n` +
      `【當前狀態（跨日追蹤實測）】\n` +
      `- ${s.first_match_date} 首次符合，區間內 ${s.match_count} 日符合條件\n` +
      `- 篩選條件：${condParts.join('；')}\n` +
      `- ${latestLine}${breakLine}\n\n` +
      `【必跑統計（呼叫時帶 confirmed=true，使用全部本地數據）】\n` +
      `1. scan_conditional_probability：indicators=["vol_squeeze"]，direction=up 與 down 各跑一次，` +
      `forward_bars=6、target_pct=3 — 重點引用 BB_Width_Pctile < ${filters.pctile} 區間的機率、Wilson CI 與樣本數，對比基線機率\n` +
      `2. analyze_event_patterns：event_type=price_surge 與 price_drop 各一次，threshold=5、n_bars=1 — ` +
      `引用 event_magnitude_stats（mean/median/p25/p75）說明歷史大漲/大跌的典型幅度\n\n` +
      `【報告要求】\n` +
      `對照當前狀態與歷史統計回答：(a) 此壓縮條件歷史上向上/向下表態的機率各多少（含樣本數與 CI）；` +
      `(b) 表態後的典型幅度；(c) 目前是否已表態（突破）、對應的操作含義與風險提示。` +
      `禁止編造數字，工具沒回傳的就明說。`
    );
    setShowPanel(false);
    toast(`切換到 ${s.code} ${s.name}，條件回測報告已送出`, 'success');
  }, [filters, setSymbol, setTimeframe, setPendingChatMessage, setShowPanel]);

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
          setHistoricalAt(null);  // 新追蹤 → 不再是歷史結果
          fetchTwTrackHistory(10).then((h) => setTrackHistory(h.items || [])).catch(() => {});
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

          {trackHistory.length > 0 && (
            <select
              value=""
              onChange={(e) => { if (e.target.value) void loadHistoryItem(e.target.value); }}
              disabled={running}
              className="px-2 py-1 rounded text-sm border-none outline-none cursor-pointer"
              style={{ background: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}
              title="回顧過去的追蹤結果（自動保留最近 30 筆）"
            >
              <option value="">📜 歷史結果</option>
              {trackHistory.map((h) => (
                <option key={h.track_id} value={h.track_id}>
                  {h.generated_at}｜BB%&lt;{String(h.params.pctile_threshold)}｜符合 {h.total_matched} 檔
                </option>
              ))}
            </select>
          )}

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
            {historicalAt && (
              <span className="px-2 py-0.5 rounded text-xs"
                style={{ background: 'rgba(88,166,255,0.12)', color: 'var(--accent-blue)' }}
                title="這是先前保存的追蹤結果；按「▶ 執行追蹤」以最新資料更新">
                📜 上次結果 @{historicalAt}
              </span>
            )}
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
