/** 阿斯拉量化系統 — K 線圖表組件 (lightweight-charts v5)
 *
 * 支援：
 * - K 線 + 成交量主圖
 * - overlay 指標（SMA、EMA、BB 等）疊在主圖上
 * - sub_chart 指標（RSI、MACD、ATR 等）獨立副圖渲染在下方
 * - 所有圖表時間軸同步
 */

import { useState, useEffect, useRef, useCallback, useMemo, memo } from 'react';
import {
  createChart,
  ColorType,
  CandlestickSeries,
  HistogramSeries,
  LineSeries,
  type IChartApi,
  type ISeriesApi,
} from 'lightweight-charts';
import { useChartStore } from '../../stores/chartStore';
import type { ActiveIndicator, OHLCVData } from '../../types';

// 時間範圍同步用（基於時間戳而非索引，確保不同數據量的圖表也能精確對齊）
interface TimeBasedRange {
  from: number;
  to: number;
}

// 根據價格自動計算合適的小數精度（BTC→2位, ADA→6位, SHIB→8位）
function getAutoPriceFormat(price: number): { type: 'price'; precision: number; minMove: number } {
  if (price >= 1000)  return { type: 'price', precision: 2, minMove: 0.01 };
  if (price >= 1)     return { type: 'price', precision: 4, minMove: 0.0001 };
  if (price >= 0.01)  return { type: 'price', precision: 6, minMove: 0.000001 };
  return { type: 'price', precision: 8, minMove: 0.00000001 };
}

// ─── 指標顏色映射 ──────────────────────────────────
const INDICATOR_COLORS: Record<string, string[]> = {
  sma:      ['#ff9800'],
  ema:      ['#2196f3'],
  bb:       ['#e91e63', '#9c27b0', '#e91e63', '#ff5722'], // upper, middle, lower, width
  keltner:  ['#00bcd4', '#009688', '#00bcd4'],
  donchian: ['#ff5722', '#795548', '#ff5722'],
  trailing_stop: ['#4caf50', '#f44336'],
  rsi:      ['#ffeb3b'],
  macd:     ['#2196f3', '#ff9800', '#26a69a'],  // MACD, Signal, Histogram
  adx:      ['#ffeb3b', '#4caf50', '#f44336'],   // ADX, +DI, -DI
  atr:      ['#ff9800'],
  bias:     ['#e040fb'],
  roc:      ['#00e5ff'],
  rel_vol:  ['#ff6e40'],
  obv:      ['#69f0ae'],
  vol_switch: ['#ea80fc'],
  max_drawdown: ['#ff5252'],
  kelly:    ['#b2ff59', '#69f0ae'],
  session:  ['#90a4ae'],
  funding:  ['#ffd740', '#ff5252', '#4caf50'],   // Rate, Alert_High, Alert_Low
  fear_greed: ['#ff6e40', '#ef5350', '#66bb6a'],  // FG, Extreme_Fear, Extreme_Greed
  stochrsi: ['#ffeb3b', '#ff9800'],               // %K, %D
  vwap:     ['#e040fb'],
  ichimoku: ['#2196f3', '#f44336', '#4caf50', '#ff9800', '#9e9e9e'],  // Tenkan, Kijun, Senkou_A, Senkou_B, Chikou
  psar:     ['#ffd740'],
  supertrend: ['#26a69a'],
  market_structure: ['#4caf50', '#f44336', '#00e676', '#ff1744', '#69f0ae', '#ff5252'],
  // Swing_High(green), Swing_Low(red), HH(bright green), LH(bright red), HL(light green), LL(light red)
  harmonic: ['#ff9800', '#ff9800', '#ff9800', '#ff9800', '#ff9800', '#ff9800', '#e040fb'],
  // Zigzag(orange), Pattern_X/A/B/C/D(orange), Classification(purple)
};

// Market Structure 系列名稱 → 顏色 / 樣式
const MS_SERIES_CONFIG: Record<string, { color: string; lineWidth: number; lineStyle: number }> = {
  Swing_High: { color: '#4caf50', lineWidth: 1, lineStyle: 2 },  // 綠虛線
  Swing_Low:  { color: '#f44336', lineWidth: 1, lineStyle: 2 },  // 紅虛線
  HH:         { color: '#00e676', lineWidth: 0, lineStyle: 0 },  // 只做 marker
  LH:         { color: '#ff1744', lineWidth: 0, lineStyle: 0 },
  HL:         { color: '#69f0ae', lineWidth: 0, lineStyle: 0 },
  LL:         { color: '#ff5252', lineWidth: 0, lineStyle: 0 },
};

// Harmonic 系列配置
const HARMONIC_SERIES_CONFIG: Record<string, { color: string; lineWidth: number; lineStyle: number; markerOnly?: boolean; markerLabel?: string }> = {
  Zigzag:        { color: '#ff9800', lineWidth: 1, lineStyle: 2 },  // 橙虛線
  Pattern_X:     { color: '#ff9800', lineWidth: 0, lineStyle: 0, markerOnly: true, markerLabel: 'X' },
  Pattern_A:     { color: '#ff9800', lineWidth: 0, lineStyle: 0, markerOnly: true, markerLabel: 'A' },
  Pattern_B:     { color: '#ff9800', lineWidth: 0, lineStyle: 0, markerOnly: true, markerLabel: 'B' },
  Pattern_C:     { color: '#ff9800', lineWidth: 0, lineStyle: 0, markerOnly: true, markerLabel: 'C' },
  Pattern_D:     { color: '#e040fb', lineWidth: 0, lineStyle: 0, markerOnly: true, markerLabel: 'D' },
};

function getSeriesColor(indicatorId: string, seriesIndex: number): string {
  const colors = INDICATOR_COLORS[indicatorId] || ['#ffffff'];
  return colors[seriesIndex % colors.length];
}

// ─── 共用圖表選項 ──────────────────────────────────
function getChartOptions(width: number, height: number) {
  return {
    width,
    height,
    layout: {
      background: { type: ColorType.Solid as const, color: '#0d1117' },
      textColor: '#8b949e',
      fontSize: 11,
    },
    grid: {
      vertLines: { color: '#21262d' },
      horzLines: { color: '#21262d' },
    },
    crosshair: {
      mode: 0 as const,
      vertLine: { color: '#58a6ff', width: 1 as const, style: 2, labelBackgroundColor: '#58a6ff' },
      horzLine: { color: '#58a6ff', width: 1 as const, style: 2, labelBackgroundColor: '#58a6ff' },
    },
    rightPriceScale: { borderColor: '#30363d' },
    timeScale: {
      borderColor: '#30363d',
      timeVisible: true,
      secondsVisible: false,
    },
  };
}

// ─── 將 OHLCV timestamp 轉為秒級 Unix Timestamp ──
// CSV 中的 timestamp 已是台北時間（UTC+8），直接以 UTC 方式解析
// 使圖表時間軸顯示與 CSV 一致的台北時間
function toChartTime(ts: string): number {
  const normalized = ts.replace(' ', 'T');
  const utcStr = normalized.endsWith('Z') ? normalized : normalized + 'Z';
  return new Date(utcStr).getTime() / 1000;
}

// ─── 統一右側價格軸最小寬度（確保所有圖表繪圖區左右對齊）─────
const PRICE_SCALE_MIN_WIDTH = 65;

// ─── Sub-Chart 子圖組件 ──────────────────────────────
interface SubChartPaneProps {
  indicator: ActiveIndicator;
  ohlcvData: OHLCVData[];
  syncRangeRef: React.MutableRefObject<TimeBasedRange | null>;
  onTimeRangeChange?: (range: TimeBasedRange | null) => void;
  isLastSubChart: boolean;
  onChartReady?: (id: string, chart: IChartApi, series: ISeriesApi<any> | null) => void;
  onChartDestroy?: (id: string) => void;
  paneHeight: number;
  onResizeStart?: (indicatorId: string, startY: number) => void;
  onRemove?: (indicatorId: string) => void;
}

const SubChartPane = memo(function SubChartPane({
  indicator, ohlcvData, syncRangeRef, onTimeRangeChange, isLastSubChart,
  onChartReady, onChartDestroy, paneHeight, onResizeStart, onRemove,
}: SubChartPaneProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesMapRef = useRef<Map<string, ISeriesApi<any>>>(new Map());
  const isSyncingRef = useRef(false);
  const subThrottleRef = useRef<ReturnType<typeof requestAnimationFrame> | null>(null);

  // 十字線數值追蹤（DOM 直接操作避免 React re-render）
  const legendRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;

    const chart = createChart(el, {
      ...getChartOptions(el.clientWidth, el.clientHeight),
      rightPriceScale: {
        borderColor: '#30363d',
        scaleMargins: { top: 0.1, bottom: 0.05 },
        minimumWidth: PRICE_SCALE_MIN_WIDTH,
      },
      timeScale: {
        borderColor: '#30363d',
        timeVisible: true,
        secondsVisible: false,
        visible: isLastSubChart,
        fixLeftEdge: false,
        fixRightEdge: false,
      },
    });

    chartRef.current = chart;

    // 十字線移動時更新數值顯示（直接操作 DOM，避免 setState 觸發重渲染）
    chart.subscribeCrosshairMove((param) => {
      const legendEl = legendRef.current;
      if (!legendEl) return;

      if (!param.time || param.seriesData.size === 0) {
        legendEl.innerHTML = '';
        return;
      }

      const htmlParts: string[] = [];
      let sIdx = 0;
      for (const [seriesName, series] of seriesMapRef.current) {
        if (/^(Alert_|Extreme_)/.test(seriesName)) { sIdx++; continue; }
        const data = param.seriesData.get(series) as any;
        if (!data) { sIdx++; continue; }
        const val = data.value ?? data.close;
        if (val == null || typeof val !== 'number') { sIdx++; continue; }
        const color = getSeriesColor(indicator.id, sIdx);
        const label = seriesName === 'Histogram' ? 'Hist' : seriesName;
        const prec = Math.abs(val) >= 100 ? 2 : Math.abs(val) >= 1 ? 4 : 6;
        htmlParts.push(
          `<span style="color:${color};margin-left:6px">${label}: ${val.toFixed(prec)}</span>`
        );
        sIdx++;
      }
      legendEl.innerHTML = htmlParts.join('');
    });

    chart.timeScale().subscribeVisibleTimeRangeChange((range: any) => {
      if (!isSyncingRef.current && onTimeRangeChange && range) {
        if (subThrottleRef.current) cancelAnimationFrame(subThrottleRef.current);
        subThrottleRef.current = requestAnimationFrame(() => {
          onTimeRangeChange({ from: range.from as number, to: range.to as number });
          subThrottleRef.current = null;
        });
      }
    });

    onChartReady?.(indicator.id, chart, null);

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          chart.applyOptions({ width, height });
        }
      }
    });
    resizeObserver.observe(el);

    return () => {
      onChartDestroy?.(indicator.id);
      if (subThrottleRef.current) cancelAnimationFrame(subThrottleRef.current);
      resizeObserver.disconnect();
      chart.remove();
      chartRef.current = null;
      seriesMapRef.current.clear();
    };
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  // 副圖接收即時同步（透過 ref 直接讀取，跳過 React re-render 延遲）
  useEffect(() => {
    const range = syncRangeRef.current;
    if (!chartRef.current || !range) return;
    isSyncingRef.current = true;
    try {
      chartRef.current.timeScale().setVisibleRange(range as any);
    } catch { /* 範圍超出數據邊界時忽略 */ }
    requestAnimationFrame(() => { isSyncingRef.current = false; });
  });

  // 顯示/隱藏時間軸隨 isLastSubChart 動態更新
  useEffect(() => {
    chartRef.current?.timeScale().applyOptions({ visible: isLastSubChart });
  }, [isLastSubChart]);

  // 更新數據
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart || !indicator.data || ohlcvData.length === 0) return;

    const timeArr = ohlcvData.map((d) => toChartTime(d.timestamp));
    const dataEntries = Object.entries(indicator.data);

    // 清除已不存在的 series
    const newKeys = new Set(dataEntries.map(([k]) => k));
    for (const [key, series] of seriesMapRef.current) {
      if (!newKeys.has(key)) {
        chart.removeSeries(series);
        seriesMapRef.current.delete(key);
      }
    }

    // MACD Histogram 使用 HistogramSeries，其他用 LineSeries
    dataEntries.forEach(([seriesName, values], idx) => {
      if (!values || values.length === 0) return;

      // 跳過非數值系列（例如 Classification 字串數據）
      const firstVal = values.find((v: any) => v != null);
      if (firstVal !== undefined && typeof firstVal !== 'number') return;

      const isHistogram = seriesName === 'Histogram';
      // 參考線（Alert/Extreme 等固定水平線）用虛線
      const isRefLine = /^(Alert_|Extreme_)/.test(seriesName);
      let series = seriesMapRef.current.get(seriesName);

      if (!series) {
        if (isHistogram) {
          series = chart.addSeries(HistogramSeries, {
            priceScaleId: 'right',
            color: getSeriesColor(indicator.id, idx),
          });
        } else {
          series = chart.addSeries(LineSeries, {
            color: getSeriesColor(indicator.id, idx),
            lineWidth: isRefLine ? 1 : 1,
            lineStyle: isRefLine ? 2 : 0, // 2 = Dashed
            priceScaleId: 'right',
            crosshairMarkerVisible: !isRefLine,
          });
        }
        seriesMapRef.current.set(seriesName, series);
      }

      const data: any[] = [];
      for (let i = 0; i < values.length && i < timeArr.length; i++) {
        if (values[i] != null) {
          if (isHistogram) {
            data.push({
              time: timeArr[i] as any,
              value: values[i],
              color: (values[i] as number) >= 0 ? 'rgba(38,166,154,0.7)' : 'rgba(239,83,80,0.7)',
            });
          } else {
            data.push({ time: timeArr[i] as any, value: values[i] });
          }
        }
      }

      if (data.length > 0) {
        series.setData(data);
      }
    });

    // 數據更新後，同步到主圖的時間範圍（即時讀取 ref）
    const range = syncRangeRef.current;
    if (range) {
      isSyncingRef.current = true;
      try {
        chart.timeScale().setVisibleRange(range as any);
      } catch { /* ignore */ }
      requestAnimationFrame(() => { isSyncingRef.current = false; });
    } else {
      chart.timeScale().fitContent();
    }

    // 更新首個 series 供 crosshair 同步使用
    const firstSeries = seriesMapRef.current.values().next().value || null;
    onChartReady?.(indicator.id, chart, firstSeries);
  }, [indicator.data, indicator.id, ohlcvData]);  // eslint-disable-line react-hooks/exhaustive-deps

  // RSI 超買超賣參考線
  const isRSI = indicator.indicator_type === 'RSI';
  const overbought = indicator.parameters?.overbought ?? 70;
  const oversold = indicator.parameters?.oversold ?? 30;

  return (
    <div
      style={{
        height: `${paneHeight}px`,
        minHeight: '60px',
        borderTop: '1px solid #30363d',
        position: 'relative',
        background: '#0d1117',
      }}
    >
      {/* 頂部拖曳把手（調整高度） */}
      <div
        style={{
          position: 'absolute',
          top: 0, left: 0, right: 0,
          height: '5px',
          cursor: 'row-resize',
          zIndex: 15,
        }}
        onMouseDown={(e) => {
          e.preventDefault();
          onResizeStart?.(indicator.id, e.clientY);
        }}
      >
        <div style={{
          position: 'absolute', top: '1px', left: '50%', transform: 'translateX(-50%)',
          width: '30px', height: '3px', borderRadius: '2px',
          background: '#30363d', opacity: 0.6,
        }} />
      </div>

      {/* 指標名稱 + 十字線數值 + 關閉按鈕 */}
      <div
        className="absolute top-1 left-2 z-10 text-xs font-medium"
        style={{ color: '#8b949e', display: 'flex', alignItems: 'center', flexWrap: 'wrap' }}
      >
        <span>{indicator.indicator_type}</span>
        {Object.keys(indicator.parameters).length > 0 && (
          <span style={{ color: '#58a6ff', marginLeft: '4px' }}>
            ({Object.values(indicator.parameters).join(', ')})
          </span>
        )}
        {isRSI && (
          <span style={{ color: '#6e7681', marginLeft: '6px' }}>
            OB:{overbought} OS:{oversold}
          </span>
        )}
        <span
          ref={legendRef}
          style={{ marginLeft: '4px', fontFamily: 'monospace', whiteSpace: 'nowrap' }}
        />
      </div>

      {/* ✕ 關閉按鈕 */}
      <button
        onClick={() => onRemove?.(indicator.id)}
        className="absolute top-1 right-2 z-10 cursor-pointer hover:opacity-80"
        style={{
          background: 'rgba(248,81,73,0.12)',
          border: 'none',
          borderRadius: '3px',
          color: '#f85149',
          fontSize: '11px',
          padding: '1px 5px',
          lineHeight: '16px',
        }}
        title={`關閉 ${indicator.indicator_type}`}
      >
        ✕
      </button>

      {/* 圖表容器 */}
      <div ref={containerRef} className="w-full h-full" />
    </div>
  );
});

// ═══════════════════════════════════════════════════
// 主圖表組件
// ═══════════════════════════════════════════════════
export default function ChartView() {
  const [markerPanelOpen, setMarkerPanelOpen] = useState(false);
  const mainContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const candleSeriesRef = useRef<ISeriesApi<any> | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const volumeSeriesRef = useRef<ISeriesApi<any> | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const overlaySeriesRef = useRef<Map<string, ISeriesApi<any>>>(new Map());
  const isSyncingRef = useRef(false);

  // 主圖十字線 overlay 數值 + OHLCV 數值
  const mainLegendRef = useRef<HTMLDivElement>(null);

  // ref 即時同步（零延遲），無需 React state 中轉
  const syncRangeRef = useRef<TimeBasedRange | null>(null);
  const syncThrottleRef = useRef<ReturnType<typeof requestAnimationFrame> | null>(null);

  // 副圖 chart 實例登記（crosshair 同步用）
  const subChartMapRef = useRef<Map<string, { chart: IChartApi; series: ISeriesApi<any> | null }>>(new Map());
  const crosshairSyncingRef = useRef(false);

  const ohlcvData = useChartStore((s) => s.ohlcvData);
  const symbol = useChartStore((s) => s.symbol);
  const timeframe = useChartStore((s) => s.timeframe);
  const loading = useChartStore((s) => s.loading);
  const activeIndicators = useChartStore((s) => s.activeIndicators);
  const annotations = useChartStore((s) => s.annotations);
  const clearAnnotations = useChartStore((s) => s.clearAnnotations);
  const toggleAnnotationGroup = useChartStore((s) => s.toggleAnnotationGroup);
  const removeAnnotationGroup = useChartStore((s) => s.removeAnnotationGroup);
  const getAnnotationGroups = useChartStore((s) => s.getAnnotationGroups);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const annotationSeriesRef = useRef<Map<string, ISeriesApi<any>>>(new Map());
  const priceLinesRef = useRef<Map<string, any>>(new Map());

  const removeIndicator = useChartStore((s) => s.removeIndicator);

  // 副圖高度管理（可拖曳）
  const [paneHeights, setPaneHeights] = useState<Record<string, number>>({});
  const resizeRef = useRef<{ indicatorId: string; startY: number; startH: number } | null>(null);

  const getPaneHeight = useCallback((id: string, isLast: boolean) => {
    return paneHeights[id] ?? (isLast ? 160 : 120);
  }, [paneHeights]);

  const handleResizeStart = useCallback((indicatorId: string, startY: number) => {
    const h = paneHeights[indicatorId] ?? 120;
    resizeRef.current = { indicatorId, startY, startH: h };

    const onMouseMove = (e: MouseEvent) => {
      if (!resizeRef.current) return;
      const delta = resizeRef.current.startY - e.clientY;
      const newH = Math.max(60, Math.min(400, resizeRef.current.startH + delta));
      setPaneHeights((prev) => ({ ...prev, [resizeRef.current!.indicatorId]: newH }));
    };

    const onMouseUp = () => {
      resizeRef.current = null;
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
    document.body.style.cursor = 'row-resize';
    document.body.style.userSelect = 'none';
  }, [paneHeights]);

  const handleRemoveIndicator = useCallback((indicatorId: string) => {
    removeIndicator(indicatorId);
  }, [removeIndicator]);

  // 統一 markers 管理（避免 overlay 和 annotation 的 setMarkers 互相覆蓋）
  type MarkerItem = { time: any; position: 'aboveBar' | 'belowBar'; color: string; shape: 'arrowDown' | 'arrowUp' | 'circle'; text: string };
  const overlayMarkersRef = useRef<MarkerItem[]>([]);
  const annotationMarkersRef = useRef<MarkerItem[]>([]);

  // 分類指標（useMemo 避免每次渲染都重新 filter）
  const overlayIndicators = useMemo(
    () => activeIndicators.filter((i) => i.visible && i.display_mode === 'overlay' && i.data),
    [activeIndicators],
  );
  const subChartIndicators = useMemo(
    () => activeIndicators.filter((i) => i.visible && i.display_mode === 'sub_chart' && i.data),
    [activeIndicators],
  );

  // 統一刷新 markers（合併 overlay + annotation 兩個來源）
  const flushMarkers = useCallback(() => {
    const cs = candleSeriesRef.current;
    if (!cs) return;
    const merged = [...overlayMarkersRef.current, ...annotationMarkersRef.current];
    merged.sort((a, b) => (a.time as number) - (b.time as number));
    try { (cs as any).setMarkers(merged); } catch { /* ignore */ }
  }, []);

  // 初始化主圖表
  useEffect(() => {
    if (!mainContainerRef.current) return;
    const el = mainContainerRef.current;

    const chart = createChart(el, {
      ...getChartOptions(el.clientWidth, el.clientHeight),
      rightPriceScale: {
        borderColor: '#30363d',
        scaleMargins: { top: 0.1, bottom: 0.25 },
        minimumWidth: PRICE_SCALE_MIN_WIDTH,
      },
    });

    // K 線序列
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#3fb950',
      downColor: '#f85149',
      borderUpColor: '#3fb950',
      borderDownColor: '#f85149',
      wickUpColor: '#3fb950',
      wickDownColor: '#f85149',
    });

    // 成交量序列
    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
    });
    chart.priceScale('volume').applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 },
    });

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;
    volumeSeriesRef.current = volumeSeries;

    // 主圖時間軸變更 → 即時寫入 ref + 節流觸發 state（雙軌同步）
    chart.timeScale().subscribeVisibleTimeRangeChange((range: any) => {
      if (!isSyncingRef.current && range) {
        const tr: TimeBasedRange = { from: range.from as number, to: range.to as number };
        syncRangeRef.current = tr;

        // 即時同步所有副圖（跳過 React re-render）
        for (const [, info] of subChartMapRef.current) {
          try { info.chart.timeScale().setVisibleRange(tr as any); } catch { /* ignore */ }
        }

        if (syncThrottleRef.current) cancelAnimationFrame(syncThrottleRef.current);
        syncThrottleRef.current = requestAnimationFrame(() => {
          syncThrottleRef.current = null;
        });
      }
    });

    // Crosshair 同步：主圖 → 副圖 + 主圖數值顯示
    chart.subscribeCrosshairMove((param: any) => {
      // 更新主圖十字線數值
      const legendEl = mainLegendRef.current;
      if (legendEl) {
        if (!param.time || param.seriesData.size === 0) {
          legendEl.innerHTML = '';
        } else {
          const htmlParts: string[] = [];
          // OHLCV 數值
          const candleData = param.seriesData.get(candleSeriesRef.current) as any;
          if (candleData && candleData.open != null) {
            const prec = ohlcvData.length > 0
              ? getAutoPriceFormat(ohlcvData[ohlcvData.length - 1].close).precision : 2;
            const o = candleData.open.toFixed(prec);
            const h = candleData.high.toFixed(prec);
            const l = candleData.low.toFixed(prec);
            const c = candleData.close.toFixed(prec);
            const clr = candleData.close >= candleData.open ? '#3fb950' : '#f85149';
            htmlParts.push(
              `<span style="color:#8b949e">O:</span><span style="color:${clr}">${o}</span>` +
              `<span style="color:#8b949e;margin-left:4px">H:</span><span style="color:${clr}">${h}</span>` +
              `<span style="color:#8b949e;margin-left:4px">L:</span><span style="color:${clr}">${l}</span>` +
              `<span style="color:#8b949e;margin-left:4px">C:</span><span style="color:${clr}">${c}</span>`
            );
          }
          // Volume
          const volData = param.seriesData.get(volumeSeriesRef.current) as any;
          if (volData && volData.value != null) {
            const vf = volData.value >= 1e6
              ? (volData.value / 1e6).toFixed(2) + 'M'
              : volData.value >= 1e3
                ? (volData.value / 1e3).toFixed(2) + 'K'
                : volData.value.toFixed(2);
            htmlParts.push(`<span style="color:#8b949e;margin-left:6px">Vol:</span><span style="color:#58a6ff">${vf}</span>`);
          }
          // overlay 指標數值
          for (const [key, series] of overlaySeriesRef.current) {
            const data = param.seriesData.get(series) as any;
            if (!data) continue;
            const val = data.value ?? data.close;
            if (val == null || typeof val !== 'number') continue;
            const seriesName = key.split('__')[1] || key;
            if (seriesName === 'BB_Width' || seriesName === 'Classification') continue;
            if (/^(Alert_|Extreme_)/.test(seriesName)) continue;
            const indId = key.split('__')[0] || '';
            const indColor = INDICATOR_COLORS[indId]?.[0] || '#ffffff';
            const prec = val >= 1000 ? 2 : val >= 1 ? 4 : 6;
            htmlParts.push(
              `<span style="color:${indColor};margin-left:6px">${seriesName}: ${val.toFixed(prec)}</span>`
            );
          }
          legendEl.innerHTML = htmlParts.join('');
        }
      }

      // 同步十字線到副圖
      if (crosshairSyncingRef.current) return;
      crosshairSyncingRef.current = true;
      try {
        for (const [, info] of subChartMapRef.current) {
          if (param.time && info.series) {
            try { info.chart.setCrosshairPosition(NaN, param.time, info.series); } catch { /* ignore */ }
          } else {
            try { info.chart.clearCrosshairPosition(); } catch { /* ignore */ }
          }
        }
      } finally {
        crosshairSyncingRef.current = false;
      }
    });

    // 自適應大小
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          chart.applyOptions({ width, height });
        }
      }
    });
    resizeObserver.observe(el);

    return () => {
      resizeObserver.disconnect();
      if (syncThrottleRef.current) cancelAnimationFrame(syncThrottleRef.current);
      chart.remove();
      chartRef.current = null;
      overlaySeriesRef.current.clear();
    };
  }, []);

  // 更新 K 線 + 成交量數據
  useEffect(() => {
    if (!candleSeriesRef.current || !volumeSeriesRef.current || ohlcvData.length === 0) return;

    // 依據最新收盤價動態設定價格精度
    const latestClose = ohlcvData[ohlcvData.length - 1].close;
    const priceFormat = getAutoPriceFormat(latestClose);
    candleSeriesRef.current.applyOptions({ priceFormat });

    const candleData = ohlcvData.map((d) => ({
      time: toChartTime(d.timestamp) as any,
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
    }));

    const volumeData = ohlcvData.map((d) => ({
      time: toChartTime(d.timestamp) as any,
      value: d.volume,
      color: d.close >= d.open ? 'rgba(63, 185, 80, 0.3)' : 'rgba(248, 81, 73, 0.3)',
    }));

    candleSeriesRef.current.setData(candleData);
    volumeSeriesRef.current.setData(volumeData);
    chartRef.current?.timeScale().fitContent();
  }, [ohlcvData]);

  // 更新 overlay 指標（含 Market Structure 特殊渲染）
  useEffect(() => {
    const chart = chartRef.current;
    const _candleSeries = candleSeriesRef.current;
    void _candleSeries;
    if (!chart || ohlcvData.length === 0) return;

    const timeArr = ohlcvData.map((d) => toChartTime(d.timestamp));
    const activeKeys = new Set<string>();

    // 收集特殊指標的 markers（Market Structure + Harmonic）
    const overlayMarkers: Array<{
      time: any;
      position: 'aboveBar' | 'belowBar';
      color: string;
      shape: 'arrowDown' | 'arrowUp' | 'circle';
      text: string;
    }> = [];

    for (const indicator of overlayIndicators) {
      if (!indicator.data) continue;

      const isMS = indicator.indicator_type === 'Market Structure';
      const isHarmonic = indicator.indicator_type === 'Harmonic Patterns 諧波型態';

      Object.entries(indicator.data).forEach(([seriesName, values], idx) => {
        if (!values || values.length === 0) return;
        if (seriesName === 'BB_Width' || seriesName === 'Classification') return;

        const key = `${indicator.id}__${seriesName}`;
        const msConf = isMS ? MS_SERIES_CONFIG[seriesName] : null;
        const harmConf = isHarmonic ? HARMONIC_SERIES_CONFIG[seriesName] : null;

        // Market Structure: HH/LH/HL/LL → marker
        if (msConf && msConf.lineWidth === 0) {
          activeKeys.add(key);
          const isHigh = seriesName === 'HH' || seriesName === 'LH';
          for (let i = 0; i < values.length && i < timeArr.length; i++) {
            if (values[i] != null) {
              overlayMarkers.push({
                time: timeArr[i] as any,
                position: isHigh ? 'aboveBar' : 'belowBar',
                color: msConf.color,
                shape: isHigh ? 'arrowDown' : 'arrowUp',
                text: seriesName,
              });
            }
          }
          return;
        }

        // Harmonic: Pattern_X/A/B/C/D → marker
        if (harmConf && harmConf.markerOnly) {
          activeKeys.add(key);
          for (let i = 0; i < values.length && i < timeArr.length; i++) {
            if (values[i] != null) {
              const isHighPoint = seriesName === 'Pattern_X' || seriesName === 'Pattern_A' || seriesName === 'Pattern_C';
              overlayMarkers.push({
                time: timeArr[i] as any,
                position: isHighPoint ? 'aboveBar' : 'belowBar',
                color: harmConf.color,
                shape: 'circle',
                text: harmConf.markerLabel || seriesName.replace('Pattern_', ''),
              });
            }
          }
          return;
        }

        activeKeys.add(key);
        const conf = msConf || harmConf;

        // overlay 指標也使用與 K 線相同的價格精度
        const overlayPriceFormat = ohlcvData.length > 0
          ? getAutoPriceFormat(ohlcvData[ohlcvData.length - 1].close)
          : undefined;

        let series = overlaySeriesRef.current.get(key);
        if (!series) {
          series = chart.addSeries(LineSeries, {
            color: conf ? conf.color : getSeriesColor(indicator.id, idx),
            lineWidth: (conf ? conf.lineWidth : 1) as any,
            lineStyle: conf ? conf.lineStyle : 0,
            priceScaleId: 'right',
            lastValueVisible: false,
            priceLineVisible: false,
            crosshairMarkerVisible: !(isMS || isHarmonic),
            ...(overlayPriceFormat ? { priceFormat: overlayPriceFormat } : {}),
          });
          overlaySeriesRef.current.set(key, series);
        } else if (overlayPriceFormat) {
          series.applyOptions({ priceFormat: overlayPriceFormat });
        }

        const data: any[] = [];
        for (let i = 0; i < values.length && i < timeArr.length; i++) {
          if (values[i] != null) {
            data.push({ time: timeArr[i] as any, value: values[i] });
          }
        }
        if (data.length > 0) {
          series.setData(data);
        }
      });
    }

    // 將 overlay markers 存入 ref，再統一刷新（避免覆蓋 annotation markers）
    overlayMarkersRef.current = overlayMarkers;
    flushMarkers();

    // 移除已不再啟用的 overlay series
    for (const [key, series] of overlaySeriesRef.current) {
      if (!activeKeys.has(key)) {
        chart.removeSeries(series);
        overlaySeriesRef.current.delete(key);
      }
    }
  }, [overlayIndicators, ohlcvData, flushMarkers]);

  // ── 渲染 Annotations（標記、垂直線、水平線、文字）──
  useEffect(() => {
    const chart = chartRef.current;
    const candleSeries = candleSeriesRef.current;
    if (!chart || !candleSeries) return;

    // 安全清除舊的 annotation series
    for (const [key, series] of annotationSeriesRef.current) {
      try { chart.removeSeries(series); } catch { /* series 可能已被移除 */ }
      annotationSeriesRef.current.delete(key);
    }

    // 安全清除舊的 price lines
    for (const [key, pl] of priceLinesRef.current) {
      try { candleSeries.removePriceLine(pl); } catch { /* ignore */ }
      priceLinesRef.current.delete(key);
    }

    const visibleAnnotations = annotations.filter((a) => a.visible !== false);

    if (visibleAnnotations.length === 0 || ohlcvData.length === 0) {
      annotationMarkersRef.current = [];
      flushMarkers();
      return;
    }

    const markers: Array<{
      time: any;
      position: 'aboveBar' | 'belowBar';
      color: string;
      shape: 'arrowDown' | 'arrowUp' | 'circle';
      text: string;
    }> = [];

    for (const ann of visibleAnnotations) {
      try {
        const color = ann.color || '#58a6ff';

        if (ann.type === 'vertical_line' && ann.startTime) {
          markers.push({
            time: toChartTime(ann.startTime) as any,
            position: 'aboveBar',
            color,
            shape: 'arrowDown',
            text: ann.text || '▼',
          });
        }

        if (ann.type === 'horizontal_line' && ann.price != null) {
          const pl = candleSeries.createPriceLine({
            price: ann.price,
            color,
            lineWidth: 1,
            lineStyle: 2,
            axisLabelVisible: true,
            title: ann.text || '',
          });
          priceLinesRef.current.set(ann.id, pl);
        }

        if (ann.type === 'text_label' && ann.startTime) {
          markers.push({
            time: toChartTime(ann.startTime) as any,
            position: ann.price != null ? 'aboveBar' : 'belowBar',
            color,
            shape: 'circle',
            text: ann.text || '',
          });
        }

        if (ann.type === 'trend_line' && ann.startTime && ann.endTime && ann.price != null && ann.endPrice != null) {
          const trendSeries = chart.addSeries(LineSeries, {
            color,
            lineWidth: (ann.lineWidth || 2) as any,
            lineStyle: ann.lineStyle ?? 0,
            priceScaleId: 'right',
            lastValueVisible: false,
            priceLineVisible: false,
            crosshairMarkerVisible: false,
          });
          trendSeries.setData([
            { time: toChartTime(ann.startTime) as any, value: ann.price },
            { time: toChartTime(ann.endTime) as any, value: ann.endPrice },
          ]);
          annotationSeriesRef.current.set(ann.id, trendSeries);

          if (ann.text) {
            markers.push({
              time: toChartTime(ann.startTime) as any,
              position: 'aboveBar',
              color,
              shape: 'circle',
              text: ann.text,
            });
          }
        }

        if (ann.type === 'highlight_range' && ann.startTime && ann.endTime) {
          const startT = toChartTime(ann.startTime);
          const endT = toChartTime(ann.endTime);
          const hlSeries = chart.addSeries(HistogramSeries, {
            priceScaleId: 'hl_overlay',
            priceFormat: { type: 'price' },
            lastValueVisible: false,
            priceLineVisible: false,
          });

          const hlData: any[] = [];
          for (const d of ohlcvData) {
            const t = toChartTime(d.timestamp);
            if (t >= startT && t <= endT) {
              hlData.push({
                time: t as any,
                value: d.high,
                color: color.startsWith('#') ? color + '30' : 'rgba(88,166,255,0.19)',
              });
            }
          }
          if (hlData.length > 0) {
            hlSeries.setData(hlData);
            chart.priceScale('hl_overlay').applyOptions({
              scaleMargins: { top: 0, bottom: 0 },
              visible: false,
            });
          }
          annotationSeriesRef.current.set(ann.id, hlSeries);
        }
      } catch (err) {
        console.warn(`[Annotation] 渲染標記失敗 (${ann.type}):`, err);
      }
    }

    // 將 annotation markers 存入 ref，再統一刷新（與 overlay markers 合併）
    annotationMarkersRef.current = markers;
    flushMarkers();
  }, [annotations, ohlcvData, flushMarkers]);

  // 副圖 → 主圖 時間軸同步回調（即時 ref + 節流 state）
  const handleSubChartTimeChange = useCallback((range: TimeBasedRange | null) => {
    if (!chartRef.current || !range) return;
    isSyncingRef.current = true;
    syncRangeRef.current = range;
    try {
      chartRef.current.timeScale().setVisibleRange(range as any);
    } catch { /* ignore */ }
    // 同步到所有其他副圖
    for (const [, info] of subChartMapRef.current) {
      try { info.chart.timeScale().setVisibleRange(range as any); } catch { /* ignore */ }
    }
    requestAnimationFrame(() => { isSyncingRef.current = false; });
  }, []);

  // 副圖 chart 實例登記 / 註銷（crosshair 同步用）
  const handleSubChartReady = useCallback((id: string, chart: IChartApi, series: ISeriesApi<any> | null) => {
    subChartMapRef.current.set(id, { chart, series });

    // 副圖 crosshair → 主圖
    chart.subscribeCrosshairMove((param: any) => {
      if (crosshairSyncingRef.current) return;
      crosshairSyncingRef.current = true;
      try {
        const mainChart = chartRef.current;
        const mainSeries = candleSeriesRef.current;
        if (mainChart && mainSeries) {
          if (param.time) {
            try { mainChart.setCrosshairPosition(NaN, param.time, mainSeries); } catch { /* ignore */ }
          } else {
            try { mainChart.clearCrosshairPosition(); } catch { /* ignore */ }
          }
        }
        // 同步到其他副圖
        for (const [otherId, info] of subChartMapRef.current) {
          if (otherId === id || !info.series) continue;
          if (param.time) {
            try { info.chart.setCrosshairPosition(NaN, param.time, info.series); } catch { /* ignore */ }
          } else {
            try { info.chart.clearCrosshairPosition(); } catch { /* ignore */ }
          }
        }
      } finally {
        crosshairSyncingRef.current = false;
      }
    });
  }, []);

  const handleSubChartDestroy = useCallback((id: string) => {
    subChartMapRef.current.delete(id);
  }, []);

  return (
    <div className="flex flex-col flex-1" style={{ minHeight: 0 }}>
      {/* ── 主圖區域 ── */}
      <div className="flex-1 relative" style={{ minHeight: '200px' }}>
        {/* 圖表資訊覆蓋層 */}
        <div
          className="absolute top-3 left-3 z-10 text-sm"
          style={{ color: '#8b949e' }}
        >
          <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '2px 0' }}>
            <span className="font-semibold" style={{ color: '#e6edf3' }}>
              {symbol}
            </span>
            <span className="ml-2">{timeframe.toUpperCase()}</span>
            {ohlcvData.length > 0 && (
              <span className="ml-2">({ohlcvData.length} bars)</span>
            )}
            {/* overlay 指標標籤 */}
            {overlayIndicators.map((ind) => (
              <span
                key={ind.id}
                className="ml-3 text-xs"
                style={{ color: getSeriesColor(ind.id, 0) }}
              >
                {ind.indicator_type}({Object.values(ind.parameters).join(',')})
              </span>
            ))}
          </div>
          {/* 十字線 OHLCV + overlay 數值 */}
          <div
            ref={mainLegendRef}
            className="text-xs"
            style={{ fontFamily: 'monospace', whiteSpace: 'nowrap', minHeight: '14px', marginTop: '2px' }}
          />
          {/* annotations 清除全部按鈕 */}
          {annotations.length > 0 && (
            <button
              onClick={() => {
                clearAnnotations();
                annotationMarkersRef.current = [];
                flushMarkers();
              }}
              className="ml-3 text-xs cursor-pointer hover:opacity-80 px-1.5 py-0.5 rounded"
              style={{ background: 'rgba(248,81,73,0.15)', color: '#f85149' }}
              title="清除所有 AI 標記"
            >
              ✕ 清除全部標記
            </button>
          )}
        </div>

        {/* AI 標記分組管理面板（可收合） */}
        {annotations.length > 0 && (() => {
          const groups = getAnnotationGroups();
          if (groups.length === 0) return null;
          return markerPanelOpen ? (
            <div
              className="absolute top-3 right-14 z-10 text-xs"
              style={{
                background: 'rgba(22,27,34,0.95)',
                border: '1px solid #30363d',
                borderRadius: '6px',
                padding: '6px 8px',
                maxWidth: '220px',
                backdropFilter: 'blur(6px)',
              }}
            >
              <div className="flex items-center justify-between" style={{ marginBottom: '4px' }}>
                <span style={{ color: '#8b949e', fontWeight: 600, fontSize: '11px' }}>
                  AI 標記
                </span>
                <button
                  onClick={() => setMarkerPanelOpen(false)}
                  className="cursor-pointer hover:opacity-70"
                  style={{ color: '#8b949e', background: 'none', border: 'none', fontSize: '13px', lineHeight: 1, padding: '0 1px' }}
                  title="收合面板"
                >
                  ▾
                </button>
              </div>
              {groups.map((g) => (
                <div
                  key={g.groupId}
                  className="flex items-center gap-1.5"
                  style={{ padding: '2px 0', opacity: g.visible ? 1 : 0.45 }}
                >
                  <span style={{ width: '8px', height: '8px', borderRadius: '2px', background: g.color, flexShrink: 0 }} />
                  <button
                    onClick={() => toggleAnnotationGroup(g.groupId)}
                    className="flex-1 text-left cursor-pointer hover:opacity-80 truncate"
                    style={{
                      color: g.visible ? '#e6edf3' : '#8b949e',
                      background: 'none', border: 'none', padding: 0, fontSize: '11px',
                      textDecoration: g.visible ? 'none' : 'line-through',
                    }}
                    title={g.visible ? `隱藏「${g.groupName}」` : `顯示「${g.groupName}」`}
                  >
                    {g.groupName}
                    <span style={{ color: '#8b949e', marginLeft: '3px' }}>({g.count})</span>
                  </button>
                  <button
                    onClick={() => removeAnnotationGroup(g.groupId)}
                    className="cursor-pointer hover:opacity-80"
                    style={{ color: '#f85149', background: 'none', border: 'none', padding: '0 2px', fontSize: '11px', lineHeight: 1 }}
                    title={`刪除「${g.groupName}」`}
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <button
              onClick={() => setMarkerPanelOpen(true)}
              className="absolute top-3 right-14 z-10 cursor-pointer hover:opacity-80"
              style={{
                background: 'rgba(22,27,34,0.85)',
                border: '1px solid #30363d',
                borderRadius: '6px',
                padding: '3px 8px',
                color: '#8b949e',
                fontSize: '11px',
                backdropFilter: 'blur(6px)',
              }}
              title="展開 AI 標記面板"
            >
              AI 標記 ({annotations.length})
            </button>
          );
        })()}

        {/* 載入中覆蓋層 */}
        {loading && (
          <div className="absolute inset-0 z-20 flex items-center justify-center bg-black/50">
            <div className="flex flex-col items-center gap-2">
              <div
                className="w-8 h-8 border-2 rounded-full animate-spin"
                style={{
                  borderColor: '#30363d',
                  borderTopColor: '#58a6ff',
                }}
              />
              <span style={{ color: '#8b949e' }}>載入數據中...</span>
            </div>
          </div>
        )}

        {/* 無數據提示 */}
        {!loading && ohlcvData.length === 0 && (
          <div className="absolute inset-0 z-10 flex items-center justify-center">
            <div className="text-center" style={{ color: '#8b949e' }}>
              <p className="text-lg mb-2">尚無數據</p>
              <p className="text-sm mb-3">
                本地尚無 {symbol} {timeframe} 的數據
              </p>
              <button
                onClick={() => useChartStore.getState().setShowSyncPanel(true)}
                className="px-4 py-2 rounded-lg text-sm font-medium cursor-pointer transition-opacity hover:opacity-80"
                style={{ background: '#58a6ff', color: '#fff' }}
              >
                前往同步數據
              </button>
              <p className="text-xs mt-3" style={{ color: '#8b949e' }}>
                或點擊頂部的「同步數據」按鈕
              </p>
            </div>
          </div>
        )}

        {/* 主圖表容器 */}
        <div ref={mainContainerRef} className="w-full h-full" />
      </div>

      {/* ── Sub-Chart 副圖區域 ── */}
      {subChartIndicators.map((indicator, idx) => {
        const isLast = idx === subChartIndicators.length - 1;
        return (
          <SubChartPane
            key={indicator.id}
            indicator={indicator}
            ohlcvData={ohlcvData}
            syncRangeRef={syncRangeRef}
            onTimeRangeChange={handleSubChartTimeChange}
            isLastSubChart={isLast}
            onChartReady={handleSubChartReady}
            onChartDestroy={handleSubChartDestroy}
            paneHeight={getPaneHeight(indicator.id, isLast)}
            onResizeStart={handleResizeStart}
            onRemove={handleRemoveIndicator}
          />
        );
      })}
    </div>
  );
}
