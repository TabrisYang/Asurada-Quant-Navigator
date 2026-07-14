/** 資金曲線圖 — lightweight-charts Area series（通用元件，回測結果視覺化）。 */

import { useEffect, useRef } from 'react';
import { createChart, AreaSeries, type IChartApi, type UTCTimestamp } from 'lightweight-charts';

interface Props {
  /** [{time: 'YYYY-MM-DD HH:MM:SS', equity: number}] — 時間需遞增 */
  points: { time: string; equity: number }[];
  height?: number;
}

function toUtcTimestamp(time: string): UTCTimestamp {
  // 與 ChartView.toChartTime 同一套規則：字串直接當 UTC 解析，
  // 確保與主圖 K 線時間軸的基準一致
  const normalized = time.includes('T') ? time : time.replace(' ', 'T');
  const suffixed = normalized.endsWith('Z') ? normalized : normalized + 'Z';
  return Math.floor(new Date(suffixed).getTime() / 1000) as UTCTimestamp;
}

export default function EquityCurveChart({ points, height = 180 }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el || points.length === 0) return;

    const initial = points[0].equity;
    const final = points[points.length - 1].equity;
    const isProfit = final >= initial;
    const lineColor = isProfit ? '#3fb950' : '#f85149';

    const chart = createChart(el, {
      width: el.clientWidth,
      height,
      layout: {
        background: { color: 'transparent' },
        textColor: '#8b949e',
        fontSize: 10,
      },
      grid: {
        vertLines: { color: 'rgba(139, 148, 158, 0.1)' },
        horzLines: { color: 'rgba(139, 148, 158, 0.1)' },
      },
      rightPriceScale: { borderVisible: false },
      timeScale: { borderVisible: false, timeVisible: false },
      handleScroll: false,
      handleScale: false,
    });
    chartRef.current = chart;

    const series = chart.addSeries(AreaSeries, {
      lineColor,
      lineWidth: 2,
      topColor: isProfit ? 'rgba(63, 185, 80, 0.25)' : 'rgba(248, 81, 73, 0.25)',
      bottomColor: 'rgba(0, 0, 0, 0)',
      priceLineVisible: false,
      lastValueVisible: true,
    });

    // 時間必須嚴格遞增，防禦性去重（降採樣後理論上已遞增）
    const data: { time: UTCTimestamp; value: number }[] = [];
    let lastTs = -Infinity;
    for (const p of points) {
      const ts = toUtcTimestamp(p.time);
      if (!Number.isFinite(ts) || ts <= lastTs) continue;
      data.push({ time: ts, value: p.equity });
      lastTs = ts;
    }
    series.setData(data);
    chart.timeScale().fitContent();

    const onResize = () => chart.applyOptions({ width: el.clientWidth });
    const observer = new ResizeObserver(onResize);
    observer.observe(el);

    return () => {
      observer.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, [points, height]);

  if (points.length === 0) return null;
  return <div ref={containerRef} style={{ width: '100%' }} />;
}
