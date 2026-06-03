/** 阿斯拉量化系統 — 策略過期 / Regime 切換 提醒 banner。
 *
 * 掛在 App.tsx 頂部（V132EvalBanner 下方）。讀後端 /api/predictions/staleness_check：
 *   ok / no_prediction / error  → 不渲染
 *   expired                     → 橘色 banner +「重跑分析」按鈕 + × dismiss(24h)
 *   regime_changed              → 紫色 banner +「重跑分析」按鈕 + × dismiss(24h)
 *
 * dismiss key 含 prediction_id：換新預測時舊 dismiss 失效。
 * 「重跑分析」透過 chartStore.setPendingChatMessage 自動填入 ChatInterface 並送出。
 */

import { useState, useEffect, useCallback } from 'react';
import { fetchStalenessCheck, type StalenessCheckResponse } from '../services/api';
import { useChartStore } from '../stores/chartStore';

const POLL_INTERVAL_MS = 5 * 60 * 1000;  // 5 分鐘
const DISMISS_TTL_MS = 24 * 60 * 60 * 1000;
const DISMISS_KEY_PREFIX = 'asura_stale_dismissed_';

const STYLE = {
  expired:        { bg: 'rgba(251,133,0,0.14)',  border: '#fb8500', text: '#ffd9a8', icon: '⏰' },
  regime_changed: { bg: 'rgba(163,113,247,0.14)', border: '#a371f7', text: '#e2d4ff', icon: '🔄' },
};

function dismissKey(symbol: string, timeframe: string, predictionId?: number): string {
  return `${DISMISS_KEY_PREFIX}${symbol.replace('/', '_')}_${timeframe}_${predictionId ?? 'none'}`;
}

export default function StaleAnalysisBanner() {
  const symbol = useChartStore((s) => s.symbol);
  const timeframe = useChartStore((s) => s.timeframe);
  const setPendingChatMessage = useChartStore((s) => s.setPendingChatMessage);

  const [data, setData] = useState<StalenessCheckResponse | null>(null);
  const [dismissed, setDismissed] = useState(false);

  const refresh = useCallback(async () => {
    if (!symbol || !timeframe) return;
    const result = await fetchStalenessCheck(symbol, timeframe);
    setData(result);
  }, [symbol, timeframe]);

  // symbol/timeframe 變更時立刻 refresh + 5 分鐘輪詢
  useEffect(() => {
    refresh();
    const t = setInterval(refresh, POLL_INTERVAL_MS);
    return () => clearInterval(t);
  }, [refresh]);

  // expired / regime_changed 狀態：localStorage 記錄 24h 不再顯示
  useEffect(() => {
    if (!data || (data.status !== 'expired' && data.status !== 'regime_changed')) {
      setDismissed(false);
      return;
    }
    const key = dismissKey(symbol, timeframe, data.prediction_id);
    const at = Number(localStorage.getItem(key) || 0);
    setDismissed(Date.now() - at < DISMISS_TTL_MS);
  }, [data, symbol, timeframe]);

  const handleDismiss = () => {
    if (!data || (data.status !== 'expired' && data.status !== 'regime_changed')) return;
    const key = dismissKey(symbol, timeframe, data.prediction_id);
    localStorage.setItem(key, String(Date.now()));
    setDismissed(true);
  };

  const handleRerun = () => {
    setPendingChatMessage(`請對 ${symbol} ${timeframe} 當前狀況重新分析`);
  };

  if (!data) return null;
  if (data.status !== 'expired' && data.status !== 'regime_changed') return null;
  if (dismissed) return null;

  const style = STYLE[data.status];

  return (
    <div
      role="status"
      style={{
        padding: '6px 12px',
        background: style.bg,
        borderBottom: `1px solid ${style.border}`,
        color: style.text,
        fontSize: '12px',
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
      }}
    >
      <span>{style.icon}</span>
      <span style={{ flex: 1 }}>
        {data.status === 'expired' && (
          <>
            <strong>{symbol} {timeframe} 分析已過期</strong>
            {typeof data.expired_hours_ago === 'number' && <>（過期 {data.expired_hours_ago} 小時）</>}
            {' '}— 建議重跑分析。
          </>
        )}
        {data.status === 'regime_changed' && (
          <>
            <strong>{symbol} {timeframe} Regime 已切換</strong>
            {data.from && data.to && <>：{data.from} → {data.to}</>}
            {typeof data.confidence === 'number' && <>（信心 {(data.confidence * 100).toFixed(0)}%）</>}
            {' '}— 原策略可能不適用，建議重跑分析。
          </>
        )}
      </span>

      <button
        onClick={handleRerun}
        style={{
          background: style.border, border: 'none', color: '#fff',
          padding: '3px 10px', borderRadius: 3,
          cursor: 'pointer', fontSize: '11px', fontWeight: 600,
        }}
      >
        重跑分析
      </button>

      <button
        onClick={handleDismiss}
        aria-label="關閉提示"
        style={{
          background: 'transparent', border: 'none', color: style.text,
          cursor: 'pointer', fontSize: '14px', padding: '0 4px',
        }}
      >
        ×
      </button>
    </div>
  );
}
