/** 阿斯拉量化系統 — 頂部工具列 */

import { useChartStore } from '../stores/chartStore';
import type { Timeframe } from '../types';
import FactorScanPanel from './FactorScanPanel/FactorScanPanel';

const TIMEFRAMES: { label: string; value: Timeframe }[] = [
  { label: '15m', value: '15m' },
  { label: '1H', value: '1h' },
  { label: '4H', value: '4h' },
  { label: '1D', value: '1d' },
  { label: '1W', value: '1w' },
];

interface TopBarProps {
  onSettingsClick: () => void;
}

export default function TopBar({ onSettingsClick }: TopBarProps) {
  const symbol = useChartStore((s) => s.symbol);
  const setSymbol = useChartStore((s) => s.setSymbol);
  const timeframe = useChartStore((s) => s.timeframe);
  const setTimeframe = useChartStore((s) => s.setTimeframe);
  const ohlcvData = useChartStore((s) => s.ohlcvData);
  const startDate = useChartStore((s) => s.startDate);
  const endDate = useChartStore((s) => s.endDate);
  const setDateRange = useChartStore((s) => s.setDateRange);
  const setShowSyncPanel = useChartStore((s) => s.setShowSyncPanel);

  return (
    <div
      className="flex items-center gap-4 px-4 py-2 border-b"
      style={{
        borderColor: 'var(--border-color)',
        background: 'var(--bg-secondary)',
      }}
    >
      {/* 系統標題 */}
      <h1
        className="text-lg font-bold mr-4"
        style={{ color: 'var(--accent-blue)' }}
      >
        阿斯拉量化系統
      </h1>

      {/* 幣種選擇 */}
      <div className="flex items-center gap-2">
        <select
          value={symbol}
          onChange={(e) => setSymbol(e.target.value)}
          className="px-3 py-1 rounded text-sm border-none outline-none cursor-pointer"
          style={{
            background: 'var(--bg-tertiary)',
            color: 'var(--text-primary)',
          }}
        >
          <option value="BTC/USDT">BTC/USDT</option>
          <option value="ETH/USDT">ETH/USDT</option>
          <option value="SOL/USDT">SOL/USDT</option>
          <option value="XRP/USDT">XRP/USDT</option>
          <option value="DOGE/USDT">DOGE/USDT</option>
          <option value="ADA/USDT">ADA/USDT</option>
          <option value="AVAX/USDT">AVAX/USDT</option>
          <option value="LINK/USDT">LINK/USDT</option>
          <option value="DOT/USDT">DOT/USDT</option>
          <option value="MATIC/USDT">MATIC/USDT</option>
        </select>
      </div>

      {/* 時間週期選擇 */}
      <div className="flex items-center gap-1">
        {TIMEFRAMES.map((tf) => (
          <button
            key={tf.value}
            onClick={() => setTimeframe(tf.value)}
            className="px-3 py-1 rounded text-sm cursor-pointer transition-colors"
            style={{
              background:
                timeframe === tf.value ? 'var(--accent-blue)' : 'var(--bg-tertiary)',
              color:
                timeframe === tf.value ? '#fff' : 'var(--text-secondary)',
            }}
          >
            {tf.label}
          </button>
        ))}
      </div>

      {/* 分隔線 */}
      <div
        className="h-6 w-px mx-2"
        style={{ background: 'var(--border-color)' }}
      />

      {/* 日期範圍 */}
      <div className="flex items-center gap-2 text-sm" style={{ color: 'var(--text-secondary)' }}>
        <input
          type="datetime-local"
          lang="zh-TW"
          className="px-2 py-1 rounded border-none outline-none text-xs"
          style={{
            background: 'var(--bg-tertiary)',
            color: 'var(--text-primary)',
            colorScheme: 'dark',
            maxWidth: '170px',
          }}
          value={startDate || ''}
          onChange={(e) => setDateRange(e.target.value || null, endDate)}
        />
        <span>~</span>
        <input
          type="datetime-local"
          lang="zh-TW"
          className="px-2 py-1 rounded border-none outline-none text-xs"
          style={{
            background: 'var(--bg-tertiary)',
            color: 'var(--text-primary)',
            colorScheme: 'dark',
            maxWidth: '170px',
          }}
          value={endDate || ''}
          onChange={(e) => setDateRange(startDate, e.target.value || null)}
        />
      </div>

      {/* 數據狀態 */}
      {ohlcvData.length > 0 && (
        <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
          {ohlcvData.length} bars
        </span>
      )}

      {/* 右側工具 */}
      <div className="ml-auto flex items-center gap-3">
        {/* 因子掃描 */}
        <FactorScanPanel />

        {/* 同步數據按鈕 */}
        <button
          onClick={() => setShowSyncPanel(true)}
          className="px-3 py-1 rounded text-sm cursor-pointer transition-opacity hover:opacity-80 font-medium"
          style={{
            background: 'var(--accent-green)',
            color: '#fff',
          }}
        >
          同步數據
        </button>

        <button
          onClick={onSettingsClick}
          className="px-3 py-1 rounded text-sm cursor-pointer transition-opacity hover:opacity-80"
          style={{
            background: 'var(--bg-tertiary)',
            color: 'var(--text-secondary)',
          }}
        >
          設定
        </button>
      </div>
    </div>
  );
}
