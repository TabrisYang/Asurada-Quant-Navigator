/** 台股掃描面板 — 共用常數與元件（v154 由 TwBBScanPanel.tsx 純搬家拆分） */

import type { TwScanFailure } from '../../services/api';

export const PCTILE_OPTIONS = [
  { value: 10, label: '強壓縮 (<10%)' },
  { value: 15, label: '中度壓縮 (<15%)' },
  { value: 20, label: '寬鬆 (<20%)' },
  { value: 25, label: '極寬鬆 (<25%)' },
];


export function Th({ label, active, onClick, right = false }: {
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


export function FailureList({ failures, expanded, onToggle }: {
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

