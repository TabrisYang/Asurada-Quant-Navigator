/** 台股掃描 — 歷史分頁（v154 由 TwBBScanPanel.tsx 純搬家拆分，邏輯零改動） */

import { useState } from 'react';
import {
  type TwScanResult,
  type TwScanSummary,
  type TwScanRevisitItem,
  type TwScanFailure,
} from '../../services/api';
import { FailureList } from './shared';

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

export function HistoryView({ history, viewing, onOpen, onBack, onDelete, onRevisit, revisitLoading, revisitData, onAnalyze }: HistoryViewProps) {
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
