/** 阿斯拉量化系統 — 預測追蹤儀表板 */

import { useState, useEffect, useCallback } from 'react';
import { useChartStore } from '../../stores/chartStore';
import {
  fetchPredictionStats, fetchActivePredictions, fetchPredictionHistory,
  updatePredictionNote, generateReview,
} from '../../services/api';

interface PredictionItem {
  id: number;
  symbol: string;
  direction: 'long' | 'short';
  entry_price: number;
  target_price: number;
  stop_price: number;
  confidence: string;
  regime: string;
  indicators: string;
  created_at: string;
  status: string;
  actual_outcome_pct: number | null;
  notes?: string;
}

export default function PredictionDashboard() {
  const chartSymbol = useChartStore((s) => s.symbol);
  const ohlcvData = useChartStore((s) => s.ohlcvData);
  const latestPrice = ohlcvData && ohlcvData.length > 0 ? ohlcvData[ohlcvData.length - 1].close : null;

  const [stats, setStats] = useState<any>(null);
  const [directionStats, setDirectionStats] = useState<any>(null);
  const [streak, setStreak] = useState<any>(null);
  const [activePreds, setActivePreds] = useState<PredictionItem[]>([]);
  const [historyPreds, setHistoryPreds] = useState<PredictionItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [viewTab, setViewTab] = useState<'active' | 'history' | 'indicators'>('active');

  // 筆記編輯
  const [editingNoteId, setEditingNoteId] = useState<number | null>(null);
  const [noteText, setNoteText] = useState('');
  const [savingNote, setSavingNote] = useState(false);

  // 覆盤報告
  const [reviewReport, setReviewReport] = useState<string | null>(null);
  const [reviewLoading, setReviewLoading] = useState(false);
  const [showReview, setShowReview] = useState(false);

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [statsRes, activeRes, historyRes] = await Promise.all([
        fetchPredictionStats(chartSymbol || undefined),
        fetchActivePredictions(chartSymbol || undefined),
        fetchPredictionHistory(chartSymbol || undefined, 50),
      ]);
      setStats(statsRes.stats || null);
      setDirectionStats(statsRes.direction_stats || null);
      setStreak(statsRes.streak || null);
      setActivePreds(activeRes.predictions || []);
      setHistoryPreds(historyRes.predictions || []);
    } catch {
      /* ignore */
    } finally {
      setLoading(false);
    }
  }, [chartSymbol]);

  useEffect(() => { loadData(); }, [loadData]);

  const handleSaveNote = async (predId: number) => {
    setSavingNote(true);
    try {
      await updatePredictionNote(predId, noteText);
      // 更新本地狀態
      const updateList = (list: PredictionItem[]) =>
        list.map((p) => (p.id === predId ? { ...p, notes: noteText } : p));
      setActivePreds(updateList);
      setHistoryPreds(updateList);
      setEditingNoteId(null);
    } catch { /* ignore */ } finally {
      setSavingNote(false);
    }
  };

  const handleGenerateReview = async () => {
    setReviewLoading(true);
    setShowReview(true);
    try {
      const res = await generateReview(undefined, undefined, chartSymbol || undefined);
      setReviewReport(res.report || res.message || '生成失敗');
    } catch {
      setReviewReport('覆盤報告生成失敗，請確認 LLM 設定是否正確。');
    } finally {
      setReviewLoading(false);
    }
  };

  const calcPnl = (pred: PredictionItem): number | null => {
    if (!latestPrice || pred.symbol !== chartSymbol) return null;
    const pct = ((latestPrice - pred.entry_price) / pred.entry_price) * 100;
    return pred.direction === 'short' ? -pct : pct;
  };

  const statusLabel = (s: string) => {
    const map: Record<string, string> = { hit_target: 'V 命中', hit_stop: 'X 止損', expired: '- 到期' };
    return map[s] || s;
  };

  const statusColor = (s: string) => {
    if (s === 'hit_target') return '#4ade80';
    if (s === 'hit_stop') return '#f87171';
    return 'var(--text-secondary)';
  };

  const confidenceColor = (c: string) => {
    if (c === 'high') return '#4ade80';
    if (c === 'medium') return '#fbbf24';
    return 'var(--text-secondary)';
  };

  if (loading) {
    return <div className="text-center py-8" style={{ color: 'var(--text-secondary)' }}>載入預測資料中...</div>;
  }

  const totalPreds = stats?.total_predictions ?? 0;

  return (
    <div className="space-y-4">
      {/* ===== 統計概覽 ===== */}
      <div className="grid grid-cols-4 gap-3">
        <StatCard label="總預測數" value={totalPreds} />
        <StatCard
          label="加權勝率"
          value={stats?.weighted_win_rate != null ? `${(stats.weighted_win_rate * 100).toFixed(1)}%` : 'N/A'}
          color={stats?.weighted_win_rate >= 0.5 ? '#4ade80' : '#f87171'}
        />
        <StatCard label="命中目標" value={stats?.hit_target ?? 0} color="#4ade80" />
        <StatCard label="觸及止損" value={stats?.hit_stop ?? 0} color="#f87171" />
      </div>

      {/* 多空對比 */}
      {directionStats && (
        <div className="grid grid-cols-2 gap-3">
          <div className="rounded-lg p-3" style={{ background: 'var(--bg-tertiary)' }}>
            <div className="text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>做多勝率</div>
            <div className="text-lg font-bold" style={{ color: '#4ade80' }}>
              {directionStats.long?.win_rate != null
                ? `${(directionStats.long.win_rate * 100).toFixed(1)}%`
                : 'N/A'}
              <span className="text-xs font-normal ml-1" style={{ color: 'var(--text-secondary)' }}>
                ({directionStats.long?.total ?? 0} 筆)
              </span>
            </div>
          </div>
          <div className="rounded-lg p-3" style={{ background: 'var(--bg-tertiary)' }}>
            <div className="text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>做空勝率</div>
            <div className="text-lg font-bold" style={{ color: '#f87171' }}>
              {directionStats.short?.win_rate != null
                ? `${(directionStats.short.win_rate * 100).toFixed(1)}%`
                : 'N/A'}
              <span className="text-xs font-normal ml-1" style={{ color: 'var(--text-secondary)' }}>
                ({directionStats.short?.total ?? 0} 筆)
              </span>
            </div>
          </div>
        </div>
      )}

      {/* 連勝/連敗 */}
      {streak && (streak.current_streak !== 0) && (
        <div className="text-xs px-2" style={{ color: streak.current_streak > 0 ? '#4ade80' : '#f87171' }}>
          {streak.current_streak > 0
            ? `目前連勝 ${streak.current_streak} 筆`
            : `目前連敗 ${Math.abs(streak.current_streak)} 筆`}
        </div>
      )}

      {/* ===== Sub-tabs ===== */}
      <div className="flex gap-2 border-b" style={{ borderColor: 'var(--border-primary)' }}>
        {(['active', 'history', 'indicators'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setViewTab(tab)}
            className="px-3 py-1.5 text-xs"
            style={{
              color: viewTab === tab ? 'var(--accent-blue)' : 'var(--text-secondary)',
              borderBottom: viewTab === tab ? '2px solid var(--accent-blue)' : '2px solid transparent',
              background: 'transparent',
            }}
          >
            {tab === 'active' ? `進行中 (${activePreds.length})` : tab === 'history' ? `歷史記錄 (${historyPreds.length})` : '指標勝率'}
          </button>
        ))}
      </div>

      {/* ===== 進行中的預測 ===== */}
      {viewTab === 'active' && (
        <div className="space-y-2">
          {activePreds.length === 0 ? (
            <div className="text-center py-6 text-xs" style={{ color: 'var(--text-secondary)' }}>
              目前沒有進行中的預測。在對話中請 AI 分析進場點位即可自動產生。
            </div>
          ) : (
            activePreds.map((pred) => {
              const pnl = calcPnl(pred);
              return (
                <div key={pred.id} className="rounded-lg p-3 text-xs" style={{ background: 'var(--bg-tertiary)' }}>
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium" style={{ color: 'var(--text-primary)' }}>
                      {pred.symbol} {pred.direction === 'long' ? 'LONG' : 'SHORT'}
                    </span>
                    <span style={{ color: confidenceColor(pred.confidence) }}>
                      {pred.confidence}
                    </span>
                  </div>
                  <div className="grid grid-cols-3 gap-2 mb-1" style={{ color: 'var(--text-secondary)' }}>
                    <div>入場: {pred.entry_price}</div>
                    <div style={{ color: '#4ade80' }}>目標: {pred.target_price}</div>
                    <div style={{ color: '#f87171' }}>止損: {pred.stop_price}</div>
                  </div>
                  {pnl !== null && (
                    <div className="flex justify-between">
                      <span style={{ color: 'var(--text-secondary)' }}>即時盈虧</span>
                      <span className="font-medium" style={{ color: pnl >= 0 ? '#4ade80' : '#f87171' }}>
                        {pnl >= 0 ? '+' : ''}{pnl.toFixed(2)}%
                      </span>
                    </div>
                  )}
                  <div className="mt-1" style={{ color: 'var(--text-secondary)' }}>
                    指標: {pred.indicators} | {pred.regime} | {new Date(pred.created_at).toLocaleDateString()}
                  </div>
                  {/* 筆記區 */}
                  {editingNoteId === pred.id ? (
                    <div className="mt-2">
                      <textarea
                        value={noteText}
                        onChange={(e) => setNoteText(e.target.value)}
                        rows={2}
                        className="w-full text-xs rounded p-1.5"
                        style={{ background: 'var(--bg-secondary)', color: 'var(--text-primary)', border: '1px solid var(--border-primary)' }}
                        placeholder="寫下你的交易筆記..."
                      />
                      <div className="flex gap-2 mt-1">
                        <button
                          onClick={() => handleSaveNote(pred.id)}
                          disabled={savingNote}
                          className="text-xs px-2 py-0.5 rounded"
                          style={{ background: 'var(--accent-blue)', color: '#fff' }}
                        >
                          {savingNote ? '儲存中...' : '儲存'}
                        </button>
                        <button
                          onClick={() => setEditingNoteId(null)}
                          className="text-xs px-2 py-0.5 rounded"
                          style={{ color: 'var(--text-secondary)' }}
                        >
                          取消
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className="mt-1 flex items-center gap-2">
                      {pred.notes && (
                        <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                          {pred.notes}
                        </span>
                      )}
                      <button
                        onClick={() => { setEditingNoteId(pred.id); setNoteText(pred.notes || ''); }}
                        className="text-xs"
                        style={{ color: 'var(--accent-blue)' }}
                      >
                        {pred.notes ? '編輯筆記' : '+ 筆記'}
                      </button>
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>
      )}

      {/* ===== 歷史預測 ===== */}
      {viewTab === 'history' && (
        <div className="space-y-2">
          {historyPreds.length === 0 ? (
            <div className="text-center py-6 text-xs" style={{ color: 'var(--text-secondary)' }}>
              尚無歷史預測記錄
            </div>
          ) : (
            historyPreds.map((pred) => (
              <div key={pred.id} className="rounded-lg p-3 text-xs" style={{ background: 'var(--bg-tertiary)' }}>
                <div className="flex items-center justify-between mb-1">
                  <span className="font-medium" style={{ color: 'var(--text-primary)' }}>
                    {pred.symbol} {pred.direction === 'long' ? 'LONG' : 'SHORT'}
                  </span>
                  <span style={{ color: statusColor(pred.status) }}>
                    {statusLabel(pred.status)}
                  </span>
                </div>
                <div className="grid grid-cols-3 gap-2" style={{ color: 'var(--text-secondary)' }}>
                  <div>入場: {pred.entry_price}</div>
                  <div>目標: {pred.target_price}</div>
                  <div>
                    結果: <span style={{ color: (pred.actual_outcome_pct ?? 0) >= 0 ? '#4ade80' : '#f87171' }}>
                      {pred.actual_outcome_pct != null ? `${pred.actual_outcome_pct.toFixed(2)}%` : 'N/A'}
                    </span>
                  </div>
                </div>
                <div className="mt-1" style={{ color: 'var(--text-secondary)' }}>
                  {pred.indicators} | {pred.regime} | {new Date(pred.created_at).toLocaleDateString()}
                </div>
                {/* 筆記區 */}
                {editingNoteId === pred.id ? (
                  <div className="mt-2">
                    <textarea
                      value={noteText}
                      onChange={(e) => setNoteText(e.target.value)}
                      rows={2}
                      className="w-full text-xs rounded p-1.5"
                      style={{ background: 'var(--bg-secondary)', color: 'var(--text-primary)', border: '1px solid var(--border-primary)' }}
                      placeholder="寫下你的交易筆記..."
                    />
                    <div className="flex gap-2 mt-1">
                      <button
                        onClick={() => handleSaveNote(pred.id)}
                        disabled={savingNote}
                        className="text-xs px-2 py-0.5 rounded"
                        style={{ background: 'var(--accent-blue)', color: '#fff' }}
                      >
                        {savingNote ? '儲存中...' : '儲存'}
                      </button>
                      <button
                        onClick={() => setEditingNoteId(null)}
                        className="text-xs px-2 py-0.5 rounded"
                        style={{ color: 'var(--text-secondary)' }}
                      >
                        取消
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="mt-1 flex items-center gap-2">
                    {pred.notes && (
                      <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                        {pred.notes}
                      </span>
                    )}
                    <button
                      onClick={() => { setEditingNoteId(pred.id); setNoteText(pred.notes || ''); }}
                      className="text-xs"
                      style={{ color: 'var(--accent-blue)' }}
                    >
                      {pred.notes ? '編輯筆記' : '+ 筆記'}
                    </button>
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      )}

      {/* ===== 指標勝率排名 ===== */}
      {viewTab === 'indicators' && (
        <div>
          {stats?.indicator_performance && Object.keys(stats.indicator_performance).length > 0 ? (
            <table className="w-full text-xs">
              <thead>
                <tr style={{ color: 'var(--text-secondary)' }}>
                  <th className="text-left py-1">指標</th>
                  <th className="text-right py-1">勝率</th>
                  <th className="text-right py-1">樣本數</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(stats.indicator_performance as Record<string, { win_rate: number; samples: number }>)
                  .sort(([, a], [, b]) => b.win_rate - a.win_rate)
                  .map(([name, data]) => (
                    <tr key={name} style={{ borderTop: '1px solid var(--border-primary)' }}>
                      <td className="py-1.5" style={{ color: 'var(--text-primary)' }}>{name}</td>
                      <td className="text-right py-1.5" style={{ color: data.win_rate >= 0.5 ? '#4ade80' : '#f87171' }}>
                        {(data.win_rate * 100).toFixed(1)}%
                      </td>
                      <td className="text-right py-1.5" style={{ color: 'var(--text-secondary)' }}>
                        {data.samples}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          ) : (
            <div className="text-center py-6 text-xs" style={{ color: 'var(--text-secondary)' }}>
              累積更多預測後即可顯示各指標的勝率排名
            </div>
          )}
        </div>
      )}

      {/* 覆盤報告 + 刷新按鈕 */}
      <div className="flex justify-center gap-3">
        <button
          onClick={handleGenerateReview}
          disabled={reviewLoading}
          className="text-xs px-3 py-1 rounded"
          style={{ background: 'var(--accent-blue)', color: '#fff' }}
        >
          {reviewLoading ? '生成中...' : '生成覆盤報告'}
        </button>
        <button
          onClick={loadData}
          className="text-xs px-3 py-1 rounded"
          style={{ color: 'var(--accent-blue)', background: 'transparent' }}
        >
          重新載入
        </button>
      </div>

      {/* 覆盤報告 Modal */}
      {showReview && (
        <div
          style={{
            position: 'fixed', inset: 0, zIndex: 9999,
            background: 'rgba(0,0,0,0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}
          onClick={() => setShowReview(false)}
        >
          <div
            className="rounded-lg p-5"
            style={{
              background: 'var(--bg-secondary)', maxWidth: 640, width: '90%', maxHeight: '80vh',
              overflow: 'auto', color: 'var(--text-primary)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-bold">覆盤報告</h3>
              <button
                onClick={() => setShowReview(false)}
                className="text-xs px-2 py-0.5 rounded"
                style={{ color: 'var(--text-secondary)' }}
              >
                關閉
              </button>
            </div>
            {reviewLoading ? (
              <div className="text-center py-8 text-xs" style={{ color: 'var(--text-secondary)' }}>
                AI 正在分析預測記錄，生成覆盤報告...
              </div>
            ) : (
              <div className="text-xs whitespace-pre-wrap leading-relaxed" style={{ color: 'var(--text-primary)' }}>
                {reviewReport}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}


function StatCard({ label, value, color }: { label: string; value: string | number; color?: string }) {
  return (
    <div className="rounded-lg p-3 text-center" style={{ background: 'var(--bg-tertiary)' }}>
      <div className="text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>{label}</div>
      <div className="text-lg font-bold" style={{ color: color || 'var(--text-primary)' }}>
        {value}
      </div>
    </div>
  );
}
