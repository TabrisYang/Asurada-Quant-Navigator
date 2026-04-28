/** 阿斯拉量化系統 — 預測追蹤儀表板 */

import { useState, useEffect, useCallback } from 'react';
import { useChartStore } from '../../stores/chartStore';
import {
  fetchPredictionStats, fetchActivePredictions, fetchPredictionHistory,
  updatePredictionNote, generateReview, clearPredictions,
  fetchScenarios, fetchAdjustments, recalculateAdjustments,
  fetchReviewHistory,
} from '../../services/api';
import { loadPersistedSession } from '../../services/session';

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
  validated_at?: string;
  status: string;
  actual_outcome_pct: number | null;
  notes?: string;
}

/** 格式化 ISO 時間為台北時區 MM/DD HH:mm */
function fmtTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleString('zh-TW', {
    timeZone: 'Asia/Taipei',
    month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit',
    hour12: false,
  });
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
  const [viewTab, setViewTab] = useState<'scenarios' | 'active' | 'history' | 'indicators' | 'adjustments' | 'reviews' | 'calibration' | 'imitation' | 'strategy'>('scenarios');
  // ★ v100：Calibration 數據
  const [calibrationData, setCalibrationData] = useState<any>(null);
  const [calibrationLoading, setCalibrationLoading] = useState(false);
  // ★ v101：模仿學習模型狀態
  const [imitationStatus, setImitationStatus] = useState<any>(null);
  const [imitationLoading, setImitationLoading] = useState(false);

  // ★ v103 4：Dashboard 觀察工具（AUC 趨勢 / SHAP 統計 / 分歧）
  const [aucHistory, setAucHistory] = useState<any[]>([]);
  const [shapTopFeatures, setShapTopFeatures] = useState<{ items: { name: string; count: number }[]; total_logs: number; days: number } | null>(null);
  const [divergenceStats, setDivergenceStats] = useState<any>(null);

  // ★ v103 5：策略績效 + 跨 symbol RS
  const [strategyPerf, setStrategyPerf] = useState<any>(null);
  const [crossSymbolRS, setCrossSymbolRS] = useState<any>(null);
  const [strategyLoading, setStrategyLoading] = useState(false);

  // 情境預測
  const [scenarioData, setScenarioData] = useState<any>(null);
  const [scenarioLoading, setScenarioLoading] = useState(false);
  const [scenarioError, setScenarioError] = useState<string | null>(null);

  // 自動調整
  const [adjustments, setAdjustments] = useState<any[]>([]);
  const [adjLoading, setAdjLoading] = useState(false);

  // 歷史覆盤報告
  const [reviews, setReviews] = useState<any[]>([]);
  const [reviewsLoading, setReviewsLoading] = useState(false);

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
      // 各自獨立 try/catch，避免一個失敗導致全部資料消失
      const [statsResult, activeResult, historyResult] = await Promise.allSettled([
        fetchPredictionStats(chartSymbol || undefined),
        fetchActivePredictions(chartSymbol || undefined),
        fetchPredictionHistory(chartSymbol || undefined, 50),
      ]);
      if (statsResult.status === 'fulfilled') {
        setStats(statsResult.value.stats || null);
        setDirectionStats(statsResult.value.direction_stats || null);
        setStreak(statsResult.value.streak || null);
      }
      if (activeResult.status === 'fulfilled') {
        setActivePreds(activeResult.value.predictions || []);
      }
      if (historyResult.status === 'fulfilled') {
        setHistoryPreds(historyResult.value.predictions || []);
      }
    } catch {
      /* ignore */
    } finally {
      setLoading(false);
    }
  }, [chartSymbol]);

  const loadScenarios = useCallback(async () => {
    if (!chartSymbol) return;
    setScenarioLoading(true);
    setScenarioError(null);
    try {
      const timeframe = useChartStore.getState().timeframe || '1d';
      const res = await fetchScenarios(chartSymbol, timeframe);
      setScenarioData(res);
    } catch (e: any) {
      const msg = e?.response?.data?.detail || e?.message || '載入失敗';
      setScenarioError(msg);
    } finally {
      setScenarioLoading(false);
    }
  }, [chartSymbol]);

  const loadAdjustments = useCallback(async () => {
    setAdjLoading(true);
    try {
      const res = await fetchAdjustments(chartSymbol || undefined);
      setAdjustments(res.adjustments || []);
    } catch { setAdjustments([]); }
    finally { setAdjLoading(false); }
  }, [chartSymbol]);

  const handleRecalculate = async () => {
    setAdjLoading(true);
    try {
      await recalculateAdjustments(chartSymbol || undefined);
      await loadAdjustments();
    } catch { /* ignore */ }
    finally { setAdjLoading(false); }
  };

  const loadReviews = useCallback(async () => {
    setReviewsLoading(true);
    try {
      const res = await fetchReviewHistory(10);
      setReviews(res.reviews || []);
    } catch { setReviews([]); }
    finally { setReviewsLoading(false); }
  }, []);

  // ★ v100：載入 calibration 數據
  const loadCalibration = useCallback(async () => {
    setCalibrationLoading(true);
    try {
      const params = chartSymbol ? `?symbol=${encodeURIComponent(chartSymbol)}&days=90` : '?days=90';
      const res = await fetch(`/api/predictions/calibration${params}`);
      const data = await res.json();
      setCalibrationData(data);
    } catch { setCalibrationData(null); }
    finally { setCalibrationLoading(false); }
  }, [chartSymbol]);

  // ★ v101：載入模型狀態
  const loadImitation = useCallback(async () => {
    setImitationLoading(true);
    try {
      const res = await fetch('/api/predictions/imitation/status');
      const data = await res.json();
      setImitationStatus(data);
    } catch { setImitationStatus(null); }
    finally { setImitationLoading(false); }
  }, []);

  // ★ v103 5：策略績效 + 跨 symbol RS
  const loadStrategyView = useCallback(async () => {
    setStrategyLoading(true);
    try {
      const [perfRes, rsRes] = await Promise.all([
        fetch('/api/predictions/strategy_performance?days=90'),
        fetch('/api/predictions/cross_symbol_rs?timeframe=4h&days=30'),
      ]);
      setStrategyPerf(await perfRes.json());
      setCrossSymbolRS(await rsRes.json());
    } catch {
      setStrategyPerf(null);
      setCrossSymbolRS(null);
    } finally {
      setStrategyLoading(false);
    }
  }, []);

  // ★ v103 4：Dashboard 觀察工具（AUC 趨勢 / SHAP 統計 / 分歧）
  const loadImitationObservatory = useCallback(async () => {
    try {
      const [aucRes, shapRes, divRes] = await Promise.all([
        fetch('/api/predictions/imitation/auc_history?limit=12'),
        fetch('/api/predictions/imitation/shap_top_features?days=30'),
        fetch('/api/predictions/imitation/divergence_stats'),
      ]);
      const aucData = await aucRes.json();
      const shapData = await shapRes.json();
      const divData = await divRes.json();
      setAucHistory(aucData.items ?? []);
      setShapTopFeatures(shapData ?? null);
      setDivergenceStats(divData ?? null);
    } catch {
      setAucHistory([]); setShapTopFeatures(null); setDivergenceStats(null);
    }
  }, []);

  useEffect(() => { loadData(); }, [loadData]);
  useEffect(() => { if (viewTab === 'scenarios') loadScenarios(); }, [viewTab, loadScenarios]);
  useEffect(() => { if (viewTab === 'adjustments') loadAdjustments(); }, [viewTab, loadAdjustments]);
  useEffect(() => { if (viewTab === 'reviews') loadReviews(); }, [viewTab, loadReviews]);
  useEffect(() => { if (viewTab === 'calibration') loadCalibration(); }, [viewTab, loadCalibration]);
  useEffect(() => {
    if (viewTab === 'imitation') {
      loadImitation();
      loadImitationObservatory();
    }
  }, [viewTab, loadImitation, loadImitationObservatory]);
  useEffect(() => { if (viewTab === 'strategy') loadStrategyView(); }, [viewTab, loadStrategyView]);

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
      const session = loadPersistedSession();
      const res = await generateReview(undefined, undefined, chartSymbol || undefined, session?.sessionId);
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

  const totalPreds = stats?.total ?? 0;

  return (
    <div className="space-y-4">
      {/* ★ v100：免責聲明 */}
      <div className="p-2 rounded text-xs" style={{ background: 'rgba(251,191,36,0.1)', border: '1px solid rgba(251,191,36,0.3)', color: 'var(--text-secondary)' }}>
        ⚠️ 系統判讀僅供參考，不構成投資建議。歷史命中率不保證未來表現，請依自身判斷下單。
      </div>

      {/* ===== 統計概覽 ===== */}
      <div className="grid grid-cols-4 gap-3">
        <StatCard label="總預測數" value={totalPreds} />
        <StatCard
          label="加權勝率"
          value={stats?.win_rate_weighted != null ? `${stats.win_rate_weighted.toFixed(1)}%` : 'N/A'}
          color={stats?.win_rate_weighted >= 50 ? '#4ade80' : '#f87171'}
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
                ? `${directionStats.long.win_rate.toFixed(1)}%`
                : 'N/A'}
              <span className="text-xs font-normal ml-1" style={{ color: 'var(--text-secondary)' }}>
                ({directionStats.long?.samples ?? 0} 筆)
              </span>
            </div>
          </div>
          <div className="rounded-lg p-3" style={{ background: 'var(--bg-tertiary)' }}>
            <div className="text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>做空勝率</div>
            <div className="text-lg font-bold" style={{ color: '#f87171' }}>
              {directionStats.short?.win_rate != null
                ? `${directionStats.short.win_rate.toFixed(1)}%`
                : 'N/A'}
              <span className="text-xs font-normal ml-1" style={{ color: 'var(--text-secondary)' }}>
                ({directionStats.short?.samples ?? 0} 筆)
              </span>
            </div>
          </div>
        </div>
      )}

      {/* 連勝/連敗 */}
      {streak && streak.current_streak > 0 && (
        <div className="text-xs px-2" style={{ color: streak.streak_type === 'hit_target' ? '#4ade80' : '#f87171' }}>
          {streak.streak_type === 'hit_target'
            ? `目前連勝 ${streak.current_streak} 筆`
            : `目前連敗 ${streak.current_streak} 筆`}
        </div>
      )}

      {/* ===== Sub-tabs ===== */}
      <div className="flex gap-2 border-b overflow-x-auto" style={{ borderColor: 'var(--border-primary)' }}>
        {(['scenarios', 'active', 'history', 'calibration', 'imitation', 'strategy', 'reviews', 'indicators', 'adjustments'] as const).map((tab) => {
          const labels: Record<string, string> = {
            scenarios: '情境預測',
            active: `進行中 (${activePreds.length})`,
            history: `歷史記錄 (${historyPreds.length})`,
            calibration: '📊 命中率',
            imitation: '🤖 模型狀態',
            strategy: '📊 策略績效',
            reviews: '覆盤報告',
            indicators: '指標勝率',
            adjustments: '自動調整',
          };
          return (
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
              {labels[tab]}
            </button>
          );
        })}
      </div>

      {/* ===== 情境預測 ===== */}
      {viewTab === 'scenarios' && (
        <div className="space-y-3">
          {scenarioLoading ? (
            <div className="text-center py-8 text-xs" style={{ color: 'var(--text-secondary)' }}>
              正在分析多源訊號，產出情境預測...
            </div>
          ) : scenarioError ? (
            <div className="text-center py-6 space-y-2">
              <div className="text-xs" style={{ color: '#f87171' }}>{scenarioError}</div>
              <button
                onClick={loadScenarios}
                className="text-xs px-3 py-1 rounded"
                style={{ background: 'var(--accent-blue)', color: '#fff' }}
              >
                重試
              </button>
            </div>
          ) : scenarioData ? (
            <>
              <div className="text-xs px-1" style={{ color: 'var(--text-secondary)' }}>
                {scenarioData.symbol} | 當前價格: {scenarioData.current_price} | 數據: {scenarioData.data_points} 根 K 線
              </div>

              {scenarioData.scenarios?.map((sc: any, i: number) => {
                const arrow = sc.direction === 'bullish' ? '\u25B2' : sc.direction === 'bearish' ? '\u25BC' : '\u25AC';
                const dirColor = sc.direction === 'bullish' ? '#4ade80' : sc.direction === 'bearish' ? '#f87171' : '#fbbf24';
                const riskColor = sc.risk_level === 'low' ? '#4ade80' : sc.risk_level === 'high' ? '#f87171' : '#fbbf24';
                return (
                  <div key={i} className="rounded-lg p-3" style={{ background: 'var(--bg-tertiary)', borderLeft: `3px solid ${dirColor}` }}>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm font-bold" style={{ color: dirColor }}>
                        {arrow} {sc.label} ({sc.probability_pct})
                      </span>
                      <span className="text-xs px-1.5 py-0.5 rounded" style={{ color: riskColor, border: `1px solid ${riskColor}` }}>
                        {sc.risk_level === 'low' ? '低風險' : sc.risk_level === 'high' ? '高風險' : '中風險'}
                      </span>
                    </div>
                    <div className="text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>
                      目標區間: {sc.price_target?.low} ~ {sc.price_target?.high}
                    </div>
                    {sc.invalidation && (
                      <div className="text-xs mb-1" style={{ color: '#f87171' }}>
                        失效條件: {sc.invalidation}
                      </div>
                    )}
                    {sc.supporting_signals?.length > 0 && (
                      <div className="mt-1 space-y-0.5">
                        {sc.supporting_signals.map((sig: any, j: number) => (
                          <div key={j} className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                            {'\u21B3'} {sig.name}: {sig.interpretation}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}

              {/* 信心來源 */}
              {scenarioData.signal_sources?.weights && (
                <div className="rounded-lg p-3 text-xs" style={{ background: 'var(--bg-tertiary)' }}>
                  <div className="mb-1 font-medium" style={{ color: 'var(--text-primary)' }}>信心來源權重</div>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(scenarioData.signal_sources.weights as Record<string, number>).map(([k, v]) => {
                      const labels: Record<string, string> = { ml: 'ML 模型', technical: '技術指標', historical: '歷史相似度', regime: '市場結構' };
                      return (
                        <span key={k} className="px-2 py-0.5 rounded" style={{ background: 'var(--bg-secondary)', color: 'var(--text-secondary)' }}>
                          {labels[k] || k}: {(v * 100).toFixed(0)}%
                        </span>
                      );
                    })}
                  </div>
                  <div className="mt-1" style={{ color: 'var(--text-secondary)' }}>
                    預測有效期: {scenarioData.scenarios?.[0]?.timeframe_bars || 5} 根 K 線 |
                    生成時間: {scenarioData.generated_at}
                  </div>
                </div>
              )}

              <div className="flex justify-center">
                <button
                  onClick={loadScenarios}
                  className="text-xs px-3 py-1 rounded"
                  style={{ background: 'var(--accent-blue)', color: '#fff' }}
                >
                  重新分析
                </button>
              </div>
            </>
          ) : (
            <div className="text-center py-6 text-xs" style={{ color: 'var(--text-secondary)' }}>
              點擊上方分頁載入情境預測
            </div>
          )}
        </div>
      )}

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
                    指標: {pred.indicators} | {pred.regime} | 入場: {fmtTime(pred.created_at)}
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
                  {pred.indicators} | {pred.regime}
                </div>
                <div className="mt-0.5" style={{ color: 'var(--text-secondary)' }}>
                  入場: {fmtTime(pred.created_at)}{pred.validated_at ? ` | 出場: ${fmtTime(pred.validated_at)}` : ''}
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

      {/* ===== v100：Calibration 命中率 ===== */}
      {viewTab === 'calibration' && (
        <div className="space-y-3">
          <div className="p-2 rounded text-xs" style={{ background: 'rgba(248,81,73,0.08)', border: '1px solid rgba(248,81,73,0.3)', color: 'var(--text-secondary)' }}>
            ⚠️ 歷史命中率不保證未來表現。樣本數少（n &lt; 10）時 CI 寬，僅作參考。
          </div>

          {calibrationLoading ? (
            <div className="text-center py-8 text-xs" style={{ color: 'var(--text-secondary)' }}>正在載入命中率數據...</div>
          ) : !calibrationData || calibrationData.total === 0 ? (
            <div className="text-center py-8 text-xs" style={{ color: 'var(--text-secondary)' }}>
              尚無已驗證預測。系統會在預測時間到期後自動驗證並更新此面板。
            </div>
          ) : (
            <>
              {/* 整體命中率卡 */}
              <div className="p-3 rounded" style={{ background: 'var(--bg-tertiary)' }}>
                <div className="text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>整體命中率（90 天加權）</div>
                <div className="font-mono text-2xl" style={{ color: 'var(--accent-blue)' }}>
                  {calibrationData.win_rate_weighted ?? 0}%
                </div>
                <div className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>
                  樣本：{calibrationData.total} 筆
                  {calibrationData.bayesian?.credible_interval_95 && (
                    <span className="ml-2">| Bayesian 95% CI：{calibrationData.bayesian.credible_interval_95[0]}-{calibrationData.bayesian.credible_interval_95[1]}%</span>
                  )}
                </div>
              </div>

              {/* Brier + ECE */}
              {calibrationData.calibration && (
                <div className="grid grid-cols-2 gap-2">
                  <div className="p-2 rounded text-xs" style={{ background: 'var(--bg-tertiary)' }}>
                    <div style={{ color: 'var(--text-secondary)' }}>Brier Score</div>
                    <div className="font-mono text-base" style={{ color: 'var(--text-primary)' }}>
                      {calibrationData.calibration.brier_score?.toFixed(3) ?? 'N/A'}
                    </div>
                    <div style={{ color: 'var(--text-secondary)', fontSize: '10px' }}>
                      {calibrationData.calibration.brier_score && calibrationData.calibration.brier_score < 0.25 ? '✅ 校準良好' : '⚠️ 校準偏差'}
                    </div>
                  </div>
                  <div className="p-2 rounded text-xs" style={{ background: 'var(--bg-tertiary)' }}>
                    <div style={{ color: 'var(--text-secondary)' }}>ECE</div>
                    <div className="font-mono text-base" style={{ color: 'var(--text-primary)' }}>
                      {calibrationData.calibration.ece?.toFixed(3) ?? 'N/A'}
                    </div>
                    <div style={{ color: 'var(--text-secondary)', fontSize: '10px' }}>
                      預期校準誤差（越低越好）
                    </div>
                  </div>
                </div>
              )}

              {/* 信心校準 */}
              {calibrationData.confidence_calibration && Object.keys(calibrationData.confidence_calibration).length > 0 && (
                <div className="p-2 rounded" style={{ background: 'var(--bg-tertiary)' }}>
                  <div className="text-xs mb-2 font-medium" style={{ color: 'var(--text-primary)' }}>信心校準（看「高信心」是否真的更準）</div>
                  <table className="w-full text-xs">
                    <thead><tr style={{ color: 'var(--text-secondary)' }}>
                      <th className="text-left py-1">信心</th><th className="text-right">命中率</th><th className="text-right">樣本</th>
                    </tr></thead>
                    <tbody>
                      {Object.entries(calibrationData.confidence_calibration).map(([conf, data]: any) => (
                        <tr key={conf}><td className="py-1">{conf === 'high' ? '高' : conf === 'medium' ? '中' : '低'}</td>
                          <td className="text-right font-mono" style={{ color: 'var(--accent-blue)' }}>{data.win_rate}%</td>
                          <td className="text-right" style={{ color: 'var(--text-secondary)' }}>{data.samples}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* 按 regime 拆分 */}
              {calibrationData.by_regime && Object.keys(calibrationData.by_regime).length > 0 && (
                <div className="p-2 rounded" style={{ background: 'var(--bg-tertiary)' }}>
                  <div className="text-xs mb-2 font-medium" style={{ color: 'var(--text-primary)' }}>按市場 regime 拆分</div>
                  <table className="w-full text-xs">
                    <thead><tr style={{ color: 'var(--text-secondary)' }}>
                      <th className="text-left py-1">Regime</th><th className="text-right">命中率</th><th className="text-right">樣本</th>
                    </tr></thead>
                    <tbody>
                      {Object.entries(calibrationData.by_regime).map(([rg, data]: any) => (
                        <tr key={rg}><td className="py-1">{rg}</td>
                          <td className="text-right font-mono" style={{ color: 'var(--accent-blue)' }}>{data.win_rate}%</td>
                          <td className="text-right" style={{ color: 'var(--text-secondary)' }}>{data.samples}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Reliability curve */}
              {calibrationData.calibration?.reliability_curve && calibrationData.calibration.reliability_curve.length > 0 && (
                <div className="p-2 rounded" style={{ background: 'var(--bg-tertiary)' }}>
                  <div className="text-xs mb-2 font-medium" style={{ color: 'var(--text-primary)' }}>可靠性曲線（預測信心 vs 實際命中率）</div>
                  <table className="w-full text-xs">
                    <thead><tr style={{ color: 'var(--text-secondary)' }}>
                      <th className="text-left py-1">信心區間</th><th className="text-right">實際命中率</th><th className="text-right">樣本</th>
                    </tr></thead>
                    <tbody>
                      {calibrationData.calibration.reliability_curve.map((bin: any, i: number) => (
                        <tr key={i}><td className="py-1">{bin.bin_label || `bin ${i+1}`}</td>
                          <td className="text-right font-mono" style={{ color: 'var(--accent-blue)' }}>
                            {(bin.actual_win_rate * 100).toFixed(1)}%
                          </td>
                          <td className="text-right" style={{ color: 'var(--text-secondary)' }}>{bin.samples}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* ===== v101：模型狀態（Champion-Challenger）===== */}
      {viewTab === 'imitation' && (
        <div className="space-y-3">
          <div className="p-2 rounded text-xs" style={{ background: 'rgba(56,139,253,0.1)', border: '1px solid var(--accent-blue)', color: 'var(--text-secondary)' }}>
            🤖 v101 模仿學習：當 Quality Gate 通過後才會啟用，使用者可能仍看到 v100 體驗。
          </div>

          {imitationLoading ? (
            <div className="text-center py-8 text-xs" style={{ color: 'var(--text-secondary)' }}>正在載入...</div>
          ) : !imitationStatus ? (
            <div className="text-center py-8 text-xs" style={{ color: 'var(--text-secondary)' }}>無資料</div>
          ) : (
            <>
              {/* Canary 狀態 */}
              <div className="p-3 rounded" style={{ background: 'var(--bg-tertiary)' }}>
                <div className="text-xs mb-2 font-medium" style={{ color: 'var(--text-primary)' }}>Canary 狀態</div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>使用者啟用：<span style={{ color: imitationStatus.canary?.active_for_users ? 'var(--accent-green, #4ade80)' : 'var(--text-secondary)' }}>
                    {imitationStatus.canary?.active_for_users ? '✓ ON' : '○ OFF（看 v100）'}
                  </span></div>
                  <div>SHADOW MODE：<span style={{ color: imitationStatus.canary?.shadow_mode ? 'var(--accent-blue)' : 'var(--text-secondary)' }}>
                    {imitationStatus.canary?.shadow_mode ? '✓ ON' : '○ OFF'}
                  </span></div>
                  <div>Quality Gate：<span style={{ color: imitationStatus.canary?.quality_gate_passed ? 'var(--accent-green, #4ade80)' : '#fbbf24' }}>
                    {imitationStatus.canary?.quality_gate_passed ? '✓ 通過' : '⏳ 等待中'}
                  </span></div>
                  <div>Canary %：<span className="font-mono" style={{ color: 'var(--accent-blue)' }}>{imitationStatus.canary?.canary_pct ?? 0}%</span></div>
                </div>
              </div>

              {/* Champion 模型 */}
              {imitationStatus.champion ? (
                <div className="p-3 rounded" style={{ background: 'var(--bg-tertiary)' }}>
                  <div className="text-xs mb-2 font-medium" style={{ color: 'var(--accent-green, #4ade80)' }}>👑 Champion 模型 v{imitationStatus.champion.version}</div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>樣本量：<span className="font-mono">{imitationStatus.champion.trainset_n}</span></div>
                    <div>Lockbox AUC：<span className="font-mono" style={{ color: 'var(--accent-blue)' }}>{imitationStatus.champion.lockbox_auc?.toFixed(3) || 'N/A'}</span></div>
                    <div>OOS AUC：<span className="font-mono">{imitationStatus.champion.auc?.toFixed(3) || 'N/A'}</span></div>
                    <div>Brier：<span className="font-mono">{imitationStatus.champion.brier?.toFixed(3) || 'N/A'}</span></div>
                  </div>
                  <div className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>訓練於 {imitationStatus.champion.trained_at}</div>

                  {/* Feature importance top 10 */}
                  {imitationStatus.champion.feature_importance && (
                    <div className="mt-2">
                      <div className="text-xs font-medium" style={{ color: 'var(--text-primary)' }}>Top 10 重要特徵</div>
                      <div className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>
                        {Object.entries(imitationStatus.champion.feature_importance as Record<string, number>)
                          .sort(([, a], [, b]) => Number(b) - Number(a))
                          .slice(0, 10)
                          .map(([k, v]) => `${k}: ${Number(v).toFixed(0)}`)
                          .join(' ｜ ')}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="p-3 rounded text-xs" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}>
                  無 Champion 模型 — 樣本累積到 50+ 後可手動觸發訓練
                </div>
              )}

              {/* Stable Fallback */}
              {imitationStatus.stable_fallback && (
                <div className="p-2 rounded text-xs" style={{ background: 'var(--bg-tertiary)' }}>
                  🛟 Stable Fallback v{imitationStatus.stable_fallback.version}（AUC {imitationStatus.stable_fallback.lockbox_auc?.toFixed(3)}）
                </div>
              )}

              {/* ★ v103 4：模型觀察工具 */}
              <div className="p-3 rounded text-xs space-y-3" style={{ background: 'var(--bg-secondary)' }}>
                <div className="font-medium" style={{ color: 'var(--accent-blue)' }}>📊 模型觀察（v103）</div>

                {/* 4A：AUC 趨勢 sparkline */}
                <div>
                  <div className="mb-1" style={{ color: 'var(--text-secondary)' }}>
                    Lockbox AUC 趨勢（最近 {aucHistory.length} 次訓練）
                  </div>
                  {aucHistory.length === 0 ? (
                    <div style={{ color: 'var(--text-secondary)' }}>尚無訓練紀錄</div>
                  ) : (
                    (() => {
                      const W = 280, H = 50, P = 4;
                      const vals = aucHistory.map(p => Number(p.lockbox_auc) || 0.5);
                      const minV = Math.min(0.4, ...vals);
                      const maxV = Math.max(0.8, ...vals);
                      const range = Math.max(0.01, maxV - minV);
                      const pts = vals.map((v, i) => {
                        const x = P + (i * (W - 2 * P)) / Math.max(1, vals.length - 1);
                        const y = H - P - ((v - minV) / range) * (H - 2 * P);
                        return `${x.toFixed(1)},${y.toFixed(1)}`;
                      }).join(' ');
                      const last = vals[vals.length - 1];
                      const first = vals[0];
                      const delta = last - first;
                      return (
                        <div>
                          <svg width={W} height={H} style={{ display: 'block' }}>
                            <line x1={P} y1={H / 2} x2={W - P} y2={H / 2} stroke="var(--text-secondary)" strokeOpacity="0.2" strokeDasharray="2 2" />
                            <polyline points={pts} fill="none" stroke="var(--accent-blue)" strokeWidth="1.5" />
                            {vals.map((v, i) => {
                              const x = P + (i * (W - 2 * P)) / Math.max(1, vals.length - 1);
                              const y = H - P - ((v - minV) / range) * (H - 2 * P);
                              return <circle key={i} cx={x} cy={y} r="2" fill="var(--accent-blue)" />;
                            })}
                          </svg>
                          <div className="mt-1" style={{ color: 'var(--text-secondary)' }}>
                            起點 {first.toFixed(3)} → 最新 {last.toFixed(3)}（
                            <span style={{ color: delta >= 0 ? '#3fb950' : '#f85149' }}>{delta >= 0 ? '+' : ''}{delta.toFixed(3)}</span>
                            ）
                          </div>
                        </div>
                      );
                    })()
                  )}
                </div>

                {/* 4B：SHAP top features 統計 */}
                <div>
                  <div className="mb-1" style={{ color: 'var(--text-secondary)' }}>
                    過去 {shapTopFeatures?.days ?? 30} 天最常被引用 feature（共 {shapTopFeatures?.total_logs ?? 0} 次推論）
                  </div>
                  {!shapTopFeatures || shapTopFeatures.items.length === 0 ? (
                    <div style={{ color: 'var(--text-secondary)' }}>尚無 SHAP 紀錄</div>
                  ) : (
                    <div className="space-y-1">
                      {shapTopFeatures.items.slice(0, 8).map((it) => {
                        const max = shapTopFeatures.items[0].count || 1;
                        const pct = (it.count / max) * 100;
                        return (
                          <div key={it.name} className="flex items-center gap-2">
                            <span className="font-mono" style={{ width: 110, fontSize: 10 }}>{it.name}</span>
                            <div style={{ flex: 1, height: 8, background: 'var(--bg-tertiary)', borderRadius: 2, overflow: 'hidden' }}>
                              <div style={{ width: `${pct}%`, height: '100%', background: 'var(--accent-blue)' }} />
                            </div>
                            <span className="font-mono" style={{ width: 24, textAlign: 'right' }}>{it.count}</span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>

                {/* 4C：模型 vs 規則分歧次數 */}
                <div>
                  <div className="mb-1" style={{ color: 'var(--text-secondary)' }}>
                    ML vs 規則分歧次數（|p_ml - p_rule| &gt; 0.2）
                  </div>
                  {!divergenceStats ? (
                    <div style={{ color: 'var(--text-secondary)' }}>無資料</div>
                  ) : (
                    <div className="grid grid-cols-3 gap-2">
                      {(['7d', '30d', 'all'] as const).map((k) => {
                        const s = divergenceStats[k];
                        if (!s) return null;
                        const pct = ((s.ratio || 0) * 100).toFixed(1);
                        return (
                          <div key={k} className="p-2 rounded" style={{ background: 'var(--bg-tertiary)' }}>
                            <div style={{ color: 'var(--text-secondary)' }}>{k === 'all' ? '全期' : k}</div>
                            <div className="font-mono" style={{ color: 'var(--accent-blue)' }}>
                              {s.divergent} / {s.total}
                            </div>
                            <div style={{ color: 'var(--text-secondary)' }}>{pct}%</div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>

              {/* ★ v102：強制啟用 v101（跳過 Quality Gate）*/}
              <button
                onClick={async () => {
                  if (!confirm(
                    '強制啟用 v101？\n\n' +
                    '✓ 使用者「全部分析」會看到 🤖 RL 戰略結論段（含 SHAP 解釋 + 路徑類比）\n' +
                    '✓ 透過 subprocess 隔離（主進程不會 segfault）\n' +
                    '⚠️ 跳過 Quality Gate 檢查（樣本量 / AUC / Brier 不再強制）\n\n' +
                    '若要退出：按下方「停用 v101」即可'
                  )) return;
                  const res = await fetch('/api/predictions/imitation/force_enable', { method: 'POST' });
                  const data = await res.json();
                  alert(`✅ ${data.message}`);
                  loadImitation();
                }}
                className="w-full py-2 rounded text-xs font-medium cursor-pointer mb-2"
                style={{ background: 'rgba(63,185,80,0.2)', color: '#3fb950', border: '1px solid #3fb950' }}
              >🚀 強制啟用 v101（讓使用者看到 RL 戰略結論段）</button>

              {/* 控制按鈕 */}
              <div className="flex gap-2">
                <button
                  onClick={async () => {
                    if (!confirm('立即重訓？需 30-60 秒。')) return;
                    const res = await fetch('/api/predictions/imitation/retrain', { method: 'POST' });
                    const data = await res.json();
                    alert(`結果：${data.status}\n${JSON.stringify(data, null, 2).slice(0, 500)}`);
                    loadImitation();
                  }}
                  className="flex-1 py-1.5 rounded text-xs cursor-pointer"
                  style={{ background: 'var(--accent-blue)', color: '#fff' }}
                >立即重訓</button>
                <button
                  onClick={async () => {
                    if (!confirm('回到 stable_fallback 模型？')) return;
                    await fetch('/api/predictions/imitation/rollback', { method: 'POST' });
                    loadImitation();
                  }}
                  className="flex-1 py-1.5 rounded text-xs cursor-pointer"
                  style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
                >退回 Stable</button>
                <button
                  onClick={async () => {
                    if (!confirm('停用 v101？所有使用者立刻退回 v100。')) return;
                    await fetch('/api/predictions/imitation/disable', { method: 'POST' });
                    loadImitation();
                  }}
                  className="flex-1 py-1.5 rounded text-xs cursor-pointer"
                  style={{ background: 'rgba(248,81,73,0.2)', color: '#f85149' }}
                >停用 v101</button>
              </div>

              {/* Quality Gate 細節 */}
              {imitationStatus.last_quality_gate && (
                <div className="p-2 rounded text-xs" style={{ background: 'var(--bg-tertiary)' }}>
                  <div className="font-medium mb-1" style={{ color: 'var(--text-primary)' }}>最近 Quality Gate 評估</div>
                  <div style={{ color: 'var(--text-secondary)' }}>
                    {imitationStatus.last_quality_gate.evaluated_at} - {imitationStatus.last_quality_gate.action_taken}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}

      {/* ===== v103 5：策略績效 + 跨 symbol RS ===== */}
      {viewTab === 'strategy' && (
        <div className="space-y-4">
          {strategyLoading ? (
            <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>載入中…</div>
          ) : (
            <>
              {/* 5A：策略類型勝率 */}
              <div>
                <div className="text-xs mb-2 font-medium" style={{ color: 'var(--accent-blue)' }}>
                  📊 策略類型勝率（過去 90 天）
                </div>
                {!strategyPerf || !strategyPerf.strategies ? (
                  <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>無資料</div>
                ) : (
                  <div className="space-y-2">
                    {Object.entries(strategyPerf.strategies as Record<string, any>).map(([name, d]) => {
                      const decided = d.wins + d.losses;
                      const winPct = decided > 0 ? d.win_rate : 0;
                      const isStrongest = strategyPerf.strongest === name;
                      const isWeakest = strategyPerf.weakest === name;
                      return (
                        <div key={name} className="p-2 rounded text-xs" style={{ background: 'var(--bg-tertiary)' }}>
                          <div className="flex justify-between items-center mb-1">
                            <span className="font-medium">
                              {isStrongest && <span style={{ color: '#3fb950' }}>👑 </span>}
                              {isWeakest && <span style={{ color: '#f85149' }}>⚠️ </span>}
                              {name}
                            </span>
                            <span className="font-mono">
                              <span style={{ color: winPct >= 50 ? '#3fb950' : '#f85149' }}>{winPct.toFixed(1)}%</span>
                              <span style={{ color: 'var(--text-secondary)' }}> ({d.wins}W / {d.losses}L)</span>
                            </span>
                          </div>
                          <div style={{ height: 6, background: 'var(--bg-secondary)', borderRadius: 2, overflow: 'hidden' }}>
                            <div style={{
                              width: `${Math.min(100, winPct)}%`,
                              height: '100%',
                              background: winPct >= 50 ? '#3fb950' : '#f85149',
                            }} />
                          </div>
                          <div className="mt-1" style={{ color: 'var(--text-secondary)' }}>
                            涵蓋 regime：{(d.regimes_included || []).join(', ') || '—'}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
                {strategyPerf?.strongest && (
                  <div className="mt-2 text-xs p-2 rounded" style={{ background: 'rgba(63,185,80,0.1)', color: 'var(--text-primary)' }}>
                    💡 你最強策略：<b>{strategyPerf.strongest}</b>
                    {strategyPerf.weakest && <>　|　最弱（建議避免）：<b style={{ color: '#f85149' }}>{strategyPerf.weakest}</b></>}
                  </div>
                )}
              </div>

              {/* 5B：跨 symbol RS（過去 30 天 4h） */}
              <div>
                <div className="text-xs mb-2 font-medium" style={{ color: 'var(--accent-blue)' }}>
                  🔥 跨 symbol 相對強弱（vs {crossSymbolRS?.base ?? 'BTC/USDT'}，過去 {crossSymbolRS?.days ?? 30} 天）
                </div>
                {!crossSymbolRS || !crossSymbolRS.items || crossSymbolRS.items.length === 0 ? (
                  <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                    {crossSymbolRS?.error ?? '無資料'}
                  </div>
                ) : (
                  <div className="space-y-1">
                    {crossSymbolRS.items.map((it: any) => {
                      const maxAbs = Math.max(...crossSymbolRS.items.map((x: any) => Math.abs(x.rs_score)), 1);
                      const pct = (Math.abs(it.rs_score) / maxAbs) * 100;
                      const positive = it.rs_score >= 0;
                      return (
                        <div key={it.symbol} className="flex items-center gap-2 text-xs">
                          <span className="font-mono" style={{ width: 90 }}>{it.symbol}</span>
                          <div style={{ flex: 1, position: 'relative', height: 8, background: 'var(--bg-tertiary)', borderRadius: 2 }}>
                            <div style={{
                              position: 'absolute',
                              left: positive ? '50%' : `${50 - pct / 2}%`,
                              width: `${pct / 2}%`,
                              height: '100%',
                              background: positive ? '#3fb950' : '#f85149',
                              borderRadius: 2,
                            }} />
                            <div style={{
                              position: 'absolute', left: '50%', top: 0, bottom: 0,
                              width: 1, background: 'var(--text-secondary)', opacity: 0.3,
                            }} />
                          </div>
                          <span className="font-mono" style={{
                            width: 70, textAlign: 'right',
                            color: positive ? '#3fb950' : '#f85149',
                          }}>
                            {positive ? '+' : ''}{it.rs_score.toFixed(2)}%
                          </span>
                          <span className="font-mono" style={{ width: 60, textAlign: 'right', color: 'var(--text-secondary)' }}>
                            ({it.return_pct >= 0 ? '+' : ''}{it.return_pct.toFixed(1)}%)
                          </span>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      )}

      {/* ===== 指標勝率排名 ===== */}
      {viewTab === 'indicators' && (
        <div>
          {stats?.indicator_performance && Object.keys(stats.indicator_performance).length > 0 ? (
            <>
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
                    .map(([name, data]) => {
                      const lowSample = data.samples < 3;
                      return (
                        <tr key={name} style={{ borderTop: '1px solid var(--border-primary)', opacity: lowSample ? 0.5 : 1 }}>
                          <td className="py-1.5" style={{ color: 'var(--text-primary)' }}>
                            {name}{lowSample ? ' *' : ''}
                          </td>
                          <td className="text-right py-1.5" style={{ color: data.win_rate >= 50 ? '#4ade80' : '#f87171' }}>
                            {data.win_rate.toFixed(1)}%
                          </td>
                          <td className="text-right py-1.5" style={{ color: 'var(--text-secondary)' }}>
                            {data.samples}
                          </td>
                        </tr>
                      );
                    })}
                </tbody>
              </table>
              <div className="text-right mt-1 text-xs" style={{ color: 'var(--text-secondary)', opacity: 0.6 }}>
                * 樣本不足，僅供參考
              </div>
            </>
          ) : (
            <div className="text-center py-6 text-xs" style={{ color: 'var(--text-secondary)' }}>
              累積更多預測後即可顯示各指標的勝率排名
            </div>
          )}
        </div>
      )}

      {/* ===== 覆盤報告列表 ===== */}
      {viewTab === 'reviews' && (
        <div className="space-y-2">
          {reviewsLoading ? (
            <div className="text-center py-6 text-xs" style={{ color: 'var(--text-secondary)' }}>
              載入覆盤報告中...
            </div>
          ) : reviews.length === 0 ? (
            <div className="text-center py-6 text-xs" style={{ color: 'var(--text-secondary)' }}>
              尚無覆盤報告。點擊下方「生成覆盤報告」或等待每週自動覆盤。
            </div>
          ) : (
            reviews.map((rev: any) => (
              <div key={rev.id} className="rounded-lg p-3 text-xs" style={{ background: 'var(--bg-tertiary)' }}>
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium" style={{ color: 'var(--text-primary)' }}>
                    {rev.is_auto ? '自動週覆盤' : '手動覆盤'}
                    {rev.symbol ? ` — ${rev.symbol}` : ''}
                  </span>
                  <span style={{ color: 'var(--text-secondary)' }}>
                    {fmtTime(rev.created_at)}
                  </span>
                </div>
                <div
                  className="whitespace-pre-wrap leading-relaxed"
                  style={{ color: 'var(--text-primary)', maxHeight: 200, overflow: 'auto' }}
                >
                  {rev.report}
                </div>
              </div>
            ))
          )}
        </div>
      )}

      {/* ===== 自動調整規則 ===== */}
      {viewTab === 'adjustments' && (
        <div>
          <div className="flex justify-between items-center mb-2">
            <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
              系統根據預測績效自動生成的強制規則
            </span>
            <button
              onClick={handleRecalculate}
              disabled={adjLoading}
              className="px-2 py-0.5 text-xs rounded"
              style={{ background: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}
            >
              {adjLoading ? '計算中...' : '重新計算'}
            </button>
          </div>
          {adjustments.length > 0 ? (
            <div className="space-y-1">
              {adjustments.map((adj: any) => {
                const typeLabels: Record<string, string> = {
                  indicator_weight: '指標權重',
                  confidence_scale: '信心校準',
                  direction_bias: '方向限制',
                  risk_multiplier: '風險控制',
                };
                const isBoost = adj.value > 1;
                const isSuppress = adj.value < 1;
                const color = isBoost ? '#4ade80' : isSuppress ? '#f87171' : 'var(--text-secondary)';
                return (
                  <div
                    key={adj.id}
                    className="flex items-center justify-between py-1.5 px-2 rounded text-xs"
                    style={{ background: 'var(--bg-tertiary)', borderLeft: `3px solid ${color}` }}
                  >
                    <div className="flex-1">
                      <span style={{ color: 'var(--text-secondary)' }}>
                        [{typeLabels[adj.adjustment_type] || adj.adjustment_type}]
                      </span>{' '}
                      <span style={{ color: 'var(--text-primary)' }}>{adj.key}</span>
                      <span style={{ color, fontWeight: 600 }}> {adj.value}x</span>
                      {adj.user_override ? (
                        <span style={{ color: '#facc15', marginLeft: 4 }}>🔒</span>
                      ) : null}
                      <div className="mt-0.5" style={{ color: 'var(--text-secondary)', fontSize: '10px' }}>
                        {adj.reason}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="text-center py-6 text-xs" style={{ color: 'var(--text-secondary)' }}>
              {adjLoading ? '載入中...' : '尚無自動調整規則。累積足夠的已驗證預測後，系統將自動生成。'}
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
        <button
          onClick={async () => {
            if (!confirm('確定要清除所有預測紀錄嗎？此操作無法復原。')) return;
            try {
              await clearPredictions();
              loadData();
            } catch (e: any) {
              alert(`清除失敗: ${e?.response?.data?.detail || e?.message || '請確認後端已重啟'}`);
            }
          }}
          className="text-xs px-3 py-1 rounded"
          style={{ color: '#f87171', background: 'transparent' }}
        >
          清除全部紀錄
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
