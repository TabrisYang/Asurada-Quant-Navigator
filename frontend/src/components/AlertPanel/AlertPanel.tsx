/** 阿斯拉量化系統 — 預警面板 */

import { useState, useEffect, useCallback } from 'react';
import { fetchActiveAlerts, fetchAlertHistory, triggerManualScan, dismissAlert } from '../../services/api';
import type { Alert } from '../../services/api';
import { useChartStore } from '../../stores/chartStore';

const ALERT_TYPE_LABELS: Record<string, string> = {
  overbought_reversal: '超買反轉',
  oversold_bounce: '超賣反彈',
  volatility_expansion: '波動擴張',
};

const DIRECTION_LABELS: Record<string, { text: string; color: string }> = {
  long: { text: '做多', color: '#3fb950' },
  short: { text: '做空', color: '#f85149' },
  neutral: { text: '不確定', color: '#8b949e' },
};

const CONFIDENCE_STYLES: Record<string, { bg: string; text: string }> = {
  high: { bg: 'rgba(63,185,80,0.15)', text: '#3fb950' },
  medium: { bg: 'rgba(210,153,34,0.15)', text: '#d29922' },
  low: { bg: 'rgba(139,148,158,0.12)', text: '#8b949e' },
};

function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

function timeRemaining(dateStr: string): string {
  const diff = new Date(dateStr).getTime() - Date.now();
  if (diff <= 0) return '已過期';
  const hours = Math.floor(diff / 3600000);
  if (hours < 1) return `${Math.floor(diff / 60000)}m`;
  if (hours < 24) return `${hours}h`;
  return `${Math.floor(hours / 24)}d ${hours % 24}h`;
}

export default function AlertPanel() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [history, setHistory] = useState<Alert[]>([]);
  const [tab, setTab] = useState<'active' | 'history'>('active');
  const [scanning, setScanning] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [expandedAlertId, setExpandedAlertId] = useState<number | null>(null);
  const setSymbol = useChartStore((s) => s.setSymbol);

  const loadAlerts = useCallback(async () => {
    try {
      const data = await fetchActiveAlerts();
      setAlerts(data.alerts || []);
    } catch {
      /* ignore */
    }
  }, []);

  const loadHistory = useCallback(async () => {
    try {
      const data = await fetchAlertHistory(undefined, 30);
      setHistory(data.alerts || []);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    loadAlerts();
    const interval = setInterval(loadAlerts, 60000);
    return () => clearInterval(interval);
  }, [loadAlerts]);

  useEffect(() => {
    if (tab === 'history') loadHistory();
  }, [tab, loadHistory]);

  const [scanResult, setScanResult] = useState<string | null>(null);

  const handleScan = async () => {
    setScanning(true);
    setScanResult(null);
    try {
      const result = await triggerManualScan();
      setScanResult(result.count > 0 ? `發現 ${result.count} 個信號` : '未發現異常');
      loadAlerts();
    } catch {
      setScanResult('掃描失敗');
    }
    setScanning(false);
    setTimeout(() => setScanResult(null), 5000);
  };

  const handleDismiss = async (id: number) => {
    await dismissAlert(id);
    setAlerts((prev) => prev.filter((a) => a.id !== id));
  };

  const handleJump = (symbol: string) => {
    const normalized = symbol.replace('/', '');
    setSymbol(normalized);
  };

  const displayAlerts = tab === 'active' ? alerts : history;

  return (
    <div
      style={{
        background: 'var(--bg-secondary)',
        borderBottom: '1px solid var(--border-color)',
        fontSize: '12px',
      }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-3 py-1.5"
        style={{ borderBottom: expanded ? '1px solid var(--border-color)' : 'none' }}
      >
        <div className="flex items-center gap-2">
          <button
            onClick={() => setExpanded(!expanded)}
            className="cursor-pointer"
            style={{ color: 'var(--text-primary)', background: 'none', border: 'none', fontSize: '12px' }}
          >
            {expanded ? '▼' : '▶'}
          </button>
          <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
            預警系統
          </span>
          {alerts.length > 0 && (
            <span
              style={{
                background: '#f85149',
                color: '#fff',
                borderRadius: '8px',
                padding: '0 6px',
                fontSize: '10px',
                fontWeight: 700,
              }}
            >
              {alerts.length}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {expanded && (
            <>
              <button
                onClick={() => setTab('active')}
                className="cursor-pointer"
                style={{
                  background: tab === 'active' ? 'var(--accent-blue)' : 'transparent',
                  color: tab === 'active' ? '#fff' : 'var(--text-secondary)',
                  border: 'none',
                  borderRadius: '4px',
                  padding: '1px 8px',
                  fontSize: '11px',
                }}
              >
                進行中
              </button>
              <button
                onClick={() => setTab('history')}
                className="cursor-pointer"
                style={{
                  background: tab === 'history' ? 'var(--accent-blue)' : 'transparent',
                  color: tab === 'history' ? '#fff' : 'var(--text-secondary)',
                  border: 'none',
                  borderRadius: '4px',
                  padding: '1px 8px',
                  fontSize: '11px',
                }}
              >
                歷史
              </button>
            </>
          )}
          <button
            onClick={handleScan}
            disabled={scanning}
            className="cursor-pointer"
            style={{
              background: scanning ? 'var(--bg-tertiary)' : 'rgba(88,166,255,0.12)',
              color: scanning ? 'var(--text-secondary)' : '#58a6ff',
              border: 'none',
              borderRadius: '4px',
              padding: '1px 8px',
              fontSize: '11px',
            }}
          >
            {scanning ? '掃描中...' : '掃描'}
          </button>
          {scanResult && (
            <span style={{
              color: scanResult.includes('失敗') ? '#f85149' : '#3fb950',
              fontSize: '10px',
            }}>
              {scanResult}
            </span>
          )}
        </div>
      </div>

      {/* Alert List */}
      {expanded && (
        <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
          {displayAlerts.length === 0 ? (
            <div
              className="text-center py-4"
              style={{ color: 'var(--text-secondary)', fontSize: '11px' }}
            >
              {tab === 'active' ? '目前沒有預警信號' : '沒有歷史預警'}
            </div>
          ) : (
            displayAlerts.map((alert) => {
              const dir = DIRECTION_LABELS[alert.direction] || DIRECTION_LABELS.neutral;
              const conf = CONFIDENCE_STYLES[alert.confidence] || CONFIDENCE_STYLES.low;
              const typeName = ALERT_TYPE_LABELS[alert.alert_type] || alert.alert_type;

              const isDetailOpen = expandedAlertId === alert.id;
              const hasProb = alert.move_probability != null && alert.move_probability > 0;

              return (
                <div key={alert.id} style={{ borderBottom: '1px solid var(--border-color)' }}>
                  <div
                    className="flex items-center justify-between px-3 py-1.5"
                    style={{
                      opacity: alert.status === 'dismissed' ? 0.4 : 1,
                      cursor: alert.evidence_summary ? 'pointer' : 'default',
                    }}
                    onClick={() => {
                      if (alert.evidence_summary) {
                        setExpandedAlertId(isDetailOpen ? null : alert.id);
                      }
                    }}
                  >
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      {/* Direction dot */}
                      <span
                        className="inline-block w-2 h-2 rounded-full flex-shrink-0"
                        style={{ background: dir.color }}
                      />
                      {/* Symbol - clickable */}
                      <button
                        onClick={(e) => { e.stopPropagation(); handleJump(alert.symbol); }}
                        className="cursor-pointer hover:underline flex-shrink-0"
                        style={{
                          color: 'var(--text-primary)',
                          fontWeight: 600,
                          background: 'none',
                          border: 'none',
                          padding: 0,
                          fontSize: '12px',
                        }}
                      >
                        {alert.symbol}
                      </button>
                      {/* Type */}
                      <span style={{ color: dir.color, fontSize: '11px' }}>
                        {typeName}
                      </span>
                      {/* Direction */}
                      <span style={{ color: dir.color, fontSize: '11px', fontWeight: 600 }}>
                        {dir.text}
                      </span>
                      {/* Confidence */}
                      <span
                        style={{
                          background: conf.bg,
                          color: conf.text,
                          borderRadius: '3px',
                          padding: '0 4px',
                          fontSize: '10px',
                        }}
                      >
                        {alert.confidence}
                      </span>
                      {/* Movement Probability */}
                      {hasProb && (
                        <span
                          style={{
                            background: alert.move_probability! > 60
                              ? 'rgba(246,106,10,0.15)'
                              : 'rgba(139,148,158,0.12)',
                            color: alert.move_probability! > 60 ? '#f66a0a' : '#8b949e',
                            borderRadius: '3px',
                            padding: '0 4px',
                            fontSize: '10px',
                            fontWeight: 600,
                          }}
                          title="未來6根K線內出現>=3%波動的機率"
                        >
                          {alert.move_probability!.toFixed(0)}%波動
                        </span>
                      )}
                      {/* Outcome (history only) */}
                      {alert.outcome_pct != null && (
                        <span
                          style={{
                            color: alert.outcome_pct > 0 ? '#3fb950' : '#f85149',
                            fontSize: '11px',
                          }}
                        >
                          {alert.outcome_pct > 0 ? '+' : ''}{alert.outcome_pct.toFixed(1)}%
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      {alert.evidence_summary && (
                        <span style={{ color: 'var(--text-secondary)', fontSize: '10px' }}>
                          {isDetailOpen ? '▲' : '▼'}
                        </span>
                      )}
                      <span style={{ color: 'var(--text-secondary)', fontSize: '10px' }}>
                        {tab === 'active'
                          ? `剩 ${timeRemaining(alert.expires_at)}`
                          : timeAgo(alert.created_at)
                        }
                      </span>
                      {tab === 'active' && alert.status === 'active' && (
                        <button
                          onClick={(e) => { e.stopPropagation(); handleDismiss(alert.id); }}
                          className="cursor-pointer"
                          style={{
                            color: 'var(--text-secondary)',
                            background: 'none',
                            border: 'none',
                            fontSize: '11px',
                            padding: '0 2px',
                          }}
                          title="忽略"
                        >
                          ×
                        </button>
                      )}
                    </div>
                  </div>
                  {/* Expanded evidence detail */}
                  {isDetailOpen && (alert.evidence_summary || alert.probability_detail) && (
                    <div
                      className="px-3 py-2"
                      style={{
                        background: 'rgba(88,166,255,0.04)',
                        borderTop: '1px solid var(--border-color)',
                        fontSize: '11px',
                        color: 'var(--text-secondary)',
                        lineHeight: '1.6',
                      }}
                    >
                      {/* 機率表格 */}
                      {alert.probability_detail && (
                        <div className="flex gap-3 mb-1.5" style={{ flexWrap: 'wrap' }}>
                          {Object.entries(alert.probability_detail).map(([horizon, data]) => (
                            <div key={horizon} style={{
                              background: 'var(--bg-tertiary)',
                              borderRadius: '4px',
                              padding: '2px 6px',
                              fontSize: '10px',
                            }}>
                              <span style={{ fontWeight: 600 }}>{horizon.replace('_bars', '')}根</span>
                              {' '}
                              <span style={{ color: '#3fb950' }}>{data.up_pct}%</span>
                              {' / '}
                              <span style={{ color: '#f85149' }}>{data.down_pct}%</span>
                              {data.any_ci && (
                                <span style={{ color: 'var(--text-secondary)', fontSize: '9px' }}>
                                  {' '}CI:{data.any_ci[0]}~{data.any_ci[1]}%
                                </span>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                      {/* 特徵歸因 */}
                      {alert.feature_attribution && alert.feature_attribution.length > 0 && (
                        <div className="flex gap-1.5 mb-1.5" style={{ flexWrap: 'wrap' }}>
                          {alert.feature_attribution.slice(0, 4).map((fa) => (
                            <span key={fa.feature} style={{
                              background: fa.lift > 1.1 ? 'rgba(63,185,80,0.1)' : 'var(--bg-tertiary)',
                              borderRadius: '3px',
                              padding: '1px 5px',
                              fontSize: '10px',
                              color: fa.lift > 1.1 ? '#3fb950' : 'var(--text-secondary)',
                            }}>
                              {fa.feature.split('=')[0]} {fa.lift > 1 ? `+${((fa.lift - 1) * 100).toFixed(0)}%` : ''}
                            </span>
                          ))}
                        </div>
                      )}
                      {/* 文字摘要 */}
                      {alert.evidence_summary}
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>
      )}
    </div>
  );
}
