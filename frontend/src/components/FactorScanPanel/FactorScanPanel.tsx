/** 阿斯拉量化系統 — 因子掃描面板 */

import { useState, useCallback } from 'react';
import { useChartStore } from '../../stores/chartStore';
import { runFactorScan } from '../../services/api';
import type { FactorScanResult, FactorScanItem } from '../../services/api';

function useGenerateBacktest() {
  const setPendingChatMessage = useChartStore((s) => s.setPendingChatMessage);
  const symbol = useChartStore((s) => s.symbol);
  const timeframe = useChartStore((s) => s.timeframe);

  return useCallback((result: FactorScanResult) => {
    const parts: string[] = [];
    parts.push(`根據因子掃描結果，請用以下有效因子對 ${symbol} ${timeframe} 做回測：`);

    const allEntries: string[] = [];
    result.quantile_analysis && Object.values(result.quantile_analysis).forEach((qa) => {
      if (qa.best_quantile?.entry_suggestion) {
        allEntries.push(`${qa.factor} ${qa.best_quantile.entry_suggestion}`);
      }
    });

    if (allEntries.length > 0) {
      parts.push(`建議進場條件：${allEntries.join('、')}`);
    }

    if ((result.positive_top?.length ?? 0) > 0) {
      const top3 = result.positive_top!.slice(0, 3).map((f) => f.factor);
      parts.push(`正相關有效因子：${top3.join('、')}`);
    }
    if ((result.negative_top?.length ?? 0) > 0) {
      const top3 = result.negative_top!.slice(0, 3).map((f) => f.factor);
      parts.push(`負相關有效因子（反向使用）：${top3.join('、')}`);
    }

    parts.push('請設計一個結合以上因子的策略並執行回測分析。');
    setPendingChatMessage(parts.join('\n'));
  }, [setPendingChatMessage, symbol, timeframe]);
}

// ─── IC 強度判讀 ────────────────────────────
function IcStrengthBar({ ic }: { ic: number }) {
  const absIc = Math.abs(ic);
  let label: string;
  let color: string;
  let pct: number;

  if (absIc > 0.15) { label = '非常強'; color = '#3fb950'; pct = 100; }
  else if (absIc > 0.10) { label = '強'; color = '#58a6ff'; pct = 80; }
  else if (absIc > 0.05) { label = '中等'; color = '#d29922'; pct = 55; }
  else if (absIc > 0.02) { label = '弱'; color = '#f0883e'; pct = 30; }
  else { label = '無效'; color = '#484f58'; pct = 10; }

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 4, minWidth: 80 }}>
      <div style={{
        width: 50, height: 6, background: 'var(--bg-tertiary)',
        borderRadius: 3, overflow: 'hidden',
      }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 3 }} />
      </div>
      <span style={{ fontSize: 10, color, whiteSpace: 'nowrap' }}>{label}</span>
    </div>
  );
}

// ─── 迷你 Sparkline ─────────────────────────
function Sparkline({ data }: { data: (number | null)[] }) {
  const vals = data.filter((v): v is number => v !== null);
  if (vals.length < 2) return <span style={{ color: 'var(--text-muted)' }}>—</span>;
  const mn = Math.min(...vals);
  const mx = Math.max(...vals);
  const range = mx - mn || 1;
  const bars = vals.map((v) => Math.round(((v - mn) / range) * 7));
  const blocks = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
  return (
    <span style={{ fontFamily: 'monospace', letterSpacing: '-1px', fontSize: 12 }}>
      {bars.map((b, i) => (
        <span key={i} style={{ color: i === bars.length - 1 ? '#58a6ff' : 'var(--text-muted)' }}>
          {blocks[b]}
        </span>
      ))}
    </span>
  );
}

// ─── 因子表格行 ─────────────────────────────
function FactorRow({ item, rank }: { item: FactorScanItem; rank: number }) {
  const trendIcon = item.decay_trend === 'rising' ? '↑' :
    item.decay_trend === 'decaying' ? '↓' : '→';
  const trendColor = item.decay_trend === 'rising' ? '#3fb950' :
    item.decay_trend === 'decaying' ? '#f85149' : 'var(--text-muted)';

  return (
    <tr style={{ borderBottom: '1px solid var(--border-color)' }}>
      <td style={{ padding: '6px 8px', textAlign: 'center', color: 'var(--text-muted)' }}>{rank}</td>
      <td style={{ padding: '6px 8px', fontWeight: 600, color: 'var(--text-primary)', maxWidth: 130, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {item.factor}
      </td>
      <td style={{ padding: '6px 8px', textAlign: 'right', fontFamily: 'monospace', color: item.ic_recent > 0 ? '#3fb950' : '#f85149' }}>
        {item.ic_recent > 0 ? '+' : ''}{item.ic_recent.toFixed(4)}
      </td>
      <td style={{ padding: '6px 4px' }}><IcStrengthBar ic={item.ic_recent} /></td>
      <td style={{ padding: '6px 8px', textAlign: 'right', fontFamily: 'monospace', color: 'var(--text-muted)', fontSize: 11 }}>
        {item.ic_full > 0 ? '+' : ''}{item.ic_full.toFixed(4)}
      </td>
      <td style={{ padding: '6px 8px', textAlign: 'center' }}><Sparkline data={item.decay_curve} /></td>
      <td style={{ padding: '6px 8px', textAlign: 'center', color: trendColor, fontWeight: 600 }}>{trendIcon}</td>
      <td style={{ padding: '6px 8px', textAlign: 'center', color: 'var(--text-muted)', fontSize: 11 }}>
        {item.half_life != null ? (item.half_life >= 6 ? '>6' : item.half_life) : '—'}
      </td>
      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11 }}>{item.status_label}</td>
    </tr>
  );
}

// ─── 組合表格 ───────────────────────────────
type ComboItem = { factor_a: string; factor_b: string; combo_ic: number; combo_abs_ic: number };

function ComboTable({ items, label }: { items: ComboItem[]; label: string }) {
  if (!items || items.length === 0) return null;
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>{label}</div>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <tbody>
          {items.map((c, i) => (
            <tr key={i} style={{ borderBottom: '1px solid var(--border-color)' }}>
              <td style={{ padding: '4px 8px', color: 'var(--text-primary)' }}>{c.factor_a}</td>
              <td style={{ padding: '4px 4px', color: 'var(--text-muted)', textAlign: 'center' }}>+</td>
              <td style={{ padding: '4px 8px', color: 'var(--text-primary)' }}>{c.factor_b}</td>
              <td style={{ padding: '4px 8px', textAlign: 'right', fontFamily: 'monospace', color: c.combo_ic > 0 ? '#3fb950' : '#f85149' }}>
                {c.combo_ic > 0 ? '+' : ''}{c.combo_ic.toFixed(4)}
              </td>
              <td style={{ padding: '4px 4px' }}><IcStrengthBar ic={c.combo_ic} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─── 主面板 ─────────────────────────────────
export default function FactorScanPanel() {
  const symbol = useChartStore((s) => s.symbol);
  const timeframe = useChartStore((s) => s.timeframe);

  const [result, setResult] = useState<FactorScanResult | null>(null);
  const [scanning, setScanning] = useState(false);
  const [showPanel, setShowPanel] = useState(false);

  const handleScan = useCallback(async () => {
    setScanning(true);
    setShowPanel(true);
    try {
      const data = await runFactorScan({ symbol, timeframe, forward_period: 5, top_n: 5 });
      setResult(data);
    } catch {
      setResult({ status: 'error', message: '掃描失敗' });
    } finally {
      setScanning(false);
    }
  }, [symbol, timeframe]);

  const thStyle: React.CSSProperties = {
    padding: '4px 8px', textAlign: 'center', fontSize: 11,
    color: 'var(--text-muted)', borderBottom: '2px solid var(--border-color)',
    whiteSpace: 'nowrap',
  };

  return (
    <>
      <button
        onClick={handleScan}
        disabled={scanning}
        title="掃描所有因子的近期預測力、Alpha Decay、組合效果和最佳進場區間"
        style={{
          display: 'flex', alignItems: 'center', gap: 6,
          padding: '6px 14px', borderRadius: 6,
          border: '1px solid var(--border-color)',
          background: scanning ? 'var(--bg-secondary)' : 'linear-gradient(135deg, #1a3a5c, #0d2137)',
          color: '#58a6ff', cursor: scanning ? 'wait' : 'pointer',
          fontSize: 13, fontWeight: 600, transition: 'all 0.2s',
        }}
      >
        {scanning ? <span style={{ display: 'inline-block', animation: 'spin 1s linear infinite' }}>⟳</span> : <span>📊</span>}
        {scanning ? '掃描中...' : '因子掃描'}
      </button>

      {showPanel && (
        <div style={{
          position: 'fixed', top: 60, right: 20, bottom: 60,
          width: 760, maxWidth: 'calc(100vw - 40px)',
          background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
          borderRadius: 12, boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
          zIndex: 1000, display: 'flex', flexDirection: 'column', overflow: 'hidden',
        }}>
          {/* Header */}
          <div style={{
            padding: '12px 16px', borderBottom: '1px solid var(--border-color)',
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            background: 'var(--bg-secondary)',
          }}>
            <div>
              <div style={{ fontWeight: 700, fontSize: 15, color: 'var(--text-primary)' }}>
                📊 因子掃描 — {symbol} {timeframe}
              </div>
              {result?.status === 'success' && (
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
                  市場體制：{result.regime?.label ?? '未知'}
                  {result.regime?.adx != null && ` (ADX=${result.regime.adx})`}
                  {' | '}近期 {result.recent_bars} 根 vs 全域 {result.total_bars} 根
                  {' | '}有效因子 {result.effective_count}/{result.total_factors_scanned}
                </div>
              )}
            </div>
            <button onClick={() => setShowPanel(false)} style={{
              background: 'none', border: 'none', color: 'var(--text-muted)',
              cursor: 'pointer', fontSize: 18, padding: '4px 8px',
            }}>✕</button>
          </div>

          {/* IC 強度圖例 */}
          <div style={{
            padding: '6px 16px', borderBottom: '1px solid var(--border-color)',
            display: 'flex', gap: 16, fontSize: 10, color: 'var(--text-muted)',
            background: 'var(--bg-secondary)',
          }}>
            <span>IC 強度：</span>
            {[
              { label: '非常強 >0.15', color: '#3fb950' },
              { label: '強 0.10~0.15', color: '#58a6ff' },
              { label: '中等 0.05~0.10', color: '#d29922' },
              { label: '弱 0.02~0.05', color: '#f0883e' },
              { label: '無效 <0.02', color: '#484f58' },
            ].map((s) => (
              <span key={s.label} style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                <span style={{ width: 8, height: 8, borderRadius: 2, background: s.color, display: 'inline-block' }} />
                {s.label}
              </span>
            ))}
          </div>

          {/* Body */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '12px 16px' }}>
            {scanning && (
              <div style={{ textAlign: 'center', padding: 40, color: 'var(--text-muted)' }}>
                <div style={{ fontSize: 24, marginBottom: 12, animation: 'spin 1s linear infinite', display: 'inline-block' }}>⟳</div>
                <div>正在掃描 ~36 個因子（含衍生因子）...</div>
              </div>
            )}

            {result?.status === 'error' && (
              <div style={{ textAlign: 'center', padding: 40, color: '#f85149' }}>{result.message}</div>
            )}

            {result?.status === 'success' && !scanning && (
              <>
                {/* 正相關 TOP */}
                {(result.positive_top?.length ?? 0) > 0 && (
                  <Section title="正相關 TOP — 因子值↑ → 未來價格↑" color="#3fb950">
                    <FactorTable items={result.positive_top!} headerStyle={thStyle} />
                  </Section>
                )}

                {/* 負相關 TOP */}
                {(result.negative_top?.length ?? 0) > 0 && (
                  <Section title="負相關 TOP — 因子值↑ → 未來價格↓" color="#f85149">
                    <FactorTable items={result.negative_top!} headerStyle={thStyle} />
                  </Section>
                )}

                {/* 分位數分析 + 最佳進場建議 */}
                {result.quantile_analysis && Object.keys(result.quantile_analysis).length > 0 && (
                  <Section title="最佳進場區間分析" color="#79c0ff">
                    {Object.entries(result.quantile_analysis).map(([label, qa]) => (
                      <div key={label} style={{ marginBottom: 12, padding: '8px 10px', background: 'var(--bg-secondary)', borderRadius: 8 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                          <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>
                            {qa.factor}
                          </span>
                          <span style={{ fontSize: 11 }}>
                            單調性: {qa.is_monotonic ? <span style={{ color: '#3fb950' }}>✅ 良好</span> : <span style={{ color: '#d29922' }}>⚠️ 非單調</span>}
                          </span>
                        </div>

                        {/* 分位數條 */}
                        <div style={{ display: 'flex', gap: 3, marginBottom: 6 }}>
                          {qa.quantile_returns_pct.map((r, i) => {
                            const range = qa.quantile_ranges?.[i];
                            const isBest = qa.best_quantile?.index === i;
                            return (
                              <div key={i} style={{
                                flex: 1, textAlign: 'center', padding: '6px 2px', borderRadius: 4,
                                fontSize: 10, fontFamily: 'monospace',
                                background: isBest ? 'rgba(88,166,255,0.2)' : r != null && r > 0 ? 'rgba(63,185,80,0.1)' : 'rgba(248,81,73,0.1)',
                                border: isBest ? '1px solid #58a6ff' : '1px solid transparent',
                              }}>
                                <div style={{ color: 'var(--text-muted)', marginBottom: 2 }}>{range?.label ?? `Q${i + 1}`}</div>
                                {range && <div style={{ color: 'var(--text-muted)', fontSize: 9, marginBottom: 2 }}>{range.low.toFixed(1)}~{range.high.toFixed(1)}</div>}
                                <div style={{ color: r != null && r > 0 ? '#3fb950' : '#f85149', fontWeight: isBest ? 700 : 400 }}>
                                  {r != null ? `${r > 0 ? '+' : ''}${r.toFixed(2)}%` : '—'}
                                </div>
                                {isBest && <div style={{ color: '#58a6ff', fontSize: 9, marginTop: 2 }}>★最佳</div>}
                              </div>
                            );
                          })}
                        </div>

                        {/* 最佳進場建議 */}
                        {qa.best_quantile?.entry_suggestion && (
                          <div style={{
                            fontSize: 12, padding: '6px 10px', borderRadius: 6,
                            background: 'rgba(88,166,255,0.1)', border: '1px solid rgba(88,166,255,0.3)',
                            color: '#58a6ff',
                          }}>
                            💡 建議進場條件：<strong>{qa.factor} {qa.best_quantile.entry_suggestion}</strong>
                            {' '}（歷史平均報酬 {qa.best_quantile.return_pct > 0 ? '+' : ''}{qa.best_quantile.return_pct.toFixed(2)}%）
                          </div>
                        )}
                      </div>
                    ))}
                  </Section>
                )}

                {/* 雙因子組合 */}
                {result.combo_top && (
                  <Section title="雙因子組合" color="#d2a8ff">
                    <ComboTable items={result.combo_top.positive_combos ?? []} label="正正組合（同方向看多增強）" />
                    <ComboTable items={result.combo_top.negative_combos ?? []} label="負負組合（同方向看空增強）" />
                    <ComboTable items={result.combo_top.hedge_combos ?? []} label="多空對沖組合（正因子 - 負因子）" />
                    {!(result.combo_top.positive_combos?.length || result.combo_top.negative_combos?.length || result.combo_top.hedge_combos?.length) && (
                      <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>無足夠因子進行組合分析</div>
                    )}
                  </Section>
                )}

                {/* 高相關警告 */}
                {(result.high_correlation_warnings?.length ?? 0) > 0 && (
                  <Section title="高相關警告" color="#d29922">
                    {result.high_correlation_warnings!.map((w, i) => (
                      <div key={i} style={{ fontSize: 12, color: '#d29922', padding: '2px 0' }}>
                        ⚠️ {w.factor_a} ↔ {w.factor_b} (r={w.correlation.toFixed(2)}) — 建議只保留預測力較強的
                      </div>
                    ))}
                  </Section>
                )}

                {/* 一鍵回測 */}
                <BacktestButton result={result} />

                {/* 底部說明 */}
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 12, padding: '8px 0', borderTop: '1px solid var(--border-color)' }}>
                  ⚠️ IC 為統計相關性，不保證因果關係。建議搭配回測驗證。
                  {' | '}Forward = {result.forward_period} 根 K 線
                </div>
              </>
            )}
          </div>
        </div>
      )}

      <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
    </>
  );
}

function BacktestButton({ result }: { result: FactorScanResult }) {
  const generateBacktest = useGenerateBacktest();
  const hasFactors = (result.positive_top?.length ?? 0) > 0 || (result.negative_top?.length ?? 0) > 0;
  if (!hasFactors) return null;
  return (
    <button
      onClick={() => generateBacktest(result)}
      style={{
        width: '100%', padding: '10px 16px', borderRadius: 8,
        border: '1px solid rgba(88,166,255,0.3)',
        background: 'linear-gradient(135deg, rgba(88,166,255,0.15), rgba(63,185,80,0.1))',
        color: '#58a6ff', cursor: 'pointer', fontSize: 13, fontWeight: 700,
        transition: 'all 0.2s', marginTop: 8,
      }}
      onMouseEnter={(e) => (e.currentTarget.style.background = 'linear-gradient(135deg, rgba(88,166,255,0.25), rgba(63,185,80,0.2))')}
      onMouseLeave={(e) => (e.currentTarget.style.background = 'linear-gradient(135deg, rgba(88,166,255,0.15), rgba(63,185,80,0.1))')}
    >
      🚀 一鍵回測 — 用掃描結果生成策略
    </button>
  );
}

function Section({ title, color, children }: { title: string; color: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontSize: 13, fontWeight: 700, color, marginBottom: 8, paddingBottom: 4, borderBottom: `1px solid ${color}33` }}>
        {title}
      </div>
      {children}
    </div>
  );
}

function FactorTable({ items, headerStyle }: { items: FactorScanItem[]; headerStyle: React.CSSProperties }) {
  return (
    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
      <thead>
        <tr>
          <th style={headerStyle}>#</th>
          <th style={{ ...headerStyle, textAlign: 'left' }}>因子</th>
          <th style={headerStyle}>近期IC</th>
          <th style={headerStyle}>強度</th>
          <th style={headerStyle}>長期IC</th>
          <th style={headerStyle}>Decay</th>
          <th style={headerStyle}>趨勢</th>
          <th style={headerStyle}>半衰期</th>
          <th style={headerStyle}>狀態</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item, i) => <FactorRow key={item.factor} item={item} rank={i + 1} />)}
      </tbody>
    </table>
  );
}
