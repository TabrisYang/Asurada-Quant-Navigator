/** 阿斯拉量化系統 — 因子掃描面板 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { useChartStore } from '../../stores/chartStore';
import { runFactorScan, triggerPreview } from '../../services/api';
import { toast } from '../Toast';
import type { FactorScanResult, FactorScanItem, StrategyTier, StrategyTierCondition, TriggerCondition } from '../../services/api';

type TierKey = 'strict' | 'moderate' | 'loose';

// ─── 從 tier conditions 轉換成調整器可編輯格式 ───
function tierToEditableConds(conditions: StrategyTierCondition[]): EditableCond[] {
  return conditions.map((c) => {
    let op = '>';
    let val = 0;
    let val2: number | undefined;
    const desc = c.description;

    if (desc.includes('~')) {
      op = 'between';
      const parts = desc.replace(/[()放寬]/g, '').split('~').map((s) => parseFloat(s.trim()));
      val = parts[0] || 0;
      val2 = parts[1] || 0;
    } else if (desc.startsWith('<')) {
      op = '<';
      val = parseFloat(desc.replace(/[<()放寬]/g, '').trim()) || 0;
    } else if (desc.startsWith('>')) {
      op = '>';
      val = parseFloat(desc.replace(/[>()放寬]/g, '').trim()) || 0;
    } else {
      const num = parseFloat(desc.replace(/[^0-9.\-]/g, ''));
      if (!isNaN(num)) { op = '>'; val = num; }
    }

    return {
      factor: c.factor, operator: op, value: val, value2: val2,
      enabled: true, abs_ic: c.abs_ic, return_pct: c.return_pct,
      original_description: c.description,
    };
  });
}

// ─── 量化概念解說（零 token，純模板） ──────────────
const QUANT_EXPLAIN: Record<string, { title: string; body: string; example?: string }> = {
  ic: {
    title: 'IC（Information Coefficient）',
    body: '衡量一個因子跟未來報酬的相關性。範圍 -1 ~ +1，絕對值越大代表預測力越強。',
    example: 'IC = +0.15 代表這個因子跟未來漲跌有很強的正相關，數值越高 → 未來越容易漲。',
  },
  alpha_decay: {
    title: 'Alpha Decay（預測力衰減）',
    body: '因子的預測力會隨時間衰減。衰減越慢代表因子越穩定可靠，衰減快的因子需要更頻繁更新。',
    example: '半衰期 = 4 表示這個因子的預測力大約 4 個週期後會降到一半。',
  },
  half_life: {
    title: '半衰期（Half-Life）',
    body: '因子的預測力衰減到一半所需的時間週期。數值越大 = 因子越持久穩定。',
    example: '半衰期 ≥ 4：穩定；2~3：中等；≤ 1：快速衰退，可能是雜訊。',
  },
  quantile: {
    title: '分位數分析（Quantile Analysis）',
    body: '把因子值從小到大分成 5 組（Q1~Q5），看每組的平均未來報酬。找出「哪個範圍進場最賺」。',
    example: 'Q1 報酬 +0.36% 且是最佳 → 當因子值在 Q1 範圍時進場，歷史上平均賺 0.36%。',
  },
  combo_ic: {
    title: '雙因子組合 IC',
    body: '兩個因子合併計算的預測力。組合 IC 高於單因子 IC 代表這兩個因子配合使用更有效。',
    example: '因子A IC=0.10 + 因子B IC=0.08 → 組合 IC=0.15，代表配合使用效果比單獨用好。',
  },
  monotonic: {
    title: '單調性（Monotonicity）',
    body: '因子值從 Q1→Q5 的報酬是否遞增或遞減。單調 = 因子值越大（或越小）報酬越好，線性關係清楚。',
    example: '非單調 ⚠️ 代表中間某些區間可能有反轉，使用時需搭配其他因子確認。',
  },
  strategy_tier: {
    title: '策略分級（Strict / Moderate / Loose）',
    body: '嚴格版用所有高品質因子 AND，精準但機會少；建議版用最佳雙因子組合 + 強因子，平衡精準和機會；寬鬆版只用最強 2 個因子，機會多但雜訊也多。',
    example: '建議先從「建議版」開始，觀察觸發次數和歷史報酬，再決定要更嚴格還是更寬鬆。',
  },
  trigger_count: {
    title: '觸發次數',
    body: '歷史上滿足所有進場條件的 K 線數量。太少（< 10）代表條件太嚴，回測統計意義不足；太多（> 50%）代表條件太鬆，可能沒有選擇性。',
    example: '觸發 50 次 / 7000 根 = 0.7%，代表條件非常嚴格，只在極少數時刻觸發。',
  },
};

function WhyButton({ concept }: { concept: keyof typeof QUANT_EXPLAIN }) {
  const [open, setOpen] = useState(false);
  const info = QUANT_EXPLAIN[concept];
  if (!info) return null;
  return (
    <span style={{ position: 'relative', display: 'inline-block' }}>
      <span
        onClick={(e) => { e.stopPropagation(); setOpen(!open); }}
        style={{
          cursor: 'pointer', fontSize: 10, opacity: 0.5,
          color: 'var(--accent-blue)', marginLeft: 3,
          userSelect: 'none',
        }}
        title="點擊瞭解這個概念"
      >❓</span>
      {open && (
        <div
          onClick={(e) => e.stopPropagation()}
          style={{
            position: 'absolute', left: 16, top: -8, zIndex: 1000,
            width: 280, padding: '12px 14px', borderRadius: 10,
            background: 'var(--bg-secondary, #161b22)',
            border: '1px solid var(--accent-blue)',
            boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
            fontSize: 11, lineHeight: 1.6,
            color: 'var(--text-primary, #e6edf3)',
          }}
        >
          <div style={{ fontWeight: 700, marginBottom: 6, color: 'var(--accent-blue)' }}>
            {info.title}
          </div>
          <p style={{ margin: 0, marginBottom: 8, color: 'var(--text-secondary)' }}>{info.body}</p>
          {info.example && (
            <div style={{
              padding: '6px 8px', borderRadius: 6,
              background: 'rgba(88,166,255,0.08)',
              border: '1px solid rgba(88,166,255,0.2)',
              fontSize: 10, color: 'var(--text-secondary)',
            }}>
              💡 {info.example}
            </div>
          )}
          <button
            onClick={() => setOpen(false)}
            style={{
              marginTop: 8, padding: '3px 10px', borderRadius: 4,
              background: 'rgba(88,166,255,0.15)', border: '1px solid rgba(88,166,255,0.3)',
              color: 'var(--accent-blue)', fontSize: 10, cursor: 'pointer',
            }}
          >關閉</button>
        </div>
      )}
    </span>
  );
}

// ─── IC 條形圖視覺化 ────────────────────────
function IcBarChart({ items, title }: { items: FactorScanItem[]; title: string }) {
  if (!items || items.length === 0) return null;
  const maxAbs = Math.max(...items.map(i => Math.abs(i.ic_recent)), 0.01);

  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 6 }}>
        {title}
      </div>
      {items.map((item, idx) => {
        const pct = Math.abs(item.ic_recent) / maxAbs * 100;
        const isPos = item.ic_recent > 0;
        const barColor = isPos ? '#3fb950' : '#f85149';
        return (
          <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 3 }}>
            <span style={{
              width: 90, fontSize: 10, color: 'var(--text-secondary)',
              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              textAlign: 'right', flexShrink: 0,
            }}>
              {item.factor}
            </span>
            <div style={{ flex: 1, height: 12, background: 'var(--bg-tertiary)', borderRadius: 3, overflow: 'hidden', position: 'relative' }}>
              <div style={{
                position: 'absolute',
                left: isPos ? '50%' : `${50 - pct / 2}%`,
                width: `${pct / 2}%`,
                height: '100%',
                background: barColor,
                borderRadius: 3,
                opacity: 0.7,
              }} />
              <div style={{
                position: 'absolute', left: '50%', top: 0, bottom: 0,
                width: 1, background: 'var(--text-muted)', opacity: 0.3,
              }} />
            </div>
            <span style={{ width: 52, fontSize: 10, fontFamily: 'monospace', color: barColor, textAlign: 'right', flexShrink: 0 }}>
              {isPos ? '+' : ''}{item.ic_recent.toFixed(4)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ─── 雙因子組合熱力圖 ─────────────────────────
type ComboItemViz = { factor_a: string; factor_b: string; combo_ic: number; combo_abs_ic: number };

function ComboHeatmap({ items }: { items: ComboItemViz[] }) {
  if (!items || items.length === 0) return null;
  const maxAbs = Math.max(...items.map(i => i.combo_abs_ic), 0.01);

  const getColor = (ic: number) => {
    const abs = Math.abs(ic);
    const intensity = Math.min(abs / maxAbs, 1);
    if (ic > 0) return `rgba(63, 185, 80, ${0.15 + intensity * 0.6})`;
    return `rgba(248, 81, 73, ${0.15 + intensity * 0.6})`;
  };

  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 6 }}>
        雙因子組合 IC 熱力圖
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))', gap: 3 }}>
        {items.map((c, i) => (
          <div key={i} style={{
            padding: '6px 8px', borderRadius: 6,
            background: getColor(c.combo_ic),
            border: '1px solid rgba(255,255,255,0.05)',
            fontSize: 10, lineHeight: 1.4,
          }}>
            <div style={{ fontWeight: 600, color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {c.factor_a}
            </div>
            <div style={{ color: 'var(--text-muted)' }}>× {c.factor_b}</div>
            <div style={{ fontFamily: 'monospace', color: c.combo_ic > 0 ? '#3fb950' : '#f85149', fontWeight: 700 }}>
              {c.combo_ic > 0 ? '+' : ''}{c.combo_ic.toFixed(4)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
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
      <td style={{ padding: '6px 8px', textAlign: 'right', fontFamily: 'monospace', fontSize: 10,
        color: (item.p_value_recent ?? 1) <= 0.01 ? '#3fb950' : (item.p_value_recent ?? 1) <= 0.05 ? '#e3a008' : '#f85149' }}>
        {item.p_value_recent != null ? item.p_value_recent.toFixed(3) : '—'}
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

function buildScanSummary(r: FactorScanResult): string {
  if (r.status !== 'success') return '';
  const parts: string[] = [];
  parts.push(`市場體制:${r.regime?.label ?? '未知'}(ADX=${r.regime?.adx ?? '?'})`);
  if (r.ic_threshold_used) parts.push(`IC門檻:${r.ic_threshold_used}`);
  if (r.positive_top?.length) {
    parts.push('正相關TOP:' + r.positive_top.slice(0, 3).map(
      (f) => `${f.factor}(IC=${f.ic_recent > 0 ? '+' : ''}${f.ic_recent.toFixed(3)},p=${(f.p_value_recent ?? 0).toFixed(3)},${f.decay_trend})`
    ).join(','));
  }
  if (r.negative_top?.length) {
    parts.push('負相關TOP:' + r.negative_top.slice(0, 3).map(
      (f) => `${f.factor}(IC=${f.ic_recent.toFixed(3)},p=${(f.p_value_recent ?? 0).toFixed(3)},${f.decay_trend})`
    ).join(','));
  }

  // 優先使用策略分級的建議版
  const moderate = r.strategy_tiers?.moderate;
  if (moderate && moderate.conditions.length > 0) {
    const condStr = moderate.conditions.map(
      (c) => `${c.factor} ${c.description}(IC=${c.abs_ic},報酬${c.return_pct > 0 ? '+' : ''}${c.return_pct}%)`
    ).join(',');
    parts.push(`建議版策略(${moderate.trigger_count}次觸發):${condStr}`);
  } else if (r.quantile_analysis) {
    const entries = Object.values(r.quantile_analysis)
      .filter((qa) => qa.best_quantile?.entry_suggestion)
      .slice(0, 3)
      .map((qa) => `${qa.factor} ${qa.best_quantile!.entry_suggestion}(報酬${qa.best_quantile!.return_pct > 0 ? '+' : ''}${qa.best_quantile!.return_pct.toFixed(1)}%)`);
    if (entries.length) parts.push('建議進場:' + entries.join(','));
  }

  parts.push(`有效因子${r.effective_count}/${r.total_factors_scanned}`);
  return parts.join('｜');
}

// ─── 主面板 ─────────────────────────────────
export default function FactorScanPanel() {
  const symbol = useChartStore((s) => s.symbol);
  const timeframe = useChartStore((s) => s.timeframe);
  const setLastFactorScan = useChartStore((s) => s.setLastFactorScan);

  const [result, setResult] = useState<FactorScanResult | null>(null);
  const [scanning, setScanning] = useState(false);
  const [showPanel, setShowPanel] = useState(false);
  const [customizerConds, setCustomizerConds] = useState<EditableCond[] | null>(null);
  const [customizerExpanded, setCustomizerExpanded] = useState(false);
  const [activeTier, setActiveTier] = useState<TierKey | null>(null);

  const handleLoadTier = useCallback((key: TierKey) => {
    const tier = result?.strategy_tiers?.[key];
    if (!tier) return;
    const conds = tierToEditableConds(tier.conditions);
    setCustomizerConds(conds);
    setCustomizerExpanded(true);
    setActiveTier(key);
  }, [result]);

  const handleScan = useCallback(async () => {
    setScanning(true);
    setShowPanel(true);
    setActiveTier(null);
    setCustomizerConds(null);
    setCustomizerExpanded(false);
    try {
      const data = await runFactorScan({ symbol, timeframe, forward_period: 5, top_n: 5 });
      setResult(data);
      if (data.status === 'success') {
        setLastFactorScan({
          symbol, timeframe,
          timestamp: Date.now(),
          summary: buildScanSummary(data),
        });
        toast(`因子掃描完成（有效因子 ${data.effective_count || 0}/${data.total_factors_scanned || 0}）`, 'success');
      }
    } catch {
      setResult({ status: 'error', message: '掃描失敗' });
      toast('因子掃描失敗', 'error');
    } finally {
      setScanning(false);
    }
  }, [symbol, timeframe, setLastFactorScan]);

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
                <div>正在掃描所有因子（含衍生因子）...</div>
              </div>
            )}

            {result?.status === 'error' && (
              <div style={{ textAlign: 'center', padding: 40, color: '#f85149' }}>{result.message}</div>
            )}

            {result?.status === 'success' && !scanning && (
              <>
                {/* 掃描警告 */}
                {(result.scan_warnings?.length ?? 0) > 0 && (
                  <div style={{ marginBottom: 12 }}>
                    {result.scan_warnings!.map((w, i) => (
                      <div key={i} style={{ fontSize: 12, color: '#d29922', padding: '4px 12px', background: 'rgba(210,153,34,0.08)', borderRadius: 6, marginBottom: 4 }}>
                        ⚠️ {w}
                      </div>
                    ))}
                  </div>
                )}

                {/* 掃描元資訊 */}
                <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', fontSize: 11, color: 'var(--text-muted)', marginBottom: 10, padding: '0 4px' }}>
                  <span>IC門檻: {result.ic_threshold_used ?? '—'}</span>
                  <span>p-value: ≤{result.p_value_cutoff ?? '—'}</span>
                  <span>分位數: OOS {result.oos_split_ratio ?? '—'}</span>
                  <span>去重: r&gt;{result.dedup_threshold ?? '—'}</span>
                </div>

                {/* 正相關 TOP */}
                {(result.positive_top?.length ?? 0) > 0 && (
                  <Section title={<>正相關 TOP — 因子值↑ → 未來價格↑<WhyButton concept="ic" /></>} color="#3fb950">
                    <FactorTable items={result.positive_top!} headerStyle={thStyle} />
                  </Section>
                )}

                {/* 負相關 TOP */}
                {(result.negative_top?.length ?? 0) > 0 && (
                  <Section title={<>負相關 TOP — 因子值↑ → 未來價格↓<WhyButton concept="ic" /></>} color="#f85149">
                    <FactorTable items={result.negative_top!} headerStyle={thStyle} />
                  </Section>
                )}

                {/* IC 視覺化條形圖 */}
                {((result.positive_top?.length ?? 0) + (result.negative_top?.length ?? 0)) > 0 && (
                  <Section title="IC 分佈視覺化" color="#8b949e">
                    <IcBarChart
                      items={[...(result.positive_top || []), ...(result.negative_top || [])].sort((a, b) => b.ic_recent - a.ic_recent)}
                      title="所有因子 IC（正 → 負）"
                    />
                  </Section>
                )}

                {/* 雙因子熱力圖 */}
                {result.combo_top && (() => {
                  const allCombos = [
                    ...(result.combo_top.positive_combos || []),
                    ...(result.combo_top.negative_combos || []),
                    ...(result.combo_top.hedge_combos || []),
                  ];
                  return allCombos.length > 0 ? (
                    <Section title="雙因子組合強度" color="#d2a8ff">
                      <ComboHeatmap items={allCombos} />
                    </Section>
                  ) : null;
                })()}

                {/* 分位數分析 + 最佳進場建議 */}
                {result.quantile_analysis && Object.keys(result.quantile_analysis).length > 0 && (
                  <Section title={<>最佳進場區間分析<WhyButton concept="quantile" /></>} color="#79c0ff">
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
                  <Section title={<>雙因子組合<WhyButton concept="combo_ic" /></>} color="#d2a8ff">
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

                {/* 策略建構精靈 */}
                <StrategyWizard result={result} onSendMessage={(msg) => {
                  const setPending = useChartStore.getState().setPendingChatMessage;
                  setPending(msg);
                }} />

                {/* 策略分級建議 */}
                <StrategyTiersSection result={result} activeTier={activeTier} onLoadTier={handleLoadTier} />

                {/* 互動式因子調整器 */}
                <FactorCustomizer
                  result={result}
                  externalConds={customizerConds}
                  expanded={customizerExpanded}
                  setExpanded={setCustomizerExpanded}
                />

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

const TIER_META: Record<TierKey, { label: string; emoji: string; color: string; desc: string }> = {
  strict: { label: '嚴格', emoji: '🎯', color: '#f85149', desc: '高品質因子（≥2星 + Alpha Decay 過濾）全部 AND' },
  moderate: { label: '建議', emoji: '⚡', color: '#58a6ff', desc: '最佳雙因子組合 + 強 IC 因子，放寬一級' },
  loose: { label: '寬鬆', emoji: '🌊', color: '#3fb950', desc: '僅保留 IC 最強 2 因子，放寬兩級' },
};

function StrategyTiersSection({ result, activeTier, onLoadTier }: {
  result: FactorScanResult;
  activeTier: TierKey | null;
  onLoadTier: (key: TierKey) => void;
}) {
  const tiers = result.strategy_tiers;

  if (!tiers) return null;
  const tierKeys: TierKey[] = ['strict', 'moderate', 'loose'];
  const available = tierKeys.filter((k) => tiers[k] && (tiers[k]!.conditions.length ?? 0) > 0);
  if (available.length === 0) return null;

  return (
    <Section title={<>策略分級建議<WhyButton concept="strategy_tier" /></>} color="#79c0ff">
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 10 }}>
        選擇一個版本載入到下方調整器，可進一步微調後再送出回測：
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {available.map((key) => {
          const tier = tiers[key]!;
          const meta = TIER_META[key];
          const isRecommended = key === 'moderate';
          const isActive = activeTier === key;

          return (
            <div key={key} style={{
              padding: '10px 12px', borderRadius: 8,
              background: isActive ? `${meta.color}12` : isRecommended ? 'rgba(88,166,255,0.08)' : 'var(--bg-secondary)',
              border: `1px solid ${isActive ? meta.color : isRecommended ? 'rgba(88,166,255,0.4)' : 'var(--border-color)'}`,
              transition: 'all 0.2s',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span>{meta.emoji}</span>
                  <span style={{ fontWeight: 700, fontSize: 13, color: meta.color }}>{meta.label}版</span>
                  {isRecommended && (
                    <span style={{
                      fontSize: 9, padding: '1px 6px', borderRadius: 4,
                      background: 'rgba(88,166,255,0.2)', color: '#58a6ff', fontWeight: 600,
                    }}>推薦</span>
                  )}
                  {isActive && (
                    <span style={{
                      fontSize: 9, padding: '1px 6px', borderRadius: 4,
                      background: `${meta.color}30`, color: meta.color, fontWeight: 600,
                    }}>已載入 ✓</span>
                  )}
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{
                    fontSize: 11, fontFamily: 'monospace',
                    color: tier.trigger_count >= 10 ? '#3fb950' : tier.trigger_count >= 5 ? '#d29922' : '#f85149',
                  }}>
                    觸發 {tier.trigger_count} 次
                    {tier.trigger_count < 5 && ' ⚠️'}
                  </span>
                  <button
                    onClick={() => onLoadTier(key)}
                    style={{
                      padding: '4px 12px', borderRadius: 6, fontSize: 11, fontWeight: 600,
                      border: `1px solid ${meta.color}44`,
                      background: isActive ? `${meta.color}30` : `${meta.color}15`,
                      color: meta.color, cursor: 'pointer', transition: 'all 0.15s',
                    }}
                    onMouseEnter={(e) => { e.currentTarget.style.background = `${meta.color}30`; }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = isActive ? `${meta.color}30` : `${meta.color}15`; }}
                  >
                    {isActive ? '✓ 已載入' : '載入此版本'}
                  </button>
                </div>
              </div>

              <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 6 }}>
                {tier.source || meta.desc}
              </div>

              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                {tier.conditions.map((c, i) => (
                  <span key={i} style={{
                    fontSize: 10, padding: '2px 8px', borderRadius: 4,
                    background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
                    color: 'var(--text-primary)', fontFamily: 'monospace',
                  }}>
                    {c.factor} {c.description}
                    <span style={{ color: 'var(--text-muted)', marginLeft: 4 }}>IC={c.abs_ic}</span>
                  </span>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </Section>
  );
}

// ─── 互動式因子調整器 ───────────────────────
interface EditableCond {
  factor: string;
  operator: string;
  value: number;
  value2?: number;
  enabled: boolean;
  abs_ic: number;
  return_pct: number;
  original_description: string;
}

function FactorCustomizer({ result, externalConds, expanded, setExpanded }: {
  result: FactorScanResult;
  externalConds: EditableCond[] | null;
  expanded: boolean;
  setExpanded: (v: boolean) => void;
}) {
  const symbol = useChartStore((s) => s.symbol);
  const timeframe = useChartStore((s) => s.timeframe);
  const setPendingChatMessage = useChartStore((s) => s.setPendingChatMessage);

  const [conditions, setConditions] = useState<EditableCond[]>([]);
  const [triggerInfo, setTriggerInfo] = useState<{ count: number; total: number; pct: number; lastTime?: string } | null>(null);
  const [previewing, setPreviewing] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout>>();

  // 接收從 tier 按鈕載入的條件
  useEffect(() => {
    if (externalConds && externalConds.length > 0) {
      setConditions(externalConds);
      setTriggerInfo(null);
    }
  }, [externalConds]);

  // 首次從掃描結果初始化（只在無外部注入時）
  useEffect(() => {
    if (externalConds && externalConds.length > 0) return;
    if (!result.quantile_analysis) return;
    const conds: EditableCond[] = [];
    for (const qa of Object.values(result.quantile_analysis)) {
      const bq = qa.best_quantile;
      if (!bq?.entry_suggestion) continue;

      let op = '>';
      let val = bq.range.low;
      let val2: number | undefined;
      const sug = bq.entry_suggestion;
      if (sug.startsWith('>')) {
        op = '>';
        val = bq.range.low;
      } else if (sug.startsWith('<')) {
        op = '<';
        val = bq.range.high;
      } else if (sug.includes('~')) {
        op = 'between';
        val = bq.range.low;
        val2 = bq.range.high;
      }

      const abs_ic = result.positive_top?.find((f) => f.factor === qa.factor)?.ic_recent
        ?? result.negative_top?.find((f) => f.factor === qa.factor)?.ic_recent
        ?? 0;

      conds.push({
        factor: qa.factor, operator: op, value: val, value2: val2,
        enabled: true, abs_ic: Math.abs(abs_ic), return_pct: bq.return_pct,
        original_description: sug,
      });
    }
    conds.sort((a, b) => b.abs_ic - a.abs_ic);
    setConditions(conds);
  }, [result]); // eslint-disable-line react-hooks/exhaustive-deps

  // debounced 觸發預覽
  const fetchPreview = useCallback((conds: EditableCond[]) => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(async () => {
      const apiConds: TriggerCondition[] = conds
        .filter((c) => c.enabled)
        .map((c) => ({ factor: c.factor, operator: c.operator, value: c.value, value2: c.value2, enabled: true }));
      if (apiConds.length === 0) {
        setTriggerInfo(null);
        return;
      }
      setPreviewing(true);
      try {
        const res = await triggerPreview(symbol, timeframe, apiConds);
        if (res.status === 'success') {
          setTriggerInfo({ count: res.trigger_count, total: res.total_bars, pct: res.trigger_pct, lastTime: res.last_trigger_time });
        }
      } catch { /* ignore */ }
      setPreviewing(false);
    }, 400);
  }, [symbol, timeframe]);

  const updateCond = useCallback((idx: number, patch: Partial<EditableCond>) => {
    setConditions((prev) => {
      const next = [...prev];
      next[idx] = { ...next[idx], ...patch };
      fetchPreview(next);
      return next;
    });
  }, [fetchPreview]);

  const handleSendBacktest = useCallback(() => {
    const enabled = conditions.filter((c) => c.enabled);
    if (enabled.length === 0) return;
    const parts: string[] = [];
    parts.push(`根據自訂因子組合，請對 ${symbol} ${timeframe} 做回測：`);
    parts.push(`進場條件（${enabled.length} 個因子）：`);
    for (const c of enabled) {
      const desc = c.operator === 'between'
        ? `${c.value.toFixed(2)} ~ ${(c.value2 ?? c.value).toFixed(2)}`
        : `${c.operator} ${c.value.toFixed(2)}`;
      parts.push(`  - ${c.factor} ${desc} (IC=${c.abs_ic.toFixed(3)})`);
    }
    if (triggerInfo) {
      parts.push(`預估歷史觸發 ${triggerInfo.count} 次 / ${triggerInfo.total} 根 (${triggerInfo.pct}%)`);
    }
    parts.push('請依照以上條件設計策略並執行回測分析。');
    setPendingChatMessage(parts.join('\n'));
  }, [conditions, symbol, timeframe, triggerInfo, setPendingChatMessage]);

  // 展開時觸發一次預覽
  useEffect(() => {
    if (conditions.length > 0 && expanded) fetchPreview(conditions);
  }, [expanded, externalConds]); // eslint-disable-line react-hooks/exhaustive-deps

  if (conditions.length === 0) return null;

  return (
    <Section title="自訂策略組合器" color="#d2a8ff">
      <button
        onClick={() => setExpanded(!expanded)}
        style={{
          width: '100%', padding: '8px 12px', borderRadius: 6, fontSize: 12,
          border: '1px solid var(--border-color)', background: 'var(--bg-secondary)',
          color: 'var(--text-primary)', cursor: 'pointer', textAlign: 'left',
        }}
      >
        {expanded ? '▼' : '▶'} {expanded ? '收起調整器' : '展開 — 自由勾選因子、調整閾值'}
      </button>

      {expanded && (
        <div style={{ marginTop: 8, padding: '8px 0' }}>
          {/* 觸發次數即時顯示 */}
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            padding: '6px 10px', borderRadius: 6, marginBottom: 8,
            background: triggerInfo && triggerInfo.count >= 10
              ? 'rgba(63,185,80,0.1)' : triggerInfo && triggerInfo.count >= 5
              ? 'rgba(210,153,34,0.1)' : 'rgba(248,81,73,0.1)',
            border: '1px solid var(--border-color)',
          }}>
            <span style={{ fontSize: 12, color: 'var(--text-primary)' }}>
              {previewing ? '⟳ 計算中...' : triggerInfo
                ? <>歷史觸發 <strong style={{
                    color: triggerInfo.count >= 10 ? '#3fb950' : triggerInfo.count >= 5 ? '#d29922' : '#f85149',
                  }}>{triggerInfo.count} 次</strong> / {triggerInfo.total} 根 ({triggerInfo.pct}%)
                  {triggerInfo.count < 10 && <span style={{ color: '#d29922', marginLeft: 4 }}>⚠ 建議 ≥10 次</span>}
                </>
                : '勾選因子後自動計算觸發次數'
              }
            </span>
            {triggerInfo?.lastTime && (
              <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>最近：{triggerInfo.lastTime.slice(0, 16)}</span>
            )}
          </div>

          {/* 因子條件列表 */}
          {conditions.map((c, i) => (
            <div key={c.factor} style={{
              display: 'flex', alignItems: 'center', gap: 6, padding: '4px 0',
              borderBottom: '1px solid var(--border-color)', opacity: c.enabled ? 1 : 0.45,
            }}>
              <input
                type="checkbox"
                checked={c.enabled}
                onChange={(e) => updateCond(i, { enabled: e.target.checked })}
                style={{ cursor: 'pointer' }}
              />
              <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-primary)', minWidth: 110, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {c.factor}
              </span>
              <span style={{ fontSize: 10, color: 'var(--text-muted)', minWidth: 55 }}>IC={c.abs_ic.toFixed(3)}</span>

              <select
                value={c.operator}
                onChange={(e) => updateCond(i, { operator: e.target.value })}
                disabled={!c.enabled}
                style={{ fontSize: 11, padding: '2px 4px', borderRadius: 4, border: '1px solid var(--border-color)', background: 'var(--bg-primary)', color: 'var(--text-primary)' }}
              >
                <option value=">">{'>'}</option>
                <option value="<">{'<'}</option>
                <option value=">=">{'>='}</option>
                <option value="<=">{' <='}</option>
                <option value="between">區間</option>
              </select>

              <input
                type="number"
                value={c.value}
                step="0.01"
                onChange={(e) => updateCond(i, { value: parseFloat(e.target.value) || 0 })}
                disabled={!c.enabled}
                style={{ width: 70, fontSize: 11, padding: '2px 6px', borderRadius: 4, border: '1px solid var(--border-color)', background: 'var(--bg-primary)', color: 'var(--text-primary)', fontFamily: 'monospace' }}
              />
              {c.operator === 'between' && (
                <>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>~</span>
                  <input
                    type="number"
                    value={c.value2 ?? c.value}
                    step="0.01"
                    onChange={(e) => updateCond(i, { value2: parseFloat(e.target.value) || 0 })}
                    disabled={!c.enabled}
                    style={{ width: 70, fontSize: 11, padding: '2px 6px', borderRadius: 4, border: '1px solid var(--border-color)', background: 'var(--bg-primary)', color: 'var(--text-primary)', fontFamily: 'monospace' }}
                  />
                </>
              )}
            </div>
          ))}

          {/* 送出回測按鈕 */}
          <button
            onClick={handleSendBacktest}
            disabled={conditions.filter((c) => c.enabled).length === 0}
            style={{
              width: '100%', padding: '8px 16px', borderRadius: 6, marginTop: 8,
              border: '1px solid rgba(210,168,255,0.3)',
              background: 'linear-gradient(135deg, rgba(210,168,255,0.15), rgba(88,166,255,0.1))',
              color: '#d2a8ff', cursor: 'pointer', fontSize: 12, fontWeight: 700, transition: 'all 0.2s',
            }}
          >
            🎛️ 用自訂條件回測
          </button>
        </div>
      )}
    </Section>
  );
}

// ─── 策略建構精靈 ───────────────────────────
function StrategyWizard({ result, onSendMessage }: { result: FactorScanResult; onSendMessage: (msg: string) => void }) {
  const [step, setStep] = useState(0);
  const [selectedFactors, setSelectedFactors] = useState<Set<string>>(new Set());
  const [direction, setDirection] = useState<'long' | 'short'>('long');
  const symbol = useChartStore(s => s.symbol);
  const timeframe = useChartStore(s => s.timeframe);

  const allFactors: FactorScanItem[] = [
    ...(result.positive_top || []),
    ...(result.negative_top || []),
  ];

  const toggleFactor = (f: string) => {
    const next = new Set(selectedFactors);
    if (next.has(f)) next.delete(f); else next.add(f);
    setSelectedFactors(next);
  };

  const buildPrompt = () => {
    const factors = allFactors.filter(f => selectedFactors.has(f.factor));
    const conditions = factors.map(f => {
      const qa = result.quantile_analysis?.[
        Object.keys(result.quantile_analysis || {}).find(k =>
          (result.quantile_analysis as Record<string, { factor: string }>)[k]?.factor === f.factor
        ) || ''
      ];
      const suggestion = qa?.best_quantile?.entry_suggestion;
      return suggestion ? `${f.factor} ${suggestion}` : f.factor;
    });
    return `請對 ${symbol} ${timeframe} 做${direction === 'long' ? '做多' : '做空'}策略回測：\n進場條件：${conditions.join('、')}\n\n選用因子：${factors.map(f => `${f.factor}(IC=${f.ic_recent > 0 ? '+' : ''}${f.ic_recent.toFixed(3)})`).join(', ')}\n\n請設計策略並執行回測分析。`;
  };

  const steps = ['選擇方向', '選擇因子', '確認送出'];

  return (
    <Section title="🧙 策略建構精靈" color="#d2a8ff">
      {/* 步驟指示器 */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 12 }}>
        {steps.map((s, i) => (
          <div key={i} style={{
            flex: 1, textAlign: 'center', padding: '4px 0',
            borderRadius: 4, fontSize: 10, fontWeight: i === step ? 700 : 400,
            background: i === step ? 'rgba(210,168,255,0.2)' : 'var(--bg-tertiary)',
            color: i === step ? '#d2a8ff' : i < step ? '#3fb950' : 'var(--text-muted)',
            border: i === step ? '1px solid rgba(210,168,255,0.4)' : '1px solid transparent',
            cursor: i < step ? 'pointer' : 'default',
          }}
            onClick={() => { if (i < step) setStep(i); }}
          >
            {i < step ? '✓ ' : ''}{s}
          </div>
        ))}
      </div>

      {/* Step 0: Direction */}
      {step === 0 && (
        <div>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 8 }}>選擇策略方向：</div>
          <div style={{ display: 'flex', gap: 8 }}>
            {(['long', 'short'] as const).map(d => (
              <button
                key={d}
                onClick={() => setDirection(d)}
                style={{
                  flex: 1, padding: '10px 0', borderRadius: 8,
                  border: direction === d ? `2px solid ${d === 'long' ? '#3fb950' : '#f85149'}` : '1px solid var(--border-color)',
                  background: direction === d ? (d === 'long' ? 'rgba(63,185,80,0.1)' : 'rgba(248,81,73,0.1)') : 'var(--bg-tertiary)',
                  color: d === 'long' ? '#3fb950' : '#f85149',
                  fontSize: 13, fontWeight: 700, cursor: 'pointer',
                }}
              >
                {d === 'long' ? '📈 做多' : '📉 做空'}
              </button>
            ))}
          </div>
          <button
            onClick={() => setStep(1)}
            style={{
              width: '100%', marginTop: 10, padding: '8px', borderRadius: 6,
              background: 'rgba(210,168,255,0.15)', border: '1px solid rgba(210,168,255,0.3)',
              color: '#d2a8ff', fontSize: 12, fontWeight: 600, cursor: 'pointer',
            }}
          >
            下一步 →
          </button>
        </div>
      )}

      {/* Step 1: Pick factors */}
      {step === 1 && (
        <div>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 8 }}>
            勾選要納入策略的因子（建議 2~4 個）<WhyButton concept="combo_ic" />
          </div>
          <div style={{ maxHeight: 200, overflowY: 'auto' }}>
            {allFactors.map(f => {
              const sel = selectedFactors.has(f.factor);
              return (
                <div
                  key={f.factor}
                  onClick={() => toggleFactor(f.factor)}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 8,
                    padding: '6px 8px', marginBottom: 2, borderRadius: 6,
                    cursor: 'pointer',
                    background: sel ? 'rgba(210,168,255,0.1)' : 'transparent',
                    border: sel ? '1px solid rgba(210,168,255,0.3)' : '1px solid transparent',
                  }}
                >
                  <span style={{ fontSize: 14 }}>{sel ? '☑️' : '⬜'}</span>
                  <span style={{ flex: 1, fontSize: 11, color: 'var(--text-primary)' }}>{f.factor}</span>
                  <span style={{
                    fontSize: 10, fontFamily: 'monospace',
                    color: f.ic_recent > 0 ? '#3fb950' : '#f85149',
                  }}>
                    IC={f.ic_recent > 0 ? '+' : ''}{f.ic_recent.toFixed(3)}
                  </span>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>{f.status_label}</span>
                </div>
              );
            })}
          </div>
          <div style={{ display: 'flex', gap: 6, marginTop: 8 }}>
            <button onClick={() => setStep(0)} style={{
              flex: 1, padding: '6px', borderRadius: 6, fontSize: 11, cursor: 'pointer',
              background: 'var(--bg-tertiary)', border: '1px solid var(--border-color)', color: 'var(--text-secondary)',
            }}>← 上一步</button>
            <button onClick={() => setStep(2)} disabled={selectedFactors.size === 0} style={{
              flex: 1, padding: '6px', borderRadius: 6, fontSize: 11, cursor: 'pointer',
              background: selectedFactors.size > 0 ? 'rgba(210,168,255,0.15)' : 'var(--bg-tertiary)',
              border: '1px solid rgba(210,168,255,0.3)', color: '#d2a8ff',
              opacity: selectedFactors.size > 0 ? 1 : 0.4,
            }}>下一步 →</button>
          </div>
        </div>
      )}

      {/* Step 2: Confirm and send */}
      {step === 2 && (
        <div>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 8 }}>確認策略：</div>
          <div style={{
            padding: '10px 12px', borderRadius: 8,
            background: 'var(--bg-tertiary)', fontSize: 11, lineHeight: 1.6,
            color: 'var(--text-primary)', marginBottom: 10,
          }}>
            <div>📊 標的：<strong>{symbol}</strong> ({timeframe})</div>
            <div>{direction === 'long' ? '📈 做多' : '📉 做空'}</div>
            <div>🔬 因子數：{selectedFactors.size} 個</div>
            <div style={{ marginTop: 4, color: 'var(--text-muted)', fontSize: 10 }}>
              {Array.from(selectedFactors).join(' + ')}
            </div>
          </div>
          <div style={{ display: 'flex', gap: 6 }}>
            <button onClick={() => setStep(1)} style={{
              flex: 1, padding: '6px', borderRadius: 6, fontSize: 11, cursor: 'pointer',
              background: 'var(--bg-tertiary)', border: '1px solid var(--border-color)', color: 'var(--text-secondary)',
            }}>← 修改</button>
            <button onClick={() => { onSendMessage(buildPrompt()); }} style={{
              flex: 1, padding: '8px', borderRadius: 6, fontSize: 12, fontWeight: 700, cursor: 'pointer',
              background: 'linear-gradient(135deg, rgba(210,168,255,0.2), rgba(88,166,255,0.15))',
              border: '1px solid rgba(210,168,255,0.4)', color: '#d2a8ff',
            }}>🚀 送出回測</button>
          </div>
        </div>
      )}
    </Section>
  );
}

function Section({ title, color, children }: { title: React.ReactNode; color: string; children: React.ReactNode }) {
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
          <th style={headerStyle}>p值</th>
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
