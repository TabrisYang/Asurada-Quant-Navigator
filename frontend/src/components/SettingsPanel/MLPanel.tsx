/** 阿斯拉量化系統 — ML 增強面板
 *
 * 整合：設定 / 訓練 / 多模型比較 / 狀態監控 / 即時預測
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useChartStore } from '../../stores/chartStore';
import {
  fetchMLModels, fetchMLSettings, updateMLSettings,
  trainMLModel, compareMLModels, predictML,
  fetchTrainedModels,
  fetchMLPerformanceHistory, checkMLHealth,
  fetchMLPredictionAccuracy,
  type MLModel, type MLSettings,
} from '../../services/api';


/** 數字輸入：本地暫存 → blur/Enter 時才送出 */
function NumInput({ value, onCommit, min, max, step, disabled, style }: {
  value: number;
  onCommit: (v: number) => void;
  min?: number; max?: number; step?: number;
  disabled?: boolean;
  style?: React.CSSProperties;
}) {
  const [local, setLocal] = useState(String(value));
  const prevValue = useRef(value);

  // 外部 value 變化時同步（例如重置按鈕）
  useEffect(() => {
    if (value !== prevValue.current) {
      setLocal(String(value));
      prevValue.current = value;
    }
  }, [value]);

  const commit = () => {
    let n = Number(local);
    if (isNaN(n)) { setLocal(String(value)); return; }
    if (min !== undefined && n < min) n = min;
    if (max !== undefined && n > max) n = max;
    setLocal(String(n));
    if (n !== value) onCommit(n);
  };

  return (
    <input
      type="number"
      min={min} max={max} step={step}
      value={local}
      disabled={disabled}
      onChange={e => setLocal(e.target.value)}
      onBlur={commit}
      onKeyDown={e => { if (e.key === 'Enter') commit(); }}
      style={style}
    />
  );
}

/** 參數說明 tooltip — hover ⓘ 顯示說明 */
function Tip({ text }: { text: string }) {
  const [show, setShow] = useState(false);
  return (
    <span className="relative inline-block ml-1"
      onMouseEnter={() => setShow(true)} onMouseLeave={() => setShow(false)}>
      <span style={{ color: 'var(--text-tertiary)', cursor: 'help', fontSize: 11 }}>ⓘ</span>
      {show && (
        <div style={{
          position: 'absolute', left: 0, top: '100%', zIndex: 50,
          width: 220, padding: '8px 10px', borderRadius: 6,
          background: 'var(--bg-primary)', border: '1px solid var(--border-color)',
          boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
          fontSize: 11, lineHeight: 1.6, color: 'var(--text-secondary)',
          pointerEvents: 'none',
        }}>
          {text}
        </div>
      )}
    </span>
  );
}

type SubTab = 'settings' | 'train' | 'compare' | 'status' | 'monitor';

/* ═══════════════════════════════════════════ */

export default function MLPanel() {
  const [subTab, setSubTab] = useState<SubTab>('settings');
  const [showGuide, setShowGuide] = useState(() => {
    return localStorage.getItem('asura_ml_guide_dismissed') !== 'true';
  });
  const [expandedStep, setExpandedStep] = useState<number | null>(null);

  const tabs: { key: SubTab; label: string }[] = [
    { key: 'settings', label: '設定' },
    { key: 'train', label: '訓練模型' },
    { key: 'compare', label: '模型比較' },
    { key: 'status', label: '狀態/預測' },
    { key: 'monitor', label: '監控' },
  ];

  return (
    <div>
      {/* ⚠️ ML 模組停用橫幅（顯著提示，始終顯示） */}
      <div style={{
        padding: '14px 16px',
        borderRadius: 8,
        marginBottom: 14,
        background: 'rgba(245,158,11,0.10)',
        border: '1px solid rgba(245,158,11,0.45)',
        fontSize: 13,
        lineHeight: 1.7,
        color: 'var(--text-primary)',
      }}>
        <div style={{ fontWeight: 700, color: 'var(--accent-orange, #f59e0b)', marginBottom: 6, fontSize: 14 }}>
          ⚠️ ML 模組目前處於實驗性停用狀態
        </div>
        <div style={{ color: 'var(--text-secondary)' }}>
          所有訓練 / 預測 / 模型管理功能皆已暫停，AI 對話也不會再注入 ML 預測。
          下方界面仍可瀏覽，但操作會回傳「模組已停用」錯誤。
        </div>
        <div style={{ marginTop: 8, color: 'var(--text-secondary)', fontSize: 12 }}>
          原因：ML pipeline 過去從未真正接通到 chat 介面（236 筆預測中 0 筆 ML 增強），
          系統的「文字預測」64% 命中率已是高 baseline。<br />
          重新啟用方式：開發者修改 <code style={{ background: 'rgba(0,0,0,0.2)', padding: '1px 4px', borderRadius: 3 }}>backend/app/core/ml/_settings.py</code> 的 <code style={{ background: 'rgba(0,0,0,0.2)', padding: '1px 4px', borderRadius: 3 }}>ML_ENABLED = True</code> 後重啟後端。
        </div>
      </div>

      {/* 新手引導卡片（可收合，始終可見） */}
      {showGuide && (
        <div style={{
          padding: '12px 14px', borderRadius: 8, marginBottom: 14,
          background: 'rgba(88,166,255,0.08)', border: '1px solid rgba(88,166,255,0.25)',
          fontSize: 12, lineHeight: 1.7,
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
            <span style={{ fontWeight: 700, color: 'var(--accent-blue)', fontSize: 13 }}>
              ML 模型使用指南
            </span>
            <button
              onClick={() => { setShowGuide(false); localStorage.setItem('asura_ml_guide_dismissed', 'true'); }}
              style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: 14 }}
            >x</button>
          </div>
          <div style={{ color: 'var(--text-primary)' }}>
            {[
              { title: 'Step 1 — 先到「同步」面板抓取至少 90 天的歷史數據（建議 180 天以上）',
                why: 'ML 模型需要足夠的歷史樣本才能學到有意義的模式。樣本太少會導致過擬合（模型背住歷史但無法預測未來）。180 天以上能覆蓋多種市場狀態（趨勢/盤整/反轉），讓模型更穩健。' },
              { title: 'Step 2 — 切到「模型比較」分頁，一鍵訓練全部模型，系統會自動找出最佳模型',
                why: '不同模型（LightGBM、XGBoost、RF、LSTM 等）擅長捕捉不同的市場特徵。透過同時比較所有模型的 OOS（樣本外）表現，避免只依賴單一演算法的偏差，找到最適合當前幣對的模型。' },
              { title: 'Step 3 — 在「設定」分頁確認啟用模式為「自動」，同步新資料時會自動重訓',
                why: '市場狀態會隨時間改變（regime change），模型的預測能力會逐漸衰退。自動重訓搭配「冠軍挑戰者」機制，確保只有表現更好的新模型才會取代舊的，防止靜默退化。' },
              { title: 'Step 4 — 到「監控」分頁追蹤預測準確率，定期檢查模型是否退化',
                why: '即使有自動重訓，仍需人工監控。監控面板提供 MFE/MAE（最大有利/不利偏移）、Wilson 信心區間等統計，幫助你判斷模型是否真正有效，還是只是運氣好。' },
            ].map((step, i) => (
              <div key={i} style={{ marginBottom: 4 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <strong style={{ whiteSpace: 'nowrap' }}>{step.title}</strong>
                  <button
                    onClick={() => setExpandedStep(expandedStep === i ? null : i)}
                    style={{
                      background: 'none', border: 'none', cursor: 'pointer', fontSize: 11,
                      color: 'var(--accent-blue)', whiteSpace: 'nowrap', padding: 0,
                    }}
                  >{expandedStep === i ? '收起' : '為什麼？'}</button>
                </div>
                {expandedStep === i && (
                  <div style={{
                    marginTop: 4, marginBottom: 4, marginLeft: 12, padding: '6px 10px',
                    borderRadius: 6, fontSize: 11, lineHeight: 1.6,
                    background: 'rgba(88,166,255,0.05)', color: 'var(--text-secondary)',
                    borderLeft: '2px solid rgba(88,166,255,0.3)',
                  }}>
                    {step.why}
                  </div>
                )}
              </div>
            ))}
          </div>
          <div style={{ marginTop: 8, color: 'var(--text-secondary)', fontSize: 11 }}>
            模型訓練完成後，聊天分析時會自動注入 ML 預測結果作為參考依據。
          </div>
        </div>
      )}

      {/* 子分頁 */}
      <div style={{ display: 'flex', gap: '4px', marginBottom: '14px' }}>
        {tabs.map(t => (
          <button
            key={t.key}
            onClick={() => setSubTab(t.key)}
            style={{
              flex: 1, padding: '5px 0', borderRadius: '6px', fontSize: '12px',
              cursor: 'pointer', transition: 'all 0.15s',
              border: subTab === t.key ? '1px solid var(--accent-blue)' : '1px solid var(--border-color)',
              background: subTab === t.key ? 'rgba(88,166,255,0.12)' : 'transparent',
              color: subTab === t.key ? 'var(--accent-blue)' : 'var(--text-secondary)',
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      {subTab === 'settings' && <SettingsTab />}
      {subTab === 'train' && <TrainTab />}
      {subTab === 'compare' && <CompareTab />}
      {subTab === 'status' && <StatusTab />}
      {subTab === 'monitor' && <MonitorTab />}
    </div>
  );
}


/* ─── 共用樣式 ─────────────────────────── */

const S = {
  label: {
    color: 'var(--text-secondary)', fontSize: '12px', marginBottom: '4px', display: 'block',
  } as React.CSSProperties,
  input: {
    width: '100%', padding: '6px 10px', borderRadius: '6px',
    border: '1px solid var(--border-color)', background: 'var(--bg-secondary)',
    color: 'var(--text-primary)', fontSize: '13px',
  } as React.CSSProperties,
  row: {
    display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '12px',
  } as React.CSSProperties,
  btn: (active: boolean) => ({
    flex: 1, padding: '6px', borderRadius: '6px', fontSize: '13px', cursor: 'pointer',
    border: active ? '1px solid var(--accent-blue)' : '1px solid var(--border-color)',
    background: active ? 'rgba(88,166,255,0.15)' : 'var(--bg-secondary)',
    color: active ? 'var(--accent-blue)' : 'var(--text-secondary)',
  } as React.CSSProperties),
  card: {
    padding: '10px 12px', borderRadius: '8px', marginBottom: '8px',
    border: '1px solid var(--border-color)', background: 'var(--bg-secondary)',
  } as React.CSSProperties,
  warn: {
    padding: '8px 10px', borderRadius: '6px', marginBottom: '8px', fontSize: '12px',
    background: 'rgba(255,200,0,0.12)', color: '#e3a008',
    border: '1px solid rgba(255,200,0,0.25)',
  } as React.CSSProperties,
  err: {
    padding: '8px 10px', borderRadius: '6px', marginBottom: '8px', fontSize: '12px',
    background: 'rgba(248,81,73,0.12)', color: '#f85149',
  } as React.CSSProperties,
  ok: {
    padding: '8px 10px', borderRadius: '6px', marginBottom: '8px', fontSize: '12px',
    background: 'rgba(63,185,80,0.12)', color: '#3fb950',
  } as React.CSSProperties,
};


/* ═══════════════════  設定  ═══════════════════ */

function SettingsTab() {
  const [settings, setSettings] = useState<MLSettings | null>(null);
  const [models, setModels] = useState<MLModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState('');
  const [showAdv, setShowAdv] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const [s, m] = await Promise.all([fetchMLSettings(), fetchMLModels()]);
        setSettings(s);
        setModels(m);
      } catch { setMsg('載入失敗'); }
      finally { setLoading(false); }
    })();
  }, []);

  const save = async (updates: Partial<MLSettings>) => {
    setSaving(true); setMsg('');
    try {
      const s = await updateMLSettings(updates);
      setSettings(s);
      const w = s._warnings;
      setMsg(w?.length ? w.join('；') : '已儲存');
      setTimeout(() => setMsg(''), 3000);
    } catch { setMsg('儲存失敗'); }
    finally { setSaving(false); }
  };

  if (loading) return <div style={{ color: 'var(--text-secondary)', padding: '16px 0', textAlign: 'center' }}>載入中...</div>;
  if (!settings) return <div style={S.err}>無法載入 ML 設定</div>;

  return (
    <div>
      {msg && <div style={msg.includes('失敗') ? S.err : S.ok}>{msg}</div>}

      {/* 啟用模式 */}
      <div style={{ marginBottom: '12px' }}>
        <label style={S.label}>啟用模式 <Tip text="auto = 有新數據時自動重訓並注入分析；on = 手動訓練但結果會注入分析；off = 完全關閉 ML 功能" /></label>
        <div style={{ display: 'flex', gap: '8px' }}>
          {(['auto', 'on', 'off'] as const).map(mode => (
            <button key={mode} onClick={() => save({ enabled: mode })} disabled={saving} style={S.btn(settings.enabled === mode)}>
              {mode === 'auto' ? '自動判斷' : mode === 'on' ? '強制開啟' : '關閉'}
            </button>
          ))}
        </div>
      </div>

      {/* 模型選擇 */}
      <div style={{ marginBottom: '12px' }}>
        <label style={S.label}>預測模型 <Tip text="LightGBM 速度快適合入門；LSTM / Transformer 適合捕捉時序規律但訓練較慢" /></label>
        <select value={settings.model_id || 'lightgbm'} onChange={e => save({ model_id: e.target.value })} style={S.input}>
          {models.map(m => (
            <option key={m.id} value={m.id} disabled={!m.available}>
              {m.name} ({m.category}){!m.available ? ' [未安裝]' : ''} — {m.training_speed}
            </option>
          ))}
        </select>
      </div>

      {/* ── 前兆模式辨識設定 ── */}
      <div style={{ marginBottom: '12px', padding: '10px 12px', borderRadius: '8px', border: '1px solid var(--border-color)', background: 'var(--bg-tertiary)' }}>
        <label style={{ ...S.label, fontWeight: 600, color: 'var(--text-primary)', fontSize: '13px', marginBottom: '8px' }}>前兆模式辨識</label>
        <div style={S.row}>
          <div>
            <label style={S.label}>預測方向 <Tip text="訓練模型預測「上漲」還是「下跌」。選 up 則模型學習上漲前的特徵模式，選 down 則學習下跌前的模式" /></label>
            <div style={{ display: 'flex', gap: '6px' }}>
              {(['up', 'down'] as const).map(d => (
                <button key={d} onClick={() => save({ target_direction: d })} disabled={saving}
                  style={{
                    ...S.btn(settings.target_direction === d),
                    flex: 1,
                    color: settings.target_direction === d
                      ? (d === 'up' ? '#3fb950' : '#f85149')
                      : 'var(--text-secondary)',
                    borderColor: settings.target_direction === d
                      ? (d === 'up' ? '#3fb950' : '#f85149')
                      : 'var(--border-color)',
                    background: settings.target_direction === d
                      ? (d === 'up' ? 'rgba(63,185,80,0.12)' : 'rgba(248,81,73,0.12)')
                      : 'var(--bg-secondary)',
                  }}>
                  {d === 'up' ? '上漲' : '下跌'}
                </button>
              ))}
            </div>
          </div>
          <div>
            <label style={S.label}>幅度門檻 <Tip text="漲/跌超過多少 % 才算「有效移動」。門檻越高，信號越少但越準確；越低信號多但雜訊也多" /></label>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <input type="range" min="0.5" max="15" step="0.5"
                value={Math.round((settings.target_threshold || 0.03) * 100 * 10) / 10}
                onChange={e => save({ target_threshold: Number(e.target.value) / 100 })}
                style={{ flex: 1 }} />
              <span style={{ color: 'var(--text-primary)', fontSize: '13px', minWidth: '40px' }}>
                {((settings.target_threshold || 0.03) * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
        <div style={S.row}>
          <div>
            <label style={S.label}>回看窗口（K 線數） <Tip text="模型觀察過去幾根 K 線的特徵來做判斷。太短資訊不足，太長會包含過期雜訊" /></label>
            <NumInput value={settings.lookback_window || 7} min={3} max={30} step={1}
              onCommit={v => save({ lookback_window: v })} style={S.input} />
          </div>
          <div>
            <label style={S.label}>預測期數（K 線數） <Tip text="預測未來幾根 K 線內是否達到門檻。期數短 = 短線交易，期數長 = 波段交易" /></label>
            <NumInput value={settings.forward_period || 5} min={1} max={100} step={1}
              onCommit={v => save({ forward_period: v })} style={S.input} />
          </div>
        </div>
        <p style={{ fontSize: '11px', color: 'var(--text-tertiary)', margin: '4px 0 0 0' }}>
          分析「未來 {settings.forward_period || 5} 根 K 線{settings.target_direction === 'down' ? '下跌' : '上漲'} {((settings.target_threshold || 0.03) * 100).toFixed(1)}% 以上」之前 {settings.lookback_window || 7} 根 K 線的特徵模式
        </p>
      </div>

      {/* 採用門檻 + 共識 */}
      <div style={S.row}>
        <div>
          <label style={S.label}>信心採用門檻 <Tip text="模型預測機率要超過此值才會被採用。60% 是平衡點，調高會減少信號但提高準確率" /></label>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <input type="range" min="52" max="85" step="1"
              value={Math.round((settings.threshold || 0.6) * 100)}
              onChange={e => save({ threshold: Number(e.target.value) / 100 })}
              style={{ flex: 1 }} />
            <span style={{ color: 'var(--text-primary)', fontSize: '13px', minWidth: '35px' }}>
              {Math.round((settings.threshold || 0.6) * 100)}%
            </span>
          </div>
        </div>
      </div>

      {/* 共識 + 解釋 */}
      <div style={S.row}>
        <div>
          <label style={S.label}>多模型共識 <Tip text="best = 只用最佳模型；vote = 多模型投票表決；weighted = 按準確率加權平均。多模型共識更穩健但需先訓練多個模型" /></label>
          <select value={settings.consensus_mode || 'best'} onChange={e => save({ consensus_mode: e.target.value })} style={S.input}>
            <option value="best">最佳單模型</option>
            <option value="vote">多數投票</option>
            <option value="weighted_avg">加權平均</option>
          </select>
        </div>
        <div>
          <label style={S.label}>顯示 ML 解釋 <Tip text="開啟後 AI 分析報告中會顯示模型認為最重要的驅動特徵及其數值" /></label>
          <button onClick={() => save({ show_explanation: !settings.show_explanation })}
            style={{
              ...S.input, cursor: 'pointer', textAlign: 'center',
              background: settings.show_explanation ? 'rgba(63,185,80,0.15)' : 'var(--bg-secondary)',
              color: settings.show_explanation ? '#3fb950' : 'var(--text-secondary)',
              border: settings.show_explanation ? '1px solid #3fb950' : '1px solid var(--border-color)',
            }}>
            {settings.show_explanation ? '顯示驅動因子' : '隱藏'}
          </button>
        </div>
      </div>

      {/* 進階設定 */}
      <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '12px' }}>
        <button onClick={() => setShowAdv(!showAdv)} style={{
          background: 'transparent', border: 'none', cursor: 'pointer',
          color: 'var(--text-secondary)', fontSize: '13px', padding: 0,
          display: 'flex', alignItems: 'center', gap: '4px',
        }}>
          <span style={{ transform: showAdv ? 'rotate(90deg)' : 'none', transition: 'transform 0.2s', display: 'inline-block' }}>
            &#9654;
          </span>
          進階 ML 設定
        </button>

        {showAdv && (
          <div style={{ marginTop: '12px' }}>
            <div style={S.row}>
              <div>
                <label style={S.label}>特徵集 <Tip text="compact = 8 個基礎特徵（快）；standard = 15 個（推薦）；full = 25+ 個（慢但能捕捉更多模式）" /></label>
                <select value={settings.feature_set || 'standard'} onChange={e => save({ feature_set: e.target.value })} style={S.input}>
                  <option value="compact">精簡 (8 特徵)</option>
                  <option value="standard">標準 (15 特徵)</option>
                  <option value="full">完整 (25+ 特徵)</option>
                </select>
              </div>
              <div>
                <label style={S.label}>訓練窗口 <Tip text="用多少根 K 線來訓練模型。500 根 4h K 線 ≈ 83 天。太少會過擬合，太多會包含過時行情" /></label>
                <NumInput value={settings.train_window || 500} min={200} max={5000} step={50}
                  onCommit={v => save({ train_window: v })} style={S.input} />
              </div>
            </div>
            <div style={S.row}>
              <div>
                <label style={S.label}>再訓練頻率（每 N 根） <Tip text="每過幾根新 K 線就重新訓練一次。頻率高 = 適應快但耗資源；頻率低 = 穩定但可能過時" /></label>
                <NumInput value={settings.retrain_interval || 50} min={10} max={500} step={10}
                  onCommit={v => save({ retrain_interval: v })} style={S.input} />
              </div>
              <div>
                <label style={S.label}>最低樣本量 <Tip text="歷史數據低於此值時不訓練模型，防止樣本太少導致過擬合" /></label>
                <NumInput value={settings.min_samples || 500} min={100} max={5000} step={50}
                  onCommit={v => save({ min_samples: v })} style={S.input} />
              </div>
            </div>
            <div style={S.row}>
              <div>
                <label style={S.label}>Walk Forward 驗證 <Tip text="模擬真實交易的前進式驗證：用過去的數據訓練、未來的數據測試。強烈建議保持開啟，關閉會高估模型表現" /></label>
                <button onClick={() => save({ walk_forward: !settings.walk_forward })}
                  style={{
                    ...S.input, cursor: 'pointer', textAlign: 'center',
                    background: settings.walk_forward ? 'rgba(63,185,80,0.15)' : 'var(--bg-secondary)',
                    color: settings.walk_forward ? '#3fb950' : 'var(--text-secondary)',
                    border: settings.walk_forward ? '1px solid #3fb950' : '1px solid var(--border-color)',
                  }}>
                  {settings.walk_forward ? '啟用' : '停用'}
                </button>
              </div>
              <div>
                <label style={S.label}>WF 窗口數 <Tip text="Walk Forward 切幾段來做前進式驗證。越多統計結果越穩定，但訓練時間越長" /></label>
                <NumInput value={settings.wf_windows || 5} min={2} max={15} step={1}
                  onCommit={v => save({ wf_windows: v })} disabled={!settings.walk_forward} style={S.input} />
              </div>
            </div>
            <button onClick={() => save({
              enabled: 'auto', model_id: 'lightgbm', feature_set: 'standard',
              forward_period: 5, threshold: 0.60, show_explanation: true,
              train_window: 500, retrain_interval: 50, walk_forward: true,
              wf_windows: 5, min_samples: 500, consensus_mode: 'best',
              target_direction: 'up', target_threshold: 0.03, lookback_window: 7,
            })} disabled={saving}
              style={{
                width: '100%', padding: '8px', borderRadius: '6px', fontSize: '13px',
                cursor: 'pointer', marginTop: '4px',
                border: '1px solid var(--border-color)', background: 'var(--bg-secondary)',
                color: 'var(--text-secondary)',
              }}>
              重置為建議值
            </button>
          </div>
        )}
      </div>

      {/* 模型一覽 */}
      <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '12px', marginTop: '12px' }}>
        <label style={{ ...S.label, fontWeight: 600, color: 'var(--text-primary)', fontSize: '13px' }}>模型一覽</label>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          {models.map(m => (
            <div key={m.id} style={{ ...S.card, opacity: m.available ? 1 : 0.5 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ color: 'var(--text-primary)', fontWeight: 500, fontSize: '12px' }}>{m.name}</span>
                <span style={{
                  fontSize: '10px', padding: '1px 6px', borderRadius: '4px',
                  background: m.available ? 'rgba(63,185,80,0.15)' : 'rgba(248,81,73,0.15)',
                  color: m.available ? '#3fb950' : '#f85149',
                }}>{m.available ? '可用' : '未安裝'}</span>
              </div>
              <div style={{ color: 'var(--text-tertiary)', fontSize: '11px', marginTop: '2px' }}>
                {m.category} · {m.training_speed}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}


/* ═══════════════════  訓練  ═══════════════════ */

function TrainTab() {
  const symbol = useChartStore(s => s.symbol);
  const timeframe = useChartStore(s => s.timeframe);

  const [models, setModels] = useState<MLModel[]>([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [training, setTraining] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchMLModels().then(m => {
      const avail = m.filter(x => x.available);
      setModels(avail);
      if (avail.length > 0 && !selectedModel) setSelectedModel(avail[0].id);
    }).catch(() => setError('無法載入模型列表'));
  }, []);

  const handleTrain = async () => {
    setTraining(true); setResult(null); setError('');
    try {
      const res = await trainMLModel({ symbol, timeframe, model_id: selectedModel });
      setResult(res);
      if (res.status === 'cooldown') {
        setError(res.message);
      }
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || '訓練失敗');
    } finally {
      setTraining(false);
    }
  };

  return (
    <div>
      <p style={{ color: 'var(--text-secondary)', fontSize: '12px', marginBottom: '12px' }}>
        使用當前交易對 <strong style={{ color: 'var(--text-primary)' }}>{symbol}</strong> {timeframe} 的歷史數據訓練 ML 模型。
      </p>

      {/* 模型選擇 */}
      <div style={{ marginBottom: '12px' }}>
        <label style={S.label}>選擇訓練模型</label>
        <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)} style={S.input}>
          {models.map(m => <option key={m.id} value={m.id}>{m.name} ({m.category}) — {m.training_speed}</option>)}
        </select>
      </div>

      {/* 訓練按鈕 */}
      <button onClick={handleTrain} disabled={training || !selectedModel}
        style={{
          width: '100%', padding: '10px', borderRadius: '8px', fontSize: '14px', fontWeight: 600,
          cursor: training ? 'wait' : 'pointer',
          border: 'none',
          background: training ? 'var(--bg-tertiary)' : 'var(--accent-blue)',
          color: training ? 'var(--text-secondary)' : '#fff',
          opacity: !selectedModel ? 0.4 : 1,
        }}>
        {training ? '訓練中...' : '開始訓練'}
      </button>

      {training && (
        <div style={{ marginTop: '10px', textAlign: 'center' }}>
          <div style={{ color: 'var(--text-secondary)', fontSize: '12px' }}>
            模型訓練中，樹模型通常 5-30 秒，神經網路可能需要數分鐘...
          </div>
          <div style={{
            width: '100%', height: '4px', borderRadius: '2px', marginTop: '8px',
            background: 'var(--border-color)', overflow: 'hidden',
          }}>
            <div style={{
              width: '40%', height: '100%', borderRadius: '2px',
              background: 'var(--accent-blue)',
              animation: 'mlprogress 1.5s ease-in-out infinite',
            }} />
          </div>
          <style>{`@keyframes mlprogress { 0% { margin-left: 0 } 50% { margin-left: 60% } 100% { margin-left: 0 } }`}</style>
        </div>
      )}

      {error && <div style={{ ...S.err, marginTop: '10px' }}>{error}</div>}

      {/* 訓練結果 */}
      {result && result.status === 'success' && <TrainResultCard result={result} />}
    </div>
  );
}


/* ─── 訓練結果卡片 ─────────────────────── */

function TrainResultCard({ result }: { result: any }) {
  const tr = result.train_result || {};
  const diag = result.diagnostics || {};
  const warns: string[] = result.warnings || [];
  const ls = result.label_stats || {};

  return (
    <div style={{ marginTop: '12px' }}>
      {/* 警告 */}
      {/* 標籤統計 */}
      {ls.total_samples > 0 && (
        <div style={{ ...S.card, marginBottom: '8px', fontSize: '12px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
            <span style={{ color: 'var(--text-secondary)' }}>
              目標：{ls.target_direction === 'down' ? '下跌' : '上漲'} {((ls.target_threshold || 0) * 100).toFixed(1)}%
            </span>
            <span style={{
              color: ls.imbalance_warning ? '#e3a008' : '#3fb950',
              fontWeight: 500,
            }}>
              正樣本 {ls.positive_count}/{ls.total_samples} ({((ls.positive_ratio || 0) * 100).toFixed(1)}%)
            </span>
          </div>
          {ls.suggested_threshold > 0 && (
            <div style={{ color: 'var(--text-tertiary)', fontSize: '11px' }}>
              建議門檻 ≤ {((ls.suggested_threshold || 0) * 100).toFixed(1)}%（歷史報酬率 P75）
            </div>
          )}
        </div>
      )}

      {warns.length > 0 && (
        <div style={S.warn}>
          <div style={{ fontWeight: 600, marginBottom: '4px' }}>風險警告</div>
          {warns.map((w, i) => <div key={i} style={{ marginTop: '2px' }}>- {w}</div>)}
        </div>
      )}

      <div style={S.card}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
          <span style={{ color: 'var(--text-primary)', fontWeight: 600, fontSize: '13px' }}>
            {result.model_id}
          </span>
          <span style={{
            fontSize: '10px', padding: '2px 8px', borderRadius: '4px',
            background: tr.is_reliable ? 'rgba(63,185,80,0.15)' : 'rgba(248,81,73,0.15)',
            color: tr.is_reliable ? '#3fb950' : '#f85149',
          }}>
            {tr.is_reliable ? '可靠' : '不可靠'}
          </span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px' }}>
          <MetricBox label="OOS 準確率" value={`${((tr.oos_accuracy || 0) * 100).toFixed(1)}%`}
            color={tr.oos_accuracy > 0.55 ? '#3fb950' : tr.oos_accuracy > 0.5 ? '#e3a008' : '#f85149'} />
          <MetricBox label="OOS AUC" value={(tr.oos_auc || 0).toFixed(3)}
            color={tr.oos_auc > 0.6 ? '#3fb950' : tr.oos_auc > 0.5 ? '#e3a008' : '#f85149'} />
          <MetricBox label="OOS F1" value={(tr.oos_f1 || 0).toFixed(3)} color="var(--text-primary)" />
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px', marginTop: '6px' }}>
          <MetricBox label="訓練準確率" value={`${((tr.train_accuracy || 0) * 100).toFixed(1)}%`} color="var(--text-secondary)" />
          <MetricBox label="過擬合落差" value={`${((diag.overfit_gap || 0) * 100).toFixed(1)}%`}
            color={diag.overfit_gap > 0.25 ? '#f85149' : diag.overfit_gap > 0.15 ? '#e3a008' : '#3fb950'} />
          <MetricBox label="樣本/特徵" value={`${diag.sample_feature_ratio || '-'}:1`}
            color={diag.sample_feature_ratio < 15 ? '#e3a008' : '#3fb950'} />
        </div>

        {/* 白話解讀 */}
        <div style={{
          marginTop: '8px', padding: '6px 10px', borderRadius: 6, fontSize: 11, lineHeight: 1.6,
          background: 'rgba(88,166,255,0.06)', borderLeft: '2px solid rgba(88,166,255,0.3)',
          color: 'var(--text-secondary)',
        }}>
          {tr.oos_accuracy >= 0.58
            ? '模型表現良好，預測結果具參考價值'
            : tr.oos_accuracy >= 0.52
              ? '模型有一定預測能力，建議搭配其他指標綜合判斷'
              : '模型效果不佳，建議增加歷史數據量或嘗試其他模型'}
          {diag.overfit_gap > 0.25
            ? '。過擬合嚴重 — 模型記住了訓練數據的雜訊，實戰效果會打折扣'
            : diag.overfit_gap > 0.15
              ? '。有輕微過擬合，可增加數據量或降低特徵集來改善'
              : '。泛化能力良好，訓練表現與實測接近'}
        </div>

        <div style={{ marginTop: '6px', fontSize: '11px', color: 'var(--text-tertiary)' }}>
          訓練樣本: {tr.n_train_samples} · 測試樣本: {tr.n_oos_samples} · 耗時: {tr.train_time_ms}ms
        </div>

        {/* 特徵重要性 TOP 5 */}
        {tr.feature_importance && Object.keys(tr.feature_importance).length > 0 && (
          <div style={{ marginTop: '10px' }}>
            <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '4px' }}>Top 特徵</div>
            {Object.entries(tr.feature_importance as Record<string, number>)
              .sort(([, a], [, b]) => b - a)
              .slice(0, 5)
              .map(([name, val]) => (
                <div key={name} style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '3px' }}>
                  <div style={{
                    height: '6px', borderRadius: '3px', background: 'var(--accent-blue)',
                    width: `${Math.min(val * 100 * 3, 100)}%`, minWidth: '4px',
                  }} />
                  <span style={{ fontSize: '11px', color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}>
                    {name} ({(val * 100).toFixed(1)}%)
                  </span>
                </div>
              ))}
          </div>
        )}
      </div>
    </div>
  );
}


/* ═══════════════════  多模型比較  ═══════════════════ */

function CompareTab() {
  const symbol = useChartStore(s => s.symbol);
  const timeframe = useChartStore(s => s.timeframe);

  const [models, setModels] = useState<MLModel[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [comparing, setComparing] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchMLModels().then(m => {
      const avail = m.filter(x => x.available);
      setModels(avail);
      setSelected(new Set(avail.map(x => x.id)));
    }).catch(() => {});
  }, []);

  const toggle = (id: string) => {
    setSelected(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const handleCompare = async () => {
    if (selected.size < 2) { setError('至少選擇 2 個模型'); return; }
    setComparing(true); setResult(null); setError('');
    try {
      const res = await compareMLModels({
        symbol, timeframe, model_ids: [...selected],
      });
      setResult(res);
    } catch (e: any) {
      setError(e?.response?.data?.detail || '比較失敗');
    } finally {
      setComparing(false);
    }
  };

  return (
    <div>
      <p style={{ color: 'var(--text-secondary)', fontSize: '12px', marginBottom: '12px' }}>
        使用 <strong style={{ color: 'var(--text-primary)' }}>{symbol}</strong> {timeframe} 同時訓練多個模型，比較表現並推薦最佳模型。
      </p>

      {/* 模型勾選 */}
      <div style={{ marginBottom: '12px' }}>
        <label style={S.label}>選擇參賽模型（至少 2 個）</label>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          {models.map(m => (
            <label key={m.id} style={{
              display: 'flex', alignItems: 'center', gap: '8px', padding: '6px 10px',
              borderRadius: '6px', cursor: 'pointer', fontSize: '12px',
              background: selected.has(m.id) ? 'rgba(88,166,255,0.08)' : 'transparent',
              border: selected.has(m.id) ? '1px solid rgba(88,166,255,0.3)' : '1px solid var(--border-color)',
              color: 'var(--text-primary)',
            }}>
              <input type="checkbox" checked={selected.has(m.id)} onChange={() => toggle(m.id)}
                style={{ accentColor: 'var(--accent-blue)' }} />
              <span>{m.name}</span>
              <span style={{ color: 'var(--text-tertiary)', marginLeft: 'auto' }}>{m.training_speed}</span>
            </label>
          ))}
        </div>
      </div>

      <button onClick={handleCompare} disabled={comparing || selected.size < 2}
        style={{
          width: '100%', padding: '10px', borderRadius: '8px', fontSize: '14px', fontWeight: 600,
          cursor: comparing ? 'wait' : 'pointer', border: 'none',
          background: comparing ? 'var(--bg-tertiary)' : '#7c3aed',
          color: comparing ? 'var(--text-secondary)' : '#fff',
          opacity: selected.size < 2 ? 0.4 : 1,
        }}>
        {comparing ? '比較中...' : `比較 ${selected.size} 個模型`}
      </button>

      {comparing && (
        <div style={{ marginTop: '10px', textAlign: 'center' }}>
          <div style={{ color: 'var(--text-secondary)', fontSize: '12px' }}>
            依據模型數量，可能需要 30 秒至數分鐘...
          </div>
          <div style={{
            width: '100%', height: '4px', borderRadius: '2px', marginTop: '8px',
            background: 'var(--border-color)', overflow: 'hidden',
          }}>
            <div style={{
              width: '30%', height: '100%', borderRadius: '2px',
              background: '#7c3aed',
              animation: 'mlprogress 2s ease-in-out infinite',
            }} />
          </div>
        </div>
      )}

      {error && <div style={{ ...S.err, marginTop: '10px' }}>{error}</div>}

      {/* 比較結果 */}
      {result && result.status === 'success' && <CompareResultCard result={result} symbol={symbol} />}
    </div>
  );
}


function CompareResultCard({ result, symbol }: { result: any; symbol: string }) {
  const ranking: any[] = result.ranking || [];
  const consensus: any[] = result.feature_consensus || [];
  const rec = result.recommendation || {};
  const [setDefault, setSetDefault] = useState(false);
  const [defaultMsg, setDefaultMsg] = useState('');

  const handleSetAsDefault = async () => {
    if (!rec.model_id) return;
    setSetDefault(true);
    try {
      await updateMLSettings({ model_id: rec.model_id, symbol });
      setDefaultMsg(`已將 ${rec.model_name} 設為 ${symbol} 預設模型`);
    } catch {
      setDefaultMsg('設定失敗');
    } finally {
      setSetDefault(false);
    }
  };

  return (
    <div style={{ marginTop: '12px' }}>
      {/* 推薦 */}
      {rec.model_name && (
        <div style={{ ...S.ok, fontWeight: 500, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span>推薦：{rec.reason}</span>
          {!defaultMsg && (
            <button
              onClick={handleSetAsDefault}
              disabled={setDefault}
              style={{
                padding: '3px 10px', borderRadius: 4, fontSize: 11, cursor: 'pointer',
                background: '#3fb950', color: '#fff', border: 'none', whiteSpace: 'nowrap',
              }}
            >
              {setDefault ? '設定中...' : `設為 ${symbol} 預設`}
            </button>
          )}
          {defaultMsg && <span style={{ fontSize: 11 }}>{defaultMsg}</span>}
        </div>
      )}

      {/* 排名表 */}
      <div style={{ ...S.card, padding: '0', overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px' }}>
          <thead>
            <tr style={{ background: 'var(--bg-tertiary)' }}>
              <th style={thStyle}>#</th>
              <th style={thStyle}>模型</th>
              <th style={thStyle}>OOS Acc</th>
              <th style={thStyle}>AUC</th>
              <th style={thStyle}>過擬合</th>
              <th style={thStyle}>狀態</th>
            </tr>
          </thead>
          <tbody>
            {ranking.map((r: any, i: number) => {
              const m = r.metrics || {};
              const diag = r.diagnostics || {};
              const warns: string[] = r.warnings || [];
              return (
                <tr key={r.model_id} style={{ borderTop: '1px solid var(--border-color)' }}>
                  <td style={tdStyle}>{r.status === 'success' ? i + 1 : '-'}</td>
                  <td style={{ ...tdStyle, fontWeight: 500, color: 'var(--text-primary)' }}>{r.model_name}</td>
                  <td style={tdStyle}>
                    {r.status === 'success' ? `${((m.oos_accuracy || 0) * 100).toFixed(1)}%` : '-'}
                  </td>
                  <td style={tdStyle}>
                    {r.status === 'success' ? (m.oos_auc || 0).toFixed(3) : '-'}
                  </td>
                  <td style={tdStyle}>
                    {r.status === 'success' ? (
                      <span style={{
                        color: diag.overfit_gap > 0.25 ? '#f85149' : diag.overfit_gap > 0.15 ? '#e3a008' : '#3fb950',
                      }}>
                        {((diag.overfit_gap || 0) * 100).toFixed(1)}%
                      </span>
                    ) : '-'}
                  </td>
                  <td style={tdStyle}>
                    {r.status === 'success' ? (
                      warns.length > 0
                        ? <span style={{ color: '#e3a008' }} title={warns.join('\n')}>&#9888;</span>
                        : <span style={{ color: '#3fb950' }}>&#10003;</span>
                    ) : (
                      <span style={{ color: '#f85149' }}>{r.reason?.slice(0, 20) || r.status}</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* 特徵共識 */}
      {consensus.length > 0 && (
        <div style={{ marginTop: '10px' }}>
          <label style={{ ...S.label, fontWeight: 600, color: 'var(--text-primary)' }}>特徵共識排名 (Top 10)</label>
          {consensus.slice(0, 10).map((c: any) => (
            <div key={c.feature} style={{
              display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '3px',
            }}>
              <div style={{
                height: '6px', borderRadius: '3px',
                width: `${Math.min(c.avg_importance * 100 * 3, 100)}%`, minWidth: '4px',
                background: c.agreement_ratio >= 0.8 ? '#3fb950' : c.agreement_ratio >= 0.5 ? '#e3a008' : 'var(--accent-blue)',
              }} />
              <span style={{ fontSize: '11px', color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}>
                {c.feature} ({(c.avg_importance * 100).toFixed(1)}%) {c.reliability || ''}
              </span>
            </div>
          ))}
        </div>
      )}

      <div style={{ marginTop: '8px', fontSize: '11px', color: 'var(--text-tertiary)' }}>
        總樣本: {result.total_samples} · 訓練: {result.train_samples} · 驗證: {result.val_samples}
      </div>
    </div>
  );
}


/* ═══════════════════  狀態/預測  ═══════════════════ */

function StatusTab() {
  const symbol = useChartStore(s => s.symbol);
  const timeframe = useChartStore(s => s.timeframe);

  const [trainedModels, setTrainedModels] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [prediction, setPrediction] = useState<any>(null);
  const [error, setError] = useState('');

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetchTrainedModels(symbol);
      setTrainedModels(res.models || []);
    } catch { setError('無法載入已訓練模型'); }
    finally { setLoading(false); }
  }, [symbol]);

  useEffect(() => { loadData(); }, [loadData]);

  const handlePredict = async (modelId?: string, consensus?: boolean) => {
    setPredicting(true); setPrediction(null); setError('');
    try {
      const res = await predictML({ symbol, timeframe, model_id: modelId, consensus });
      setPrediction(res);
    } catch (e: any) {
      setError(e?.response?.data?.detail || '預測失敗');
    } finally {
      setPredicting(false);
    }
  };

  if (loading) return <div style={{ color: 'var(--text-secondary)', padding: '16px 0', textAlign: 'center' }}>載入中...</div>;

  return (
    <div>
      <p style={{ color: 'var(--text-secondary)', fontSize: '12px', marginBottom: '12px' }}>
        <strong style={{ color: 'var(--text-primary)' }}>{symbol}</strong> 已訓練模型和即時預測。
      </p>

      {trainedModels.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '20px 0', color: 'var(--text-secondary)', fontSize: '13px' }}>
          尚未為此交易對訓練任何模型。
          <br />
          <span style={{ fontSize: '12px' }}>請至「訓練模型」頁籤開始訓練。</span>
        </div>
      ) : (
        <>
          {/* 多模型共識預測 */}
          {new Set(trainedModels.map((m: any) => m.model_id)).size >= 2 && (
            <button onClick={() => handlePredict(undefined, true)} disabled={predicting}
              style={{
                width: '100%', padding: '8px 12px', borderRadius: '6px', fontSize: '13px',
                cursor: 'pointer', marginBottom: '10px', fontWeight: 600,
                border: '1px solid var(--accent-blue)',
                background: 'rgba(88,166,255,0.1)', color: 'var(--accent-blue)',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px',
              }}>
              {predicting ? '預測中...' : `多模型共識預測 (${new Set(trainedModels.map((m: any) => m.model_id)).size} 個模型)`}
            </button>
          )}
          {trainedModels.map((m: any) => (
            <div key={m.id} style={{ ...S.card, marginBottom: '8px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <span style={{ color: 'var(--text-primary)', fontWeight: 600, fontSize: '13px' }}>
                    {m.model_id}
                  </span>
                  <span style={{ color: 'var(--text-tertiary)', fontSize: '11px', marginLeft: '8px' }}>
                    {m.timeframe}
                  </span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span style={{
                    fontSize: '10px', padding: '1px 6px', borderRadius: '4px',
                    background: m.is_reliable ? 'rgba(63,185,80,0.15)' : 'rgba(248,81,73,0.15)',
                    color: m.is_reliable ? '#3fb950' : '#f85149',
                  }}>
                    {m.is_reliable ? '可靠' : '不可靠'}
                  </span>
                  <button onClick={() => handlePredict(m.model_id)} disabled={predicting}
                    style={{
                      padding: '3px 10px', borderRadius: '4px', fontSize: '11px',
                      cursor: 'pointer', border: '1px solid var(--accent-blue)',
                      background: 'rgba(88,166,255,0.1)', color: 'var(--accent-blue)',
                    }}>
                    {predicting ? '...' : '預測'}
                  </button>
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '6px', marginTop: '6px' }}>
                <MiniMetric label="OOS Acc" value={`${((m.oos_accuracy || 0) * 100).toFixed(1)}%`} />
                <MiniMetric label="AUC" value={(m.oos_auc || 0).toFixed(3)} />
                <MiniMetric label="特徵集" value={m.feature_set} />
                <MiniMetric label="訓練時間" value={m.created_at?.slice(5, 16) || '-'} />
              </div>
            </div>
          ))}
        </>
      )}

      {error && <div style={{ ...S.err, marginTop: '10px' }}>{error}</div>}

      {/* 預測結果 */}
      {prediction && prediction.status === 'success' && (
        <PredictionResultCard prediction={prediction} />
      )}
    </div>
  );
}


function PredictionResultCard({ prediction }: { prediction: any }) {
  const p = prediction.prediction || {};
  const q = p.model_quality || {};
  const topFeatures: any[] = Array.isArray(p.top_features) ? p.top_features : [];
  const consensusMode = p.consensus_mode as string | undefined;
  const modelsConsidered = p.models_considered as number | undefined;
  const voteDetail = p.vote_detail as Record<string, number> | undefined;
  const individualModels = p.individual_models as any[] | undefined;

  const dirColor = p.direction === 'long' ? '#3fb950' : p.direction === 'short' ? '#f85149' : '#e3a008';
  const confLabel = p.confidence === 'high' ? '高' : p.confidence === 'medium' ? '中' : '低';
  const modeLabel: Record<string, string> = { best: '最佳單模型', vote: '多數投票', weighted_avg: '加權平均' };

  return (
    <div style={{ ...S.card, marginTop: '12px', border: `1px solid ${dirColor}33` }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
        <div>
          <span style={{ fontSize: '20px', fontWeight: 700, color: dirColor }}>
            {p.direction === 'long' ? 'LONG' : p.direction === 'short' ? 'SHORT' : 'NEUTRAL'}
          </span>
          <span style={{ fontSize: '12px', color: 'var(--text-secondary)', marginLeft: '8px' }}>
            信心: {confLabel}
          </span>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: '22px', fontWeight: 700, color: dirColor }}>
            {((p.probability || 0) * 100).toFixed(1)}%
          </div>
          <div style={{ fontSize: '10px', color: 'var(--text-tertiary)' }}>上漲機率</div>
        </div>
      </div>

      {/* 共識模式資訊 */}
      {consensusMode && modelsConsidered && modelsConsidered > 1 && (
        <div style={{
          padding: '6px 10px', borderRadius: '6px', marginBottom: '8px',
          background: 'rgba(88,166,255,0.08)', border: '1px solid rgba(88,166,255,0.2)',
          fontSize: '11px', color: 'var(--accent-blue)',
        }}>
          <div style={{ fontWeight: 600, marginBottom: '4px' }}>
            共識模式：{modeLabel[consensusMode] || consensusMode}（{modelsConsidered} 個模型）
          </div>
          {voteDetail && (
            <div style={{ color: 'var(--text-secondary)' }}>
              投票：做多 {voteDetail.long ?? 0} · 做空 {voteDetail.short ?? 0} · 中立 {voteDetail.neutral ?? 0}
            </div>
          )}
          {individualModels && individualModels.length > 0 && (
            <div style={{ marginTop: '4px' }}>
              {individualModels.map((im: any, i: number) => {
                const c = im.direction === 'long' ? '#3fb950' : im.direction === 'short' ? '#f85149' : '#e3a008';
                return (
                  <div key={i} style={{ display: 'flex', justifyContent: 'space-between', marginTop: '2px' }}>
                    <span style={{ color: 'var(--text-primary)' }}>{im.model_id}</span>
                    <span style={{ color: c, fontWeight: 500 }}>
                      {im.direction === 'long' ? 'LONG' : im.direction === 'short' ? 'SHORT' : 'NEUTRAL'}{' '}
                      {((im.probability || 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* 單模型時顯示模型名 */}
      {p.model_id && (!consensusMode || modelsConsidered === 1) && (
        <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '6px' }}>
          模型：{p.model_id}
        </div>
      )}

      {!q.is_reliable && q.oos_accuracy !== undefined && (
        <div style={{ ...S.warn, marginBottom: '8px', fontSize: '11px' }}>
          模型品質不足（OOS Acc {((q.oos_accuracy || 0) * 100).toFixed(1)}%），結果僅供參考。
        </div>
      )}

      {/* 驅動因子 */}
      {topFeatures.length > 0 && (
        <div>
          <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '4px' }}>主要驅動因子</div>
          {topFeatures.slice(0, 5).map((f: any, i: number) => (
            <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', marginBottom: '2px' }}>
              <span style={{ color: 'var(--text-primary)' }}>{f.name || f[0]}</span>
              <span style={{ color: 'var(--text-secondary)' }}>
                {typeof f.importance === 'number' ? f.importance.toFixed(4) : (typeof f.value === 'number' ? f.value.toFixed(4) : f[1])}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}


/* ─── 小型通用組件 ─────────────────────── */

function MetricBox({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{
      padding: '6px 8px', borderRadius: '6px',
      background: 'var(--bg-tertiary)', textAlign: 'center',
    }}>
      <div style={{ fontSize: '10px', color: 'var(--text-tertiary)' }}>{label}</div>
      <div style={{ fontSize: '14px', fontWeight: 600, color }}>{value}</div>
    </div>
  );
}

function MiniMetric({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: '9px', color: 'var(--text-tertiary)' }}>{label}</div>
      <div style={{ fontSize: '12px', color: 'var(--text-primary)', fontWeight: 500 }}>{value}</div>
    </div>
  );
}

const thStyle: React.CSSProperties = {
  padding: '6px 8px', textAlign: 'left', fontSize: '10px',
  color: 'var(--text-secondary)', fontWeight: 600,
};

const tdStyle: React.CSSProperties = {
  padding: '6px 8px', color: 'var(--text-secondary)', fontSize: '11px',
};


// ─── 監控 Tab ────────────────────

interface HistoryPoint {
  model_id: string;
  oos_accuracy: number | null;
  oos_auc: number | null;
  is_reliable: number;
  created_at: string;
}

function MonitorTab() {
  const symbol = useChartStore((s) => s.symbol);
  const timeframe = useChartStore((s) => s.timeframe);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [health, setHealth] = useState<any>(null);
  const [accuracy, setAccuracy] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [histRes, healthRes, accRes] = await Promise.all([
        fetchMLPerformanceHistory(symbol || undefined),
        symbol ? checkMLHealth(symbol, timeframe || '4h') : Promise.resolve({ status: 'no_model' }),
        fetchMLPredictionAccuracy(symbol || undefined),
      ]);
      setHistory(histRes.history || []);
      setHealth(healthRes);
      setAccuracy(accRes);
    } catch { /* ignore */ } finally {
      setLoading(false);
    }
  }, [symbol, timeframe]);

  useEffect(() => { load(); }, [load]);

  if (loading) {
    return <div style={{ textAlign: 'center', padding: '24px', color: 'var(--text-secondary)', fontSize: 12 }}>載入中...</div>;
  }

  const threshold = 0.55;
  const accData = history.filter((h) => h.oos_accuracy != null).map((h) => h.oos_accuracy as number);

  return (
    <div style={{ fontSize: 12 }}>
      {/* 健康度徽章 */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12,
        padding: '8px 12px', borderRadius: 8, background: 'var(--bg-tertiary)',
      }}>
        <span style={{
          width: 10, height: 10, borderRadius: '50%', flexShrink: 0,
          background: !health || health.status === 'no_model' ? '#888'
            : health.needs_retrain ? '#f87171' : '#4ade80',
        }} />
        <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
          {!health || health.status === 'no_model'
            ? '尚無模型'
            : health.needs_retrain
              ? `模型品質下降 — 建議重新訓練 (OOS: ${((health.oos_accuracy || 0) * 100).toFixed(1)}%)`
              : `模型健康 (OOS: ${((health.oos_accuracy || 0) * 100).toFixed(1)}%)`}
        </span>
      </div>

      {/* OOS 準確率趨勢圖 */}
      {accData.length > 1 ? (
        <div style={{ marginBottom: 12 }}>
          <div style={{ color: 'var(--text-secondary)', marginBottom: 6 }}>OOS 準確率趨勢</div>
          <AccuracyChart data={accData} threshold={threshold} />
        </div>
      ) : (
        <div style={{ textAlign: 'center', padding: 16, color: 'var(--text-secondary)' }}>
          需要至少 2 次訓練記錄才能顯示趨勢圖
        </div>
      )}

      {/* 預測準確率 */}
      {accuracy && accuracy.validated > 0 && (
        <div style={{
          marginBottom: 12, padding: '10px 12px', borderRadius: 8,
          background: 'var(--bg-tertiary)',
        }}>
          <div style={{ color: 'var(--text-secondary)', marginBottom: 8, fontWeight: 600 }}>
            預測準確率
          </div>
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 8 }}>
            <div>
              <span style={{ color: 'var(--text-secondary)' }}>方向正確率：</span>
              <span style={{
                fontWeight: 700,
                color: accuracy.direction_accuracy >= 0.55 ? '#4ade80'
                  : accuracy.direction_accuracy >= 0.50 ? '#facc15' : '#f87171',
              }}>
                {(accuracy.direction_accuracy * 100).toFixed(1)}%
              </span>
              {accuracy.confidence_interval && (
                <span style={{ color: 'var(--text-tertiary)', fontSize: 10, marginLeft: 4 }}>
                  (95% CI: {(accuracy.confidence_interval[0] * 100).toFixed(0)}-{(accuracy.confidence_interval[1] * 100).toFixed(0)}%)
                </span>
              )}
            </div>
            <div>
              <span style={{ color: 'var(--text-secondary)' }}>已驗證：</span>
              <span style={{ color: 'var(--text-primary)' }}>
                {accuracy.validated} / {accuracy.total_predictions} 筆
              </span>
            </div>
            <div>
              <span style={{ color: 'var(--text-secondary)' }}>平均漲跌：</span>
              <span style={{
                color: accuracy.avg_outcome_pct >= 0 ? '#4ade80' : '#f87171',
              }}>
                {accuracy.avg_outcome_pct >= 0 ? '+' : ''}{accuracy.avg_outcome_pct.toFixed(2)}%
              </span>
            </div>
          </div>
          {/* MFE / MAE */}
          {(accuracy.avg_mfe_pct != null || accuracy.avg_mae_pct != null) && (
            <div style={{ display: 'flex', gap: 16, marginBottom: 8 }}>
              {accuracy.avg_mfe_pct != null && (
                <div>
                  <span style={{ color: 'var(--text-secondary)' }}>平均 MFE：</span>
                  <span style={{ color: '#4ade80' }}>+{accuracy.avg_mfe_pct.toFixed(2)}%</span>
                </div>
              )}
              {accuracy.avg_mae_pct != null && (
                <div>
                  <span style={{ color: 'var(--text-secondary)' }}>平均 MAE：</span>
                  <span style={{ color: '#f87171' }}>{accuracy.avg_mae_pct.toFixed(2)}%</span>
                </div>
              )}
            </div>
          )}
          {/* 按模型統計 */}
          {accuracy.by_model && accuracy.by_model.length > 1 && (
            <div style={{ marginTop: 4 }}>
              <div style={{ color: 'var(--text-secondary)', marginBottom: 4, fontSize: 11 }}>按模型統計</div>
              {accuracy.by_model.map((m: any) => (
                <div key={m.model_id} style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 2 }}>
                  <span style={{ color: 'var(--text-primary)', minWidth: 80 }}>{m.model_id}</span>
                  <span style={{
                    fontWeight: 600,
                    color: m.accuracy >= 0.55 ? '#4ade80' : m.accuracy >= 0.50 ? '#facc15' : '#f87171',
                  }}>
                    {(m.accuracy * 100).toFixed(1)}%
                  </span>
                  <span style={{ color: 'var(--text-tertiary)', fontSize: 10 }}>
                    ({m.correct}/{m.total},
                    CI: {(m.confidence_interval[0] * 100).toFixed(0)}-{(m.confidence_interval[1] * 100).toFixed(0)}%)
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      {accuracy && accuracy.validated === 0 && accuracy.total_predictions > 0 && (
        <div style={{
          marginBottom: 12, padding: '10px 12px', borderRadius: 8,
          background: 'var(--bg-tertiary)', color: 'var(--text-secondary)',
        }}>
          已有 {accuracy.total_predictions} 筆預測，同步新資料後將自動驗證準確率
        </div>
      )}

      {/* 版本歷史表 */}
      {history.length > 0 && (
        <div>
          <div style={{ color: 'var(--text-secondary)', marginBottom: 6 }}>訓練記錄</div>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={thStyle}>模型</th>
                <th style={{ ...thStyle, textAlign: 'right' }}>OOS Acc</th>
                <th style={{ ...thStyle, textAlign: 'right' }}>AUC</th>
                <th style={{ ...thStyle, textAlign: 'center' }}>可靠</th>
                <th style={thStyle}>日期</th>
              </tr>
            </thead>
            <tbody>
              {[...history].reverse().map((h, i) => (
                <tr key={i} style={{ borderTop: '1px solid var(--border-primary)' }}>
                  <td style={tdStyle}>{h.model_id}</td>
                  <td style={{
                    ...tdStyle, textAlign: 'right',
                    color: (h.oos_accuracy || 0) >= threshold ? '#4ade80' : '#f87171',
                  }}>
                    {h.oos_accuracy != null ? `${(h.oos_accuracy * 100).toFixed(1)}%` : 'N/A'}
                  </td>
                  <td style={{ ...tdStyle, textAlign: 'right' }}>
                    {h.oos_auc != null ? h.oos_auc.toFixed(3) : 'N/A'}
                  </td>
                  <td style={{ ...tdStyle, textAlign: 'center' }}>
                    {h.is_reliable ? 'V' : 'X'}
                  </td>
                  <td style={tdStyle}>{new Date(h.created_at).toLocaleDateString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div style={{ textAlign: 'center', marginTop: 8 }}>
        <button onClick={load} style={{ color: 'var(--accent-blue)', background: 'transparent', border: 'none', cursor: 'pointer', fontSize: 11 }}>
          重新載入
        </button>
      </div>
    </div>
  );
}


/** 簡易 SVG 準確率折線圖 */
function AccuracyChart({ data, threshold }: { data: number[]; threshold: number }) {
  const w = 480, h = 120, pad = 24;
  const minY = Math.min(...data, threshold) - 0.05;
  const maxY = Math.max(...data, threshold) + 0.05;
  const scaleX = (i: number) => pad + (i / (data.length - 1)) * (w - pad * 2);
  const scaleY = (v: number) => h - pad - ((v - minY) / (maxY - minY)) * (h - pad * 2);

  const points = data.map((v, i) => `${scaleX(i)},${scaleY(v)}`).join(' ');
  const thresholdY = scaleY(threshold);

  return (
    <svg width="100%" viewBox={`0 0 ${w} ${h}`} style={{ background: 'var(--bg-tertiary)', borderRadius: 6 }}>
      {/* 閾值線 */}
      <line x1={pad} y1={thresholdY} x2={w - pad} y2={thresholdY}
        stroke="#f87171" strokeWidth={1} strokeDasharray="4,4" opacity={0.6} />
      <text x={w - pad + 4} y={thresholdY + 3} fontSize={9} fill="#f87171">{(threshold * 100).toFixed(0)}%</text>

      {/* 折線 */}
      <polyline points={points} fill="none" stroke="var(--accent-blue)" strokeWidth={2} />

      {/* 圓點 */}
      {data.map((v, i) => (
        <circle key={i} cx={scaleX(i)} cy={scaleY(v)} r={3}
          fill={v >= threshold ? '#4ade80' : '#f87171'} />
      ))}
    </svg>
  );
}
