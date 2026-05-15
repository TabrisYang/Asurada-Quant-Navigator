/** 阿斯拉量化系統 — v132 編排管線 2 週評估結果 banner。
 *
 * 掛在 App.tsx 頂部。讀後端 /api/system/v132_eval_status：
 *   pending   (今日<5/29 + 無 marker)  → 不渲染
 *   scheduled (今日≥5/29 + marker 未生成) → 細灰 banner
 *   passed    (marker + 無劣化)        → 綠 banner + ×關閉（localStorage 24h 不再顯示）
 *   degraded  (marker + 偵測到劣化)     → 紅 banner +「立即回滾」按鈕
 *   error     → 紅底錯誤訊息
 */

import { useState, useEffect, useCallback } from 'react';
import {
  fetchV132EvalStatus, triggerV132Rollback,
  type V132EvalStatusResponse,
} from '../services/api';
import { toast } from './Toast';

const POLL_INTERVAL_MS = 5 * 60 * 1000;  // 5 分鐘
const DISMISSED_KEY = 'asura_v132_eval_passed_dismissed_at';
const DISMISS_TTL_MS = 24 * 60 * 60 * 1000;  // passed 狀態 24h 內不再顯示

const STYLE = {
  scheduled: { bg: 'rgba(139,148,158,0.12)', border: '#8b949e', text: '#c9d1d9', icon: '⏳' },
  passed:    { bg: 'rgba(63,185,80,0.12)',   border: '#3fb950', text: '#aff5b4', icon: '✅' },
  degraded:  { bg: 'rgba(248,81,73,0.12)',   border: '#f85149', text: '#ffdcd7', icon: '⚠️' },
  error:     { bg: 'rgba(210,153,34,0.12)',  border: '#d29922', text: '#ffe39e', icon: 'ℹ️' },
};

export default function V132EvalBanner() {
  const [data, setData] = useState<V132EvalStatusResponse | null>(null);
  const [showDetails, setShowDetails] = useState(false);
  const [rollbackInFlight, setRollbackInFlight] = useState(false);
  const [confirmRollback, setConfirmRollback] = useState(false);
  const [dismissed, setDismissed] = useState(false);

  const refresh = useCallback(async () => {
    const result = await fetchV132EvalStatus();
    setData(result);
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, POLL_INTERVAL_MS);
    return () => clearInterval(t);
  }, [refresh]);

  // passed 狀態：localStorage 記錄 24h 不再顯示
  useEffect(() => {
    if (data?.status !== 'passed') {
      setDismissed(false);
      return;
    }
    const dismissedAt = Number(localStorage.getItem(DISMISSED_KEY) || 0);
    setDismissed(Date.now() - dismissedAt < DISMISS_TTL_MS);
  }, [data?.status]);

  const handleDismiss = () => {
    localStorage.setItem(DISMISSED_KEY, String(Date.now()));
    setDismissed(true);
  };

  const handleRollback = async () => {
    setRollbackInFlight(true);
    try {
      const r = await triggerV132Rollback();
      if (r.ok) {
        toast(r.message, r.needs_restart ? 'warning' : 'info', 8000);
      } else {
        toast(r.message || '回滾失敗', 'error', 6000);
      }
    } catch (err) {
      toast(`回滾失敗：${(err as Error).message}`, 'error', 6000);
    } finally {
      setRollbackInFlight(false);
      setConfirmRollback(false);
      refresh();
    }
  };

  if (!data) return null;
  if (data.status === 'pending') return null;
  if (data.status === 'passed' && dismissed) return null;

  const style = STYLE[data.status as keyof typeof STYLE] ?? STYLE.error;

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
        flexDirection: 'column',
        gap: '6px',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span>{style.icon}</span>
        <span style={{ flex: 1 }}>
          {data.status === 'scheduled' && (
            <>v132 編排管線評估已過排程日（{data.target_date}），等下次系統開啟 09:00 自動觸發。</>
          )}
          {data.status === 'passed' && (
            <>
              <strong>v132 評估通過</strong>
              {data.evaluated_at && <>（{data.evaluated_at} 跑）</>}
              {' '}— 新版品質未劣化，可保留。
            </>
          )}
          {data.status === 'degraded' && (
            <>
              <strong>v132 評估偵測到品質劣化</strong>
              {data.evaluated_at && <>（{data.evaluated_at} 跑）</>}
              {' '}— 建議立即回滾到舊版。
            </>
          )}
          {data.status === 'error' && <>v132 評估狀態讀取失敗：{data.message || '未知錯誤'}</>}
        </span>

        {data.log_tail && (
          <button
            onClick={() => setShowDetails(v => !v)}
            style={{
              background: 'transparent', border: `1px solid ${style.border}`,
              color: style.text, padding: '2px 8px', borderRadius: 3,
              cursor: 'pointer', fontSize: '11px',
            }}
          >
            {showDetails ? '收起' : '查看詳情'}
          </button>
        )}

        {data.status === 'degraded' && !confirmRollback && (
          <button
            onClick={() => setConfirmRollback(true)}
            disabled={rollbackInFlight}
            style={{
              background: '#f85149', border: 'none', color: '#fff',
              padding: '3px 10px', borderRadius: 3,
              cursor: rollbackInFlight ? 'not-allowed' : 'pointer',
              fontSize: '11px', fontWeight: 600,
            }}
          >
            立即回滾
          </button>
        )}

        {data.status === 'passed' && (
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
        )}
      </div>

      {confirmRollback && data.status === 'degraded' && (
        <div
          style={{
            display: 'flex', alignItems: 'center', gap: '8px',
            paddingTop: '4px', borderTop: `1px dashed ${style.border}`,
          }}
        >
          <span>確認回滾?將關閉 v132 新版,需手動重啟後端才會生效。</span>
          <button
            onClick={handleRollback}
            disabled={rollbackInFlight}
            style={{
              background: '#f85149', border: 'none', color: '#fff',
              padding: '3px 10px', borderRadius: 3,
              cursor: rollbackInFlight ? 'not-allowed' : 'pointer',
              fontSize: '11px',
            }}
          >
            {rollbackInFlight ? '回滾中...' : '確認回滾'}
          </button>
          <button
            onClick={() => setConfirmRollback(false)}
            disabled={rollbackInFlight}
            style={{
              background: 'transparent', border: `1px solid ${style.border}`,
              color: style.text, padding: '3px 10px', borderRadius: 3,
              cursor: 'pointer', fontSize: '11px',
            }}
          >
            取消
          </button>
        </div>
      )}

      {showDetails && data.log_tail && (
        <pre
          style={{
            margin: 0, padding: '8px', maxHeight: '200px', overflowY: 'auto',
            background: 'rgba(0,0,0,0.3)', borderRadius: 3,
            fontSize: '11px', fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
            whiteSpace: 'pre-wrap', wordBreak: 'break-word',
          }}
        >
          {data.log_tail}
        </pre>
      )}
    </div>
  );
}
