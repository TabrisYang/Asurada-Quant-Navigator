/** 阿斯拉量化系統 — 全域通知系統（Toast） */

import { useState, useEffect, useCallback, useRef } from 'react';

export type ToastLevel = 'success' | 'info' | 'warning' | 'error';

interface ToastItem {
  id: number;
  message: string;
  level: ToastLevel;
  duration: number;
}

const LEVEL_STYLE: Record<ToastLevel, { bg: string; border: string; icon: string }> = {
  success: { bg: 'rgba(63,185,80,0.12)', border: '#3fb950', icon: '✅' },
  info:    { bg: 'rgba(88,166,255,0.12)', border: '#58a6ff', icon: 'ℹ️' },
  warning: { bg: 'rgba(210,153,34,0.12)', border: '#d29922', icon: '⚠️' },
  error:   { bg: 'rgba(248,81,73,0.12)',  border: '#f85149', icon: '❌' },
};

let _nextId = 0;
let _addToast: ((msg: string, level: ToastLevel, duration?: number) => void) | null = null;

export function toast(message: string, level: ToastLevel = 'info', duration = 3500) {
  _addToast?.(message, level, duration);
}

export function ToastContainer() {
  const [items, setItems] = useState<ToastItem[]>([]);
  const timers = useRef<Map<number, ReturnType<typeof setTimeout>>>(new Map());

  const add = useCallback((message: string, level: ToastLevel, duration = 3500) => {
    const id = ++_nextId;
    setItems(prev => {
      const next = [...prev, { id, message, level, duration }];
      return next.length > 5 ? next.slice(-5) : next;
    });
    const timer = setTimeout(() => {
      setItems(prev => prev.filter(t => t.id !== id));
      timers.current.delete(id);
    }, duration);
    timers.current.set(id, timer);
  }, []);

  useEffect(() => {
    _addToast = add;
    return () => { _addToast = null; };
  }, [add]);

  const dismiss = (id: number) => {
    setItems(prev => prev.filter(t => t.id !== id));
    const timer = timers.current.get(id);
    if (timer) { clearTimeout(timer); timers.current.delete(id); }
  };

  if (items.length === 0) return null;

  return (
    <div style={{
      position: 'fixed', top: 16, right: 16, zIndex: 99999,
      display: 'flex', flexDirection: 'column', gap: 8,
      pointerEvents: 'none', maxWidth: 380,
    }}>
      {items.map(item => {
        const s = LEVEL_STYLE[item.level];
        return (
          <div
            key={item.id}
            style={{
              pointerEvents: 'auto',
              display: 'flex', alignItems: 'center', gap: 8,
              padding: '10px 14px', borderRadius: 8,
              background: s.bg, border: `1px solid ${s.border}`,
              backdropFilter: 'blur(8px)',
              boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
              animation: 'toastSlideIn 0.25s ease-out',
              cursor: 'pointer',
              fontSize: 12, color: 'var(--text-primary, #e6edf3)',
            }}
            onClick={() => dismiss(item.id)}
          >
            <span style={{ fontSize: 14, flexShrink: 0 }}>{s.icon}</span>
            <span style={{ flex: 1, lineHeight: 1.4 }}>{item.message}</span>
            <span style={{ flexShrink: 0, opacity: 0.4, fontSize: 10 }}>✕</span>
          </div>
        );
      })}
      <style>{`
        @keyframes toastSlideIn {
          from { opacity: 0; transform: translateX(40px); }
          to   { opacity: 1; transform: translateX(0); }
        }
      `}</style>
    </div>
  );
}
