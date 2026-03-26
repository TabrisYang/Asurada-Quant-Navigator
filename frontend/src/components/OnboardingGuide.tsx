/** 阿斯拉量化系統 — 新手引導組件
 *
 * 首次使用時自動顯示互動式引導，介紹系統核心功能。
 * 引導完成後記錄到 localStorage，不再重複顯示。
 * 支援區域高亮（透過 CSS Selector 定位 + 遮罩切口）。
 */

import { useState, useCallback, useEffect, useRef } from 'react';

const STORAGE_KEY = 'asura_onboarding_done';

interface Step {
  title: string;
  content: string;
  highlight: string;
  selector?: string;
}

const STEPS: Step[] = [
  {
    title: '歡迎使用阿斯拉量化系統',
    content:
      '這是一套結合 AI 助手的加密貨幣量化分析系統。接下來用 30 秒快速了解核心功能。',
    highlight: '',
  },
  {
    title: '步驟 1：同步數據',
    content:
      '點擊頂部工具列的「同步數據」按鈕，選擇交易對（如 BTC/USDT）、時間範圍和級別，從交易所下載歷史 K 線數據。數據會存在本地，日後只需增量更新。',
    highlight: 'top',
    selector: '[data-guide="topbar"]',
  },
  {
    title: '步驟 2：查看 K 線圖',
    content:
      '中央區域顯示互動式 K 線圖。你可以縮放、拖動來瀏覽不同時間段。圖表會自動適應價格精度（BTC 顯示 2 位小數，ADA 顯示 4 位等）。',
    highlight: 'center',
    selector: '[data-guide="chart"]',
  },
  {
    title: '步驟 3：使用技術指標',
    content:
      '左側面板提供 30 種技術指標，涵蓋趨勢、動量、量能、波動率、風險管理等。點擊即可疊加到圖表，參數可即時調整。滑鼠懸停指標名稱可查看定義說明。',
    highlight: 'left',
    selector: '[data-guide="indicator-panel"]',
  },
  {
    title: '步驟 4：與 AI 助手對話',
    content:
      '右側是 AI 助手。用自然語言描述你的需求，例如：\n' +
      '• 「分析 BTC 目前的趨勢和支撐壓力」\n' +
      '• 「在圖表上標記 RSI 低於 30 的時間段」\n' +
      '• 「畫出趨勢線」\n' +
      '• 「回測 RSI 超賣買入策略的績效」\n' +
      'AI 會自動操作圖表、計算指標、生成分析報告。',
    highlight: 'right',
    selector: '[data-guide="chat"]',
  },
  {
    title: '步驟 5：因子掃描',
    content:
      '頂部的「因子掃描」按鈕可以自動分析所有指標的預測力，找出近期最有效的因子。結果包含 IC 分析、Alpha Decay、分位數最佳區間、策略分級建議等。你可以用「策略建構精靈」一鍵送出回測。',
    highlight: 'top',
    selector: '[data-guide="topbar"]',
  },
  {
    title: '步驟 6：設定 LLM',
    content:
      '使用 AI 助手前，需先到「設定」中輸入 LLM API Key（支援 Gemini、OpenAI、Claude、Ollama）。系統會自動偵測可用模型讓你選擇。API Key 在關閉瀏覽器後自動失效。',
    highlight: 'top',
    selector: '[data-guide="topbar"]',
  },
  {
    title: '準備就緒！',
    content:
      '你已經了解所有核心功能。系統會越用越聰明——過去的分析結果會自動融入未來的回答中，減少 Token 消耗。你還可以在知識庫中手動添加學習筆記。\n\n祝你交易順利！',
    highlight: '',
  },
];

export function useOnboarding() {
  const [visible, setVisible] = useState(() => {
    try {
      return !localStorage.getItem(STORAGE_KEY);
    } catch {
      return true;
    }
  });

  const dismiss = useCallback(() => {
    setVisible(false);
    try {
      localStorage.setItem(STORAGE_KEY, '1');
    } catch { /* ignore */ }
  }, []);

  const reset = useCallback(() => {
    setVisible(true);
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch { /* ignore */ }
  }, []);

  return { visible, dismiss, reset };
}

interface OnboardingGuideProps {
  onDismiss: () => void;
}

export default function OnboardingGuide({ onDismiss }: OnboardingGuideProps) {
  const [step, setStep] = useState(0);
  const current = STEPS[step];
  const isLast = step === STEPS.length - 1;
  const isFirst = step === 0;

  const [highlightRect, setHighlightRect] = useState<DOMRect | null>(null);
  const rafRef = useRef(0);

  useEffect(() => {
    if (!current.selector) {
      setHighlightRect(null);
      return;
    }
    const update = () => {
      const el = document.querySelector(current.selector!);
      if (el) {
        setHighlightRect(el.getBoundingClientRect());
      } else {
        setHighlightRect(null);
      }
    };
    update();
    rafRef.current = requestAnimationFrame(update);
    return () => cancelAnimationFrame(rafRef.current);
  }, [current.selector, step]);

  const cutoutStyle: React.CSSProperties = highlightRect ? {
    clipPath: `polygon(
      0% 0%, 100% 0%, 100% 100%, 0% 100%, 0% 0%,
      ${highlightRect.left - 4}px ${highlightRect.top - 4}px,
      ${highlightRect.left - 4}px ${highlightRect.bottom + 4}px,
      ${highlightRect.right + 4}px ${highlightRect.bottom + 4}px,
      ${highlightRect.right + 4}px ${highlightRect.top - 4}px,
      ${highlightRect.left - 4}px ${highlightRect.top - 4}px
    )`,
  } : {};

  return (
    <>
      {/* Overlay backdrop with cutout for highlighted area */}
      <div
        className="fixed inset-0 z-50"
        style={{
          background: 'rgba(0,0,0,0.65)',
          pointerEvents: 'auto',
          transition: 'clip-path 0.3s ease',
          ...cutoutStyle,
        }}
      />

      {/* Highlight border ring */}
      {highlightRect && (
        <div
          className="fixed z-[55] pointer-events-none"
          style={{
            left: highlightRect.left - 4,
            top: highlightRect.top - 4,
            width: highlightRect.width + 8,
            height: highlightRect.height + 8,
            borderRadius: 8,
            border: '2px solid #58a6ff',
            boxShadow: '0 0 20px rgba(88,166,255,0.3)',
            transition: 'all 0.3s ease',
          }}
        />
      )}

      {/* Guide dialog */}
      <div
        className="fixed z-[60] flex items-center justify-center inset-0"
        style={{ pointerEvents: 'none' }}
      >
        <div
          className="rounded-xl shadow-2xl p-6 max-w-md w-full mx-4"
          style={{
            background: 'var(--bg-secondary, #161b22)',
            border: '1px solid var(--border-color, #30363d)',
            pointerEvents: 'auto',
          }}
        >
          {/* Progress indicator */}
          <div className="flex gap-1 mb-4">
            {STEPS.map((_, i) => (
              <div
                key={i}
                className="h-1 flex-1 rounded-full transition-colors"
                style={{
                  background: i <= step ? 'var(--accent-blue, #58a6ff)' : 'var(--bg-tertiary, #21262d)',
                }}
              />
            ))}
          </div>

          {/* Step counter */}
          <p className="text-xs mb-2" style={{ color: 'var(--text-secondary, #8b949e)' }}>
            {step + 1} / {STEPS.length}
          </p>

          {/* Title */}
          <h3 className="text-base font-bold mb-3" style={{ color: 'var(--text-primary, #e6edf3)' }}>
            {current.title}
          </h3>

          {/* Content */}
          <div
            className="text-sm mb-6 whitespace-pre-line leading-relaxed"
            style={{ color: 'var(--text-secondary, #8b949e)' }}
          >
            {current.content}
          </div>

          {/* Buttons */}
          <div className="flex items-center gap-3">
            <button
              onClick={onDismiss}
              className="text-xs cursor-pointer"
              style={{ color: 'var(--text-secondary, #8b949e)' }}
            >
              跳過引導
            </button>

            <div className="flex-1" />

            {!isFirst && (
              <button
                onClick={() => setStep((s) => s - 1)}
                className="px-3 py-1.5 rounded-lg text-sm cursor-pointer"
                style={{ background: 'var(--bg-tertiary, #21262d)', color: 'var(--text-primary, #e6edf3)' }}
              >
                上一步
              </button>
            )}

            <button
              onClick={() => {
                if (isLast) {
                  onDismiss();
                } else {
                  setStep((s) => s + 1);
                }
              }}
              className="px-4 py-1.5 rounded-lg text-sm cursor-pointer"
              style={{ background: 'var(--accent-blue, #58a6ff)', color: '#fff' }}
            >
              {isLast ? '開始使用' : '下一步'}
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
