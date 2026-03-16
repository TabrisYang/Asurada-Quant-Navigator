/** 阿斯拉量化系統 — 新手引導組件
 *
 * 首次使用時自動顯示互動式引導，介紹系統核心功能。
 * 引導完成後記錄到 localStorage，不再重複顯示。
 */

import { useState, useCallback } from 'react';

const STORAGE_KEY = 'asura_onboarding_done';

interface Step {
  title: string;
  content: string;
  highlight: string;
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
  },
  {
    title: '步驟 2：查看 K 線圖',
    content:
      '中央區域顯示互動式 K 線圖。你可以縮放、拖動來瀏覽不同時間段。圖表會自動適應價格精度（BTC 顯示 2 位小數，ADA 顯示 4 位等）。',
    highlight: 'center',
  },
  {
    title: '步驟 3：使用技術指標',
    content:
      '左側面板提供 30 種技術指標，涵蓋趨勢、動量、量能、波動率、風險管理等。點擊即可疊加到圖表，參數可即時調整。滑鼠懸停指標名稱可查看定義說明。',
    highlight: 'left',
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
  },
  {
    title: '步驟 5：設定 LLM',
    content:
      '使用 AI 助手前，需先到「設定」中輸入 LLM API Key（支援 Gemini、OpenAI、Claude、Ollama）。系統會自動偵測可用模型讓你選擇。API Key 在關閉瀏覽器後自動失效。',
    highlight: 'top',
  },
  {
    title: '準備就緒！',
    content:
      '你已經了解所有核心功能。系統會越用越聰明——過去的分析結果會自動融入未來的回答中，減少 Token 消耗。\n\n祝你交易順利！',
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

  // @ts-expect-error reserved for future highlight feature
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const highlightStyle = (area: string): React.CSSProperties => {
    if (!current.highlight || current.highlight !== area) return {};
    return {
      boxShadow: '0 0 0 4px rgba(88, 166, 255, 0.4)',
      borderRadius: '8px',
      position: 'relative' as const,
      zIndex: 60,
    };
  };

  return (
    <>
      {/* Overlay backdrop */}
      <div
        className="fixed inset-0 z-50"
        style={{ background: 'rgba(0,0,0,0.65)', pointerEvents: 'auto' }}
      />

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
