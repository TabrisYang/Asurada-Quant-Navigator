/** 阿斯拉量化系統 — Google Sheet 匯出設定精靈（一次性）
 *
 * 五步驟引導使用者部署 Apps Script Web App：
 *   1. 設定匯出密碼（嵌入腳本 + 存後端本機 config，之後匯出全自動）
 *   2. 開啟 script.google.com
 *   3. 一鍵複製腳本貼上
 *   4. 照指示部署（網頁應用程式 / 以我執行 / 任何人）
 *   5. 貼回 /exec 網址 → 後端 ping 驗證密碼相符後儲存
 */

import { useState } from 'react';
import { submitGoogleSheetsPassword, saveGoogleSheetsConfig } from '../../services/api';
import { toast } from '../Toast';

const STEP_TITLES = ['設定密碼', '開啟 Apps Script', '貼上程式碼', '部署', '完成設定'];

function apiErrMsg(e: unknown): string {
  const detail = (e as { response?: { data?: { detail?: { message?: string } | string } } })?.response?.data?.detail;
  if (typeof detail === 'string') return detail;
  return detail?.message || (e as Error)?.message || '未知錯誤';
}

export function GoogleSheetSetupWizard({ onComplete, onClose }: {
  onComplete: () => void;
  onClose: () => void;
}) {
  const [step, setStep] = useState(0);
  const [password, setPassword] = useState('');
  const [password2, setPassword2] = useState('');
  const [scriptCode, setScriptCode] = useState('');
  const [webhookUrl, setWebhookUrl] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePasswordNext = async () => {
    if (password.length < 10) { setError('密碼至少需 10 個字元'); return; }
    if (password !== password2) { setError('兩次輸入的密碼不一致'); return; }
    setBusy(true); setError(null);
    try {
      const r = await submitGoogleSheetsPassword(password);
      setScriptCode(r.script_code);
      setStep(1);
    } catch (e) {
      setError(apiErrMsg(e));
    } finally {
      setBusy(false);
    }
  };

  const handleCopyScript = async () => {
    try {
      await navigator.clipboard.writeText(scriptCode);
      toast('已複製程式碼', 'success');
    } catch {
      toast('複製失敗，請手動全選複製', 'warning');
    }
  };

  const handleFinish = async () => {
    if (!webhookUrl.trim()) { setError('請貼上部署後的網址'); return; }
    setBusy(true); setError(null);
    try {
      await saveGoogleSheetsConfig(webhookUrl.trim());
      toast('Google Sheet 匯出設定完成 ✅', 'success');
      onComplete();
    } catch (e) {
      setError(apiErrMsg(e));
    } finally {
      setBusy(false);
    }
  };

  const inputStyle = {
    background: 'var(--bg-tertiary)', color: 'var(--text-primary)',
    border: '1px solid var(--border-color)',
  } as const;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ background: 'rgba(0,0,0,0.6)' }}>
      <div
        className="rounded-xl shadow-2xl flex flex-col w-[560px] max-w-[92vw] max-h-[85vh]"
        style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}
      >
        {/* 標題列 */}
        <div className="flex items-center justify-between px-5 py-3 border-b shrink-0" style={{ borderColor: 'var(--border-color)' }}>
          <div className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
            📤 Google Sheet 匯出 — 一次性設定（約 5 分鐘）
          </div>
          <button onClick={onClose} className="cursor-pointer text-lg leading-none hover:opacity-70" style={{ color: 'var(--text-secondary)' }}>✕</button>
        </div>

        {/* 步驟進度 */}
        <div className="flex items-center gap-1 px-5 py-3 shrink-0">
          {STEP_TITLES.map((t, i) => (
            <div key={t} className="flex items-center gap-1">
              <div
                className="w-5 h-5 rounded-full flex items-center justify-center text-[11px] font-medium shrink-0"
                style={{
                  background: i <= step ? 'var(--accent-blue)' : 'var(--bg-tertiary)',
                  color: i <= step ? '#fff' : 'var(--text-secondary)',
                }}
              >{i + 1}</div>
              <span className="text-xs whitespace-nowrap" style={{ color: i === step ? 'var(--text-primary)' : 'var(--text-secondary)' }}>{t}</span>
              {i < STEP_TITLES.length - 1 && <span style={{ color: 'var(--text-secondary)' }}>›</span>}
            </div>
          ))}
        </div>

        {/* 內容 */}
        <div className="px-5 pb-4 overflow-y-auto text-sm space-y-3" style={{ color: 'var(--text-primary)' }}>
          {step === 0 && (
            <>
              <p>設定一組「匯出密碼」。它會嵌入你的 Google 腳本裡當通行檢查 — 別人就算知道網址，沒密碼也寫不進你的試算表。密碼<b>只存在後端記憶體</b>（不寫入磁碟），平常匯出全自動，後端重啟後第一次匯出會再問一次。</p>
              <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                ⚠️ 至少 10 個字元；密碼會以明文嵌在你的 Google 腳本中，請<b>勿使用</b>與其他網站相同的密碼。
              </p>
              <input
                type="password" value={password} onChange={(e) => setPassword(e.target.value)}
                placeholder="匯出密碼（至少 10 字元）"
                className="w-full px-3 py-2 rounded outline-none" style={inputStyle}
              />
              <input
                type="password" value={password2} onChange={(e) => setPassword2(e.target.value)}
                placeholder="再輸入一次確認"
                className="w-full px-3 py-2 rounded outline-none" style={inputStyle}
              />
            </>
          )}

          {step === 1 && (
            <>
              <p>開啟 Google Apps Script 編輯器（會用你的 Google 帳號登入、自動建立一個新專案）：</p>
              <button
                onClick={() => window.open('https://script.google.com/create', '_blank')}
                className="px-4 py-2 rounded cursor-pointer hover:opacity-90"
                style={{ background: 'var(--accent-blue)', color: '#fff' }}
              >開啟 script.google.com ↗</button>
              <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>開好後回到這裡按「下一步」。</p>
            </>
          )}

          {step === 2 && (
            <>
              <p>複製下面的程式碼，回到 Apps Script 編輯器，<b>全選並刪除</b>編輯器裡原本的內容（那幾行 <code>function myFunction()...</code>），再貼上：</p>
              <div className="relative">
                <textarea
                  readOnly value={scriptCode} rows={10}
                  className="w-full px-3 py-2 rounded outline-none font-mono text-xs resize-none"
                  style={inputStyle}
                />
                <button
                  onClick={handleCopyScript}
                  className="absolute top-2 right-2 px-2 py-1 rounded text-xs cursor-pointer hover:opacity-90"
                  style={{ background: 'var(--accent-blue)', color: '#fff' }}
                >📋 一鍵複製</button>
              </div>
              <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>貼上後按編輯器的 💾（或 Cmd+S）儲存。</p>
            </>
          )}

          {step === 3 && (
            <>
              <p>在 Apps Script 編輯器右上角照以下順序點（一生只做這一次）：</p>
              <ol className="list-decimal pl-5 space-y-1.5">
                <li>點右上角藍色「<b>部署</b>」→「<b>新增部署作業</b>」</li>
                <li>左上齒輪 ⚙️ → 類型選「<b>網頁應用程式</b>」</li>
                <li>「執行身分」選「<b>我</b>」（你的帳號）</li>
                <li>「誰可以存取」選「<b>任何人</b>」（放心 — 有匯出密碼把關）</li>
                <li>按「<b>部署</b>」→ 跳出授權視窗 → 選你的帳號</li>
                <li>若出現「Google 尚未驗證這個應用程式」：點「<b>進階</b>」→「<b>前往(不安全的網頁)</b>」→「<b>允許</b>」（這是你自己寫的腳本，Google 對所有個人腳本都會這樣提示）</li>
                <li>完成後畫面會顯示「<b>網頁應用程式 網址</b>」（結尾是 <code>/exec</code>）→ 按旁邊「複製」</li>
              </ol>
            </>
          )}

          {step === 4 && (
            <>
              <p>把剛剛複製的「網頁應用程式網址」貼到下面，系統會自動連線驗證密碼是否相符：</p>
              <input
                type="text" value={webhookUrl} onChange={(e) => setWebhookUrl(e.target.value)}
                placeholder="https://script.google.com/macros/s/…/exec"
                className="w-full px-3 py-2 rounded outline-none font-mono text-xs" style={inputStyle}
              />
            </>
          )}

          {error && <p className="text-xs" style={{ color: '#dc2626' }}>⚠️ {error}</p>}
        </div>

        {/* 底部按鈕 */}
        <div className="flex justify-between px-5 py-3 border-t shrink-0" style={{ borderColor: 'var(--border-color)' }}>
          <button
            onClick={() => { setError(null); step === 0 ? onClose() : setStep(step - 1); }}
            disabled={busy}
            className="px-4 py-1.5 rounded text-sm cursor-pointer hover:opacity-90"
            style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
          >{step === 0 ? '取消' : '← 上一步'}</button>

          {step === 0 && (
            <button onClick={handlePasswordNext} disabled={busy}
              className="px-4 py-1.5 rounded text-sm font-medium cursor-pointer hover:opacity-90"
              style={{ background: 'var(--accent-blue)', color: '#fff', opacity: busy ? 0.6 : 1 }}
            >{busy ? '處理中…' : '下一步 →'}</button>
          )}
          {(step === 1 || step === 2 || step === 3) && (
            <button onClick={() => { setError(null); setStep(step + 1); }}
              className="px-4 py-1.5 rounded text-sm font-medium cursor-pointer hover:opacity-90"
              style={{ background: 'var(--accent-blue)', color: '#fff' }}
            >下一步 →</button>
          )}
          {step === 4 && (
            <button onClick={handleFinish} disabled={busy}
              className="px-4 py-1.5 rounded text-sm font-medium cursor-pointer hover:opacity-90"
              style={{ background: 'var(--accent-blue)', color: '#fff', opacity: busy ? 0.6 : 1 }}
            >{busy ? '驗證中…' : '✓ 驗證並完成'}</button>
          )}
        </div>
      </div>
    </div>
  );
}
