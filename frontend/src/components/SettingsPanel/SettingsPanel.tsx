/** 阿斯拉量化系統 — 設定面板組件
 *
 * 流程：選供應商 → 輸入 API Key → 探測可用模型 → 選模型 → 測試連線（用選的模型）→ 儲存
 *
 * 核心原則：
 * - 不使用預設模型，使用者必須自己選擇
 * - 測試連線會用使用者選的模型，確認真的能用
 * - 模型清單完全來自 API 動態偵測，不寫死
 */

import { useState, useRef, useEffect, useCallback, type CSSProperties } from 'react';
import { useChartStore } from '../../stores/chartStore';
import {
  configureLLM, testLLMConnection, discoverModels, exportKnowledgePDF,
  fetchStrategies, addStrategy, updateStrategy, deleteStrategy,
  fetchSystemSettings, updateSystemSettings, clearSessionCache,
  type UserStrategy,
} from '../../services/api';
import { persistSession } from '../../services/session';
import type { LLMProvider } from '../../types';
import MLPanel from './MLPanel';
import PredictionDashboard from '../PredictionDashboard/PredictionDashboard';

const LLM_PROVIDERS: { id: LLMProvider; name: string; requiresKey: boolean; desc: string }[] = [
  { id: 'openai', name: 'OpenAI (GPT-4/4o)', requiresKey: true, desc: '最成熟的 Function Calling（僅 API Key，無訂閱串接）' },
  { id: 'gemini', name: 'Google Gemini', requiresKey: true, desc: '免費額度較高（僅 API Key，無訂閱串接）' },
  { id: 'claude', name: 'Anthropic Claude (API Key)', requiresKey: true, desc: '推理能力強，按 token 計費' },
  { id: 'claude_subscription', name: 'Claude 訂閱制', requiresKey: false, desc: '用 Claude 訂閱額度（本機登入，或各自帶 setup-token）' },
  { id: 'ollama', name: '本地 Ollama', requiresKey: false, desc: '完全免費，無需 API Key' },
];

interface ModelInfo {
  id: string;
  name: string;
  description: string;
}

interface SettingsPanelProps {
  onClose: () => void;
}

export default function SettingsPanel({ onClose }: SettingsPanelProps) {
  const llmConfig = useChartStore((s) => s.llmConfig);
  const setLLMConfig = useChartStore((s) => s.setLLMConfig);
  const [activeTab, setActiveTab] = useState<'llm' | 'export' | 'strategies' | 'ml' | 'predictions' | 'scan'>('llm');
  const [apiKey, setApiKey] = useState('');
  const [baseUrl, setBaseUrl] = useState(llmConfig.baseUrl || 'http://localhost:11434');

  // 模型探測
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>(llmConfig.modelName || '');
  const [loadingModels, setLoadingModels] = useState(false);
  const [modelsMessage, setModelsMessage] = useState('');

  // 測試連線
  const [testStatus, setTestStatus] = useState<'idle' | 'testing' | 'success' | 'failed'>('idle');
  const [testMessage, setTestMessage] = useState('');

  // 儲存
  const [saving, setSaving] = useState(false);

  // 教學模式
  const [teachingMode, setTeachingMode] = useState(false);
  useEffect(() => {
    fetchSystemSettings().then((s) => setTeachingMode(!!s.teaching_mode)).catch(() => {});
  }, []);
  const handleTeachingToggle = async () => {
    const next = !teachingMode;
    setTeachingMode(next);
    await updateSystemSettings({ teaching_mode: next }).catch(() => setTeachingMode(!next));
  };

  // 清除 Session 快取
  const [clearingCache, setClearingCache] = useState(false);
  const [cacheMsg, setCacheMsg] = useState('');
  const handleClearCache = async () => {
    setClearingCache(true);
    setCacheMsg('');
    try {
      const msg = await clearSessionCache();
      setCacheMsg(msg || '已清除');
    } catch {
      setCacheMsg('清除失敗，請稍後再試');
    } finally {
      setClearingCache(false);
      setTimeout(() => setCacheMsg(''), 3000);
    }
  };

  const selectedProvider = LLM_PROVIDERS.find((p) => p.id === llmConfig.provider);
  const hasSession = !!llmConfig.sessionId;

  // ─── Step 1: 探測可用模型 ────────────────
  const handleDiscoverModels = async () => {
    setLoadingModels(true);
    setModelsMessage('');
    setAvailableModels([]);
    setSelectedModel('');
    setTestStatus('idle');
    setTestMessage('');

    try {
      const result = await discoverModels(
        llmConfig.provider,
        llmConfig.sessionId,
        // claude_subscription：帶上選填的 OAuth token（部署到無本機登入的伺服器時用得到）
        (selectedProvider?.requiresKey || llmConfig.provider === 'claude_subscription')
          ? (apiKey || undefined)
          : undefined,
        llmConfig.provider === 'ollama' ? baseUrl : undefined,
      );

      const models: ModelInfo[] = result.models || [];
      setAvailableModels(models);

      if (models.length === 0) {
        setModelsMessage('未偵測到可用模型，請檢查 API Key 是否正確');
      } else if (models.length === 1) {
        setSelectedModel(models[0].id);
        setModelsMessage(`偵測到 1 個可用模型，已自動選取`);
      } else {
        setModelsMessage(`偵測到 ${models.length} 個可用模型，請選擇一個`);
        // 如果之前有選過，嘗試保留
        if (llmConfig.modelName && models.some((m) => m.id === llmConfig.modelName)) {
          setSelectedModel(llmConfig.modelName);
        }
      }
    } catch (err: any) {
      const rawDetail = err?.response?.data?.detail;
      const detail = typeof rawDetail === 'string' ? rawDetail
        : Array.isArray(rawDetail) ? rawDetail.map((d: any) => d.msg || JSON.stringify(d)).join('; ')
        : '';
      if (detail.includes('SESSION_EXPIRED')) {
        clearPersistedSession();
        setLLMConfig({ sessionId: undefined });
        setModelsMessage('先前的連線已過期，請重新輸入 API Key 後再偵測模型');
      } else if (detail.includes('401') || detail.includes('無效')) {
        setModelsMessage('API Key 無效，無法偵測模型');
      } else {
        setModelsMessage(detail || '偵測模型失敗，請檢查 API Key 和網路');
      }
    } finally {
      setLoadingModels(false);
    }
  };

  // ─── Step 2: 測試連線（使用選中的模型）────────────────
  const handleTest = async () => {
    if (!selectedModel) return;
    setTestStatus('testing');
    setTestMessage('');

    try {
      const result = await testLLMConnection(
        llmConfig.provider,
        llmConfig.sessionId,
        // claude_subscription：用選填的 OAuth token 實際驗證該使用者的訂閱能否連線
        (selectedProvider?.requiresKey || llmConfig.provider === 'claude_subscription')
          ? (apiKey || undefined)
          : undefined,
        llmConfig.provider === 'ollama' ? baseUrl : undefined,
        selectedModel,
      );
      setTestStatus('success');
      setTestMessage(result.message || '連線成功');
    } catch (err: any) {
      setTestStatus('failed');
      const rawDetail2 = err?.response?.data?.detail;
      const detail = typeof rawDetail2 === 'string' ? rawDetail2
        : Array.isArray(rawDetail2) ? rawDetail2.map((d: any) => d.msg || JSON.stringify(d)).join('; ')
        : '';
      if (detail.includes('SESSION_EXPIRED')) {
        clearPersistedSession();
        setLLMConfig({ sessionId: undefined });
        setTestMessage('先前的連線已過期，請重新輸入 API Key');
      } else if (detail.includes('RESOURCE_EXHAUSTED') || detail.includes('429') || detail.includes('quota')) {
        setTestMessage(`模型 ${selectedModel} 的免費額度已用完，請選擇其他模型或稍後再試`);
      } else if (detail.includes('401') || detail.includes('UNAUTHENTICATED') || detail.includes('API key')) {
        setTestMessage('API Key 無效，請檢查後重新輸入');
      } else if (detail.includes('404') || detail.includes('NOT_FOUND')) {
        setTestMessage(`模型 ${selectedModel} 不可用，請選擇其他模型`);
      } else {
        setTestMessage(detail || '連線失敗，請檢查 API Key 和網路');
      }
    }
  };

  // ─── Step 3: 儲存 ────────────────
  const handleSave = async () => {
    setSaving(true);
    try {
      const result = await configureLLM(
        llmConfig.provider,
        // claude_subscription：apiKey 此處承載「選填」的個人 OAuth token；留空則用本機登入
        (selectedProvider?.requiresKey || llmConfig.provider === 'claude_subscription')
          ? (apiKey || undefined)
          : undefined,
        selectedModel,
        llmConfig.provider === 'ollama' ? baseUrl : undefined,
      );

      if (result.session_id) {
        setLLMConfig({
          provider: llmConfig.provider,
          sessionId: result.session_id,
          modelName: selectedModel,
          baseUrl: llmConfig.provider === 'ollama' ? baseUrl : undefined,
        });

        persistSession(result.session_id, llmConfig.provider, selectedModel);
        setApiKey('');
        onClose();
      }
    } catch (err: any) {
      setTestStatus('failed');
      const rawSaveErr = err?.response?.data?.detail;
      const saveErrMsg = typeof rawSaveErr === 'string' ? rawSaveErr
        : Array.isArray(rawSaveErr) ? rawSaveErr.map((d: any) => d.msg || JSON.stringify(d)).join('; ')
        : '儲存失敗';
      setTestMessage(saveErrMsg);
    } finally {
      setSaving(false);
    }
  };

  // 能否探測模型：有 key 或已有 session（Ollama 不需要 key）
  const canDiscover = !loadingModels && (!!apiKey || hasSession || !selectedProvider?.requiresKey);

  // 能否測試：必須已選模型
  const canTest = !!selectedModel && testStatus !== 'testing';

  // 能否儲存：已選模型 + 測試成功
  const canSave = !saving && !!selectedModel && testStatus === 'success';

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ background: 'rgba(0,0,0,0.6)' }}
      onClick={onClose}
    >
      <div
        className={`rounded-xl shadow-2xl p-6 max-h-[80vh] overflow-y-auto ${activeTab === 'ml' ? 'w-[600px]' : 'w-[520px]'}`}
        style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="text-lg font-bold mb-4" style={{ color: 'var(--text-primary)' }}>
          系統設定
        </h2>

        {/* ===== 分頁標籤 ===== */}
        <div className="flex gap-2 mb-5" style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>
          <button
            onClick={() => setActiveTab('llm')}
            className="px-3 py-1.5 rounded-t text-sm cursor-pointer"
            style={{
              color: activeTab === 'llm' ? 'var(--accent-blue)' : 'var(--text-secondary)',
              borderBottom: activeTab === 'llm' ? '2px solid var(--accent-blue)' : '2px solid transparent',
              background: 'transparent',
            }}
          >
            LLM 設定
          </button>
          <button
            onClick={() => setActiveTab('strategies')}
            className="px-3 py-1.5 rounded-t text-sm cursor-pointer"
            style={{
              color: activeTab === 'strategies' ? 'var(--accent-blue)' : 'var(--text-secondary)',
              borderBottom: activeTab === 'strategies' ? '2px solid var(--accent-blue)' : '2px solid transparent',
              background: 'transparent',
            }}
          >
            分析策略庫
          </button>
          <button
            onClick={() => setActiveTab('ml')}
            className="px-3 py-1.5 rounded-t text-sm cursor-pointer"
            style={{
              color: activeTab === 'ml' ? 'var(--accent-blue)' : 'var(--text-secondary)',
              borderBottom: activeTab === 'ml' ? '2px solid var(--accent-blue)' : '2px solid transparent',
              background: 'transparent',
            }}
          >
            ML 增強
          </button>
          <button
            onClick={() => setActiveTab('predictions')}
            className="px-3 py-1.5 rounded-t text-sm cursor-pointer"
            style={{
              color: activeTab === 'predictions' ? 'var(--accent-blue)' : 'var(--text-secondary)',
              borderBottom: activeTab === 'predictions' ? '2px solid var(--accent-blue)' : '2px solid transparent',
              background: 'transparent',
            }}
          >
            預測追蹤
          </button>
          <button
            onClick={() => setActiveTab('export')}
            className="px-3 py-1.5 rounded-t text-sm cursor-pointer"
            style={{
              color: activeTab === 'export' ? 'var(--accent-blue)' : 'var(--text-secondary)',
              borderBottom: activeTab === 'export' ? '2px solid var(--accent-blue)' : '2px solid transparent',
              background: 'transparent',
            }}
          >
            匯出/匯入
          </button>
          <button
            onClick={() => setActiveTab('scan')}
            className="px-3 py-1.5 rounded-t text-sm cursor-pointer"
            style={{
              color: activeTab === 'scan' ? 'var(--accent-blue)' : 'var(--text-secondary)',
              borderBottom: activeTab === 'scan' ? '2px solid var(--accent-blue)' : '2px solid transparent',
              background: 'transparent',
            }}
          >
            掃描設定
          </button>
        </div>

        {activeTab === 'strategies' && <StrategySection />}
        {activeTab === 'export' && <ExportImportSection onClose={onClose} />}
        {activeTab === 'ml' && <MLPanel />}
        {activeTab === 'predictions' && <PredictionDashboard />}
        {activeTab === 'scan' && <ScanSettingsSection />}

        {activeTab === 'llm' && <>
        {/* ===== 1. LLM 供應商選擇 ===== */}
        <div className="mb-6">
          <StepLabel step={1} text="選擇供應商" />
          <div className="space-y-2">
            {LLM_PROVIDERS.map((provider) => (
              <button
                key={provider.id}
                onClick={() => {
                  setLLMConfig({ provider: provider.id });
                  setTestStatus('idle');
                  setTestMessage('');
                  setModelsMessage('');
                  setAvailableModels([]);
                  setSelectedModel('');
                }}
                className="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-left cursor-pointer transition-colors"
                style={{
                  background: llmConfig.provider === provider.id ? 'rgba(88, 166, 255, 0.1)' : 'var(--bg-tertiary)',
                  border: llmConfig.provider === provider.id ? '1px solid var(--accent-blue)' : '1px solid transparent',
                }}
              >
                <div
                  className="w-4 h-4 rounded-full border-2 flex items-center justify-center shrink-0"
                  style={{ borderColor: llmConfig.provider === provider.id ? 'var(--accent-blue)' : 'var(--border-color)' }}
                >
                  {llmConfig.provider === provider.id && (
                    <div className="w-2 h-2 rounded-full" style={{ background: 'var(--accent-blue)' }} />
                  )}
                </div>
                <div>
                  <p className="text-sm" style={{ color: 'var(--text-primary)' }}>{provider.name}</p>
                  <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>{provider.desc}</p>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* ===== 2. API Key 輸入 + 探測按鈕 ===== */}
        {selectedProvider?.requiresKey && (
          <div className="mb-6">
            <StepLabel step={2} text="輸入 API Key 並偵測模型" />

            {hasSession && (
              <div
                className="mb-2 px-3 py-2 rounded-lg text-xs flex items-center gap-2"
                style={{ background: 'rgba(63, 185, 80, 0.1)', color: 'var(--accent-green)' }}
              >
                <span>&#x1f512;</span>
                <span>API Key 已加密儲存（重新輸入可更換）</span>
              </div>
            )}

            <div className="flex gap-2">
              <input
                type="password"
                value={apiKey}
                onChange={(e) => {
                  setApiKey(e.target.value);
                  // Key 變了，清除舊的模型和測試結果
                  setAvailableModels([]);
                  setSelectedModel('');
                  setTestStatus('idle');
                  setTestMessage('');
                  setModelsMessage('');
                }}
                placeholder={hasSession ? '已設定（重新輸入可更換）' : `輸入 ${selectedProvider.name} API Key`}
                className="flex-1 px-4 py-2 rounded-lg text-sm border-none outline-none"
                style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
              />
              <button
                onClick={handleDiscoverModels}
                disabled={!canDiscover}
                className="px-4 py-2 rounded-lg text-sm cursor-pointer transition-opacity hover:opacity-80 whitespace-nowrap disabled:opacity-40"
                style={{ background: 'var(--accent-blue)', color: '#fff' }}
              >
                {loadingModels ? '偵測中...' : '偵測模型'}
              </button>
            </div>
            <p className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>
              輸入 API Key 後點「偵測模型」，從可用模型中選擇
            </p>
          </div>
        )}

        {/* ===== Claude 訂閱制：無需 Key，直接偵測 ===== */}
        {llmConfig.provider === 'claude_subscription' && (
          <div className="mb-6">
            <StepLabel step={2} text="（選填）訂閱 Token 並偵測模型" />
            <p className="text-xs mb-2" style={{ color: 'var(--text-secondary)' }}>
              留空 → 使用本機 Claude Code 登入憑證（自用）。<br />
              想用自己的訂閱額度 → 在自己電腦執行{' '}
              <code style={{ background: 'var(--bg-tertiary)', padding: '0 4px', borderRadius: 3 }}>claude setup-token</code>{' '}
              ，把產生的 token 貼到下方，分析就計入你自己的 Claude 訂閱。
            </p>
            <SubscriptionTokenGuide />
            <div className="flex gap-2">
              <input
                type="password"
                value={apiKey}
                onChange={(e) => {
                  setApiKey(e.target.value);
                  setAvailableModels([]);
                  setSelectedModel('');
                  setTestStatus('idle');
                  setTestMessage('');
                  setModelsMessage('');
                }}
                placeholder="（選填）貼上 claude setup-token 產生的 token"
                className="flex-1 px-4 py-2 rounded-lg text-sm border-none outline-none"
                style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
              />
              <button
                onClick={handleDiscoverModels}
                disabled={!canDiscover}
                className="px-4 py-2 rounded-lg text-sm cursor-pointer transition-opacity hover:opacity-80 whitespace-nowrap disabled:opacity-40"
                style={{ background: 'var(--accent-blue)', color: '#fff' }}
              >
                {loadingModels ? '偵測中...' : '偵測模型'}
              </button>
            </div>
            <p className="text-xs mt-2" style={{ color: 'var(--accent-orange, #d29922)' }}>
              ⚠️ token 等同你的 Claude 帳號存取權，僅加密暫存於後端記憶體、24 小時過期、不寫入磁碟。
              請確認你信任此服務再貼上；個人訂閱供他人共用可能違反 Anthropic 使用條款，風險請自行評估。
            </p>
          </div>
        )}

        {/* ===== Ollama Base URL + 探測 ===== */}
        {llmConfig.provider === 'ollama' && (
          <div className="mb-6">
            <StepLabel step={2} text="設定伺服器並偵測模型" />
            <div className="flex gap-2">
              <input
                type="text"
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder="http://localhost:11434"
                className="flex-1 px-4 py-2 rounded-lg text-sm border-none outline-none"
                style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
              />
              <button
                onClick={handleDiscoverModels}
                disabled={!canDiscover}
                className="px-4 py-2 rounded-lg text-sm cursor-pointer transition-opacity hover:opacity-80 whitespace-nowrap disabled:opacity-40"
                style={{ background: 'var(--accent-blue)', color: '#fff' }}
              >
                {loadingModels ? '偵測中...' : '偵測模型'}
              </button>
            </div>
          </div>
        )}

        {/* ===== 偵測訊息 ===== */}
        {modelsMessage && (
          <div
            className="mb-4 px-3 py-2 rounded-lg text-xs"
            style={{
              background: availableModels.length > 0 ? 'rgba(63, 185, 80, 0.1)' : 'rgba(248, 81, 73, 0.1)',
              color: availableModels.length > 0 ? 'var(--accent-green)' : 'var(--accent-red, #f85149)',
            }}
          >
            {modelsMessage}
          </div>
        )}

        {/* ===== 3. 模型選擇 ===== */}
        {availableModels.length > 0 && (
          <div className="mb-6">
            <StepLabel step={3} text="選擇模型" />
            <div className="space-y-1 max-h-[200px] overflow-y-auto pr-1">
              {availableModels.map((m) => (
                <button
                  key={m.id}
                  onClick={() => {
                    setSelectedModel(m.id);
                    // 換模型後要重新測試
                    setTestStatus('idle');
                    setTestMessage('');
                  }}
                  className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left cursor-pointer transition-colors"
                  style={{
                    background: selectedModel === m.id ? 'rgba(88, 166, 255, 0.1)' : 'transparent',
                    border: selectedModel === m.id ? '1px solid rgba(88, 166, 255, 0.4)' : '1px solid transparent',
                  }}
                >
                  <div
                    className="w-3 h-3 rounded-full border-2 flex items-center justify-center shrink-0"
                    style={{ borderColor: selectedModel === m.id ? 'var(--accent-blue)' : 'var(--border-color)' }}
                  >
                    {selectedModel === m.id && (
                      <div className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--accent-blue)' }} />
                    )}
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="text-xs font-medium truncate" style={{ color: 'var(--text-primary)' }}>
                      {m.name || m.id}
                    </p>
                    {m.description && (
                      <p className="text-xs truncate" style={{ color: 'var(--text-secondary)', fontSize: '10px' }}>
                        {m.description}
                      </p>
                    )}
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* ===== 4. 測試連線（用選中的模型）===== */}
        {selectedModel && (
          <div className="mb-4">
            <StepLabel step={4} text={`測試連線（${selectedModel}）`} />
            <button
              onClick={handleTest}
              disabled={!canTest}
              className="w-full px-4 py-2 rounded-lg text-sm cursor-pointer transition-opacity hover:opacity-80 disabled:opacity-40"
              style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}
            >
              {testStatus === 'testing' ? `正在測試 ${selectedModel}...` : `測試 ${selectedModel} 連線`}
            </button>
          </div>
        )}

        {/* ===== 測試結果 ===== */}
        {testMessage && (
          <div
            className="mb-4 px-3 py-2 rounded-lg text-xs whitespace-pre-wrap"
            style={{
              background: testStatus === 'success' ? 'rgba(63, 185, 80, 0.1)' : 'rgba(248, 81, 73, 0.1)',
              color: testStatus === 'success' ? 'var(--accent-green)' : 'var(--accent-red, #f85149)',
            }}
          >
            {testMessage}
          </div>
        )}

        {/* ===== 按鈕列 ===== */}
        <div className="flex items-center gap-3 mt-6">
          <div className="flex-1" />
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg text-sm cursor-pointer"
            style={{ background: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}
          >
            取消
          </button>
          <button
            onClick={handleSave}
            disabled={!canSave}
            className="px-4 py-2 rounded-lg text-sm cursor-pointer disabled:opacity-40"
            style={{ background: 'var(--accent-blue)', color: '#fff' }}
            title={!canSave ? '請先偵測模型、選擇模型、並測試連線成功' : ''}
          >
            {saving ? '儲存中...' : '儲存'}
          </button>
        </div>

        {/* 底部提示 */}
        {!canSave && availableModels.length === 0 && (
          <p className="text-xs mt-3 text-center" style={{ color: 'var(--text-secondary)' }}>
            請先輸入 API Key → 偵測模型 → 選擇模型 → 測試連線 → 儲存
          </p>
        )}
        {!canSave && selectedModel && testStatus !== 'success' && (
          <p className="text-xs mt-3 text-center" style={{ color: 'var(--text-secondary)' }}>
            請測試連線成功後再儲存
          </p>
        )}

        {/* ===== AI 教學模式開關 ===== */}
        <div className="mt-6 pt-4" style={{ borderTop: '1px solid var(--border-primary)' }}>
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
                AI 教學模式
              </div>
              <div className="text-xs mt-0.5" style={{ color: 'var(--text-secondary)' }}>
                開啟後 AI 會在分析中解釋指標意義、信號邏輯與策略風險
              </div>
            </div>
            <button
              onClick={handleTeachingToggle}
              className="relative inline-flex h-6 w-11 items-center rounded-full transition-colors"
              style={{
                backgroundColor: teachingMode ? 'var(--accent-blue)' : 'var(--bg-tertiary)',
              }}
            >
              <span
                className="inline-block h-4 w-4 rounded-full bg-white transition-transform"
                style={{
                  transform: teachingMode ? 'translateX(22px)' : 'translateX(4px)',
                }}
              />
            </button>
          </div>
        </div>

        {/* ===== 清除 Session 快取 ===== */}
        {llmConfig.provider === 'claude_subscription' && (
        <div className="mt-4 pt-4" style={{ borderTop: '1px solid var(--border-primary)' }}>
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
                清除 Session 快取
              </div>
              <div className="text-xs mt-0.5" style={{ color: 'var(--text-secondary)' }}>
                若遇到回應異常或想重置對話快取，可手動清除
              </div>
            </div>
            <button
              onClick={handleClearCache}
              disabled={clearingCache}
              className="px-3 py-1.5 text-xs rounded transition-colors"
              style={{
                backgroundColor: 'var(--bg-tertiary)',
                color: 'var(--text-primary)',
                opacity: clearingCache ? 0.5 : 1,
              }}
            >
              {clearingCache ? '清除中...' : '清除快取'}
            </button>
          </div>
          {cacheMsg && (
            <div className="text-xs mt-2" style={{ color: 'var(--accent-green)' }}>
              {cacheMsg}
            </div>
          )}
        </div>
        )}
        </>}
      </div>
    </div>
  );
}


// ─── 分析策略庫組件 ─────────────────────────

function StrategySection() {
  const [strategies, setStrategies] = useState<UserStrategy[]>([]);
  const [loading, setLoading] = useState(true);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');
  const [editContent, setEditContent] = useState('');
  const [showAdd, setShowAdd] = useState(false);
  const [newTitle, setNewTitle] = useState('');
  const [newContent, setNewContent] = useState('');
  const [saving, setSaving] = useState(false);

  const loadStrategies = useCallback(async () => {
    try {
      setLoading(true);
      const list = await fetchStrategies();
      setStrategies(list);
    } catch {
      /* ignore */
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadStrategies(); }, [loadStrategies]);

  const handleAdd = async () => {
    if (!newTitle.trim() || !newContent.trim()) return;
    setSaving(true);
    try {
      await addStrategy(newTitle.trim(), newContent.trim());
      setNewTitle('');
      setNewContent('');
      setShowAdd(false);
      await loadStrategies();
    } finally { setSaving(false); }
  };

  const handleUpdate = async (id: string) => {
    setSaving(true);
    try {
      await updateStrategy(id, { title: editTitle, content: editContent });
      setEditingId(null);
      await loadStrategies();
    } finally { setSaving(false); }
  };

  const handleDelete = async (id: string) => {
    if (!confirm('確定刪除這條分析策略？')) return;
    await deleteStrategy(id);
    await loadStrategies();
  };

  const handleToggle = async (s: UserStrategy) => {
    await updateStrategy(s.id, { enabled: !s.enabled });
    await loadStrategies();
  };

  const inputStyle: React.CSSProperties = {
    width: '100%',
    padding: '8px 10px',
    borderRadius: '6px',
    border: '1px solid var(--border-color)',
    background: 'var(--bg-secondary)',
    color: 'var(--text-primary)',
    fontSize: '13px',
  };

  const textareaStyle: React.CSSProperties = {
    ...inputStyle,
    minHeight: '120px',
    resize: 'vertical',
    fontFamily: 'monospace',
    fontSize: '12px',
    lineHeight: '1.5',
  };

  return (
    <div>
      <p className="text-xs mb-3" style={{ color: 'var(--text-secondary)' }}>
        在這裡管理你的自訂分析方法論。AI 助手每次分析時會優先參考已啟用的策略，
        但不會侷限於此 — 它也會結合系統內建的 30 種指標和自身的分析知識。
      </p>

      {loading ? (
        <div className="text-center py-4" style={{ color: 'var(--text-secondary)' }}>
          載入中...
        </div>
      ) : (
        <div className="space-y-3">
          {strategies.map((s) => (
            <div
              key={s.id}
              className="rounded-lg p-3"
              style={{
                border: '1px solid var(--border-color)',
                background: s.enabled ? 'var(--bg-secondary)' : 'transparent',
                opacity: s.enabled ? 1 : 0.6,
              }}
            >
              {editingId === s.id ? (
                <div className="space-y-2">
                  <input
                    value={editTitle}
                    onChange={(e) => setEditTitle(e.target.value)}
                    style={inputStyle}
                    placeholder="策略名稱"
                  />
                  <textarea
                    value={editContent}
                    onChange={(e) => setEditContent(e.target.value)}
                    style={textareaStyle}
                  />
                  <div className="flex gap-2">
                    <button
                      onClick={() => handleUpdate(s.id)}
                      disabled={saving}
                      className="px-3 py-1 rounded text-xs cursor-pointer"
                      style={{ background: 'var(--accent-blue)', color: '#fff' }}
                    >
                      {saving ? '儲存中...' : '儲存'}
                    </button>
                    <button
                      onClick={() => setEditingId(null)}
                      className="px-3 py-1 rounded text-xs cursor-pointer"
                      style={{ background: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}
                    >
                      取消
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => handleToggle(s)}
                        className="cursor-pointer"
                        style={{
                          width: '16px', height: '16px', borderRadius: '3px',
                          border: '1px solid var(--border-color)',
                          background: s.enabled ? 'var(--accent-blue)' : 'transparent',
                          color: '#fff', fontSize: '10px', lineHeight: '14px', textAlign: 'center',
                        }}
                        title={s.enabled ? '停用' : '啟用'}
                      >
                        {s.enabled ? '✓' : ''}
                      </button>
                      <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                        {s.title}
                      </span>
                    </div>
                    <div className="flex gap-1">
                      <button
                        onClick={() => { setEditingId(s.id); setEditTitle(s.title); setEditContent(s.content); }}
                        className="px-2 py-0.5 rounded text-xs cursor-pointer hover:opacity-80"
                        style={{ background: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}
                      >
                        編輯
                      </button>
                      <button
                        onClick={() => handleDelete(s.id)}
                        className="px-2 py-0.5 rounded text-xs cursor-pointer hover:opacity-80"
                        style={{ background: 'rgba(248,81,73,0.15)', color: '#f85149' }}
                      >
                        刪除
                      </button>
                    </div>
                  </div>
                  <pre
                    className="text-xs whitespace-pre-wrap mt-1"
                    style={{
                      color: 'var(--text-secondary)',
                      maxHeight: '200px',
                      overflow: 'auto',
                      lineHeight: '1.5',
                    }}
                  >
                    {s.content.length > 500 ? s.content.slice(0, 500) + '...' : s.content}
                  </pre>
                </>
              )}
            </div>
          ))}

          {strategies.length === 0 && !showAdd && (
            <div className="text-center py-6" style={{ color: 'var(--text-secondary)' }}>
              <p className="mb-2">尚未設定任何分析策略</p>
              <p className="text-xs">點擊下方「新增策略」來添加你的分析方法論</p>
            </div>
          )}
        </div>
      )}

      {showAdd ? (
        <div
          className="mt-3 p-3 rounded-lg space-y-2"
          style={{ border: '1px solid var(--accent-blue)', background: 'var(--bg-secondary)' }}
        >
          <input
            value={newTitle}
            onChange={(e) => setNewTitle(e.target.value)}
            style={inputStyle}
            placeholder="策略名稱（例如：30 層機構交易分析框架）"
          />
          <textarea
            value={newContent}
            onChange={(e) => setNewContent(e.target.value)}
            style={textareaStyle}
            placeholder="策略內容（你的分析方法論...）"
          />
          <div className="flex gap-2">
            <button
              onClick={handleAdd}
              disabled={saving || !newTitle.trim() || !newContent.trim()}
              className="px-3 py-1.5 rounded text-xs cursor-pointer"
              style={{
                background: newTitle.trim() && newContent.trim() ? 'var(--accent-blue)' : 'var(--bg-tertiary)',
                color: newTitle.trim() && newContent.trim() ? '#fff' : 'var(--text-secondary)',
              }}
            >
              {saving ? '儲存中...' : '確認新增'}
            </button>
            <button
              onClick={() => { setShowAdd(false); setNewTitle(''); setNewContent(''); }}
              className="px-3 py-1.5 rounded text-xs cursor-pointer"
              style={{ background: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}
            >
              取消
            </button>
          </div>
        </div>
      ) : (
        <button
          onClick={() => setShowAdd(true)}
          className="mt-3 w-full py-2 rounded-lg text-sm cursor-pointer hover:opacity-80"
          style={{
            border: '1px dashed var(--border-color)',
            background: 'transparent',
            color: 'var(--accent-blue)',
          }}
        >
          + 新增策略
        </button>
      )}
    </div>
  );
}


// ─── 匯出/匯入設定組件 ─────────────────────────

function ExportImportSection({ onClose }: { onClose: () => void }) {
  const store = useChartStore();
  const { llmConfig } = store;
  const hasSession = !!llmConfig.sessionId;
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [importMsg, setImportMsg] = useState('');
  const [importStatus, setImportStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const [pdfExporting, setPdfExporting] = useState(false);
  const [pdfMsg, setPdfMsg] = useState('');
  const [pdfIncludeFragments] = useState(true);
  const [pdfIncludeCache] = useState(false);
  const [pdfIncludeHistory] = useState(true);

  const handleExport = () => {
    const config = {
      _format: 'asura_quant_settings_v1',
      _exportedAt: new Date().toISOString(),
      chart: {
        symbol: store.symbol,
        timeframe: store.timeframe,
        startDate: store.startDate,
        endDate: store.endDate,
      },
      indicators: store.activeIndicators.map((ind) => ({
        indicator_type: ind.indicator_type,
        parameters: ind.parameters,
        display_mode: ind.display_mode,
        visible: ind.visible,
      })),
      annotations: store.annotations.map((ann) => ({
        type: ann.type,
        startTime: ann.startTime,
        endTime: ann.endTime,
        price: ann.price,
        endPrice: ann.endPrice,
        text: ann.text,
        color: ann.color,
        lineWidth: ann.lineWidth,
        lineStyle: ann.lineStyle,
      })),
    };

    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `阿斯拉設定_${store.symbol}_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    setImportMsg('');
    setImportStatus('idle');
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const raw = JSON.parse(ev.target?.result as string);
        if (raw._format !== 'asura_quant_settings_v1') {
          setImportStatus('error');
          setImportMsg('檔案格式不正確，請選擇由本系統匯出的 JSON 檔案');
          return;
        }

        // 匯入圖表設定
        if (raw.chart) {
          if (raw.chart.symbol) store.setSymbol(raw.chart.symbol);
          if (raw.chart.timeframe) store.setTimeframe(raw.chart.timeframe);
          if (raw.chart.startDate || raw.chart.endDate) {
            store.setDateRange(raw.chart.startDate, raw.chart.endDate);
          }
        }

        // 匯入指標
        if (Array.isArray(raw.indicators) && raw.indicators.length > 0) {
          store.clearIndicators();
          for (const ind of raw.indicators) {
            store.addIndicator({
              id: `${ind.indicator_type}_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
              indicator_type: ind.indicator_type,
              parameters: ind.parameters || {},
              display_mode: ind.display_mode || 'sub_chart',
              visible: ind.visible !== false,
            });
          }
        }

        // 匯入標記
        if (Array.isArray(raw.annotations) && raw.annotations.length > 0) {
          store.clearAnnotations();
          for (const ann of raw.annotations) {
            store.addAnnotation({
              id: `imp_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
              ...ann,
            });
          }
        }

        setImportStatus('success');
        setImportMsg(`匯入成功！已載入 ${raw.indicators?.length || 0} 個指標、${raw.annotations?.length || 0} 個標記`);
      } catch {
        setImportStatus('error');
        setImportMsg('檔案解析失敗，請確認為有效的 JSON 檔案');
      }
    };
    reader.readAsText(file);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div className="space-y-5">
      {/* 匯出 */}
      <div>
        <h3 className="text-sm font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
          匯出目前設定
        </h3>
        <p className="text-xs mb-3" style={{ color: 'var(--text-secondary)' }}>
          將目前的圖表設定、指標組合、圖表標記匯出為 JSON 檔案，方便備份或分享。
        </p>
        <button
          onClick={handleExport}
          className="px-4 py-2 rounded-lg text-sm cursor-pointer"
          style={{ background: 'var(--accent-blue)', color: '#fff' }}
        >
          匯出設定檔
        </button>
      </div>

      <hr style={{ borderColor: 'var(--border-color)' }} />

      {/* 匯入 */}
      <div>
        <h3 className="text-sm font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
          匯入設定
        </h3>
        <p className="text-xs mb-3" style={{ color: 'var(--text-secondary)' }}>
          從先前匯出的 JSON 檔案還原設定。匯入後會覆蓋目前的指標和標記。
        </p>
        <input
          ref={fileInputRef}
          type="file"
          accept=".json"
          onChange={handleImport}
          className="hidden"
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          className="px-4 py-2 rounded-lg text-sm cursor-pointer"
          style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)', border: '1px solid var(--border-color)' }}
        >
          選擇 JSON 檔案匯入
        </button>
      </div>

      {importMsg && (
        <div
          className="text-xs px-3 py-2 rounded"
          style={{
            background: importStatus === 'success' ? 'rgba(63,185,80,0.1)' : 'rgba(248,81,73,0.1)',
            color: importStatus === 'success' ? 'var(--accent-green)' : 'var(--accent-red, #f85149)',
          }}
        >
          {importMsg}
        </div>
      )}

      <hr style={{ borderColor: 'var(--border-color)' }} />

      {/* 匯出分析報告 PDF */}
      <div>
        <h3 className="text-sm font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
          匯出 AI 分析報告 (PDF)
        </h3>
        <p className="text-xs mb-3" style={{ color: 'var(--text-secondary)' }}>
          {hasSession
            ? 'AI 將根據所有歷史分析資料，自動撰寫結構化的量化分析報告（會消耗少量 token）。'
            : '需要先設定 API Key 才能生成 AI 分析報告。未設定時僅匯出原始資料附錄。'}
        </p>
        {!hasSession && (
          <p className="text-xs mb-3 px-2 py-1 rounded" style={{ background: 'rgba(255,200,0,0.15)', color: 'var(--text-secondary)' }}>
            提示：請先在上方設定 LLM API Key 並連線成功後，再匯出報告以獲得最佳效果。
          </p>
        )}
        <button
          onClick={async () => {
            setPdfExporting(true);
            setPdfMsg('');
            try {
              await exportKnowledgePDF(
                pdfIncludeFragments,
                pdfIncludeCache,
                pdfIncludeHistory,
                llmConfig.sessionId
              );
              setPdfMsg('PDF 已下載');
            } catch (err: any) {
              setPdfMsg(err?.message || '匯出失敗');
            } finally {
              setPdfExporting(false);
            }
          }}
          disabled={pdfExporting}
          className="px-4 py-2 rounded-lg text-sm cursor-pointer disabled:opacity-50"
          style={{ background: '#7c3aed', color: '#fff' }}
        >
          {pdfExporting
            ? (hasSession ? 'AI 正在撰寫報告...' : '報告生成中...')
            : (hasSession ? '匯出 AI 分析報告' : '匯出原始資料報告')}
        </button>
        {pdfMsg && (
          <p className="text-xs mt-2" style={{ color: pdfMsg.includes('失敗') ? 'var(--accent-red, #f85149)' : 'var(--accent-green)' }}>
            {pdfMsg}
          </p>
        )}
      </div>

      <hr style={{ borderColor: 'var(--border-color)' }} />

      {/* 目前設定摘要 */}
      <div className="text-xs rounded-lg p-3" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}>
        <p className="font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>目前設定摘要</p>
        <p>標的：{store.symbol} · {store.timeframe}</p>
        <p>啟用指標：{store.activeIndicators.length} 個</p>
        <p>圖表標記：{store.annotations.length} 個</p>
      </div>

      <div className="flex justify-end">
        <button
          onClick={onClose}
          className="px-4 py-2 rounded-lg text-sm cursor-pointer"
          style={{ background: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}
        >
          關閉
        </button>
      </div>
    </div>
  );
}


// ─── 步驟標籤組件 ─────────────────────────

function StepLabel({ step, text }: { step: number; text: string }) {
  return (
    <h3 className="text-sm font-semibold mb-2 flex items-center gap-2" style={{ color: 'var(--text-primary)' }}>
      <span
        className="w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold shrink-0"
        style={{ background: 'var(--accent-blue)', color: '#fff' }}
      >
        {step}
      </span>
      {text}
    </h3>
  );
}


/** Claude 訂閱制 token 取得教學（可展開）— 給不熟 Claude Code 的使用者自助取得並貼上 token */
function SubscriptionTokenGuide() {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  const copyCmd = async (cmd: string) => {
    try {
      await navigator.clipboard.writeText(cmd);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // 剪貼簿不可用（如非 https）時忽略，使用者仍可手動複製
    }
  };

  const codeStyle: CSSProperties = {
    background: 'var(--bg-primary, #0d1117)',
    padding: '1px 6px',
    borderRadius: 4,
    fontFamily: 'monospace',
    color: 'var(--text-primary)',
  };

  return (
    <div className="mb-2">
      <button
        onClick={() => setOpen((v) => !v)}
        className="text-xs cursor-pointer flex items-center gap-1 hover:opacity-80"
        style={{ color: 'var(--accent-blue)', background: 'none', border: 'none', padding: 0 }}
      >
        <span>{open ? '▾' : '▸'}</span>
        <span>❓ 我沒有 token？看如何取得（約 1 分鐘）</span>
      </button>

      {open && (
        <div
          className="mt-2 p-3 rounded-lg text-xs leading-relaxed"
          style={{ background: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}
        >
          <p className="mb-2" style={{ color: 'var(--text-primary)' }}>
            前提：需有 Claude <strong>Pro / Max 訂閱</strong>，且自己電腦已安裝 <strong>Claude Code</strong>。
          </p>
          <p className="mb-1">
            尚未安裝 Claude Code？在終端機執行：<br />
            <code style={codeStyle}>npm i -g @anthropic-ai/claude-code</code>
          </p>

          <div className="mt-3 mb-1" style={{ color: 'var(--text-primary)' }}>步驟：</div>
          <ol className="list-decimal ml-4 space-y-1.5">
            <li>
              在<strong>自己電腦</strong>的終端機執行：
              <div className="flex items-center gap-2 mt-1">
                <code style={codeStyle}>claude setup-token</code>
                <button
                  onClick={() => copyCmd('claude setup-token')}
                  className="px-2 py-0.5 rounded cursor-pointer hover:opacity-80"
                  style={{ background: 'var(--accent-blue)', color: '#fff', fontSize: 11 }}
                >
                  {copied ? '✓ 已複製' : '複製指令'}
                </button>
              </div>
            </li>
            <li>瀏覽器會跳出 → 登入自己的 Claude 帳號並授權。</li>
            <li>
              終端機會印出一長串 <code style={codeStyle}>sk-ant-oat01-…</code>，複製它。
            </li>
            <li>貼到上方的 token 輸入框 → 點「偵測模型」選一個模型 → 「測試連線」→「儲存」。</li>
          </ol>

          <div className="mt-3 mb-1" style={{ color: 'var(--text-primary)' }}>常見問題：</div>
          <ul className="list-disc ml-4 space-y-1">
            <li>沒有 Claude Code 或沒有 Pro/Max 訂閱 → 改用「Anthropic Claude (API Key)」供應商。</li>
            <li>儲存後 24 小時 session 會過期，回設定再貼一次同一串 token 即可（token 本身長效）。</li>
            <li>⚠️ 這串 token 等同你的 Claude 帳號存取權，請只貼給你信任的服務。</li>
          </ul>
        </div>
      )}
    </div>
  );
}


// MLSettingsSection moved to ./MLPanel.tsx


// ─── 掃描設定組件 ─────────────────────────────────

const AVAILABLE_SYMBOLS = [
  'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT',
  'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT', 'MATIC/USDT',
  'BNB/USDT', 'TRX/USDT', 'TON/USDT', 'SHIB/USDT', 'LTC/USDT',
  'BCH/USDT', 'UNI/USDT', 'NEAR/USDT', 'APT/USDT', 'FIL/USDT',
];

const TIMEFRAME_OPTIONS = [
  { value: '15m', label: '15 分鐘' },
  { value: '1h', label: '1 小時' },
  { value: '4h', label: '4 小時' },
  { value: '1d', label: '1 天' },
  { value: '1w', label: '1 週' },
];

const INTERVAL_OPTIONS = [1, 2, 4, 6, 8, 12, 24];

function ScanSettingsSection() {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState('');

  const [enabled, setEnabled] = useState(true);
  const [symbols, setSymbols] = useState<string[]>([]);
  const [timeframe, setTimeframe] = useState('4h');
  const [interval, setInterval] = useState(4);
  const [threshold, setThreshold] = useState(3.0);
  const [customSymbol, setCustomSymbol] = useState('');

  useEffect(() => {
    fetchSystemSettings().then((s) => {
      setEnabled(s.scan_enabled !== false);
      if (Array.isArray(s.scan_symbols)) setSymbols(s.scan_symbols as string[]);
      if (s.scan_timeframe) setTimeframe(s.scan_timeframe as string);
      if (s.scan_interval_hours) setInterval(s.scan_interval_hours as number);
      if (s.move_threshold_pct) setThreshold(s.move_threshold_pct as number);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  const handleSave = async () => {
    setSaving(true);
    setMsg('');
    try {
      // 先檢查自訂幣種是否有數據
      const customSyms = symbols.filter((s) => !AVAILABLE_SYMBOLS.includes(s));
      if (customSyms.length > 0) {
        const { checkSymbolsData } = await import('../../services/api');
        const dataCheck = await checkSymbolsData(customSyms, timeframe);
        const noData = customSyms.filter((s) => !dataCheck[s]);
        if (noData.length > 0) {
          setMsg(`以下幣種無本地數據，請先同步：${noData.join(', ')}`);
          setSaving(false);
          return;
        }
      }
      await updateSystemSettings({
        scan_enabled: enabled,
        scan_symbols: symbols,
        scan_timeframe: timeframe,
        scan_interval_hours: interval,
        move_threshold_pct: threshold,
      });
      setMsg('設定已儲存，即時生效');
    } catch {
      setMsg('儲存失敗');
    }
    setSaving(false);
  };

  const toggleSymbol = (sym: string) => {
    setSymbols((prev) =>
      prev.includes(sym) ? prev.filter((s) => s !== sym) : [...prev, sym]
    );
  };

  const addCustomSymbol = () => {
    const sym = customSymbol.trim().toUpperCase();
    if (sym && !symbols.includes(sym)) {
      setSymbols((prev) => [...prev, sym]);
      setCustomSymbol('');
    }
  };

  if (loading) return <div style={{ color: 'var(--text-secondary)', fontSize: '13px' }}>載入中...</div>;

  const inputStyle = {
    background: 'var(--bg-tertiary)',
    color: 'var(--text-primary)',
    border: '1px solid var(--border-color)',
    borderRadius: '6px',
    padding: '6px 10px',
    fontSize: '13px',
    outline: 'none',
  };

  return (
    <div style={{ fontSize: '13px' }}>
      <h3 style={{ color: 'var(--text-primary)', fontWeight: 600, marginBottom: '16px' }}>
        自動掃描預警設定
      </h3>

      {/* 啟用開關 */}
      <div className="flex items-center gap-3 mb-4">
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={enabled}
            onChange={(e) => setEnabled(e.target.checked)}
            style={{ accentColor: 'var(--accent-blue)' }}
          />
          <span style={{ color: 'var(--text-primary)' }}>啟用自動掃描</span>
        </label>
        {!enabled && (
          <span style={{ color: 'var(--text-secondary)', fontSize: '11px' }}>
            停用後系統不會自動掃描，但仍可手動觸發
          </span>
        )}
      </div>

      {/* 掃描間隔 + 時間框架 + 波動門檻 */}
      <div className="flex gap-4 mb-4" style={{ flexWrap: 'wrap' }}>
        <div>
          <label style={{ color: 'var(--text-secondary)', fontSize: '11px', display: 'block', marginBottom: '4px' }}>
            掃描間隔
          </label>
          <select
            value={interval}
            onChange={(e) => setInterval(Number(e.target.value))}
            style={{ ...inputStyle, cursor: 'pointer' }}
          >
            {INTERVAL_OPTIONS.map((h) => (
              <option key={h} value={h}>{h} 小時</option>
            ))}
          </select>
        </div>
        <div>
          <label style={{ color: 'var(--text-secondary)', fontSize: '11px', display: 'block', marginBottom: '4px' }}>
            K 線週期
          </label>
          <select
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value)}
            style={{ ...inputStyle, cursor: 'pointer' }}
          >
            {TIMEFRAME_OPTIONS.map((tf) => (
              <option key={tf.value} value={tf.value}>{tf.label}</option>
            ))}
          </select>
        </div>
        <div>
          <label style={{ color: 'var(--text-secondary)', fontSize: '11px', display: 'block', marginBottom: '4px' }}>
            波動門檻 (%)
          </label>
          <input
            type="number"
            min={1}
            max={20}
            step={0.5}
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
            style={{ ...inputStyle, width: '80px' }}
          />
        </div>
      </div>

      {/* 掃描幣種 */}
      <div className="mb-4">
        <label style={{ color: 'var(--text-secondary)', fontSize: '11px', display: 'block', marginBottom: '6px' }}>
          掃描幣種（已選 {symbols.length} 個）
        </label>
        <div className="flex flex-wrap gap-1.5 mb-2">
          {AVAILABLE_SYMBOLS.map((sym) => {
            const active = symbols.includes(sym);
            return (
              <button
                key={sym}
                onClick={() => toggleSymbol(sym)}
                className="cursor-pointer"
                style={{
                  padding: '2px 8px',
                  borderRadius: '4px',
                  fontSize: '11px',
                  fontWeight: active ? 600 : 400,
                  background: active ? 'rgba(88,166,255,0.15)' : 'var(--bg-tertiary)',
                  color: active ? '#58a6ff' : 'var(--text-secondary)',
                  border: active ? '1px solid rgba(88,166,255,0.3)' : '1px solid var(--border-color)',
                }}
              >
                {sym.replace('/USDT', '')}
              </button>
            );
          })}
        </div>

        {/* 自訂幣種中不在預設列表的 */}
        {symbols.filter((s) => !AVAILABLE_SYMBOLS.includes(s)).length > 0 && (
          <div className="flex flex-wrap gap-1.5 mb-2">
            {symbols.filter((s) => !AVAILABLE_SYMBOLS.includes(s)).map((sym) => (
              <button
                key={sym}
                onClick={() => toggleSymbol(sym)}
                className="cursor-pointer"
                style={{
                  padding: '2px 8px',
                  borderRadius: '4px',
                  fontSize: '11px',
                  fontWeight: 600,
                  background: 'rgba(246,106,10,0.15)',
                  color: '#f66a0a',
                  border: '1px solid rgba(246,106,10,0.3)',
                }}
              >
                {sym} ×
              </button>
            ))}
          </div>
        )}

        {/* 自訂新增 */}
        <div className="flex gap-2 items-center">
          <input
            type="text"
            placeholder="自訂幣種（如 PEPE/USDT）"
            value={customSymbol}
            onChange={(e) => setCustomSymbol(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') addCustomSymbol(); }}
            style={{ ...inputStyle, width: '200px' }}
          />
          <button
            onClick={addCustomSymbol}
            className="cursor-pointer"
            style={{
              padding: '5px 12px',
              borderRadius: '6px',
              fontSize: '12px',
              background: 'var(--bg-tertiary)',
              color: 'var(--text-secondary)',
              border: '1px solid var(--border-color)',
            }}
          >
            新增
          </button>
        </div>
      </div>

      {/* 儲存 */}
      <div className="flex items-center gap-3">
        <button
          onClick={handleSave}
          disabled={saving}
          className="cursor-pointer"
          style={{
            padding: '6px 20px',
            borderRadius: '6px',
            fontSize: '13px',
            fontWeight: 600,
            background: saving ? 'var(--bg-tertiary)' : 'var(--accent-blue)',
            color: '#fff',
            border: 'none',
          }}
        >
          {saving ? '儲存中...' : '儲存設定'}
        </button>
        {msg && (
          <span style={{
            color: msg.includes('失敗') ? '#f85149' : '#3fb950',
            fontSize: '12px',
          }}>
            {msg}
          </span>
        )}
      </div>

      {/* CPU 負載警告 */}
      {interval <= 2 && symbols.length > 15 && (
        <div style={{
          background: 'rgba(248,81,73,0.08)',
          border: '1px solid rgba(248,81,73,0.2)',
          borderRadius: '6px',
          padding: '8px 12px',
          marginTop: '12px',
          fontSize: '11px',
          color: '#f85149',
        }}>
          間隔 {interval} 小時 + {symbols.length} 個幣種可能造成較高 CPU 負載。建議將間隔調整為 4 小時或減少幣種數量。
        </div>
      )}
      <p style={{ color: 'var(--text-secondary)', fontSize: '11px', marginTop: '12px' }}>
        儲存後即時生效。建議掃描間隔不低於 K 線週期（如 4h K 線搭配 4 小時間隔）。
      </p>

      {/* 特徵分析 */}
      <FeatureProfilesStatus />

      {/* 校準狀態 */}
      <CalibrationStatus />
    </div>
  );
}


/** 全量歷史特徵分析狀態 */
function FeatureProfilesStatus() {
  const [profiles, setProfiles] = useState<Record<string, any> | null>(null);
  const [computing, setComputing] = useState(false);
  const [msg, setMsg] = useState('');

  const loadProfiles = useCallback(async () => {
    try {
      const { fetchFeatureProfiles } = await import('../../services/api');
      const data = await fetchFeatureProfiles();
      if (data && Object.keys(data).length > 0) setProfiles(data);
    } catch { /* ignore */ }
  }, []);

  useEffect(() => { loadProfiles(); }, [loadProfiles]);

  const handleRecompute = async () => {
    setComputing(true);
    setMsg('');
    try {
      const { recomputeFeatureProfiles } = await import('../../services/api');
      const result = await recomputeFeatureProfiles();
      setMsg(`完成：${result.total_samples} 窗口, ${result.features_with_weight} 有效特徵`);
      await loadProfiles();
    } catch {
      setMsg('計算失敗');
    }
    setComputing(false);
  };

  const meta = profiles?._meta;

  // 排序：按 weight 降序
  const featureList = profiles
    ? Object.entries(profiles)
        .filter(([k]) => !k.startsWith('_'))
        .sort(([, a]: [string, any], [, b]: [string, any]) => (b.weight || 0) - (a.weight || 0))
    : [];

  return (
    <div style={{
      marginTop: '20px',
      borderTop: '1px solid var(--border-color)',
      paddingTop: '16px',
    }}>
      <div className="flex items-center justify-between mb-3">
        <h4 style={{ color: 'var(--text-primary)', fontWeight: 600, fontSize: '13px', margin: 0 }}>
          全量歷史特徵分析
        </h4>
        <div className="flex items-center gap-2">
          {msg && (
            <span style={{
              fontSize: '10px',
              color: msg.includes('失敗') ? '#f85149' : '#3fb950',
            }}>{msg}</span>
          )}
          <button
            onClick={handleRecompute}
            disabled={computing}
            className="cursor-pointer"
            style={{
              padding: '2px 10px',
              borderRadius: '4px',
              fontSize: '11px',
              background: computing ? 'var(--bg-tertiary)' : 'rgba(88,166,255,0.12)',
              color: computing ? 'var(--text-secondary)' : '#58a6ff',
              border: 'none',
            }}
          >
            {computing ? '分析中...' : '重新分析'}
          </button>
        </div>
      </div>

      {!meta ? (
        <div style={{
          background: 'rgba(88,166,255,0.06)',
          border: '1px solid rgba(88,166,255,0.15)',
          borderRadius: '6px',
          padding: '10px 12px',
          fontSize: '11px',
          color: 'var(--text-secondary)',
        }}>
          尚未執行全量歷史特徵分析。系統啟動時會自動分析，或按「重新分析」手動觸發。
        </div>
      ) : (
        <>
          <div className="flex gap-4 mb-3" style={{ flexWrap: 'wrap', fontSize: '11px' }}>
            <div>
              <span style={{ color: 'var(--text-secondary)' }}>歷史窗口：</span>
              <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
                {meta.total_samples?.toLocaleString()}
              </span>
            </div>
            <div>
              <span style={{ color: 'var(--text-secondary)' }}>幣種：</span>
              <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{meta.symbols_used}</span>
            </div>
            <div>
              <span style={{ color: 'var(--text-secondary)' }}>基線命中率：</span>
              <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{meta.baseline_prob}%</span>
            </div>
            <div>
              <span style={{ color: 'var(--text-secondary)' }}>建議門檻：</span>
              <span style={{ color: '#58a6ff', fontWeight: 600 }}>{meta.score_threshold}</span>
            </div>
          </div>

          {/* 特徵表格 */}
          <div style={{ maxHeight: '200px', overflowY: 'auto', fontSize: '10px' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ color: 'var(--text-secondary)', textAlign: 'left' }}>
                  <th style={{ padding: '2px 4px' }}>特徵</th>
                  <th style={{ padding: '2px 4px', textAlign: 'right' }}>IC</th>
                  <th style={{ padding: '2px 4px', textAlign: 'right' }}>一致性</th>
                  <th style={{ padding: '2px 4px', textAlign: 'right' }}>Lift</th>
                  <th style={{ padding: '2px 4px', textAlign: 'right' }}>權重</th>
                </tr>
              </thead>
              <tbody>
                {featureList.map(([name, p]: [string, any]) => {
                  const absIc = Math.abs(p.ic || 0);
                  const icColor = absIc >= 0.1 ? '#3fb950' : absIc >= 0.05 ? '#d29922' : 'var(--text-secondary)';
                  return (
                    <tr key={name} style={{ borderTop: '1px solid var(--border-color)' }}>
                      <td style={{ padding: '3px 4px', color: 'var(--text-primary)' }}>
                        {name.replace(/_/g, ' ')}
                      </td>
                      <td style={{ padding: '3px 4px', textAlign: 'right', color: icColor, fontWeight: 600 }}>
                        {p.ic?.toFixed(3)}
                      </td>
                      <td style={{ padding: '3px 4px', textAlign: 'right', color: 'var(--text-secondary)' }}>
                        {((p.ic_consistency || 0) * 100).toFixed(0)}%
                      </td>
                      <td style={{ padding: '3px 4px', textAlign: 'right', color: (p.lift || 0) > 1.2 ? '#d29922' : 'var(--text-secondary)' }}>
                        {p.lift?.toFixed(2)}x
                      </td>
                      <td style={{
                        padding: '3px 4px', textAlign: 'right', fontWeight: 600,
                        color: (p.weight || 0) >= 1.0 ? '#3fb950' : (p.weight || 0) >= 0.3 ? '#d29922' : 'var(--text-secondary)',
                      }}>
                        {p.weight?.toFixed(2)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}


/** 掃描器自適應校準狀態顯示 */
function CalibrationStatus() {
  const [cal, setCal] = useState<Record<string, any> | null>(null);
  const [loadError, setLoadError] = useState(false);
  const [resetting, setResetting] = useState(false);

  const loadCal = useCallback(async () => {
    try {
      const { fetchScannerCalibration } = await import('../../services/api');
      const data = await fetchScannerCalibration();
      setCal(data);
    } catch {
      setLoadError(true);
    }
  }, []);

  useEffect(() => { loadCal(); }, [loadCal]);

  if (loadError) return null;
  if (!cal) return (
    <div style={{ color: 'var(--text-secondary)', fontSize: '11px', marginTop: '16px' }}>
      載入校準狀態...
    </div>
  );

  const handleReset = async () => {
    setResetting(true);
    try {
      const { resetScannerCalibration } = await import('../../services/api');
      await resetScannerCalibration();
      await loadCal();
    } catch { /* ignore */ }
    setResetting(false);
  };

  const totalSamples = cal.total_samples || 0;
  const isActive = cal.calibration_active;
  const overallHitRate = cal.overall_hit_rate;
  const lastUpdated = cal.last_updated;
  const featureWeights = cal.feature_weights || {};
  const scoreThreshold = cal.score_threshold || {};
  const confBoundaries = cal.confidence_boundaries || {};
  const simThreshold = cal.similarity_threshold || {};

  // 找出有調整的特徵
  const adjustedFeatures = Object.entries(featureWeights)
    .filter(([, v]: [string, any]) => v.samples > 0 && Math.abs(v.adjusted - v.default) > 0.01)
    .sort(([, a]: [string, any], [, b]: [string, any]) => b.samples - a.samples);

  return (
    <div style={{
      marginTop: '20px',
      borderTop: '1px solid var(--border-color)',
      paddingTop: '16px',
    }}>
      <div className="flex items-center justify-between mb-3">
        <h4 style={{ color: 'var(--text-primary)', fontWeight: 600, fontSize: '13px', margin: 0 }}>
          自適應校準狀態
        </h4>
        {isActive && (
          <button
            onClick={handleReset}
            disabled={resetting}
            className="cursor-pointer"
            style={{
              padding: '2px 10px',
              borderRadius: '4px',
              fontSize: '11px',
              background: 'rgba(248,81,73,0.08)',
              color: '#f85149',
              border: '1px solid rgba(248,81,73,0.2)',
            }}
          >
            {resetting ? '重置中...' : '重置為預設'}
          </button>
        )}
      </div>

      {/* 整體統計 */}
      <div className="flex gap-4 mb-3" style={{ flexWrap: 'wrap' }}>
        <div style={{ fontSize: '11px' }}>
          <span style={{ color: 'var(--text-secondary)' }}>已驗證預警：</span>
          <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{totalSamples} 筆</span>
        </div>
        {overallHitRate != null && (
          <div style={{ fontSize: '11px' }}>
            <span style={{ color: 'var(--text-secondary)' }}>命中率：</span>
            <span style={{
              color: overallHitRate >= 50 ? '#3fb950' : overallHitRate >= 35 ? '#d29922' : '#f85149',
              fontWeight: 600,
            }}>{overallHitRate}%</span>
          </div>
        )}
        {lastUpdated && (
          <div style={{ fontSize: '11px' }}>
            <span style={{ color: 'var(--text-secondary)' }}>最後校準：</span>
            <span style={{ color: 'var(--text-primary)' }}>
              {new Date(lastUpdated).toLocaleString('zh-TW', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
            </span>
          </div>
        )}
      </div>

      {!isActive ? (
        <div style={{
          background: 'rgba(88,166,255,0.06)',
          border: '1px solid rgba(88,166,255,0.15)',
          borderRadius: '6px',
          padding: '10px 12px',
          fontSize: '11px',
          color: 'var(--text-secondary)',
        }}>
          需要至少 20 筆已驗證預警才會開始自適應校準（目前：{totalSamples} 筆）。
          系統會在每次掃描驗證後自動檢查是否可開始校準。
        </div>
      ) : (
        <>
          {/* 門檻區 */}
          <div className="flex gap-3 mb-3" style={{ flexWrap: 'wrap' }}>
            <CalThresholdBadge
              label="觸發門檻"
              defaultVal={scoreThreshold.default ?? 5.5}
              adjustedVal={scoreThreshold.adjusted ?? 5.5}
            />
            <CalThresholdBadge
              label="High 分界"
              defaultVal={confBoundaries.high_detail?.default ?? 12}
              adjustedVal={confBoundaries.high ?? 12}
            />
            <CalThresholdBadge
              label="Medium 分界"
              defaultVal={confBoundaries.medium_detail?.default ?? 8}
              adjustedVal={confBoundaries.medium ?? 8}
            />
            <CalThresholdBadge
              label="相似度門檻"
              defaultVal={simThreshold.default ?? 0.85}
              adjustedVal={simThreshold.adjusted ?? 0.85}
            />
          </div>

          {/* 特徵權重表 */}
          {adjustedFeatures.length > 0 && (
            <div style={{ marginBottom: '8px' }}>
              <div style={{ color: 'var(--text-secondary)', fontSize: '10px', marginBottom: '4px' }}>
                已調整的特徵權重
              </div>
              <div className="flex flex-wrap gap-1.5">
                {adjustedFeatures.map(([name, v]: [string, any]) => {
                  const isUp = v.adjusted > v.default;
                  return (
                    <span key={name} style={{
                      padding: '2px 6px',
                      borderRadius: '3px',
                      fontSize: '10px',
                      background: isUp ? 'rgba(63,185,80,0.1)' : 'rgba(248,81,73,0.1)',
                      color: isUp ? '#3fb950' : '#f85149',
                    }}>
                      {name.replace(/_/g, ' ')} {v.default}→{v.adjusted}
                      {v.hit_rate != null && ` (${v.hit_rate}%)`}
                    </span>
                  );
                })}
              </div>
            </div>
          )}

          {adjustedFeatures.length === 0 && (
            <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
              校準已啟用，目前所有特徵權重與預設值一致。
            </div>
          )}
        </>
      )}
    </div>
  );
}


function CalThresholdBadge({ label, defaultVal, adjustedVal }: {
  label: string; defaultVal: number; adjustedVal: number;
}) {
  const changed = Math.abs(adjustedVal - defaultVal) > 0.001;
  return (
    <div style={{
      background: 'var(--bg-tertiary)',
      borderRadius: '4px',
      padding: '3px 8px',
      fontSize: '10px',
    }}>
      <span style={{ color: 'var(--text-secondary)' }}>{label}: </span>
      {changed ? (
        <>
          <span style={{ color: 'var(--text-secondary)', textDecoration: 'line-through' }}>
            {defaultVal}
          </span>
          {' → '}
          <span style={{
            color: adjustedVal > defaultVal ? '#3fb950' : '#f85149',
            fontWeight: 600,
          }}>
            {adjustedVal}
          </span>
        </>
      ) : (
        <span style={{ color: 'var(--text-primary)' }}>{defaultVal}</span>
      )}
    </div>
  );
}


// Re-export for backward compatibility
export { loadPersistedSession } from '../../services/session';

export function clearPersistedSession() {
  sessionStorage.removeItem('asura_llm_session');
}
