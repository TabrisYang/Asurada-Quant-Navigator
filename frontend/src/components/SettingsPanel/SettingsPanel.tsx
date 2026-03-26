/** 阿斯拉量化系統 — 設定面板組件
 *
 * 流程：選供應商 → 輸入 API Key → 探測可用模型 → 選模型 → 測試連線（用選的模型）→ 儲存
 *
 * 核心原則：
 * - 不使用預設模型，使用者必須自己選擇
 * - 測試連線會用使用者選的模型，確認真的能用
 * - 模型清單完全來自 API 動態偵測，不寫死
 */

import { useState, useRef, useEffect, useCallback } from 'react';
import { useChartStore } from '../../stores/chartStore';
import {
  configureLLM, testLLMConnection, discoverModels, exportKnowledgePDF,
  fetchStrategies, addStrategy, updateStrategy, deleteStrategy,
  fetchSystemSettings, updateSystemSettings,
  type UserStrategy,
} from '../../services/api';
import { persistSession } from '../../services/session';
import type { LLMProvider } from '../../types';
import MLPanel from './MLPanel';
import PredictionDashboard from '../PredictionDashboard/PredictionDashboard';

const LLM_PROVIDERS: { id: LLMProvider; name: string; requiresKey: boolean; desc: string }[] = [
  { id: 'openai', name: 'OpenAI (GPT-4/4o)', requiresKey: true, desc: '最成熟的 Function Calling 支援' },
  { id: 'gemini', name: 'Google Gemini', requiresKey: true, desc: '免費額度較高' },
  { id: 'claude', name: 'Anthropic Claude', requiresKey: true, desc: '推理能力強' },
  { id: 'claude_subscription', name: 'Claude 訂閱制', requiresKey: false, desc: '使用 Claude Code 訂閱額度（需已登入）' },
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
  const [activeTab, setActiveTab] = useState<'llm' | 'export' | 'strategies' | 'ml' | 'predictions'>('llm');
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
        selectedProvider?.requiresKey ? apiKey : undefined,
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
        selectedProvider?.requiresKey ? apiKey : undefined,
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
        selectedProvider?.requiresKey ? apiKey : undefined,
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
        </div>

        {activeTab === 'strategies' && <StrategySection />}
        {activeTab === 'export' && <ExportImportSection onClose={onClose} />}
        {activeTab === 'ml' && <MLPanel />}
        {activeTab === 'predictions' && <PredictionDashboard />}

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
            <StepLabel step={2} text="偵測可用模型" />
            <p className="text-xs mb-2" style={{ color: 'var(--text-secondary)' }}>
              系統將自動使用 Claude Code 的訂閱憑證，無需輸入 API Key
            </p>
            <button
              onClick={handleDiscoverModels}
              disabled={!canDiscover}
              className="px-4 py-2 rounded-lg text-sm cursor-pointer transition-opacity hover:opacity-80 whitespace-nowrap disabled:opacity-40"
              style={{ background: 'var(--accent-blue)', color: '#fff' }}
            >
              {loadingModels ? '偵測中...' : '偵測模型'}
            </button>
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


// MLSettingsSection moved to ./MLPanel.tsx


// Re-export for backward compatibility
export { loadPersistedSession } from '../../services/session';

export function clearPersistedSession() {
  sessionStorage.removeItem('asura_llm_session');
}
