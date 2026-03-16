/** LLM Session 持久化工具（使用 sessionStorage，關閉瀏覽器後自動清除） */

const STORAGE_KEY = 'asura_llm_session';

export function persistSession(sessionId: string, provider: string, modelName?: string) {
  try {
    const payload = JSON.stringify({ s: sessionId, p: provider, m: modelName, t: Date.now() });
    const encoded = btoa(unescape(encodeURIComponent(payload)));
    sessionStorage.setItem(STORAGE_KEY, encoded);
  } catch {
    // sessionStorage 不可用時靜默失敗
  }
}

export function loadPersistedSession(): { sessionId: string; provider: string; modelName?: string } | null {
  try {
    const encoded = sessionStorage.getItem(STORAGE_KEY);
    if (!encoded) return null;
    const payload = JSON.parse(decodeURIComponent(escape(atob(encoded))));
    if (Date.now() - payload.t > 24 * 60 * 60 * 1000) {
      sessionStorage.removeItem(STORAGE_KEY);
      return null;
    }
    return { sessionId: payload.s, provider: payload.p, modelName: payload.m };
  } catch {
    sessionStorage.removeItem(STORAGE_KEY);
    return null;
  }
}
