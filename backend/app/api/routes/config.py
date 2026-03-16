"""阿斯拉量化系統 — 設定路由（含 API Key 安全管理 + 動態模型探測 + 用量追蹤 + 自訂策略庫）"""

import re
from typing import Optional
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

from app.core.llm.adapter import create_adapter
from app.core.llm.model_discovery import discover_models
from app.core.security.key_manager import key_manager
from app.core.usage_tracker import usage_tracker
from app.core import user_strategies
from app.models.schemas import LLMConfigRequest, LLMProvider

router = APIRouter()

_OLLAMA_URL_RE = re.compile(r"^https?://[\w.\-]+(:\d+)?/?$")


def _validate_ollama_url(url: str) -> str:
    """驗證 Ollama base_url，防止 SSRF"""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="Ollama URL 必須是 http 或 https")
    if not parsed.hostname:
        raise HTTPException(status_code=400, detail="Ollama URL 格式不正確")
    # 只允許 localhost / 127.x / 私有 IP / 自訂域名
    host = parsed.hostname.lower()
    allowed = (
        host in ("localhost", "127.0.0.1", "0.0.0.0", "::1")
        or host.startswith("192.168.")
        or host.startswith("10.")
        or host.startswith("172.")
    )
    if not allowed and not _OLLAMA_URL_RE.match(url):
        raise HTTPException(status_code=400, detail="Ollama URL 僅允許本機或私有網路位址")
    return url.rstrip("/")


@router.post("/llm")
async def configure_llm(request: LLMConfigRequest):
    """
    設定 LLM 供應商和 API Key。

    API Key 會被加密儲存在後端記憶體中，回傳 session_id。
    後續所有對話請求只需帶 session_id，不再傳輸明文 Key。
    """
    provider = request.provider.value

    # Ollama 不需要 Key
    if provider == "ollama":
        base = _validate_ollama_url(request.base_url or "http://localhost:11434")
        session_id = key_manager.store_key(
            api_key="",
            provider=provider,
            base_url=base,
        )
        return {
            "status": "ok",
            "provider": provider,
            "session_id": session_id,
            "message": f"已設定 LLM 供應商為 {provider}（無需 API Key）",
        }

    # 需要 API Key 的供應商
    if not request.api_key:
        raise HTTPException(status_code=400, detail=f"{provider} 需要 API Key")

    # 加密儲存，回傳 session_id
    session_id = key_manager.store_key(
        api_key=request.api_key,
        provider=provider,
        model_name=request.model_name,
    )

    return {
        "status": "ok",
        "provider": provider,
        "session_id": session_id,
        "message": f"API Key 已加密儲存，session 有效期 24 小時",
    }


@router.post("/llm/test")
async def test_llm_connection(request: LLMConfigRequest):
    """
    測試 LLM 連線。

    支援兩種方式：
    1. 帶 session_id → 從加密儲存中取出 Key 測試
    2. 帶 api_key → 直接用明文 Key 測試（一次性，不儲存）
    """
    provider = request.provider.value

    api_key = None
    session_expired = False
    if request.session_id:
        api_key = key_manager.get_key(request.session_id)
        if not api_key:
            session_expired = True
    if not api_key and request.api_key:
        api_key = request.api_key

    if not api_key and provider != "ollama":
        if session_expired:
            raise HTTPException(
                status_code=401,
                detail="SESSION_EXPIRED:先前的 API Key 已失效，請重新輸入 API Key",
            )
        raise HTTPException(status_code=400, detail="需要 API Key 或有效的 session_id")

    # 嘗試建立 adapter 並發送測試訊息
    try:
        adapter = create_adapter(
            provider=provider,
            api_key=api_key,
            model_name=request.model_name,
            base_url=request.base_url,
        )

        # 發送一個簡短的測試訊息
        response = await adapter.chat(
            messages=[{"role": "user", "content": "Hi, 回覆 OK 即可"}],
        )

        # Gemini adapter 回傳的訊息可能包含配額錯誤提示
        if response.message and "額度已用完" in response.message:
            return {
                "status": "ok",
                "provider": provider,
                "message": response.message,
                "warning": "quota_exhausted",
            }

        if response.message:
            model_info = request.model_name or "未指定模型"
            return {
                "status": "ok",
                "provider": provider,
                "model": request.model_name,
                "message": f"✓ 模型 {model_info} 連線測試成功",
                "test_response": response.message[:100],
            }
        else:
            return {
                "status": "ok",
                "provider": provider,
                "message": "已收到回應（可能為空）",
            }

    except Exception as e:
        err_str = str(e)
        logger.error(f"LLM 連線測試失敗 ({provider}): {e}")

        # 分類錯誤訊息
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
            raise HTTPException(
                status_code=429,
                detail="模型免費額度已用完。請選擇其他模型（如 Gemini 2.0 Flash Lite），或稍後再試。",
            )
        elif "401" in err_str or "UNAUTHENTICATED" in err_str or "invalid" in err_str.lower():
            raise HTTPException(
                status_code=401,
                detail="API Key 無效或已過期，請檢查後重新輸入。",
            )
        elif "403" in err_str or "PERMISSION_DENIED" in err_str:
            raise HTTPException(
                status_code=403,
                detail="API Key 權限不足，請確認已啟用對應的 API 服務。",
            )
        else:
            raise HTTPException(
                status_code=502,
                detail=f"連線測試失敗: {err_str[:200]}",
            )


@router.get("/llm/session/{session_id}")
async def get_session_info(session_id: str):
    """查詢 session 狀態（不回傳 API Key）"""
    info = key_manager.get_session_info(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="Session 不存在或已過期")
    return info


@router.delete("/llm/session/{session_id}")
async def revoke_session(session_id: str):
    """撤銷（登出）一個 session"""
    success = key_manager.revoke_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session 不存在")
    return {"status": "ok", "message": "Session 已撤銷"}


@router.get("/llm/providers")
async def list_llm_providers():
    """列出支援的 LLM 供應商"""
    return {
        "providers": [
            {
                "id": "openai",
                "name": "OpenAI (GPT-4/4o)",
                "requires_key": True,
                "description": "最成熟的 Function Calling 支援",
            },
            {
                "id": "gemini",
                "name": "Google Gemini",
                "requires_key": True,
                "description": "免費額度較高",
            },
            {
                "id": "claude",
                "name": "Anthropic Claude",
                "requires_key": True,
                "description": "推理能力強",
            },
            {
                "id": "ollama",
                "name": "本地 Ollama",
                "requires_key": False,
                "description": "完全免費，無需 API Key",
            },
        ]
    }


@router.post("/llm/models")
async def list_available_models(request: LLMConfigRequest):
    """
    用 API Key 動態探測該供應商可用的模型清單。

    流程：輸入 Key → 呼叫供應商 API → 回傳可用模型列表
    使用者再從中選擇一個。
    """
    provider = request.provider.value

    api_key = None
    session_expired = False
    if request.session_id:
        api_key = key_manager.get_key(request.session_id)
        if not api_key:
            session_expired = True
    if not api_key and request.api_key:
        api_key = request.api_key

    if not api_key and provider != "ollama":
        if session_expired:
            raise HTTPException(
                status_code=401,
                detail="SESSION_EXPIRED:先前的 API Key 已失效（後端重啟或 Session 過期），請重新輸入 API Key",
            )
        raise HTTPException(status_code=400, detail="需要 API Key 才能探測可用模型")

    try:
        models = await discover_models(
            provider=provider,
            api_key=api_key,
            base_url=request.base_url,
        )
        return {
            "status": "ok",
            "provider": provider,
            "models": models,
            "total": len(models),
        }
    except Exception as e:
        err_str = str(e)
        logger.error(f"探測模型失敗 ({provider}): {e}")
        if "401" in err_str or "UNAUTHENTICATED" in err_str or "invalid" in err_str.lower():
            raise HTTPException(status_code=401, detail="API Key 無效，無法探測可用模型")
        raise HTTPException(status_code=502, detail=f"探測模型失敗: {err_str[:200]}")


# ─── Token 用量查詢端點 ────────────────────────

@router.post("/usage/summary")
async def get_usage_summary(request: LLMConfigRequest):
    """
    查詢 API Key 的累計 token 用量摘要。

    回傳：今日、本月、歷史總計的 token 數量與費用。
    注意：費用為估算值，實際帳單請查閱供應商後台。
    """
    # 解析 API Key
    api_key = None
    if request.session_id:
        api_key = key_manager.get_key(request.session_id)
    if not api_key and request.api_key:
        api_key = request.api_key

    if not api_key:
        raise HTTPException(status_code=400, detail="需要有效的 session 才能查詢用量")

    summary = usage_tracker.get_summary(api_key)
    return {"status": "ok", **summary}


@router.post("/usage/daily")
async def get_usage_daily(request: LLMConfigRequest):
    """查詢最近 30 天的每日用量明細"""
    api_key = None
    if request.session_id:
        api_key = key_manager.get_key(request.session_id)
    if not api_key and request.api_key:
        api_key = request.api_key

    if not api_key:
        raise HTTPException(status_code=400, detail="需要有效的 session 才能查詢用量")

    daily = usage_tracker.get_daily_breakdown(api_key, days=30)
    return {"status": "ok", "daily": daily}


# ─── 使用者自訂分析策略庫 ─────────────────────────


class StrategyRequest(BaseModel):
    title: str
    content: str
    enabled: bool = True


class StrategyUpdateRequest(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    enabled: Optional[bool] = None


@router.get("/strategies")
async def list_strategies():
    """列出所有自訂分析策略"""
    return {"status": "ok", "strategies": user_strategies.list_strategies()}


@router.post("/strategies")
async def add_strategy(req: StrategyRequest):
    """新增一條自訂分析策略"""
    result = user_strategies.add_strategy(req.title, req.content, req.enabled)
    return {"status": "ok", "strategy": result}


@router.put("/strategies/{strategy_id}")
async def update_strategy(strategy_id: str, req: StrategyUpdateRequest):
    """更新一條自訂分析策略"""
    result = user_strategies.update_strategy(
        strategy_id, title=req.title, content=req.content, enabled=req.enabled,
    )
    if not result:
        raise HTTPException(status_code=404, detail="策略不存在")
    return {"status": "ok", "strategy": result}


@router.delete("/strategies/{strategy_id}")
async def delete_strategy(strategy_id: str):
    """刪除一條自訂分析策略"""
    ok = user_strategies.delete_strategy(strategy_id)
    if not ok:
        raise HTTPException(status_code=404, detail="策略不存在")
    return {"status": "ok"}
