"""阿斯拉量化系統 — 設定路由（含 API Key 安全管理 + 動態模型探測 + 用量追蹤 + 自訂策略庫）"""

import json
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

from app.core.config.settings import settings as _app_settings
from app.core.llm.adapter import create_adapter
from app.core.llm.model_discovery import discover_models
from app.core.security.key_manager import key_manager
from app.core.usage_tracker import usage_tracker
from app.core import user_strategies
from app.models.schemas import LLMConfigRequest

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

    # 不需要 API Key 的供應商
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

    if provider == "claude_subscription":
        # api_key 在此「選填」：使用者各自貼上自己的 OAuth token（claude setup-token 產生），
        # 則計入該使用者自己的訂閱；留空則用本機 Claude Code 登入憑證（站長本機模式）。
        # token 沿用 key_manager 的 Fernet 加密 + 24h TTL，僅存記憶體、不落地、不寫 log。
        has_token = bool(request.api_key)
        session_id = key_manager.store_key(
            api_key=request.api_key or "",
            provider=provider,
            model_name=request.model_name,
        )
        return {
            "status": "ok",
            "provider": provider,
            "session_id": session_id,
            "message": (
                "已設定為 Claude 訂閱制（使用你提供的個人訂閱 token）"
                if has_token
                else "已設定為 Claude 訂閱制（使用本機 Claude Code 登入憑證）"
            ),
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
        "message": "API Key 已加密儲存，session 有效期 24 小時",
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

    if not api_key and provider not in ("ollama", "claude_subscription"):
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
                "description": "最成熟的 Function Calling 支援（僅支援 API Key，無訂閱串接）",
            },
            {
                "id": "gemini",
                "name": "Google Gemini",
                "requires_key": True,
                "description": "免費額度較高（僅支援 API Key，無訂閱串接）",
            },
            {
                "id": "claude",
                "name": "Anthropic Claude (API Key)",
                "requires_key": True,
                "description": "推理能力強，按 token 計費",
            },
            {
                "id": "claude_subscription",
                "name": "Claude 訂閱制",
                "requires_key": False,
                "description": (
                    "用 Claude 訂閱額度（Pro/Max）。本機留空 token 即用本機 Claude Code 登入；"
                    "雲端/他人使用時，各自貼上自己 `claude setup-token` 產生的 token，計入自己的訂閱。"
                ),
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

    if not api_key and provider not in ("ollama", "claude_subscription"):
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


# ═══════════════════════════════════════════════════════
#  系統通用設定（教學模式等）
# ═══════════════════════════════════════════════════════

_SYSTEM_SETTINGS_FILE = Path(_app_settings.db_path) / "system_settings.json"


_SCAN_SETTINGS_DEFAULTS = {
    "scan_enabled": True,
    "scan_symbols": [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT",
        "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", "MATIC/USDT",
    ],
    "scan_timeframe": "4h",
    "scan_interval_hours": 4,
    "move_threshold_pct": 3.0,
}


def load_system_settings() -> dict:
    """讀取系統通用設定（外部模組可直接呼叫）"""
    defaults = {"teaching_mode": False, **_SCAN_SETTINGS_DEFAULTS}
    if _SYSTEM_SETTINGS_FILE.exists():
        try:
            with open(_SYSTEM_SETTINGS_FILE) as f:
                data = json.load(f)
            defaults.update(data)
        except Exception:
            pass
    return defaults


@router.get("/system-settings")
async def get_system_settings():
    """取得系統通用設定"""
    return {"status": "ok", "settings": load_system_settings()}


@router.post("/clear-session-cache")
async def clear_session_cache():
    """清除 Claude CLI 的 session 快取，釋放磁碟空間並重置 prompt cache"""
    import asyncio
    try:
        proc = await asyncio.create_subprocess_exec(
            "claude", "--clear-conversation-history",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
        if proc.returncode == 0:
            return {"status": "ok", "message": "Session 快取已清除"}
        else:
            err = stderr.decode().strip() if stderr else "未知錯誤"
            logger.warning(f"清除 session cache 失敗: {err}")
            return {"status": "ok", "message": f"清除指令已執行（returncode={proc.returncode}）"}
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="未安裝 Claude CLI，無法清除 session cache")
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="清除 session cache 超時")
    except Exception as e:
        logger.error(f"清除 session cache 失敗: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/system-settings")
async def update_system_settings(body: dict):
    """更新系統通用設定（含掃描設定）"""
    current = load_system_settings()

    if "teaching_mode" in body:
        current["teaching_mode"] = bool(body["teaching_mode"])

    # 掃描設定
    if "scan_enabled" in body:
        current["scan_enabled"] = bool(body["scan_enabled"])
    if "scan_symbols" in body:
        symbols = body["scan_symbols"]
        if isinstance(symbols, list) and len(symbols) <= 30:
            current["scan_symbols"] = [s.strip() for s in symbols if isinstance(s, str) and s.strip()]
    if "scan_timeframe" in body:
        tf = body["scan_timeframe"]
        if tf in ("15m", "1h", "4h", "1d", "1w"):
            current["scan_timeframe"] = tf
    if "scan_interval_hours" in body:
        interval = body["scan_interval_hours"]
        if isinstance(interval, (int, float)) and 1 <= interval <= 24:
            # CPU 保護：幣種多時強制最低間隔
            n_symbols = len(current.get("scan_symbols", []))
            min_interval = 2 if n_symbols > 20 else 1
            current["scan_interval_hours"] = max(int(interval), min_interval)
    if "move_threshold_pct" in body:
        threshold = body["move_threshold_pct"]
        if isinstance(threshold, (int, float)) and 1.0 <= threshold <= 20.0:
            current["move_threshold_pct"] = float(threshold)

    _SYSTEM_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_SYSTEM_SETTINGS_FILE, "w") as f:
        json.dump(current, f, ensure_ascii=False)

    # 掃描設定變更 → 即時重啟掃描任務
    scan_keys = {"scan_enabled", "scan_symbols", "scan_timeframe", "scan_interval_hours", "move_threshold_pct"}
    if scan_keys & body.keys():
        _restart_scan_task()

    return {"status": "ok", "settings": current}


def _restart_scan_task():
    """取消舊的掃描任務並立即重建，使設定即時生效。"""
    import asyncio
    try:
        from app.main import app_state
        old_task = app_state.get("scan_task")
        factory = app_state.get("scan_factory")
        if old_task and factory:
            old_task.cancel()
            new_task = asyncio.create_task(factory())
            app_state["scan_task"] = new_task
            logger.info("[掃描設定] 已重啟掃描任務，新設定即時生效")
    except Exception as e:
        logger.warning(f"[掃描設定] 重啟掃描任務失敗: {e}")


# ─── 掃描器校準 API ──────────────────────────


@router.get("/scanner-calibration")
async def get_scanner_calibration():
    """取得掃描器自適應校準狀態。"""
    from app.core.scanner_calibrator import scanner_calibrator
    cal = scanner_calibrator.get_active_calibrations()
    return {"status": "ok", "calibration": cal}


@router.post("/scanner-calibration/reset")
async def reset_scanner_calibration():
    """重置所有校準值為預設。"""
    from app.core.scanner_calibrator import scanner_calibrator
    scanner_calibrator.reset_all()
    return {"status": "ok", "message": "已重置為預設值"}


@router.put("/scanner-calibration/{cal_id}/override")
async def override_calibration(cal_id: int, body: dict):
    """用戶手動覆寫校準值。"""
    from app.core.scanner_calibrator import scanner_calibrator
    result = scanner_calibrator.set_override(cal_id, body.get("value"))
    return {"status": "ok", "calibration": result}


# ─── 全量歷史特徵分析 API ──────────────────────────


@router.get("/feature-profiles")
async def get_feature_profiles():
    """取得全量歷史特徵分析結果。"""
    from app.core.scanner_feature_profiler import scanner_feature_profiler
    profiles = scanner_feature_profiler.get_feature_profiles()
    return {"status": "ok", "profiles": profiles}


@router.post("/feature-profiles/recompute")
async def recompute_feature_profiles():
    """重新計算全量歷史特徵分析。"""
    from app.core.scanner_feature_profiler import scanner_feature_profiler
    result = scanner_feature_profiler.compute_all_profiles()
    return {"status": "ok", "result": result}
