"""阿斯拉量化系統 — 動態模型探測

用 API Key 呼叫各供應商的「列出模型」API，
回傳該 Key 可存取的模型清單。
"""

from typing import Optional

import httpx
from loguru import logger


async def discover_models(
    provider: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> list[dict]:
    """
    探測供應商可用的模型清單。

    Returns:
        [{"id": "gemini-2.0-flash-lite", "name": "...", "description": "..."}, ...]
    """
    if provider == "gemini":
        return await _discover_gemini(api_key)
    elif provider == "openai":
        return await _discover_openai(api_key)
    elif provider == "claude":
        return _discover_claude_static()
    elif provider == "claude_subscription":
        return await _discover_claude_subscription()
    elif provider == "ollama":
        return await _discover_ollama(base_url or "http://localhost:11434")
    else:
        return []


# ─── Gemini ──────────────────────────────────────

# 只顯示適合聊天/function calling 的模型
_GEMINI_CHAT_KEYWORDS = {"flash", "pro", "ultra"}
_GEMINI_SKIP = {"embedding", "aqa", "imagen", "veo", "tts", "live"}


async def _discover_gemini(api_key: Optional[str]) -> list[dict]:
    """透過 google-genai SDK 列出 Gemini 可用模型"""
    from google import genai

    client = genai.Client(api_key=api_key)
    models_list = []

    try:
        for model in client.models.list():
            model_id = model.name  # e.g., "models/gemini-2.0-flash-lite"
            short_id = model_id.replace("models/", "") if model_id.startswith("models/") else model_id

            # 過濾：只保留可用於 generateContent 的聊天模型
            supported = getattr(model, "supported_generation_methods", []) or []
            if supported and "generateContent" not in supported:
                continue

            # 跳過非聊天用途模型（embedding, tts 等）
            lower_id = short_id.lower()
            if any(skip in lower_id for skip in _GEMINI_SKIP):
                continue

            display_name = getattr(model, "display_name", short_id) or short_id
            desc = getattr(model, "description", "") or ""

            models_list.append({
                "id": short_id,
                "name": display_name,
                "description": desc[:120],
            })
    except Exception as e:
        logger.error(f"Gemini 模型列表取得失敗: {e}")
        raise

    # 排序：flash-lite 優先，然後 flash，然後 pro
    def _sort_key(m: dict) -> tuple:
        mid = m["id"].lower()
        if "flash-lite" in mid:
            return (0, mid)
        if "flash" in mid:
            return (1, mid)
        if "pro" in mid:
            return (2, mid)
        return (3, mid)

    models_list.sort(key=_sort_key)
    logger.info(f"Gemini 探測到 {len(models_list)} 個可用模型")
    return models_list


# ─── OpenAI ──────────────────────────────────────

_OPENAI_CHAT_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "o1", "o3"}


def _openai_fallback_models() -> list[dict]:
    """當 API Key 權限不足時，回傳常用 OpenAI 模型讓使用者手動選"""
    return [
        {"id": "gpt-4o", "name": "GPT-4o", "description": "最強多模態模型"},
        {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "description": "快速、低成本"},
        {"id": "gpt-4-turbo", "name": "GPT-4 Turbo", "description": "128K 上下文"},
        {"id": "gpt-4", "name": "GPT-4", "description": "經典模型"},
        {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "description": "最快、最便宜"},
        {"id": "o3-mini", "name": "o3-mini", "description": "推理模型（小型）"},
        {"id": "o1-mini", "name": "o1-mini", "description": "推理模型"},
    ]


async def _discover_openai(api_key: Optional[str]) -> list[dict]:
    """透過 OpenAI API 列出可用的 GPT 聊天模型"""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)

    try:
        response = await client.models.list()
        models_list = []

        for model in response.data:
            model_id = model.id
            if not any(prefix in model_id for prefix in _OPENAI_CHAT_MODELS):
                continue
            if any(skip in model_id for skip in ["realtime", "audio", "search"]):
                continue

            models_list.append({
                "id": model_id,
                "name": model_id,
                "description": "",
            })

        def _sort_key(m: dict) -> tuple:
            mid = m["id"]
            if mid == "gpt-4o":
                return (0, mid)
            if mid.startswith("gpt-4o"):
                return (1, mid)
            if mid.startswith("o3") or mid.startswith("o1"):
                return (2, mid)
            if mid.startswith("gpt-4"):
                return (3, mid)
            return (4, mid)

        models_list.sort(key=_sort_key)
        logger.info(f"OpenAI 探測到 {len(models_list)} 個可用模型")
        return models_list

    except Exception as e:
        err_str = str(e)
        if "403" in err_str or "permission" in err_str.lower() or "scope" in err_str.lower():
            logger.warning(f"OpenAI API Key 權限不足（無法列出模型），改用預設模型清單: {e}")
            return _openai_fallback_models()
        logger.error(f"OpenAI 模型列表取得失敗: {e}")
        raise


# ─── Claude ──────────────────────────────────────

def _discover_claude_static() -> list[dict]:
    """Anthropic API 已知可用模型"""
    return [
        {"id": "claude-opus-4-20250514", "name": "Claude Opus 4", "description": "最強推理模型"},
        {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4", "description": "推理能力強，旗艦模型"},
        {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet", "description": "性價比高"},
        {"id": "claude-3-5-sonnet-20240620", "name": "Claude 3.5 Sonnet (v1)", "description": "3.5 Sonnet 初版"},
        {"id": "claude-3-5-haiku-20241022", "name": "Claude 3.5 Haiku", "description": "回應最快"},
        {"id": "claude-3-haiku-20240307", "name": "Claude 3 Haiku", "description": "最輕量，成本最低"},
    ]


async def _discover_claude_subscription() -> list[dict]:
    """動態偵測 Claude 訂閱制可用模型（透過 claude CLI）+ 靜態清單合併"""
    import asyncio

    discovered: list[dict] = []

    # 嘗試用 claude CLI 查詢可用模型
    try:
        proc = await asyncio.create_subprocess_exec(
            "claude", "--print", "--model", "sonnet",
            "-p", "List all available Claude model IDs in a simple list, one per line. Only output model IDs like claude-opus-4-6, claude-sonnet-4-6, etc.",
            "--max-turns", "0",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        text = stdout.decode().strip()

        # 解析模型 ID
        seen = set()
        for line in text.split("\n"):
            line = line.strip().strip("`").strip("- ")
            if line.startswith("claude-") and line not in seen:
                seen.add(line)
                name, desc = _claude_model_display(line)
                discovered.append({"id": line, "name": name, "description": desc})

    except Exception as e:
        logger.debug(f"Claude CLI 模型偵測失敗: {e}")

    # 合併靜態清單：CLI 偵測到的優先，再補上靜態清單中未出現的
    static = _discover_claude_subscription_static()
    discovered_ids = {m["id"] for m in discovered}
    for m in static:
        if m["id"] not in discovered_ids:
            discovered.append(m)

    return discovered if discovered else static


def _claude_model_display(model_id: str) -> tuple[str, str]:
    """根據模型 ID 生成顯示名稱與描述"""
    if "opus-4-7" in model_id:
        return f"Claude Opus 4.7 ({model_id})", "最新推理模型（需 Max 訂閱）"
    if "opus-4-6" in model_id:
        return f"Claude Opus 4.6 ({model_id})", "推理模型（需 Max 訂閱）"
    if "opus" in model_id:
        return f"Claude Opus ({model_id})", "推理模型（需 Max 訂閱）"
    if "sonnet" in model_id and "3-5" not in model_id and "3.5" not in model_id:
        return f"Claude Sonnet ({model_id})", "推理能力強，旗艦模型"
    if "3-5-sonnet" in model_id or "3.5-sonnet" in model_id:
        return f"Claude 3.5 Sonnet ({model_id})", "性價比高"
    if "haiku" in model_id:
        return f"Claude Haiku ({model_id})", "回應最快"
    return model_id, ""


def _discover_claude_subscription_static() -> list[dict]:
    """靜態模型清單：所有 Claude 模型都可選"""
    return [
        # ── Opus 系列（需 Max 訂閱）──
        {"id": "claude-opus-4-7", "name": "Claude Opus 4.7", "description": "最新最強推理模型（需 Max 訂閱）"},
        {"id": "claude-opus-4-6", "name": "Claude Opus 4.6", "description": "推理模型（需 Max 訂閱）"},
        {"id": "claude-opus-4-20250514", "name": "Claude Opus 4 (API 版)", "description": "Opus 4 固定版本"},
        # ── Sonnet 系列 ──
        {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6", "description": "推理能力強，旗艦模型"},
        {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4 (API 版)", "description": "Sonnet 4 固定版本"},
        {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet", "description": "性價比高，穩定可靠"},
        {"id": "claude-3-5-sonnet-20240620", "name": "Claude 3.5 Sonnet (v1)", "description": "3.5 Sonnet 初版"},
        # ── Haiku 系列 ──
        {"id": "claude-haiku-4-5-20251001", "name": "Claude Haiku 4.5", "description": "回應最快"},
        {"id": "claude-3-5-haiku-20241022", "name": "Claude 3.5 Haiku", "description": "快速、低成本"},
        {"id": "claude-3-haiku-20240307", "name": "Claude 3 Haiku", "description": "最輕量，成本最低"},
    ]


# ─── Ollama ──────────────────────────────────────

async def _discover_ollama(base_url: str) -> list[dict]:
    """透過 Ollama API 列出本地已下載的模型"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{base_url.rstrip('/')}/api/tags")
            resp.raise_for_status()
            data = resp.json()

        models_list = []
        for model in data.get("models", []):
            name = model.get("name", "")
            size = model.get("size", 0)
            size_gb = f"{size / 1e9:.1f}GB" if size else ""
            models_list.append({
                "id": name,
                "name": name,
                "description": size_gb,
            })

        logger.info(f"Ollama 探測到 {len(models_list)} 個本地模型")
        return models_list

    except Exception as e:
        logger.error(f"Ollama 模型列表取得失敗: {e}")
        raise
