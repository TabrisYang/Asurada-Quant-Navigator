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
        # api_key 在訂閱模式承載「選填」的 per-user OAuth token；無則走本機登入
        return await _discover_claude_subscription(oauth_token=api_key or None)
    elif provider == "codex_subscription":
        return await _discover_codex_subscription()
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


# 別名 → CLI 永遠解析成「該等級最新版本」
_CLI_ALIASES = ["opus", "sonnet", "haiku", "fable"]
# 偵測結果快取（避免每次開設定都付費跑 CLI）；只在 live 偵測成功時才寫入
_SUB_CACHE: dict = {"data": None, "ts": 0.0}
_SUB_CACHE_TTL = 6 * 3600


async def _resolve_alias(alias: str, oauth_token: Optional[str] = None) -> Optional[str]:
    """跑一次最小 CLI 查詢，從 modelUsage 取得別名實際解析到的最新模型 ID。

    比「叫模型自己列出 ID」可靠：modelUsage 的 key 就是 CLI 真正路由到的具體模型，
    未來 CLI/訂閱拿到 4.8、4.9 會自動跟上，不需改程式碼。

    oauth_token 有值時用 CLAUDE_CODE_OAUTH_TOKEN 注入（部署到無本機登入的伺服器時，
    用呼叫者自己的訂閱憑證解析）；無則沿用本機登入。
    """
    import asyncio
    import json
    import os
    import shutil
    import tempfile

    cfg_dir = None
    env = None
    if oauth_token:
        cfg_dir = tempfile.mkdtemp(prefix="claude_cfg_")
        env = {**os.environ, "CLAUDE_CODE_OAUTH_TOKEN": oauth_token, "CLAUDE_CONFIG_DIR": cfg_dir}

    try:
        proc = await asyncio.create_subprocess_exec(
            "claude", "--print", "--output-format", "json",
            "--model", alias, "--tools", "", "-p", "hi",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=tempfile.gettempdir(),  # 避開專案 CLAUDE.md，降低 token 成本
            env=env,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        data = json.loads(stdout.decode())
        ids = [k for k in (data.get("modelUsage") or {}) if k.startswith("claude-")]
        return ids[0] if ids else None
    except Exception as e:
        logger.debug(f"別名 {alias} 解析失敗: {e}")
        return None
    finally:
        if cfg_dir:
            shutil.rmtree(cfg_dir, ignore_errors=True)


async def _discover_claude_subscription(oauth_token: Optional[str] = None) -> list[dict]:
    """動態偵測 Claude 訂閱制可用模型（透過 claude CLI 解析別名）+ 靜態清單合併"""
    import asyncio
    import time

    now = time.monotonic()
    if _SUB_CACHE["data"] and now - _SUB_CACHE["ts"] < _SUB_CACHE_TTL:
        return _SUB_CACHE["data"]

    # 併發解析三個別名，各取得實際路由到的最新具體模型 ID
    resolved = await asyncio.gather(*[_resolve_alias(a, oauth_token) for a in _CLI_ALIASES])

    discovered: list[dict] = []
    seen: set[str] = set()
    for model_id in resolved:
        if model_id and model_id not in seen:
            seen.add(model_id)
            name, desc = _claude_model_display(model_id)
            discovered.append({"id": model_id, "name": name, "description": desc})

    # 合併靜態清單：CLI 偵測到的優先，再補上靜態清單中未出現的
    static = _discover_claude_subscription_static()
    for m in static:
        if m["id"] not in seen:
            discovered.append(m)

    result = discovered if discovered else static
    # 只在 live 偵測成功（至少解析到一個具體 ID）時才寫快取，避免污染
    if seen:
        _SUB_CACHE["data"] = result
        _SUB_CACHE["ts"] = now
    return result


# 家族 → (顯示前綴, 描述)
_FAM = {
    "opus": ("Claude Opus", "推理模型（需 Max 訂閱）"),
    "sonnet": ("Claude Sonnet", "推理能力強，旗艦模型"),
    "haiku": ("Claude Haiku", "回應最快"),
    "fable": ("Claude Fable", "最強旗艦模型（高階，方案需支援）"),
    "mythos": ("Claude Mythos", "最強旗艦模型（Project Glasswing）"),
}


def _claude_model_display(model_id: str) -> tuple[str, str]:
    """根據模型 ID 生成顯示名稱與描述（版號無關，未來新版自動正確渲染）"""
    import re

    mid = model_id.lower()
    fam_m = re.search(r"(opus|sonnet|haiku|fable|mythos)", mid)
    if not fam_m:
        return model_id, ""
    fam = fam_m.group(1)
    base, desc = _FAM[fam]

    # 舊式版號在家族前：claude-3-5-sonnet-... / claude-3-haiku-...
    pre = re.search(rf"claude-(\d+)(?:-(\d+))?-{fam}", mid)
    if pre:
        ver = pre.group(1) + (f".{pre.group(2)}" if pre.group(2) else "")
    else:  # 新式版號在家族後：opus-4-7 / haiku-4-5-<date>（minor 限 1-2 位數，排除日期）
        new = re.search(rf"{fam}-(\d+)(?:-(\d{{1,2}})(?!\d))?", mid)
        ver = (new.group(1) + (f".{new.group(2)}" if new.group(2) else "")) if new else ""

    label = f"{base} {ver}".strip()
    return f"{label} ({model_id})", desc


def _discover_claude_subscription_static() -> list[dict]:
    """靜態模型清單：所有 Claude 模型都可選"""
    return [
        # ── Fable 系列（高階旗艦，方案需支援）──
        {"id": "claude-fable-5", "name": "Claude Fable 5", "description": "最強旗艦模型（高階，方案需支援）"},
        # ── Opus 系列（需 Max 訂閱）──
        {"id": "claude-opus-4-8", "name": "Claude Opus 4.8", "description": "最新最強推理模型（需 Max 訂閱）"},
        {"id": "claude-opus-4-7", "name": "Claude Opus 4.7", "description": "推理模型（需 Max 訂閱）"},
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


# ─── Codex（ChatGPT 訂閱制）──────────────────────────

# 靜態 fallback（cache 不存在時用；名單會過時，僅保底）
_CODEX_STATIC = [
    {"id": "gpt-5.6-terra", "name": "GPT-5.6-Terra", "description": "均衡主力模型"},
    {"id": "gpt-5.6-luna", "name": "GPT-5.6-Luna", "description": "快速省額度"},
    {"id": "gpt-5.5", "name": "GPT-5.5", "description": "前沿推理模型"},
    {"id": "gpt-5.4-mini", "name": "GPT-5.4-Mini", "description": "輕量低耗"},
]

# 不適合對話用途的內部模型（不列給使用者）
_CODEX_SKIP_SLUGS = ("auto-review",)


def _codex_models_cache_path():
    """Codex app/CLI 會把帳號可用模型清單同步到本機 cache。"""
    from pathlib import Path as _P
    import os
    codex_home = os.environ.get("CODEX_HOME") or str(_P.home() / ".codex")
    return _P(codex_home) / "models_cache.json"


async def _discover_codex_subscription() -> list[dict]:
    """列出 ChatGPT 訂閱制（Codex）可用模型。

    首選：讀本機 ~/.codex/models_cache.json — 這是 Codex app/CLI 向帳號
    同步下來的「此帳號實際可用」最新清單（含 GPT-5.6 等新版），零額度成本、
    永遠跟上 OpenAI 發版。cache 不存在（從未跑過 codex）→ 靜態保底清單。
    """
    import json

    cache_path = _codex_models_cache_path()
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        models = []
        for m in data.get("models", []):
            slug = m.get("slug") or ""
            if not slug or any(skip in slug for skip in _CODEX_SKIP_SLUGS):
                continue
            models.append({
                "id": slug,
                "name": m.get("display_name") or slug,
                "description": (m.get("description") or "")[:80],
            })
        if models:
            fetched = str(data.get("fetched_at", ""))[:10]
            logger.info(f"Codex 模型清單：讀取帳號快取 {len(models)} 個（同步於 {fetched}）")
            return models
    except FileNotFoundError:
        logger.info("Codex models_cache.json 不存在（未跑過 codex），回靜態清單")
    except Exception as e:
        logger.warning(f"Codex 模型快取讀取失敗，回靜態清單: {e}")
    return list(_CODEX_STATIC)
