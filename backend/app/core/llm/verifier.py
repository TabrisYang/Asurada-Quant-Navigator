"""阿斯拉量化系統 — LLM 回答自動覆核（低一階模型交叉檢查）

在 fact_checker（純 regex 數值比對）與 mechanical_audit（純 Python 審查）之上的第三層：
用比主模型低一個家族的便宜模型，交叉檢查四類現有規則層抓不到的錯誤：
  number     — 引用數字與系統實際數據不符
  direction  — 方向結論（看多/看空）與其引用的數據明顯矛盾
  fabricated — 引用了系統數據中不存在、也推算不出的具體數據
  logic      — 推理鏈前後自相矛盾

成本控制：
- 覆核模型自動降家族（Opus→Sonnet→Haiku 等），可用 settings.verify_model_override 覆寫
- 輸入只送「回答全文 + ~3KB 數據摘要」，不重送 200KB chart_state
- 短文 / 無數字 / 閒聊 intent 直接跳過（should_verify）

設計原則（與 fact_checker / mechanical_audit 一致）：事後標註、不阻擋、graceful —
任何失敗只 log 不上拋，絕不影響主回答。
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Optional

from loguru import logger

from app.core.config.settings import settings

# ─── 降階對照 ───────────────────────────────────────────────

# claude 家族順序（訂閱 CLI 用別名，保證解析到帳號可用的該家族最新版）
_CLAUDE_FAMILY_DOWN = {
    "fable": "sonnet",
    "mythos": "sonnet",
    "opus": "sonnet",
    "sonnet": "haiku",
    "haiku": "haiku",  # 已最低階：同級交叉檢查仍有價值
}
_CLAUDE_FAMILY_RE = re.compile(r"(opus|sonnet|haiku|fable|mythos)")

# claude API 版用具體 model ID（API 不吃別名）
_CLAUDE_API_DOWN = {
    "opus": "claude-sonnet-4-20250514",
    "fable": "claude-sonnet-4-20250514",
    "mythos": "claude-sonnet-4-20250514",
    "sonnet": "claude-3-5-haiku-20241022",
    "haiku": "claude-3-haiku-20240307",
}

_OPENAI_DOWN = {
    "gpt-4o": "gpt-4o-mini",
    "gpt-4.1": "gpt-4.1-mini",
    "o3": "o4-mini",
    "o1": "o3-mini",
}


def pick_verifier_model(provider: str, main_model: str, override: str = "") -> str:
    """挑覆核模型：override 優先；否則依 provider 降一個家族/檔次。"""
    if override:
        return override
    mid = (main_model or "").lower()

    if provider == "claude_subscription":
        m = _CLAUDE_FAMILY_RE.search(mid)
        return _CLAUDE_FAMILY_DOWN.get(m.group(1), "sonnet") if m else "sonnet"

    if provider == "claude":
        m = _CLAUDE_FAMILY_RE.search(mid)
        return _CLAUDE_API_DOWN.get(m.group(1), "claude-3-5-haiku-20241022") if m else "claude-3-5-haiku-20241022"

    if provider == "openai":
        if "mini" in mid or "nano" in mid:
            return main_model
        for prefix, small in _OPENAI_DOWN.items():
            if mid.startswith(prefix):
                return small
        return "gpt-4o-mini"

    if provider == "gemini":
        if "lite" in mid:
            return main_model
        if "pro" in mid:
            return mid.replace("pro", "flash")
        if "flash" in mid:
            return f"{main_model}-lite"
        return "gemini-2.0-flash-lite"

    # ollama 等本地模型：無成本差，同模型自我覆核
    return main_model


# ─── 數據摘要（覆核的比對基準，控制在 ~3KB）───────────────────

def _fmt_num(v: Any) -> Optional[str]:
    try:
        return str(round(float(v), 4))
    except (TypeError, ValueError):
        return None


def build_data_digest(chart_state: Optional[dict], exec_result: Optional[dict],
                      max_bytes: int = 3072) -> str:
    """從 chart_state / exec_result 抽關鍵實際值，組成緊湊 key=value 摘要。"""
    cs = chart_state or {}
    lines: list[str] = []

    for key in ("symbol", "timeframe", "currentPrice"):
        if cs.get(key) is not None:
            lines.append(f"{key}={cs[key]}")

    regime = cs.get("currentRegime")
    if isinstance(regime, dict) and regime.get("regime"):
        lines.append(f"regime={regime['regime']}")

    for name, val in (cs.get("indicatorValues") or {}).items():
        s = _fmt_num(val)
        if s is not None:
            lines.append(f"{name}={s}")

    ext = cs.get("external_signals") or {}
    deriv = ext.get("derivatives") or {}
    sent = ext.get("sentiment") or {}
    if deriv.get("funding_rate_pct") is not None:
        lines.append(f"funding_rate_pct={deriv['funding_rate_pct']}")
    if deriv.get("global_long_short_ratio") is not None:
        lines.append(f"global_long_short_ratio={deriv['global_long_short_ratio']}")
    if sent.get("fear_greed_value") is not None:
        lines.append(f"fear_greed_value={sent['fear_greed_value']}")

    ra = cs.get("recent_accuracy") or {}
    wr = ra.get("win_rate_30d") or ra.get("win_rate")
    if wr is not None:
        lines.append(f"win_rate_30d={wr}")

    dp = cs.get("donchian_position_pct")
    if dp is not None:
        lines.append(f"donchian_position_pct={dp}")

    # 回測數字重用 fact_checker 的抽取（fact_checker 不 import core/llm，無循環）
    try:
        from app.core.fact_checker import _extract_backtest_facts
        for k, v in _extract_backtest_facts(exec_result).items():
            lines.append(f"backtest_{k}={v}")
    except Exception:
        pass

    out: list[str] = []
    size = 0
    for line in lines:
        size += len(line.encode("utf-8")) + 1
        if size > max_bytes:
            out.append("...(digest truncated)")
            break
        out.append(line)
    return "\n".join(out)


# ─── 覆核 prompt ─────────────────────────────────────────────

_VERIFY_SYSTEM_PROMPT = """你是量化分析報告的覆核員。對照【系統實際數據】檢查【待覆核回答】，只找四類錯誤：
1. number：回答引用的數字與系統實際數據不符（超出合理捨入範圍）
2. direction：方向結論（看多/看空/中性）與其引用的數據明顯矛盾
3. fabricated：引用了系統數據中不存在、也不可能由其推算出的具體數據，且回答宣稱該數據來自系統
4. logic：推理鏈內部自相矛盾（前段結論與後段引用衝突）

嚴格原則：
- 只標「你能確定」的錯誤；不確定一律不標
- 數據摘要沒涵蓋的數字不可標為錯誤（不可假設它是編造的，除非回答明確宣稱該數字來自系統計算而摘要明確缺少該類別）
- 風格、遺漏、保守程度、主觀判斷差異都不是錯誤
- 最多列 5 條；quote 必須逐字摘自回答且不超過 80 字

只輸出 JSON，不要任何其他文字：
{"verdict":"pass"|"issues","issues":[{"type":"number|direction|fabricated|logic","severity":"high|medium","quote":"...","why":"...","correction":"..."}]}"""

_MAX_ANSWER_HEAD = 3000
_MAX_ANSWER_TAIL = 12000


def _build_user_message(digest: str, answer: str) -> str:
    if len(answer) > _MAX_ANSWER_HEAD + _MAX_ANSWER_TAIL:
        # 結論卡在尾部，保尾為主
        answer = (answer[:_MAX_ANSWER_HEAD] + "\n...(中段省略)...\n"
                  + answer[-_MAX_ANSWER_TAIL:])
    return f"【系統實際數據】\n{digest or '(無)'}\n\n【待覆核回答】\n{answer}"


# ─── JSON 解析（LLM 可能包 fence / 夾雜文字）─────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_VALID_TYPES = {"number", "direction", "fabricated", "logic"}


def _parse_verify_json(text: str) -> Optional[dict]:
    if not text:
        return None
    s = text.strip()
    fence = _FENCE_RE.search(s)
    if fence:
        s = fence.group(1).strip()
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(s[start:end + 1])
    except Exception:
        return None
    if not isinstance(data, dict) or data.get("verdict") not in ("pass", "issues"):
        return None
    issues = []
    for item in (data.get("issues") or [])[:8]:
        if not isinstance(item, dict) or item.get("type") not in _VALID_TYPES:
            continue
        issues.append({
            "type": item["type"],
            "severity": str(item.get("severity", "medium"))[:10],
            "quote": str(item.get("quote", ""))[:120],
            "why": str(item.get("why", ""))[:200],
            "correction": str(item.get("correction", ""))[:200],
        })
        if len(issues) >= 5:
            break
    verdict = "issues" if issues else "pass"
    return {"verdict": verdict, "issues": issues}


# ─── 跳過條件 ────────────────────────────────────────────────

_VERIFY_INTENTS = {
    "analysis", "backtest", "quant_research", "calibrate", "event_analysis",
    "conditional_prob", "scenario", "smc", "deep_analysis",
    "deep_phase1", "deep_phase2", "deep_phase3", "comprehensive_analysis",
    "factor_validation", "strategy_backtest", "regime_analysis",
    "momentum_analysis", "fundamental_analysis", "crypto_fundamental",
    "sector_analysis",
}


def should_verify(final_text: Optional[str], intents: Optional[set]) -> bool:
    """短文 / 無數字 / 閒聊 intent / 主模型失敗訊息 → 不覆核（省成本）"""
    if not final_text or len(final_text) < settings.verify_min_text_len:
        return False
    if final_text.lstrip().startswith("⚠️"):
        return False
    if not re.search(r"\d", final_text):
        return False
    if not intents or not (set(intents) & _VERIFY_INTENTS):
        return False
    return True


# ─── 主入口 ──────────────────────────────────────────────────

async def verify_answer(
    final_text: str,
    chart_state: Optional[dict],
    exec_result: Optional[dict],
    provider: str,
    api_key: Optional[str],
    base_url: Optional[str],
    main_model: str,
    override: str = "",
    timeout_sec: int = 120,
) -> Optional[dict]:
    """呼叫低一階模型覆核。失敗回 None（呼叫端視為略過），絕不上拋。"""
    try:
        model = pick_verifier_model(provider, main_model, override)
        digest = build_data_digest(chart_state, exec_result)
        user_msg = _build_user_message(digest, final_text)

        from app.core.llm.adapter import create_adapter
        adapter = create_adapter(
            provider=provider, api_key=api_key, model_name=model, base_url=base_url,
        )
        resp = await asyncio.wait_for(
            adapter.chat(
                [{"role": "user", "content": user_msg}],
                force_text=True,
                system_prompt=_VERIFY_SYSTEM_PROMPT,
            ),
            timeout=timeout_sec,
        )
        # adapter 失敗訊息以 ⚠️ 開頭（如 CLI 不可用 / 限流）→ 視為覆核失敗
        if not resp.message or resp.message.lstrip().startswith("⚠️"):
            logger.debug(f"[verify] model={model} 回應無效，略過")
            return None
        parsed = _parse_verify_json(resp.message)
        if parsed is None:
            logger.debug(f"[verify] model={model} JSON 解析失敗，略過")
            return None
        result = {
            "status": parsed["verdict"],
            "model": model,
            "issues": parsed["issues"],
            "usage": resp.usage,
        }
        logger.info(
            f"[verify] model={model} verdict={parsed['verdict']} "
            f"n_issues={len(parsed['issues'])}"
        )
        return result
    except asyncio.TimeoutError:
        logger.debug(f"[verify] 覆核逾時（{timeout_sec}s），略過")
        return None
    except Exception as e:
        logger.debug(f"[verify] 覆核失敗（不影響主流程）: {e}")
        return None


# ─── 可見區塊（鏡射 fact_check 區塊風格）─────────────────────

_TYPE_LABELS = {
    "number": "數字錯誤",
    "direction": "方向矛盾",
    "fabricated": "疑似編造",
    "logic": "邏輯矛盾",
}


def format_verify_block(result: dict) -> str:
    lines = ["", "", f"═══ 🔎 AI 覆核（{result.get('model', '?')} 交叉檢查）═══"]
    for issue in result.get("issues", []):
        label = _TYPE_LABELS.get(issue["type"], issue["type"])
        sev = "❗" if issue.get("severity") == "high" else "•"
        lines.append(f"  {sev} [{label}] 「{issue['quote']}」")
        lines.append(f"     — {issue['why']}")
        if issue.get("correction"):
            lines.append(f"     → {issue['correction']}")
    lines.append("⚠️ 上列段落請以系統實際值與修正說明為準，覆核模型也可能誤判，重大決策請自行複核")
    lines.append("═══════════════════════════════════════")
    return "\n".join(lines)
