"""阿斯拉量化系統 — v106 C1：Reflection / Critic agent。

第二個（cheap）LLM 看完主分析的 final_text 後做 critique：
- 抓邏輯謬誤（資料說 A，結論說反 A）
- 抓未引用的關鍵警示（external_signals 極端但結論沒提）
- 抓 confidence 是否與證據相符

設計原則：
- **非阻塞**：透過獨立 endpoint POST /chat/critique 呼叫，主分析完成後使用者手動觸發
- **便宜**：強制使用 gpt-4o-mini（成本約 1/30 of gpt-4o），最多 600 tokens
- **可選**：UI toggle 控制；不啟用就完全不付費
- **獨立 graceful**：critique 失敗不影響主分析

回傳結構：
{
  "critique": "...邏輯一致性 / 風險警示 / 信心相符性...",
  "issues_found": ["技術指標說 RSI=72 超買，但結論建議加碼", ...],
  "verdict": "consistent / concerns / contradiction",
  "tokens_used": 543,
  "model": "gpt-4o-mini"
}
"""

from __future__ import annotations

import json
import os
from typing import Optional

from loguru import logger


CRITIC_SYSTEM_PROMPT = """你是阿斯拉量化系統的「審查 LLM」（Reflection Critic）。
你的工作是檢視主分析 LLM 產出的結論，找出：

1. **邏輯不一致**：chart_state 中的事實 vs 結論之間的矛盾（例：external_signals.funding_rate=+0.15% 是過熱，但結論還寫「市場情緒中性」）
2. **未引用的重大警示**：upcoming_events 有高影響事件但結論沒提；regime confidence < 30% 但建議大倉位；user_positions 反向但沒警示
3. **信心過高**：用詞「強烈建議」「肯定」「保證」等，但 chart_state 顯示樣本不足 / Wilson CI 寬 / 多訊號分歧
4. **數值錯誤**：RR 算錯、價位寫錯（跟 chart_state 對比）

不要重寫結論。**僅指出問題**，每個問題附上 chart_state 對應欄位作為證據。
若沒找到問題，verdict="consistent" 並說明你檢查了哪些面向。

輸出 JSON 格式：
{
  "verdict": "consistent | concerns | contradiction",
  "issues_found": ["具體問題 1", "具體問題 2", ...],
  "summary": "1-2 句話總結"
}
僅輸出 JSON，不要其他文字。
"""


async def critique(
    final_text: str,
    chart_state: Optional[dict],
    user_question: str,
    api_key: Optional[str] = None,
) -> dict:
    """呼叫 cheap LLM 對 final_text 做審查。

    Args:
        final_text: 主分析 LLM 的最終文字輸出
        chart_state: 該次分析的 chart_state snapshot
        user_question: 使用者原始問題
        api_key: OpenAI API key（若無傳入則用 OPENAI_API_KEY env）

    Returns:
        dict with verdict / issues_found / summary / tokens_used
    """
    if not final_text or len(final_text.strip()) < 50:
        return {"verdict": "skipped", "reason": "final_text 太短，無需審查"}

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        return {"verdict": "skipped", "reason": "無 OpenAI API key"}

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=key, timeout=20.0)

        # 精簡 chart_state（只保留關鍵欄位避免 prompt 過大）
        compact_state = _compact_chart_state(chart_state or {})

        user_msg = (
            f"使用者原始問題：{user_question[:300]}\n\n"
            f"主分析結論（待審查）：\n{final_text[:6000]}\n\n"
            f"當時的 chart_state（精簡）：\n{json.dumps(compact_state, ensure_ascii=False, indent=2)[:4000]}"
        )

        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=600,
            response_format={"type": "json_object"},
        )

        content = resp.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {"verdict": "concerns", "issues_found": [], "summary": content[:300]}

        usage = resp.usage
        return {
            **parsed,
            "tokens_used": (usage.total_tokens if usage else 0),
            "model": "gpt-4o-mini",
        }
    except Exception as e:
        logger.warning(f"[reflection_critic] 失敗: {e}")
        return {"verdict": "skipped", "reason": f"critic 呼叫失敗：{str(e)[:200]}"}


_KEEP_FIELDS = {
    "symbol", "timeframe", "current_price", "currentRegime", "regime_subtype",
    "external_signals_summary", "upcoming_events", "calendar_meta",
    "user_positions", "portfolio_summary", "rl_strategic_insight",
    "historical_insights", "social_sentiment",
    "regimeWarning", "indicatorValues", "smcStructure", "auto_position_multiplier",
    "fundamentals_summary", "drift_status",
}


def _compact_chart_state(state: dict) -> dict:
    """只保留審查關鍵欄位 → 控制 prompt 大小。"""
    out = {}
    for k in _KEEP_FIELDS:
        if k in state:
            out[k] = state[k]
    return out
