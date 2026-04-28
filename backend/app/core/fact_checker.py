"""阿斯拉量化系統 — v104 Q3：LLM 數值編造偵測。

掃 LLM 生成文本中出現的具體數值（RSI / MACD / ATR / 價格 / 命中率 / funding / OI / 多空比 / Fear&Greed），
跟 chart_state 對照。差異超過容忍度就標 mismatch，前端顯示警示，使用者立刻知道哪段不能信。

設計原則：
- 不阻擋（事後標註，不重跑 LLM）
- 容忍度寬（避免假陽性）
- 只抓「明確指名」的數值（「RSI 65」必抓；「動量強」這種文字描述不抓）
- 失敗 graceful：抓不到就 skip，不影響主流程
"""

from __future__ import annotations

import re
from typing import Any, Optional

from loguru import logger

# ─── 數值抓取 regex ──────────────────────────────────
# 「RSI 65.4」「RSI=65」「RSI 為 65」「RSI: 65」
_INDICATOR_PATTERN = re.compile(
    r"(RSI|MACD|ATR|ADX|MFI|CCI|stoch|布林|BB)"
    r"(?:_\d+)?\s*[:=：為是]*\s*"  # 只接 _14 這種底線後綴；不要無底線（會 gobble 進 value）
    r"(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# 「funding -0.005%」「funding rate=0.01%」「資金費率 0.005%」
_FUNDING_PATTERN = re.compile(
    r"(?:funding[_\s]*rate|資金費率|費率)\s*[=:：]*\s*"
    r"(-?\d+(?:\.\d+)?)\s*%?",
    re.IGNORECASE,
)

# 「多空比 1.5」「long_short_ratio=2.3」
_LS_PATTERN = re.compile(
    r"(?:多空比|long[_\s]*short[_\s]*ratio|LS\s*ratio|持倉比)\s*[=:：]*\s*"
    r"(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# 「Fear&Greed 33」「恐懼貪婪 33」「FNG=25」
_FNG_PATTERN = re.compile(
    r"(?:fear[\s&]*greed|恐懼貪婪|FNG|F&G)\s*[=:：值]*\s*"
    r"(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# 「命中率 65%」「win rate = 70.5%」「歷史勝率 60%」
_WINRATE_PATTERN = re.compile(
    r"(?:命中率|勝率|win[_\s]*rate|hit[_\s]*rate)\s*[=:：為是]*\s*"
    r"(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)

# ─── 容忍度 ──────────────────────────────────
_TOLERANCE = {
    "indicator": 0.10,   # ±10%（指標值）
    "funding": 0.0005,   # ±0.05%（絕對差，funding 本來就小）
    "ls_ratio": 0.20,    # ±0.2（絕對差）
    "fng": 5,            # ±5（FNG 是整數）
    "winrate": 5.0,      # ±5%（命中率）
}


def _within(actual: float, claimed: float, tol: float, mode: str = "abs") -> bool:
    if mode == "abs":
        return abs(actual - claimed) <= tol
    if mode == "rel":
        if actual == 0:
            return abs(claimed) <= tol
        return abs((actual - claimed) / actual) <= tol
    return False


def check_text_against_chart_state(
    text: str,
    chart_state: dict,
) -> dict:
    """掃文本並回報 mismatch list。

    Returns: {
      "checked_count": int,
      "mismatches": [
        {"type": "indicator", "name": "RSI", "claimed": 65, "actual": 58, "snippet": "..."},
        ...
      ],
      "summary": "短訊息給 UI 顯示"
    }
    """
    result: dict[str, Any] = {"checked_count": 0, "mismatches": [], "summary": ""}
    if not text or not chart_state:
        return result

    # ─── 指標值 ───────────────────────────────
    indicator_values = chart_state.get("indicatorValues") or {}
    if indicator_values:
        for m in _INDICATOR_PATTERN.finditer(text):
            name = m.group(1).upper()
            try:
                claimed = float(m.group(2))
            except ValueError:
                continue

            # 在 indicator_values 裡找對應 key（容忍命名差異）
            actual = None
            for key, val in indicator_values.items():
                k_up = key.upper()
                if name in k_up or k_up.startswith(name):
                    try:
                        actual = float(val) if val is not None else None
                    except (TypeError, ValueError):
                        continue
                    break
            if actual is None:
                continue

            result["checked_count"] += 1
            if not _within(actual, claimed, _TOLERANCE["indicator"], "rel"):
                result["mismatches"].append({
                    "type": "indicator",
                    "name": name,
                    "claimed": claimed,
                    "actual": round(actual, 4),
                    "snippet": _snippet(text, m.start(), m.end()),
                })

    # ─── 衍生品 / 情緒 ───────────────────────────────
    ext = chart_state.get("external_signals") or {}
    deriv = ext.get("derivatives") or {}
    sent = ext.get("sentiment") or {}

    if "funding_rate_pct" in deriv:
        actual_funding = deriv["funding_rate_pct"]  # %
        for m in _FUNDING_PATTERN.finditer(text):
            try:
                claimed = float(m.group(1))
            except ValueError:
                continue
            # text 可能寫 0.005 或 0.005%；統一當 % 比較
            result["checked_count"] += 1
            if not _within(actual_funding, claimed, 0.05, "abs"):  # 容忍 ±0.05%
                result["mismatches"].append({
                    "type": "funding",
                    "claimed": claimed,
                    "actual": actual_funding,
                    "snippet": _snippet(text, m.start(), m.end()),
                })

    if "global_long_short_ratio" in deriv:
        actual_ls = deriv["global_long_short_ratio"]
        for m in _LS_PATTERN.finditer(text):
            try:
                claimed = float(m.group(1))
            except ValueError:
                continue
            result["checked_count"] += 1
            if not _within(actual_ls, claimed, _TOLERANCE["ls_ratio"], "abs"):
                result["mismatches"].append({
                    "type": "long_short_ratio",
                    "claimed": claimed,
                    "actual": actual_ls,
                    "snippet": _snippet(text, m.start(), m.end()),
                })

    if "fear_greed_value" in sent:
        actual_fng = sent["fear_greed_value"]
        for m in _FNG_PATTERN.finditer(text):
            try:
                claimed = float(m.group(1))
            except ValueError:
                continue
            result["checked_count"] += 1
            if not _within(actual_fng, claimed, _TOLERANCE["fng"], "abs"):
                result["mismatches"].append({
                    "type": "fear_greed",
                    "claimed": claimed,
                    "actual": actual_fng,
                    "snippet": _snippet(text, m.start(), m.end()),
                })

    # ─── 命中率 ───────────────────────────────
    ra = chart_state.get("recent_accuracy") or {}
    actual_wr = ra.get("win_rate_30d") or ra.get("win_rate")
    if actual_wr is not None:
        try:
            actual_wr = float(actual_wr)
        except (TypeError, ValueError):
            actual_wr = None

    if actual_wr is not None:
        for m in _WINRATE_PATTERN.finditer(text):
            try:
                claimed = float(m.group(1))
            except ValueError:
                continue
            result["checked_count"] += 1
            if not _within(actual_wr, claimed, _TOLERANCE["winrate"], "abs"):
                result["mismatches"].append({
                    "type": "winrate",
                    "claimed": claimed,
                    "actual": round(actual_wr, 1),
                    "snippet": _snippet(text, m.start(), m.end()),
                })

    # ─── 摘要 ───────────────────────────────
    n_mis = len(result["mismatches"])
    if n_mis > 0:
        result["summary"] = (
            f"⚠️ 偵測到 {n_mis} 處數值跟 chart_state 不一致（檢查 {result['checked_count']} 處）— "
            f"請使用者核對標記區塊"
        )
        logger.warning(
            f"[fact_checker] {n_mis} mismatches: "
            + ", ".join(f"{m['type']}={m['claimed']} vs {m['actual']}" for m in result["mismatches"][:3])
        )
    else:
        result["summary"] = f"✓ 抽查 {result['checked_count']} 處數值無編造"

    return result


def _snippet(text: str, start: int, end: int, padding: int = 25) -> str:
    """截取 mismatch 周圍上下文（給 UI 顯示）。"""
    s = max(0, start - padding)
    e = min(len(text), end + padding)
    out = text[s:e].replace("\n", " ")
    if s > 0:
        out = "…" + out
    if e < len(text):
        out = out + "…"
    return out
