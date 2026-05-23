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

# v108 Phase 2：「位於區間 X%」「donchian position 位置 X%」
_DONCHIAN_POS_PATTERN = re.compile(
    r"(?:位於區間|區間位置|donchian[_\s]*position)\s*[（(]?\s*"
    r"(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)

# ─── v149：回測 / Monte Carlo 數字（對照 exec_result，非 chart_state）──
# 這些數字目前完全沒驗證，LLM 改寫/湊整會無聲流到用戶（Q5）。
# 不收 win_rate（與既有 recent_accuracy 勝率比對衝突，避免假陽性）。
_PF_PATTERN = re.compile(
    r"(?:profit[\s_]*factor|盈虧比|獲利因子|\bPF\b)\s*[:=：約達為是]*\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_SHARPE_PATTERN = re.compile(
    r"sharpe(?:\s*ratio|\s*比率|\s*值)?\s*[:=：約達為是]*\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_SORTINO_PATTERN = re.compile(
    r"sortino(?:\s*ratio)?\s*[:=：約達為是]*\s*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_MDD_PATTERN = re.compile(
    r"(?:max[\s_]*drawdown|最大回撤|\bMDD\b)\s*[:=：約達為是]*\s*(-?\d+(?:\.\d+)?)\s*%?",
    re.IGNORECASE,
)
_MC_RUIN_PATTERN = re.compile(
    r"(?:破產(?:機率|概率|風險)|ruin(?:\s*probability)?)\s*[:=：約達為是]*\s*(\d+(?:\.\d+)?)\s*%?",
    re.IGNORECASE,
)
_MC_PROFIT_PATTERN = re.compile(
    r"(?:獲利機率|獲利概率|profit[\s_]*probability)\s*[:=：約達為是]*\s*(\d+(?:\.\d+)?)\s*%?",
    re.IGNORECASE,
)

# ─── 容忍度 ──────────────────────────────────
# v108 Phase 2：指標分組容忍度（從原 ±10% 統一收緊）
# 振盪類 (0-100 範圍) 用絕對差更精確；其餘用相對差
_INDICATOR_TOLERANCE: dict[str, tuple[str, float]] = {
    "RSI": ("abs", 2.0),
    "STOCH": ("abs", 2.0),
    "MFI": ("abs", 2.0),
    "ADX": ("abs", 2.0),
    "CCI": ("rel", 0.05),
    "MACD": ("rel", 0.05),
    "ATR": ("rel", 0.05),
    "BB": ("rel", 0.02),
    "布林": ("rel", 0.02),
}
_INDICATOR_DEFAULT_TOLERANCE: tuple[str, float] = ("rel", 0.05)  # 未列名指標用 ±5%

_TOLERANCE = {
    # 舊欄位保留（funding / ls_ratio / fng / winrate 維持原值）
    "funding": 0.0005,   # ±0.05%（絕對差）
    "ls_ratio": 0.20,    # ±0.2（絕對差）
    "fng": 5,            # ±5（FNG 是整數）
    "winrate": 5.0,      # ±5%（命中率）
    # v108 新增：donchian_position_pct（百分位）絕對差 ±1%
    "donchian_position": 1.0,
}


def _lookup_indicator_tolerance(name: str) -> tuple[str, float]:
    """v108：依指標名查容忍度，未列名 fallback 到 default ±5%。"""
    name_up = name.upper()
    for key, tol in _INDICATOR_TOLERANCE.items():
        if name_up == key.upper() or name_up.startswith(key.upper()):
            return tol
    return _INDICATOR_DEFAULT_TOLERANCE


def _within(actual: float, claimed: float, tol: float, mode: str = "abs") -> bool:
    if mode == "abs":
        return abs(actual - claimed) <= tol
    if mode == "rel":
        if actual == 0:
            return abs(claimed) <= tol
        return abs((actual - claimed) / actual) <= tol
    return False


def _extract_backtest_facts(exec_result: Optional[dict]) -> dict:
    """從 exec_result（run_quant_research / run_backtest）抽出回測關鍵數字實際值。

    只抽無歧義、且目前完全沒驗證的指標：PF / Sharpe / Sortino / MDD / MC 破產率 / MC 獲利率。
    抽不到就略過（不產生假陽性）。
    """
    facts: dict[str, float] = {}
    if not isinstance(exec_result, dict):
        return facts

    def _num(x):
        try:
            return float(x) if x is not None else None
        except (TypeError, ValueError):
            return None

    for r in (exec_result.get("results") or []):
        if not isinstance(r, dict):
            continue
        res = r.get("result")
        if not isinstance(res, dict):
            continue
        bt = res.get("backtest") if isinstance(res.get("backtest"), dict) else None
        # run_backtest 可能把績效放頂層
        if bt is None and ("profit_factor" in res or "sharpe_ratio" in res):
            bt = res
        if bt and not bt.get("error"):
            for fk, key in (("pf", "profit_factor"), ("sharpe", "sharpe_ratio"),
                            ("sortino", "sortino_ratio"), ("mdd", "max_drawdown_pct")):
                v = _num(bt.get(key))
                if v is not None:
                    facts.setdefault(fk, v)
        mc = res.get("monte_carlo")
        if isinstance(mc, dict) and mc.get("status") == "success":
            for fk, key in (("mc_ruin", "ruin_probability"), ("mc_profit", "profit_probability")):
                v = _num(mc.get(key))
                if v is not None:
                    facts.setdefault(fk, v)
    return facts


def check_text_against_chart_state(
    text: str,
    chart_state: dict,
    exec_result: Optional[dict] = None,
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
    if not text:
        return result
    # chart_state 可能空，但 exec_result 仍可能有回測數字要驗；各段已各自 .get 防護
    chart_state = chart_state or {}

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
            # v108 Phase 2：依指標分組容忍度
            tol_mode, tol_val = _lookup_indicator_tolerance(name)
            if not _within(actual, claimed, tol_val, tol_mode):
                result["mismatches"].append({
                    "type": "indicator",
                    "name": name,
                    "claimed": claimed,
                    "actual": round(actual, 4),
                    "tolerance": f"±{tol_val} ({tol_mode})",
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

    # ─── v108 Phase 2：donchian_position_pct ─────────────────
    actual_dp = chart_state.get("donchian_position_pct")
    if actual_dp is not None:
        try:
            actual_dp = float(actual_dp)
        except (TypeError, ValueError):
            actual_dp = None
    if actual_dp is not None:
        for m in _DONCHIAN_POS_PATTERN.finditer(text):
            try:
                claimed = float(m.group(1))
            except ValueError:
                continue
            result["checked_count"] += 1
            if not _within(actual_dp, claimed, _TOLERANCE["donchian_position"], "abs"):
                result["mismatches"].append({
                    "type": "donchian_position",
                    "claimed": claimed,
                    "actual": round(actual_dp, 1),
                    "tolerance": f"±{_TOLERANCE['donchian_position']}",
                    "snippet": _snippet(text, m.start(), m.end()),
                })

    # ─── v149：回測 / MC 數字（對照 exec_result）──
    bt_facts = _extract_backtest_facts(exec_result)
    if bt_facts:
        # (label, pattern, fact_key, 容忍度abs, 是否取絕對值比較)
        _bt_checks = [
            ("PF", _PF_PATTERN, "pf", 0.3, False),
            ("Sharpe", _SHARPE_PATTERN, "sharpe", 0.3, False),
            ("Sortino", _SORTINO_PATTERN, "sortino", 0.3, False),
            ("MDD", _MDD_PATTERN, "mdd", 3.0, True),
            ("破產機率", _MC_RUIN_PATTERN, "mc_ruin", 5.0, False),
            ("獲利機率", _MC_PROFIT_PATTERN, "mc_profit", 5.0, False),
        ]
        for label, pat, fkey, tol, use_abs_mag in _bt_checks:
            actual = bt_facts.get(fkey)
            if actual is None:
                continue
            for m in pat.finditer(text):
                try:
                    claimed = float(m.group(1))
                except ValueError:
                    continue
                a, c = (abs(actual), abs(claimed)) if use_abs_mag else (actual, claimed)
                result["checked_count"] += 1
                if not _within(a, c, tol, "abs"):
                    result["mismatches"].append({
                        "type": "backtest",
                        "name": label,
                        "claimed": claimed,
                        "actual": round(actual, 3),
                        "tolerance": f"±{tol}",
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
