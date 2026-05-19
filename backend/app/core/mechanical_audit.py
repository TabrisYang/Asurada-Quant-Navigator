"""阿斯拉量化系統 — v107.1：機械審查（取代 v106 C1 的 LLM critic）。

純 Python，不打 LLM。每次主分析 done 之前自動跑（< 50ms）。
比 LLM 自審更可靠：永遠執行成功、零成本、零延遲。

檢查的事：
1. **數值一致性**：報告寫的 RSI/ATR/lastClose/funding 等數字 vs chart_state 對比
2. **必引用欄位**：chart_state 有 X 但報告沒提 → 標記
3. **資料缺失誤導語句**：資料缺失時若報告寫「目前無事件」「市場情緒中性」 → 標記
"""

from __future__ import annotations

import re
from typing import Any


_NUM_TOL_PCT = 5.0  # 數值差異 ≥ 5% 才算不一致


# ─── 數值抓取正則 ──────────────────────────────

# 抓「RSI=72」「RSI 約 72」「RSI: 72」「RSI 為 72」這類
_RSI_PAT = re.compile(r"RSI[\s:=約為]*?([0-9]+\.?[0-9]*)", re.IGNORECASE)
_ATR_PAT = re.compile(r"ATR[\s:=約為]*?([0-9]+\.?[0-9]*)", re.IGNORECASE)

# v141.3：只抓「明確宣稱當前 RSI」的句型，避免誤抓歷史統計（如「大漲前 RSI 中位數 59.3」
# 「RSI 60-70 甜蜜區」「RSI 65.3–70.2 條件機率」）造成 false positive。
# 必須 RSI 前 10 字內出現「當前/目前/現值/最新/latest/current」等當下語境關鍵字。
_RSI_CURRENT_PAT = re.compile(
    r"(?:當前|目前|現價|現值|現為|現在|最新|此刻|latest|current)[^。\n]{0,10}?RSI[\s:=約為（(]*?([0-9]+\.?[0-9]*)",
    re.IGNORECASE,
)
# 排除「RSI 60-70」這種區間（後面緊跟 - – ~ 到 / 表示是範圍非點值）
_RSI_RANGE_SUFFIX = re.compile(r"^[\s]*[-–~/到至]")
_FUNDING_PAT = re.compile(r"funding[_\s]*rate[\s:=約為]*?([+\-]?[0-9]+\.?[0-9]*)\s*%", re.IGNORECASE)
_PRICE_PAT = re.compile(r"(?:lastClose|last_close|現價|當前價|目前價)[\s:=約為]*?\$?([0-9]+[,0-9]*\.?[0-9]*)", re.IGNORECASE)


def _safe_float(s: Any) -> float | None:
    if s is None:
        return None
    try:
        return float(str(s).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _diff_pct(a: float, b: float) -> float:
    if a == 0:
        return 0 if b == 0 else 100.0
    return abs(a - b) / abs(a) * 100


def _extract_first_match(pattern: re.Pattern, text: str) -> float | None:
    m = pattern.search(text or "")
    if not m:
        return None
    return _safe_float(m.group(1))


# ─── 檢查 1：數值一致性 ──────────────────────────


def _check_value_consistency(
    final_text: str, chart_state: dict, issues: list[str]
) -> int:
    """回傳檢查項數。差異 > tolerance 加入 issues。"""
    n_checks = 0
    inds = (chart_state or {}).get("indicatorValues") or {}
    price_ov = (chart_state or {}).get("priceOverview") or {}
    deriv = ((chart_state or {}).get("external_signals") or {}).get("derivatives") or {}

    # RSI — v141.3：只比對「明確宣稱當前 RSI」的句型，避免誤抓歷史統計/區間
    written_rsi = _extract_first_match(_RSI_CURRENT_PAT, final_text)
    actual_rsi_obj = inds.get("rsi") or inds.get("RSI") or {}
    actual_rsi_vals = actual_rsi_obj.get("values") if isinstance(actual_rsi_obj, dict) else None
    if written_rsi is not None and actual_rsi_vals:
        n_checks += 1
        last = next((v for v in reversed(actual_rsi_vals) if v is not None), None)
        if last is not None and _diff_pct(last, written_rsi) > _NUM_TOL_PCT:
            issues.append(
                f"當前 RSI 數值不一致：報告寫 {written_rsi:.1f}，實際 {last:.1f}"
            )

    # lastClose
    written_price = _extract_first_match(_PRICE_PAT, final_text)
    actual_price = _safe_float(price_ov.get("lastClose"))
    if written_price is not None and actual_price is not None:
        n_checks += 1
        if _diff_pct(actual_price, written_price) > _NUM_TOL_PCT:
            issues.append(
                f"現價數值不一致：報告寫 ${written_price:,.2f}，實際 ${actual_price:,.2f}"
            )

    # funding rate
    written_fund = _extract_first_match(_FUNDING_PAT, final_text)
    actual_fund = _safe_float(deriv.get("funding_rate_pct"))
    if written_fund is not None and actual_fund is not None:
        n_checks += 1
        # funding 都是小數值，用絕對差判斷（0.05% vs 0.06% 才該算 OK）
        if abs(actual_fund - written_fund) > 0.05:
            issues.append(
                f"funding rate 數值不一致：報告寫 {written_fund:+.3f}%，實際 {actual_fund:+.3f}%"
            )

    return n_checks


# ─── 檢查 2：必引用欄位 ──────────────────────────


def _check_required_mentions(
    final_text: str, chart_state: dict, issues: list[str]
) -> int:
    """chart_state 有但報告沒提 → 加入 issues。"""
    n_checks = 0
    text = final_text or ""
    cs = chart_state or {}

    # regimeWarning
    if cs.get("regimeWarning"):
        n_checks += 1
        if not any(kw in text for kw in ("低信心", "regime confidence", "信心過低", "低 confidence")):
            issues.append("regimeWarning 存在但報告未引用「低信心」相關警示")

    # 高優先事件
    upcoming = cs.get("upcoming_events") or []
    high_imminent = [
        e for e in upcoming
        if isinstance(e, dict)
        and (e.get("severity") == "high" or e.get("impact") == "high")
        and _safe_float(e.get("hours_to_event")) is not None
        and _safe_float(e.get("hours_to_event")) <= 24
    ]
    if high_imminent:
        n_checks += 1
        if not any(kw in text for kw in ("事件警示", "⚠️", "事件公布", "重大事件")):
            issues.append(
                f"24h 內高影響事件 {len(high_imminent)} 個但報告未引用警示"
            )

    # calendar stale
    cal_meta = cs.get("calendar_meta") or {}
    if cal_meta.get("is_stale"):
        n_checks += 1
        if not any(kw in text for kw in ("日曆已", "天未更新", "calendar", "事件日期可能不準")):
            issues.append("calendar_meta.is_stale=true 但報告未提日曆過期")

    # historical_insights
    if cs.get("historical_insights"):
        n_checks += 1
        if not any(kw in text for kw in ("Wilson", "歷史", "樣本", "winrate", "勝率")):
            issues.append("historical_insights 存在但報告未引用樣本量 / Wilson 下界")

    # portfolio 偏倚
    portfolio = cs.get("portfolio_summary") or {}
    lsr = _safe_float(portfolio.get("long_short_ratio"))
    if lsr is not None and (lsr > 3 or (0 < lsr < 0.34)):
        n_checks += 1
        if not any(kw in text for kw in ("組合偏多", "組合偏空", "整體組合", "long_short_ratio")):
            issues.append(
                f"portfolio long/short ratio={lsr:.2f}（極端）但報告未提組合偏倚警示"
            )

    return n_checks


# ─── 檢查 3：資料缺失誤導語句 ──────────────────────────


def _check_missing_data_misleading(
    final_text: str, chart_state: dict, issues: list[str]
) -> int:
    """資料缺失時報告卻寫「市場無事件 / 情緒中性」 → 嚴重誤導。"""
    n_checks = 0
    text = final_text or ""
    cs = chart_state or {}

    # 沒有 upcoming_events 卻寫「目前無事件影響」
    has_events = bool(cs.get("upcoming_events"))
    if not has_events:
        n_checks += 1
        misleading = [
            "目前無重大事件影響", "近期無事件", "無重要事件",
            "市場無事件", "no major events", "沒有重要事件",
        ]
        for kw in misleading:
            if kw in text:
                issues.append(
                    f"資料缺失誤導：upcoming_events 無資料但報告寫「{kw}」（應改寫為「無事件資料可判斷」）"
                )
                break

    # 沒有 social_sentiment 卻寫「市場情緒中性」
    has_sent = bool(cs.get("social_sentiment"))
    if not has_sent:
        n_checks += 1
        misleading = [
            "市場情緒中性", "社群情緒平穩", "情緒平和",
            "市場情緒平衡", "整體情緒中性",
        ]
        for kw in misleading:
            if kw in text:
                issues.append(
                    f"資料缺失誤導：social_sentiment 無資料但報告寫「{kw}」（應改寫為「無社群情緒資料」）"
                )
                break

    # 沒有 derivatives 卻寫「衍生品市場平衡」
    has_deriv = bool(((cs.get("external_signals") or {}).get("derivatives")))
    if not has_deriv:
        n_checks += 1
        misleading = [
            "衍生品市場平衡", "funding 中性", "OI 平穩",
            "多空比平衡",
        ]
        for kw in misleading:
            if kw in text:
                issues.append(
                    f"資料缺失誤導：derivatives 無資料但報告寫「{kw}」（應改寫為「無衍生品快照」）"
                )
                break

    return n_checks


# ─── Public API ──────────────────────────


def audit_final_text(final_text: str, chart_state: dict | None) -> dict:
    """每次主分析 done 之前呼叫。極快、不打網路 / LLM。

    Returns:
        {
          "passed": bool,
          "issues": list[str],
          "n_checks": int,           # 總檢查項數
          "n_failures": int,
          "summary": str,            # 「✅ 機械審查通過 (5/5)」or「⚠️ 5/5 通過，1 項未過」
        }
    """
    if not final_text or len(final_text.strip()) < 10:
        return {
            "passed": True,
            "issues": [],
            "n_checks": 0,
            "n_failures": 0,
            "summary": "—（報告過短，跳過審查）",
        }

    cs = chart_state or {}
    issues: list[str] = []
    n_checks = 0

    try:
        n_checks += _check_value_consistency(final_text, cs, issues)
    except Exception as e:
        issues.append(f"數值一致性檢查例外：{e}")

    try:
        n_checks += _check_required_mentions(final_text, cs, issues)
    except Exception as e:
        issues.append(f"必引用欄位檢查例外：{e}")

    try:
        n_checks += _check_missing_data_misleading(final_text, cs, issues)
    except Exception as e:
        issues.append(f"資料缺失誤導檢查例外：{e}")

    n_failures = len(issues)
    passed = n_failures == 0
    if n_checks == 0:
        summary = "—（無可審查項）"
    elif passed:
        summary = f"✅ 機械審查通過 ({n_checks}/{n_checks})"
    else:
        summary = f"⚠️ {n_checks - n_failures}/{n_checks} 通過，{n_failures} 項未過"

    return {
        "passed": passed,
        "issues": issues,
        "n_checks": n_checks,
        "n_failures": n_failures,
        "summary": summary,
    }
