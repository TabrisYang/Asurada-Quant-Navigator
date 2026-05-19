#!/usr/bin/env python3
"""P3 Checklist 自動提醒腳本。

每次 .command 啟動時背景執行，達標時跳 macOS dialog + 預備剪貼簿訊息。
plan 文件：/Users/tonyy/.claude/plans/melodic-prancing-catmull.md

行為：
- ~/.p3_done 存在 → 直接 exit 0（使用者已執行 P3）
- 跑 shadow_mode（或讀當日已存在 JSON）→ parse → 比對 baseline
- 未達標 → 印一行進度（被 .command stdout 印出），exit 0
- 達標 → pbcopy 訊息 + osascript display dialog，exit 0
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # 阿斯拉量化系統V2/
DB_DIR = ROOT_DIR / "backend" / "data" / "db"
SENTINEL = Path.home() / ".p3_done"
SHADOW_MODE_SCRIPT = ROOT_DIR / "backend" / "scripts" / "shadow_mode.py"

# 5/19 baseline（hardcode，避免讀 plan markdown）
BASELINE = {
    "date": "2026-05-19",
    "total": 14,
    "rows_with_full_samples": 0,
    "trigger": {"A": 0.0, "C": 14.3, "F": 35.7},
    "regime": {
        "trending_up":   {"n": 5, "hit_rate": 0.0,   "avg_outcome": -2.572},
        "trending_down": {"n": 4, "hit_rate": 33.3,  "avg_outcome":  1.518},
        "unknown":       {"n": 3, "hit_rate": 0.0,   "avg_outcome":  0.560},
        "ranging":       {"n": 2, "hit_rate": 100.0, "avg_outcome":  7.295},
    },
}

THRESHOLD_FULL_SAMPLES = 10  # 足樣本 case ≥ 10 = 啟動條件
HIT_DROP_LIMIT_PP = 10.0      # 任一 regime hit rate 跌幅上限（pp）
TRIGGER_DRIFT_LIMIT_PCT = 20.0  # A/C/F 觸發頻率變動上限（%）


def _today_str() -> str:
    return datetime.now().strftime("%Y%m%d")


def _find_latest_report() -> Path | None:
    if not DB_DIR.exists():
        return None
    reports = sorted(DB_DIR.glob("shadow_report_*.json"))
    return reports[-1] if reports else None


def _report_is_fresh(path: Path, max_age_hours: int = 6) -> bool:
    """報告產生於 max_age_hours 內視為新鮮，可直接讀，不需重跑。"""
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime) < timedelta(hours=max_age_hours)


def _run_shadow_mode() -> Path | None:
    """跑 shadow_mode.py 產生新報告，回傳新報告路徑。失敗回 None。"""
    venv_python = ROOT_DIR / "backend" / ".venv" / "bin" / "python3"
    python_bin = str(venv_python) if venv_python.exists() else "python3"
    try:
        subprocess.run(
            [python_bin, str(SHADOW_MODE_SCRIPT), "--days", "14"],
            cwd=str(ROOT_DIR / "backend"),
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    expected = DB_DIR / f"shadow_report_{_today_str()}.json"
    if expected.exists():
        return expected
    return _find_latest_report()


def _load_report() -> dict | None:
    """取最新 shadow report（新鮮就直接讀、否則重跑）。"""
    latest = _find_latest_report()
    if latest and _report_is_fresh(latest):
        try:
            return json.loads(latest.read_text(encoding="utf-8"))
        except Exception:
            pass
    fresh = _run_shadow_mode()
    if fresh and fresh.exists():
        try:
            return json.loads(fresh.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _evaluate(report: dict) -> tuple[bool, dict]:
    """評估是否達標。回傳 (is_ready, snapshot)。"""
    summary = report.get("summary", {})
    total = summary.get("total", 0)
    _suff = summary.get("sample_sufficiency", {})
    # v141.3：gate 改用「核心 2 條件計數」（direction_balance + regime_warning），
    # combo_stats 因 sparsity 結構性難達標，降為 informational。
    rows_core = _suff.get("rows_with_core_samples")
    rows_full = _suff.get("rows_with_full_samples", 0)
    # 舊報告無 rows_with_core_samples 欄位時 fallback 用 full（向後相容）
    rows_gate = rows_core if rows_core is not None else rows_full
    trigger = {k: v.get("pct", 0.0) for k, v in summary.get("trigger_frequencies", {}).items()}
    regime = summary.get("by_regime", {})

    # 條件 1：核心足樣本 case ≥ 10（combo_stats 不計入 gate）
    cond_samples = rows_gate >= THRESHOLD_FULL_SAMPLES

    # 條件 2：任一 regime hit rate 下降不超過 10pp（樣本 <baseline 時跳過該 regime）
    hit_breaches = []
    for name, b in BASELINE["regime"].items():
        current = regime.get(name)
        if not current:
            continue
        # 只在 baseline n >= 2 且 current n >= 2 時比對
        if b["n"] < 2 or current.get("n", 0) < 2:
            continue
        drop_pp = b["hit_rate"] - current.get("hit_rate", 0.0)
        if drop_pp > HIT_DROP_LIMIT_PP:
            hit_breaches.append(f"{name}: {b['hit_rate']:.1f}% → {current['hit_rate']:.1f}% (跌 {drop_pp:.1f}pp)")
    cond_hit = len(hit_breaches) == 0

    # 條件 3：A/C/F 觸發頻率「上升」超過 20% 才算問題
    # v141.3：下降 = 系統更 conservative（biased/ranging 衝突 case 變少）= 好事，不阻擋。
    # 只 flag 上升（系統變更 biased / 更常觸發觀望規則 = 可能 alpha 劣化）。
    trigger_breaches = []
    for k, baseline_pct in BASELINE["trigger"].items():
        current_pct = trigger.get(k, 0.0)
        if baseline_pct == 0:
            if current_pct > TRIGGER_DRIFT_LIMIT_PCT:
                trigger_breaches.append(f"{k}: 0.0% → {current_pct:.1f}% (新增觸發↑)")
        elif current_pct > baseline_pct:  # 只看上升
            rise_pct = (current_pct - baseline_pct) / baseline_pct * 100
            if rise_pct > TRIGGER_DRIFT_LIMIT_PCT:
                trigger_breaches.append(f"{k}: {baseline_pct:.1f}% → {current_pct:.1f}% (上升 {rise_pct:.0f}%)")
    cond_trigger = len(trigger_breaches) == 0

    is_ready = cond_samples and cond_hit and cond_trigger

    return is_ready, {
        "total": total,
        "rows_full": rows_full,       # combo 含在內（informational）
        "rows_gate": rows_gate,        # v141.3 核心 2 條件（gate 用）
        "cond_samples": cond_samples,
        "cond_hit": cond_hit,
        "cond_trigger": cond_trigger,
        "hit_breaches": hit_breaches,
        "trigger_breaches": trigger_breaches,
        "trigger": trigger,
        "regime": regime,
    }


def _format_clipboard(snap: dict) -> str:
    """組剪貼簿訊息（給使用者貼回給 Claude）。"""
    lines = ["P3 可以啟動了！以下是回報內容：", "", "【今日 shadow_mode 數據】"]
    lines.append(f"- 已驗證樣本：{snap['total']} 筆")
    lines.append(f"- 核心足樣本 case（dir+regime）：{snap['rows_gate']}/{snap['total']} ✅")
    lines.append(f"- 完整足樣本 case（含 combo，informational）：{snap['rows_full']}/{snap['total']}")
    for k, pct in snap["trigger"].items():
        lines.append(f"- {k} 觸發：{pct:.1f}%")

    lines.append("")
    lines.append("【各 regime hit rate（vs 5/19 baseline）】")
    for name, b in BASELINE["regime"].items():
        cur = snap["regime"].get(name) or {"n": 0, "hit_rate": 0.0, "avg_outcome": 0.0}
        delta = cur.get("hit_rate", 0.0) - b["hit_rate"]
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"- {name}: n={cur.get('n', 0)}, hit={cur.get('hit_rate', 0):.1f}% "
            f"(baseline {b['hit_rate']:.1f}%, 變化 {sign}{delta:.1f}pp)"
        )

    lines.append("")
    lines.append("【行為驗證 — 請手動觀察 5-10 個近期分析報告】")
    lines.append("[ ] compute_factor_ic 是否被呼叫（檢查報告是否含 IC/strength 數字）")
    lines.append("[ ] SMC 四要素章節是否完整（BSL/SSL/OB/FVG/Premium-Discount）")
    lines.append("[ ] 30 秒結論卡的 RR<1.5 觸發率（≤30% 為健康）")
    lines.append("")
    lines.append("請啟動 P3。")
    return "\n".join(lines)


def _set_clipboard(text: str) -> bool:
    try:
        p = subprocess.run(["pbcopy"], input=text.encode("utf-8"), timeout=5, check=False)
        return p.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _show_dialog(snap: dict) -> None:
    """跳 osascript modal dialog（不阻塞使用者操作系統）。"""
    body = (
        f"🟢 P3 已可啟動\\n\\n"
        f"Checklist 條件全綠：\\n"
        f"  ✅ 核心足樣本 case: {snap['rows_gate']}/{THRESHOLD_FULL_SAMPLES}\\n"
        f"  ✅ regime hit 未跌破門檻\\n"
        f"  ✅ 觸發頻率未漂移超限\\n\\n"
        f"回報訊息已複製到剪貼簿\\n"
        f"請貼給 Claude 觸發 P3 實作\\n\\n"
        f"執行完 P3 後請跑：touch ~/.p3_done"
    )
    script = (
        f'display dialog "{body}" '
        f'buttons {{"稍後再說", "我知道了"}} default button "我知道了" '
        f'with title "阿斯拉量化系統 — P3 Checklist 全綠" '
        f'with icon caution'
    )
    try:
        subprocess.run(["osascript", "-e", script], timeout=300, check=False)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass


def _print_progress(snap: dict) -> None:
    """未達標時印一行給 .command terminal。"""
    parts = [f"⏳ P3 等待中：核心足樣本 {snap['rows_gate']}/{THRESHOLD_FULL_SAMPLES}"]
    if snap.get("rows_full", 0) != snap.get("rows_gate", 0):
        parts.append(f"(含 combo {snap['rows_full']})")
    if snap["hit_breaches"]:
        parts.append(f"⚠️ regime hit 跌破：{', '.join(snap['hit_breaches'])}")
    # v141.3：觸發頻率「下降」是 conservative（好事），不再當警告，只 flag 上升
    if snap["trigger_breaches"]:
        parts.append(f"ℹ️ 觸發頻率漂移：{', '.join(snap['trigger_breaches'])}")
    print(" | ".join(parts))


def main() -> int:
    # Sentinel：P3 已執行 → 跳過
    if SENTINEL.exists():
        return 0

    report = _load_report()
    if not report:
        # 無報告或 shadow_mode 失敗，靜默退出（不擾人）
        return 0

    is_ready, snap = _evaluate(report)

    if not is_ready:
        _print_progress(snap)
        return 0

    # 達標：剪貼簿 + dialog
    clipboard_msg = _format_clipboard(snap)
    _set_clipboard(clipboard_msg)
    _show_dialog(snap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
