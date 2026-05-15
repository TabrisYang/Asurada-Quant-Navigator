"""阿斯拉量化系統 v132 — Shadow Mode 兩份報告比對工具。

用途：
  v132 編排管線上線後 2 週，跑一次 `shadow_mode.py --days 14`，
  再用本腳本比對 baseline → 列出劣化 >10% 的指標 → 給出回滾建議。

使用方式：
  # 基本比對（兩份 JSON 都已存在）
  python3 backend/scripts/compare_shadow_reports.py \\
      backend/data/db/shadow_report_20260514.json \\
      backend/data/db/shadow_report_20260529.json

  # --auto：先跑 shadow_mode.py 產出新報告，再自動比對 baseline
  python3 backend/scripts/compare_shadow_reports.py --auto

  # --target-date YYYY-MM-DD（launchd 用）：只在指定日期當天或之後執行
  # 配 marker 檔避免重複觸發
  python3 backend/scripts/compare_shadow_reports.py \\
      --auto --target-date 2026-05-29 \\
      --marker backend/data/db/.shadow_pipeline_eval_done

設計：
  - 「劣化 >10%」依 CLAUDE.md 規則，超過即建議關 flag 回滾
  - by_regime hit_rate / avg_return 用相對誤差
  - 觸發頻率 A/C/F 用絕對 percentage point
  - 看漲說漲 verdict 轉變(GREY_ZONE → BUG / ALPHA → BUG)視為重大變化
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path


DEGRADATION_THRESHOLD_PCT = 10.0  # >10% 相對劣化 → 觸發警告
TRIGGER_FREQ_DELTA_PP = 10.0      # 觸發頻率絕對 percentage point 差 ≥10pp 視為顯著


def _rel_delta_pct(baseline: float, new: float) -> float:
    """相對變化 %（負值 = 變差）。baseline=0 時退回絕對差。"""
    if baseline == 0:
        return new - baseline
    return (new - baseline) / abs(baseline) * 100.0


def _run_shadow_mode(days: int = 14) -> Path:
    """跑 shadow_mode.py --days 14，回傳新生成的 JSON 路徑。"""
    backend_dir = Path(__file__).resolve().parent.parent
    venv_py = backend_dir / ".venv" / "bin" / "python3"
    if not venv_py.exists():
        venv_py = "python3"  # fallback
    cmd = [str(venv_py), "scripts/shadow_mode.py", "--days", str(days)]
    print(f"[compare] 跑 shadow_mode.py --days {days} 中...")
    result = subprocess.run(
        cmd, cwd=backend_dir, capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print("[compare] ❌ shadow_mode.py 執行失敗")
        print(result.stderr[-2000:])
        sys.exit(1)
    # 從輸出找出 JSON 報告路徑
    for line in result.stdout.splitlines():
        if "JSON 報告" in line and ".json" in line:
            return Path(line.split("：", 1)[-1].strip())
    # 後備：找今天日期的報告
    today_str = date.today().strftime("%Y%m%d")
    candidate = backend_dir / "data" / "db" / f"shadow_report_{today_str}.json"
    if candidate.exists():
        return candidate
    print("[compare] ❌ 找不到新生成的 shadow_report JSON")
    sys.exit(1)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compare(baseline_path: Path, new_path: Path) -> tuple[bool, list[str]]:
    """比對兩份報告，回傳 (has_degradation, finding_lines)。"""
    base = _load(baseline_path).get("summary", {})
    new = _load(new_path).get("summary", {})
    findings: list[str] = []
    has_deg = False

    findings.append("=" * 60)
    findings.append("Shadow Mode 報告比對")
    findings.append(f"  Baseline : {baseline_path.name} (total={base.get('total', 'N/A')})")
    findings.append(f"  New      : {new_path.name} (total={new.get('total', 'N/A')})")
    findings.append("=" * 60)

    # 1) by_regime hit_rate / avg_return（相對誤差）
    findings.append("\n[1] 各 regime 命中率 / 平均報酬")
    base_reg = base.get("by_regime", {})
    new_reg = new.get("by_regime", {})
    for regime in sorted(set(base_reg) | set(new_reg)):
        b = base_reg.get(regime, {}) or {}
        n = new_reg.get(regime, {}) or {}
        b_hit = b.get("hit_rate", 0) or 0
        n_hit = n.get("hit_rate", 0) or 0
        b_avg = b.get("avg_outcome", 0) or 0
        n_avg = n.get("avg_outcome", 0) or 0
        b_n = b.get("n", 0) or 0
        n_n = n.get("n", 0) or 0

        hit_delta = _rel_delta_pct(b_hit, n_hit)
        avg_delta = _rel_delta_pct(b_avg, n_avg)

        hit_flag = ""
        avg_flag = ""
        if b_hit > 0 and n_n >= 3 and hit_delta < -DEGRADATION_THRESHOLD_PCT:
            hit_flag = f"  ⚠️ hit_rate 劣化 {hit_delta:+.1f}%"
            has_deg = True
        if b_avg != 0 and n_n >= 3 and avg_delta < -DEGRADATION_THRESHOLD_PCT:
            avg_flag = f"  ⚠️ avg_return 劣化 {avg_delta:+.1f}%"
            has_deg = True

        line = (
            f"  {regime:14s} "
            f"hit: {b_hit:5.1f}% → {n_hit:5.1f}% ({hit_delta:+6.1f}%){hit_flag} | "
            f"avg: {b_avg:+5.2f}% → {n_avg:+5.2f}% ({avg_delta:+6.1f}%){avg_flag} | "
            f"n: {b_n}→{n_n}"
        )
        findings.append(line)

    # 2) trigger_frequencies（絕對 percentage point）
    findings.append("\n[2] A/C/F 觸發頻率（絕對 pp 變化）")
    b_tf = base.get("trigger_frequencies", {})
    n_tf = new.get("trigger_frequencies", {})
    for key in ("A", "C", "F"):
        bv = (b_tf.get(key, {}) or {}).get("pct", 0) or 0
        nv = (n_tf.get(key, {}) or {}).get("pct", 0) or 0
        delta = nv - bv
        flag = ""
        if abs(delta) >= TRIGGER_FREQ_DELTA_PP:
            flag = f"  ⚠️ 絕對變化 {delta:+.1f}pp ≥ {TRIGGER_FREQ_DELTA_PP}pp"
            has_deg = True
        findings.append(f"  {key:8s} {bv:5.1f}% → {nv:5.1f}% ({delta:+5.1f}pp){flag}")

    # 3) 看漲說漲 verdict
    findings.append("\n[3] 看漲說漲 verdict")
    b_v = (base.get("看漲說漲_quantification") or {}).get("verdict", "N/A")
    n_v = (new.get("看漲說漲_quantification") or {}).get("verdict", "N/A")
    findings.append(f"  {b_v} → {n_v}")
    # GREY_ZONE/ALPHA → BUG 視為惡化
    if b_v in ("GREY_ZONE", "ALPHA") and n_v == "BUG":
        findings.append(f"  ⚠️ verdict 惡化（{b_v} → BUG）")
        has_deg = True

    # 4) sample_sufficiency
    findings.append("\n[4] 樣本充足度")
    b_ss = (base.get("sample_sufficiency") or {}).get("rows_with_full_samples", 0)
    n_ss = (new.get("sample_sufficiency") or {}).get("rows_with_full_samples", 0)
    findings.append(f"  rows_with_full_samples: {b_ss} → {n_ss}")
    if n_ss < 10:
        findings.append(
            f"  ⚠️ 新報告 full samples {n_ss} < 10，統計顯著性不足 — "
            f"上方所有 >10% 警告需謹慎解讀（可能是樣本噪音而非真劣化）"
        )

    findings.append("\n" + "=" * 60)
    if has_deg:
        findings.append("❌ 偵測到指標劣化 → 建議回滾 v132 編排管線")
        findings.append("")
        findings.append("回滾步驟：")
        findings.append("  1. 編輯 backend/.env，把這行改為 false 或刪除：")
        findings.append("     COMPREHENSIVE_PIPELINE_ENABLED=true")
        findings.append("  2. 重啟後端（雙擊「阿斯拉量化系統.command」）")
        findings.append("  3. 「全部分析」恢復走舊 seg2 monolith")
    else:
        findings.append("✅ 未偵測到 >10% 劣化 → 編排管線可繼續啟用")
    findings.append("=" * 60)

    return has_deg, findings


def main():
    ap = argparse.ArgumentParser(description="比對兩份 shadow_mode.py 報告")
    ap.add_argument("baseline", nargs="?", help="baseline JSON 路徑（預設 shadow_report_20260514.json）")
    ap.add_argument("new", nargs="?", help="新報告 JSON 路徑（--auto 模式可省略）")
    ap.add_argument("--auto", action="store_true",
                    help="先跑 shadow_mode.py --days 14 產生新報告，再比對 baseline")
    ap.add_argument("--target-date", type=str, default=None,
                    help="守門日期 YYYY-MM-DD：今天 < 該日期 → 不執行（launchd 重複觸發用）")
    ap.add_argument("--marker", type=str, default=None,
                    help="marker 檔路徑：執行後寫入，避免 launchd 重複跑")
    args = ap.parse_args()

    # ─── 日期守門（launchd 模式）────────────────────────
    if args.target_date:
        try:
            target = datetime.strptime(args.target_date, "%Y-%m-%d").date()
        except ValueError:
            print(f"❌ --target-date 格式錯誤：{args.target_date}（需 YYYY-MM-DD）")
            sys.exit(2)
        today = date.today()
        if today < target:
            print(f"[guard] 今天 {today} < 目標日 {target}，跳過執行。")
            sys.exit(0)
        if args.marker and Path(args.marker).exists():
            print(f"[guard] marker 已存在（{args.marker}），跳過重複執行。")
            sys.exit(0)

    backend_dir = Path(__file__).resolve().parent.parent
    baseline_path = Path(args.baseline) if args.baseline else (
        backend_dir / "data" / "db" / "shadow_report_20260514.json"
    )
    if not baseline_path.exists():
        print(f"❌ baseline 不存在：{baseline_path}")
        sys.exit(1)

    if args.auto:
        new_path = _run_shadow_mode(days=14)
    elif args.new:
        new_path = Path(args.new)
    else:
        print("❌ 需提供 new 報告路徑，或加 --auto")
        sys.exit(2)
    if not new_path.exists():
        print(f"❌ new 報告不存在：{new_path}")
        sys.exit(1)

    has_deg, lines = compare(baseline_path, new_path)
    print("\n".join(lines))

    # 寫 marker（launchd 不再重複觸發）
    if args.marker:
        try:
            Path(args.marker).write_text(
                f"completed_at={datetime.now().isoformat()}\n"
                f"baseline={baseline_path}\n"
                f"new={new_path}\n"
                f"has_degradation={has_deg}\n",
                encoding="utf-8",
            )
        except Exception as e:
            print(f"[warn] 寫 marker 失敗（不致命）：{e}")

    sys.exit(1 if has_deg else 0)


if __name__ == "__main__":
    main()
