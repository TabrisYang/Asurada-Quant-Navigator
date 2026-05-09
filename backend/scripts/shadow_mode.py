"""阿斯拉量化系統 — Shadow Mode（pre-flight check 用）。

回放 predictions.db 中的歷史紀錄，對每筆 prediction 計算「當時若採用 A/C/F
改動規則會發生什麼」，並統計：

  1. 「看漲說漲」量化指標：
     - direction_balance.biased_long=true 的 case 中，系統實際給 LONG 的比例
     - 這類 case 的平均 actual_outcome_pct + 命中率
     - 若報酬 < 0% → 確認是 bug，A 改動有依據
     - 若報酬 > 0% → 警告這是 alpha，A 改動需重新評估

  2. A / C / F 各自的觸發頻率（每筆 prediction 是否會被觸發）。

  3. 各 regime_std 的勝率分布（baseline）。

執行：
    python3 backend/scripts/shadow_mode.py --days 90 --symbol ETH/USDT

輸出：
    stdout 文字摘要 + JSON 報告（shadow_report_YYYYMMDD.json）

設計：
    本腳本不重跑 LLM，僅以 SQL 重現「當時的 direction_balance / regime_warning /
    combo_stats」三個統計訊號（as-of 計算），然後判定 A/C/F 觸發狀態。
    這比重跑分析便宜 1000 倍，且足以驗證觸發規則的歷史頻率與報酬分布。

    A/C/F 觸發定義（與 plan witty-scribbling-fog 一致）：
      A：direction_balance.biased + regime_warning.win_rate<50%(n>=10)
         + combo_stats.win_rate<50%(n>=10) 三條件全成立
      C：regime_std == 'ranging' AND direction in ('long', 'short')
         （C 真實版需 RSI/EMA60 連續同向資料；本腳本以 ranging+有方向偏向作近似）
      F：direction_balance.biased_long OR biased_short
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _SCRIPT_DIR.parent
_DB = _BACKEND_DIR / "data" / "db" / "predictions.db"

# 觸發門檻（與 plan 一致）
BIAS_LONG_THRESHOLD = 75.0
BIAS_SHORT_THRESHOLD = 25.0
DIR_BALANCE_MIN_SAMPLES = 10
REGIME_WARN_MIN_SAMPLES = 10
COMBO_MIN_SAMPLES = 10
WIN_RATE_THRESHOLD = 50.0


def _open_db() -> sqlite3.Connection:
    if not _DB.exists():
        print(f"✗ 找不到 DB：{_DB}", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(_DB)
    conn.row_factory = sqlite3.Row
    return conn


def _direction_balance_as_of(
    conn: sqlite3.Connection, symbol: str, as_of: str, lookback_days: int = 30
) -> dict:
    """重現 prediction_tracker.get_direction_stats 但以 as_of 為基準時間。

    Returns: {"long_n": int, "short_n": int, "long_pct": float,
              "biased_long": bool, "biased_short": bool, "samples": int}
    """
    cutoff_dt = datetime.fromisoformat(as_of) - timedelta(days=lookback_days)
    rows = conn.execute(
        "SELECT direction FROM predictions "
        "WHERE symbol = ? AND created_at > ? AND created_at < ? "
        "AND status NOT IN ('active', 'invalidated')",
        (symbol, cutoff_dt.isoformat(), as_of),
    ).fetchall()
    long_n = sum(1 for r in rows if r["direction"] == "long")
    short_n = sum(1 for r in rows if r["direction"] == "short")
    total = long_n + short_n
    if total < DIR_BALANCE_MIN_SAMPLES:
        return {
            "long_n": long_n, "short_n": short_n,
            "long_pct": 0.0, "samples": total,
            "biased_long": False, "biased_short": False,
        }
    long_pct = long_n / total * 100
    return {
        "long_n": long_n, "short_n": short_n,
        "long_pct": round(long_pct, 1), "samples": total,
        "biased_long": long_pct > BIAS_LONG_THRESHOLD,
        "biased_short": long_pct < BIAS_SHORT_THRESHOLD,
    }


def _regime_warning_as_of(
    conn: sqlite3.Connection, symbol: Optional[str], regime_std: Optional[str],
    as_of: str, lookback_days: int = 90,
) -> dict:
    """重現 regime class 命中率（規則同 prediction_tracker.get_regime_class_stats）。

    Returns: {"win_rate": float, "samples": int}
    """
    if not regime_std or regime_std == "unknown":
        return {"win_rate": 0.0, "samples": 0}
    cutoff_dt = datetime.fromisoformat(as_of) - timedelta(days=lookback_days)
    sql = (
        "SELECT status FROM predictions "
        "WHERE regime_std = ? AND created_at > ? AND created_at < ? "
        "AND status IN ('hit_target', 'hit_stop')"
    )
    params: list = [regime_std, cutoff_dt.isoformat(), as_of]
    if symbol:
        sql += " AND symbol = ?"
        params.append(symbol)
    rows = conn.execute(sql, params).fetchall()
    if not rows:
        return {"win_rate": 0.0, "samples": 0}
    wins = sum(1 for r in rows if r["status"] == "hit_target")
    return {
        "win_rate": round(wins / len(rows) * 100, 1),
        "samples": len(rows),
    }


def _combo_stats_as_of(
    conn: sqlite3.Connection, symbol: Optional[str], buckets: dict,
    as_of: str, lookback_days: int = 90,
) -> dict:
    """重現訊號組合命中率。對 buckets 中每個非 UNKNOWN bucket 做 AND match。"""
    filtered = {k: v for k, v in buckets.items() if v and v != "UNKNOWN"}
    if not filtered:
        return {"win_rate": 0.0, "samples": 0}
    cutoff_dt = datetime.fromisoformat(as_of) - timedelta(days=lookback_days)
    sql = (
        "SELECT status FROM predictions "
        "WHERE buckets_json IS NOT NULL "
        "AND created_at > ? AND created_at < ? "
        "AND status IN ('hit_target', 'hit_stop')"
    )
    params: list = [cutoff_dt.isoformat(), as_of]
    for sig, bucket in filtered.items():
        sql += " AND json_extract(buckets_json, '$.' || ?) = ?"
        params.extend([sig, bucket])
    if symbol:
        sql += " AND symbol = ?"
        params.append(symbol)
    rows = conn.execute(sql, params).fetchall()
    if not rows:
        return {"win_rate": 0.0, "samples": 0}
    wins = sum(1 for r in rows if r["status"] == "hit_target")
    return {
        "win_rate": round(wins / len(rows) * 100, 1),
        "samples": len(rows),
    }


def analyze(
    days: int, symbol: Optional[str] = None,
) -> tuple[list[dict], dict]:
    """對近 N 天的 prediction 跑 shadow analysis。回傳 (per_row, summary)。"""
    conn = _open_db()
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    sql = (
        "SELECT id, created_at, symbol, timeframe, direction, regime_std, "
        "buckets_json, status, actual_outcome_pct "
        "FROM predictions "
        "WHERE created_at > ? AND status NOT IN ('active', 'invalidated') "
        "ORDER BY created_at"
    )
    params: list = [cutoff]
    if symbol:
        sql = sql.replace("WHERE created_at > ?", "WHERE created_at > ? AND symbol = ?")
        params = [cutoff, symbol]
    rows = conn.execute(sql, params).fetchall()

    per_row: list[dict] = []
    for r in rows:
        buckets = {}
        if r["buckets_json"]:
            try:
                buckets = json.loads(r["buckets_json"])
            except json.JSONDecodeError:
                pass

        dir_bal = _direction_balance_as_of(conn, r["symbol"], r["created_at"])
        regime_warn = _regime_warning_as_of(
            conn, r["symbol"], r["regime_std"], r["created_at"],
        )
        combo = _combo_stats_as_of(
            conn, r["symbol"], buckets, r["created_at"],
        )

        biased = dir_bal["biased_long"] or dir_bal["biased_short"]

        # A：三條件全成立
        a_trigger = (
            biased
            and regime_warn["samples"] >= REGIME_WARN_MIN_SAMPLES
            and regime_warn["win_rate"] < WIN_RATE_THRESHOLD
            and combo["samples"] >= COMBO_MIN_SAMPLES
            and combo["win_rate"] < WIN_RATE_THRESHOLD
        )
        # C：regime=ranging 且該預測有方向（近似版）
        c_trigger = (
            r["regime_std"] == "ranging"
            and r["direction"] in ("long", "short")
        )
        # F：偏向觸發
        f_trigger = biased

        # 判斷該 prediction 是否「順勢」（biased 同向）
        is_pro_trend = False
        if dir_bal["biased_long"] and r["direction"] == "long":
            is_pro_trend = True
        elif dir_bal["biased_short"] and r["direction"] == "short":
            is_pro_trend = True

        per_row.append({
            "id": r["id"],
            "created_at": r["created_at"],
            "symbol": r["symbol"],
            "timeframe": r["timeframe"],
            "direction": r["direction"],
            "regime_std": r["regime_std"],
            "status": r["status"],
            "actual_outcome_pct": r["actual_outcome_pct"],
            "direction_balance": dir_bal,
            "regime_warning": regime_warn,
            "combo_stats": combo,
            "would_trigger_A": a_trigger,
            "would_trigger_C": c_trigger,
            "would_trigger_F": f_trigger,
            "is_pro_trend_during_bias": is_pro_trend,
        })

    summary = _summarize(per_row)
    conn.close()
    return per_row, summary


def _summarize(per_row: list[dict]) -> dict:
    n = len(per_row)
    if n == 0:
        return {"total": 0, "note": "no validated predictions in window"}

    a_n = sum(1 for r in per_row if r["would_trigger_A"])
    c_n = sum(1 for r in per_row if r["would_trigger_C"])
    f_n = sum(1 for r in per_row if r["would_trigger_F"])

    biased_cases = [r for r in per_row if r["would_trigger_F"]]
    pro_trend_cases = [r for r in biased_cases if r["is_pro_trend_during_bias"]]

    def _avg_outcome(rows: list[dict]) -> float:
        vals = [r["actual_outcome_pct"] for r in rows if r["actual_outcome_pct"] is not None]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    def _hit_rate(rows: list[dict]) -> float:
        decided = [r for r in rows if r["status"] in ("hit_target", "hit_stop")]
        if not decided:
            return 0.0
        wins = sum(1 for r in decided if r["status"] == "hit_target")
        return round(wins / len(decided) * 100, 1)

    a_cases = [r for r in per_row if r["would_trigger_A"]]

    by_regime: dict = defaultdict(list)
    for r in per_row:
        by_regime[r["regime_std"] or "unknown"].append(r)
    regime_perf = {
        rg: {
            "n": len(rows),
            "hit_rate": _hit_rate(rows),
            "avg_outcome": _avg_outcome(rows),
        }
        for rg, rows in by_regime.items()
    }

    bug_or_alpha_avg = _avg_outcome(pro_trend_cases)

    return {
        "total": n,
        "trigger_frequencies": {
            "A": {"n": a_n, "pct": round(a_n / n * 100, 1)},
            "C": {"n": c_n, "pct": round(c_n / n * 100, 1)},
            "F": {"n": f_n, "pct": round(f_n / n * 100, 1)},
        },
        "看漲說漲_quantification": {
            "biased_cases_total": len(biased_cases),
            "pro_trend_count": len(pro_trend_cases),
            "pro_trend_ratio_pct": (
                round(len(pro_trend_cases) / len(biased_cases) * 100, 1)
                if biased_cases else 0.0
            ),
            "pro_trend_avg_outcome_pct": bug_or_alpha_avg,
            "pro_trend_hit_rate_pct": _hit_rate(pro_trend_cases),
            "verdict": (
                "BUG" if bug_or_alpha_avg < -2.0
                else "ALPHA" if bug_or_alpha_avg > 2.0
                else "GREY_ZONE"
            ),
        },
        "A_trigger_cases_performance": {
            "n": len(a_cases),
            "avg_outcome_pct": _avg_outcome(a_cases),
            "hit_rate_pct": _hit_rate(a_cases),
            "note": (
                "若 hit_rate < 50% → A 防護有效，這些 case 確實該被攔下"
                if _hit_rate(a_cases) < 50.0 and len(a_cases) >= 5
                else "樣本不足或命中率高，A 觸發條件可能太鬆"
            ),
        },
        "by_regime": regime_perf,
        "sample_sufficiency": {
            "direction_balance_min": DIR_BALANCE_MIN_SAMPLES,
            "regime_warn_min": REGIME_WARN_MIN_SAMPLES,
            "combo_min": COMBO_MIN_SAMPLES,
            "rows_with_full_samples": sum(
                1 for r in per_row
                if r["direction_balance"]["samples"] >= DIR_BALANCE_MIN_SAMPLES
                and r["regime_warning"]["samples"] >= REGIME_WARN_MIN_SAMPLES
                and r["combo_stats"]["samples"] >= COMBO_MIN_SAMPLES
            ),
        },
    }


def _print_report(summary: dict, days: int, symbol: Optional[str]) -> None:
    print()
    print("═" * 64)
    print(f"  Shadow Mode 報告（近 {days} 天{', symbol=' + symbol if symbol else ''}）")
    print("═" * 64)

    if summary["total"] == 0:
        print(f"\n  ⚠ 視窗內無已驗證預測：{summary.get('note')}")
        return

    print(f"\n  已驗證樣本：{summary['total']} 筆")

    print("\n  ── A/C/F 觸發頻率 ──")
    tf = summary["trigger_frequencies"]
    print(f"    A（三防線同時失效）: {tf['A']['n']} 筆 ({tf['A']['pct']}%)")
    print(f"    C（ranging+有方向）: {tf['C']['n']} 筆 ({tf['C']['pct']}%)")
    print(f"    F（biased_long/short）: {tf['F']['n']} 筆 ({tf['F']['pct']}%)")

    print("\n  ── 「看漲說漲」量化 ──")
    q = summary["看漲說漲_quantification"]
    print(f"    biased 期間 case 數：{q['biased_cases_total']}")
    print(f"    其中順勢比例：{q['pro_trend_ratio_pct']}% ({q['pro_trend_count']} 筆)")
    print(f"    順勢 case 平均報酬：{q['pro_trend_avg_outcome_pct']}%")
    print(f"    順勢 case 命中率：{q['pro_trend_hit_rate_pct']}%")
    verdict_label = {
        "BUG": "✅ 確認是 bug，A 改動有依據（順勢虧錢）",
        "ALPHA": "❌ 這是 alpha 不是 bug，A 改動會傷邊際",
        "GREY_ZONE": "⚠️ 灰色地帶，建議 A 觸發門檻調嚴",
    }
    print(f"    判定：{verdict_label.get(q['verdict'], q['verdict'])}")

    print("\n  ── A 觸發 case 表現（驗證 A 是否該攔） ──")
    a = summary["A_trigger_cases_performance"]
    print(f"    n={a['n']}, 平均報酬={a['avg_outcome_pct']}%, 命中率={a['hit_rate_pct']}%")
    print(f"    {a['note']}")

    print("\n  ── 各 regime 表現 ──")
    for rg, perf in sorted(
        summary["by_regime"].items(), key=lambda x: -x[1]["n"]
    ):
        print(f"    {rg or 'unknown':15s}: n={perf['n']:3d}, hit={perf['hit_rate']:5.1f}%, avg={perf['avg_outcome']:+.2f}%")

    print("\n  ── 樣本充足度 ──")
    s = summary["sample_sufficiency"]
    print(f"    三條件樣本都 ≥ 門檻的 case：{s['rows_with_full_samples']} / {summary['total']}")
    if s["rows_with_full_samples"] < 10:
        print(f"    ⚠ 警告：足樣本 case <10，A/C/F 觸發頻率統計不夠可靠，需收集更多資料")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Shadow Mode：回放歷史 predictions，量化 A/C/F 改動的觸發頻率與報酬影響"
    )
    parser.add_argument("--days", type=int, default=90, help="回放天數（預設 90）")
    parser.add_argument("--symbol", type=str, default=None, help="目標 symbol（如 ETH/USDT），預設全部")
    parser.add_argument("--output", type=str, default=None, help="JSON 輸出路徑（預設 shadow_report_YYYYMMDD.json 在 backend/data/db/）")
    args = parser.parse_args()

    per_row, summary = analyze(args.days, args.symbol)

    out_path = (
        Path(args.output) if args.output
        else _BACKEND_DIR / "data" / "db" / f"shadow_report_{datetime.now():%Y%m%d}.json"
    )
    out_path.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(),
        "days": args.days,
        "symbol": args.symbol,
        "summary": summary,
        "per_row": per_row,
    }, ensure_ascii=False, indent=2))

    _print_report(summary, args.days, args.symbol)
    print(f"  JSON 報告：{out_path}")
    print()


if __name__ == "__main__":
    main()
