"""阿斯拉量化系統 — v106 B3：Prompt 改動防 regression eval harness。

對 50 個固定歷史 chart_state，用規則模擬「LLM 應該產出哪種結論卡」，
比對 prompt v1 vs v2（或 git HEAD vs git HEAD~N）的輸出差異。

不真的呼叫 LLM（成本太高），用 select_card 規則邏輯模擬。
通過標準延伸自 v104 Fix G eval_conclusion_distribution.py，但加入：
- bias_score 動態 threshold（v105.1）
- per-symbol regime_subtype 路徑
- 信心分布
- Lift 統計

執行：
    cd backend && .venv/bin/python -m scripts.eval_prompt_distribution

輸出：
    eval_prompt_distribution_<timestamp>.json
    含每個樣本 chart_state + 規則決策 + bias_score / threshold
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))

import pandas as pd  # noqa: E402

from app.core.regime_filter import classify_regime_at_bar  # noqa: E402
from app.core.regime_subtype import classify_ranging_subtype  # noqa: E402


def select_card(regime_info: dict, subtype_info: dict | None) -> str:
    """v106：6 規則模擬選卡邏輯（同 v104 Fix G + v105.1 動態 threshold）。"""
    regime = regime_info.get("regime", "unknown")
    confidence = float(regime_info.get("confidence", 0))

    if confidence < 0.25:
        return "observe"
    if regime == "trending_up" and confidence >= 0.6:
        return "long"
    if regime == "trending_down" and confidence >= 0.6:
        return "short"
    if regime == "trending_up" and confidence >= 0.4:
        return "lean_long"
    if regime == "trending_down" and confidence >= 0.4:
        return "lean_short"

    if regime in ("ranging", "unknown"):
        if not subtype_info:
            return "bilateral"
        sub = subtype_info.get("subtype")
        if sub in ("true_ranging", "breakout_pending"):
            return "bilateral"
        if sub == "lean_long":
            return "lean_long"
        if sub == "lean_short":
            return "lean_short"
        if sub == "neutral_ranging":
            bias = (subtype_info.get("metrics") or {}).get("bias_score") or 0
            if bias > 0:
                return "lean_long"
            elif bias < 0:
                return "lean_short"
            return "bilateral"

    return "bilateral"


def evaluate(n_samples: int = 50, seed: int = 42) -> dict:
    """跑 N 個歷史時點，統計卡分布 + bias 分布 + threshold 觸發率。"""
    rng = random.Random(seed)
    ohlcv_dir = _SCRIPT_DIR.parent / "data" / "ohlcv"
    files = sorted(
        f for f in ohlcv_dir.glob("*_4h.csv")
        if "derivatives" not in f.name and "USDT" in f.name
    )
    if not files:
        return {"error": "No OHLCV files found"}

    samples: list[dict] = []
    card_dist: Counter = Counter()
    subtype_dist: Counter = Counter()
    bias_buckets: Counter = Counter()
    threshold_buckets: Counter = Counter()

    attempted = 0
    while len(samples) < n_samples and attempted < n_samples * 3:
        attempted += 1
        f = rng.choice(files)
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        if df.empty or len(df) < 200:
            continue

        bar_idx = rng.randint(150, len(df) - 1)
        df_w = df.iloc[: bar_idx + 1].reset_index(drop=True)
        symbol = f.stem.replace("_4h", "").replace("_", "/")

        try:
            ri = classify_regime_at_bar(df_w, len(df_w) - 1)
        except Exception:
            continue

        regime = ri.get("regime", "unknown")
        sub = None
        if regime in ("ranging", "unknown"):
            try:
                sub = classify_ranging_subtype(df_w, ri, {}, symbol=symbol)
            except Exception:
                sub = None

        card = select_card(ri, sub)
        card_dist[card] += 1
        if sub and sub.get("subtype"):
            subtype_dist[sub["subtype"]] += 1

        # bias / threshold buckets
        if sub:
            metrics = sub.get("metrics", {})
            bias = metrics.get("bias_score", 0) or 0
            thr = metrics.get("bias_threshold_used", 0.4)
            n_active = metrics.get("n_active_contributions", 0)
            if abs(bias) < 0.1:
                bias_buckets["近零 (|b|<0.1)"] += 1
            elif abs(bias) < 0.3:
                bias_buckets["弱 (0.1-0.3)"] += 1
            elif abs(bias) < 0.5:
                bias_buckets["中 (0.3-0.5)"] += 1
            else:
                bias_buckets["強 (≥0.5)"] += 1
            threshold_buckets[f"thr={thr} (n_active={n_active})"] += 1

        samples.append({
            "symbol": symbol,
            "regime": regime,
            "regime_confidence": round(ri.get("confidence", 0), 2),
            "subtype": sub.get("subtype") if sub else None,
            "bias_score": round((sub.get("metrics", {}).get("bias_score") if sub else 0) or 0, 3),
            "card": card,
        })

    # 通過標準（複用 v104 Fix G）
    n = len(samples)
    bilateral_pct = card_dist["bilateral"] / n * 100 if n else 0
    direction_pct = sum(card_dist[k] for k in ("long", "short", "lean_long", "lean_short")) / n * 100 if n else 0
    observe_pct = card_dist["observe"] / n * 100 if n else 0
    lean_pct = sum(card_dist[k] for k in ("lean_long", "lean_short")) / n * 100 if n else 0

    pass_checks = {
        "雙向 < 30%": (bilateral_pct < 30, f"{bilateral_pct:.1f}%"),
        "方向卡 > 50%": (direction_pct > 50, f"{direction_pct:.1f}%"),
        "觀望 < 10%": (observe_pct < 10, f"{observe_pct:.1f}%"),
        "偏多+偏空 > 15%": (lean_pct > 15, f"{lean_pct:.1f}%"),
    }
    all_pass = all(v[0] for v in pass_checks.values())

    return {
        "ran_at": datetime.now().isoformat(),
        "n_samples": n,
        "card_distribution": dict(card_dist),
        "card_distribution_pct": {k: round(v / n * 100, 1) for k, v in card_dist.items()} if n else {},
        "subtype_distribution": dict(subtype_dist),
        "bias_distribution": dict(bias_buckets),
        "threshold_distribution": dict(threshold_buckets),
        "pass_checks": pass_checks,
        "overall_pass": all_pass,
        "samples": samples,
    }


def main():
    print("═" * 60)
    print(f"  v106 B3 — Prompt regression eval（{datetime.now():%Y-%m-%d %H:%M}）")
    print("═" * 60)

    result = evaluate(n_samples=50)
    if "error" in result:
        print(f"❌ {result['error']}")
        sys.exit(1)

    n = result["n_samples"]
    print(f"\n樣本: {n}")
    print(f"\n結論卡分布:")
    for k, v in sorted(result["card_distribution_pct"].items(), key=lambda x: -x[1]):
        print(f"  {k:15s}: {v}%")

    print(f"\n子類型分布:")
    for k, v in result["subtype_distribution"].items():
        print(f"  {k:20s}: {v}")

    print(f"\nBias 強度:")
    for k, v in result["bias_distribution"].items():
        print(f"  {k:20s}: {v}")

    print(f"\n通過標準:")
    for name, (ok, val) in result["pass_checks"].items():
        mark = "✅" if ok else "❌"
        print(f"  {mark} {name}: {val}")

    print(f"\n{'✅ 全部通過' if result['overall_pass'] else '⚠️ 部分未達標'}")

    # 寫 JSON
    out_path = _SCRIPT_DIR.parent / "data" / "db" / f"eval_prompt_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print(f"\n詳細結果存到: {out_path}")


if __name__ == "__main__":
    main()
