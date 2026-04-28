"""阿斯拉量化系統 — v104 Fix G：結論卡分布量化驗證。

不實際呼叫 LLM（成本太高），改用「規則模擬」測試 v104 規則層是否會把雙向比例壓下來。

流程：
1. 從 backend/data/ohlcv/*.csv 抽 N 個歷史時點
2. 每個時點重建 chart_state（regime + regime_subtype + bias_score）
3. 套 v104 6 規則 → 預測該時點 LLM 應該選哪種結論卡
4. 統計分布

通過標準：
- 雙向比例 < 30%（修正前 60-70%）
- 有方向卡（做多 + 做空 + 偏多 + 偏空）合計 > 50%
- 觀望 < 10%
- 偏多 + 偏空 > 15%（lean 卡有實際使用）

執行：
    cd backend && .venv/bin/python3 scripts/eval_conclusion_distribution.py [--n 50]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))

import pandas as pd  # noqa: E402

from app.core.regime_filter import classify_regime_at_bar  # noqa: E402
from app.core.regime_subtype import classify_ranging_subtype  # noqa: E402


def select_card(regime_info: dict, subtype_info: dict | None) -> str:
    """v104 6 規則：給定 regime + subtype，回傳該選哪張結論卡。

    return: "long" | "short" | "lean_long" | "lean_short" | "bilateral" | "observe"
    """
    regime = regime_info.get("regime", "unknown")
    confidence = float(regime_info.get("confidence", 0))

    # 1. 信心極低 → 觀望
    if confidence < 0.25:
        return "observe"

    # 2-3. trending 高信心
    if regime == "trending_up" and confidence >= 0.6:
        return "long"
    if regime == "trending_down" and confidence >= 0.6:
        return "short"

    # 4-5. trending 中信心 → lean
    if regime == "trending_up" and confidence >= 0.4:
        return "lean_long"
    if regime == "trending_down" and confidence >= 0.4:
        return "lean_short"

    # 6-10. ranging / unknown 都用 subtype 分流
    if regime in ("ranging", "unknown"):
        if not subtype_info:
            return "bilateral"  # subtype 缺失 fallback
        sub = subtype_info.get("subtype")
        if sub in ("true_ranging", "breakout_pending"):
            return "bilateral"
        if sub == "lean_long":
            return "lean_long"
        if sub == "lean_short":
            return "lean_short"
        if sub == "neutral_ranging":
            # 任何非零 bias 都用 lean（避免不必要的雙向）；只有 bias 完全 0 才 bilateral
            bias = (subtype_info.get("metrics") or {}).get("bias_score") or 0
            if bias > 0.0:
                return "lean_long"
            elif bias < 0.0:
                return "lean_short"
            return "bilateral"  # bias 真的是 0 才回退

    # 其他 regime（high_vol / low_vol）
    return "bilateral"


def select_card_old(regime_info: dict) -> str:
    """v103 4 規則（修正前）— 給定 regime，回傳該選哪張結論卡。"""
    regime = regime_info.get("regime", "unknown")
    confidence = float(regime_info.get("confidence", 0))

    if confidence < 0.3:
        return "observe"
    if regime == "trending_up" and confidence >= 0.6:
        return "long"
    if regime == "trending_down" and confidence >= 0.6:
        return "short"
    if regime == "ranging":
        return "bilateral"
    return "bilateral"  # 其他都進雙向


def _fake_chart_state(df: pd.DataFrame) -> dict:
    """組最小 chart_state 給 regime_subtype 用（沒有 external_signals/breadth）。"""
    return {}


def evaluate(n_samples: int, seed: int = 42) -> dict:
    """掃 OHLCV CSV 抽 n 個歷史時點，統計舊 vs 新規則的卡分布。"""
    rng = random.Random(seed)
    ohlcv_dir = _SCRIPT_DIR.parent / "data" / "ohlcv"
    files = sorted(
        f for f in ohlcv_dir.glob("*_4h.csv")
        if "derivatives" not in f.name and "USDT" in f.name
    )

    if not files:
        return {"error": "找不到 OHLCV 檔案"}

    samples_old: list[str] = []
    samples_new: list[str] = []
    breakdown_new = Counter()

    attempted = 0
    while len(samples_new) < n_samples and attempted < n_samples * 3:
        attempted += 1
        f = rng.choice(files)
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        if df.empty or len(df) < 200:
            continue

        # 隨機抽一根 K 線（保證前面有 100+ 根歷史可用）
        bar_idx = rng.randint(150, len(df) - 1)
        df_window = df.iloc[: bar_idx + 1].reset_index(drop=True)

        try:
            ri = classify_regime_at_bar(df_window, len(df_window) - 1)
        except Exception:
            continue

        regime = ri.get("regime", "unknown")
        sub = None
        if regime in ("ranging", "unknown"):
            try:
                sub = classify_ranging_subtype(df_window, ri, _fake_chart_state(df_window))
            except Exception:
                sub = None

        old_card = select_card_old(ri)
        new_card = select_card(ri, sub)

        samples_old.append(old_card)
        samples_new.append(new_card)
        breakdown_new[new_card] += 1
        # 細分 ranging subtype
        if regime == "ranging" and sub:
            breakdown_new[f"  ranging→{sub.get('subtype')}"] += 1

    return {
        "n_samples": len(samples_new),
        "old_dist": dict(Counter(samples_old)),
        "new_dist": dict(Counter(samples_new)),
        "breakdown_with_subtype": dict(breakdown_new),
    }


def _pct(d: dict, total: int) -> dict:
    return {k: f"{v} ({v / total * 100:.1f}%)" for k, v in sorted(d.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="抽幾個歷史時點")
    ap.add_argument("--json", action="store_true", help="JSON 輸出")
    args = ap.parse_args()

    print("═" * 60)
    print(f"  v104 結論卡分布量化驗證（n={args.n}）")
    print("═" * 60)

    result = evaluate(args.n)
    if "error" in result:
        print(f"❌ {result['error']}")
        sys.exit(1)

    n = result["n_samples"]
    print(f"\n樣本數：{n}\n")

    print("─── 舊規則（v103 4 規則）───")
    for k, v in _pct(result["old_dist"], n).items():
        print(f"  {k:20s}: {v}")

    print("\n─── 新規則（v104 6 規則）───")
    for k, v in _pct(result["new_dist"], n).items():
        print(f"  {k:20s}: {v}")

    print("\n─── 細分（含 ranging subtype）───")
    for k, v in result["breakdown_with_subtype"].items():
        if k.startswith("  "):
            print(f"  {k}: {v} ({v / n * 100:.1f}%)")

    # 通過標準檢查
    new = result["new_dist"]
    bilateral_pct = new.get("bilateral", 0) / n * 100
    direction_pct = sum(new.get(k, 0) for k in ("long", "short", "lean_long", "lean_short")) / n * 100
    observe_pct = new.get("observe", 0) / n * 100
    lean_pct = sum(new.get(k, 0) for k in ("lean_long", "lean_short")) / n * 100

    print("\n─── 通過標準 ───")
    checks = [
        (f"雙向 < 30%", bilateral_pct < 30, f"{bilateral_pct:.1f}%"),
        (f"方向卡 > 50%", direction_pct > 50, f"{direction_pct:.1f}%"),
        (f"觀望 < 10%", observe_pct < 10, f"{observe_pct:.1f}%"),
        (f"偏多+偏空 > 15%", lean_pct > 15, f"{lean_pct:.1f}%"),
    ]
    all_pass = True
    for name, ok, val in checks:
        mark = "✅" if ok else "❌"
        print(f"  {mark} {name}: {val}")
        if not ok:
            all_pass = False

    print(f"\n{'✅ 全部通過' if all_pass else '⚠️ 部分未達標 — 規則或 subtype 分類器可能要調整'}")

    if args.json:
        print("\n" + json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
