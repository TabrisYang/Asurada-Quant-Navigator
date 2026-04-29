"""v105 A1 + C1：對既有 predictions 補 horizon_class + regime_std。

用既有 timeframe_hours 算 horizon_class，用 regime_mapping 算 regime_std。
跑一次即可（idempotent — 重跑不會傷資料）。
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))


def main():
    from app.core.prediction_tracker import prediction_tracker
    from app.core.regime_mapping import standardize_regime

    prediction_tracker._ensure_db()
    conn = prediction_tracker._conn

    rows = conn.execute(
        "SELECT id, timeframe_hours, regime FROM predictions"
    ).fetchall()
    print(f"總 predictions: {len(rows)}")

    updated_h = 0
    updated_r = 0
    horizon_dist = {"short": 0, "medium": 0, "long": 0}
    regime_dist: dict[str, int] = {}

    for r in rows:
        pred_id = r["id"]
        h = prediction_tracker.classify_horizon(r["timeframe_hours"])
        rs = standardize_regime(r["regime"])
        horizon_dist[h] = horizon_dist.get(h, 0) + 1
        regime_dist[rs] = regime_dist.get(rs, 0) + 1
        conn.execute(
            "UPDATE predictions SET horizon_class=?, regime_std=? WHERE id=?",
            (h, rs, pred_id),
        )
        updated_h += 1
        updated_r += 1

    conn.commit()
    print(f"更新 horizon_class: {updated_h} 筆")
    print(f"更新 regime_std: {updated_r} 筆")
    print()
    print("─── horizon 分布 ───")
    for k, v in horizon_dist.items():
        print(f"  {k}: {v}")
    print()
    print("─── regime_std 分布 ───")
    for k, v in sorted(regime_dist.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    # 看可訓練樣本中各 regime 的數量
    print()
    print("─── 可訓練樣本（hit_target+hit_stop）按 regime_std ───")
    rows = conn.execute(
        "SELECT regime_std, COUNT(*) FROM predictions "
        "WHERE status IN ('hit_target', 'hit_stop') GROUP BY regime_std"
    ).fetchall()
    for r in rows:
        print(f"  {r[0]}: {r[1]} 筆")


if __name__ == "__main__":
    main()
