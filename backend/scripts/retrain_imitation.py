"""阿斯拉量化系統 — v101 模仿學習自動重訓（每週日 02:00 由 launchd 觸發）。

執行流程：
  1. Drift 檢查（adversarial validation）— 顯著漂移強制重訓
  2. 新樣本檢查（< 10 筆 + 沒漂移 → 跳過）
  3. 訓練（含 Champion-Challenger 拒絕條件）
  4. Auto-rollback health check
  5. Quality Gate 評估 + canary progression

執行：
  cd backend && .venv/bin/python3 scripts/retrain_imitation.py

日誌：寫到 data/db/launchd_retrain.log（launchd 配置）
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))


def main():
    print("═" * 60)
    print("  阿斯拉量化系統 — v101 模仿學習自動重訓")
    print(f"  時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 60)

    # 1. Drift 檢查
    try:
        from app.core.drift_monitor import adversarial_validation
        drift = adversarial_validation()
        print(f"\n[1/4] Drift 偵測：{drift.get('message', drift)}")
        force_retrain = drift.get("action") == "force_retrain"
    except Exception as e:
        print(f"[1/4] Drift 偵測失敗（容忍）：{e}")
        force_retrain = False

    # 2. 新樣本檢查
    try:
        from app.core.prediction_tracker import prediction_tracker
        prediction_tracker._ensure_db()
        new_count_row = prediction_tracker._conn.execute(
            """SELECT COUNT(*) FROM predictions p
               LEFT JOIN imitation_model_metrics m ON 1=1
               WHERE p.status IN ('hit_target', 'hit_stop')
                 AND p.validated_at > COALESCE(
                   (SELECT MAX(trained_at) FROM imitation_model_metrics WHERE is_champion = 1),
                   '2000-01-01'
                 )"""
        ).fetchone()
        new_count = new_count_row[0] if new_count_row else 0
        print(f"\n[2/4] 新樣本：{new_count} 筆")
        if new_count < 10 and not force_retrain:
            print("    新樣本 < 10 且無漂移 → 跳過訓練")
            return
    except Exception as e:
        print(f"[2/4] 新樣本檢查失敗（繼續嘗試訓練）：{e}")

    # 3. 訓練
    try:
        from app.core.imitation_trainer import train_imitation_model
        result = train_imitation_model(min_samples=50, force=force_retrain)
        print(f"\n[3/4] 訓練結果：{result.get('status')}")
        print(json.dumps({k: v for k, v in result.items() if k not in ('feature_importance',)},
                         ensure_ascii=False, indent=2, default=str))
    except Exception as e:
        print(f"[3/4] 訓練失敗：{e}")
        return

    # 4. Auto-rollback health + Quality Gate
    try:
        from app.core.auto_rollback import check_v101_health
        health = check_v101_health()
        print(f"\n[4a/4] Auto-rollback：{health.get('rollback_triggered', '正常')}")

        from app.core.v101_self_validator import evaluate_v101_readiness, canary_progression
        gate = evaluate_v101_readiness()
        print(f"[4b/4] Quality Gate：ready={gate.get('ready')} action={gate.get('action_taken')}")

        progression = canary_progression()
        print(f"[4c/4] Canary：{progression}")
    except Exception as e:
        print(f"[4/4] 健康檢查 / Quality Gate 失敗：{e}")

    print(f"\n✅ 完成：{datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
