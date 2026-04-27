"""阿斯拉量化系統 — v101 模仿學習自動重訓（每週日 02:00 由 launchd 觸發）。

執行流程：
  1. Catch-up 檢查（> 7 天未跑就立刻補跑）
  2. Drift 檢查（adversarial validation）— 顯著漂移強制重訓
  3. 新樣本檢查（< 10 筆 + 沒漂移 → 跳過）
  4. 訓練（含 Champion-Challenger 拒絕條件）
  5. Auto-rollback health check
  6. Quality Gate 評估 + canary progression
  7. 老舊模型清理（disk 不爆）
  8. 寫狀態 JSON + 發 macOS 通知

執行：
  cd backend && .venv/bin/python3 scripts/retrain_imitation.py

日誌：寫到 data/db/launchd_retrain.log
狀態：寫到 data/db/imitation_status.json（前端可讀）
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))

# 狀態檔案
_BACKEND_DIR = _SCRIPT_DIR.parent
_STATUS_DIR = _BACKEND_DIR / "data" / "db"
_LAST_RUN_FILE = _STATUS_DIR / ".imitation_last_run"
_STATUS_JSON = _STATUS_DIR / "imitation_status.json"
_FAILURE_COUNT_FILE = _STATUS_DIR / ".imitation_failure_count"
_MODELS_DIR = _BACKEND_DIR / "models"


def _notify_macos(title: str, message: str) -> None:
    """macOS 通知中心發訊（不影響主流程，失敗靜默）。"""
    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                f'display notification "{message}" with title "{title}"',
            ],
            timeout=3, check=False, capture_output=True,
        )
    except Exception:
        pass


def _read_failure_count() -> int:
    if _FAILURE_COUNT_FILE.exists():
        try:
            return int(_FAILURE_COUNT_FILE.read_text().strip())
        except Exception:
            return 0
    return 0


def _write_failure_count(n: int) -> None:
    try:
        _FAILURE_COUNT_FILE.write_text(str(n))
    except Exception:
        pass


def _was_last_run_too_long_ago(threshold_days: int = 7) -> bool:
    """看是否錯過排程（catch-up 機制）。"""
    if not _LAST_RUN_FILE.exists():
        return True  # 從沒跑過
    try:
        last = datetime.fromisoformat(_LAST_RUN_FILE.read_text().strip())
        return (datetime.now() - last) > timedelta(days=threshold_days)
    except Exception:
        return True


def _write_last_run(timestamp: datetime) -> None:
    try:
        _LAST_RUN_FILE.write_text(timestamp.isoformat())
    except Exception:
        pass


def _write_status_json(payload: dict) -> None:
    try:
        _STATUS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    except Exception:
        pass


def _cleanup_old_models(keep_recent: int = 5) -> dict:
    """清理 .pkl 檔案 — 只保留：
      - Champion / Stable Fallback（永遠保留）
      - 最近 N 個版本（給 rollback 用）
      - 其餘的 .pkl 刪除（DB metrics 仍保留歷史）
    """
    from app.core.prediction_tracker import prediction_tracker
    prediction_tracker._ensure_db()
    if not prediction_tracker._conn or not _MODELS_DIR.exists():
        return {"deleted": 0, "kept": 0}

    # 找要保留的版本
    keep_versions: set[int] = set()

    # Champion + Stable Fallback
    rows = prediction_tracker._conn.execute(
        "SELECT version FROM imitation_model_metrics WHERE is_champion=1 OR is_stable_fallback=1"
    ).fetchall()
    keep_versions.update(r[0] for r in rows)

    # 最近 N 個成功訓練
    rows = prediction_tracker._conn.execute(
        """SELECT version FROM imitation_model_metrics
           WHERE status = 'activated' ORDER BY version DESC LIMIT ?""",
        (keep_recent,),
    ).fetchall()
    keep_versions.update(r[0] for r in rows)

    # 刪除其餘 .pkl（DB metrics row 不刪 — 歷史紀錄）
    deleted = 0
    kept = 0
    for pkl_file in _MODELS_DIR.glob("imitation_v*.pkl"):
        try:
            # 從檔名取 version
            name = pkl_file.stem  # 例：imitation_v4 / imitation_v4_plain
            parts = name.split("_v")
            if len(parts) < 2:
                continue
            version_part = parts[1].split("_")[0]
            version = int(version_part)
            if version in keep_versions:
                kept += 1
                continue
            pkl_file.unlink()
            deleted += 1
        except Exception:
            continue
    return {"deleted": deleted, "kept": kept, "kept_versions": sorted(keep_versions)}


def main():
    started_at = datetime.now()
    is_catchup = _was_last_run_too_long_ago(7)
    status: dict = {
        "started_at": started_at.isoformat(),
        "is_catchup": is_catchup,
        "steps": {},
    }

    print("═" * 60)
    print("  阿斯拉量化系統 — v101 模仿學習自動重訓")
    print(f"  時間：{started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    if is_catchup:
        print("  ⏰ 偵測到上次執行超過 7 天 — 補跑模式")
    print("═" * 60)

    try:
        # 1. Drift 檢查
        try:
            from app.core.drift_monitor import adversarial_validation
            drift = adversarial_validation()
            print(f"\n[1/6] Drift 偵測：{drift.get('message', drift)}")
            status["steps"]["drift"] = drift
            force_retrain = drift.get("action") == "force_retrain"
        except Exception as e:
            print(f"[1/6] Drift 偵測失敗（容忍）：{e}")
            status["steps"]["drift"] = {"error": str(e)}
            force_retrain = False

        # 2. 新樣本檢查
        new_count = 0
        try:
            from app.core.prediction_tracker import prediction_tracker
            prediction_tracker._ensure_db()
            new_count = prediction_tracker._conn.execute(
                """SELECT COUNT(*) FROM predictions
                   WHERE status IN ('hit_target', 'hit_stop')
                     AND validated_at > COALESCE(
                       (SELECT MAX(trained_at) FROM imitation_model_metrics WHERE is_champion = 1),
                       '2000-01-01'
                     )"""
            ).fetchone()[0]
            print(f"\n[2/6] 新樣本：{new_count} 筆")
            status["steps"]["new_samples"] = new_count
            if new_count < 10 and not force_retrain and not is_catchup:
                print("    新樣本 < 10 + 無漂移 + 非補跑 → 跳過訓練")
                status["steps"]["training"] = {"status": "skipped", "reason": "insufficient_new_samples"}
                _finalize(status, started_at, success=True)
                return
        except Exception as e:
            print(f"[2/6] 新樣本檢查失敗（繼續嘗試訓練）：{e}")

        # 3. 訓練
        try:
            from app.core.imitation_trainer import train_imitation_model
            result = train_imitation_model(min_samples=50, force=force_retrain)
            print(f"\n[3/6] 訓練結果：{result.get('status')}")
            status["steps"]["training"] = result
            print(json.dumps({k: v for k, v in result.items() if k != 'feature_importance'},
                             ensure_ascii=False, indent=2, default=str))
        except Exception as e:
            print(f"[3/6] 訓練失敗：{e}")
            status["steps"]["training"] = {"error": str(e)}
            _finalize(status, started_at, success=False)
            return

        # 4. Auto-rollback health
        try:
            from app.core.auto_rollback import check_v101_health
            health = check_v101_health()
            print(f"\n[4/6] Auto-rollback：{'觸發' if health.get('rollback_triggered') else '正常'}")
            status["steps"]["auto_rollback"] = health
        except Exception as e:
            print(f"[4/6] Auto-rollback 失敗：{e}")
            status["steps"]["auto_rollback"] = {"error": str(e)}

        # 5. Quality Gate + canary progression
        try:
            from app.core.v101_self_validator import evaluate_v101_readiness, canary_progression
            gate = evaluate_v101_readiness()
            print(f"\n[5/6] Quality Gate：ready={gate.get('ready')} action={gate.get('action_taken')}")
            status["steps"]["quality_gate"] = {
                "ready": gate.get("ready"),
                "failed_gates": gate.get("failed_gates"),
                "action": gate.get("action_taken"),
            }

            progression = canary_progression()
            print(f"      Canary：{progression.get('action', 'no_action')}")
            status["steps"]["canary"] = progression
        except Exception as e:
            print(f"[5/6] Quality Gate 失敗：{e}")
            status["steps"]["quality_gate"] = {"error": str(e)}

        # 6. 老舊模型清理
        try:
            cleanup = _cleanup_old_models(keep_recent=5)
            print(f"\n[6/6] 模型清理：刪 {cleanup['deleted']} 留 {cleanup['kept']}")
            status["steps"]["cleanup"] = cleanup
        except Exception as e:
            print(f"[6/6] 清理失敗：{e}")
            status["steps"]["cleanup"] = {"error": str(e)}

        _finalize(status, started_at, success=True)

    except Exception as e:
        # 捕獲總體例外
        status["fatal_error"] = str(e)
        _finalize(status, started_at, success=False)
        raise


def _finalize(status: dict, started_at: datetime, success: bool) -> None:
    """記錄結果 + 通知 + 連續失敗追蹤。"""
    finished_at = datetime.now()
    elapsed = (finished_at - started_at).total_seconds()
    status["finished_at"] = finished_at.isoformat()
    status["elapsed_sec"] = round(elapsed, 1)
    status["success"] = success

    # 連續失敗追蹤
    failures = _read_failure_count()
    if success:
        if failures > 0:
            print(f"\n✅ 從 {failures} 次失敗中恢復")
        _write_failure_count(0)
        _write_last_run(finished_at)
    else:
        failures += 1
        _write_failure_count(failures)
        status["consecutive_failures"] = failures

    status["consecutive_failures"] = failures
    _write_status_json(status)

    # 通知
    if success:
        # 成功通知（簡訊）
        train_status = status.get("steps", {}).get("training", {}).get("status", "")
        gate_ready = status.get("steps", {}).get("quality_gate", {}).get("ready", False)
        msg_parts = [f"完成 ({elapsed:.0f}s)"]
        if train_status == "activated":
            msg_parts.append(f"v{status['steps']['training'].get('version')} 上線")
        elif train_status == "skipped":
            msg_parts.append("跳過訓練")
        if gate_ready:
            msg_parts.append("Quality Gate ✓")
        _notify_macos("v101 重訓", " | ".join(msg_parts))
    else:
        if failures >= 3:
            # 連續 3 次失敗 → 醒目警告
            _notify_macos(
                "⚠️ v101 重訓連續失敗",
                f"已連續 {failures} 次失敗，請檢查 launchd_retrain.error.log",
            )
        else:
            _notify_macos("v101 重訓失敗", f"第 {failures} 次失敗，{status.get('fatal_error', '見 log')}")

    print(f"\n{'✅' if success else '❌'} 完成於 {finished_at.strftime('%H:%M:%S')}（{elapsed:.0f}s）")


if __name__ == "__main__":
    main()
