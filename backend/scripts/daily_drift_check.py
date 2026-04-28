"""阿斯拉量化系統 — v103 3B：每日 quick drift check（短窗 PSI）。

每天 03:00 由 launchd 觸發。

執行流程：
  1. 取最近 5 天 vs 之前 30 天的 prediction_features
  2. 對每個特徵算 PSI
  3. max PSI > 0.20 → 立刻觸發 train_imitation_model(force=True)
     （不等週日 retrain，regime shift 反應從 7 天縮到 24 小時）
  4. 寫狀態 + 漂移時 macOS 通知（穩定時不通知，避免噪音）

執行：
  cd backend && .venv/bin/python3 scripts/daily_drift_check.py

日誌：data/db/launchd_drift.log
狀態：data/db/daily_drift_status.json
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))

_BACKEND_DIR = _SCRIPT_DIR.parent
_STATUS_DIR = _BACKEND_DIR / "data" / "db"
_STATUS_JSON = _STATUS_DIR / "daily_drift_status.json"

PSI_TRIGGER = 0.20  # max PSI > 0.20 → 強制重訓
RECENT_DAYS = 5     # 短窗
REFERENCE_DAYS = 30  # 對照窗（recent 之前的 30 天）


def _notify(title: str, message: str) -> None:
    try:
        subprocess.run(
            ["osascript", "-e", f'display notification "{message}" with title "{title}"'],
            timeout=3, check=False, capture_output=True,
        )
    except Exception:
        pass


def _load_features() -> pd.DataFrame:
    from app.core.prediction_tracker import prediction_tracker

    prediction_tracker._ensure_db()
    if not prediction_tracker._conn:
        return pd.DataFrame()

    cursor = prediction_tracker._conn.execute(
        """SELECT pf.*, p.created_at AS p_created_at
           FROM prediction_features pf
           JOIN predictions p ON pf.prediction_id = p.id"""
    )
    cols = [d[0] for d in cursor.description]
    rows = cursor.fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=cols)
    df["created_at"] = pd.to_datetime(df["p_created_at"], errors="coerce")
    return df


def main():
    started = datetime.now()
    status: dict = {"started_at": started.isoformat()}

    print("═" * 60)
    print("  阿斯拉量化系統 — v103 每日 drift check")
    print(f"  時間：{started.strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 60)

    try:
        from app.core.feature_extractor import FEATURE_COLUMNS
        from app.core.drift_monitor import compute_psi
    except Exception as e:
        status["error"] = f"import 失敗：{e}"
        _STATUS_JSON.write_text(json.dumps(status, ensure_ascii=False, indent=2, default=str))
        print(f"❌ import 失敗：{e}")
        return

    df = _load_features()
    if df.empty:
        status["result"] = "no_data"
        _STATUS_JSON.write_text(json.dumps(status, ensure_ascii=False, indent=2, default=str))
        print("⚠️ 沒有特徵資料，跳過")
        return

    now = datetime.now()
    recent_cut = now - timedelta(days=RECENT_DAYS)
    reference_cut = recent_cut - timedelta(days=REFERENCE_DAYS)

    recent = df[df["created_at"] >= recent_cut]
    reference = df[(df["created_at"] >= reference_cut) & (df["created_at"] < recent_cut)]

    print(f"recent ({RECENT_DAYS} 天): {len(recent)} 筆")
    print(f"reference ({RECENT_DAYS}-{RECENT_DAYS+REFERENCE_DAYS} 天前): {len(reference)} 筆")

    if len(recent) < 5 or len(reference) < 20:
        status.update({
            "result": "insufficient_samples",
            "n_recent": len(recent),
            "n_reference": len(reference),
        })
        _STATUS_JSON.write_text(json.dumps(status, ensure_ascii=False, indent=2, default=str))
        print(f"⚠️ 樣本不足（recent={len(recent)} reference={len(reference)}），跳過")
        return

    psi_results: dict[str, float] = {}
    for col in FEATURE_COLUMNS:
        if col not in recent.columns or col not in reference.columns:
            continue
        try:
            r = pd.to_numeric(reference[col], errors="coerce").to_numpy()
            c = pd.to_numeric(recent[col], errors="coerce").to_numpy()
            psi = compute_psi(r, c)
            psi_results[col] = round(psi, 4)
        except Exception:
            continue

    valid = [(k, v) for k, v in psi_results.items() if v is not None]
    valid.sort(key=lambda x: x[1], reverse=True)
    max_psi = max((v for _, v in valid), default=0.0)
    top_drift = valid[:5]

    status.update({
        "n_recent": len(recent),
        "n_reference": len(reference),
        "max_psi": round(max_psi, 4),
        "top_drift": top_drift,
        "trigger_threshold": PSI_TRIGGER,
    })

    print(f"\nmax PSI = {max_psi:.4f} (門檻 {PSI_TRIGGER})")
    for name, val in top_drift:
        print(f"  {name}: {val}")

    triggered = max_psi > PSI_TRIGGER
    status["triggered"] = triggered

    if triggered:
        print(f"\n⚠️ PSI > {PSI_TRIGGER} → 立刻觸發 retrain (force=True)")
        try:
            from app.core.imitation_trainer import train_imitation_model
            result = train_imitation_model(min_samples=50, force=True)
            status["retrain_result"] = result
            print(f"retrain 結果: {result.get('status')}")
            _notify(
                "⚠️ 模型漂移偵測",
                f"max PSI {max_psi:.2f} → 已觸發 retrain ({result.get('status')})",
            )
        except Exception as e:
            status["retrain_error"] = str(e)
            print(f"❌ retrain 失敗：{e}")
            _notify("❌ Drift retrain 失敗", str(e))
    else:
        print("✅ 穩定，不需 retrain")

    _STATUS_JSON.write_text(json.dumps(status, ensure_ascii=False, indent=2, default=str))
    elapsed = (datetime.now() - started).total_seconds()
    print(f"\n完成（{elapsed:.1f}s）")


if __name__ == "__main__":
    main()
