"""阿斯拉量化系統 — 自動健康檢查腳本（每月由 launchd 觸發）。

檢查項目：
1. LLM 預測命中率：30 天 vs 90 天 baseline
2. 各策略命中率（從 predictions.db 統計）
3. Regime classifier 是否有 unknown 樣本累積（需要關注）
4. 退化判定：任何指標下降 > 15% → degraded

輸出：
- backend/data/db/system_health.json   ← 最新狀態（前端讀取顯示警告）
- backend/data/db/health_history.log   ← 歷史趨勢（每次跑都 append）

執行：
- 手動：python3 backend/scripts/audit_system_health.py
- 自動：launchd 每天 0:30（透過 install_audit_launchd.sh 安裝）
"""

import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_DB_DIR = _SCRIPT_DIR.parent / "data" / "db"
_PRED_DB = _DB_DIR / "predictions.db"
_UNKNOWN_REGIME_DB = _DB_DIR / "unknown_regime_log.db"
_HEALTH_FILE = _DB_DIR / "system_health.json"
_HEALTH_LOG = _DB_DIR / "health_history.log"

# 退化判定閾值
_DEGRADATION_THRESHOLD_PCT = 15.0


def _section(title: str):
    print()
    print("─" * 60)
    print(f"  {title}")
    print("─" * 60)


def compute_pred_accuracy(days: int) -> dict:
    """過去 N 天的預測命中率（從 predictions.db）。"""
    if not _PRED_DB.exists():
        return {"n": 0, "win_rate": None, "note": "predictions.db 不存在"}

    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    conn = sqlite3.connect(_PRED_DB)
    rows = conn.execute(
        "SELECT status FROM predictions "
        "WHERE created_at >= ? AND status IN ('hit_target', 'hit_stop')",
        (cutoff,),
    ).fetchall()
    conn.close()

    if not rows:
        return {"n": 0, "win_rate": None, "note": f"過去 {days} 天無已驗證預測"}

    hits = sum(1 for r in rows if r[0] == "hit_target")
    return {
        "n": len(rows),
        "win_rate": round(hits / len(rows) * 100, 1),
        "hits": hits,
        "stops": len(rows) - hits,
    }


def count_unknown_regimes(days: int) -> dict:
    """過去 N 天系統判定為 unknown 的次數（從 unknown_regime_log.db）。"""
    if not _UNKNOWN_REGIME_DB.exists():
        return {"n": 0, "note": "unknown_regime_log.db 尚未建立（要等 Level 1 自動記錄啟用後）"}

    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    try:
        conn = sqlite3.connect(_UNKNOWN_REGIME_DB)
        n = conn.execute(
            "SELECT COUNT(*) FROM unknown_regimes WHERE timestamp >= ?",
            (cutoff,),
        ).fetchone()[0]
        conn.close()
        return {"n": n}
    except sqlite3.OperationalError:
        return {"n": 0, "note": "unknown_regimes 表尚未建立"}


def detect_degradation(recent: dict, baseline: dict) -> dict:
    """判定是否退化。

    退化條件：
      - recent.win_rate 比 baseline.win_rate 下降超過 _DEGRADATION_THRESHOLD_PCT 個百分點
      - 或 recent.n < 3（樣本太少不可靠）
    """
    if recent.get("win_rate") is None or baseline.get("win_rate") is None:
        return {"is_degraded": False, "reason": "資料不足無法判定"}

    if recent["n"] < 3:
        return {"is_degraded": False, "reason": f"近期樣本太少（{recent['n']} 筆）", "warning": "建議增加交易/預測累積樣本"}

    drop = baseline["win_rate"] - recent["win_rate"]
    is_degraded = drop > _DEGRADATION_THRESHOLD_PCT
    return {
        "is_degraded": is_degraded,
        "drop_pct_points": round(drop, 1),
        "threshold": _DEGRADATION_THRESHOLD_PCT,
        "reason": f"recent {recent['win_rate']}% vs baseline {baseline['win_rate']}%（下降 {drop:.1f}pp）" if is_degraded else "未顯著退化",
    }


def generate_action_items(health: dict) -> list[str]:
    """根據健康狀態產生具體行動建議。"""
    actions: list[str] = []
    deg = health.get("degradation", {})

    if deg.get("is_degraded"):
        actions.append(
            f"⚠️ 預測命中率退化 {deg['drop_pct_points']} 個百分點，建議：① 暫時縮小倉位 ② 檢查最近策略是否被市場 regime 變化影響"
        )

    pred_30 = health.get("pred_accuracy_30d", {})
    if pred_30.get("n", 0) > 0 and pred_30.get("win_rate", 0) < 50:
        actions.append(
            f"近 30 天預測命中率 {pred_30['win_rate']}% < 50%，建議檢查是否有特定標的長期失準"
        )

    unknown = health.get("unknown_regime_30d", {})
    if unknown.get("n", 0) > 30:
        actions.append(
            f"近 30 天有 {unknown['n']} 次 regime 信心過低，市場可能進入新型態，建議手動 review"
        )

    if not actions:
        actions.append("✅ 系統運作正常，無需特別操作")

    return actions


def main():
    print()
    print("═" * 60)
    print("  阿斯拉量化系統 — 自動健康檢查")
    print(f"  時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 60)

    if not _DB_DIR.exists():
        print(f"\n✗ 找不到 DB 目錄：{_DB_DIR}")
        sys.exit(1)

    _section("1. LLM 預測命中率")
    pred_30d = compute_pred_accuracy(30)
    pred_90d = compute_pred_accuracy(90)
    print(f"  近 30 天：{pred_30d}")
    print(f"  近 90 天（baseline）：{pred_90d}")

    _section("2. Regime classifier 健康度")
    unknown_30d = count_unknown_regimes(30)
    print(f"  近 30 天 unknown 次數：{unknown_30d}")

    _section("3. 退化判定")
    deg = detect_degradation(pred_30d, pred_90d)
    print(f"  退化狀態：{'⚠️ 是' if deg.get('is_degraded') else '✓ 否'}")
    print(f"  原因：{deg.get('reason', 'N/A')}")

    health = {
        "checked_at": datetime.now().isoformat(),
        "status": "degraded" if deg.get("is_degraded") else "healthy",
        "pred_accuracy_30d": pred_30d,
        "pred_accuracy_90d": pred_90d,
        "unknown_regime_30d": unknown_30d,
        "degradation": deg,
    }
    health["action_items"] = generate_action_items(health)

    _section("4. 行動建議")
    for i, action in enumerate(health["action_items"], 1):
        print(f"  {i}. {action}")

    # 寫檔
    with _HEALTH_FILE.open("w", encoding="utf-8") as f:
        json.dump(health, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 寫入 {_HEALTH_FILE.name}")

    # Append 歷史 log
    with _HEALTH_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": health["checked_at"],
            "status": health["status"],
            "win_rate_30d": pred_30d.get("win_rate"),
            "win_rate_90d": pred_90d.get("win_rate"),
            "unknown_30d": unknown_30d.get("n", 0),
        }, ensure_ascii=False) + "\n")
    print(f"✓ Append 到 {_HEALTH_LOG.name}")

    print()


if __name__ == "__main__":
    main()
