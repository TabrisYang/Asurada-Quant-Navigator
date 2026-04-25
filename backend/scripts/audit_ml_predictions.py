"""ML 預測有效性診斷工具。

回答：
1. predictions.db 裡有沒有 ml_enhanced=1 的紀錄？(ML pipeline 有沒有接通？)
2. ML 模組在 ml.db 有沒有訓練紀錄？(模型有沒有被訓練？)
3. ML 預測 vs 非 ML 預測命中率差異？(ML 有沒有 alpha？)

執行：
    python3 backend/scripts/audit_ml_predictions.py
"""

import sqlite3
import sys
from collections import Counter
from pathlib import Path

_DB = Path(__file__).resolve().parent.parent / "data" / "db"


def _section(title: str):
    print()
    print("─" * 60)
    print(f"  {title}")
    print("─" * 60)


def audit_predictions():
    """檢查 predictions.db 的 ML 標記與命中率。"""
    _section("1. predictions.db 概況")
    p_path = _DB / "predictions.db"
    if not p_path.exists():
        print(f"  ✗ 找不到 {p_path}")
        return

    conn = sqlite3.connect(p_path)
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    ml_total = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE ml_enhanced=1"
    ).fetchone()[0]
    print(f"  總筆數：{total}")
    print(f"  ML 標記筆數 (ml_enhanced=1)：{ml_total}")

    # 狀態分布
    status_dist = conn.execute(
        "SELECT status, COUNT(*) FROM predictions GROUP BY status ORDER BY 2 DESC"
    ).fetchall()
    print(f"\n  狀態分布：")
    for status, n in status_dist:
        print(f"    {status:20s}: {n}")

    _section("2. 命中率分析（已驗證的 hit_target / hit_stop）")
    for ml_flag, label in ((0, "非 ML 預測"), (1, "ML 預測")):
        rows = conn.execute(
            "SELECT status FROM predictions "
            "WHERE COALESCE(ml_enhanced, 0)=? AND status IN ('hit_target','hit_stop')",
            (ml_flag,),
        ).fetchall()
        if not rows:
            print(f"  {label}: 無已驗證紀錄")
            continue
        hits = sum(1 for r in rows if r[0] == "hit_target")
        rate = hits / len(rows) * 100
        print(f"  {label}: {hits}/{len(rows)} = {rate:.1f}% 命中（hit_target / 已驗證）")

    # 時間分布（看資料新不新）
    _section("3. 預測時間分布")
    earliest = conn.execute(
        "SELECT MIN(created_at), MAX(created_at) FROM predictions"
    ).fetchone()
    print(f"  最早：{earliest[0]}")
    print(f"  最晚：{earliest[1]}")

    # 來源分布
    sources = conn.execute(
        "SELECT source_question, COUNT(*) FROM predictions "
        "GROUP BY source_question ORDER BY 2 DESC LIMIT 5"
    ).fetchall()
    print(f"\n  來源 Top 5：")
    for src, n in sources:
        truncated = (src or "")[:60] + ("..." if src and len(src) > 60 else "")
        print(f"    {n:4d} × {truncated}")

    conn.close()
    return total, ml_total


def audit_ml_db():
    """檢查 ml.db 的訓練紀錄。"""
    _section("4. ml.db 訓練狀態")
    m_path = _DB / "ml.db"
    if not m_path.exists():
        print(f"  ✗ 找不到 {m_path}")
        return

    conn = sqlite3.connect(m_path)
    tables = [
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    ]
    print(f"  資料表：{tables}")
    for t in tables:
        try:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            cols = [c[1] for c in conn.execute(f"PRAGMA table_info({t})").fetchall()]
            print(f"    {t}: {n} 筆 | 欄位 {cols}")
        except Exception as e:
            print(f"    {t}: 讀取失敗 ({e})")

    conn.close()


def conclude(total: int, ml_total: int):
    _section("結論與建議")

    if total == 0:
        print("  ⚠ predictions.db 完全沒有資料")
        print("  → 系統的「預測追蹤」功能本身可能未啟用")
        return

    if ml_total == 0:
        print("  ⚠ ML 預測從未被存入 predictions.db（ml_enhanced=1 的紀錄為 0）")
        print()
        print("  可能原因：")
        print("    (a) 前端 chart_state 從未送出 mlPrediction 欄位")
        print("    (b) 前端 ML 模組沒 surface 結果給 chat 介面")
        print("    (c) 後端 [chat.py:2017] 的條件 `ml_pred and ml_pred.get('probability')` 從未成立")
        print()
        print("  建議：")
        print("    1. 檢查前端 chartStore 的 mlPrediction state 是否真的被填值")
        print("    2. 若 ML 功能根本沒開：考慮移除 ML 模組或 UI 標示「未啟用」")
        print("    3. 若有開但 surfacing 路徑斷掉：修補資料流（前端送 → chat_state → 後端 store）")
        return

    # 有 ML 紀錄的情況
    ratio = ml_total / total * 100
    print(f"  ML 預測佔總預測 {ratio:.1f}%（{ml_total}/{total}）")
    print()
    print("  下一步：跑「ML vs 非 ML 命中率對照」決定保留 / 降級 / 移除")
    print("    - 命中率 ≤ 52%：ML 是雜訊，建議降級或移除")
    print("    - 53-58%：保留為輔助參考")
    print("    - > 58%：加強整合")


def main():
    print()
    print("═" * 60)
    print("  阿斯拉量化系統 — ML 預測有效性診斷")
    print("═" * 60)

    if not _DB.exists():
        print(f"\n✗ 找不到 DB 目錄：{_DB}")
        sys.exit(1)

    result = audit_predictions()
    audit_ml_db()
    if result:
        conclude(*result)

    print()


if __name__ == "__main__":
    main()
