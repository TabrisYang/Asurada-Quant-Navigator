"""v120 regression tests：訊號層回測（修「看漲說漲」徹底版）。

設計：對每個訊號（funding / OI / premium / long_short / order_book / fear_greed）
做歷史命中率追蹤，配合 v118 regime_warning 一起警告 LLM「該訊號組合過去無 alpha」。

6 個 sub-tasks：
- v120.1: predictions schema 加 signal_at_entry 欄位 + JSON 彈性容器
- v120.2: signal bucket classifier
- v120.3: prediction_tracker.store capture signals
- v120.4: historical 回填腳本
- v120.5: get_signal_combo_stats + chat.py 注入
- v120.6: function_defs.py v120 警告規則
"""

import pathlib

from app.core.prediction_tracker import prediction_tracker


# ─── v120.1：predictions schema migration ─────

def test_v120_1_predictions_table_has_signal_columns():
    """確認 v120 新增的 9 個欄位都存在於 predictions 表。"""
    prediction_tracker._ensure_db()
    cur = prediction_tracker._conn.execute("PRAGMA table_info(predictions)")
    cols = {row[1] for row in cur.fetchall()}

    required = {
        "funding_at_entry",
        "oi_24h_change_at_entry",
        "premium_at_entry",
        "long_short_at_entry",
        "etf_flow_7d_at_entry",
        "ob_imbalance_at_entry",
        "fear_greed_at_entry",
        "signals_json",
        "buckets_json",
    }
    missing = required - cols
    assert not missing, f"v120 schema 缺少欄位：{missing}"


def test_v120_1_signal_columns_default_null():
    """新欄位都該是 nullable（既有 prediction 不會被影響）。"""
    prediction_tracker._ensure_db()
    cur = prediction_tracker._conn.execute("PRAGMA table_info(predictions)")
    rows = cur.fetchall()
    for row in rows:
        col_name = row[1]
        if col_name.endswith("_at_entry") or col_name.endswith("_json"):
            # 應該是 nullable（notnull=0 或有 default）
            notnull = row[3]
            dflt = row[4]
            assert notnull == 0 or dflt is not None, (
                f"{col_name} 不是 nullable 也沒 default，會影響既有資料"
            )
