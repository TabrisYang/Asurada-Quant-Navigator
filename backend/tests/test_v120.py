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


# ─── v120.2：signal bucket classifier ─────────────

from app.core.signal_buckets import (  # noqa: E402
    classify_funding,
    classify_oi_change,
    classify_premium,
    classify_long_short,
    classify_ob_imbalance,
    classify_fear_greed,
    classify_etf_flow,
    classify_all_signals,
)


def test_v120_2_classify_funding_buckets():
    assert classify_funding(0.10) == "POSITIVE_HIGH"
    assert classify_funding(0.03) == "POSITIVE"
    assert classify_funding(0.0) == "NEUTRAL"
    assert classify_funding(-0.005) == "NEUTRAL"
    assert classify_funding(-0.03) == "NEGATIVE"
    assert classify_funding(-0.10) == "NEGATIVE_HIGH"
    assert classify_funding(None) == "UNKNOWN"


def test_v120_2_classify_oi_change_buckets():
    assert classify_oi_change(20) == "RISING_FAST"
    assert classify_oi_change(8) == "RISING"
    assert classify_oi_change(0) == "FLAT"
    assert classify_oi_change(-8) == "FALLING"
    assert classify_oi_change(-20) == "FALLING_FAST"


def test_v120_2_classify_premium_buckets():
    assert classify_premium(0.10) == "POSITIVE_HIGH"
    assert classify_premium(0.02) == "POSITIVE"
    assert classify_premium(0) == "NEUTRAL"
    assert classify_premium(-0.03) == "NEGATIVE"
    assert classify_premium(-0.10) == "NEGATIVE_HIGH"


def test_v120_2_classify_long_short_buckets():
    assert classify_long_short(4.0) == "BULLISH_HEAVY"
    assert classify_long_short(2.0) == "BULLISH"
    assert classify_long_short(1.0) == "BALANCED"
    assert classify_long_short(0.6) == "BEARISH"
    assert classify_long_short(0.3) == "BEARISH_HEAVY"


def test_v120_2_classify_ob_imbalance_buckets():
    assert classify_ob_imbalance(2.5) == "BUY_HEAVY"
    assert classify_ob_imbalance(1.7) == "BUY_PRESSURE"
    assert classify_ob_imbalance(1.0) == "BALANCED"
    assert classify_ob_imbalance(0.6) == "SELL_PRESSURE"
    assert classify_ob_imbalance(0.3) == "SELL_HEAVY"


def test_v120_2_classify_fear_greed_buckets():
    assert classify_fear_greed(85) == "EXTREME_GREED"
    assert classify_fear_greed(60) == "GREED"
    assert classify_fear_greed(50) == "NEUTRAL"
    assert classify_fear_greed(35) == "FEAR"
    assert classify_fear_greed(15) == "EXTREME_FEAR"


def test_v120_2_classify_all_signals_one_shot():
    """從 external_signals 結構一鍵分類全部。"""
    derivatives = {
        "funding_rate_pct": -0.03,
        "open_interest_24h_change_pct": 8,
        "coinbase_premium_pct": 0.02,
        "global_long_short_ratio": 1.0,
        "ob_imbalance_ratio": 1.7,
    }
    sentiment = {"fear_greed_value": 35}
    result = classify_all_signals(derivatives, sentiment)
    assert result["funding"] == "NEGATIVE"
    assert result["oi_change"] == "RISING"
    assert result["premium"] == "POSITIVE"
    assert result["long_short"] == "BALANCED"
    assert result["ob_imbalance"] == "BUY_PRESSURE"
    assert result["fear_greed"] == "FEAR"
    assert result["etf_flow"] == "UNKNOWN"  # ETF Flow 沒抓


def test_v120_2_classify_unknown_when_none():
    """所有 classifier 對 None 回 UNKNOWN（不 raise）。"""
    for fn in (classify_funding, classify_oi_change, classify_premium,
               classify_long_short, classify_ob_imbalance,
               classify_fear_greed, classify_etf_flow):
        assert fn(None) == "UNKNOWN"
