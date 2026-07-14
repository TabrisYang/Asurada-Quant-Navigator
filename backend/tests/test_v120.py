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


# ─── v120.3：store() capture signals ───────────

def test_v120_3_store_captures_signals_from_chart_state():
    """store() 帶 chart_state → DB 行寫入 signal_at_entry 欄位。"""
    import json as _json
    chart_state = {
        "external_signals": {
            "derivatives": {
                "funding_rate_pct": -0.03,
                "open_interest_24h_change_pct": 8.5,
                "coinbase_premium_pct": 0.02,
                "global_long_short_ratio": 1.0,
                "ob_imbalance_ratio": 1.7,
            },
            "sentiment": {"fear_greed_value": 35},
        },
    }
    pred = {
        "direction": "long",
        "entry_price": 100.0,
        "target_price": 110.0,
        "stop_price": 95.0,
        "timeframe_hours": 24,
        "confidence": "medium",
        "regime": "trending_up",
        "indicators": "test",
    }
    # 用獨特 symbol 避免汙染
    test_symbol = f"V120_TEST_{__import__('uuid').uuid4().hex[:8]}/USDT"
    pid = prediction_tracker.store(
        symbol=test_symbol, timeframe="4h",
        prediction=pred, source_question="v120.3 test",
        chart_state=chart_state,
    )
    try:
        assert pid > 0, "store 應回傳 valid pid"
        row = prediction_tracker._conn.execute(
            "SELECT funding_at_entry, premium_at_entry, fear_greed_at_entry, "
            "buckets_json, signals_json FROM predictions WHERE id=?",
            (pid,),
        ).fetchone()
        assert row["funding_at_entry"] == -0.03
        assert row["premium_at_entry"] == 0.02
        assert row["fear_greed_at_entry"] == 35
        # bucket 應該有
        buckets = _json.loads(row["buckets_json"])
        assert buckets["funding"] == "NEGATIVE"
        assert buckets["premium"] == "POSITIVE"
        assert buckets["fear_greed"] == "FEAR"
        # raw signals 也存了
        signals = _json.loads(row["signals_json"])
        assert signals["derivatives"]["funding_rate_pct"] == -0.03
    finally:
        # cleanup
        prediction_tracker._conn.execute(
            "DELETE FROM predictions WHERE symbol=?", (test_symbol,)
        )
        prediction_tracker._conn.commit()


# ─── v120.5：get_signal_combo_stats + chat.py 注入 ─────

def test_v120_5_get_single_signal_stats_returns_correct_structure():
    """get_single_signal_stats 回 {win_rate, samples, wins, losses}。"""
    result = prediction_tracker.get_single_signal_stats(
        symbol=None, signal_name="funding", bucket="POSITIVE", days=180,
    )
    assert "win_rate" in result
    assert "samples" in result
    assert "wins" in result
    assert isinstance(result["win_rate"], (int, float))
    assert isinstance(result["samples"], int)


def test_v120_5_get_signal_combo_stats_filters_unknown():
    """current_buckets 中的 UNKNOWN 應該被過濾，不參與 SQL match。"""
    result = prediction_tracker.get_signal_combo_stats(
        symbol=None,
        current_buckets={"funding": "UNKNOWN", "fear_greed": "FEAR"},
        days=180,
    )
    # 只剩 fear_greed 一個訊號被 match
    assert result["matched_signals"] == ["fear_greed"]


def test_v120_5_get_signal_combo_stats_empty_buckets():
    """全部 UNKNOWN → 回 samples=0 不 raise。"""
    result = prediction_tracker.get_signal_combo_stats(
        symbol=None,
        current_buckets={"funding": "UNKNOWN", "fear_greed": "UNKNOWN"},
        days=180,
    )
    assert result["samples"] == 0
    assert result["win_rate"] == 0


def test_v120_5_chat_py_injects_signal_history():
    """chat.py 必須在 recent_accuracy 注入後加 signal_history。"""
    chat_py = (
        pathlib.Path(__file__).resolve().parent.parent / "app" / "api" / "routes" / "chat.py"
    )
    src = chat_py.read_text(encoding="utf-8")
    assert "signal_history" in src, "chat.py 必須注入 signal_history（v120.5）"
    assert "get_signal_combo_stats" in src, "chat.py 必須呼叫 get_signal_combo_stats"
    assert "classify_all_signals" in src, "chat.py 必須呼叫 classify_all_signals"


# ─── v120.6：function_defs.py v120 警告規則 ─────

def test_v120_6_function_defs_has_combo_stats_rule():
    """function_defs 必須含 combo_stats < 50% 強制警告規則。"""
    _llm_dir = pathlib.Path(__file__).resolve().parent.parent / "app" / "core" / "llm"
    # prompt 規則內容已抽至 prompt_modules.py，兩檔串接檢查
    src = (_llm_dir / "function_defs.py").read_text(encoding="utf-8") + (
        _llm_dir / "prompt_modules.py"
    ).read_text(encoding="utf-8")
    assert "combo_stats" in src, "function_defs 必須含 combo_stats 規則"
    assert "signal_history" in src, "function_defs 必須提及 signal_history 結構"
    # 必須要有「< 50%」之類的閾值
    assert "< 50%" in src or "命中率僅" in src, (
        "v120 規則必須含「命中率 < 50% 強制警告」"
    )


def test_v120_6_function_defs_has_single_signal_rule():
    """function_defs 必須含 single_signal_stats 多訊號不利警告規則。"""
    _llm_dir = pathlib.Path(__file__).resolve().parent.parent / "app" / "core" / "llm"
    # prompt 規則內容已抽至 prompt_modules.py，兩檔串接檢查
    src = (_llm_dir / "function_defs.py").read_text(encoding="utf-8") + (
        _llm_dir / "prompt_modules.py"
    ).read_text(encoding="utf-8")
    assert "single_signal_stats" in src, "function_defs 必須含 single_signal_stats 規則"


def test_v120_6_function_defs_has_sample_threshold_rule():
    """function_defs 必須含 samples < 5 樣本不足提示規則。"""
    _llm_dir = pathlib.Path(__file__).resolve().parent.parent / "app" / "core" / "llm"
    # prompt 規則內容已抽至 prompt_modules.py，兩檔串接檢查
    src = (_llm_dir / "function_defs.py").read_text(encoding="utf-8") + (
        _llm_dir / "prompt_modules.py"
    ).read_text(encoding="utf-8")
    assert "樣本不足" in src or "samples < 5" in src, (
        "v120 規則必須含「樣本不足提示」"
    )


def test_v120_3_store_without_chart_state_works():
    """不傳 chart_state 時 store 仍能正常運作（向後相容）。"""
    pred = {
        "direction": "long", "entry_price": 100.0, "target_price": 110.0,
        "stop_price": 95.0, "timeframe_hours": 24, "confidence": "low",
        "regime": "trending_up", "indicators": "test",
    }
    test_symbol = f"V120_NOCS_{__import__('uuid').uuid4().hex[:8]}/USDT"
    pid = prediction_tracker.store(
        symbol=test_symbol, timeframe="4h",
        prediction=pred, source_question="no chart_state test",
    )
    try:
        assert pid > 0
        row = prediction_tracker._conn.execute(
            "SELECT funding_at_entry, signals_json FROM predictions WHERE id=?",
            (pid,),
        ).fetchone()
        # 沒 chart_state → 新欄位都是 None
        assert row["funding_at_entry"] is None
        assert row["signals_json"] is None
    finally:
        prediction_tracker._conn.execute(
            "DELETE FROM predictions WHERE symbol=?", (test_symbol,)
        )
        prediction_tracker._conn.commit()
