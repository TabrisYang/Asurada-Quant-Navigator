"""邊界契約測試 — executor 工具回傳結構（v154）

目的：釘住 LLM 工具的回傳 schema。任何人改壞欄位名/結構，這裡先紅。
全部用合成資料（patch_executor_data），CI 無 OHLCV 也全綠。
"""

import asyncio

import pytest

from tests.conftest import HAS_LOCAL_OHLCV

import app.core.llm.executor as ex


def _run(coro):
    return asyncio.run(coro)


class TestAnalyzeEventPatterns:
    def test_success_contract(self, patch_executor_data):
        r = _run(ex._exec_analyze_event_patterns(
            {"event_type": "price_drop", "threshold": 1.0, "n_bars": 1},
            "TEST/USDT", "1d"))
        assert r["status"] == "success"
        assert isinstance(r["events_found"], int) and r["events_found"] > 0
        assert isinstance(r["event_dates"], list)
        assert isinstance(r["common_patterns"], dict)
        # v153：事件幅度聚合（使用者問「通常跌多少」的答案）
        mag = r["event_magnitude_stats"]
        assert mag is not None
        for key in ("mean", "median", "std", "min", "max", "p25", "p75", "samples"):
            assert key in mag, f"event_magnitude_stats 缺 {key}"
        assert isinstance(r["annotations"], list)
        if r["annotations"]:
            a = r["annotations"][0]
            assert "type" in a and "startTime" in a and "groupId" in a
        assert isinstance(r["data_range"], str)
        assert isinstance(r["total_bars"], int)

    def test_no_events_has_suggestion(self, patch_executor_data):
        r = _run(ex._exec_analyze_event_patterns(
            {"event_type": "price_drop", "threshold": 99.0, "n_bars": 1},
            "TEST/USDT", "1d"))
        assert r["status"] == "no_events"
        assert "suggestion" in r


class TestConditionalProbScan:
    def test_needs_confirmation_gate(self, patch_executor_data):
        r = _run(ex._exec_conditional_prob_scan({}, "TEST/USDT", "1d"))
        assert r["status"] == "needs_confirmation"

    def test_success_contract(self, patch_executor_data):
        r = _run(ex._exec_conditional_prob_scan(
            {"confirmed": True, "indicators": ["rsi"], "target_pct": 2.0},
            "TEST/USDT", "1d"))
        assert r["status"] == "success"
        assert isinstance(r["indicators"], dict) and r["indicators"]
        for _, ind in r["indicators"].items():
            assert "bins" in ind and isinstance(ind["bins"], list)
            assert "baseline_prob_pct" in ind
            assert "best_range" in ind
        ob = r["overall_best"]
        for key in ("indicator", "range", "prob_pct", "baseline_pct", "lift"):
            assert key in ob
        assert "warning" in r

    def test_bad_range_falls_back_to_full(self, monkeypatch, synthetic_ohlcv):
        # v153 防呆：指定載不到資料的日期範圍 → 自動退全量而非 0 根硬失敗
        import pandas as pd

        def _range_aware_load(symbol, timeframe, start=None, end=None):
            if start or end:
                return pd.DataFrame(columns=synthetic_ohlcv.columns)  # 範圍載不到
            return synthetic_ohlcv.copy()

        monkeypatch.setattr(ex, "_load_local_data", _range_aware_load)
        r = _run(ex._exec_conditional_prob_scan(
            {"confirmed": True, "start_date": "2031-01-01", "end_date": "2031-02-01"},
            "TEST/USDT", "1d"))
        assert r["status"] == "success"
        assert "range_notice" in r


class TestSqueezeBreakoutTiming:
    def test_success_contract(self, patch_executor_data):
        # 放寬條件確保合成資料也有進入事件
        r = _run(ex._exec_squeeze_breakout_timing(
            {"pctb_max": 60, "width_pctile_max": 60, "horizon_bars": 20},
            "TEST/USDT", "1d"))
        assert r["status"] == "success", r.get("message")
        assert isinstance(r["n_entries"], int) and r["n_entries"] > 0
        for side in ("up_first", "down_first"):
            s = r[side]
            assert "count" in s and "pct" in s and "bars_to_cross" in s
            if s["bars_to_cross"] is not None:
                for key in ("median", "mean", "p25", "p75", "min", "max"):
                    assert key in s["bars_to_cross"]
        assert "count" in r["no_cross_within_horizon"]
        assert isinstance(r["recent_entry_dates"], list)
        assert "resolved_pct" in r and "condition" in r

    def test_no_events_has_suggestion(self, patch_executor_data):
        r = _run(ex._exec_squeeze_breakout_timing(
            {"pctb_max": 0.0001, "width_pctile_max": 0.0001},
            "TEST/USDT", "1d"))
        assert r["status"] in ("no_events", "success")
        if r["status"] == "no_events":
            assert "suggestion" in r


class TestCompareStrategies:
    def test_missing_strategies_error(self, patch_executor_data):
        r = _run(ex._exec_compare_strategies({"confirmed": True}, "TEST/USDT", "1d"))
        assert r["status"] == "error"

    def test_too_many_strategies_error(self, patch_executor_data):
        strategies = [{"name": f"s{i}"} for i in range(6)]
        r = _run(ex._exec_compare_strategies(
            {"strategies": strategies, "confirmed": True}, "TEST/USDT", "1d"))
        assert r["status"] == "error"

    def test_all_invalid_returns_error_not_empty_success(self, patch_executor_data):
        # v153 防呆回歸鎖：全部策略無效時不可回「成功但空」
        r = _run(ex._exec_compare_strategies(
            {"strategies": [{"name": "壞策略", "entry_conditions": [], "exit_conditions": []}],
             "confirmed": True},
            "TEST/USDT", "1d"))
        assert r["status"] == "error"
        assert isinstance(r["per_strategy_errors"], list) and r["per_strategy_errors"]
        assert "suggestion" in r

    def test_valid_strategies_contract(self, patch_executor_data):
        strategies = [
            {"name": "RSI 低買", "direction": "long",
             "entry_conditions": [{"indicator": "rsi", "operator": "<", "value": 35}],
             "exit_conditions": [{"indicator": "rsi", "operator": ">", "value": 65}]},
            {"name": "RSI 高買", "direction": "long",
             "entry_conditions": [{"indicator": "rsi", "operator": ">", "value": 60}],
             "exit_conditions": [{"indicator": "rsi", "operator": "<", "value": 40}]},
        ]
        r = _run(ex._exec_compare_strategies(
            {"strategies": strategies, "confirmed": True}, "TEST/USDT", "1d"))
        assert r["status"] == "success", r.get("message")
        assert r["total_strategies"] == 2
        assert isinstance(r["comparison"], list) and len(r["comparison"]) == 2
        ok = [c for c in r["comparison"] if c.get("status") == "success"]
        assert ok, "至少一個策略應回測成功"
        assert "metrics" in ok[0] and "rank" in ok[0]


@pytest.mark.skipif(not HAS_LOCAL_OHLCV, reason="本地無真實 OHLCV（CI 環境）")
class TestRealDataSmoke:
    """真資料選配冒煙 — 只在本地跑，CI 自動 skip"""

    def test_event_patterns_btc(self):
        r = _run(ex._exec_analyze_event_patterns(
            {"event_type": "price_drop", "threshold": 5.0, "n_bars": 1},
            "BTC/USDT", "1d"))
        assert r["status"] in ("success", "no_events")
