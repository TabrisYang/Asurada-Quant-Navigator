"""邊界契約測試 — R2 精簡 chart_state（v154）

`from app.core.llm.adapter import ...` 這個 import 路徑本身就是契約：
adapter.py 拆成 package 後必須以 façade 保持它可用（本測試是拆檔守門員）。
"""

import json

from app.core.llm.adapter import _minimal_r2_chart_state


def _full_state() -> dict:
    return {
        "symbol": "BTC/USDT",
        "timeframe": "1d",
        "currentPrice": 64000,
        "currentRegime": {"regime": "ranging", "confidence": 55},
        "indicatorValues": {"RSI": 42.1, "BB_Width": 14.2, "ADX": 18.5},
        "external_signals": {
            "derivatives": {"funding_rate_pct": 0.01, "big_field": list(range(100))},
            "sentiment": {"fear_greed_value": 27},
        },
        "signal_history": {"huge": list(range(1000))},
        "recent_accuracy": {
            "win_rate_30d": 62.0,
            "signal_history": {"combo_stats": {"win_rate": 58.0, "samples": 40}},
        },
        "donchian_position_pct": 48.3,
    }


class TestMinimalR2ChartState:
    def test_keeps_current_values_strips_history(self):
        out = _minimal_r2_chart_state(_full_state())
        # 保留：識別欄位 + 當前指標值（v153）
        assert out["symbol"] == "BTC/USDT"
        assert out["currentRegime"]["regime"] == "ranging"
        assert out["indicatorValues"]["BB_Width"] == 14.2
        assert out["donchian_position_pct"] == 48.3
        # external_signals 壓成 summary
        assert out["external_signals_summary"]["funding_rate_pct"] == 0.01
        assert out["external_signals_summary"]["fear_greed_value"] == 27
        assert "external_signals" not in out
        # 剝掉大宗
        assert "signal_history" not in out
        # recent_accuracy 只剩摘要
        assert out["recent_accuracy"]["win_rate_30d"] == 62.0
        assert "signal_history" not in out["recent_accuracy"]
        # R2 註記存在
        assert "_r2_note" in out

    def test_indicator_values_size_fuse(self):
        state = _full_state()
        # 塞超過 8KB 的 indicatorValues → 保險絲丟棄（防 v118-120 TTFT 回歸）
        state["indicatorValues"] = {f"IND_{i}": "x" * 50 for i in range(400)}
        assert len(json.dumps(state["indicatorValues"])) > 8000
        out = _minimal_r2_chart_state(state)
        assert "indicatorValues" not in out

    def test_none_and_empty_passthrough(self):
        assert _minimal_r2_chart_state(None) is None
        assert _minimal_r2_chart_state({}) == {}
