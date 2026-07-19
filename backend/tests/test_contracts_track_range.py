"""邊界契約測試 — 跨日追蹤 result 事件結構（v154）

釘住 stream() 的 result 契約：這是前端 TwTrackRangeResult / TwDailyFeatures 型別的
後端對應面。欄位集合改變（尤其 daily_features）必須同步改前端型別與此測試。
"""

import asyncio

from tests.conftest import make_ohlcv

import app.core.tw_track_range as ttr


class _FakeEngine:
    async def fetch_ohlcv(self, symbol, timeframe, start_date=None, end_date=None):
        # 400 根涵蓋 backfill 200 天 + 追蹤區間，bb_pctile 需 >=140 根
        return make_ohlcv(start="2025-01-01", n=560)


def _collect(start_date: str, end_date: str, **kw):
    async def _go():
        events = []
        async for evt in ttr.tw_track_range.stream(
            start_date=start_date, end_date=end_date, scope="recent_scan", **kw
        ):
            events.append(evt)
        return events
    return asyncio.run(_go())


_DAILY_FEATURE_KEYS = {"close", "bb_pctile", "bb_width", "change_20d", "vol_5d", "matched", "breakout"}


class TestTrackRangeResultContract:
    def _patch(self, monkeypatch):
        monkeypatch.setattr(ttr.tw_track_range, "_engine", _FakeEngine())
        monkeypatch.setattr(
            ttr, "get_recent_scan_symbols",
            lambda within_days=30: ("scan_test", [
                {"code": "3260", "name": "威剛", "market": "otc", "industry": "半導體業"},
            ]),
        )

    def test_result_contract(self, monkeypatch):
        self._patch(monkeypatch)
        # pctile_threshold=101 → 全部 matched，結果確定性
        events = _collect("2026-05-01", "2026-05-15", pctile_threshold=101)
        assert events[0]["type"] == "progress"
        assert events[0]["phase"] == "starting"
        result = events[-1]
        assert result["type"] == "result"

        # 頂層契約
        for key in ("scan_dates", "symbols", "total_scanned", "total_matched",
                    "duration_sec", "scope", "pctile_threshold", "generated_at", "data_check"):
            assert key in result, f"result 缺 {key}"
        assert all(len(d) == 10 and d[4] == "-" for d in result["scan_dates"])  # YYYY-MM-DD

        # symbol 契約
        assert result["symbols"], "應至少一檔符合"
        s = result["symbols"][0]
        for key in ("code", "name", "market", "industry", "first_match_date",
                    "last_match_date", "match_count", "latest_close",
                    "latest_close_date", "daily_features"):
            assert key in s, f"symbol 缺 {key}"
        assert isinstance(s["latest_close"], float)

        # daily_features 契約：恰好這組 key（break_up/break_down 必須已被 pop —
        # 前端 TwDailyFeatures 對齊鎖）
        non_null = [f for f in s["daily_features"].values() if f is not None]
        assert non_null, "應有非 null 的 daily features"
        assert set(non_null[0].keys()) == _DAILY_FEATURE_KEYS

        # data_check 契約
        dc = result["data_check"]
        assert dc is not None
        for key in ("passed", "n_checks", "n_issues", "issues"):
            assert key in dc

    def test_invalid_date_order_errors(self, monkeypatch):
        self._patch(monkeypatch)
        events = _collect("2026-05-15", "2026-05-01")
        assert events[0]["type"] == "error"

    def test_empty_pool_errors(self, monkeypatch):
        monkeypatch.setattr(ttr.tw_track_range, "_engine", _FakeEngine())
        monkeypatch.setattr(ttr, "get_recent_scan_symbols", lambda within_days=30: (None, []))
        events = _collect("2026-05-01", "2026-05-15")
        assert any(e["type"] == "error" for e in events)

    def test_track_history_roundtrip(self, monkeypatch, tmp_path):
        # v155：追蹤落地 save→list→get，result 契約與 SSE result 一致
        from app.core.tw_scan_history import TwScanHistory
        self._patch(monkeypatch)
        events = _collect("2026-05-01", "2026-05-15", pctile_threshold=101)
        result = events[-1]

        store = TwScanHistory(db_path=tmp_path / "t.db")
        tid = store.save_track(params={"start_date": "2026-05-01"}, result=result)
        items = store.list_track_recent()
        assert items and items[0]["track_id"] == tid
        assert items[0]["total_matched"] == result["total_matched"]
        loaded = store.get_track(tid)
        assert loaded is not None
        r2 = loaded["result"]
        assert set(r2["symbols"][0]["daily_features"][result["scan_dates"][0]] or
                   next(f for f in r2["symbols"][0]["daily_features"].values() if f)) \
            .issuperset({"close", "bb_pctile", "matched"})
        assert r2["generated_at"] == result["generated_at"]
        assert store.get_track("nonexistent") is None

    def test_export_csv_smoke(self, monkeypatch):
        self._patch(monkeypatch)
        events = _collect("2026-05-01", "2026-05-15", pctile_threshold=101)
        result = events[-1]
        csv_text = ttr.export_to_csv(result, filter_note="測試註記")
        assert isinstance(csv_text, str)
        header = csv_text.splitlines()[0]
        for col in ("代號", "最新價", "帶寬%", "突破"):
            assert col in header
        assert "測試註記" in csv_text
