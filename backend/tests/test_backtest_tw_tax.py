"""台股證交稅回測契約（v155）

- /TWD 標的：賣出端自動加 0.3% 證交稅（long 出場、short 進場）
- 加密標的 / 無 symbol：bit-identical 不變（回歸鎖）
- tw_tax_enabled=False：台股與加密相同
"""

import pytest

from tests.conftest import make_ohlcv

from app.core.backtest import run_backtest
from app.core.config.settings import settings


_ENTRY = [{"indicator": "rsi", "operator": "<", "value": 40}]
_EXIT = [{"indicator": "rsi", "operator": ">", "value": 60}]


def _run(symbol: str, direction: str = "long"):
    df = make_ohlcv(n=500, seed=11)
    return run_backtest(
        df=df, entry_conditions=_ENTRY, exit_conditions=_EXIT,
        direction=direction, symbol=symbol,
    )


class TestTwTax:
    def test_crypto_identical_to_no_symbol(self):
        r_btc = _run("BTC/USDT")
        r_none = _run("")
        assert r_btc.metrics == r_none.metrics, "加密/無 symbol 路徑必須 bit-identical"

    def test_tw_symbol_pays_sell_tax(self):
        r_crypto = _run("BTC/USDT")
        r_tw = _run("2330/TWD")
        assert r_tw.trades and r_crypto.trades
        assert len(r_tw.trades) == len(r_crypto.trades), "稅不應改變交易次數（同進出訊號）"
        # 每筆做多交易：台股出場價比加密低 ~0.3%（賣出端扣稅）
        for t_tw, t_c in zip(r_tw.trades, r_crypto.trades):
            assert t_tw.pnl_pct < t_c.pnl_pct, "台股每筆損益應劣於未稅版本"
        # 總報酬劣化
        assert r_tw.metrics["total_return_pct"] < r_crypto.metrics["total_return_pct"]
        # warning 註明
        assert any("證交稅" in w for w in r_tw.warnings)
        assert not any("證交稅" in w for w in r_crypto.warnings)

    def test_tax_disabled_matches_crypto(self, monkeypatch):
        monkeypatch.setattr(settings, "tw_tax_enabled", False)
        r_tw = _run("2330/TWD")
        r_crypto = _run("BTC/USDT")
        assert r_tw.metrics == r_crypto.metrics
        assert not any("證交稅" in w for w in r_tw.warnings)

    def test_short_direction_tax_on_entry(self):
        r_tw = _run("2330/TWD", direction="short")
        r_crypto = _run("BTC/USDT", direction="short")
        if not r_tw.trades:
            pytest.skip("合成資料 short 未觸發交易")
        assert len(r_tw.trades) == len(r_crypto.trades)
        # 做空進場＝賣出扣稅 → 台股進場價低於未稅版本
        for t_tw, t_c in zip(r_tw.trades, r_crypto.trades):
            assert t_tw.entry_price < t_c.entry_price
            assert t_tw.pnl_pct < t_c.pnl_pct
