"""阿斯拉量化系統 — 歷史 OHLCV 補深 / 刷新腳本。

用途（對應 plan 軌道 A）：
  1. 把加密貨幣現有 CSV 從 2023 往前補抓到 default_history_start（預設 2018-01-01）。
  2. 把 2330/TWII 等只到 2023 的台股補抓到 2020-01-01。
  3. 順帶把所有檔案往後刷新到今日（fetch_ohlcv 的 need_fetch_after 增量路徑）。

原理：engine.fetch_ohlcv 帶入比現有資料更早的 start_date 時，會走 need_fetch_before
分頁往前補抓；不需改動 engine 邏輯。單次執行即可補深 + 刷新。

執行：
    # 全部（預設清單 + 預設 timeframe）
    python3 backend/scripts/backfill_history.py

    # 只補特定標的
    python3 backend/scripts/backfill_history.py --symbols BTC/USDT 2330/TWD --timeframes 1d

    # 只刷新到今日、不往前補深
    python3 backend/scripts/backfill_history.py --refresh-only
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))

from app.core.config.settings import settings  # noqa: E402
from app.data.fetchers.crypto_engine import crypto_engine  # noqa: E402
from app.data.fetchers.tw_stock_engine import tw_stock_engine  # noqa: E402
from app.utils.symbol import is_tw_stock  # noqa: E402

# 台股 engine 僅支援日/週線
_TW_TIMEFRAMES = {"1d", "1w"}


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


async def _backfill_one(symbol: str, timeframe: str, crypto_start: datetime,
                        tw_start: datetime, refresh_only: bool) -> tuple[str, str, int, str]:
    """補深/刷新單一 symbol+timeframe，回傳 (symbol, tf, bars, status)。"""
    tw = is_tw_stock(symbol)
    if tw and timeframe not in _TW_TIMEFRAMES:
        return (symbol, timeframe, 0, "skip(台股僅 1d/1w)")

    # refresh_only → 不帶 start_date，只走增量往後刷新；否則帶早於現有資料的起點觸發往前補抓
    start_date = None
    if not refresh_only:
        start_date = tw_start if tw else crypto_start

    engine = tw_stock_engine if tw else crypto_engine
    try:
        df = await engine.fetch_ohlcv(symbol=symbol, timeframe=timeframe, start_date=start_date)
        first = str(df["timestamp"].min())[:10] if not df.empty else "-"
        last = str(df["timestamp"].max())[:10] if not df.empty else "-"
        return (symbol, timeframe, len(df), f"ok {first}~{last}")
    except Exception as e:  # noqa: BLE001
        return (symbol, timeframe, 0, f"error: {e}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="歷史 OHLCV 補深 / 刷新")
    parser.add_argument("--symbols", nargs="*", default=None,
                        help="標的清單，預設 settings.default_symbols + default_tw_symbols")
    parser.add_argument("--timeframes", nargs="*", default=None,
                        help="時間框架，預設 settings.default_timeframes")
    parser.add_argument("--crypto-start", default=settings.default_history_start,
                        help=f"加密回溯起點（預設 {settings.default_history_start}）")
    parser.add_argument("--tw-start", default="2020-01-01", help="台股回溯起點（預設 2020-01-01）")
    parser.add_argument("--refresh-only", action="store_true",
                        help="只往後刷新到今日，不往前補深")
    args = parser.parse_args()

    symbols = args.symbols or (settings.default_symbols + settings.default_tw_symbols)
    timeframes = args.timeframes or settings.default_timeframes
    crypto_start = _parse_date(args.crypto_start)
    tw_start = _parse_date(args.tw_start)

    mode = "刷新到今日" if args.refresh_only else f"補深(加密→{args.crypto_start} / 台股→{args.tw_start}) + 刷新"
    print(f"▶ 開始 {mode}")
    print(f"  標的 {len(symbols)} 個 × 時間框架 {timeframes}")

    ok = 0
    fail = 0
    for symbol in symbols:
        for tf in timeframes:
            sym, t, bars, status = await _backfill_one(symbol, tf, crypto_start, tw_start, args.refresh_only)
            flag = "✓" if status.startswith("ok") else ("·" if status.startswith("skip") else "✗")
            print(f"  {flag} {sym:14s} {t:4s} {bars:6d} 根  {status}")
            if status.startswith("ok"):
                ok += 1
            elif status.startswith("error"):
                fail += 1

    print(f"▶ 完成：成功 {ok}、失敗 {fail}")


if __name__ == "__main__":
    asyncio.run(main())
