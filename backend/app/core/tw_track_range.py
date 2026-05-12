"""阿斯拉量化系統 — 台股 BB 掃描跨日追蹤

對「最近一次掃描的標的池」或「全市場」執行：
1. 批次撈 OHLCV（自動斷點續傳 + 本地快取）
2. 逐日重算 4 個指標：close / bb_pctile / change_20d / vol_5d
3. 標出該日是否符合條件（BB% < pctile_threshold，預設 20）
4. 計算 first_match_date、match_count

回傳：以 async iterator 流式產出 progress + result 事件，外層 SSE 直接轉發。
"""

from __future__ import annotations

import asyncio
import io
import csv
import json
import sqlite3
import time
from datetime import datetime, timedelta
from typing import AsyncIterator, Optional

import numpy as np
import pandas as pd
from loguru import logger

from app.core.config.settings import settings
from app.data.fetchers.tw_stock_engine import TwStockEngine
from app.data.fetchers.tw_universe import tw_universe


_DEFAULT_CONCURRENCY = 25
_DEFAULT_PCTILE_THRESHOLD = 20.0  # BB% < 20 視為符合條件
_BB_PERIOD = 20
_BB_LOOKBACK = 120                 # 計算 BB Width 百分位的滾動視窗
_BACKFILL_BUFFER_DAYS = 200       # 起算日往前多撈幾天，確保 BB lookback 充足


# ──────────────────────────────────────────────
# 標的池取得
# ──────────────────────────────────────────────

def get_recent_scan_symbols() -> tuple[Optional[str], list[dict]]:
    """從 tw_scan_history.db 撈最近一次掃描結果的標的清單。

    Returns:
        (scan_id, [{"code", "name", "market", "industry"}, ...])
        若無歷史則回 (None, [])
    """
    db_path = settings.db_path / "tw_scan_history.db"
    if not db_path.exists():
        return None, []

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT scan_id, results_json FROM tw_bb_scans "
            "ORDER BY scanned_at DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()

    if not row:
        return None, []

    scan_id, results_json = row
    try:
        results = json.loads(results_json)
    except Exception as e:
        logger.warning(f"tw_scan_history 最新 scan results 解析失敗: {e}")
        return scan_id, []

    symbols = [
        {
            "code": r.get("code", ""),
            "name": r.get("name", ""),
            "market": r.get("market", ""),
            "industry": r.get("industry", ""),
        }
        for r in results
        if r.get("code")
    ]
    return scan_id, symbols


async def get_full_market_symbols() -> list[dict]:
    """取得全市場標的清單（上市 + 上櫃，~1938 檔）。"""
    stocks = await tw_universe.fetch_all(use_cache=True)
    return [
        {
            "code": s.code,
            "name": s.name,
            "market": s.market,
            "industry": s.industry,
        }
        for s in stocks
    ]


# ──────────────────────────────────────────────
# 指標計算（單日切片）
# ──────────────────────────────────────────────

def _compute_features_on_df(df: pd.DataFrame, target_date: pd.Timestamp) -> Optional[dict]:
    """在 df 上找到 target_date 對應的那一根 K 線，重算 4 個指標。

    回傳：{"close", "bb_pctile", "change_20d", "vol_5d"} 或 None（資料不足）。
    """
    if df is None or df.empty:
        return None

    # 找該日（或之前最接近的交易日）的 index
    df_sub = df[df["timestamp"] <= target_date]
    if df_sub.empty:
        return None

    last_idx = df_sub.index[-1]
    last_ts = df_sub.iloc[-1]["timestamp"]

    # 必須是目標日當天（不是更早的交易日）— 否則該日不存在這檔的資料
    if pd.Timestamp(last_ts).date() != target_date.date():
        return None

    # 收盤價
    close = float(df.iloc[last_idx]["close"])

    # BB Width 百分位（_BB_PERIOD=20, _BB_LOOKBACK=120）
    bb_pctile: Optional[float] = None
    if last_idx >= _BB_PERIOD - 1:
        # 取截至 target_date 的所有 close（含 target_date）
        close_series = df.iloc[: last_idx + 1]["close"]
        sma = close_series.rolling(window=_BB_PERIOD).mean()
        std = close_series.rolling(window=_BB_PERIOD).std()
        safe_sma = sma.replace(0, np.nan)
        bb_width = (2 * std / safe_sma) * 100

        # 滾動 lookback 百分位（最後一根）
        if len(bb_width.dropna()) >= _BB_LOOKBACK:
            recent = bb_width.tail(_BB_LOOKBACK)
            current = bb_width.iloc[-1]
            if pd.notna(current):
                rank = (recent < current).sum() + (recent == current).sum() / 2
                bb_pctile = float(rank / len(recent) * 100)

    # 20 日漲跌幅
    change_20d: Optional[float] = None
    if last_idx >= 20:
        prev_close = float(df.iloc[last_idx - 20]["close"])
        if prev_close > 0:
            change_20d = (close - prev_close) / prev_close * 100

    # 5 日均量（單位：張，÷1000）
    vol_5d: Optional[int] = None
    if last_idx >= 4 and "volume" in df.columns:
        recent_vol = df.iloc[last_idx - 4 : last_idx + 1]["volume"].mean()
        if pd.notna(recent_vol):
            vol_5d = int(recent_vol / 1000)

    return {
        "close": round(close, 2),
        "bb_pctile": round(bb_pctile, 2) if bb_pctile is not None else None,
        "change_20d": round(change_20d, 2) if change_20d is not None else None,
        "vol_5d": vol_5d,
    }


# ──────────────────────────────────────────────
# 主聚合
# ──────────────────────────────────────────────

class TwTrackRange:
    """跨日追蹤聚合器。"""

    def __init__(self):
        self._engine = TwStockEngine()

    async def stream(
        self,
        start_date: str,
        end_date: str,
        scope: str = "recent_scan",
        concurrency: int = _DEFAULT_CONCURRENCY,
        pctile_threshold: float = _DEFAULT_PCTILE_THRESHOLD,
    ) -> AsyncIterator[dict]:
        """執行跨日追蹤，以 async iterator 產出事件：

          {"type": "progress", "phase": str, "current": int, "total": int}
          {"type": "result", "scan_dates": [...], "symbols": [...]}
          {"type": "error", "error": str}
        """
        try:
            sd = pd.Timestamp(start_date).normalize()
            ed = pd.Timestamp(end_date).normalize()
        except Exception as e:
            yield {"type": "error", "error": f"日期格式錯誤: {e}"}
            return

        if sd > ed:
            yield {"type": "error", "error": "start_date 不能晚於 end_date"}
            return

        # 1. 取標的池
        if scope == "full_market":
            symbols_meta = await get_full_market_symbols()
            source_scan_id = None
        else:
            source_scan_id, symbols_meta = get_recent_scan_symbols()

        if not symbols_meta:
            yield {
                "type": "error",
                "error": (
                    "標的池為空：請先執行一次掃描" if scope == "recent_scan"
                    else "全市場標的池為空，請檢查 tw_universe 快取"
                ),
            }
            return

        total = len(symbols_meta)
        logger.info(
            f"TwTrackRange: 開始追蹤 scope={scope} total={total} "
            f"range=[{sd.date()}, {ed.date()}] threshold={pctile_threshold}"
        )

        yield {
            "type": "progress",
            "phase": "starting",
            "current": 0,
            "total": total,
            "scope": scope,
            "source_scan_id": source_scan_id,
        }

        # 2. 撈 OHLCV + 算指標（並發）
        # 為了讓 BB lookback (120) 在 start_date 當天就有值，往前多撈 _BACKFILL_BUFFER_DAYS
        fetch_start = sd - pd.Timedelta(days=_BACKFILL_BUFFER_DAYS)
        fetch_end = ed + pd.Timedelta(days=1)

        # 預生成日期範圍（每日，跳過週末）
        date_range = pd.date_range(start=sd, end=ed, freq="D")
        scan_dates_str = [d.strftime("%Y-%m-%d") for d in date_range]

        sem = asyncio.Semaphore(concurrency)
        progress_state = {"current": 0}
        progress_lock = asyncio.Lock()
        result_queue: asyncio.Queue[dict] = asyncio.Queue()
        symbols_out: list[dict] = []

        async def _fetch_and_compute(meta: dict) -> Optional[dict]:
            code = meta["code"]
            symbol = f"{code}/TWD"
            try:
                df = await self._engine.fetch_ohlcv(
                    symbol=symbol,
                    timeframe="1d",
                    start_date=fetch_start.to_pydatetime(),
                    end_date=fetch_end.to_pydatetime(),
                )
                if df is None or df.empty:
                    return None
                df = df.copy()
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp").reset_index(drop=True)

                daily_features: dict[str, Optional[dict]] = {}
                matched_dates: list[str] = []
                for date_str, ts in zip(scan_dates_str, date_range):
                    feats = _compute_features_on_df(df, ts)
                    if feats is None:
                        daily_features[date_str] = None
                        continue
                    matched = (
                        feats["bb_pctile"] is not None
                        and feats["bb_pctile"] < pctile_threshold
                    )
                    daily_features[date_str] = {**feats, "matched": matched}
                    if matched:
                        matched_dates.append(date_str)

                if not matched_dates:
                    # 此標的範圍內全未符合 → 不納入結果（避免表格塞爆）
                    return None

                return {
                    "code": code,
                    "name": meta["name"],
                    "market": meta["market"],
                    "industry": meta["industry"],
                    "first_match_date": matched_dates[0],
                    "last_match_date": matched_dates[-1],
                    "match_count": len(matched_dates),
                    "daily_features": daily_features,
                }
            except Exception as e:
                logger.debug(f"TwTrackRange {code} 失敗: {e}")
                return None

        async def _worker(meta: dict):
            async with sem:
                row = await _fetch_and_compute(meta)

            async with progress_lock:
                progress_state["current"] += 1
                current = progress_state["current"]

            if row is not None:
                await result_queue.put(row)

            # 每 10 個（或最後一個）emit 一次 progress
            if current % 10 == 0 or current == total:
                await result_queue.put({
                    "__progress__": True,
                    "current": current,
                    "total": total,
                })

        start_ts = time.monotonic()
        tasks = [asyncio.create_task(_worker(m)) for m in symbols_meta]

        while True:
            try:
                evt = await asyncio.wait_for(result_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                evt = None

            if evt is not None:
                if evt.get("__progress__"):
                    yield {
                        "type": "progress",
                        "phase": "computing",
                        "current": evt["current"],
                        "total": evt["total"],
                    }
                else:
                    symbols_out.append(evt)

            if all(t.done() for t in tasks) and result_queue.empty():
                break

        await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.monotonic() - start_ts

        # 排序：first_match_date 升序，再 match_count 降序
        symbols_out.sort(key=lambda r: (r["first_match_date"], -r["match_count"]))

        logger.info(
            f"TwTrackRange: 完成 total={total} matched={len(symbols_out)} "
            f"elapsed={elapsed:.1f}s"
        )

        yield {
            "type": "result",
            "scan_dates": scan_dates_str,
            "symbols": symbols_out,
            "total_scanned": total,
            "total_matched": len(symbols_out),
            "duration_sec": round(elapsed, 1),
            "scope": scope,
            "source_scan_id": source_scan_id,
            "pctile_threshold": pctile_threshold,
        }


# ──────────────────────────────────────────────
# CSV 匯出
# ──────────────────────────────────────────────

def export_to_csv(result: dict) -> str:
    """將跨日追蹤結果組成 CSV 文字（4 指標 × N 天）。"""
    scan_dates: list[str] = result.get("scan_dates", [])
    symbols: list[dict] = result.get("symbols", [])

    buf = io.StringIO()
    # Excel 開 UTF-8 CSV 加 BOM 才不會亂碼
    buf.write("﻿")

    writer = csv.writer(buf)

    header = ["代號", "名稱", "市場", "產業", "首次符合", "符合天數"]
    for d in scan_dates:
        header.extend([f"{d}_BB%", f"{d}_收盤", f"{d}_20日漲跌%", f"{d}_5日均量"])
    writer.writerow(header)

    for s in symbols:
        row = [
            s.get("code", ""),
            s.get("name", ""),
            s.get("market", ""),
            s.get("industry", ""),
            s.get("first_match_date", ""),
            s.get("match_count", 0),
        ]
        feats_map = s.get("daily_features", {})
        for d in scan_dates:
            feats = feats_map.get(d) or {}
            row.extend([
                feats.get("bb_pctile", "") if feats.get("bb_pctile") is not None else "",
                feats.get("close", "") if feats.get("close") is not None else "",
                feats.get("change_20d", "") if feats.get("change_20d") is not None else "",
                feats.get("vol_5d", "") if feats.get("vol_5d") is not None else "",
            ])
        writer.writerow(row)

    return buf.getvalue()


# 模組單例
tw_track_range = TwTrackRange()
