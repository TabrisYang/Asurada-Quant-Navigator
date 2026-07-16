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
_DEFAULT_RECENT_DAYS = 30         # v128: 跨日追蹤標的池 = 最近 N 天掃描聯集（去重）
                                   # 30 天 ≈ 21 個交易日，覆蓋 BB 壓縮研究完整週期：
                                   #   - 壓縮事件 5-15 個交易日後傾向突破
                                   #   - 突破後再加 1-2 週走勢觀察
                                   # 避免「最近 N 次」邏輯：若用戶月跑 1 次、5 次 = 5 個月過舊


# ──────────────────────────────────────────────
# 標的池取得
# ──────────────────────────────────────────────

def get_recent_scan_symbols(within_days: int = _DEFAULT_RECENT_DAYS) -> tuple[Optional[str], list[dict]]:
    """從 tw_scan_history.db 撈最近 N 天內掃描結果的**聯集**標的清單（去重）。

    v128: 從「LIMIT N 次」（v126）改成「WHERE scanned_at >= now - N days」（v128）。
    動機：解決「掃描頻率低 → N 次池含過老資料」的盲點。
    30 天涵蓋 BB 壓縮研究完整週期（壓縮事件 5-15 個交易日後突破 + 走勢觀察）。

    去重邏輯：保留每個 code 第一次出現（最近一次掃描）的 meta 欄位。

    Args:
        within_days: 取過去幾天內的掃描做聯集。預設 30 天。

    Returns:
        (latest_scan_id, [{"code", "name", "market", "industry"}, ...])
        若無歷史則回 (None, [])
        若 N 天內無掃描 → fallback 取最近 1 筆（避免完全空池）
    """
    db_path = settings.db_path / "tw_scan_history.db"
    if not db_path.exists():
        return None, []

    # 用 Python 算 cutoff（與 SQLite tz 解耦）
    cutoff = (datetime.now() - timedelta(days=max(1, within_days))).isoformat(timespec="seconds")

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT scan_id, scanned_at, results_json FROM tw_bb_scans "
            "WHERE scanned_at >= ? ORDER BY scanned_at DESC",
            (cutoff,),
        ).fetchall()
        # Fallback：N 天內 0 筆 → 取最近 1 筆，避免完全空池
        if not rows:
            rows = conn.execute(
                "SELECT scan_id, scanned_at, results_json FROM tw_bb_scans "
                "ORDER BY scanned_at DESC LIMIT 1"
            ).fetchall()
            if rows:
                logger.warning(
                    f"get_recent_scan_symbols: 過去 {within_days} 天無掃描歷史，"
                    f"降級用最近 1 筆 (scanned_at={rows[0][1]})"
                )
    finally:
        conn.close()

    if not rows:
        return None, []

    # 聯集去重：保留每個 code 第一次出現（最近一次掃描）的 meta
    seen: set[str] = set()
    union: list[dict] = []
    for scan_id, scanned_at, results_json in rows:
        try:
            results = json.loads(results_json)
        except Exception as e:
            logger.warning(f"tw_scan_history scan {scan_id} 解析失敗: {e}")
            continue
        for r in results:
            code = r.get("code", "")
            if not code or code in seen:
                continue
            seen.add(code)
            union.append({
                "code": code,
                "name": r.get("name", ""),
                "market": r.get("market", ""),
                "industry": r.get("industry", ""),
            })

    latest_scan_id = rows[0][0]
    logger.info(
        f"get_recent_scan_symbols: 聯集過去 {within_days} 天內 {len(rows)} 次掃描，"
        f"去重後 {len(union)} 個標的 (latest_scan_id={latest_scan_id})"
    )
    return latest_scan_id, union


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
    """在 df 上找到 target_date 對應的那一根 K 線，重算 5 個指標。

    回傳：{"close", "bb_pctile", "bb_width", "change_20d", "vol_5d"} 或 None（資料不足）。
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

    # BB Width 百分位（_BB_PERIOD=20, _BB_LOOKBACK=120）+ 帶寬絕對值 % + 突破上/下軌事件
    bb_pctile: Optional[float] = None
    bb_width_abs: Optional[float] = None  # (2*std/sma)*100，與掃描端定義一致
    break_up = False    # 當日收盤首次站上上軌（昨日仍在軌內）
    break_down = False  # 當日收盤首次跌破下軌
    if last_idx >= _BB_PERIOD - 1:
        # 取截至 target_date 的所有 close（含 target_date）
        close_series = df.iloc[: last_idx + 1]["close"]
        sma = close_series.rolling(window=_BB_PERIOD).mean()
        std = close_series.rolling(window=_BB_PERIOD).std()
        safe_sma = sma.replace(0, np.nan)
        bb_width = (2 * std / safe_sma) * 100

        current = bb_width.iloc[-1]
        if pd.notna(current):
            bb_width_abs = float(current)

        # 突破偵測：今昨兩日都要有完整 band
        if last_idx >= _BB_PERIOD:
            upper = sma + 2 * std
            lower = sma - 2 * std
            c_now, c_prev = float(close_series.iloc[-1]), float(close_series.iloc[-2])
            u_now, u_prev = upper.iloc[-1], upper.iloc[-2]
            l_now, l_prev = lower.iloc[-1], lower.iloc[-2]
            if pd.notna(u_now) and pd.notna(u_prev):
                break_up = bool(c_now > float(u_now) and c_prev <= float(u_prev))
                break_down = bool(c_now < float(l_now) and c_prev >= float(l_prev))

        # 滾動 lookback 百分位（最後一根）
        if len(bb_width.dropna()) >= _BB_LOOKBACK:
            recent = bb_width.tail(_BB_LOOKBACK)
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
        "bb_width": round(bb_width_abs, 2) if bb_width_abs is not None else None,
        "change_20d": round(change_20d, 2) if change_20d is not None else None,
        "vol_5d": vol_5d,
        "break_up": break_up,
        "break_down": break_down,
    }


# ──────────────────────────────────────────────
# 數據機械檢核（純程式、零成本；仿 mechanical_audit 哲學：只標記不阻擋）
# ──────────────────────────────────────────────

# 台股現股漲跌停 ±10%；加 0.5% 容差。超過幾乎必為除權息未還原或資料錯誤。
_MAX_DAILY_JUMP_PCT = 10.5
_MAX_MISSING_RATIO = 0.4   # 非週末日缺漏比例上限
_MAX_STALE_DAYS = 5        # 最新收盤落後 end_date 的日曆日上限
_MAX_ISSUES = 20


def run_data_checks(scan_dates: list[str], symbols: list[dict], end_date: str) -> dict:
    """對追蹤結果做合理性檢核，回傳 {passed, n_checks, n_issues, issues}。"""
    issues: list[str] = []
    n_checks = 0

    def _add(msg: str):
        if len(issues) < _MAX_ISSUES:
            issues.append(msg)

    weekdays = [d for d in scan_dates if pd.Timestamp(d).dayofweek < 5]

    for s in symbols:
        label = f"{s.get('code', '?')} {s.get('name', '')}"
        feats_map = s.get("daily_features") or {}

        prev_close: Optional[float] = None
        prev_date = ""
        n_missing_weekday = 0
        for d in scan_dates:
            f = feats_map.get(d)
            if f is None:
                if pd.Timestamp(d).dayofweek < 5:
                    n_missing_weekday += 1
                continue
            n_checks += 1
            bb_pctile = f.get("bb_pctile")
            if bb_pctile is not None and not (0 <= bb_pctile <= 100):
                _add(f"{label} {d[5:]}: BB% 超出 0-100 範圍（{bb_pctile}）")
            bb_width = f.get("bb_width")
            if bb_width is not None and not (0 < bb_width < 100):
                _add(f"{label} {d[5:]}: 帶寬異常（{bb_width}%）")
            close = f.get("close")
            if close is not None and close <= 0:
                _add(f"{label} {d[5:]}: 收盤價異常（{close}）")
            vol = f.get("vol_5d")
            if vol is not None and vol < 0:
                _add(f"{label} {d[5:]}: 均量為負（{vol}）")
            # 單日跳動 > 漲跌停限制 → 疑似除權息未還原 / 資料錯誤
            if close is not None and close > 0 and prev_close:
                jump = (close - prev_close) / prev_close * 100
                if abs(jump) > _MAX_DAILY_JUMP_PCT:
                    _add(
                        f"{label} {prev_date[5:]}→{d[5:]}: 單日跳動 {jump:+.1f}% "
                        f"超過台股 ±10% 漲跌停，疑似除權息未還原或資料異常"
                    )
            if close is not None and close > 0:
                prev_close, prev_date = close, d

        if weekdays and n_missing_weekday / len(weekdays) > _MAX_MISSING_RATIO:
            _add(f"{label}: 區間內 {n_missing_weekday}/{len(weekdays)} 個交易日無資料，缺漏率偏高")

        lcd = s.get("latest_close_date")
        if lcd:
            n_checks += 1
            stale = (pd.Timestamp(end_date) - pd.Timestamp(lcd)).days
            if stale > _MAX_STALE_DAYS:
                _add(f"{label}: 最新收盤停在 {lcd}，落後區間終點 {stale} 天，資料可能未更新")

    return {
        "passed": len(issues) == 0,
        "n_checks": n_checks,
        "n_issues": len(issues),
        "issues": issues,
    }


def _safe_data_check(scan_dates: list[str], symbols: list[dict], end_date: str) -> Optional[dict]:
    """檢核失敗不影響主流程（graceful）"""
    try:
        return run_data_checks(scan_dates, symbols, end_date)
    except Exception as e:
        logger.debug(f"run_data_checks 失敗（不影響主流程）: {e}")
        return None


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
        max_abs_bb_width: float = 0.0,  # 帶寬絕對值上限 %（AND 進每日 matched）；0 = 不過濾
        max_close: float = 0.0,         # 最新收盤價上限（標的級剔除）；0 = 不過濾
        min_vol_5d: float = 0.0,        # 最新 5 日均量下限（張，標的級剔除）；0 = 不過濾
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
                latest_close: Optional[float] = None
                latest_close_date: Optional[str] = None
                latest_vol: Optional[int] = None
                for date_str, ts in zip(scan_dates_str, date_range):
                    feats = _compute_features_on_df(df, ts)
                    if feats is None:
                        daily_features[date_str] = None
                        continue
                    matched = (
                        feats["bb_pctile"] is not None
                        and feats["bb_pctile"] < pctile_threshold
                        and (
                            max_abs_bb_width <= 0
                            or (feats["bb_width"] is not None and feats["bb_width"] < max_abs_bb_width)
                        )
                    )
                    # 壓縮後突破：範圍內先出現過符合日（或當日仍符合），才把突破事件標出來
                    break_up = feats.pop("break_up", False)
                    break_down = feats.pop("break_down", False)
                    breakout: Optional[str] = None
                    if matched_dates or matched:
                        if break_up:
                            breakout = "up"
                        elif break_down:
                            breakout = "down"
                    daily_features[date_str] = {**feats, "matched": matched, "breakout": breakout}
                    if matched:
                        matched_dates.append(date_str)
                    if feats["close"] is not None:
                        latest_close = feats["close"]
                        latest_close_date = date_str
                    if feats["vol_5d"] is not None:
                        latest_vol = feats["vol_5d"]

                if not matched_dates:
                    # 此標的範圍內全未符合 → 不納入結果（避免表格塞爆）
                    return None

                # 標的級篩選：用範圍內最新一日的值（排除高價股 / 流動性不足）
                if max_close > 0 and latest_close is not None and latest_close > max_close:
                    return None
                if min_vol_5d > 0 and latest_vol is not None and latest_vol < min_vol_5d:
                    return None

                return {
                    "code": code,
                    "name": meta["name"],
                    "market": meta["market"],
                    "industry": meta["industry"],
                    "first_match_date": matched_dates[0],
                    "last_match_date": matched_dates[-1],
                    "match_count": len(matched_dates),
                    "latest_close": latest_close,
                    "latest_close_date": latest_close_date,
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
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "data_check": _safe_data_check(scan_dates_str, symbols_out, end_date),
        }


# ──────────────────────────────────────────────
# CSV 匯出
# ──────────────────────────────────────────────

def export_to_csv(result: dict, filter_note: str = "") -> str:
    """將跨日追蹤結果組成 CSV 文字（5 指標＋突破 × N 天；filter_note 附加為最後一列）。"""
    scan_dates: list[str] = result.get("scan_dates", [])
    symbols: list[dict] = result.get("symbols", [])

    buf = io.StringIO()
    # Excel 開 UTF-8 CSV 加 BOM 才不會亂碼
    buf.write("﻿")

    writer = csv.writer(buf)

    header = ["代號", "名稱", "市場", "產業", "首次符合", "符合天數", "最新價", "最新價日期"]
    for d in scan_dates:
        header.extend([f"{d}_BB%", f"{d}_帶寬%", f"{d}_收盤", f"{d}_20日漲跌%", f"{d}_5日均量", f"{d}_突破"])
    writer.writerow(header)

    for s in symbols:
        row = [
            s.get("code", ""),
            s.get("name", ""),
            s.get("market", ""),
            s.get("industry", ""),
            s.get("first_match_date", ""),
            s.get("match_count", 0),
            s.get("latest_close", "") if s.get("latest_close") is not None else "",
            s.get("latest_close_date", "") or "",
        ]
        feats_map = s.get("daily_features", {})
        for d in scan_dates:
            feats = feats_map.get(d) or {}
            row.extend([
                feats.get("bb_pctile", "") if feats.get("bb_pctile") is not None else "",
                feats.get("bb_width", "") if feats.get("bb_width") is not None else "",
                feats.get("close", "") if feats.get("close") is not None else "",
                feats.get("change_20d", "") if feats.get("change_20d") is not None else "",
                feats.get("vol_5d", "") if feats.get("vol_5d") is not None else "",
                {"up": "↑", "down": "↓"}.get(feats.get("breakout") or "", ""),
            ])
        writer.writerow(row)

    if filter_note:
        writer.writerow([])
        writer.writerow(["篩選條件", filter_note])

    return buf.getvalue()


# 模組單例
tw_track_range = TwTrackRange()
