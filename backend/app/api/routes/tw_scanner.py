"""阿斯拉量化系統 — 台股掃描器 API

提供：
- SSE 串流掃描（即時進度 + 結果）
- 掃描歷史列表 / 單次結果 / 刪除
- 「回看」：對歷史某次結果取當前價格，算後續漲跌幅
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from fastapi.responses import Response

from app.core.tw_bb_scanner import tw_bb_scanner
from app.core.tw_scan_history import tw_scan_history
from app.core.tw_track_range import export_to_csv, tw_track_range
from app.data.fetchers.tw_fundamental import tw_fundamental
from app.data.fetchers.tw_stock_engine import TwStockEngine


router = APIRouter()


class TwBBScanRequest(BaseModel):
    timeframe: str = Field(default="1d", description="日線或週線（目前僅支援 1d）")
    pctile_threshold: float = Field(default=20.0, ge=1.0, le=50.0, description="BB Width 百分位門檻")
    markets: Optional[list[str]] = Field(default=None, description="['listed','otc']，預設兩個都掃")
    # 進階過濾
    min_volume: int = Field(default=0, ge=0, description="5 日均量門檻（張）；0 = 不過濾")
    require_healthy_trend: bool = Field(default=False, description="趨勢健康：MA60 斜率 > 0 或 收盤 > MA20 > MA60")
    max_adx: Optional[float] = Field(default=None, description="ADX 上限（< 25 為真盤整）；None 不過濾")
    persistence_bars: int = Field(default=1, ge=1, le=10, description="壓縮持續性：最近 N 根都要 < 門檻")
    min_abs_bb_width: float = Field(default=0.0, ge=0.0, le=50.0, description="絕對 BB Width 下限 %（排除常年低波動）")
    history_days: int = Field(default=400, ge=220, le=3000, description="抓取歷史天數（日曆日）；下限 220 ≈ 150 交易日")
    # v137：v136 Bollinger 訊號（預設開）；設 False 可 fallback 純 BB Width 掃描
    enable_v136: bool = Field(default=True, description="是否算 v136 完整布林通道訊號（Squeeze / 突破 / Walking / 反轉）")


def _sse_event(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/tw-bb-width")
async def scan_tw_bb_width(request: TwBBScanRequest):
    """啟動台股 BB Width 壓縮掃描，SSE 串流回傳進度和結果，結束後存入歷史。"""

    async def event_stream():
        collected_results: list[dict] = []
        collected_failures: list[dict] = []
        try:
            async for evt in tw_bb_scanner.scan(
                timeframe=request.timeframe,
                pctile_threshold=request.pctile_threshold,
                markets=request.markets,
                min_volume=request.min_volume,
                require_healthy_trend=request.require_healthy_trend,
                max_adx=request.max_adx,
                persistence_bars=request.persistence_bars,
                min_abs_bb_width=request.min_abs_bb_width,
                history_days=request.history_days,
                enable_v136=request.enable_v136,
            ):
                evt_type = evt.pop("type")
                if evt_type == "result":
                    collected_results.append(evt)
                elif evt_type == "failure":
                    collected_failures.append(evt)
                elif evt_type == "done":
                    # 以掃描器回傳的 failures 為權威來源（理論上與 collected 一致）
                    failures_payload = evt.pop("failures", None) or collected_failures
                    # 保存到歷史 DB
                    scan_id = tw_scan_history.save(
                        timeframe=request.timeframe,
                        params=request.model_dump(exclude_none=False),
                        results=collected_results,
                        total_scanned=evt.get("total_scanned", 0),
                        total_found=evt.get("total_found", 0),
                        total_fail=evt.get("total_fail", 0),
                        duration_sec=evt.get("duration_sec", 0.0),
                        failures=failures_payload,
                    )
                    evt["scan_id"] = scan_id
                    logger.info(
                        f"掃描結果已存歷史: {scan_id}（失敗 {len(failures_payload)} 檔）"
                    )
                yield _sse_event(evt_type, evt)
        except Exception as e:
            logger.exception("掃描過程發生未預期錯誤")
            yield _sse_event("error", {"error": f"掃描錯誤: {e}"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ─── 歷史 API ────────────────────────────────────

@router.get("/tw-bb-width/history")
async def list_scan_history(limit: int = 20):
    """列出最近 N 次掃描。"""
    return {"scans": tw_scan_history.list_recent(limit=limit)}


@router.get("/tw-bb-width/history/{scan_id}")
async def get_scan_result(scan_id: str):
    """取特定一次完整結果。"""
    result = tw_scan_history.get(scan_id)
    if not result:
        raise HTTPException(status_code=404, detail="找不到該掃描紀錄")
    return result


@router.delete("/tw-bb-width/history/{scan_id}")
async def delete_scan(scan_id: str):
    if not tw_scan_history.delete(scan_id):
        raise HTTPException(status_code=404, detail="找不到該掃描紀錄")
    return {"status": "deleted", "scan_id": scan_id}


@router.get("/tw-bb-width/history/{scan_id}/revisit")
async def revisit_scan(scan_id: str):
    """回看：取歷史某次結果的每檔當前價，算後續漲跌幅。

    回傳每檔含：
      - scan_price / scan_date（當時掃到的價格與日期）
      - current_price / current_date（現在的收盤）
      - return_pct（漲跌幅 %）
    按 return_pct 降序排序，讓「當時掃到後續漲最多」的排最上面。
    """
    scan = tw_scan_history.get(scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="找不到該掃描紀錄")

    engine = TwStockEngine()
    enriched: list[dict] = []

    async def fetch_one(item: dict):
        code = item["code"]
        try:
            df = await engine.fetch_ohlcv(symbol=f"{code}/TWD", timeframe=scan["timeframe"])
            if df is None or df.empty:
                return None
            last = df.iloc[-1]
            cur_price = float(last["close"])
            cur_date = str(last["timestamp"])[:10]
            scan_price = float(item.get("price", 0))
            ret = (cur_price - scan_price) / scan_price * 100 if scan_price > 0 else 0.0
            return {
                **item,
                "scan_price": scan_price,
                "scan_date": item.get("price_date"),
                "current_price": round(cur_price, 2),
                "current_date": cur_date,
                "return_pct": round(ret, 2),
            }
        except Exception as e:
            logger.debug(f"revisit {code} 失敗: {e}")
            return None

    sem = asyncio.Semaphore(10)

    async def bounded(item):
        async with sem:
            return await fetch_one(item)

    tasks = [bounded(r) for r in scan["results"]]
    results = await asyncio.gather(*tasks)
    enriched = [r for r in results if r is not None]
    enriched.sort(key=lambda r: r.get("return_pct", 0), reverse=True)

    return {
        "scan_id": scan_id,
        "scanned_at": scan["scanned_at"],
        "timeframe": scan["timeframe"],
        "total_original": len(scan["results"]),
        "total_revisited": len(enriched),
        "results": enriched,
    }


# ─── 跨日追蹤 API ────────────────────────────────

@router.get("/tw-bb-width/range")
async def stream_tw_bb_range(
    start_date: str,
    end_date: str,
    scope: str = "recent_scan",
    pctile_threshold: float = 20.0,
    max_abs_bb_width: float = 0.0,
    max_close: float = 0.0,
    min_vol_5d: float = 0.0,
):
    """跨日追蹤：對「最近一次掃描標的池」或「全市場」執行：
    - 撈 OHLCV
    - 逐日重算 5 個指標（close / bb_pctile / bb_width / change_20d / vol_5d）
    - 標出該日是否符合 BB% < pctile_threshold（且帶寬 < max_abs_bb_width，若有設）
    - max_close / min_vol_5d 用最新一日的值做標的級剔除（0 = 不過濾）

    SSE 流：progress / result / error
    """
    if scope not in ("recent_scan", "full_market"):
        raise HTTPException(status_code=400, detail="scope 必須是 recent_scan 或 full_market")

    async def event_stream():
        last_result: Optional[dict] = None
        try:
            async for evt in tw_track_range.stream(
                start_date=start_date,
                end_date=end_date,
                scope=scope,
                pctile_threshold=pctile_threshold,
                max_abs_bb_width=max_abs_bb_width,
                max_close=max_close,
                min_vol_5d=min_vol_5d,
            ):
                evt_type = evt.pop("type")
                if evt_type == "result":
                    last_result = evt
                yield _sse_event(evt_type, evt)
        except Exception as e:
            logger.exception("跨日追蹤過程發生未預期錯誤")
            yield _sse_event("error", {"error": f"追蹤錯誤: {e}"})

        # 將 last_result 快取到記憶體，供 export 端點使用（用 query string params 重抓也可，但成本高）
        # 註：目前不做快取，export 端點重新跑一次 stream 取結果。

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/tw-bb-width/range/export")
async def export_tw_bb_range(
    start_date: str,
    end_date: str,
    scope: str = "recent_scan",
    pctile_threshold: float = 20.0,
    max_abs_bb_width: float = 0.0,
    max_close: float = 0.0,
    min_vol_5d: float = 0.0,
):
    """跨日追蹤結果 CSV 匯出。

    回傳 text/csv（含 BOM，Excel 開不亂碼）。
    重新跑一次追蹤聚合（不依賴 SSE 快照）。
    """
    if scope not in ("recent_scan", "full_market"):
        raise HTTPException(status_code=400, detail="scope 必須是 recent_scan 或 full_market")

    final_result: Optional[dict] = None
    async for evt in tw_track_range.stream(
        start_date=start_date,
        end_date=end_date,
        scope=scope,
        pctile_threshold=pctile_threshold,
        max_abs_bb_width=max_abs_bb_width,
        max_close=max_close,
        min_vol_5d=min_vol_5d,
    ):
        if evt.get("type") == "result":
            final_result = evt
            break

    if final_result is None:
        raise HTTPException(status_code=500, detail="追蹤未產生結果")

    note_parts = [f"BB% < {pctile_threshold}%"]
    if max_abs_bb_width > 0:
        note_parts.append(f"帶寬 < {max_abs_bb_width}%")
    if max_close > 0:
        note_parts.append(f"收盤價 ≤ {max_close}")
    if min_vol_5d > 0:
        note_parts.append(f"5日均量 ≥ {min_vol_5d}張")
    note_parts.append(f"區間 {start_date}~{end_date}")
    note_parts.append("標的池：全市場" if scope == "full_market" else "標的池：最近30天掃描聯集")
    csv_text = export_to_csv(final_result, filter_note="；".join(note_parts))
    filename = f"tw_bb_track_{start_date}_to_{end_date}_{scope}.csv"
    return Response(
        content=csv_text,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


# ─── EPS 批次查詢（跨日追蹤表格的「上年度 / 最新季 EPS」欄位）────

# In-memory cache：code → (timestamp, ttl, eps_dict)。
# 成功（quality=ok）快取 24h；失敗只快取 10 分鐘 — 否則 FinMind 額度用盡等暫時性失敗
# 會讓 EPS 欄整天空白（v152 修復：失敗不再佔用 24h cache）。
_EPS_CACHE: dict[str, tuple[float, float, dict]] = {}
_EPS_TTL_OK = 24 * 3600
_EPS_TTL_MISS = 600
_EPS_BATCH_CONCURRENCY = 8  # 並行抓 yfinance 上限，避免被 rate limit


async def _get_eps_with_cache(code: str) -> dict:
    """單檔 EPS 抓取 + cache（成功 24h、失敗 10 分鐘後自動重試）。"""
    import time
    now = time.time()
    cached = _EPS_CACHE.get(code)
    if cached and now - cached[0] < cached[1]:
        return cached[2]
    eps = await tw_fundamental.fetch_eps_breakdown(code)
    ttl = _EPS_TTL_OK if (eps or {}).get("quality") == "ok" else _EPS_TTL_MISS
    _EPS_CACHE[code] = (now, ttl, eps)
    return eps


@router.get("/tw-stock/eps_batch")
async def get_eps_batch(codes: str):
    """批次取台股 EPS（上年度 + 最新季）。

    Query:
      codes — 逗號分隔的股票代號，例：codes=2330,2337,4967
    回傳：
      {"data": {"2330": {annual_prev, annual_prev_label, quarter_latest, quarter_latest_label, eps_trailing}, ...}}
    Cache：in-memory 24h，失敗也快取（避免反覆打）。
    """
    code_list = [c.strip() for c in codes.split(",") if c.strip()]
    if not code_list:
        return {"data": {}}
    if len(code_list) > 200:
        raise HTTPException(status_code=400, detail="一次最多 200 檔")

    sem = asyncio.Semaphore(_EPS_BATCH_CONCURRENCY)

    async def _one(code: str) -> tuple[str, dict]:
        async with sem:
            return code, await _get_eps_with_cache(code)

    results = await asyncio.gather(*(_one(c) for c in code_list), return_exceptions=True)

    data: dict[str, dict] = {}
    for r in results:
        if isinstance(r, Exception):
            logger.warning(f"[eps_batch] 抓取失敗：{r}")
            continue
        code, eps = r
        data[code] = eps
    return {"data": data}
