"""互動式教學 API — 課程內容 + 瘦回測端點。

「學理論 → 立即回測驗證」的直呼通道：不經 LLM function calling，
複用 run_backtest 純函式與既有資料引擎，點了秒回。
條件由後端課程範本產生（前端只傳參數值），杜絕任意條件注入。
"""

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from app.core.backtest import run_backtest
from app.core.learn.lessons import build_conditions, get_lesson, get_lesson_summaries
from app.data.fetchers.crypto_engine import crypto_engine
from app.data.fetchers.tw_stock_engine import tw_stock_engine
from app.utils.symbol import is_tw_stock

router = APIRouter()

_MIN_BARS = 200          # 教學回測至少需要的 K 線數（統計上才有意義）
_EQUITY_MAX_POINTS = 500  # equity curve 降採樣上限（前端繪圖用）


class LearnBacktestRequest(BaseModel):
    lesson_id: str
    symbol: str
    timeframe: str | None = None  # 不傳則用課程範本預設
    params: dict[str, float] = Field(default_factory=dict)  # 只接受 tunable_params 宣告的鍵


@router.get("/lessons")
async def list_lessons():
    return {"lessons": get_lesson_summaries()}


@router.get("/lessons/{lesson_id}")
async def lesson_detail(lesson_id: str):
    lesson = get_lesson(lesson_id)
    if lesson is None:
        raise HTTPException(status_code=404, detail=f"找不到課程 {lesson_id}")
    return lesson


@router.post("/run")
async def run_lesson_backtest(req: LearnBacktestRequest):
    """執行課程範本回測（全部本地歷史資料，與聊天回測共用引擎與數據）。"""
    spec = build_conditions(req.lesson_id, req.params)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"找不到課程 {req.lesson_id}")

    timeframe = req.timeframe or spec["timeframe"]
    engine = tw_stock_engine if is_tw_stock(req.symbol) else crypto_engine
    df = engine.load_local_data(req.symbol, timeframe)
    if df is None or df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"找不到 {req.symbol} {timeframe} 的本地數據，請先到「同步數據」補資料。",
        )
    if len(df) < _MIN_BARS:
        raise HTTPException(
            status_code=422,
            detail=f"{req.symbol} {timeframe} 只有 {len(df)} 根 K 線（教學回測至少需 {_MIN_BARS} 根），請先補深歷史資料。",
        )

    try:
        result = run_backtest(
            df=df,
            entry_conditions=spec["entry_conditions"],
            exit_conditions=spec["exit_conditions"],
            direction=spec["direction"],
            entry_logic=spec["entry_logic"],
            exit_logic=spec["exit_logic"],
            timeframe=timeframe,
        )
    except Exception as e:  # 引擎層例外統一轉為可讀錯誤
        logger.error(f"教學回測失敗 [{req.lesson_id} / {req.symbol}]: {e}")
        raise HTTPException(status_code=500, detail=f"回測執行失敗：{e}")

    d = result.to_dict()

    # equity curve：補上真實 timestamp（bar index 對應 df 同位置），降採樣後回傳
    timestamps = df["timestamp"].astype(str).tolist()
    curve = [
        {"time": timestamps[p["bar"]], "equity": p["equity"]}
        for p in result.equity_curve
        if p["bar"] < len(timestamps)
    ]
    if len(curve) > _EQUITY_MAX_POINTS:
        stride = len(curve) / _EQUITY_MAX_POINTS
        sampled = [curve[int(i * stride)] for i in range(_EQUITY_MAX_POINTS)]
        if sampled[-1] is not curve[-1]:
            sampled.append(curve[-1])  # 終值必留，避免總報酬視覺失真
        curve = sampled
    d["equity_curve"] = curve

    d["data_range"] = {
        "start": timestamps[0][:10] if timestamps else None,
        "end": timestamps[-1][:10] if timestamps else None,
        "bars": len(df),
        "timeframe": timeframe,
    }
    d["resolved_params"] = spec["resolved_params"]
    return d
