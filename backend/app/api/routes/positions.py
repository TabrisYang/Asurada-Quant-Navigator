"""阿斯拉量化系統 — v106 A2：持倉追蹤 API endpoints。

純本地 SQLite，不接交易所 API、不上雲。
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.position_tracker import position_tracker

router = APIRouter(prefix="/api/positions", tags=["positions"])


class PositionCreate(BaseModel):
    symbol: str = Field(..., description="例：BTC/USDT")
    direction: str = Field(..., description="long / short")
    size_usd: float = Field(..., gt=0, description="美元曝險")
    avg_entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    leverage: float = 1.0
    notes: str = ""


class PositionUpdate(BaseModel):
    size_usd: Optional[float] = None
    avg_entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    leverage: Optional[float] = None
    notes: Optional[str] = None
    status: Optional[str] = None


@router.get("/list")
async def list_positions(symbol: Optional[str] = None, all: bool = False) -> dict:
    """列持倉。all=True 回傳含已關閉的。"""
    if all:
        positions = position_tracker.list_all(limit=200)
    else:
        positions = position_tracker.list_open(symbol=symbol)
    return {"status": "success", "positions": positions, "count": len(positions)}


@router.get("/summary")
async def get_summary() -> dict:
    """整體組合摘要：總曝險、各 symbol 比重、淨多空比、過期警示。"""
    summary = position_tracker.get_summary()
    return {"status": "success", **summary}


@router.post("/add")
async def add_position(body: PositionCreate) -> dict:
    """新增持倉。"""
    if body.direction not in ("long", "short"):
        raise HTTPException(400, "direction 必須是 long 或 short")
    pid = position_tracker.add(
        symbol=body.symbol,
        direction=body.direction,
        size_usd=body.size_usd,
        avg_entry_price=body.avg_entry_price,
        stop_loss=body.stop_loss,
        take_profit=body.take_profit,
        leverage=body.leverage,
        notes=body.notes,
    )
    return {"status": "success", "id": pid}


@router.patch("/{pid}")
async def update_position(pid: int, body: PositionUpdate) -> dict:
    """更新持倉欄位。"""
    fields = body.model_dump(exclude_none=True)
    if not fields:
        raise HTTPException(400, "至少需提供一個更新欄位")
    ok = position_tracker.update(pid, **fields)
    if not ok:
        raise HTTPException(404, f"找不到 position id={pid}")
    return {"status": "success", "updated_fields": list(fields.keys())}


@router.post("/{pid}/close")
async def close_position(pid: int) -> dict:
    """關閉持倉（不刪除，標 status=closed 保留歷史）。"""
    ok = position_tracker.close(pid)
    if not ok:
        raise HTTPException(404, f"找不到 position id={pid}")
    return {"status": "success", "closed": pid}


@router.delete("/{pid}")
async def delete_position(pid: int) -> dict:
    """真刪除（罕用，通常用 close）。"""
    ok = position_tracker.delete(pid)
    if not ok:
        raise HTTPException(404, f"找不到 position id={pid}")
    return {"status": "success", "deleted": pid}
