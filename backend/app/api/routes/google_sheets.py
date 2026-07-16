"""阿斯拉量化系統 — Google Sheets 匯出 API（Apps Script Webhook）

提供：
- 設定精靈：設密碼 → 取得嵌密碼的腳本 → 驗證並儲存部署網址
- 匯出：前端送畫面表格（headers + rows），轉發給使用者的 Apps Script 寫入試算表
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from app.core.google_sheets import GoogleSheetsError, google_sheets_webhook


router = APIRouter()

# GoogleSheetsError.code → HTTP status
_ERROR_STATUS = {
    "not_configured": 409,
    "password_required": 401,
    "bad_password": 400,
    "bad_url": 400,
    "unauthorized": 400,
    "webhook_unreachable": 502,
    "remote_error": 502,
}


_CODE_ALIAS = {
    "not_configured": "gsheet_not_configured",
    "password_required": "gsheet_password_required",
}


def _http_error(e: GoogleSheetsError) -> HTTPException:
    return HTTPException(
        status_code=_ERROR_STATUS.get(e.code, 500),
        detail={"code": _CODE_ALIAS.get(e.code, e.code), "message": e.message},
    )


class SetupRequest(BaseModel):
    password: str = Field(..., description="匯出密碼（將嵌入 Apps Script）")


class ConfigRequest(BaseModel):
    webhook_url: str = Field(..., description="Apps Script Web App 部署網址（/exec 結尾）")


class SheetExportRequest(BaseModel):
    spreadsheet_url: str = Field(..., description="目標 Google 試算表網址")
    sheet_title: Optional[str] = Field(default=None, description="新分頁名稱（同名自動加序號）")
    headers: list[str] = Field(..., min_length=1)
    rows: list[list[Any]] = Field(default_factory=list)
    colors: Optional[list[list[Optional[str]]]] = Field(
        default=None, description="與 rows 同形狀的字體色 hex 矩陣（null = 預設色）"
    )


@router.get("/setup")
async def get_setup_status():
    """精靈開場：是否已完成設定 + 密碼是否已在記憶體"""
    return {
        "configured": google_sheets_webhook.is_configured(),
        "password_loaded": google_sheets_webhook.has_password(),
    }


@router.post("/setup")
async def set_password(req: SetupRequest):
    """精靈第一步：設定密碼，回傳嵌入密碼的 Apps Script 腳本"""
    try:
        google_sheets_webhook.set_password(req.password)
        return {"script_code": google_sheets_webhook.build_script_code()}
    except GoogleSheetsError as e:
        raise _http_error(e)


class UnlockRequest(BaseModel):
    password: str = Field(..., description="匯出密碼（後端重啟後重新輸入）")


@router.post("/password")
async def unlock_password(req: UnlockRequest):
    """後端重啟後重新輸入密碼：ping 驗證與腳本相符才收進記憶體"""
    try:
        await google_sheets_webhook.unlock(req.password)
        return {"ok": True}
    except GoogleSheetsError as e:
        raise _http_error(e)


@router.post("/config")
async def save_webhook(req: ConfigRequest):
    """精靈最後一步：ping 驗證部署網址（密碼須對得上）後儲存"""
    try:
        await google_sheets_webhook.save_webhook_url(req.webhook_url)
        return {"ok": True}
    except GoogleSheetsError as e:
        raise _http_error(e)


@router.delete("/config")
async def reset_config():
    """清除設定（重跑精靈 / 換密碼用）"""
    google_sheets_webhook.reset()
    return {"ok": True}


@router.post("/export")
async def export_to_sheet(req: SheetExportRequest):
    """把表格寫入試算表新分頁；未設定時回 409 讓前端開精靈"""
    if not google_sheets_webhook.is_configured():
        raise HTTPException(
            status_code=409,
            detail={"code": "gsheet_not_configured", "message": "尚未完成 Google Sheet 匯出設定"},
        )
    if not req.rows:
        raise HTTPException(status_code=400, detail={"code": "empty", "message": "沒有可匯出的資料"})
    if req.colors is not None and len(req.colors) != len(req.rows):
        raise HTTPException(status_code=400, detail={"code": "bad_colors", "message": "colors 與 rows 列數不一致"})
    try:
        result = await google_sheets_webhook.export_table(
            spreadsheet_url=req.spreadsheet_url,
            sheet_title=req.sheet_title or "匯出",
            headers=req.headers,
            rows=req.rows,
            colors=req.colors,
        )
        logger.info(
            f"[GoogleSheets] 匯出成功：{result['sheet_title']}（{len(req.rows)} 列）"
        )
        return result
    except GoogleSheetsError as e:
        raise _http_error(e)
