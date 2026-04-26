"""阿斯拉量化系統 — 系統健康狀態 API。

GET /api/system/health  → 回傳最新 system_health.json 內容（前端啟動時讀，顯示警告 banner）

來源資料是 audit_system_health.py 寫的，由 launchd 每天 0:30 更新；
也可手動 `python3 backend/scripts/audit_system_health.py` 觸發。
"""

import json
from pathlib import Path

from fastapi import APIRouter

from app.core.config.settings import settings


router = APIRouter()
_HEALTH_FILE = settings.db_path / "system_health.json"


@router.get("/health")
async def get_system_health():
    """回傳最新健康檢查結果，供前端啟動時讀取。"""
    if not _HEALTH_FILE.exists():
        return {
            "status": "no_data",
            "message": "尚未執行健康檢查（執行 backend/scripts/audit_system_health.py 或安裝 launchd 排程）",
        }

    try:
        return json.loads(_HEALTH_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        return {
            "status": "error",
            "message": f"讀取健康狀態失敗：{e}",
        }
