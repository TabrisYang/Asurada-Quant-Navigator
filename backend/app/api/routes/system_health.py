"""阿斯拉量化系統 — 系統健康狀態 API。

GET /api/system/health             → 最新 system_health.json（前端啟動時讀，顯示警告 banner）
GET /api/system/v132_eval_status   → v132 編排管線 2 週評估結果（讀 launchd 寫的 marker）
POST /api/system/v132_rollback     → 把 .env 的 COMPREHENSIVE_PIPELINE_ENABLED 改回 false
"""

import json
import logging
import os
import re
import tempfile
from datetime import date, datetime
from pathlib import Path

from fastapi import APIRouter

from app.core.config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter()

_HEALTH_FILE = settings.db_path / "system_health.json"

# v132：launchd `com.asurada.shadow_pipeline_eval` 寫的 marker / log 路徑
# 與 ~/Library/LaunchAgents/com.asurada.shadow_pipeline_eval.plist 內的路徑一致
_V132_MARKER_FILE = settings.db_path / ".shadow_pipeline_eval_done"
_V132_LOG_FILE = settings.db_path / "launchd_shadow_eval.log"
_V132_TARGET_DATE = date(2026, 5, 29)  # plist 內的 --target-date

# .env 路徑：launchd plist 工作目錄是 backend/，所以 .env 跟 backend/ 同層
_BACKEND_DIR = Path(__file__).resolve().parents[3]  # routes/system_health.py → backend/
_ENV_FILE = _BACKEND_DIR / ".env"
_FLAG_KEY = "COMPREHENSIVE_PIPELINE_ENABLED"


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


# ─── v132 編排管線 2 週評估狀態 ────────────────────────────


def _parse_marker(text: str) -> dict:
    """compare_shadow_reports.py:232 寫的 marker 格式是 key=value 純文字。"""
    out: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def _tail_log(path: Path, max_lines: int = 50) -> str:
    """回傳 log 檔尾段 N 行（給 banner 摺疊區顯示）。"""
    if not path.exists():
        return ""
    try:
        # 簡單實作：read + split。launchd log 不會很大（每天最多幾百行）
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max_lines:])
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[v132_eval] 讀 log tail 失敗：{e}")
        return ""


@router.get("/v132_eval_status")
async def get_v132_eval_status():
    """v132 編排管線上線 2 週後的自動評估結果。

    回傳:
      - status: "pending" | "scheduled" | "passed" | "degraded" | "error"
      - pending   = 今日 < target 且無 marker（還沒到評估日）
      - scheduled = 今日 >= target 但 marker 未生成（等下次 09:00 launchd 觸發）
      - passed    = marker 存在 + has_degradation=False
      - degraded  = marker 存在 + has_degradation=True（應該回滾）
      - error     = marker 解析失敗或意外錯誤
    """
    today = datetime.now().date()
    target_iso = _V132_TARGET_DATE.isoformat()

    if not _V132_MARKER_FILE.exists():
        if today < _V132_TARGET_DATE:
            return {
                "status": "pending",
                "target_date": target_iso,
                "message": f"評估排程在 {target_iso} 之後系統開啟時自動觸發",
            }
        return {
            "status": "scheduled",
            "target_date": target_iso,
            "message": f"已過排程日 {target_iso}，等下次 09:00 launchd 觸發",
        }

    try:
        marker = _parse_marker(_V132_MARKER_FILE.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        logger.error(f"[v132_eval] marker 解析失敗：{e}")
        return {
            "status": "error",
            "message": f"評估資料讀取失敗：{e}",
        }

    has_deg_raw = marker.get("has_degradation", "").lower()
    has_degradation = has_deg_raw in ("true", "1", "yes")

    return {
        "status": "degraded" if has_degradation else "passed",
        "target_date": target_iso,
        "evaluated_at": marker.get("completed_at", ""),
        "has_degradation": has_degradation,
        "baseline": marker.get("baseline", ""),
        "new_report": marker.get("new", ""),
        "log_tail": _tail_log(_V132_LOG_FILE),
    }


# ─── v132 一鍵回滾（.env 把 flag 改 false）────────────────────


def _atomic_rewrite_env(set_flag_false: bool) -> tuple[bool, str]:
    """原子寫入 .env：把 COMPREHENSIVE_PIPELINE_ENABLED 那行改成 false。

    回傳 (changed, message)：
      changed=True  → 真的改了某行
      changed=False → .env 不存在 / 該行不存在 / 已經是 false（idempotent）
    """
    if not _ENV_FILE.exists():
        return False, ".env 不存在，無需處理（flag 預設 False）"

    try:
        original = _ENV_FILE.read_text(encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        return False, f"讀 .env 失敗：{e}"

    # 找 COMPREHENSIVE_PIPELINE_ENABLED=... 那行（允許前後空白、引號）
    pattern = re.compile(
        rf'^(\s*){re.escape(_FLAG_KEY)}\s*=\s*["\']?([^"\'#\s]+)["\']?\s*(#.*)?$',
        re.MULTILINE,
    )
    m = pattern.search(original)
    if not m:
        return False, f".env 內無 {_FLAG_KEY} 設定（預設即 False，無需處理）"

    current_value = m.group(2).lower()
    if current_value in ("false", "0", "no", "off"):
        return False, f"{_FLAG_KEY} 已是 false，無需處理"

    if not set_flag_false:
        return False, "未指定要改 false（內部錯誤）"

    new_text = pattern.sub(
        lambda mt: f"{mt.group(1)}{_FLAG_KEY}=false" + (f"  {mt.group(3)}" if mt.group(3) else ""),
        original,
        count=1,
    )

    # Atomic write：寫 .env.tmp → os.replace (POSIX atomic rename)
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8",
            dir=_BACKEND_DIR, prefix=".env.", suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp.write(new_text)
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, _ENV_FILE)
    except Exception as e:  # noqa: BLE001
        logger.error(f"[v132_rollback] 寫 .env 失敗：{e}")
        return False, f"寫 .env 失敗：{e}"

    return True, f"已將 {_FLAG_KEY} 改為 false"


@router.post("/v132_rollback")
async def v132_rollback():
    """把 .env 的 COMPREHENSIVE_PIPELINE_ENABLED 改 false。

    不自動重啟後端 —— 自殺式 kill 有風險，且使用者可能正在用其他功能。
    回傳指示使用者雙擊 .command 重啟。idempotent：已是 false 直接回 ok。
    """
    changed, msg = _atomic_rewrite_env(set_flag_false=True)
    logger.info(f"[v132_rollback] changed={changed}, msg={msg}")
    return {
        "ok": True,
        "changed": changed,
        "needs_restart": changed,
        "message": (
            f"{msg}。請雙擊 阿斯拉量化系統.command 重啟後端，新版即關閉。"
            if changed
            else f"{msg}。新版本來就沒開，無需重啟。"
        ),
    }
