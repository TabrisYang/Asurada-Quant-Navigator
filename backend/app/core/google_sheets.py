"""阿斯拉量化系統 — Google Sheets 匯出（Apps Script Webhook）

使用者一次性在 script.google.com 部署 doPost 腳本（內嵌自訂密碼），
之後系統把表格資料 POST 到該 Web App，由腳本以使用者身分寫入試算表新分頁。

安全模型：
- 密碼「僅存記憶體」不落地 — 後端重啟後第一次匯出需重新輸入（前端會提示）。
- 設定檔 data/google_sheets/config.json 只存 webhook_url（chmod 600、不進 git）。
- 舊版曾把密碼寫進 config.json：載入時自動遷移到記憶體並改寫檔案移除。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

import httpx
from loguru import logger

from app.core.config.settings import settings

MIN_PASSWORD_LEN = 10

# Apps Script Web App 部署網址（含 Workspace 網域的 /a/macros/<domain>/ 形式）
_WEBHOOK_URL_RE = re.compile(
    r"^https://script\.google\.com/(?:a/macros/[^/]+/|macros/)s/[\w-]+/exec$"
)
_SPREADSHEET_ID_RE = re.compile(r"/spreadsheets/d/([a-zA-Z0-9\-_]+)")

_SCRIPT_TEMPLATE = """\
const SECRET = '{{SECRET}}';  // 你的匯出密碼，勿外流

function doPost(e) {
  let out;
  try {
    const req = JSON.parse(e.postData.contents);
    if (req.secret !== SECRET) throw new Error('unauthorized');
    if (req.action === 'ping') {
      out = { ok: true, pong: true };
    } else if (req.action === 'export') {
      const ss = SpreadsheetApp.openByUrl(req.spreadsheet_url);
      let title = req.sheet_title, n = 2;
      while (ss.getSheetByName(title)) title = req.sheet_title + ' (' + n++ + ')';
      const sheet = ss.insertSheet(title);
      const values = [req.headers, ...req.rows.map(r => r.map(v => v === null ? '' : v))];
      sheet.getRange(1, 1, values.length, req.headers.length).setValues(values);
      // 字體顏色（與系統畫面一致）；舊版腳本沒有這段也能匯出、只是沒顏色
      if (req.colors && req.colors.length) {
        sheet.getRange(2, 1, req.colors.length, req.headers.length).setFontColors(req.colors);
      }
      out = { ok: true, sheet_title: title, gid: sheet.getSheetId(), updated_cells: values.length * req.headers.length };
    } else throw new Error('unknown action');
  } catch (err) {
    out = { ok: false, error: String(err && err.message || err) };
  }
  return ContentService.createTextOutput(JSON.stringify(out)).setMimeType(ContentService.MimeType.JSON);
}
"""


class GoogleSheetsError(Exception):
    """code: not_configured / password_required / bad_password / bad_url /
    webhook_unreachable / unauthorized / remote_error"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class GoogleSheetsWebhook:
    """管理 webhook 設定（磁碟）＋ 密碼（僅記憶體）＋ 轉發匯出請求"""

    def __init__(self):
        self._password: Optional[str] = None  # 僅存記憶體，重啟即清空

    @property
    def _config_path(self) -> Path:
        return Path(settings.data_dir) / "google_sheets" / "config.json"

    def _load(self) -> dict:
        try:
            cfg = json.loads(self._config_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except Exception as e:
            logger.warning(f"[GoogleSheets] config.json 讀取失敗：{e}")
            return {}
        # 舊版把密碼落地：遷移到記憶體並改寫檔案移除
        if "password" in cfg:
            self._password = self._password or cfg.pop("password")
            cfg.pop("password", None)
            self._save(cfg)
            logger.info("[GoogleSheets] 已把舊版落地密碼遷移為僅存記憶體")
        return cfg

    def _save(self, cfg: dict) -> None:
        path = self._config_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        os.chmod(path, 0o600)

    def is_configured(self) -> bool:
        """是否已有部署網址（密碼是否在記憶體另看 has_password）"""
        return bool(self._load().get("webhook_url"))

    def has_password(self) -> bool:
        return self._password is not None

    def set_password(self, password: str) -> None:
        """精靈第一步：設密碼（記憶體）。換密碼＝腳本要重新部署，舊 webhook 作廢。"""
        if len(password) < MIN_PASSWORD_LEN:
            raise GoogleSheetsError("bad_password", f"密碼至少需 {MIN_PASSWORD_LEN} 個字元")
        self._password = password
        cfg = self._load()
        if cfg.pop("webhook_url", None):
            self._save(cfg)

    async def unlock(self, password: str) -> None:
        """後端重啟後重新輸入密碼：先 ping 驗證與腳本內密碼相符才收進記憶體"""
        cfg = self._load()
        webhook_url = cfg.get("webhook_url")
        if not webhook_url:
            raise GoogleSheetsError("not_configured", "尚未完成 Google Sheet 匯出設定")
        await self._call(webhook_url, {"secret": password, "action": "ping"})
        self._password = password

    def build_script_code(self) -> str:
        if not self._password:
            raise GoogleSheetsError("password_required", "尚未設定匯出密碼")
        escaped = self._password.replace("\\", "\\\\").replace("'", "\\'")
        return _SCRIPT_TEMPLATE.replace("{{SECRET}}", escaped)

    def reset(self) -> None:
        self._password = None
        self._config_path.unlink(missing_ok=True)

    async def _call(self, webhook_url: str, payload: dict) -> dict:
        """POST 到 Apps Script。回應是 302 → script.googleusercontent.com，需跟隨。"""
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
                resp = await client.post(webhook_url, json=payload)
        except httpx.HTTPError as e:
            raise GoogleSheetsError("webhook_unreachable", f"無法連線到 Apps Script：{e}")
        try:
            data = resp.json()
        except Exception:
            # 拿到 HTML（多半是 Google 登入頁）＝ 部署存取權沒設「任何人」
            raise GoogleSheetsError(
                "webhook_unreachable",
                "Apps Script 回應無法解析，請確認部署時「存取權」設為「任何人」且網址以 /exec 結尾",
            )
        if not data.get("ok"):
            err = str(data.get("error", "未知錯誤"))
            if "unauthorized" in err:
                raise GoogleSheetsError(
                    "unauthorized", "密碼不符：與腳本內的密碼不一致"
                )
            raise GoogleSheetsError("remote_error", f"Apps Script 執行失敗：{err}")
        return data

    async def save_webhook_url(self, webhook_url: str) -> None:
        webhook_url = webhook_url.strip()
        if not _WEBHOOK_URL_RE.match(webhook_url):
            raise GoogleSheetsError(
                "bad_url",
                "網址格式不正確：應為 https://script.google.com/macros/s/…/exec（結尾是 /exec）",
            )
        if not self._password:
            raise GoogleSheetsError("password_required", "尚未設定匯出密碼，請先完成精靈第一步")
        await self._call(webhook_url, {"secret": self._password, "action": "ping"})
        cfg = self._load()
        cfg["webhook_url"] = webhook_url
        self._save(cfg)
        logger.info("[GoogleSheets] webhook 驗證成功並已儲存")

    async def export_table(
        self,
        spreadsheet_url: str,
        sheet_title: str,
        headers: list[str],
        rows: list[list[Any]],
        colors: Optional[list[list[Optional[str]]]] = None,
    ) -> dict:
        cfg = self._load()
        webhook_url = cfg.get("webhook_url")
        if not webhook_url:
            raise GoogleSheetsError("not_configured", "尚未完成 Google Sheet 匯出設定")
        if not self._password:
            raise GoogleSheetsError(
                "password_required", "後端重啟後需重新輸入匯出密碼"
            )
        m = _SPREADSHEET_ID_RE.search(spreadsheet_url)
        if not m:
            raise GoogleSheetsError("bad_url", "這不是 Google 試算表網址（找不到 /spreadsheets/d/…）")
        data = await self._call(
            webhook_url,
            {
                "secret": self._password,
                "action": "export",
                "spreadsheet_url": spreadsheet_url,
                "sheet_title": sheet_title,
                "headers": headers,
                "rows": rows,
                "colors": colors or [],
            },
        )
        sheet_url = f"https://docs.google.com/spreadsheets/d/{m.group(1)}/edit#gid={data.get('gid', 0)}"
        return {
            "sheet_url": sheet_url,
            "sheet_title": data.get("sheet_title", sheet_title),
            "updated_cells": data.get("updated_cells", 0),
        }


google_sheets_webhook = GoogleSheetsWebhook()
