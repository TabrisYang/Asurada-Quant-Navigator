"""邊界契約測試 — chart_state 欄位 vs 治理文件一致性（v154）

chart_state 是全系統最大的共享契約（後端注入、前端讀、prompt 注入、覆核比對）。
check_repo_health 管「數量」，本測試管「名單」：chat 路由家族每個注入的 top-level key
都必須出現在 CHART_STATE_SCHEMA.md — 新欄位沒補文件就紅燈。

v157：chat.py 拆分後注入邏輯搬到 chat_context.py，改掃 routes/chat*.py 全家族，
避免拆檔後這道契約默默失效（掃單一檔案會 0 命中而假性全綠）。
"""

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_ROUTES = _BACKEND / "app" / "api" / "routes"
_CHAT_FILES = sorted(_ROUTES.glob("chat*.py"))
_SCHEMA_MD = _BACKEND / "docs" / "CHART_STATE_SCHEMA.md"

_INJECT_RE = re.compile(r'chart_state\["(\w+)"\]\s*=')
_DOC_FIELD_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")


class TestChartStateSchemaConsistency:
    def test_injected_keys_documented(self):
        assert _CHAT_FILES, "找不到 routes/chat*.py（路徑改了？）"
        injected: set[str] = set()
        for path in _CHAT_FILES:
            injected |= set(_INJECT_RE.findall(path.read_text(encoding="utf-8")))
        assert injected, "chat 路由家族應有 chart_state 注入（regex 失效？）"

        documented = set(_DOC_FIELD_RE.findall(_SCHEMA_MD.read_text(encoding="utf-8")))
        missing = sorted(injected - documented)
        assert not missing, (
            f"以下 chart_state 欄位在 chat 路由家族注入但未寫進 docs/CHART_STATE_SCHEMA.md：{missing}\n"
            "→ 請依 CLAUDE.md 紀律補文件（用途、是否進 round2），或移除欄位。"
        )
