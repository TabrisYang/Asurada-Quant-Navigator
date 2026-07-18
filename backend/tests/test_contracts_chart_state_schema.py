"""邊界契約測試 — chart_state 欄位 vs 治理文件一致性（v154）

chart_state 是全系統最大的共享契約（後端注入、前端讀、prompt 注入、覆核比對）。
check_repo_health 管「數量」，本測試管「名單」：chat.py 每個注入的 top-level key
都必須出現在 CHART_STATE_SCHEMA.md — 新欄位沒補文件就紅燈。
"""

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_CHAT_PY = _BACKEND / "app" / "api" / "routes" / "chat.py"
_SCHEMA_MD = _BACKEND / "docs" / "CHART_STATE_SCHEMA.md"

_INJECT_RE = re.compile(r'chart_state\["(\w+)"\]\s*=')
_DOC_FIELD_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")


class TestChartStateSchemaConsistency:
    def test_injected_keys_documented(self):
        injected = set(_INJECT_RE.findall(_CHAT_PY.read_text(encoding="utf-8")))
        assert injected, "chat.py 應有 chart_state 注入（regex 失效？）"

        documented = set(_DOC_FIELD_RE.findall(_SCHEMA_MD.read_text(encoding="utf-8")))
        missing = sorted(injected - documented)
        assert not missing, (
            f"以下 chart_state 欄位在 chat.py 注入但未寫進 docs/CHART_STATE_SCHEMA.md：{missing}\n"
            "→ 請依 CLAUDE.md 紀律補文件（用途、是否進 round2），或移除欄位。"
        )
