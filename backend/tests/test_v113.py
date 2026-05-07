"""v113 regression tests：防止 LLM CLI subprocess `_line_timeout` 退回 60 秒。

歷史 bug（v107~v112 隱藏）：
adapter.py 三處 `_line_timeout = 60` 對「全部分析」太短 — 全部分析 Round 2
LLM 從段落 N 切到段落 N+1 整合 compare_strategies 結果時，思考間隔輕易超過
60 秒 → wait_for(readline) 觸發 TimeoutError → break → terminate subprocess
→ 報告中斷在「## 4. 🧪 多策」幾個字後。

修法：60 → 300（5 分鐘），給 LLM 充足思考時間，且仍小於 frontend
STREAM_TIMEOUT_MS = 600_000 ms（10 分鐘）。
"""

import pathlib
import re


def _adapter_src() -> str:
    p = pathlib.Path(__file__).resolve().parent.parent / "app" / "core" / "llm" / "adapter.py"
    return p.read_text(encoding="utf-8")


def _frontend_api_src() -> str:
    # backend/tests/test_v113.py → 阿斯拉量化系統V2/frontend/src/services/api.ts
    p = (
        pathlib.Path(__file__).resolve().parent.parent.parent
        / "frontend" / "src" / "services" / "api.ts"
    )
    return p.read_text(encoding="utf-8")


def test_line_timeout_at_least_300():
    """確認 adapter.py 中所有 `_line_timeout = N` 賦值 N >= 300，不再是 60。"""
    src = _adapter_src()
    matches = re.findall(r"_line_timeout\s*=\s*(\d+)", src)
    assert matches, "adapter.py 找不到 _line_timeout 賦值（測試 grep 失效或變數已被刪除）"
    for val_str in matches:
        val = int(val_str)
        assert val >= 300, (
            f"adapter.py 出現 _line_timeout = {val}（< 300）"
            f"——歷史 bug 復活，會導致全部分析報告中斷。修回 >= 300。"
        )


def test_line_timeout_smaller_than_frontend_stream_timeout():
    """backend timeout 必須短於 frontend STREAM_TIMEOUT_MS，避免 frontend 先超時顯示誤導訊息。"""
    backend_src = _adapter_src()
    frontend_src = _frontend_api_src()

    backend_vals = [int(v) for v in re.findall(r"_line_timeout\s*=\s*(\d+)", backend_src)]
    assert backend_vals, "找不到 backend _line_timeout 設定"
    backend_max_seconds = max(backend_vals)

    fe_match = re.search(r"STREAM_TIMEOUT_MS\s*=\s*([\d_]+)", frontend_src)
    assert fe_match, "frontend api.ts 找不到 STREAM_TIMEOUT_MS"
    frontend_ms = int(fe_match.group(1).replace("_", ""))
    frontend_seconds = frontend_ms / 1000

    assert backend_max_seconds < frontend_seconds, (
        f"backend _line_timeout 最大值 {backend_max_seconds}s "
        f"必須小於 frontend STREAM_TIMEOUT_MS {frontend_seconds}s，"
        f"否則 frontend 會先觸發顯示「分析連線無回應超過 N 秒」誤導訊息"
    )
