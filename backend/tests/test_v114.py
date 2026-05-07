"""v114 regression tests：防止 client_disconnect 時 partial 內容被丟掉。

歷史 bug（v110~v113 隱藏）：
chat.py 的 `except asyncio.CancelledError:` 直接 `return`，沒把已生成的
final_text 寫進 DB。結果：使用者「全部分析」跑到一半被 ASGI cancel scope 殺
（瀏覽器/系統/網路層外部因素，不可控），重整頁面看不到 partial 內容，必須
打「請繼續」重跑 8 分鐘 + 大量 token 才能看到完整報告。

修法：在 except CancelledError 分支用 sync chat_history.save_message 把
partial 內容寫進 DB，並附加「報告生成中斷」標記。
"""

import pathlib
import re


CHAT_PY = (
    pathlib.Path(__file__).resolve().parent.parent
    / "app" / "api" / "routes" / "chat.py"
)


def _extract_cancelled_handler() -> str:
    """提取 stream_gen 內 `except asyncio.CancelledError:` 整段（到下一個同縮排的 except / finally 為止）。"""
    src = CHAT_PY.read_text(encoding="utf-8")
    lines = src.split("\n")
    start_idx = None
    base_indent = None
    for i, line in enumerate(lines):
        if "except asyncio.CancelledError:" not in line:
            continue
        # 認 stream_gen 內的那個（下一行含 "Streaming chat 被取消"）
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        if "Streaming chat 被取消" in next_line:
            start_idx = i
            base_indent = len(line) - len(line.lstrip())
            break
    assert start_idx is not None, "找不到 stream_gen 內的 except asyncio.CancelledError"

    end_idx = start_idx + 1
    while end_idx < len(lines):
        line = lines[end_idx]
        if not line.strip():
            end_idx += 1
            continue
        cur_indent = len(line) - len(line.lstrip())
        if cur_indent <= base_indent:
            break
        end_idx += 1
    return "\n".join(lines[start_idx:end_idx])


def test_cancel_handler_calls_save_message():
    """except CancelledError 區塊必須呼叫 chat_history.save_message 寫 partial 內容到 DB。"""
    handler = _extract_cancelled_handler()
    assert "chat_history.save_message" in handler, (
        "except asyncio.CancelledError 區塊必須呼叫 chat_history.save_message，"
        "否則 client 斷線時 partial 內容會丟掉（重整頁面看不到，使用者必須重跑）"
    )


def test_cancel_handler_appends_interrupted_marker():
    """partial 內容必須附加『報告生成中斷』標記，讓使用者區分『LLM 自然結束』vs『斷線截斷』。"""
    handler = _extract_cancelled_handler()
    assert "報告生成中斷" in handler, (
        "partial 內容尾端必須附加 '報告生成中斷' 標記"
    )


def test_cancel_handler_skips_save_when_final_text_empty():
    """final_text 是空 string 時不該呼叫 save_message（避免存空訊息汙染對話）。"""
    handler = _extract_cancelled_handler()
    assert ".strip()" in handler, (
        "必須先檢查 final_text.strip() 才呼叫 save_message"
    )


def test_cancel_handler_strips_internal_markers():
    """partial 內容寫 DB 前必須剝除 KEY_INSIGHTS / PREDICTIONS / SYSTEM_DISTILL 內部 markup，
    避免 DB 訊息包含使用者不該看到的內部標記。
    """
    handler = _extract_cancelled_handler()
    assert "strip_key_insights" in handler, "必須剝除 KEY_INSIGHTS 標記"
    assert "strip_predictions" in handler, "必須剝除 PREDICTIONS 標記"
    assert "strip_system_distill" in handler, "必須剝除 SYSTEM_DISTILL 標記"


def test_cancel_handler_save_message_is_sync_not_awaited():
    """save_message 必須是 sync 呼叫（沒有 await）— 當前 task 已被 cancel，
    await 會立刻 raise CancelledError，partial-save 永遠跑不完。
    """
    handler = _extract_cancelled_handler()
    # 找 save_message 呼叫前 30 字內不能有 await
    m = re.search(r"(.{0,30})chat_history\.save_message", handler)
    assert m, "找不到 chat_history.save_message 呼叫"
    prefix = m.group(1)
    assert "await" not in prefix, (
        f"save_message 不該用 await（前 30 字含: {prefix!r}），"
        "當前 task 已被 cancel，await 會立刻 raise CancelledError"
    )
