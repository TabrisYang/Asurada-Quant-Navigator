"""v117 regression tests：防 backend event loop 卡死的兩個關鍵修復。

歷史 bug（v100~v116 隱藏）：

(1) chat.py finally 內 `await t` cleanup 沒 timeout
    若 task 內部是 asyncio.to_thread(...) 包的 CPU-bound sync 操作，
    cancel 訊號等到 thread 跑完才生效。期間整個 event loop 卡住，所有
    endpoint（/health / /api/chat/stream / /api/chart/data）都 timeout。
    使用者實測：跑「全部分析」一次後 backend 整個卡死，curl 15 秒無回應。

(2) _post_process_chat_message 是 async def 但內部全是 sync SQLite ops
    （chat_history / analysis_cache / semantic_cache / fragment_store /
    prediction_tracker 都是 sync），→ 仍跑在主 event loop 上。當 SQLite
    fsync / embedding encode 慢時，阻塞所有其他 endpoints。

修法：
- (1) `await t` → `await asyncio.wait_for(t, timeout=2.0)` + 加 TimeoutError catch
- (2) `async def` → `def`；caller 改 `asyncio.create_task(asyncio.to_thread(...))`
"""

import inspect
import pathlib
import re


CHAT_PY = (
    pathlib.Path(__file__).resolve().parent.parent
    / "app" / "api" / "routes" / "chat.py"
)


def _chat_src() -> str:
    return CHAT_PY.read_text(encoding="utf-8")


def test_active_tasks_cleanup_uses_wait_for_timeout():
    """finally cleanup 必須用 asyncio.wait_for(t, timeout=...) 包，不能裸 await t。

    裸 await t 在 task 內部是 to_thread CPU-bound 死循環時會永遠等，
    卡住整個 event loop（v117 之前已實測 reproduce）。
    """
    src = _chat_src()
    # 找 _active_tasks cleanup block
    m = re.search(
        r"for t in _active_tasks:.*?\n\s+if not t\.done\(\):.*?\n\s+t\.cancel\(\).*?\n\s+try:.*?\n\s+(.+?)\n\s+except",
        src, re.DOTALL,
    )
    assert m, "找不到 _active_tasks cleanup block"
    cleanup_call = m.group(1).strip()
    assert "wait_for" in cleanup_call, (
        f"_active_tasks cleanup 必須用 asyncio.wait_for(t, timeout=N) 包，"
        f"避免 task 內部 thread 死循環時卡住 event loop。"
        f"目前 cleanup 行：{cleanup_call!r}"
    )
    assert "timeout=" in cleanup_call, (
        f"wait_for 必須有 timeout 參數，目前：{cleanup_call!r}"
    )


def test_active_tasks_cleanup_catches_timeout_error():
    """除了 CancelledError，TimeoutError 也必須被 except 吃掉，否則 wait_for timeout
    會把 exception 推到 generator 外。"""
    src = _chat_src()
    m = re.search(
        r"for t in _active_tasks:.*?\n.*?wait_for.*?\n\s+except\s*\(([^)]+)\)",
        src, re.DOTALL,
    )
    assert m, "找不到 _active_tasks cleanup 的 except clause"
    except_clause = m.group(1)
    assert "TimeoutError" in except_clause, (
        f"_active_tasks cleanup 的 except 必須含 asyncio.TimeoutError，"
        f"否則 wait_for timeout 會 propagate。目前：except ({except_clause})"
    )


def test_post_process_chat_message_is_sync():
    """_post_process_chat_message 必須是 sync function（不是 async def）。

    內部全是 sync SQLite/embedding 操作，async def 但無 await 會仍在主 event
    loop 跑，阻塞其他 endpoints。改 sync + caller 用 to_thread 丟 thread pool。
    """
    from app.api.routes.chat import _post_process_chat_message
    assert not inspect.iscoroutinefunction(_post_process_chat_message), (
        "_post_process_chat_message 必須是 sync function（v117）。"
        "若改回 async def，sync DB ops 會阻塞 event loop 造成 backend hang。"
    )


def test_post_process_chat_message_called_via_to_thread():
    """caller 必須用 asyncio.to_thread 包 _post_process_chat_message，
    確保跑在 thread pool 不阻塞主 event loop。"""
    src = _chat_src()
    # 找 _post_process_chat_message 的 caller
    # 期望模式：asyncio.create_task(asyncio.to_thread(_post_process_chat_message, ...))
    m = re.search(
        r"asyncio\.create_task\(\s*asyncio\.to_thread\(\s*_post_process_chat_message",
        src,
    )
    assert m, (
        "caller 必須用 asyncio.create_task(asyncio.to_thread(_post_process_chat_message, ...))，"
        "v117 修復了 _post_process 阻塞 event loop 的 bug，"
        "若 caller 變回 asyncio.create_task(_post_process_chat_message(...)) 會 regression。"
    )


def test_post_process_chat_message_signature_unchanged():
    """v117 改動只是移除 async + 加 to_thread，signature 仍是 keyword-only args。"""
    from app.api.routes.chat import _post_process_chat_message
    sig = inspect.signature(_post_process_chat_message)
    params = sig.parameters
    expected_keys = {
        "final_text", "request_message", "chart_state",
        "chart_symbol_for_save", "conversation_id", "total_usage",
    }
    assert set(params.keys()) == expected_keys, (
        f"_post_process_chat_message signature 變了：{set(params.keys())} vs {expected_keys}"
    )
    # 全部 keyword-only
    for name, param in params.items():
        assert param.kind == inspect.Parameter.KEYWORD_ONLY, (
            f"{name} 必須是 keyword-only（原 signature 用 `*,` 強制）"
        )
