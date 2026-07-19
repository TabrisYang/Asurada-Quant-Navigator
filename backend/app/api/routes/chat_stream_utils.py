"""chat 路由的串流基礎設施（v157 拆分：從 chat.py 機械搬移，邏輯零改動）。

職責：SSE 事件格式化、心跳保活（防 LLM 思考期間連線被判 idle 而斷）、
速率限制、以及把重 CPU 的 function call 丟到獨立 thread 跑。

⚠️ _stream_with_heartbeat 的 asyncio.shield 設計是 v110 修 stream 中斷的關鍵，
改動前務必讀該函式的 docstring。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict


from app.core.llm.executor import execute_function_calls
from app.core.llm.function_result_formatter import _json_safe_default

_rate_limit_store: dict[str, list[float]] = defaultdict(list)


# ─── 簡易速率限制（每 IP 每分鐘最多 30 次 chat 請求）───
_RATE_LIMIT_WINDOW = 60  # 秒
_RATE_LIMIT_MAX = 30     # 每窗口最大請求數
_rate_limit_store: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(client_ip: str) -> bool:
    """檢查是否超過速率限制，返回 True 表示允許"""
    now = time.time()
    timestamps = _rate_limit_store[client_ip]
    # 清除過期記錄
    _rate_limit_store[client_ip] = [t for t in timestamps if now - t < _RATE_LIMIT_WINDOW]
    if len(_rate_limit_store[client_ip]) >= _RATE_LIMIT_MAX:
        return False
    _rate_limit_store[client_ip].append(now)
    return True


def _sse_event(event_type: str, data: dict) -> str:
    """格式化 SSE event"""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False, default=_json_safe_default)}\n\n"


# v110：SSE 心跳機制 — 防止 LLM 思考期間（無 token）SSE 連線被 OS / browser / proxy 任何層判定 idle 而斷
# 業界標準：long-lived SSE 連線每 N 秒送 heartbeat 維持 active
# 演進：15s（首版）→ 5s（實測仍會斷）→ 2s（v110 final）
# 實測 5s 仍不夠：v108 後 system prompt 變大（含 bilateral_plan / indicators_snapshot / 各規則段）
# 加上對話歷史累積 → LLM TTFT 達 4-5 秒，5s timeout「剛好」太晚 → client 在第 5 秒已斷
# 2s 給最積極保護：每 2 秒至少一個心跳，遠低於任何 OS / browser / fetch 內部 idle 偵測
_HEARTBEAT_INTERVAL = 2.0  # 秒
_HEARTBEAT_SENTINEL = object()  # 用 sentinel 物件標記心跳事件


async def _stream_with_heartbeat(stream_iter, interval: float = _HEARTBEAT_INTERVAL):
    """包裝 LLM adapter 的 async iterator，無事件達 interval 秒 → yield 心跳 sentinel。

    使用方式：
        async for evt in _stream_with_heartbeat(adapter.chat_stream_events(...)):
            if evt is _HEARTBEAT_SENTINEL:
                # 心跳：呼叫端應 yield SSE heartbeat event 給 client
                continue
            # 處理真實 evt（既有邏輯）

    ★★★ 關鍵設計（v110 fix）：用 asyncio.shield 保護 pending task ★★★
    舊版直接 `await asyncio.wait_for(_iter.__anext__(), timeout=interval)` 在 timeout 時
    會 cancel inner coroutine（即 LLM adapter 的 __anext__），導致 adapter generator 內部
    state（subprocess / readline 等）被破壞。下次 __anext__ 立刻 StopAsyncIteration → 主
    流程看到 streaming 結束 → _r2_text_buf 為空 → 走「未能產生文字報告」 fallback。

    正確做法：把 pending task 存起來，每輪 wait_for 用 asyncio.shield 包它。timeout 取消
    的是 shield wrapper 不是 pending 本身，pending 繼續活著等 LLM 下一個 event。
    """
    _iter = stream_iter.__aiter__() if hasattr(stream_iter, "__aiter__") else stream_iter
    pending: asyncio.Task | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(_iter.__anext__())
            try:
                evt = await asyncio.wait_for(asyncio.shield(pending), timeout=interval)
                # 拿到 event，pending 已 done，下輪重新建立
                pending = None
                yield evt
            except asyncio.TimeoutError:
                # pending 仍活著（被 shield 保護），下輪迴圈繼續 await 同一個 pending
                yield _HEARTBEAT_SENTINEL
            except StopAsyncIteration:
                pending = None
                return
    finally:
        # 確保 pending 在 generator 結束時被清理（避免 task leak）
        if pending is not None and not pending.done():
            pending.cancel()
            try:
                await pending
            except (asyncio.CancelledError, StopAsyncIteration, Exception):
                pass


def _execute_function_calls_in_thread(*args, **kwargs):
    """在 worker thread 內開新 event loop 跑 execute_function_calls。

    ★ 為什麼需要這個：
    execute_function_calls 內部會呼叫重 CPU 的 sync 操作（ML 訓練、回測、
    Monte Carlo、Walk Forward 等），這些是 NumPy/pandas/sklearn 的 sync
    呼叫，會阻塞 asyncio event loop。

    若直接 await execute_function_calls(...)，主 event loop 在重操作期間
    無法處理任何其他協程 → 心跳 yield SSE event 排不上時程 → 前端誤判
    超時斷線。

    解法：用 asyncio.to_thread + 新 event loop，讓 execute 在獨立 thread
    執行，主 event loop 完全不被佔用。

    用法：
        _fc_task = asyncio.create_task(asyncio.to_thread(
            _execute_function_calls_in_thread, fcs, chart_state=..., progress=...
        ))
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(execute_function_calls(*args, **kwargs))
    finally:
        loop.close()
