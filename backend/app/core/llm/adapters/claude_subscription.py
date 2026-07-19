"""ClaudeSubscriptionAdapter — v154 由 adapter.py 純搬家拆出（原 L1252-1926），邏輯零改動。"""

import asyncio
import json
import re
from typing import Any, Optional

from loguru import logger

from app.core.llm.function_defs import FUNCTION_DEFINITIONS

from app.core.llm.adapters.base import (
    BaseLLMAdapter,
    LLMResponse,
    StreamEvent,
    TokenUsage,
    _looks_rate_limited,
    _minimal_r2_chart_state,
)

class ClaudeSubscriptionAdapter(BaseLLMAdapter):
    """透過 claude CLI 使用訂閱額度的適配器。

    oauth_token：
      - None（站長本機）→ 不帶特殊環境變數，CLI 走本機登入憑證（行為與改動前完全相同）。
      - 有值（雲端/其他使用者）→ 以 CLAUDE_CODE_OAUTH_TOKEN 逐請求注入，並用獨立
        CLAUDE_CONFIG_DIR 隔離，使每位使用者各自計入自己的訂閱額度、互不污染。
        token 只存在記憶體+加密 store、只經 env 傳給子進程，永不寫 log / 不落地明文。
    """

    _PROVIDER_NAME = "claude_subscription"  # 子類（codex_subscription）覆寫

    def __init__(self, model: str = "sonnet", oauth_token: Optional[str] = None):
        self.model = model
        self.oauth_token = oauth_token or None

    def _build_subprocess_env(self) -> tuple[Optional[dict], Optional[str]]:
        """為 claude CLI 子進程建環境變數。

        有 oauth_token → (env, 臨時 config 目錄)：用 CLAUDE_CODE_OAUTH_TOKEN 注入該使用者
        的訂閱憑證，並以獨立 CLAUDE_CONFIG_DIR 隔離，避免多使用者互相污染。
        無 oauth_token → (None, None)：env=None 時子進程沿用父進程環境（本機登入），
        行為與改動前完全一致。回傳的 cfg_dir 由呼叫端在 finally 清除。
        """
        if not self.oauth_token:
            return None, None
        import os
        import tempfile
        cfg_dir = tempfile.mkdtemp(prefix="claude_cfg_")
        env = {
            **os.environ,
            "CLAUDE_CODE_OAUTH_TOKEN": self.oauth_token,
            "CLAUDE_CONFIG_DIR": cfg_dir,
        }
        return env, cfg_dir

    @staticmethod
    def _cleanup_cfg_dir(cfg_dir: Optional[str]) -> None:
        """清除 per-user 臨時 config 目錄（含其中可能的憑證快取）。"""
        if not cfg_dir:
            return
        import shutil
        shutil.rmtree(cfg_dir, ignore_errors=True)

    # ── 共用 ──

    # Round 2 只需要這 3 個繪圖/指標函式（chat.py L1133-1136 已過濾）
    _R2_ALLOWED_FUNCTIONS = {"annotate_chart", "draw_pattern", "manage_indicator"}

    @staticmethod
    def _format_tools_for_prompt(r2_mode: bool = False) -> str:
        """將 FUNCTION_DEFINITIONS 轉成文字，注入 system prompt。
        r2_mode=True 時只包含繪圖相關函式，大幅減少 token。
        """
        funcs = FUNCTION_DEFINITIONS
        if r2_mode:
            funcs = [
                fd for fd in FUNCTION_DEFINITIONS
                if (fd.get("function", fd)).get("name", "") in ClaudeSubscriptionAdapter._R2_ALLOWED_FUNCTIONS
            ]

        lines = [
            "\n\n【可用工具 — Function Call】",
            "當你需要執行操作時，用以下 XML 格式呼叫工具：",
            "<tool_call>{\"name\": \"函式名\", \"arguments\": {參數}}</tool_call>",
            "",
            "可用函式列表：",
        ]
        for i, fd in enumerate(funcs, 1):
            func = fd.get("function", fd)
            name = func.get("name", "")
            desc = func.get("description", "")
            lines.append(f"\n{i}. {name} — {desc}")
            params = func.get("parameters", {})
            props = params.get("properties", {})
            required = set(params.get("required", []))
            if props:
                lines.append("   參數:")
                for pname, pinfo in props.items():
                    ptype = pinfo.get("type", "string")
                    pdesc = pinfo.get("description", "")
                    enum_vals = pinfo.get("enum")
                    req_mark = " (必填)" if pname in required else ""
                    enum_str = f", 可選值: {enum_vals}" if enum_vals else ""
                    lines.append(f"   - {pname} ({ptype}{req_mark}): {pdesc}{enum_str}")
        if r2_mode:
            lines.append("\n你只能使用以上列出的工具來標記圖表和管理指標。")
        else:
            lines.append("\n重要：你必須主動呼叫這些工具來獲取真實數據，不要憑空猜測。")
        return "\n".join(lines)

    def _build_system_message(
        self,
        chart_state: Optional[dict] = None,
        system_prompt: Optional[str] = None,
        include_tools: bool = True,
        r2_mode: bool = False,
    ) -> str:
        """覆寫父類：附加工具定義文字。r2_mode=True 時只附加繪圖函式 + 精簡 chart_state。"""
        # R1: r2_mode=True 時用精簡 chart_state（移除 signal_history 等已在 round1 用過的欄位）
        effective_chart_state = (
            _minimal_r2_chart_state(chart_state) if r2_mode and chart_state else chart_state
        )
        prompt = super()._build_system_message(effective_chart_state, system_prompt)
        if include_tools:
            prompt += self._format_tools_for_prompt(r2_mode=r2_mode)
        else:
            # Round 3：工具已執行完畢，指示 LLM 直接生成文字分析
            prompt += (
                "\n\n【重要提示】所有工具呼叫已在前面的步驟中執行完畢，數據結果已包含在對話歷史中。"
                "你現在的任務是：根據已獲得的數據和計算結果，用文字詳細回答使用者的問題。"
                "不要再嘗試呼叫任何工具或函式，直接提供分析結論、關鍵數據解讀和交易建議。"
            )
        # P2: log prompt size 便於追蹤 chart_state 是否再次膨脹
        sys_kb = len(prompt.encode("utf-8")) / 1024
        cs_kb = (
            len(json.dumps(effective_chart_state, ensure_ascii=False).encode("utf-8")) / 1024
            if effective_chart_state else 0
        )
        if sys_kb >= 50:  # >50KB 才印，避免無關 log 噪音
            logger.info(
                f"[prompt_size] sys={sys_kb:.1f}KB chart_state={cs_kb:.1f}KB r2_mode={r2_mode}"
            )
        return prompt

    def _build_user_prompt(self, messages: list[dict]) -> str:
        """組裝對話歷史 + 當前訊息為單一 prompt"""
        prompt_parts: list[str] = []
        last_user_msg = ""
        for msg in messages:
            role_label = "使用者" if msg["role"] == "user" else "助手"
            content = msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
            if msg["role"] == "user":
                last_user_msg = content
            prompt_parts.append(f"{role_label}: {content}")
        return "\n\n".join(prompt_parts) if len(messages) > 1 else last_user_msg

    @staticmethod
    def _parse_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
        """解析 <tool_call> XML 並回傳 (clean_text, function_calls)"""
        tool_call_re = re.compile(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>")
        function_calls: list[dict[str, Any]] = []
        for match in tool_call_re.finditer(text):
            try:
                obj = json.loads(match.group(1))
                fc = {"name": obj.get("name", ""), "arguments": obj.get("arguments", {})}
                if fc["name"]:
                    function_calls.append(fc)
            except json.JSONDecodeError:
                pass
        clean_text = tool_call_re.sub("", text).strip()
        return clean_text, function_calls

    @staticmethod
    def _build_usage(usage_data: dict, model: str) -> TokenUsage:
        inp = usage_data.get("input_tokens", 0) or 0
        out = usage_data.get("output_tokens", 0) or 0
        cache_read = usage_data.get("cache_read_input_tokens", 0) or 0
        cache_create = usage_data.get("cache_creation_input_tokens", 0) or 0
        return TokenUsage(
            prompt_tokens=inp + cache_read + cache_create,
            completion_tokens=out,
            total_tokens=inp + out + cache_read + cache_create,
            model=model,
            provider="claude_subscription",  # staticmethod（claude stream-json 專用，codex 子類不經此路徑）
        )

    async def chat(
        self,
        messages: list[dict],
        chart_state: Optional[dict] = None,
        force_text: bool = False,
        system_prompt: Optional[str] = None,
        chart_screenshot: Optional[str] = None,
        r2_mode: bool = False,
    ) -> LLMResponse:
        """使用 stream-json 模式逐行讀取。

        v149：首個輸出快速失敗（120s）+ 限流偵測退避重試。
        舊版對「首個輸出」也套用 1200s timeout → LLM 限流/卡住時要硬等 20 分鐘。
        改成：首 token 120s 無回應即視為卡住/限流，並行 drain stderr 拿原因，
        指數退避重試最多 2 次，最終回傳清楚錯誤訊息給用戶（而非靜默 20 分）。
        """
        import os
        import tempfile

        sys_prompt_file = None
        try:
            sys_prompt = self._build_system_message(
                chart_state, system_prompt, include_tools=not force_text, r2_mode=r2_mode,
            )
            user_prompt = self._build_user_prompt(messages)

            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write(sys_prompt)
                sys_prompt_file = f.name

            _BACKOFFS = [5, 15]          # 限流退避秒數（最多重試 len 次）
            _MAX_ATTEMPTS = len(_BACKOFFS) + 1
            for attempt in range(_MAX_ATTEMPTS):
                full_text, usage, result_data, stderr_text, no_output = (
                    await self._run_cli_attempt(user_prompt, sys_prompt_file)
                )

                if full_text:
                    if not usage:
                        usage = TokenUsage(
                            prompt_tokens=0, completion_tokens=0, total_tokens=0,
                            model=self.model, provider=self._PROVIDER_NAME,
                        )
                    clean_text, function_calls = self._parse_tool_calls(full_text)
                    return LLMResponse(
                        message=clean_text,
                        function_calls=function_calls,
                        raw_response=result_data,
                        usage=usage,
                    )

                # 空回應：判斷是否限流/卡住 → 退避重試
                rate_limited = no_output or _looks_rate_limited(stderr_text)
                if rate_limited and attempt < _MAX_ATTEMPTS - 1:
                    wait_s = _BACKOFFS[attempt]
                    logger.warning(
                        f"Claude CLI 無回應/疑似限流（第 {attempt + 1} 次），{wait_s}s 後退避重試。"
                        f"stderr={stderr_text.strip()[:300]!r}"
                    )
                    await asyncio.sleep(wait_s)
                    continue

                reason = (
                    stderr_text.strip()[:200]
                    or ("首個輸出逾時無回應（疑似限流/忙碌）" if no_output else "回應為空")
                )
                logger.error(f"Claude CLI 最終失敗（attempt {attempt + 1}）：{reason}")
                return LLMResponse(
                    message=f"⚠️ LLM 服務無回應或限流，請稍後重試（原因：{reason}）"
                )

            return LLMResponse(message="⚠️ LLM 服務無回應，請稍後重試")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Claude CLI 請求失敗: {e}")
            return LLMResponse(message=f"Claude 訂閱制請求失敗: {str(e)}")
        finally:
            if sys_prompt_file:
                try:
                    os.unlink(sys_prompt_file)
                except OSError:
                    pass

    async def _run_cli_attempt(
        self, user_prompt: str, sys_prompt_file: str,
    ) -> tuple[str, Optional["TokenUsage"], Optional[dict], str, bool]:
        """跑一次 Claude CLI。回傳 (full_text, usage, result_data, stderr_text, no_output)。

        no_output=True 表示「首個輸出 120s 內零回應」（疑似限流/卡住）。
        並行 drain stderr：(a) 避免 stderr pipe-buffer 填滿導致 CLI 阻塞，
        (b) 拿到真正的失敗原因（限流訊息常寫在 stderr）。
        """
        _FIRST_LINE_TIMEOUT = 120   # 尚未收到任何輸出時：快速失敗
        _IDLE_TIMEOUT = 300         # 已開始輸出後：行與行間最大間隔

        proc = None
        stderr_task = None
        stderr_chunks: list[bytes] = []
        env, cfg_dir = self._build_subprocess_env()
        try:
            proc = await asyncio.create_subprocess_exec(
                "claude", "-p",
                "--output-format", "stream-json",
                "--verbose",
                "--include-partial-messages",
                "--model", self.model,
                "--system-prompt-file", sys_prompt_file,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=1024 * 1024,
                env=env,
            )

            proc.stdin.write(user_prompt.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()

            async def _drain_stderr() -> None:
                try:
                    while True:
                        chunk = await proc.stderr.read(4096)
                        if not chunk:
                            break
                        stderr_chunks.append(chunk)
                except Exception:
                    pass

            stderr_task = asyncio.create_task(_drain_stderr())

            full_text = ""
            usage = None
            result_data = None
            got_first = False
            no_output = False

            while True:
                timeout = _IDLE_TIMEOUT if got_first else _FIRST_LINE_TIMEOUT
                try:
                    line = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout)
                except asyncio.TimeoutError:
                    if not got_first:
                        no_output = True
                        logger.warning(
                            f"Claude CLI 首個輸出 {_FIRST_LINE_TIMEOUT}s 無回應（疑似限流/忙碌）"
                        )
                    else:
                        logger.warning(f"Claude CLI 行間 {_IDLE_TIMEOUT}s 無新輸出，結束讀取")
                    break

                if not line:
                    break
                got_first = True

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                evt_type = data.get("type")
                if evt_type == "stream_event":
                    event = data.get("event", {})
                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            full_text += delta.get("text", "")
                elif evt_type == "result":
                    full_text = data.get("result", full_text)
                    usage = self._build_usage(data.get("usage", {}), self.model)
                    result_data = data

            # 讓 stderr 收尾（拿完整錯誤原因，尤其限流訊息），最多等 2s
            try:
                await asyncio.wait_for(stderr_task, timeout=2)
            except (asyncio.TimeoutError, Exception):
                pass

            stderr_text = b"".join(stderr_chunks).decode("utf-8", "replace")
            return full_text, usage, result_data, stderr_text, no_output

        finally:
            if stderr_task and not stderr_task.done():
                stderr_task.cancel()
            if proc and proc.returncode is None:
                try:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=5)
                    except asyncio.TimeoutError:
                        proc.kill()
                        await proc.wait()
                except Exception:
                    pass
            self._cleanup_cfg_dir(cfg_dir)

    async def chat_stream(
        self, messages: list[dict], chart_state: Optional[dict] = None, system_prompt: Optional[str] = None,
        r2_mode: bool = False,
    ):
        """真正的 streaming — 逐 chunk yield 文字"""
        import os
        import tempfile

        proc = None
        sys_prompt_file = None
        env, cfg_dir = self._build_subprocess_env()

        try:
            sys_prompt = self._build_system_message(chart_state, system_prompt, r2_mode=r2_mode)
            user_prompt = self._build_user_prompt(messages)

            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write(sys_prompt)
                sys_prompt_file = f.name

            proc = await asyncio.create_subprocess_exec(
                "claude", "-p",
                "--output-format", "stream-json",
                "--verbose",
                "--include-partial-messages",
                "--model", self.model,
                "--system-prompt-file", sys_prompt_file,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=1024 * 1024,
                env=env,
            )

            proc.stdin.write(user_prompt.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()

            # v149：首個輸出快速失敗（120s）；已開始輸出後行間放寬到 300s
            # （舊版首 token 也套 1200s → 限流/卡住時硬等 20 分鐘）
            _FIRST_LINE_TIMEOUT = 120
            _IDLE_TIMEOUT = 300
            got_first = False

            while True:
                timeout = _IDLE_TIMEOUT if got_first else _FIRST_LINE_TIMEOUT
                try:
                    line = await asyncio.wait_for(
                        proc.stdout.readline(), timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    if not got_first:
                        logger.warning(
                            f"Claude CLI streaming 首個輸出 {_FIRST_LINE_TIMEOUT}s 無回應（疑似限流/忙碌）"
                        )
                    break
                if not line:
                    break
                got_first = True

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if data.get("type") == "stream_event":
                    event = data.get("event", {})
                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta", {})
                        text = delta.get("text", "")
                        if text:
                            yield text

            await proc.wait()

        except (asyncio.CancelledError, GeneratorExit):
            pass  # 清理在 finally 中統一處理
        except Exception as e:
            logger.error(f"Claude CLI streaming 失敗: {e}")
            yield f"Claude 訂閱制請求失敗: {str(e)}"
        finally:
            # 確保子進程不會殘留
            if proc and proc.returncode is None:
                try:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=5)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        proc.kill()
                        await proc.wait()
                    logger.info("Claude CLI streaming 子進程已終止")
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            if sys_prompt_file:
                try:
                    os.unlink(sys_prompt_file)
                except OSError:
                    pass
            self._cleanup_cfg_dir(cfg_dir)

    async def chat_stream_events(
        self,
        messages: list[dict],
        chart_state: Optional[dict] = None,
        force_text: bool = False,
        system_prompt: Optional[str] = None,
        chart_screenshot: Optional[str] = None,
        r2_mode: bool = False,
    ):
        """訂閱版 Claude CLI 真串流：邊讀 stdout 邊 yield，並處理 <tool_call> XML。

        策略：
        - 累積完整 text buffer
        - 每次新 delta 進來時，先掃描已累積的 buffer 找完整 <tool_call>
        - 只 yield「不含 tool_call XML」的安全文字（保留尾端可能正在形成 tag 的部分）
        - 串流結束後 flush 剩餘 + yield usage event
        """
        import os
        import tempfile

        TOOL_OPEN = "<tool_call>"
        TOOL_CLOSE_RE = re.compile(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>")

        proc = None
        sys_prompt_file = None
        text_buffer = ""
        yielded_until = 0  # text_buffer[:yielded_until] 已 yield 過或屬於 tool_call
        usage: Optional[TokenUsage] = None
        env, cfg_dir = self._build_subprocess_env()

        try:
            sys_prompt = self._build_system_message(
                chart_state, system_prompt, include_tools=not force_text, r2_mode=r2_mode,
            )
            user_prompt = self._build_user_prompt(messages)

            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write(sys_prompt)
                sys_prompt_file = f.name

            proc = await asyncio.create_subprocess_exec(
                "claude", "-p",
                "--output-format", "stream-json",
                "--verbose",
                "--include-partial-messages",
                "--model", self.model,
                "--system-prompt-file", sys_prompt_file,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=1024 * 1024,
                env=env,
            )

            proc.stdin.write(user_prompt.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()

            # v149：首個輸出快速失敗（120s）；已開始輸出後行間放寬到 300s
            # （舊版首 token 也套 1200s → 限流/卡住時硬等 20 分鐘）
            _FIRST_LINE_TIMEOUT = 120
            _IDLE_TIMEOUT = 300
            got_first = False

            while True:
                timeout = _IDLE_TIMEOUT if got_first else _FIRST_LINE_TIMEOUT
                try:
                    line = await asyncio.wait_for(
                        proc.stdout.readline(), timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    if not got_first:
                        logger.warning(
                            f"Claude CLI stream 首個輸出 {_FIRST_LINE_TIMEOUT}s 無回應（疑似限流/忙碌）"
                        )
                    else:
                        logger.warning(f"Claude CLI stream 行間 {_IDLE_TIMEOUT}s 無新輸出，結束讀取")
                    break
                if not line:
                    break
                got_first = True

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                evt_type = data.get("type")

                if evt_type == "stream_event":
                    event = data.get("event", {})
                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text_buffer += delta.get("text", "")

                            # 嘗試提取完整 tool_call
                            while True:
                                match = TOOL_CLOSE_RE.search(text_buffer, yielded_until)
                                if not match:
                                    break
                                # yield tool_call 之前的安全文字
                                if match.start() > yielded_until:
                                    yield StreamEvent(
                                        type="text_delta",
                                        text=text_buffer[yielded_until:match.start()],
                                    )
                                # 解析並 yield function_call
                                try:
                                    obj = json.loads(match.group(1))
                                    fc = {
                                        "name": obj.get("name", ""),
                                        "arguments": obj.get("arguments", {}),
                                    }
                                    if fc["name"]:
                                        yield StreamEvent(type="function_call", function_call=fc)
                                except json.JSONDecodeError:
                                    pass
                                yielded_until = match.end()

                            # yield 安全文字（保留可能正在形成 <tool_call> 的尾端）
                            partial_open = text_buffer.rfind(TOOL_OPEN, yielded_until)
                            safe_until = (
                                partial_open if partial_open >= 0 else len(text_buffer)
                            )
                            # 若沒看到 <tool_call> 開頭，但結尾可能是 "<tool" 之類的部分匹配 → 也要保留
                            if partial_open < 0:
                                tail = text_buffer[max(yielded_until, len(text_buffer) - 11):]
                                # 結尾若可能是 <tool_call> 的部分前綴，hold 那段
                                for i in range(len(tail), 0, -1):
                                    if TOOL_OPEN.startswith(tail[-i:]):
                                        safe_until = len(text_buffer) - i
                                        break

                            if safe_until > yielded_until:
                                yield StreamEvent(
                                    type="text_delta",
                                    text=text_buffer[yielded_until:safe_until],
                                )
                                yielded_until = safe_until

                elif evt_type == "result":
                    usage_data = data.get("usage", {})
                    usage = self._build_usage(usage_data, self.model)

            await proc.wait()

            # Flush 剩餘 buffer（嘗試最後一次 tool_call 提取）
            while True:
                match = TOOL_CLOSE_RE.search(text_buffer, yielded_until)
                if not match:
                    break
                if match.start() > yielded_until:
                    yield StreamEvent(
                        type="text_delta",
                        text=text_buffer[yielded_until:match.start()],
                    )
                try:
                    obj = json.loads(match.group(1))
                    fc = {"name": obj.get("name", ""), "arguments": obj.get("arguments", {})}
                    if fc["name"]:
                        yield StreamEvent(type="function_call", function_call=fc)
                except json.JSONDecodeError:
                    pass
                yielded_until = match.end()

            # 剩餘文字：若是 unclosed tool_call 就丟掉，否則 yield
            remaining = text_buffer[yielded_until:]
            if remaining:
                if TOOL_OPEN in remaining and "</tool_call>" not in remaining:
                    pass  # 不完整的 tool_call，丟棄
                else:
                    yield StreamEvent(type="text_delta", text=remaining)

            # yield usage + stop
            if usage:
                yield StreamEvent(type="usage", usage=usage)
            yield StreamEvent(type="stop", stop_reason="end_turn")

        except (asyncio.CancelledError, GeneratorExit):
            # v110 修正：之前訊息 [client_disconnect] 誤導 — CancelledError 來源除了 client 真的斷
            # 還包括「上層 wait_for timeout」「正常 generator close」等情況。改中性訊息。
            logger.debug("Claude CLI stream 被取消（可能：client 斷線 / wait_for timeout / generator close）")
            raise
        except Exception as e:
            logger.error(f"Claude CLI streaming events 失敗: {e}")
            yield StreamEvent(type="text_delta", text=f"\n\n[Claude 訂閱版串流錯誤] {str(e)}")
            yield StreamEvent(type="stop", stop_reason="error")
        finally:
            if proc and proc.returncode is None:
                try:
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=5)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        proc.kill()
                        await proc.wait()
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            if sys_prompt_file:
                try:
                    os.unlink(sys_prompt_file)
                except OSError:
                    pass
            self._cleanup_cfg_dir(cfg_dir)


