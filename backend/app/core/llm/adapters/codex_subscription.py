"""CodexSubscriptionAdapter — 透過 OpenAI Codex CLI 使用 ChatGPT 訂閱額度（v156）

與 ClaudeSubscriptionAdapter 同構：本機官方 CLI ＋「Sign in with ChatGPT」OAuth 憑證
（~/.codex/auth.json），額度計入使用者的 ChatGPT Plus/Pro 訂閱，不需 API key。

實作方式：繼承 ClaudeSubscriptionAdapter 複用整套「文字協議 function calling
（<tool_call>）、prompt 組裝、限流退避」邏輯，只覆寫 CLI 呼叫層：
- `codex exec` 非互動模式；prompt（system＋user 合併）走 stdin
- `--output-last-message <file>` 取最終回覆（比解析 JSONL 事件穩定、跨版本相容）
- `--sandbox read-only` ＋ cwd=/tmp（避開專案 AGENTS.md、禁止檔案寫入）

限制（v1）：
- 串流為「假串流」：等完整回應後分塊送出（Codex CLI 的事件格式跨版本變動大，
  先求穩定；體感是等待較久後整段快速出現）
- 模型限 Codex 系列（GPT-5.x-Codex 等），非 ChatGPT 選單全部模型
- token 用量 CLI 不穩定提供 → usage 記 0（訂閱額度制，無按量費用）
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Optional

from loguru import logger

from app.core.llm.adapters.base import TokenUsage
from app.core.llm.adapters.claude_subscription import ClaudeSubscriptionAdapter

# 總逾時：codex exec 是一次性等完整結果（非逐行），給完整分析足夠時間
_CODEX_TOTAL_TIMEOUT = 900


def _find_codex_bin() -> Optional[str]:
    """尋找 codex 執行檔（v156.1）。

    npm 全域安裝不一定在後端進程的 PATH（雙擊 .command 啟動時 PATH 精簡），
    且 Codex 桌面版/IDE 擴充把 CLI 裝在 ~/.codex/plugins/ 下 → 依序找：
    1. settings.codex_cli_path（手動覆寫）
    2. PATH 上的 codex
    3. ~/.codex/plugins/.plugin-appserver/codex（桌面版/擴充內建）
    4. 常見 npm 全域位置
    """
    import shutil

    from app.core.config.settings import settings as _settings
    if _settings.codex_cli_path and Path(_settings.codex_cli_path).exists():
        return _settings.codex_cli_path
    found = shutil.which("codex")
    if found:
        return found
    candidates = [
        Path.home() / ".codex" / "plugins" / ".plugin-appserver" / "codex",
        Path("/usr/local/bin/codex"),
        Path("/opt/homebrew/bin/codex"),
        Path.home() / ".npm-global" / "bin" / "codex",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def check_codex_cli_available() -> dict:
    """檢查 codex CLI 是否可用（PATH / 桌面版內建 / 手動路徑）且已登入。"""
    if _find_codex_bin() is None:
        return {
            "available": False,
            "error": (
                "找不到 codex 執行檔。請安裝 OpenAI Codex CLI（npm install -g @openai/codex）"
                "或 Codex 桌面版；也可在 .env 設 CODEX_CLI_PATH=/完整/路徑/codex 手動指定。"
            ),
        }
    auth = Path.home() / ".codex" / "auth.json"
    if not auth.exists():
        return {
            "available": False,
            "error": "Codex CLI 尚未登入。請執行 `codex login` 以 ChatGPT 帳號登入。",
        }
    return {"available": True, "error": None}


class CodexSubscriptionAdapter(ClaudeSubscriptionAdapter):
    """用 ChatGPT 訂閱額度的適配器（Codex CLI）。"""

    _PROVIDER_NAME = "codex_subscription"

    def __init__(self, model: str = "gpt-5.6-terra"):
        super().__init__(model=model, oauth_token=None)

    async def _run_cli_attempt(
        self, user_prompt: str, sys_prompt_file: str,
    ) -> tuple[str, Optional["TokenUsage"], Optional[dict], str, bool]:
        """跑一次 codex exec。回傳與父類同形狀 (full_text, usage, result_data, stderr, no_output)。

        codex 沒有 system prompt 參數 → 讀回 sys_prompt_file 與 user_prompt 合併走 stdin。
        """
        try:
            sys_prompt = Path(sys_prompt_file).read_text(encoding="utf-8")
        except Exception:
            sys_prompt = ""
        full_prompt = (
            f"[SYSTEM INSTRUCTIONS — 嚴格遵守]\n{sys_prompt}\n[END SYSTEM INSTRUCTIONS]\n\n{user_prompt}"
            if sys_prompt else user_prompt
        )

        out_file = None
        proc = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                out_file = f.name

            codex_bin = _find_codex_bin() or "codex"
            proc = await asyncio.create_subprocess_exec(
                codex_bin, "exec",
                "--skip-git-repo-check",
                "--sandbox", "read-only",
                "--color", "never",
                "-m", self.model,
                "--output-last-message", out_file,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,  # 進度輸出不解析（跨版本格式不穩），成品讀 out_file
                stderr=asyncio.subprocess.PIPE,
                cwd=tempfile.gettempdir(),  # 避開專案 AGENTS.md，降低 token 與干擾
            )
            try:
                _, stderr_b = await asyncio.wait_for(
                    proc.communicate(full_prompt.encode("utf-8")),
                    timeout=_CODEX_TOTAL_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning(f"Codex CLI 總逾時（{_CODEX_TOTAL_TIMEOUT}s），終止子進程")
                proc.kill()
                return "", None, None, f"總逾時 {_CODEX_TOTAL_TIMEOUT}s 無完成", True

            stderr_text = (stderr_b or b"").decode("utf-8", errors="replace")

            full_text = ""
            try:
                full_text = Path(out_file).read_text(encoding="utf-8").strip()
            except Exception:
                pass

            if not full_text:
                # exit code 非 0 或空輸出：交給父類的限流判定與退避重試
                logger.warning(
                    f"Codex CLI 空回應（exit={proc.returncode}）stderr={stderr_text.strip()[:200]!r}"
                )
                return "", None, None, stderr_text, proc.returncode is None

            usage = TokenUsage(
                prompt_tokens=0, completion_tokens=0, total_tokens=0,
                model=self.model, provider=self._PROVIDER_NAME,
            )
            return full_text, usage, None, stderr_text, False
        except FileNotFoundError:
            return "", None, None, "找不到 codex 命令（npm install -g @openai/codex）", False
        except asyncio.CancelledError:
            if proc and proc.returncode is None:
                proc.kill()
            raise
        finally:
            if out_file:
                Path(out_file).unlink(missing_ok=True)

    # ── 串流：v1 假串流（等完整回應後分塊）— 覆寫掉父類的 claude stream-json 實作 ──

    async def chat_stream(self, messages, chart_state=None, system_prompt=None):
        response = await self.chat(
            messages, chart_state=chart_state, system_prompt=system_prompt,
        )
        text = response.message or ""
        _CHUNK = 80
        for i in range(0, len(text), _CHUNK):
            yield text[i:i + _CHUNK]

    async def chat_stream_events(
        self, messages, chart_state=None, force_text=False,
        system_prompt=None, chart_screenshot=None, r2_mode=False,
    ):
        # 用 BaseLLMAdapter 的預設假串流（chat() 完成後一次 yield 事件），
        # 跳過父類針對 claude stream-json 的真串流實作。
        from app.core.llm.adapters.base import BaseLLMAdapter
        async for evt in BaseLLMAdapter.chat_stream_events(
            self, messages, chart_state=chart_state, force_text=force_text,
            system_prompt=system_prompt, chart_screenshot=chart_screenshot, r2_mode=r2_mode,
        ):
            yield evt
