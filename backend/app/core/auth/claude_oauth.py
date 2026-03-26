"""Claude Code CLI availability checker

Checks if the `claude` CLI is installed and authenticated,
so we can use it as a proxy for subscription-based access.
"""

import json
import shutil
import subprocess

from loguru import logger


def check_claude_cli_available() -> dict:
    """Check if Claude Code CLI is installed and authenticated.

    Returns:
        {"available": bool, "error": str|None}
    """
    if not shutil.which("claude"):
        return {
            "available": False,
            "error": (
                "找不到 claude 命令。"
                "請先安裝 Claude Code：https://docs.anthropic.com/en/docs/claude-code"
            ),
        }

    try:
        result = subprocess.run(
            ["claude", "auth", "status"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout.strip()

        # CLI 回傳 JSON：{"loggedIn": true, ...}
        try:
            data = json.loads(output)
            if data.get("loggedIn") is True:
                logger.debug("Claude Code CLI is authenticated")
                return {"available": True, "error": None}
        except (json.JSONDecodeError, ValueError):
            pass

        # 向下相容：舊版 CLI 可能用文字輸出
        combined = output + result.stderr
        if "Logged in" in combined or "authenticated" in combined.lower():
            return {"available": True, "error": None}

        return {
            "available": False,
            "error": (
                "Claude Code 尚未登入。"
                "請執行 `claude auth login` 完成登入。"
            ),
        }
    except subprocess.TimeoutExpired:
        return {"available": False, "error": "Claude CLI 回應超時"}
    except Exception as e:
        return {"available": False, "error": f"檢查 Claude CLI 失敗: {e}"}
