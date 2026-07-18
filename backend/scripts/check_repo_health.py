"""阿斯拉量化系統 — Repo Health Check（pre-commit hook）。

檢查項目：
  1. 大型檔案行數限制（block 主要肥大檔案再膨脹）
  2. chat.py 中 chart_state[X] 賦值次數限制（防 chart_state 再膨脹）
  3. function_defs.py 中 _PROMPT_MODULES 數量限制
  4. CLAUDE.md / SHADOW_FEATURES.md / CHART_STATE_SCHEMA.md 是否仍存在

執行：
    python3 backend/scripts/check_repo_health.py        # 檢查所有規則
    python3 backend/scripts/check_repo_health.py --soft # warning 不 block

可作為 pre-commit hook 安裝：
    bash backend/scripts/install_git_hooks.sh
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# ─── 規則定義 ──────────────────────────────────────────

# 規則 1：大型檔案行數限制
# warning：黃線；block：紅線（超過會 fail commit）
LINE_LIMITS: dict[str, tuple[int, int]] = {
    # 檔案路徑（相對 repo root）: (warning 行數, block 行數)
    "backend/app/api/routes/chat.py": (3700, 4500),
    "backend/app/core/llm/function_defs.py": (3000, 3500),
    "backend/app/core/llm/prompt_modules.py": (2600, 3200),
    "backend/app/core/llm/executor.py": (2600, 3000),
    # v154：adapter.py 拆為 adapters/ package（façade ~40 行）；護欄跟著搬
    "backend/app/core/llm/adapter.py": (100, 200),
    "backend/app/core/llm/adapters/base.py": (600, 800),
    "backend/app/core/llm/adapters/claude_subscription.py": (900, 1100),
    "backend/app/core/llm/adapters/gemini_adapter.py": (500, 700),
    "backend/app/core/llm/adapters/claude_adapter.py": (450, 600),
    "backend/app/core/llm/adapters/openai_adapter.py": (450, 600),
    "backend/app/core/prediction_tracker.py": (1900, 2300),
    "frontend/src/components/ChatInterface/ChatInterface.tsx": (2400, 2900),
    "frontend/src/stores/chartStore.ts": (700, 900),
    # v154：TwBBScanPanel 拆分後的護欄
    "frontend/src/components/TwBBScanPanel/TwBBScanPanel.tsx": (1000, 1300),
    "frontend/src/components/TwBBScanPanel/TrackView.tsx": (1100, 1400),
}

# 規則 2：chat.py 中 chart_state[X] = ... 的賦值次數限制
# regex 涵蓋字母 / 底線 / 數字（如 signal_v2、dummy_1）
CHAT_STATE_INJECT_PATTERN = re.compile(r'chart_state\["[a-zA-Z_0-9]+"\]\s*=')
CHAT_STATE_MAX = 28   # 目前 baseline 23（含 chart_state[X] / request.chart_state[X]），警戒線 28
CHAT_STATE_BLOCK = 33  # 硬性上限 33（容許 +10 緩衝）

# 規則 3：function_defs.py 中 prompt module 數量限制
PROMPT_MODULE_PATTERN = re.compile(r'_PROMPT_MODULES\["[a-z_0-9]+"\]\s*=')
PROMPT_MODULE_MAX = 35  # 目前 33，警戒線 35，硬性上限 40
PROMPT_MODULE_BLOCK = 40

# 規則 4：必要文件存在
REQUIRED_DOCS = [
    "CLAUDE.md",
    "backend/docs/SHADOW_FEATURES.md",
    "backend/docs/CHART_STATE_SCHEMA.md",
]


# ─── 檢查函式 ──────────────────────────────────────────


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)


def check_line_limits(soft: bool) -> tuple[list[str], list[str]]:
    """檢查大型檔案行數。回傳 (warnings, blocks)。"""
    warnings: list[str] = []
    blocks: list[str] = []
    for rel_path, (warn_n, block_n) in LINE_LIMITS.items():
        path = _REPO_ROOT / rel_path
        if not path.exists():
            continue
        n = _count_lines(path)
        if n >= block_n and not soft:
            blocks.append(
                f"❌ {rel_path}: {n} 行 ≥ block 上限 {block_n}。請拆分為多個檔案再 commit。"
            )
        elif n >= warn_n:
            warnings.append(
                f"⚠️  {rel_path}: {n} 行 ≥ warning {warn_n}（block 在 {block_n}）。考慮拆分。"
            )
    return warnings, blocks


def check_chart_state_inject(soft: bool) -> tuple[list[str], list[str]]:
    """檢查 chat.py 中 chart_state 賦值次數。"""
    warnings: list[str] = []
    blocks: list[str] = []
    chat_py = _REPO_ROOT / "backend/app/api/routes/chat.py"
    if not chat_py.exists():
        return warnings, blocks
    text = chat_py.read_text(encoding="utf-8", errors="ignore")
    matches = CHAT_STATE_INJECT_PATTERN.findall(text)
    n = len(matches)
    if n >= CHAT_STATE_BLOCK and not soft:
        blocks.append(
            f"❌ chat.py 中 chart_state[X]= 賦值 {n} 次 ≥ block {CHAT_STATE_BLOCK}。"
            f"chart_state 已過度膨脹。新增前必讀 backend/docs/CHART_STATE_SCHEMA.md，"
            f"並評估能否合併現有欄位。"
        )
    elif n >= CHAT_STATE_MAX:
        warnings.append(
            f"⚠️  chat.py 中 chart_state[X]= 賦值 {n} 次 ≥ warning {CHAT_STATE_MAX}（block 在 {CHAT_STATE_BLOCK}）。"
            f"請更新 backend/docs/CHART_STATE_SCHEMA.md。"
        )
    return warnings, blocks


def check_prompt_modules(soft: bool) -> tuple[list[str], list[str]]:
    """檢查 _PROMPT_MODULES 數量。

    內容主體已抽至 prompt_modules.py，跨兩檔統計以防有人繞回
    function_defs.py 直接加模組。
    """
    warnings: list[str] = []
    blocks: list[str] = []
    text = ""
    for rel in ("backend/app/core/llm/prompt_modules.py",
                "backend/app/core/llm/function_defs.py"):
        path = _REPO_ROOT / rel
        if path.exists():
            text += path.read_text(encoding="utf-8", errors="ignore")
    if not text:
        return warnings, blocks
    matches = PROMPT_MODULE_PATTERN.findall(text)
    n = len(matches)
    if n >= PROMPT_MODULE_BLOCK and not soft:
        blocks.append(
            f"❌ prompt_modules.py/function_defs.py 中 _PROMPT_MODULES 數量 {n} ≥ block {PROMPT_MODULE_BLOCK}。"
            f"prompt module 過多。新增前評估能否合併現有 module。"
        )
    elif n >= PROMPT_MODULE_MAX:
        warnings.append(
            f"⚠️  prompt_modules.py/function_defs.py 中 _PROMPT_MODULES 數量 {n} ≥ warning {PROMPT_MODULE_MAX}（block 在 {PROMPT_MODULE_BLOCK}）。"
        )
    return warnings, blocks


def check_required_docs() -> tuple[list[str], list[str]]:
    """檢查治理文件是否存在。"""
    warnings: list[str] = []
    blocks: list[str] = []
    for rel_path in REQUIRED_DOCS:
        path = _REPO_ROOT / rel_path
        if not path.exists():
            blocks.append(
                f"❌ 必要治理文件遺失：{rel_path}。重新建立或從 git 還原。"
            )
    return warnings, blocks


# ─── 主流程 ──────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="阿斯拉量化系統 — Repo health check")
    parser.add_argument(
        "--soft", action="store_true",
        help="僅顯示警告，不 block commit（適合手動跑）",
    )
    args = parser.parse_args()

    print("─" * 60)
    print("  阿斯拉量化系統 — Repo Health Check")
    print("─" * 60)

    all_warnings: list[str] = []
    all_blocks: list[str] = []

    for check_fn in (
        check_line_limits,
        check_chart_state_inject,
        check_prompt_modules,
    ):
        w, b = check_fn(args.soft)
        all_warnings.extend(w)
        all_blocks.extend(b)

    # 必要文件檢查不受 --soft 影響（治理文件必須在）
    w, b = check_required_docs()
    all_warnings.extend(w)
    all_blocks.extend(b)

    if all_warnings:
        print("\n  Warnings:")
        for line in all_warnings:
            print(f"    {line}")

    if all_blocks:
        print("\n  Blocks:")
        for line in all_blocks:
            print(f"    {line}")
        print()
        print("─" * 60)
        print("  ❌ Repo health check FAILED")
        print("─" * 60)
        sys.exit(1)

    if not all_warnings:
        print("\n  ✅ 全部通過")

    print()
    print("─" * 60)
    print("  ✅ Repo health check PASSED")
    print("─" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
