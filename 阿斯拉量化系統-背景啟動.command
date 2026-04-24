#!/bin/bash
# ╔═══════════════════════════════════════════════════╗
# ║   阿斯拉量化系統 — 背景啟動（獨立 Terminal 視窗） ║
# ║                                                   ║
# ║   用途：從 VSCode 整合終端 / iTerm / Finder 任一  ║
# ║   位置啟動，服務都會跑在獨立的 Terminal.app       ║
# ║   視窗，關閉呼叫端（如 VSCode）不會影響系統。     ║
# ╚═══════════════════════════════════════════════════╝

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET="$SCRIPT_DIR/阿斯拉量化系統.command"

if [ ! -f "$TARGET" ]; then
    echo "❌ 找不到主啟動腳本：$TARGET"
    exit 1
fi

# 確保主腳本有可執行權限
chmod +x "$TARGET"

# 檢查是否已經在 Terminal.app 裡執行（避免在同一個 Terminal 視窗套娃）
# 若是從 VSCode 整合終端執行，TERM_PROGRAM 會是 vscode
if [ "$TERM_PROGRAM" = "Apple_Terminal" ] && [ "$1" != "--force-spawn" ]; then
    # 已經在 Terminal.app 裡，直接執行即可
    exec "$TARGET"
fi

echo "🚀 正在開啟獨立 Terminal 視窗啟動阿斯拉量化系統..."

# 將路徑中的單引號替換成 '\'' 以便安全嵌入 AppleScript 單引號字串
ESCAPED_TARGET="${TARGET//\'/\'\\\'\'}"

# 用 osascript 叫 Terminal.app 在新視窗執行主腳本
osascript <<OSA
tell application "Terminal"
    activate
    do script "'${ESCAPED_TARGET}'"
end tell
OSA

echo "✅ 已在新 Terminal 視窗啟動，您可以安全關閉此視窗或 VSCode"
echo "   主介面將自動開啟：http://localhost:5173"
