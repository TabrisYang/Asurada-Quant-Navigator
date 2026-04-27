#!/bin/bash
# 安裝 macOS launchd 自動模仿學習重訓排程：每週日 02:00 跑 retrain_imitation.py
# 跟系統有沒有開無關，只要 Mac 醒著時間到就會跑

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
RETRAIN_SCRIPT="$SCRIPT_DIR/retrain_imitation.py"

if [ -x "$BACKEND_DIR/.venv/bin/python3" ]; then
    PYTHON_BIN="$BACKEND_DIR/.venv/bin/python3"
else
    PYTHON_BIN="$(command -v python3)"
fi

if [ -z "$PYTHON_BIN" ]; then
    echo "❌ 找不到 python3"
    exit 1
fi

PLIST_DIR="$HOME/Library/LaunchAgents"
PLIST_PATH="$PLIST_DIR/com.asurada.imitation_retrain.plist"
LABEL="com.asurada.imitation_retrain"
LOG_DIR="$BACKEND_DIR/data/db"

mkdir -p "$PLIST_DIR" "$LOG_DIR"

cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON_BIN}</string>
        <string>${RETRAIN_SCRIPT}</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Weekday</key>
        <integer>0</integer>
        <key>Hour</key>
        <integer>2</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>${LOG_DIR}/launchd_retrain.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/launchd_retrain.error.log</string>
    <key>WorkingDirectory</key>
    <string>${BACKEND_DIR}</string>
</dict>
</plist>
EOF

# 重新載入
launchctl unload "$PLIST_PATH" 2>/dev/null || true
launchctl load "$PLIST_PATH"

echo "✅ 已安裝排程：$LABEL"
echo "   plist：$PLIST_PATH"
echo "   每週日 02:00 自動執行模仿學習重訓"
echo ""
echo "查看狀態：launchctl list | grep imitation"
echo "立刻執行一次：launchctl start $LABEL"
echo "卸載：launchctl unload \"$PLIST_PATH\" && rm \"$PLIST_PATH\""
