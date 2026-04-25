#!/bin/bash
# 安裝 macOS launchd 自動備份排程：每天 0:00 自動執行 backup_databases.py
# 跟系統有沒有開無關，只要 Mac 醒著時間到就會跑

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
BACKUP_SCRIPT="$SCRIPT_DIR/backup_databases.py"

# 優先用專案 venv，否則 fallback 系統 python3
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
PLIST_PATH="$PLIST_DIR/com.asurada.daily_backup.plist"
LABEL="com.asurada.daily_backup"
LOG_DIR="$BACKEND_DIR/data/db/backups"

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
        <string>${BACKUP_SCRIPT}</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>0</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>${LOG_DIR}/launchd_backup.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/launchd_backup.error.log</string>
    <key>WorkingDirectory</key>
    <string>${BACKEND_DIR}</string>
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
EOF

# 重新載入（先 unload 舊的，再 load 新的）
launchctl unload "$PLIST_PATH" 2>/dev/null || true
launchctl load "$PLIST_PATH"

echo "✅ 已安裝排程：$LABEL"
echo "   plist：$PLIST_PATH"
echo "   Python：$PYTHON_BIN"
echo "   每天 0:00 自動執行備份"
echo ""
echo "查看狀態：launchctl list | grep asurada"
echo "立刻執行一次：launchctl start $LABEL"
echo "卸載：launchctl unload \"$PLIST_PATH\" && rm \"$PLIST_PATH\""
