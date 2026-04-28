#!/bin/bash
# v103 3B：每日 drift check launchd 排程（每天 03:00 跑 daily_drift_check.py）
# 比週日 retrain (02:00) 晚一小時，確保不衝突。
# 主要任務：短窗 PSI 偵測，max PSI > 0.20 立刻觸發 retrain。

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
DRIFT_SCRIPT="$SCRIPT_DIR/daily_drift_check.py"

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
PLIST_PATH="$PLIST_DIR/com.asurada.daily_drift_check.plist"
LABEL="com.asurada.daily_drift_check"
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
        <string>${DRIFT_SCRIPT}</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>3</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>${LOG_DIR}/launchd_drift.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/launchd_drift.error.log</string>
    <key>WorkingDirectory</key>
    <string>${BACKEND_DIR}</string>
</dict>
</plist>
EOF

launchctl unload "$PLIST_PATH" 2>/dev/null || true
launchctl load "$PLIST_PATH"

echo "✅ 已安裝排程：$LABEL"
echo "   plist：$PLIST_PATH"
echo "   每天 03:00 自動執行 drift 偵測（max PSI > 0.20 觸發 retrain）"
echo ""
echo "查看狀態：launchctl list | grep daily_drift"
echo "立刻執行一次：launchctl start $LABEL"
echo "卸載：launchctl unload \"$PLIST_PATH\" && rm \"$PLIST_PATH\""
