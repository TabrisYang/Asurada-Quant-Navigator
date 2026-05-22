#!/bin/bash
# 每週 shadow mode launchd 排程（每週一 04:00 跑 shadow_mode.py --days 90）
# 用途：核心邏輯改動上線後的 2 週並行監控自動化。
# 每次產出 data/db/shadow_report_YYYYMMDD.json，累積多週報告供比對
# PF / 勝率 / 觸發頻率，劣化 > 10% 應回滾（見 CLAUDE.md「量化系統改動原則」）。
# 排在週日 02:00 retrain、每日 03:00 drift 之後的週一 04:00，避免資源競爭。

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
SHADOW_SCRIPT="$SCRIPT_DIR/shadow_mode.py"

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
PLIST_PATH="$PLIST_DIR/com.asurada.weekly_shadow_mode.plist"
LABEL="com.asurada.weekly_shadow_mode"
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
        <string>${SHADOW_SCRIPT}</string>
        <string>--days</string>
        <string>90</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Weekday</key>
        <integer>1</integer>
        <key>Hour</key>
        <integer>4</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>${LOG_DIR}/launchd_shadow.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/launchd_shadow.error.log</string>
    <key>WorkingDirectory</key>
    <string>${BACKEND_DIR}</string>
</dict>
</plist>
EOF

launchctl unload "$PLIST_PATH" 2>/dev/null || true
launchctl load "$PLIST_PATH"

echo "✅ 已安裝排程：$LABEL"
echo "   plist：$PLIST_PATH"
echo "   每週一 04:00 自動跑 shadow_mode --days 90，報告存 $LOG_DIR/shadow_report_YYYYMMDD.json"
echo ""
echo "查看狀態：launchctl list | grep weekly_shadow"
echo "立刻執行一次：launchctl start $LABEL"
echo "看 log：tail -f $LOG_DIR/launchd_shadow.log"
echo "卸載：launchctl unload \"$PLIST_PATH\" && rm \"$PLIST_PATH\""
