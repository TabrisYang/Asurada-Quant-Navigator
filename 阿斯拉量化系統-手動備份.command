#!/bin/bash
# 手動觸發 SQLite 資料庫備份（雙擊執行）
# 自動備份由 launchd 在每天 0:00 跑；這個腳本是給「立刻備份」用的

cd "$(dirname "$0")"
ROOT_DIR="$(pwd)"

# ---------- 顏色 ----------
CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

clear
echo -e "${CYAN}╔═══════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║      阿斯拉量化系統 — 手動資料庫備份              ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════╝${NC}"
echo ""

# 優先用專案 venv，否則 fallback 系統 python3
if [ -x "$ROOT_DIR/backend/.venv/bin/python3" ]; then
    PYTHON_BIN="$ROOT_DIR/backend/.venv/bin/python3"
else
    PYTHON_BIN="$(command -v python3)"
fi

if [ -z "$PYTHON_BIN" ]; then
    echo -e "${RED}❌ 找不到 python3${NC}"
    echo ""
    echo "按任意鍵關閉..."
    read -n 1
    exit 1
fi

"$PYTHON_BIN" "$ROOT_DIR/backend/scripts/backup_databases.py"
RC=$?

echo ""
if [ $RC -eq 0 ]; then
    echo -e "${GREEN}備份完成。${NC}"
else
    echo -e "${RED}備份過程發生錯誤（exit code: $RC）${NC}"
fi
echo ""
echo "按任意鍵關閉..."
read -n 1
