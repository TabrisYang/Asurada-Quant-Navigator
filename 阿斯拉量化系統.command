#!/bin/bash
# ╔═══════════════════════════════════════════════════╗
# ║         阿斯拉量化系統 — 一鍵啟動腳本              ║
# ║         雙擊此檔案即可啟動系統                      ║
# ╚═══════════════════════════════════════════════════╝

# 切換到腳本所在目錄
cd "$(dirname "$0")"
ROOT_DIR="$(pwd)"

# ---------- 顏色定義 ----------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # 無顏色

# ---------- 顯示標題 ----------
clear
echo ""
echo -e "${CYAN}╔═══════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                                                   ║${NC}"
echo -e "${CYAN}║         ${BLUE}阿斯拉量化系統 v1.0.0${CYAN}                   ║${NC}"
echo -e "${CYAN}║         加密貨幣量化分析平台                       ║${NC}"
echo -e "${CYAN}║                                                   ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════╝${NC}"
echo ""

# ---------- 清理函式（Ctrl+C 時自動關閉所有程序）----------
BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
    echo ""
    echo -e "${YELLOW}正在關閉系統...${NC}"
    if [ -n "$BACKEND_PID" ]; then
        kill "$BACKEND_PID" 2>/dev/null
        wait "$BACKEND_PID" 2>/dev/null
        echo -e "${GREEN}[✓]${NC} 後端已關閉"
    fi
    if [ -n "$FRONTEND_PID" ]; then
        kill "$FRONTEND_PID" 2>/dev/null
        wait "$FRONTEND_PID" 2>/dev/null
        echo -e "${GREEN}[✓]${NC} 前端已關閉"
    fi
    echo -e "${CYAN}感謝使用阿斯拉量化系統，再見！${NC}"
    echo ""
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# ---------- 1. 檢查 Python ----------
echo -e "${YELLOW}[1/6]${NC} 檢查 Python 環境..."

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    echo -e "${GREEN}  [✓]${NC} $PYTHON_VERSION"
else
    echo -e "${RED}  [✗] 找不到 Python3！${NC}"
    echo -e "    請先安裝 Python 3.10 以上版本："
    echo -e "    ${BLUE}https://www.python.org/downloads/${NC}"
    echo ""
    echo "按 Enter 關閉..."
    read
    exit 1
fi

# ---------- 2. 檢查 Node.js ----------
echo -e "${YELLOW}[2/6]${NC} 檢查 Node.js 環境..."

if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version 2>&1)
    echo -e "${GREEN}  [✓]${NC} Node.js $NODE_VERSION"
else
    echo -e "${RED}  [✗] 找不到 Node.js！${NC}"
    echo -e "    請先安裝 Node.js 18 以上版本："
    echo -e "    ${BLUE}https://nodejs.org/${NC}"
    echo ""
    echo "按 Enter 關閉..."
    read
    exit 1
fi

# ---------- 3. 檢查 / 建立 Python 虛擬環境 ----------
echo -e "${YELLOW}[3/6]${NC} 檢查 Python 虛擬環境..."

VENV_DIR="$ROOT_DIR/backend/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo -e "  ${YELLOW}首次啟動，正在建立虛擬環境...${NC}"
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo -e "${RED}  [✗] 虛擬環境建立失敗${NC}"
        echo "按 Enter 關閉..."
        read
        exit 1
    fi
    echo -e "${GREEN}  [✓]${NC} 虛擬環境已建立"
else
    echo -e "${GREEN}  [✓]${NC} 虛擬環境已存在"
fi

# 啟動虛擬環境
source "$VENV_DIR/bin/activate"

# ---------- 4. 安裝 / 檢查 Python 依賴 ----------
echo -e "${YELLOW}[4/6]${NC} 檢查後端依賴..."

# 檢查所有關鍵套件是否都已安裝
NEED_INSTALL=false
python3 -c "import fastapi" 2>/dev/null || NEED_INSTALL=true
python3 -c "from google import genai" 2>/dev/null || NEED_INSTALL=true
python3 -c "import ccxt" 2>/dev/null || NEED_INSTALL=true
python3 -c "import pandas_ta" 2>/dev/null || NEED_INSTALL=true
python3 -c "from cryptography.fernet import Fernet" 2>/dev/null || NEED_INSTALL=true
python3 -c "import lightgbm" 2>/dev/null || NEED_INSTALL=true
python3 -c "import xgboost" 2>/dev/null || NEED_INSTALL=true
python3 -c "import sklearn" 2>/dev/null || NEED_INSTALL=true

if [ "$NEED_INSTALL" = true ]; then
    echo -e "  ${YELLOW}正在安裝/更新依賴（約需 1-2 分鐘）...${NC}"
    pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -q -r "$ROOT_DIR/backend/requirements.txt" 2>&1 | tail -3
    if [ $? -ne 0 ]; then
        echo -e "${RED}  [✗] 依賴安裝失敗，請檢查網路連線${NC}"
        echo "按 Enter 關閉..."
        read
        exit 1
    fi
    echo -e "${GREEN}  [✓]${NC} 後端依賴已安裝"
else
    echo -e "${GREEN}  [✓]${NC} 後端依賴已就緒"
fi

# ---------- 5. 安裝 / 檢查前端依賴 ----------
echo -e "${YELLOW}[5/6]${NC} 檢查前端依賴..."

if [ ! -d "$ROOT_DIR/frontend/node_modules" ]; then
    echo -e "  ${YELLOW}首次啟動，正在安裝前端依賴（約需 1-2 分鐘）...${NC}"
    cd "$ROOT_DIR/frontend"
    npm install --silent 2>&1 | tail -3
    if [ $? -ne 0 ]; then
        echo -e "${RED}  [✗] 前端依賴安裝失敗${NC}"
        echo "按 Enter 關閉..."
        read
        exit 1
    fi
    cd "$ROOT_DIR"
    echo -e "${GREEN}  [✓]${NC} 前端依賴已安裝"
else
    echo -e "${GREEN}  [✓]${NC} 前端依賴已就緒"
fi

# ---------- 6. 啟動服務 ----------
echo -e "${YELLOW}[6/6]${NC} 啟動服務..."

# 確保 data 目錄存在
mkdir -p "$ROOT_DIR/backend/data"

# 啟動後端（背景執行）
echo -e "  ${YELLOW}啟動後端 (port 8000)...${NC}"
cd "$ROOT_DIR/backend"
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level warning &
BACKEND_PID=$!
cd "$ROOT_DIR"

# 等待後端就緒
echo -ne "  等待後端就緒"
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
        echo ""
        echo -e "${GREEN}  [✓]${NC} 後端已啟動"
        break
    fi
    echo -n "."
    sleep 1
done

# 檢查後端是否真的啟動了
if ! curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
    echo ""
    echo -e "${RED}  [✗] 後端啟動超時${NC}"
    echo "按 Enter 關閉..."
    read
    exit 1
fi

# 啟動前端（背景執行）
echo -e "  ${YELLOW}啟動前端 (port 5173)...${NC}"
cd "$ROOT_DIR/frontend"
npx vite --host 0.0.0.0 --port 5173 > /dev/null 2>&1 &
FRONTEND_PID=$!
cd "$ROOT_DIR"

# 等待前端就緒
sleep 3
echo -e "${GREEN}  [✓]${NC} 前端已啟動"

# ---------- 完成！開啟瀏覽器 ----------
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}  系統已成功啟動！${NC}"
echo ""
echo -e "  主介面：${BLUE}http://localhost:5173${NC}"
echo -e "  API 文件：${BLUE}http://localhost:8000/docs${NC}"
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${YELLOW}按 Ctrl+C 可停止系統${NC}"
echo ""

# 自動開啟瀏覽器
open "http://localhost:5173" 2>/dev/null || xdg-open "http://localhost:5173" 2>/dev/null

# 保持腳本運行，等待使用者按 Ctrl+C
wait
