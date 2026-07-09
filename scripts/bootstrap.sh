#!/usr/bin/env bash
# 阿斯拉量化系統 — 一鍵上手 bootstrap（macOS / Linux）
#
# 從零把專案準備到「可啟動」：檢查環境 → 建 venv + 裝後端依賴 →
# 裝前端依賴 → 抓好幾個預設標的的行情資料。跑完照提示啟動前後端即可。
#
# 用法：
#   bash scripts/bootstrap.sh              # 完整（含抓資料）
#   bash scripts/bootstrap.sh --no-data    # 跳過抓行情資料
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"

FETCH_DATA=1
for arg in "$@"; do
  [ "$arg" = "--no-data" ] && FETCH_DATA=0
done

info() { printf "\033[1;36m▶ %s\033[0m\n" "$1"; }
ok()   { printf "\033[1;32m✓ %s\033[0m\n" "$1"; }
die()  { printf "\033[1;31m✗ %s\033[0m\n" "$1" >&2; exit 1; }

# ── 1. 環境檢查 ─────────────────────────────────────────────
info "檢查環境需求（Python 3.11+ / Node 20+）"

command -v python3 >/dev/null 2>&1 || die "找不到 python3，請先安裝 Python 3.11 以上。"
PY_OK=$(python3 -c 'import sys; print(1 if sys.version_info[:2] >= (3,11) else 0)')
[ "$PY_OK" = "1" ] || die "Python 版本過舊（需 3.11+）。目前：$(python3 -V)"
ok "Python: $(python3 -V)"

command -v node >/dev/null 2>&1 || die "找不到 node，請先安裝 Node.js 20 以上。"
NODE_MAJOR=$(node -p "process.versions.node.split('.')[0]")
[ "$NODE_MAJOR" -ge 20 ] || die "Node 版本過舊（需 20+）。目前：$(node -v)"
ok "Node: $(node -v)"

# ── 2. 後端 venv + 依賴 ─────────────────────────────────────
info "建立後端虛擬環境並安裝依賴"
cd "$BACKEND_DIR"
[ -d .venv ] || python3 -m venv .venv
VENV_PY="$BACKEND_DIR/.venv/bin/python"
"$VENV_PY" -m pip install --upgrade pip >/dev/null
if [ -f requirements.lock.txt ]; then
  info "使用 requirements.lock.txt 精確安裝（完全重現可用組合）"
  "$VENV_PY" -m pip install -r requirements.lock.txt
else
  "$VENV_PY" -m pip install -r requirements.txt
fi
ok "後端依賴安裝完成"

# ── 3. 前端依賴 ─────────────────────────────────────────────
info "安裝前端依賴（npm ci）"
cd "$FRONTEND_DIR"
if [ -f package-lock.json ]; then npm ci; else npm install; fi
ok "前端依賴安裝完成"

# ── 4. 抓預設行情資料 ───────────────────────────────────────
if [ "$FETCH_DATA" = "1" ]; then
  info "抓取預設標的行情（BTC/USDT、ETH/USDT 日線）— 首次可能需幾分鐘"
  cd "$BACKEND_DIR"
  "$VENV_PY" scripts/backfill_history.py --symbols BTC/USDT ETH/USDT --timeframes 1d || \
    printf "\033[1;33m⚠ 抓資料失敗（可能是網路問題），可稍後手動重跑 backfill_history.py\033[0m\n"
else
  info "略過抓資料（--no-data）。日後可跑 backend/scripts/backfill_history.py 補資料"
fi

# ── 5. 完成，印啟動指令 ─────────────────────────────────────
ok "Bootstrap 完成！"
cat <<EOF

接下來分別啟動後端與前端：

  後端（終端機 1）：
    cd backend && .venv/bin/python run.py         # → http://localhost:8000（/api）

  前端（終端機 2）：
    cd frontend && npm run dev                    # → http://localhost:5173

然後開瀏覽器 http://localhost:5173 ，右上角「設定」填入 LLM 供應商與 API Key 即可使用。
EOF
