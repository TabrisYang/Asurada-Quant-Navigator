# 阿斯拉量化系統 — 一鍵上手 bootstrap（Windows / PowerShell）
#
# 從零把專案準備到「可啟動」：檢查環境 → 建 venv + 裝後端依賴 →
# 裝前端依賴 → 抓好幾個預設標的的行情資料。
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1 -NoData
param([switch]$NoData)

$ErrorActionPreference = "Stop"
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir    = Split-Path -Parent $ScriptDir
$BackendDir = Join-Path $RootDir "backend"
$FrontendDir = Join-Path $RootDir "frontend"

function Info($m) { Write-Host "▶ $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "✓ $m" -ForegroundColor Green }
function Die($m)  { Write-Host "✗ $m" -ForegroundColor Red; exit 1 }

# ── 1. 環境檢查 ──────────────────────────────────────────────
Info "檢查環境需求（Python 3.11+ / Node 20+）"
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { Die "找不到 python，請先安裝 Python 3.11 以上。" }
$pyOk = python -c "import sys; print(1 if sys.version_info[:2] >= (3,11) else 0)"
if ($pyOk.Trim() -ne "1") { Die "Python 版本過舊（需 3.11+）。目前：$(python -V)" }
Ok "Python: $(python -V)"

if (-not (Get-Command node -ErrorAction SilentlyContinue)) { Die "找不到 node，請先安裝 Node.js 20 以上。" }
$nodeMajor = (node -p "process.versions.node.split('.')[0]").Trim()
if ([int]$nodeMajor -lt 20) { Die "Node 版本過舊（需 20+）。目前：$(node -v)" }
Ok "Node: $(node -v)"

# ── 2. 後端 venv + 依賴 ──────────────────────────────────────
Info "建立後端虛擬環境並安裝依賴"
Set-Location $BackendDir
if (-not (Test-Path ".venv")) { python -m venv .venv }
$VenvPy = Join-Path $BackendDir ".venv\Scripts\python.exe"
& $VenvPy -m pip install --upgrade pip | Out-Null
if (Test-Path "requirements.lock.txt") {
  Info "使用 requirements.lock.txt 精確安裝（完全重現可用組合）"
  & $VenvPy -m pip install -r requirements.lock.txt
} else {
  & $VenvPy -m pip install -r requirements.txt
}
Ok "後端依賴安裝完成"

# ── 3. 前端依賴 ──────────────────────────────────────────────
Info "安裝前端依賴（npm ci）"
Set-Location $FrontendDir
if (Test-Path "package-lock.json") { npm ci } else { npm install }
Ok "前端依賴安裝完成"

# ── 4. 抓預設行情資料 ────────────────────────────────────────
if (-not $NoData) {
  Info "抓取預設標的行情（BTC/USDT、ETH/USDT 日線）— 首次可能需幾分鐘"
  Set-Location $BackendDir
  try { & $VenvPy scripts/backfill_history.py --symbols BTC/USDT ETH/USDT --timeframes 1d }
  catch { Write-Host "⚠ 抓資料失敗（可能是網路問題），可稍後手動重跑 backfill_history.py" -ForegroundColor Yellow }
} else {
  Info "略過抓資料（-NoData）。日後可跑 backend/scripts/backfill_history.py 補資料"
}

# ── 5. 完成，印啟動指令 ──────────────────────────────────────
Ok "Bootstrap 完成！"
Write-Host @"

接下來分別啟動後端與前端：

  後端（終端機 1）：
    cd backend; .venv\Scripts\python run.py        # → http://localhost:8000（/api）

  前端（終端機 2）：
    cd frontend; npm run dev                       # → http://localhost:5173

然後開瀏覽器 http://localhost:5173 ，右上角「設定」填入 LLM 供應商與 API Key 即可使用。
"@
