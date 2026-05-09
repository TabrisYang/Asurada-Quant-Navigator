# 阿斯拉量化系統 V2

加密貨幣 / 台股量化分析平台。後端 FastAPI + SQLite，前端 React + TypeScript。

## 必讀治理文件（修改前先讀）

| 文件 | 何時必讀 |
|---|---|
| [backend/docs/SHADOW_FEATURES.md](backend/docs/SHADOW_FEATURES.md) | 移除任何「看似 dead code」的功能前 |
| [backend/docs/CHART_STATE_SCHEMA.md](backend/docs/CHART_STATE_SCHEMA.md) | 修改 chart_state 結構（新增/移除欄位）前 |
| 本檔案 | 任何接手者第一次進來 |

## 量化系統改動原則

任何觸及以下核心檔案的改動，必須先執行 pre-flight check：

- `backend/app/core/regime_filter.py`
- `backend/app/core/regime_subtype.py`
- `backend/app/core/llm/function_defs.py` 的決策卡規則段
- `backend/app/core/indicators/` 下任何檔案
- `backend/app/core/ml/scenario_predictor.py`
- `backend/app/core/prediction_tracker.py`

### Pre-flight check 流程（必跑）

1. 執行 `python3 backend/scripts/shadow_mode.py --days 14` 收集 baseline 數據
2. 確認 `prediction_tracker` 樣本量足夠（direction_balance ≥10、combo_stats ≥10）
3. 確認 CPCV PF 基線（基準 ~1.45）
4. 改動上線後並行跑 shadow mode 至少 2 週
5. 比對前後 PF / 勝率 / 觸發頻率，劣化 >10% 必須回滾

### 為什麼有這個原則

系統現有 alpha 是統計強健的（CPCV 一致性 100% / PF 1.45 / MC 99.7% / Walk Forward Alpha pass）。盲改核心邏輯（regime / bias_score）會傷邊際。任何改動都必須先用 shadow mode 量化「問題是真 bug 還是 alpha」，避免基於主觀感受改代碼。

任何 AI 工具看到此檔案都應遵守此原則：在動上述核心檔案前，先跑 `shadow_mode.py` 並輸出 baseline 報告，不要直接修改。

## 重構與清理紀律

**過去教訓**：v100→v120 累積疊加修補導致 chat.py 3611 行、chart_state 26 個注入欄位，最終引發 stream 中斷 regression。最近 50 commit 中 72% 是 bug fix、僅 2% refactor，技術債利息超過償還速度。

### 規則

1. **新增欄位 / 模組前必先檢查能否合併現有功能**
   - 修改 chart_state 前讀 [CHART_STATE_SCHEMA.md](backend/docs/CHART_STATE_SCHEMA.md)
   - 新增 `_PROMPT_MODULES` 前先查既有 31 個是否已涵蓋
   - 新增 LLM adapter 前先確認不是同一供應商的不同 transport

2. **「看似 dead code」必先驗證**（見 [SHADOW_FEATURES.md](backend/docs/SHADOW_FEATURES.md)）
   - `flag = False` ≠ dead code，可能有 shadow mode 或 API 動態啟用
   - grep 全引用後才能決定

3. **重構配額：每 5 個 feature/bug-fix commit 至少 1 個 refactor commit**
   - 過去 50 commit 中 refactor 僅 2%（健康應 ≥ 15%）
   - 任何超過 100 行的新增需附「能否從現有結構擴展」的評估

4. **commit message 規範**
   - chart_state 變更：`chart_state: 新增/移除/修改 X 欄位（用途：Y、進 round2: Z）`
   - 大型重構：`refactor: 描述 + 影響範圍`
   - 補丁：`fix(VXXX): 描述`
   - 在 stream 修復 / chart_state 重組類改動，commit message 含「為何」更勝「做了什麼」

### 自動化護欄

[backend/scripts/check_repo_health.py](backend/scripts/check_repo_health.py) 強制執行：
- 大型檔案行數限制（chat.py 4500 行、function_defs.py 3500 行 等）
- chart_state 賦值次數上限（35 次 hard block）
- prompt module 數量上限（40 個 hard block）

**安裝為 pre-commit hook**：
```bash
bash backend/scripts/install_git_hooks.sh
```

每次 git commit 前自動跑檢查，違反 hard limit 會 block commit。

## 快捷指令

Claude Code 使用者可直接用 `/preflight` 觸發 pre-flight check 流程（見 `.claude/commands/preflight.md`）。

## 主要目錄

- `backend/app/` — FastAPI 應用
- `backend/app/core/` — 核心邏輯（indicators / regime / llm / ml / backtest）
- `backend/app/api/routes/chat.py` — 主分析流程入口
- `backend/data/db/` — SQLite 資料庫（predictions.db、ml.db 等）
- `backend/scripts/` — 維運腳本（audit / backfill / drift check / shadow mode / health check）
- `backend/docs/` — 治理文件（SHADOW_FEATURES、CHART_STATE_SCHEMA）
- `frontend/src/` — React 前端
