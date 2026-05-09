# Shadow Mode / Opt-in 功能清單

## 目的

**這個文件存在是因為一次踩雷的教訓。**

2026-05-09 嘗試清理「dead code」時，我把 `imitation_learning_enabled = False` 解讀成「整套 imitation learning 是 dead code」，準備刪除。實際上這是錯誤判斷：

- `imitation_shadow_mode = True`（**預設開**）— 主流程仍會走 v101 推論，只是不暴露給使用者
- `auto_rollback_enabled = True`（**永遠開**）— 是 fail-safe 機制
- `champion_challenger_enabled = False` — 但 [predictions.py:86,100,136](../app/api/routes/predictions.py) 提供 API 端點動態啟用
- `imitation_learning_enabled = False` — 但 [predictions.py:117](../app/api/routes/predictions.py) 提供 API 動態啟用

**教訓**：在這個專案，`flag = False` ≠ dead code。任何要清理的功能都必須先驗證所有引用點。

任何 AI 工具（含 Claude Code agents、Cursor、其他）看到此檔案，**移除以下任何模組前必須先讀此文件**。

---

## Shadow Mode 功能清單

### 1. v101 Imitation Learning（強烈不可移除）

**主開關**：[settings.py:82-93](../app/core/config/settings.py#L82-L93)
```python
imitation_learning_enabled: bool = False     # user 端注入主開關
imitation_shadow_mode: bool = True            # ★ 預設開：subprocess 推論 + 累積資料
champion_challenger_enabled: bool = False     # 自動模型切換
auto_rollback_enabled: bool = True            # ★ 永遠開：表現變差自動關閉
adversarial_val_enabled: bool = False         # Drift 偵測
```

**關鍵理解**：
- `imitation_learning_enabled OR imitation_shadow_mode` → 主流程進入 v101 路徑（[chat.py:942](../app/api/routes/chat.py#L942)）
- shadow mode 時 subprocess 推論但不注入給使用者，**累積資料用於 quality gate**
- API 端可以動態切到 user mode（[predictions.py:117](../app/api/routes/predictions.py)）

**涉及檔案（全部不可移除）**：
- `app/core/canary.py` — % rollout 決定哪些 user 收到 v101 結果
- `app/core/auto_rollback.py` — 表現變差時自動關閉（永遠開）
- `app/core/champion_challenger.py` — 模型切換邏輯
- `app/core/v101_self_validator.py` — 每週 self-validation
- `app/core/imitation_predictor.py` — 推論引擎
- `app/core/ml_client.py` — subprocess 包裝（避免 lightgbm/shap 衝突 segfault）
- `app/core/feature_extractor.py` — 39 個特徵抽取

**驗證方式**（移除前必跑）：
```bash
grep -rn "imitation_learning_enabled\|imitation_shadow_mode\|canary\|auto_rollback\|champion_challenger" backend/app --include="*.py"
```

如果有任何引用點 → 不可移除。

---

### 2. ML Pipeline（可選但有 reader）

**主開關**：[ml/_settings.py](../app/core/ml/_settings.py)
```python
ML_ENABLED: bool  # 預設依環境
```

**關鍵理解**：
- `ML_ENABLED = False` 時 [chat.py:_inject_ml_prediction](../app/api/routes/chat.py) 直接 return，不注入 mlPrediction
- 但 chart_state.mlPrediction 欄位被 fact_checker 與 prompt 引用
- 移除 ML 模組會讓 chart_state.mlPrediction 永遠為 None，需要同步清理 reader

---

### 3. RL Strategic Insight（在 v101 imitation learning 之上）

[chat.py:994](../app/api/routes/chat.py#L994) 注入 `rl_strategic_insight` 到 chart_state，需要：
- 通過 v101 6 層守衛（chart_symbol、intent、chart_state、shadow_mode）
- subprocess 推論成功
- 通過 canary % rollout

如果你看到這個欄位「沒在用」，那是因為 canary % 沒命中，**不是因為功能廢棄**。

---

## 通用驗證流程

任何打算清理「看起來廢棄」的功能前：

```bash
# 1. grep 所有 .py 檔案
grep -rn "FUNCTION_OR_FLAG_NAME" backend/app --include="*.py"

# 2. 檢查 settings.py 是否還有其他相關 flag
grep -n "TARGET" backend/app/core/config/settings.py

# 3. 檢查 API 端點是否動態啟用
grep -rn "TARGET" backend/app/api --include="*.py"

# 4. 確認所有引用點後才能決定是否移除
```

**只有當以上步驟全部通過 + 所有引用點都被處理 + 跑過 shadow_mode + ETH/USDT 全部分析驗證 → 才可以移除。**

---

## 歷史紀錄

| 日期 | 事件 | 教訓 |
|---|---|---|
| 2026-05-09 | Explore agent 將 imitation_learning 視為 dead code | flag=False 不代表 dead；shadow mode 仍在跑 |
| 2026-05-09 | Explore agent 將 ClaudeAdapter 視為 90% 重複 | 兩個 adapter 是不同 transport（API vs CLI） |
| 2026-05-09 | 嘗試合併 donchian 三個欄位 | fact_checker.py + function_defs.py prompt 直接讀 key，不可合 |
| 2026-05-09 | 嘗試移除 external_signals_summary | context_compressor.py 與 adapter.py 都讀，不可移 |
