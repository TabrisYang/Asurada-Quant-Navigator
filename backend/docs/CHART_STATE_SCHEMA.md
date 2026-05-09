# chart_state 結構規範

## 目的

`chart_state` 是系統的「狀態總線」，每次分析都會傳給 LLM。**v118-v120 期間因為不斷往內塞欄位導致 round2 prompt 膨脹到 200-300KB、引發 stream 中斷**（見 [witty-scribbling-fog.md](../../...
/.claude/plans/witty-scribbling-fog.md) Context 段）。

本文件規範 `chart_state` 結構，**任何新增/移除欄位必須先更新此文件**。pre-commit hook 會檢查此文件與實際注入點的一致性。

任何 AI 工具看到此檔案：**修改 chart_state 結構前必讀**。

---

## 欄位總覽

目前 chart_state 共 **26 個後端注入欄位 + 11 個前端送來欄位**，總計 ~37 個 key。

任何新增需走 schema 變更流程（見最末段）。

---

## 前端送來欄位（11 個）

| 欄位 | 來源 | 用途 | 大小 | 進 Round 2? |
|---|---|---|---|---|
| `symbol` | [chartStore:443](../../frontend/src/stores/chartStore.ts) | 標的識別 | <50B | ✅ 必要 |
| `timeframe` | chartStore:444 | 時間級別 | <10B | ✅ 必要 |
| `startDate` / `endDate` | chartStore:445-446 | 數據範圍 | <50B | ✅ 必要 |
| `dataPoints` | chartStore:447 | K 線數量 | <10B | ✅ 必要 |
| `activeIndicators` | chartStore:448-452 | 已啟用指標清單 | ~1KB | ❌ 精簡（只留名稱） |
| `annotationCount` | chartStore:453 | 圖表標記數 | <10B | ✅ 必要 |
| `priceOverview` | chartStore:467-476 | 當前價 / 區間高低 / 漲跌幅 | ~200B | ✅ 必要 |
| `recentCandles` | chartStore:483-489 | 最近 N 根 K 線 | 5-15KB | ❌ 精簡（前端已送） |
| `indicatorValues` | chartStore:493-522 | 最近 5 根指標值 + 趨勢 | 2-10KB | ❌ 精簡 |
| `factorScanSummary` | chartStore:530（可選） | 因子掃描結果摘要 | <1KB | ✅ 必要 |

---

## 後端注入欄位（26 個）

### A. Cache 層（內部使用，**不進 LLM**，2 個）

| 欄位 | 注入位置 | 用途 | 進 LLM? |
|---|---|---|---|
| `_cached_df` | [chat.py:353](../app/api/routes/chat.py#L353) | 快取 OHLCV DataFrame | ❌ 內部用 |
| `_cached_df_key` | chat.py:354 | DataFrame 快取 key | ❌ 內部用 |

**規則**：所有 `_` 開頭欄位都不應序列化進 LLM prompt。adapter.py 應跳過。

### B. 市場狀態（5 個）

| 欄位 | 注入位置 | 消費者 | 進 Round 2? |
|---|---|---|---|
| `data_availability` | chat.py:359 | LLM prompt | ✅ |
| `decomposition` | chat.py:374 | LLM prompt（STL 分解） | ✅ |
| `currentRegime` | chat.py:382 | **fact_checker + prompt + 多處** | ✅ 必要 |
| `regimeWarning` | chat.py:384 | LLM prompt（低信心警示） | ✅ |
| `regime_subtype` | chat.py:488 | LLM prompt（ranging 細分） | ✅ |

### C. 外部訊號（2 個）

| 欄位 | 注入位置 | 消費者 | 進 Round 2? |
|---|---|---|---|
| `external_signals` | chat.py:405 | LLM prompt（funding/OI/sentiment 詳情） | ❌ 精簡（round2 改用 summary） |
| `external_signals_summary` | chat.py:406 | **context_compressor.py + adapter.py:83** | ✅（round2 用） |

**注意**：兩者都不可移除。`_summary` 是 round2 精簡版本。

### D. 事件 / 籌碼（4 個）

| 欄位 | 注入位置 | 消費者 | 進 Round 2? |
|---|---|---|---|
| `upcoming_events` | chat.py:425 | LLM prompt | ✅ |
| `calendar_meta` | chat.py:431 | LLM prompt（行事曆過期警示） | ✅ |
| `crossStockSignals` | chat.py:445 | LLM prompt（族群廣度） | ✅ |
| `social_sentiment` | chat.py:457 | LLM prompt（Reddit/CryptoPanic） | ✅ |

### E. 組合 / 倉位（2 個）

| 欄位 | 注入位置 | 消費者 | 進 Round 2? |
|---|---|---|---|
| `portfolio_summary` | chat.py:471 | LLM prompt（組合風控） | ✅ |
| `bilateral_plan` | chat.py:534 | LLM prompt（雙向計劃 ranging 用） | ✅ |

### F. Donchian 三件套（3 個）

| 欄位 | 注入位置 | 消費者 |
|---|---|---|
| `donchian_position_pct` | chat.py:508 | **fact_checker.py:244** + **function_defs.py 多處 prompt** |
| `donchian_upper` | chat.py:509 | function_defs.py prompt |
| `donchian_lower` | chat.py:510 | function_defs.py prompt |

**禁止合併**：fact_checker 與 prompt 直接讀 key，合併會 break 驗證與 LLM 解讀。

### G. 預測 / ML（4 個）

| 欄位 | 注入位置 | 消費者 | 進 Round 2? |
|---|---|---|---|
| `historical_insights` | chat.py:551 | LLM prompt | ✅ |
| `active_alerts` | chat.py:624 | LLM prompt（自動掃描預警） | ✅ |
| `mlPrediction` | chat.py:706 | LLM prompt + fact_checker | ✅ |
| `rl_strategic_insight` | chat.py:994 | LLM prompt（v101 imitation learning） | ✅ |

### H. recent_accuracy 統計（1 個 + 3 sub）

| 欄位 / 子欄位 | 注入位置 | 消費者 |
|---|---|---|
| `recent_accuracy` | chat.py:806 | LLM prompt + 多處 |
| `recent_accuracy.regime_warning` | chat.py:838 | v118 三道防線之一 |
| `recent_accuracy.direction_balance` | chat.py:859 | v118 三道防線之一 |
| `recent_accuracy.signal_history` | chat.py:911 | v120.5 訊號組合命中率 |

**規模警示**：`signal_history` 在 v120.5 加入後每 request 多 10-24KB（這是 stream 中斷的根因）。Round 2 必須精簡，見 [adapter.py:_minimal_r2_chart_state](../app/core/llm/adapter.py)。

---

## Round 2 精簡規則

[adapter.py:_minimal_r2_chart_state](../app/core/llm/adapter.py) 將 chart_state 精簡到 < 1KB 進 round 2，規則：

**保留**（必要）：
- symbol, timeframe, currentPrice, currentRegime
- recent_accuracy 的 top-level 摘要（win_rate / total_predictions）
- recent_accuracy 內三個 summary line（regime_warning / direction_balance / combo_stats 各一行）
- external_signals 兩個關鍵數值（funding_rate_pct, fear_greed_value）

**移除**（已在 Round 1 提供）：
- indicators 完整 JSON
- recentCandles
- signal_history 詳細 single_signal_stats
- external_signals 完整詳情（macro / orderbook / etf 等）
- recent_accuracy.regime_warning / direction_balance 細節

預期效果：round 2 prompt 從 200-300KB → 30-50KB。

---

## 大小監控

[adapter.py](../app/core/llm/adapter.py) 在 `_build_system_message` 結尾自動 log 當 sys prompt > 50KB：

```python
[prompt_size] sys=XX.XKB chart_state=X.XKB r2_mode=True/False
```

**正常數值**：
- Round 1：sys 50-80KB（含完整 system prompt + chart_state）
- Round 2：sys 50-65KB（system prompt 不變，chart_state 精簡到 < 1KB）

**異常警示**：sys > 100KB 必須調查 chart_state 是否再次膨脹。

---

## 新增 / 修改欄位流程

任何 AI 工具或工程師若需要新增 chart_state 欄位：

1. **先讀此文件**，確認是否真的需要新欄位（很多時候已有同類欄位可重用）
2. **更新此文件**：在對應分類下新增條目，含「注入位置」「消費者」「進 Round 2?」
3. **更新 [adapter.py:_minimal_r2_chart_state](../app/core/llm/adapter.py)**：明確標記新欄位是否進 Round 2
4. **跑 ETH/USDT 全部分析**驗證 round 2 prompt size 沒超過 80KB
5. **commit message** 含「chart_state: 新增 X 欄位（用途：Y、進 round2: Z）」

**禁止繞過此流程直接 push 到 chat.py。**

---

## 移除欄位流程

任何 AI 工具或工程師若要移除 chart_state 欄位：

1. **不要假設「看起來廢棄」就可以移除**（見 [SHADOW_FEATURES.md](SHADOW_FEATURES.md) 教訓）
2. **必跑 grep 全引用**：
   ```bash
   grep -rn "FIELD_NAME" backend/app --include="*.py"
   grep -rn "FIELD_NAME" frontend/src --include="*.ts" --include="*.tsx"
   grep -rn "FIELD_NAME" backend/app/core/llm/function_defs.py
   ```
3. **檢查 fact_checker.py** 是否在驗證該欄位
4. **檢查 prompt 模組**（function_defs.py）是否提示 LLM 讀該欄位
5. **所有引用點同步處理** + 跑端到端測試
6. **更新此文件**：刪除對應條目，commit message 含「chart_state: 移除 X 欄位」

---

## 變更紀錄

| 日期 | 變更 | 緣由 |
|---|---|---|
| 2026-05-09 | 建立此文件 | v118-v120 累積疊加 chart_state 欄位導致 stream 中斷後的治理舉措 |
