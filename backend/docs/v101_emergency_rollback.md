# v101 緊急回退 SOP（5 分鐘內完成）

> 出問題時快速回到 v100 完整可用狀態。優先級：**user 體驗 > 系統完整性 > 資料保留**。

---

## Level 1：Feature Flag 關閉（30 秒）

最快、最安全的回退。改設定就好，不動程式碼，不影響資料。

```bash
# 編輯 settings 或 .env
vim /Users/tonyy/阿斯拉量化系統V2/backend/.env

# 加入 / 改成：
IMITATION_LEARNING_ENABLED=false
IMITATION_SHADOW_MODE=true
IMITATION_CANARY_PCT=0

# 重啟後端
cd /Users/tonyy/阿斯拉量化系統V2/backend
pm2 restart all  # 或 .venv/bin/python -m uvicorn app.main:app --reload
```

**確認**（1 分鐘）：
1. 開啟 frontend 跑一次「全部分析」
2. 確認沒有 `🤖 RL 戰略結論` 段落
3. 結論卡 9 欄位完整顯示（v100 行為）

✅ **若恢復 → 留在這個狀態，慢慢 debug v101**。

---

## Level 2：Git Rollback（2 分鐘）

Level 1 沒救，可能是 v101 程式碼破壞了 v100 路徑。

```bash
cd /Users/tonyy/阿斯拉量化系統V2

# 查看可用 tag
git tag -l | grep v10

# 回到 v100.0（v101 改動前）
git checkout v100.0

# 重啟
cd backend
pm2 restart all
```

**確認**：跑「全部分析」確認 v100 行為。

⚠️ **此時你會回到 v101 改動前的程式碼狀態**。要繼續開發 v101 時：
```bash
git checkout main  # 回到最新 main
```

---

## Level 3：DB Rollback（極端情況，3 分鐘）

只在 Level 2 還壞才做，極少需要（v101 嚴格純加法應永遠安全）。

```bash
# 先停服務
cd /Users/tonyy/阿斯拉量化系統V2/backend
pm2 stop all

# 找最近的快照
ls data/db_pre_v101_*

# 還原（替換 YYYYMMDD 為實際日期）
cp -r data/db_pre_v101_YYYYMMDD/* data/db/

# 重啟
pm2 restart all
```

⚠️ **此操作會丟失快照之後的所有對話 / 預測 / 蒸餾資料**。只在系統完全壞掉時用。

---

## 確認回退成功的 Checklist

```
☐ 「全部分析」可以跑（5 秒內回應）
☐ v100 結論卡 9 欄位完整顯示
☐ Calibration 面板可開啟（PredictionDashboard）
☐ 圖表「歷史預測」toggle 可開關
☐ 沒有「🤖 RL 戰略結論」段落（Level 1+ 的預期）
☐ Backend log 無 ERROR（tail -f data/db/launchd_*.log）
```

---

## 各 Level 觸發時機判斷

| 症狀 | Level | 為什麼 |
|---|---|---|
| 報告變慢 / 額外段落怪怪的 | 1 | flag 關掉 v101 邏輯 |
| 「全部分析」直接 500 error | 1 → 2 | 先試 flag，沒救再回 git |
| Backend 無法啟動 | 2 | 程式碼層出問題 |
| DB schema migration 錯誤 | 3 | 純加法理論上不該發生，極端 |
| user 體驗異常但功能正常 | 1 | flag 即可 |

---

## 預防勝於治療

### 部署前 checklist

```
☐ 所有 regression tests 通過：cd backend && .venv/bin/pytest tests/test_v100_regression.py
☐ 手動跑 1 次「全部分析」確認 v100 行為（在 dev 環境）
☐ DB 快照已建立：cp -r data/db data/db_pre_v101_$(date +%Y%m%d)
☐ 所有新 feature flag 預設 OFF（settings.py 確認）
☐ Git tag 已打：git tag v101-phaseN
☐ 心智狀態 OK（不要疲勞時 deploy）
☐ 部署時間在週二 / 週三早上（不是週五，不是月底）
```

### 部署後第一週

```
☐ 設定異常 email 通知
☐ 每天看一次 backend log（grep ERROR / WARN）
☐ 觀察 quality_gate_log 表的記錄
☐ 若 user 投訴 → 立刻走 Level 1
```

---

## 聯絡

系統管理員：tabris212@gmail.com
緊急時也可手動關閉所有 flag（Level 1）後再聯絡。
