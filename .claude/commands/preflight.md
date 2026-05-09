---
description: 量化系統改動前的 pre-flight check（跑 shadow mode + 解讀報告）
---

執行量化系統改動前的安全檢查。

# 步驟

1. **跑 shadow mode**：
   `python3 backend/scripts/shadow_mode.py --days 90` （全 symbol）
   或 `python3 backend/scripts/shadow_mode.py --days 90 --symbol $ARGUMENTS` （指定 symbol）

2. **讀取最新 shadow_report**：`backend/data/db/shadow_report_*.json`，挑日期最新的

3. **報告以下五項**：

   **(a) 樣本充足度**
   - `summary.sample_sufficiency.rows_with_full_samples` ≥ 10 才算可靠
   - 若 <10：警告「樣本不足，目前任何 A/C/F 改動都缺乏驗證依據」

   **(b) prediction_tracker 健康度**
   - 總筆數 = `summary.total`，至少要 ≥ 30 才有統計意義
   - 各 regime 樣本是否平均（看 `by_regime` 各 n）

   **(c) 「看漲說漲」量化判定**
   - `summary.看漲說漲_quantification.verdict`：
     - `BUG`（順勢期平均報酬 < -2%）→ ✅ A 改動有依據，可進入階段 1
     - `ALPHA`（順勢期平均報酬 > +2%）→ ❌ A/C 都不該做，只保留 F
     - `GREY_ZONE`（-2% ~ +2%）→ ⚠️ 仍可進入階段 1，但 A 觸發門檻調嚴

   **(d) A/C/F 觸發頻率**
   - A：應 < 15%（過高代表三防線太鬆，會誤觸正常 case）
   - C：< 30%（C 設計上是「ranging + 偏向」，27% 上下合理）
   - F：< 25%（biased_long/short 觸發頻率）

   **(e) A 觸發 case 表現**
   - 若有 ≥5 筆觸發 → 看 hit_rate
     - hit_rate < 50%：A 防護有效，這些 case 確實該被攔
     - hit_rate ≥ 50%：A 觸發條件太鬆，需提高樣本門檻

4. **給出最終建議**：
   - 「✅ 改動安全，可進入階段 1」
   - 「⚠️ 樣本不足，需先收集 N 週資料再說」
   - 「❌ 數據顯示「看漲說漲」是 alpha，停止 A/C 計劃」

# 觸發此命令的時機

- 動 `regime_filter.py`、`regime_subtype.py`、`function_defs.py` 決策卡規則段、`indicators/`、`scenario_predictor.py`、`prediction_tracker.py` 之前
- 修改 A/C/F 三項規則的觸發門檻之前
- 每次 release 前快檢

# 引用

詳細規則見 [CLAUDE.md](../../CLAUDE.md) 「量化系統改動原則」段落。
