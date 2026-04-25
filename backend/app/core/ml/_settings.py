"""ML 模組總開關。

設為 True 即可全面恢復（含 UI、訓練、chat 注入）。
設為 False（預設）則：
- chat.py 的 _inject_ml_prediction 早返，不注入任何 ML 預測
- data_sync.py 跳過 increment_bars + auto_retrain_if_needed
- /api/ml 所有 endpoint 回傳「ML 模組已停用」
- frontend MLPanel 顯示停用橫幅

歷史背景：
- 2026-04-25 audit 結果：predictions.db 236 筆預測中 0 筆 ml_enhanced=1，
  顯示 ML pipeline 從未真正接通到 chat 介面
- 「文字預測」64% 命中率已是高 baseline，ML 預期收益不明顯
- 決定降級保留程式碼但停用，未來重新評估時可快速重啟

要重新啟用，將以下改為 True 並重啟後端即可：
"""

ML_ENABLED = False
