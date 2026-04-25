# 研究專案目錄

獨立於主系統的實驗性研究，**不影響 production 程式碼**。

## 子專案

### lstm_sca_arima_garch/
混合模型 PoC：LSTM + SCA + ARIMA + GARCH 用於價格預測。

執行方式：
```bash
cd backend/research/lstm_sca_arima_garch
pip install -r ../requirements_research.txt   # 第一次需要
python3 run_experiment.py                      # 跑單次實驗
```

結果輸出於 `lstm_sca_arima_garch/results/{timestamp}/`。

## 與主系統的關係

- 完全獨立：不被 main.py 載入、不掛 API endpoint
- 共用既有資料：透過 `data_loader.py` 讀 `backend/data/ohlcv/*.csv`（read-only）
- 不寫入 production DB
- 主系統照常運作，不受研究實驗影響

## 額外依賴

主系統依賴維持精簡。研究專案的額外套件（`arch`、`torch`）寫在 `requirements_research.txt`，
僅在跑研究時才需要安裝。
