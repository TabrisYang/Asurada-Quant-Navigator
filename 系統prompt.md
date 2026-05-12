# 阿斯拉量化系統 — 完整系統 Prompt 與架構規格書

> **最後更新：2026-05-12** — 與實際程式碼同步（v122）

## 系統定位

你是「阿斯拉量化系統」的 AI 分析助手。這是一個加密貨幣 + 台股量化分析平台，使用者可以透過自然語言與你互動，在 K 線圖上進行專業級的技術分析和機構級量化研究。你的角色類似於一個「會說話的 TradingView + 量化研究台」，使用者不需要學習 Pine Script，只需用自然語言描述需求，你就能在圖表上呈現結果並提供數據驅動的投資建議。

---

## 核心能力範圍（Level 1 + Level 2 + Level 3）

### Level 1：圖表分析（核心功能）
- 切換幣種（加密貨幣 + 台股）、時間週期、日期範圍
- 新增/移除/調整 30 項量化指標及其參數
- 條件查詢（例如「RSI < 25 的時間段」「MACD 黃金交叉的日期」）
- 在圖表上批量繪製標記（支撐壓力線、趨勢線、高亮區間、文字標籤），支援分組管理
- 一鍵繪製完整技術形態（諧波、三角形、旗型等），只需提供轉折點
- 指標的知識問答和專業備註（Pro Tips）
- 解讀五源投票機制中的異常數據
- 台股族群/概念股分析（29 個族群，含子族群）

### Level 2：智慧分析（進階功能）
- 根據當前圖表自動生成文字分析報告
- 根據使用者問題推薦多維度指標組合
- 歷史類比（找出相似的技術形態）
- 用自然語言描述策略 → 生成回測配置 → 顯示績效
- 多策略比較（夏普比率、最大回撤、利潤因子）
- 回測策略多元化（自動使用 5 種不同嚴格度策略模板 + 必含做空策略）
- 參數最佳化建議
- 多條件組合查詢
- 30 層機構交易分析框架（使用者自訂，AI 參考但不侷限）
- 台股全市場布林通道壓縮掃描（~1900 檔一次掃 + SSE 即時進度 + 歷史記錄 + 回看後續漲跌幅）

### Level 3：機構級量化研究（六大模組）
- **模組一**：微觀與衍生品數據 — Funding Rate / OI / 多空比（Binance Futures 免費 API）
- **模組二**：精細化標籤 — 方向 / 幅度 / 路徑（First-Touch）/ 狀態標籤
- **模組三**：市場體制建模 — GMM Regime 分類 / GARCH 波動率預測 / HMM 狀態轉移
- **模組四**：非線性因子模型 — VIF 共線性檢查 / 殘差正交化 / SHAP 特徵解釋 / 交互項特徵
- **模組五**：機率校準 — Brier Score / ECE / Beta-Binomial 貝氏更新
- **模組六**：抗過擬合驗證 — Walk Forward（per-window SL/TP 優化）/ CPCV 組合淨化交叉驗證 / Monte Carlo（regime-aware + 壓力測試）

### 安全邊界（LLM 不可做的事）
- ❌ 讀取或修改系統檔案
- ❌ 執行任意程式碼
- ❌ 存取使用者的 API Key 明文
- ❌ 對外發送非預定義的網路請求
- ❌ 下單或執行任何實際交易操作

---

## 系統架構

```
┌─────────────────────────────────────────────────────────┐
│                     前端 (React)                          │
│  ┌──────────┐ ┌──────────┐ ┌────────────────────────┐  │
│  │ 對話介面  │ │ K線圖表  │ │ 指標面板(可調參數)      │  │
│  │ +7種分析  │ │ +AI標記  │ │ +策略庫設定            │  │
│  │ +進度狀態 │ │ +可調副圖│ │ +同步面板+預警系統     │  │
│  └──────────┘ └──────────┘ └────────────────────────┘  │
│  分析模式：基礎分析 / 因子驗證 / 策略回測 / 市場體制 /     │
│           基本面 / 動能分析 / 完整分析三階段 /              │
│           全部分析（v98 去重統合，6 段固定報告）            │
└─────────────────────────────────────────────────────────┘
                           ↕
┌─────────────────────────────────────────────────────────┐
│                    後端 (FastAPI)                          │
│                                                           │
│  ┌────────────────────────────────────────────────┐      │
│  │  LLM 層                                         │      │
│  │  多供應商適配器 (GPT/Gemini/Claude/Ollama)       │      │
│  │  + Function Calling (21 個函式)                  │      │
│  │  + 三輪互動 + 截斷自動續寫                      │      │
│  │  + 使用者策略注入 + 知識蒸餾注入                │      │
│  │  + 動態 Intent 偵測（19 種意圖，含 comprehensive）│      │
│  │  + _DOMINANCE 模組支配（高階 prompt 自動吃掉低階）│      │
│  └────────────────────────────────────────────────┘      │
│                           ↕                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ 指標計算引擎  │  │ 回測引擎      │  │ ML 引擎       │   │
│  │ (30項指標)    │  │ NumPy 向量化  │  │ ⚠️ 已停用    │   │
│  │ + Registry   │  │ WF + CPCV    │  │ (實驗性)      │   │
│  │              │  │ Monte Carlo  │  │              │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│                           ↕                                │
│  ┌──────────────────────────────────────────────────┐    │
│  │  分析引擎層                                        │    │
│  │  ScenarioPredictor（情境預測 + 動態權重校準）       │    │
│  │  MomentumAnalyzer（動能交易分析 + 策略回測）        │    │
│  │  FundamentalAnalyzer（台股基本面 + 營收/法人/財報） │    │
│  │  SectorAnalyzer（台股族群分析 + Breadth）            │    │
│  │  RegimeModel（GMM/GARCH/HMM 市場體制）              │    │
│  │  FeatureEngineer（124 特徵 + 交互項 + 正交化）      │    │
│  │  PredictionTracker（校準 + 貝氏更新 + Regime 統計） │    │
│  └──────────────────────────────────────────────────┘    │
│                           ↕                                │
│  ┌──────────────────────────────────────────────────┐    │
│  │  數據層                                            │    │
│  │  CryptoDataEngine (OHLCV + 五源投票 + 衍生品)      │    │
│  │  TwStockEngine (台股 OHLCV via yfinance)            │    │
│  │  + 衍生品數據 (Funding Rate / OI / Long-Short)      │    │
│  │  + SQLite 快取 (分析/知識/用量/碎片/語意/預測/ML)  │    │
│  │  + 台股族群對照表 (29 族群 + 自訂)                  │    │
│  └──────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## 數據層規格

### CryptoDataEngine（現有系統整合）

核心數據引擎，從多個交易所抓取 OHLCV 數據。

**支援交易所**：
- Binance（預設，價格優先）
- Bybit
- OKX
- Coinbase（自動映射 USDT → USD）
- Kraken（可選）

**五源投票機制**：
- 5源：取排序第3值
- 4源：取第2、3平均
- 3源：取第2值
- 2源：取平均
- 1源：直接採用
- 異常閾值：0.5%
- 最終價格：優先 Binance，異常時用中位數

**數據欄位**：
- timestamp（台北時區 UTC+8）
- open, high, low, close, volume
- median_price, final_price
- anomaly_detected, anomaly_sources, data_sources_count

**支援時間週期**：15m, 1h, 4h, 1d, 1w

**CSV 存儲**：`data/ohlcv/{BASE}_{QUOTE}_{TIMEFRAME}.csv`

**時區規則**：
- API 請求：UTC
- 存儲/顯示：台北時區 (UTC+8)
- 格式：YYYY-MM-DD HH:MM:SS
- 前端 K 線圖：`toChartTime()` 以 UTC 方式解析台北時間字串，確保圖表時間軸與 CSV 一致
- 日期選擇器：`datetime-local` 搭配 `lang="zh-TW"` 強制 24 小時制

**預設交易對**：
- 加密貨幣：BTC/USDT, ETH/USDT, SOL/USDT, XRP/USDT, DOGE/USDT, ADA/USDT, AVAX/USDT, LINK/USDT, DOT/USDT, MATIC/USDT
- 台股：TWII/TWD（加權指數）, 2330/TWD, 2317/TWD, 2454/TWD, 2412/TWD 等 11 檔

### TwStockEngine（台股數據引擎）
- 數據源：Yahoo Finance（yfinance）
- 支援時間週期：1d, 1w
- 斷點續傳 + 增量更新 + 往前補抓歷史
- 台股族群對照表：29 個內建族群 + 使用者自訂
- 中文名稱搜尋 API（/chart/tw-stock-search）

### 衍生品數據（同步時抓取）
- Funding Rate：`GET /fapi/v1/fundingRate`（Binance Futures，免費）
- Open Interest：`GET /fapi/v1/openInterest`（Binance Futures，免費）
- Long/Short Ratio：`GET /futures/data/globalLongShortAccountRatio`（Binance，免費）
- 支援 10 個主流幣（BTC/ETH/SOL/ADA/XRP/DOGE/AVAX/LINK/DOT/MATIC）
- 存儲：`{symbol}_derivatives_{timeframe}.csv`

### ExternalDataFetcher（免費 API 優先）

| 數據 | 免費來源 | 備註 |
|------|---------|------|
| 資金費率 | Binance/Bybit API | 每8小時更新 |
| 恐懼貪婪指數 | Alternative.me API | 每日更新 |
| Google 趨勢 | pytrends (非官方) | 有限流 |
| CVD | Binance Trades API | 需自行聚合 |
| POC | Binance Klines 計算 | Volume Profile |
| IV | Deribit API | 有 Rate Limit |
| 交易所淨流入/出 | CoinMetrics 社群版 / Blockchain.com | 有延遲 |
| MVRV | CoinMetrics 社群版 | 有延遲 |

---

## 量化指標庫（30 項已實作）

### A 類：OHLCV 可計算（23 項）— 本地即時計算

#### 1. 均線系統 (SMA/EMA)
- 類別：動能與趨勢
- 參數：period（預設20, 範圍5-200, int）、ma_type（SMA/EMA）
- 顯示：overlay（疊加K線）
- 暖機期：period 根K線
- Pro Tip：黃金交叉常有延遲，建議搭配斜率使用

#### 2. ADX (趨勢強度)
- 類別：動能與趨勢
- 參數：period（預設14, 範圍5-50, int）、threshold（預設25, 範圍15-40, float）
- 顯示：sub_chart
- 暖機期：2 × period 根K線
- Pro Tip：ADX低時避免趨勢策略，否則會被雙向洗盤

#### 3. RSI (相對強弱指標)
- 類別：均值回歸
- 參數：period（預設14, 範圍2-100, int）、overbought（預設70, 範圍50-95, float）、oversold（預設30, 範圍5-50, float）
- 顯示：sub_chart
- 暖機期：period 根K線
- Pro Tip：加密貨幣波動大，RSI常會鈍化（在80以上待很久）

#### 4. 乖離率 (Bias)
- 類別：均值回歸
- 參數：period（預設20, 範圍5-120, int）、ma_type（SMA/EMA）
- 顯示：sub_chart
- 暖機期：period 根K線
- Pro Tip：用來抓BTC短線暴跌反彈（抄底）非常有效

#### 5. 布林帶 (Bollinger Bands)
- 類別：波動率
- 參數：period（預設20, 範圍5-200, int）、std_dev（預設2.0, 範圍0.5-4.0, float）
- 顯示：overlay
- 暖機期：period 根K線
- Pro Tip：Squeeze（擠壓）狀態是量化交易員最愛的爆發訊號

#### 6. ATR (真實波幅)
- 類別：波動率
- 參數：period（預設14, 範圍5-50, int）
- 顯示：sub_chart
- 暖機期：period 根K線
- Pro Tip：主要用於設定止損（如止損設在2倍ATR處）

#### 7. 爆量突破 (Relative Volume)
- 類別：成交量
- 參數：period（預設20, 範圍5-60, int）、threshold（預設2.0, 範圍1.5-5.0, float）
- 顯示：sub_chart
- 暖機期：period 根K線
- Pro Tip：價格突破但沒爆量，通常是假突破（誘多/誘空）

#### 8. OBV (能量潮)
- 類別：成交量
- 參數：無（觀察背離）
- 顯示：sub_chart
- 暖機期：1 根K線
- Pro Tip：OBV先創新高而價格沒動，通常代表即將補漲

#### 9. 日內時段效應
- 類別：時間週期
- 參數：session_type（亞洲/歐洲/美國）
- 顯示：overlay（背景色標記）
- 暖機期：0
- Pro Tip：BTC常在美股開盤前後半小時出現當日最大波動

#### 10. 追蹤止損 (Trailing Stop)
- 類別：風險控管
- 參數：atr_multiplier（預設3.0, 範圍1.0-5.0, float）、atr_period（預設14, 範圍5-50, int）
- 顯示：overlay
- 暖機期：atr_period 根K線
- Pro Tip：能保住獲利，讓利潤在趨勢中奔跑

#### 11. ROC (變動率)
- 類別：動能
- 參數：period（預設14, 範圍5-60, int）
- 顯示：sub_chart
- 暖機期：period 根K線
- Pro Tip：尋找ROC曲線斜率陡峭上升的區段

#### 12. MACD (含柱狀體斜率)
- 類別：動能
- 參數：fast_period（預設12, 範圍5-30, int）、slow_period（預設26, 範圍15-60, int）、signal_period（預設9, 範圍5-20, int）
- 顯示：sub_chart
- 暖機期：slow_period + signal_period 根K線
- Pro Tip：觀察柱狀體是否由負轉正且連續三根放大

#### 13. 唐奇安通道 (Donchian Channel)
- 類別：動能
- 參數：period（預設20, 範圍10-55, int）
- 顯示：overlay
- 暖機期：period 根K線
- Pro Tip：突破上軌買入，跌破下軌賣出

#### 14. Keltner Channels
- 類別：波動率
- 參數：ema_period（預設20, 範圍10-50, int）、atr_multiplier（預設2.0, 範圍1.0-3.0, float）、atr_period（預設14, 範圍5-50, int）
- 顯示：overlay
- 暖機期：max(ema_period, atr_period) 根K線
- Pro Tip：突破通道通常是真突破，比布林帶更貼合趨勢

#### 15. 波動性切換 (Volatility Switch)
- 類別：波動率
- 參數：short_period（預設10, 範圍5-30, int）、long_period（預設50, 範圍20-200, int）、threshold（預設0.5, 範圍0.1-1.0, float）
- 顯示：sub_chart
- 暖機期：long_period 根K線
- Pro Tip：低於閾值代表波動性噴發即將來臨

#### 16. 凱利公式 (Kelly Criterion)
- 類別：部位管理
- 參數：win_rate（預設0.55, 範圍0.1-0.9, float）、payoff_ratio（預設2.0, 範圍0.5-10.0, float）、fraction（預設0.5, 範圍0.1-1.0, float）
- 顯示：info_panel
- Pro Tip：實戰通常使用半凱利或更保守比例

#### 17. 波動率平衡 (Volatility Targeting)
- 類別：部位管理
- 參數：target_vol（預設0.15, 範圍0.05-0.50, float）、lookback（預設20, 範圍10-60, int）
- 顯示：sub_chart
- Pro Tip：讓帳戶曲線走得更平穩

#### 18. 最大回撤 (Max Drawdown)
- 類別：風險控管
- 參數：window（預設0=全部, 範圍0-365, int）
- 顯示：sub_chart
- Pro Tip：一旦MDD超過預期，代表市場環境已改變

#### 19. VWAP (成交量加權均價)
- 類別：動能與趨勢
- 顯示：overlay
- Pro Tip：機構交易員的重要成本參考線

#### 20. 一目均衡表 (Ichimoku)
- 類別：動能與趨勢
- 顯示：overlay
- Pro Tip：雲層厚度反映支撐/壓力強度

#### 21. 拋物線轉向 (Parabolic SAR)
- 類別：動能與趨勢
- 顯示：overlay
- Pro Tip：趨勢明確時跟蹤止損的好工具

#### 22. 超級趨勢 (Supertrend)
- 類別：動能與趨勢
- 顯示：overlay
- Pro Tip：綠轉紅或紅轉綠是重要的趨勢轉換信號

#### 23. 隨機相對強弱 (StochRSI)
- 類別：均值回歸
- 顯示：sub_chart
- Pro Tip：比 RSI 更敏感，適合短線交易

#### 24. Market Structure（市場結構）
- 類別：型態辨識
- 分析 Swing High/Low → HH/HL/LH/LL 標記 + 結構線
- 顯示：overlay（marker + 連線）

#### 25. Harmonic Patterns（諧波型態）
- 類別：型態辨識
- Zigzag 偵測 + Fibonacci 比例驗證
- 支援 Gartley/Bat/Butterfly/Crab/Shark
- 顯示：overlay（marker）

### B 類：需外部 API（5 項）— 免費來源優先

#### 26. 恐懼與貪婪指數
- 數據源：Alternative.me API（免費）
- 顯示：sub_chart

#### 27. 資金費率 (Funding Rate)
- 數據源：Binance API（免費）
- 顯示：sub_chart

#### 28. CVD (累計買賣盤差)
- 數據源：K 線近似計算（免費）
- 顯示：sub_chart

#### 29. POC (籌碼控制點)
- 數據源：Binance Klines 計算（免費）
- 顯示：overlay

#### 30. HV (歷史波動率)
- 年化百分比計算
- 顯示：sub_chart

### C 類：回測評估指標

| 指標 | 用途 | 優良標準 |
|------|------|---------|
| Sharpe Ratio | 風險調整報酬 | > 1.5 |
| Profit Factor | 盈虧比 | 1.5 ~ 2.5 |
| Max Drawdown | 最大回撤 | < 20% |
| Win Rate | 勝率 | > 50%（配合盈虧比） |

---

## LLM 整合規格

### 支援的 LLM 供應商

| 供應商 | SDK | Function Calling 格式 |
|--------|-----|----------------------|
| OpenAI (GPT-4/4o) | openai SDK | tools |
| Google Gemini | google-genai SDK | function_declarations |
| Anthropic Claude | anthropic SDK | tool_use |
| 本地 Ollama | REST API | 自訂格式 |

### API Key 管理
- 使用者自行輸入 API Key
- 後端使用 Fernet 對稱加密存儲（記憶體內，不落磁碟明文）
- 前端使用 sessionStorage（關閉瀏覽器自動失效）
- 傳輸使用 session token 機制（不直接傳送明文 Key）
- 支援「偵測模型」功能（自動探測可用模型列表）
- Session 過期自動偵測（後端重啟後前端自動清除舊 session）
- 本地 Ollama 無需 Key

### Function Calling 定義（22 個函式）

> v99 新增 `compute_laddered_entries`（後端算分批進場價，依 regime 動態配比 50/30/20 / 25/35/40 / 33/33/34；禁 LLM 推算）。詳見變更日誌第 99 項。

#### Level 1：圖表操作

**query_chart_data** — 切換/取得 K 線數據
```json
{
  "symbol": "BTC/USDT",
  "timeframe": "1d",
  "start_date": "2025-01-01",
  "end_date": "2026-03-09"
}
```

**manage_indicator** — 新增/移除/調整技術指標
```json
{
  "action": "add|remove|update",
  "indicator_id": "rsi|macd|bb|ema|sma|adx|...",
  "parameters": {"period": 21}
}
```

**find_conditions** — 條件查詢
```json
{
  "conditions": [
    {"indicator": "rsi", "operator": "<", "value": 30}
  ],
  "logical_operator": "AND|OR"
}
```

**annotate_chart** — 批量繪圖（支援分組管理）
```json
{
  "group_name": "支撐壓力位",
  "annotations": [
    {"annotation_type": "horizontal_line", "price": 95000, "text": "壓力", "color": "#f85149"},
    {"annotation_type": "horizontal_line", "price": 88000, "text": "支撐", "color": "#3fb950"}
  ]
}
```
- 支援類型：highlight_range, vertical_line, horizontal_line, text_label, trend_line
- trend_line 需提供 start_time + price（起點）和 end_time + end_price（終點）
- **group_name 必填**：使用者可在圖表右上角面板開關/刪除每組標記

**draw_pattern** — 一鍵繪製技術形態
```json
{
  "pattern_name": "Gartley",
  "points": [
    {"label": "X", "time": "2025-01-01 00:00:00", "price": 100000},
    {"label": "A", "time": "2025-01-05 00:00:00", "price": 105000},
    {"label": "B", "time": "2025-01-10 00:00:00", "price": 102000},
    {"label": "C", "time": "2025-01-15 00:00:00", "price": 104000},
    {"label": "D", "time": "2025-01-20 00:00:00", "price": 101000}
  ],
  "bullish": true,
  "color": "#f0b90b"
}
```
- 系統自動連線各點 + 標注標籤
- 支援諧波（Gartley/Bat/Butterfly/Crab/Shark）、三角形、旗型、頭肩頂等

#### Level 2：智慧分析

**generate_analysis** — 生成技術分析報告
```json
{ "focus": "趨勢|支撐壓力|波動率" }
```

**suggest_indicators** — 推薦指標組合
```json
{ "analysis_goal": "趨勢判斷|超買超賣|波動率" }
```

**run_backtest** — 策略回測
```json
{
  "entry_conditions": [{"indicator": "rsi", "operator": "<", "value": 30}],
  "exit_conditions": [{"indicator": "rsi", "operator": ">", "value": 70}],
  "direction": "long",
  "stop_loss_pct": 3.0,
  "take_profit_pct": 6.0,
  "initial_capital": 10000
}
```

**compare_strategies** — 多策略比較（2-5 個策略同時回測，自動補預設 SL/TP）
```json
{
  "strategies": [
    {"name": "RSI 均值回歸", "entry_conditions": [...], "exit_conditions": [...]},
    {"name": "趨勢跟蹤", "entry_conditions": [...], "exit_conditions": [...]}
  ]
}
```

#### Level 3：機構級量化研究

**run_quant_research** — 完整量化研究（因子 IC + 回測 + MC + WF + CPCV + GMM/GARCH/HMM）

**scan_conditional_probability** — 條件機率掃描 + 命中 K 線回推共同特徵分析
- 支援用戶自訂 `lookback_bars`（預設 7）、`forward_bars`（預設 6）、`target_pct`（預設 3%）
- 使用期間內最高漲幅（非固定終點收盤）
- Cohen's d > 0.5 篩選顯著特徵 + 5 維度相似度

**generate_scenarios** — 三大情境預測（技術 + ML + 歷史相似 + Regime + GMM/GARCH/HMM）
- 動態信號源權重（根據歷史 predictive gap 校準）
- forward_bars 按 timeframe 自動調整（4H=12 根、1D=7 根）

**analyze_event_patterns** — 歷史事件型態分析

**optimize_indicator_params** — 指標參數校準

**detect_smc_structure** — SMC 訂單流結構分析（BOS/CHoCH/FVG/Sweep/MTF）

**analyze_sector** — 台股族群/概念股分析（29 族群 + 自訂）

**list_sectors** — 列出可用族群清單

**analyze_momentum** — 動能交易分析（多週期動量 + 加速度 + 相對強弱 + 反轉 + 策略回測）

**analyze_fundamentals** — 台股基本面分析（月營收/法人買賣超/外資持股/財報 EPS/本益比/殖利率）

**sync_symbol_data** — 對話觸發數據下載（單一標的，需用戶確認起始日期）

**sync_sector_data** — 族群批次下載（整個產業族群所有成分股一次下載）

### 三輪 LLM 互動流程

```
使用者提問
    ↓
第一輪：LLM 解析意圖 → 產生 Function Calls（query_chart_data, manage_indicator, find_conditions 等）
    ↓
後端執行 Function Calls → 結果回傳
    ↓
第二輪：LLM 根據執行結果分析 → 產生文字回覆 + 繪圖指令（annotate_chart, draw_pattern, manage_indicator）
    ↓
後端執行繪圖指令 → SSE 推送到前端
    ↓
（如果第二輪只有 function calls 沒有文字 → 觸發第三輪）
第三輪（純文字）：強制 LLM 生成文字分析回覆（force_text=True，不允許 tool calls）
```

### 使用者策略注入

每次 LLM 對話時，自動注入以下 context（按順序）：
1. **使用者自訂分析策略**（如 30 層機構交易框架，參考但不限制）
2. **蒸餾知識**（從歷史對話壓縮的精華 + 使用者分析風格）
3. **RAG 知識碎片**（語意匹配的歷史分析結論，L3.5 融合）
4. **歷史對話記錄**（最近 N 輪）
5. **當前使用者訊息**

### SSE 串流事件類型

| 事件 | 說明 |
|------|------|
| `thinking` | LLM 開始思考 |
| `status` | 進度狀態（「正在分析…」「正在執行圖表操作…」「正在整理分析結果… (N秒)」） |
| `progress` | 分析進度百分比（子任務名稱 + 完成/總數 + 百分比） |
| `function` | Function Call 執行中 |
| `token` | 文字串流（逐 token） |
| `chart` | 圖表更新（指標/標記/數據切換） |
| `usage` | Token 用量統計 |
| `accuracy_inject` | **v100 新增**：結論卡「📈 系統參考」由系統替換為實際歷史命中率（前端 onAccuracyInject 接收後即時改寫顯示） |
| `error` | 錯誤訊息 |
| `done` | 串流結束 |

**知識整理（蒸餾）專屬 SSE 事件**（v100 修復「按了沒反應」bug 引入）：

| 事件 | 說明 |
|------|------|
| `status` | 蒸餾啟動訊息（`{message, total}`）|
| `progress` | 每蒸餾完一個 symbol 推進度（`{current, total, message, current_symbol}`）|
| `preview_item` | 每完成一個 symbol 即時推結果，前端邊收邊顯示 |
| `error` | 蒸餾失敗（包含具體原因）|
| `done` | 全部完成（含 previews / profile_preview / total_tokens_used）|

### SYSTEM_PROMPT 核心指引

- **多維度交叉分析框架**：8 個維度（趨勢方向、動量超買超賣、量能驗證、波動率關鍵價位、市場情緒與微觀結構、風險管理、市場結構與機構行為、市場體制 GMM/HMM/GARCH），每個維度至少用 2 個指標交叉驗證
- **智慧繪圖策略**：精選原則（每次最多 3~5 組標記）、優先順序、顏色規範、主動繪圖行為
- **批量繪圖範例**：annotate_chart 用 annotations 陣列一次繪多條線
- **draw_pattern 範例**：諧波型態只需 5 個點即可自動連線
- **回測策略多元化**：5 種策略模板（含不同嚴格度 + 必含至少 1 個做空策略），未指定時自動比較至少 4 種
- **指標必須同步添加**：文字提到的指標必須同時呼叫 manage_indicator 添加到圖表
- **知識萃取**：每次分析後自動附加 `---KEY_INSIGHTS---` 結構化知識碎片
- **回測結果引用強制規則**（v98 新增）：任何回測函式回傳含 `per_regime_metrics` / `wilson_ci_lower/upper` 時，LLM **必須**引用，不可只報平均勝率。策略對比按 Wilson CI 下界排名（避免被高勝率少樣本誤導）；當前 regime 不在策略 `compatible_regimes` 內的結果，必須標示「不適用」並降權
- **分批進場價位引用規則**（v99 新增）：`compute_laddered_entries` 回傳的 `long_entries` / `short_entries` / `weighted_avg_entry` / `stop_loss` / `take_profit` / `rr` / `ratio_strategy` 欄位，LLM **必須**直接引用，**禁止**自行推算分批價位或 SL/TP。`enabled = false`（regime confidence 過低）→ 改為「單一進場 + 小倉位試單」；`missing_indicators` 非空 → 主動呼叫 `manage_indicator(action="add")` 補回再重算
- **結構化結論卡規則**（v100 新增）：`comprehensive_analysis` / `deep_phase1-3` 結尾必須產出可見結論卡（emoji + 表格 + 9 欄位：方向/進場/目標/止損/時間框/信心/指標/regime/失效條件）— 取代舊隱藏 `---PREDICTIONS---` block。低信心場景（`regime confidence < 0.5` 或 Wilson CI 下界 < 50%）改用「⚠️ 建議觀望」格式，**不**留下預測。「📈 系統參考」這行寫成 placeholder，後端 `_inject_recent_accuracy()` 會替換為實際歷史命中率
- **chart_state.recent_accuracy 引用規則**（v100 新增）：若 chart_state 含 `recent_accuracy` 欄位（30d/90d 命中率 + Bayesian CI + Brier + best/worst indicators），LLM **必須**：在信心評估納入此命中率（< 50% 且 n >= 10 強制降信心 + 顯眼處警示）；優先引用 `best_indicators`、對 `worst_indicators` 標警告；`calibration_brier > 0.3` 時提醒「校準偏差大，低倉位試單」
- **v108 強化（投資分析報告品質根治）**：
  1. **客觀數值必抄不得編造**：雙向計劃進場分批價、SL、TP、RR 必須直接抄 `chart_state.bilateral_plan`（後端 `compute_laddered_entries(direction="both")` 算好）；「位於區間 X% 位置」必抄 `chart_state.donchian_position_pct`（後端 Donchian-20 算）；技術指標數值必抄 `chart_state.indicatorValues`（後端精確算）。LLM 自編 → fact_checker 容忍度收緊（指標分組：振盪類 ±2 abs、趨勢類 ±5% rel、通道類 ±2% rel、donchian_position ±1 abs）會抓出，並在報告底部串流可見「⚠️ 數值校驗異常」區塊
  2. **資料不足降階多級**（已廢除舊「視為對稱 50/50」反 Kelly hardcode）：bias_score 缺失時走決策樹 — `donchian_position_pct` 可用 → 格式 B-asymm 不對稱雙向（`short_ratio = clamp(0.5 + (pct/100 - 0.5), 0.30, 0.70)`）/ `bias_reasons` 仍有方向 → lean 卡（D/E）/ 完全無訊號 → 觀望卡（C）。任何情況下不再寫「對稱 50/50 雙向開單」
  3. **失效條件結構化 + watcher 自動標記**：失效條件文字（如「跌破 $X」「突破 $Y + 放量 > 1.5×」）寫入 predictions 後，後端 `parse_invalidation_to_json()` 自動解析為 `invalidation_json` 結構化條件（價格條件 + 量過濾）；[watcher.py](backend/app/core/watcher.py) 在資料同步完成 / 系統啟動時 sweep（60min 最小存活 + 2 根 K 線確認 + shadow mode 預設 ON），觸發即標 `status='invalidated'`，hit-rate 計算自動排除（`status NOT IN ('active', 'invalidated')`）；報告底部顯示「📛 N 筆因失效條件已排除」

### 意圖驅動的 SYSTEM_PROMPT 動態組裝（v98 強化）

每次對話前由 `detect_intents()` 對訊息做關鍵字匹配，產出意圖集合 → `_INTENT_TO_MODULES` 對應模組 → `assemble_system_prompt_split()` 套用 `_DOMINANCE` 支配關係 → 拆 (static, dynamic) 兩段（static 進 cache_control）。

**意圖支配關係（避免高低階模組同時載入造成重複）**：

| 高階模組 | 支配（移除）的低階模組 |
|---|---|
| `quant_research` | `factor_validation` |
| `output_deep_phase3` | `output_deep_phase1` / `output_deep_phase2` |
| `comprehensive_analysis`（全部分析）| `output_deep_phase1/2/3` + `output_lite/full` + `factor_validation_mode` + `momentum_analysis_mode` + `regime_analysis_mode` + `strategy_backtest_mode` |

**關鍵字 → 意圖**（節錄）：

| 意圖 | 關鍵字 |
|---|---|
| `comprehensive_analysis` | 全部分析、完整全面分析、一次完整、comprehensive |
| `deep_phase1/2/3` | 完整分析一/二/三、全面分析一/二/三、深度分析一/二/三 |
| `deep_analysis` | 完整分析、全面分析、詳細分析、深度分析 |
| `factor_validation` | 因子驗證、因子排名、IC 排名、哪些因子有效 |
| `momentum_analysis` | 動能、動量、追強、相對強弱、ROC、加速、減速 |

`comprehensive_analysis` 命中後 `detect_intents` 會自動 `discard()` 所有低階分析意圖（deep_analysis / deep_phase1-3 / analysis / factor_validation / momentum_analysis / regime_analysis / strategy_backtest），確保只走「統合報告」單一路徑，不會多載重複 prompt。

### 全部分析統合報告 prompt（v98 新增 / v99 + v100 強化）

`_PROMPT_MODULES["comprehensive_analysis"]` 載入時定義 6 段固定報告結構：

1. 📊 **市場環境**（regime + currentRegime + sector/basket breadth + crossStockSignals）
2. 🏛 **結構分析**（SMC + 趨勢方向 + STL decomposition）
3. ⚡ **動能特徵**（多週期動量 + 加速度 + 相對強弱 + RSI 背離）
4. 🧪 **多策略回測比較**（系統預跑 8 策略 + per_regime_metrics + Wilson CI）
5. 🔬 **量化研究**（IC + Walk Forward + Monte Carlo + 因子有效性）
6. 🎯 **跨維度結論 + 倉位建議**（綜合 #1-5 交叉驗證 + 引用 `compute_laddered_entries` 結果）

**禁止重複規則**（嚴格遵守）：
- 「動能」結論只寫 #3，#2/#6 提及時用「見動能段落」
- 「因子」結論只寫 #5，#4 引用 IC 數據而非重新解讀
- 「regime」結論只寫 #1，後續引用 regime 標籤即可
- 結論段（#6）不重述前段，只做交叉驗證 + 決策

**字數預算**：總計 1500-2000 字（vs 改造前 4000+ 字 → 報告長度減半）

**v99 強化**：#6 結論段必含「分批進場建議」表格（多空各 3 檔具體價位 + 加權均價 + SL/TP / RR），完全引用 `compute_laddered_entries` 後端計算結果，禁止 LLM 推算。

**v100 強化**：#6 結論段最末必須產出「結構化結論卡」（取代舊隱藏 PREDICTIONS block）— 直接顯示給使用者的可驗證格式，9 欄位 + 「📈 系統參考」placeholder 由後端 `_inject_recent_accuracy()` 替換為實際歷史命中率。低信心場景改用「⚠️ 建議觀望」格式，**不**留下預測（避免汙染統計）。

### 預測驗證閉環（v100 新增）

完成全部分析後，系統自動串連這個閉環：

```
LLM 產出可見結論卡
   ↓ regex 解析（_VISIBLE_CARD_PATTERN）
存入 predictions.db
   ↓ 每次對話完成後 validate_all_active() 自動驗證
比對後續 K 線 high/low → hit_target / hit_stop / expired
   ↓ 累積樣本 → prediction_tracker.get_stats() 算 Brier / ECE / Bayesian
chart_state.recent_accuracy 結構化注入下次分析
   ↓ LLM 看到歷史命中率 → 自動校準信心
低信心 → 走「⚠️ 建議觀望」 → 不再記錄
```

**前端面板**：
- `PredictionDashboard` 加「📊 命中率」tab → 顯示加權勝率 + Bayesian 95% CI + Brier/ECE + 信心校準表（看高信心是否真更準）+ regime 拆分 + 可靠性曲線
- `ChartView` 加「📊 歷史預測」toggle → 自動拉 `/api/predictions/by_symbol` + 用 horizontal_line annotation 在圖上畫過往 entry/SL/TP（多綠空紅）
- 頂部 + Calibration tab 加免責橫幅「歷史命中率不保證未來表現」

**API endpoints**：
- `GET /api/predictions/calibration?symbol=X&days=90` — 給 Calibration 面板
- `GET /api/predictions/by_symbol?symbol=X&days=90` — 給圖表標註

### LLM 安全防護

1. **輸入過濾**：送入 LLM 前檢查是否包含注入攻擊語句
2. **輸出驗證**：確認回傳的 JSON 結構合法，只包含白名單內的函式呼叫
3. **Function 白名單**：LLM 只能呼叫上述 10 個預定義函式
4. **Token 控制**：
   - 不將原始 OHLCV 數據送入 LLM（只傳 20 根 K 線摘要）
   - LLM 只負責意圖解析，計算交給後端
   - 對話歷史超過閾值時自動摘要壓縮
   - 顯示預估 Token 消耗和費用
5. **超時保護**：60s 統一超時 + asyncio.wait_for 雙重保護
6. **除零防護**：ADX/RSI 等指標安全除法

---

## 使用者互動模式

### 8 種分析模式（前端選項）

| 模式 | 內容 | 耗時 | 結論卡 |
|------|------|------|------|
| **基礎分析** | 市場環境 + 八維度技術 + SMC + 情境預測 + GMM/GARCH/HMM | ~15 秒 | ❌ |
| **因子驗證** | 因子 IC 排名 + 組合 IC + Bucket 評分 + 條件機率 | ~20 秒 | ❌ |
| **策略回測** | 多策略比較 + MC + WF + CPCV + SHAP | ~30 秒 | ❌ |
| **市場體制** | GMM Regime + GARCH 波動率 + HMM 狀態轉移 + 事件型態 | ~15 秒 | ❌ |
| **基本面** | 月營收趨勢 + 法人買賣超 + 財報指標 + 綜合評分（台股限定） | ~10 秒 | ❌ |
| **動能分析** | 多週期動量 + 加速度 + 相對強弱 + 反轉偵測 + 策略回測 | ~15 秒 | ❌ |
| **完整分析三階段**（按鈕）| sequence 自動接力 phase1→2→3（市場環境 → 多策略回測 → 量化研究 + 倉位建議）| ~2-3 分鐘 | ✅ v100 |
| **全部分析**（v98 去重統合 + v99 ladder + v100 結論卡）| 一次跑：市場結構 + 動能 + 8 策略回測 + 因子驗證 + 量化研究 + ladder 分批進場 + 可驗證結論卡，**禁止重複** 6 段固定結構（市場/結構/動能/回測/量化/結論），prompt 強制每段只寫一次 | **~8-15 分鐘**（vs 舊版 15-30 分，省 50%）| ✅ v100 |

**結論卡欄位**（v100 新增）：方向 / 進場（含 ladder 多檔分批）/ 目標（RR）/ 止損 / 時間框 / 信心（依 Wilson CI + regime 信心）/ 主要指標 / 市場 regime / 失效條件 / 📈 系統參考（自動填入歷史命中率）。低信心 → 改用「⚠️ 建議觀望」格式不留下預測。

**「全部分析」與「完整分析三階段」差異**：
- **三階段**：3 次獨立 LLM 對話，phase1 完成 → 排隊 phase2 → 排隊 phase3，每階段獨立輸出
- **全部分析**：1 次 LLM 對話含 6 段固定報告，預先做 8 策略回測 + 5 個函式呼叫（含 run_quant_research），統合輸出
- 適用：要看完整連續決策過程 → 三階段；要省時 + 拿到一份統合報告 → 全部分析

### 雙軌操作（完全等價，雙向同步）

**方式 A：LLM 對話**
- 輸入自然語言 → LLM 解析 → 自動更新圖表和參數
- LLM 改的參數會同步到 UI 面板
- 可自訂快捷問題（常用問題一鍵發送，localStorage 存儲）
- 進度狀態即時回饋（「正在分析…」+ 子任務進度 + 耗時計時器）
- 回覆截斷自動續寫（偵測 stop_reason=length → 自動追加第二段）
- 訊息排隊機制（分析中可繼續輸入，最多排 3 個，依序自動執行，可一鍵清除）
- 智慧捲動（用戶往上看歷史時不強制跳到底部，滾回底部後恢復自動捲動）

**方式 B：手動 UI 面板**
- 指標清單（30項，分類別摺疊）
- 每個指標旁開關（控制顯示/隱藏）
- 參數滑桿 + 數字輸入框 + Tooltip 說明
- 一鍵恢復預設值 / 一鍵清除全部指標
- 手動改的參數會同步到 LLM 上下文

**集中式狀態管理**：所有參數變更無論來源，都走同一個 state update。

---

## API 端點設計

```
# 對話（SSE Streaming）
POST   /api/chat/stream             # LLM 串流對話
POST   /api/chat/                   # LLM 非串流對話
GET    /api/chat/history             # 歷史對話列表
GET    /api/chat/history/{id}        # 特定對話內容
GET    /api/chat/distill/status      # 知識蒸餾狀態
POST   /api/chat/distill/preview     # 蒸餾預覽
POST   /api/chat/distill/confirm     # 確認蒸餾
GET    /api/chat/distill/knowledge   # 已蒸餾知識

# 圖表
GET    /api/chart/data               # 取得 K 線數據
GET    /api/chart/available/list     # 可用交易對
GET    /api/chart/tw-stock-search   # 台股搜尋（代碼或中文名）

# 指標
GET    /api/indicators/list          # 取得可用指標清單（30 個）
POST   /api/indicators/calculate     # 計算指標
POST   /api/indicators/search        # 條件搜尋

# 設定
POST   /api/config/llm               # 設定 LLM + API Key（回傳 session token）
POST   /api/config/llm/test          # 測試 LLM 連線
POST   /api/config/llm/models        # 偵測可用模型
GET    /api/config/llm/providers     # 可用供應商列表
GET    /api/config/llm/session/{id}  # 查詢 session 狀態
DELETE /api/config/llm/session/{id}  # 撤銷 session
POST   /api/config/usage/summary     # 累計 Token 用量
POST   /api/config/usage/daily       # 每日用量明細
GET    /api/config/strategies         # 列出自訂分析策略
POST   /api/config/strategies         # 新增策略
PUT    /api/config/strategies/{id}    # 更新策略
DELETE /api/config/strategies/{id}    # 刪除策略

# 數據同步
GET    /api/data/sync-status         # 同步狀態
POST   /api/data/sync                # 觸發數據同步
GET    /api/data/sync-task/{id}      # 同步任務進度
GET    /api/data/available-exchanges # 可用交易所
GET    /api/data/available-symbols   # 可用交易對

# 匯出
GET    /api/export/knowledge-pdf     # 匯出 AI 分析報告 PDF（LLM 生成結構化報告）

# 健康檢查
GET    /api/health                   # 系統健康狀態
```

---

## 前端圖表規格

### 技術選型
- TradingView Lightweight Charts v5（核心 K 線圖）
- React 18 + TypeScript
- Zustand（狀態管理）
- Tailwind CSS（樣式）

### 圖表功能
- K 線主圖 + 多個子圖（可拖曳調整高度，最小 60px，最大 400px）
- 指標 overlay 疊加
- 時間區間高亮標記
- 滑鼠懸停顯示詳細數值（Crosshair + OHLCV + overlay 指標值 + 副圖指標值）
- 時間範圍拖曳選擇
- 響應式設計
- 副圖右上角 ✕ 關閉按鈕（直接移除指標）
- AI 標記分組管理面板（可收合，預設收合為小按鈕，展開後可開關/刪除每組標記）
- 動態價格精度（根據幣種價格等級自動調整小數位）
- 主圖/副圖時間軸精確同步（統一 Y 軸寬度 + ref 即時同步 + crosshair 跨圖同步）

### 指標顯示分類
- **Overlay**：疊在K線上（均線、布林帶、Keltner、唐奇安、追蹤止損、VWAP、Ichimoku、PSAR、Supertrend、Market Structure、Harmonic）
- **Sub Chart**：獨立子圖（RSI、MACD、ATR、OBV、ROC、ADX、Bias、StochRSI、Rel_Vol、Vol_Switch、Max_Drawdown、HV、CVD、Fear_Greed、Funding 等）
- **Info Panel**：資訊面板（凱利公式、回測結果）

---

## 設定面板

### 分頁結構

1. **LLM 設定**：選供應商 → 輸入 API Key → 偵測模型 → 選模型 → 測試連線 → 儲存
2. **分析策略庫**：管理使用者自訂分析方法論（新增/編輯/刪除/啟停用），AI 助手會參考已啟用策略但不受限制
3. **匯出/匯入**：JSON 設定匯出匯入 + AI 分析報告 PDF 匯出（LLM 生成結構化報告，fpdf2 渲染，支援 CJK）

---

## 安全性規格

### Prompt Injection 防護
- 輸入過濾層（檢查注入攻擊語句）
- LLM 輸出驗證（結構合法性）
- Function Calling 白名單（10 個函式）
- 不允許執行任意程式碼

### API Key 安全
- 後端 Fernet 加密（記憶體內，不落磁碟明文）
- 前端 sessionStorage（關閉瀏覽器自動清除）
- Session Token 機制（HTTP 傳輸不暴露明文 Key）
- 用量追蹤使用 SHA-256 Hash（不可逆）
- Session 過期自動偵測（`SESSION_EXPIRED` 信號 + 前端自動清除）
- 偵測模型 + 連線測試功能

---

## 效能優化

### 快取策略（五層攔截 + RAG 知識融合，越用越省 token 且越用越聰明）
- L1：知識快取（14 筆常見 TA 問答，關鍵字匹配，零 token 消耗）
- L2：分析快取（SQLite，相同問題 hash + 相同數據指紋 → 直接回傳，近期 6h / 歷史 30d TTL）
- L3：語意快取（高置信度 ≥0.92 直接返回，零 token 消耗）
  - 模型：`paraphrase-multilingual-MiniLM-L12-v2`（支援中英混合，384 維向量）
  - 每次 LLM 新回答自動存入向量庫，擴大覆蓋範圍
  - 重複問題自動去重（相似度 > 0.95 不重複存入）
- L3.5：知識融合（中度相似 0.75~0.92 → RAG 注入歷史知識碎片到 LLM prompt，低 token 消耗）
  - 知識碎片自動從 LLM 回答中提取（`---KEY_INSIGHTS---` 結構化輸出）
  - 碎片類型：support_resistance / trend / pattern / indicator / strategy / volume / sentiment
  - 每個幣種最多 200 筆碎片，90 天 TTL，自動淘汰低命中率碎片
  - 14 筆種子知識（BTC/ETH 通用 + 策略經驗）永不過期，首次使用即可受益
  - 幣種交叉比對：問題中提到的幣種 ≠ 圖表幣種時，跳過快取避免誤判
- L4：LLM 呼叫（以上全部未命中才走 API，消耗 token）
- 額外：知識蒸餾（舊對話壓縮為精華知識，注入 LLM context 使 AI 更懂你）
- 額外：使用者策略注入（自訂分析框架注入 LLM context，參考但不限制）
- 額外：CSV 本地存儲（歷史 OHLCV 數據，存放在 `data/ohlcv/`）

### 指標暖機處理
- 每個指標定義 warmup_periods
- 查詢時自動往前多抓 max(warmup_periods) 的數據
- 前端不顯示暖機期數據

### 回應速度
- LLM Streaming 回應（SSE Server-Sent Events，邊生成邊顯示）
- 圖表載入動畫（spinner）
- 大量歷史數據異步處理 + 進度回報
- Three-Round LLM 交互（Function Call 執行 → 繪圖指令 → 強制文字回覆）
- 60s 統一超時 + asyncio.wait_for 雙重保護

### 資料持久化
- **對話歷史**：SQLite `data/db/chat_history.db`（保留 60 天）
- **Token 用量**：SQLite `data/db/usage.db`（SHA-256 Hash 追蹤，保留 180 天）
- **知識蒸餾**：SQLite `data/db/knowledge.db`（壓縮後的分析記憶 + 使用者風格）
- **分析快取**：SQLite `data/db/analysis_cache.db`（數據指紋自動失效）
- **語意快取**：SQLite `data/db/semantic_cache.db`（嵌入向量 + 問答，越用越準）
- **知識碎片**：SQLite `data/db/knowledge_fragments.db`（RAG 碎片 + 向量 + 種子知識）
- **使用者策略**：JSON `data/db/user_strategies.json`（自訂分析方法論，持久化）
- **OHLCV 數據**：CSV `data/ohlcv/`（與資料庫隔離，避免誤刪）
- **衍生品數據**：CSV `data/ohlcv/{symbol}_derivatives_{tf}.csv`（Funding Rate/OI/多空比）
- **基本面數據**：CSV `data/fundamental/{code}_revenue.csv` 等（營收/法人/持股/財報）
- **預測追蹤**：SQLite `data/db/predictions.db`（預測生命週期 + 貝氏參數）
- **ML 模型**：`data/models/`（LightGBM/XGBoost/RF 模型 + scaler）

---

## 錯誤處理

| 情境 | 處理方式 |
|------|---------|
| LLM API 超時/失敗 | 友善錯誤提示，建議重試或換模型（60s 超時保護） |
| 交易所 API 全掛 | 使用本地快取，標註「數據可能非最新」|
| 指標計算 NaN/Inf | 安全除法過濾異常值，不讓圖表崩潰 |
| API Key 無效 | 即時檢測，引導重新設定 |
| Session 過期 | 自動偵測 SESSION_EXPIRED，前端清除舊 session 引導重新輸入 |
| 前端渲染失敗 | ErrorBoundary 全覆蓋捕獲，顯示錯誤訊息 + 重試按鈕，不影響其他區域 |
| 數據不足（暖機期）| 提示需要更多歷史數據 |
| OpenAI 401 權限不足 | 特殊錯誤處理，友善提示 |
| Gemini 429 配額用盡 | 動態模型切換建議 |

---

## 使用者設定持久化

- sessionStorage 存儲 LLM 設定（API Key 加密 session token，關閉瀏覽器自動失效）
- Zustand 集中式狀態管理（圖表狀態、指標、對話、LLM 設定）
- 對話歷史後端 SQLite 持久化（可透過「📋 歷史」按鈕回顧/還原）
- Token 用量後端持久化（可透過「📊 用量」面板查看累計消耗）
- 知識蒸餾（可透過「🧠 整理」按鈕觸發，壓縮舊對話為精華知識）
- 自訂快捷問題（localStorage 存儲，可新增/修改/刪除常用問題）
- 自訂分析策略（JSON 持久化，可多條策略分別啟停用）

---

## 已完成功能清單

1. ✅ 專案骨架 + 環境設定（.command 一鍵啟動）
2. ✅ 數據層：CryptoDataEngine 整合（五源投票、斷點續傳、增量更新）
3. ✅ 30 項技術指標計算引擎（含 A 類 OHLCV + B 類外部數據 + Market Structure + Harmonic + CVD + POC + HV）
4. ✅ LLM 多供應商適配器（OpenAI / Gemini / Claude / Ollama + Function Calling + 動態模型探測）
5. ✅ 後端 API 層（FastAPI + SSE Streaming + 三輪互動）
6. ✅ 前端 K 線圖渲染（lightweight-charts v5 + 主圖/副圖/overlay + 動態價格精度）
7. ✅ 前端 LLM 對話 + 指標疊加 + 參數面板（雙軌同步）
8. ✅ AI 助手圖表操控（切換幣種/時間週期、新增指標、圖表標記）
9. ✅ 安全防護（Fernet 加密、Session Token、Session 過期偵測、Prompt Injection 過濾）
10. ✅ 對話歷史持久化 + Token 用量追蹤 + 知識蒸餾
11. ✅ 五層快取 + RAG 知識融合（L1 知識 → L2 分析 → L3 語意 → L3.5 碎片融合 → L4 LLM）
12. ✅ ErrorBoundary 前端全覆蓋崩潰保護
13. ✅ 資料目錄隔離（DB vs OHLCV 分離 + 自動遷移）
14. ✅ K 線圖時區修正（台北時間一致顯示）+ 日期時間選擇器精確到時分（24 小時制）
15. ✅ 主圖/副圖時間軸精確對齊（統一 Y 軸寬度 + ref 即時同步 + crosshair 跨圖同步）
16. ✅ 趨勢線（斜線）繪製支援 + 批量繪圖 + 分組管理
17. ✅ Market Structure 市場結構指標（Swing High/Low + HH/HL/LH/LL 標記 + 結構線）
18. ✅ 自動知識融合 RAG（LLM 回答自動提取 KEY_INSIGHTS → 向量化碎片 → 新問題注入歷史經驗）
19. ✅ 快取幣種交叉比對（問題提到的幣種 ≠ 圖表幣種時跳過快取）
20. ✅ 14 筆種子知識預載（BTC/ETH 通用指標經驗 + 通用交易策略）
21. ✅ 回測引擎實作（NumPy 向量化 + 保守滑點 0.05% + 手續費 0.1% + 止損止盈 + IS/OOS 分割 + 完整績效統計）
22. ✅ 多策略比較（compare_strategies function call，同時回測 2-5 個策略，Sharpe 排名）
23. ✅ 智慧繪圖策略（精選原則、顏色規範、諧波繪圖指南、主動繪圖行為）
24. ✅ 諧波型態偵測指標（Zigzag + Fibonacci 比例驗證 + Gartley/Bat/Butterfly/Crab/Shark）
25. ✅ LLM 適配器超時保護（60s 統一超時 + asyncio.wait_for 雙重保護）
26. ✅ 技術指標除零防護（ADX/RSI 安全除法 + Registry 空數據防護）
27. ✅ 回測引擎修正（跳空缺口處理 + 進場資金記錄 + 止損止盈取真實價格）
28. ✅ 前端效能優化（API 統一錯誤攔截 + ChartView useMemo/memo + Store 精細訂閱）
29. ✅ 對話資源管理（final_text 50KB 上限 + TimeoutError 捕獲 + finally 清理）
30. ✅ 外部數據完善（CVD 累計量差 + POC 成交量密集區 + HV 歷史波動率）
31. ✅ 匯出/匯入設定（JSON 格式，含圖表設定 + 指標組合 + 標記）
32. ✅ 基礎測試框架（pytest 23 個測試，覆蓋指標計算 + 回測引擎）
33. ✅ 進度狀態即時回饋（SSE status 事件 + 前端動態進度列 + 耗時計時器）
34. ✅ 動態價格精度（根據幣種價格等級自動調整小數位數）
35. ✅ 台股支援（TwStockEngine via yfinance + 台股族群/概念股 29 族群 + 中文名搜尋）
36. ✅ Monte Carlo 優化（自相關偵測、regime-aware bootstrap、壓力測試、Kelly 回饋）
37. ✅ Walk Forward 優化（不重疊窗口 + embargo + per-window SL/TP 優化）
38. ✅ CPCV 組合淨化交叉驗證（C(5,2)=10 組合 + purge + embargo）
39. ✅ 六大核心模組量化升級：
    - 模組一：衍生品數據（Funding Rate / OI / Long-Short Ratio，Binance Futures 免費 API）
    - 模組二：精細化標籤（方向 / 幅度 / 路徑 First-Touch / 狀態）
    - 模組三：GMM Regime 分類 + GARCH 波動率預測 + HMM 狀態轉移
    - 模組四：VIF 因子淨化 + 殘差正交化 + SHAP 特徵解釋 + 交互項特徵
    - 模組五：Brier Score + ECE + Beta-Binomial 貝氏更新
    - 模組六：CPCV + Walk Forward + Monte Carlo 三重交叉驗證
40. ✅ 過度保守修復（滑價分級 + 預設止損 + 評分調整 + 觀望規則彈性化）
41. ✅ 預測系統強化（情境預測回測驗證 + 動態權重校準 + regime 追蹤 + 自適應衰減）
42. ✅ 命中 K 線回推共同特徵分析（Cohen's d + 5 維度相似度 + 漂移偵測）
43. ✅ 策略有效期修復（按 timeframe 分級 + 止損放寬 + forward_bars 自動調整）
44. ✅ LLM 回覆截斷自動續寫 + 前端標記去重
45. ✅ 因子群 Bucket 評分（趨勢/動量/波動/量能/結構五群 -2~+2 分）
46. ✅ 7 種分析模式（基礎 / 因子驗證 / 策略回測 / 市場體制 / 動能分析 / 完整分析）
47. ✅ 動能交易分析模組（多週期動量 + 加速度 + 相對強弱 + 反轉偵測 + 策略回測）
48. ✅ 數據同步修復（增量更新不需 start_date + 往前補抓 + 自動加入下拉選單）
49. ✅ 訊息排隊機制（分析中可繼續輸入發送，最多排 3 個，依序自動執行，可一鍵清除）
50. ✅ 智慧捲動（用戶往上看歷史對話時不被 streaming 強制拉到底部）
51. ✅ 對話觸發數據下載（sync_symbol_data 單檔 + sync_sector_data 族群批次，LLM 主動建議下載）
52. ✅ 台股基本面分析模組（月營收/法人買賣超/外資持股/財報 EPS/本益比/殖利率）
53. ✅ LLM 回覆截斷自動續寫（stop_reason=length 偵測 → 自動追加第二段）
54. ✅ 前端標記去重（多輪 function call 不再產生重複 BB/支撐壓力標記）
55. ✅ 切換幣種自動清除 annotations + 日期範圍（避免殘留標記和數據限縮）
56. ✅ 過度觀望修復第二輪（止損放寬 + 評分扣分再降 + 觀望規則鬆綁 + 折衷建議）
57. ✅ 系統能力聲明（PROMPT_CORE 最頂層禁止回覆「不支援台股」）
58. ✅ 數據不足自動提示下載（query_chart_data 回傳 hint + 三層防護確保 LLM 呼叫 sync）
59. ✅ 過度觀望修復第二輪（止損放寬 + 評分扣分再降 + 觀望規則鬆綁 + 30-45 分折衷建議）
60. ✅ 回測策略多元化（不同嚴格度 + 必含至少 1 個做空策略 + 至少 4 種）
61. ✅ 台股中文名稱改用 TWSE/TPEx 官方 API（涵蓋所有上市櫃 ~1800 檔）
62. ✅ 上櫃股票判斷修正（3/4/5/6/8 開頭預設 .TWO + 失敗自動切換重試）
63. ✅ SyncPanel 同步清單用戶可編輯（localStorage 持久化 + 可刪除 + 自動中文名）
64. ✅ TopBar 下拉選單由 ✎ 編輯面板控制（同步後自動加入 + symbols-updated 事件）
65. ✅ .command 啟動腳本自動清理殘留 port
66. ✅ 上市/上櫃自動判斷改用 TWSE/TPEx 官方 API（不再需要例外清單，涵蓋所有上市 1351 + 上櫃 10635 檔）
67. ✅ CC BY-NC-SA 4.0 License + README.md（公開前準備）
68. ✅ Claude 訂閱制模型動態偵測（透過 CLI 自動取得最新可用模型）
69. ✅ 排隊機制修復（佇列處理改用 useEffect 監聽 chatLoading，不再卡住）
35. ✅ 十字線數值顯示（主圖 OHLCV + overlay 指標值 + 副圖指標值，DOM 直接操作避免重渲染）
36. ✅ 自訂快捷問題（可新增/修改/刪除常用問題，localStorage 持久化）
37. ✅ AI 分析報告 PDF 匯出（LLM 生成結構化報告 → fpdf2 渲染 → CJK 字型支援）
38. ✅ OpenAI 三輪互動修復（第二輪只有 function calls 時觸發第三輪純文字生成）
39. ✅ Session 過期自動偵測（SESSION_EXPIRED 信號 + 前端自動清除引導重新輸入）
40. ✅ 多維度交叉分析框架（6 維度交叉驗證 + 信心等級判斷）
41. ✅ draw_pattern 型態繪圖函式（一鍵繪製諧波/三角形/旗型等，自動連線+標注）
42. ✅ annotate_chart 批量繪圖 + 分組管理（group_name + annotations 陣列）
43. ✅ AI 標記管理面板（可收合/展開，分組開關/刪除，類 TradingView）
44. ✅ 副圖可拖曳調整高度（drag handle，60-400px）
45. ✅ 副圖 ✕ 關閉按鈕（直接在圖表上移除指標）
46. ✅ 使用者自訂分析策略庫（CRUD API + 前端管理 UI + 注入 LLM context）
47. ✅ 回測策略多元化指引（5 種策略模板 + 不同嚴格度 + 必含做空 + 自動比較至少 4 種）
48. ✅ 預設 30 層機構交易分析框架（首次啟動自動載入）
70. ✅ 背景啟動腳本（`阿斯拉量化系統-背景啟動.command`）：從 VSCode 整合終端 / iTerm / Finder 任一位置執行，服務都跑在獨立 Terminal.app 視窗，關閉呼叫端不影響系統；主啟動腳本每次啟動自動寫 `backend/.port` 同步前端 proxy 目標 port
71. ✅ 台股 BB Width 壓縮掃描器（`tw_bb_scanner`）：全市場 ~1900 檔一次掃 + 10 並發 + SSE 即時進度 / 結果 / 失敗 / 警告 / 完成事件 + 失敗率 > 10% 自動中止（避免 yfinance 封 IP）
72. ✅ 掃描歷史 SQLite 持久化（`tw_scan_history.db`）+ 歷史頁面檢視任一次完整結果 + 「回看」功能（取當前價算後續漲跌幅，按 return_pct 降序）
73. ✅ 掃描失敗清單持久化（`failures_json` 欄位）：SSE `failure` 事件即時推送 + DB 保存 + 歷史頁可展開檢視（代號 / 名稱 / 市場 / 產業 / 失敗原因）
74. ✅ 掃描條件優化：壓縮持續性（最近 N 根都 < 門檻）+ 絕對 BB Width 下限（排常年低波動 ETF/控股）+ 趨勢健康（MA60 斜率 > 0 或 收盤 > MA20 > MA60，取代原本單看收盤 vs MA60）+ 拿掉 ADX（與 BB 壓縮邏輯重疊）+ 5 日均量門檻降為 200 張
75. ✅ 掃描歷史自動保留最新 100 筆（`_MAX_HISTORY_ROWS`，save 時同 commit 刪最舊；手動刪除按鈕行為不變）
76. ✅ Scanner 觸發 `fetch_ohlcv` backfill 修復：明確傳入 `start_date = now - history_days` 讓 engine 自動補齊本地歷史不足的 CSV（原 bug：本地檔案存在但歷史不足時 `need_backfill=False` 永遠不補）
77. ✅ 掃描失敗訊息語意化：`bars insufficient (N)` → 「歷史資料僅 N 筆（需至少 150；新上市或資料源斷檔）」或「無歷史資料（ticker 不存在、下市股或資料源異常）」
78. ✅ 掃描歷史天數可自訂（進階過濾 UI 新增「抓取歷史天數」欄位，預設 400 日曆日，下限 220 ≈ 150 根交易日）
79. ✅ SQLite 資料庫自動備份：hot backup（`.backup` API）+ GFS 分層保留（每天 7 + 每週 4 + 每月 12 + 每年永久）+ launchd 排程每天 0:00 自動執行 + 跳過純快取 DB（`analysis_cache` / `semantic_cache`）；提供 `阿斯拉量化系統-手動備份.command` 雙擊立即備份
80. ✅ 台股掃描器加速：並發 10 → 25 + Token Bucket 限速（10 RPS / burst 20）防止 yfinance 被打爆，預期 28 分 → 12-15 分
81. ✅ yfinance Circuit Breaker：模組層單例追蹤連續失敗，連續 5 次失敗 → 暫停 30 秒（half-open 試一次再決定）+ 指數退避重試（1s → 2s → 4s）；保護所有 `fetch_ohlcv` 呼叫端不會在 yfinance 故障時打到被封 IP
82. ✅ Anthropic Prompt Caching：system prompt 拆成「靜態（CORE + 模組，~11000 字 / ~2700 tokens）+ 動態（時間戳、chart_state）」兩 block，靜態段加 `cache_control: ephemeral` 5 分鐘 TTL；tools 區塊也快取（21 個 function 定義）；`TokenUsage` 加 `cache_creation_tokens` / `cache_read_tokens` 欄位追蹤 cache hit；OpenAI 自動快取、Gemini implicit cache、Ollama 不適用
83. ✅ 掃描資料區間顯示：掃描器面板新增「📅 資料區間：YYYY-MM-DD ~ YYYY-MM-DD（N 天，判斷最新一根 K 線的壓縮程度）」即時提示，隨「抓取歷史天數」設定變動而即時更新；掃描開始時鎖定區間（避免掃描中改參數導致顯示混亂），掃描完成摘要也帶出實際使用的區間
84. ✅ 手動備份快捷腳本（`阿斯拉量化系統-手動備份.command`）：雙擊立即執行 `backup_databases.py`，補強 launchd 自動排程「Mac 關機時不會跑」的盲點，適用「重大操作前」「累積重要對話後」「Mac 連續幾天沒開」等情境
85. ✅ ML 預測有效性診斷工具（`backend/scripts/audit_ml_predictions.py`）：查 `predictions.db` ML 標記分布 + 命中率對照 + ml.db 訓練狀態。實測發現 236 筆預測 `ml_enhanced=1` 為 0、64% 整體命中率（126 hit_target / 197 已驗證）— ML pipeline 從未真正接通到 chat 介面
86. ✅ SSE 120s 超時誤報修復（chat.py 後處理改背景任務）：原本 stream 完後 line 1951-2065 同步跑 30-300 秒 post-processing（embedding × N、predictions validation）導致沉默期間前端誤判超時。抽出 `_post_process_chat_message()` 改 `asyncio.create_task` detach 在背景，先 yield done 給前端；前端 `STREAM_TIMEOUT_MS` 120s → 180s 雙重保險
87. ✅ LLM 真串流統一介面（adapter.py）：新增 `StreamEvent` dataclass（type=text_delta/function_call/usage/stop）+ `BaseLLMAdapter.chat_stream_events()`（含預設 fallback 實作呼叫 chat() 後一次性 yield）。四個 adapter 全部 override 為真串流：
    - **ClaudeAdapter（Anthropic API）**：用 `client.messages.stream()`，邊收 content_block_delta 邊 yield；tool_use 用 input_json_delta 累積；最終 message 拿 cache 用量
    - **ClaudeSubscriptionAdapter（CLI 訂閱版）**：擴充原有 stream-json 解析，加 `<tool_call>` XML 緩衝邏輯（看到開頭就 hold、看到結尾才解析 yield function_call）+ usage/stop 事件
    - **OpenAIAdapter**：用 `chat.completions.create(stream=True)` + `stream_options={"include_usage": True}`；tool_calls 按 index 累積 delta JSON 拼接
    - **GeminiAdapter**：用 `generate_content_stream` + `asyncio.to_thread` 包同步 iterator
88. ✅ chat.py 路由消費 streaming（Round 2/3/續寫 三處改即時 yield）：邊收 token 邊發 SSE token event，使用者看到字一個個冒出來而非「等完整段落湧出」。內建 `_MARKER_RE` 偵測 `KEY_INSIGHTS`/`PREDICTIONS`/`SYSTEM_DISTILL` 標記、看到就停止 yield 避免內部標記洩漏給使用者；保留原本的「假串流」fallback（如果 adapter 未 override）。實際效果：第一個 token 出現後 5-15 秒就開始看到內容、整體耗時類似但體感大幅改善
89. ✅ ML 模組降級（_settings.py + 全鏈路條件）：`backend/app/core/ml/_settings.py` 加 `ML_ENABLED = False` 總開關；`_inject_ml_prediction()` 早返不注入；data_sync.py 跳過 ML 訓練；`/api/ml` 路由統一回 503；前端 MLPanel 顯示停用橫幅。保留所有程式碼（未來改 ML_ENABLED=True 即可恢復），但 chat 流程徹底跟 ML 解耦
90. ✅ STL 時序分解（`backend/app/core/timeseries_decomposition.py`）：用 `statsmodels.tsa.seasonal.STL` 把 close 序列拆「趨勢/季節/殘差」三層；自動注入 `chart_state.decomposition`，給 LLM 看到 trend_direction、seasonal_strength、residual_volatility_pct 等立體結構；系統 prompt 加解讀指引（典型場景對應建議）
91. ✅ 跨股票群體訊號（`backend/app/core/cross_stock_signals.py`）：給台股自動算所屬族群 + 龍頭股表現 + breadth + 個股 RS；自動注入 `chart_state.crossStockSignals`，避免 LLM 給「逆勢進場」這類錯誤建議；解讀指引包含「個股強但族群弱 → 警示拉高出貨」「龍頭強 + 個股 RS>1.5 → 跟趨勢」等典型場景
92. ✅ LSTM-SCA-ARIMA-GARCH 混合模型 PoC（`backend/research/lstm_sca_arima_garch/`）：完整研究框架，獨立於主系統。8 個模組：data_loader / decomposer / arima_trend / garch_volatility / lstm_residual / sca_optimizer / hybrid_pipeline / evaluate；用 BTC/USDT 1d 跑通首次實驗（ARIMA walk-forward baseline + 混合 5 步預測）；額外依賴在 `requirements_research.txt`（statsmodels / arch / torch / matplotlib），不污染 production requirements
93. ✅ 跨股票群體訊號擴展到加密貨幣（`compute_signals_crypto`）：完全動態 basket（從本地有資料的 USDT 對掃描，排除穩定幣 USDT/USDC/BUSD/DAI 等 + 衍生品），門檻 ≥ 3 個成員。提供：basket size、breadth_pct_advancing（最近一根 K 線收漲的 %）、BTC 龍頭信號、basket 平均 5K 線漲跌、自身 RS（vs basket）、**BTC dominance proxy**（用 USD 成交額 = volume × price 計算，避開 base coin 量級不可比的問題）、**alt outperform count**（跑贏 BTC 的 alt 數）、**market_regime 推斷**（btc_led / alt_season / bearish / mixed）。系統 prompt 加 4 種 regime 對應的決策建議（例：market_regime='alt_season' + 你分析 BTC → 提示「考慮挑強勢 alt 而非 BTC」）
94. ✅ STL 時序分解視覺化（兩個新指標 `stl_trend` + `stl_oscillator`）：用既有指標註冊系統，零前端改動 — 自動繼承指標面板開關、副圖渲染、十字線數值。`stl_trend` 為 overlay（主圖疊加平滑趨勢線、無滯後）；`stl_oscillator` 為 sub_chart（顯示 Seasonal 規律週期 + Residual 隨機雜訊兩條序列）。預設關閉避免擠畫面，使用者主動到指標面板開啟。`decompose_full_series()` 函式回傳完整序列，1250 根日線約 50-100ms（接受成本，不加 cache）
95. ✅ 修復「完整分析」期間 SSE timeout 誤報：根因是 chat.py 的「預備期重操作」沒有心跳：
    - **預回測**（line 1580）：6 個策略回測累計 60-180+ 秒，原本 `await execute_function_calls()` 期間完全沉默
    - **補齊缺失函式**（line 1709）：含 run_quant_research 等重操作，可能 30-90 秒沉默
    - **自動校準**（line 1525）：sync 函式 run_calibration 直接呼叫，可能阻塞 10-60 秒
    解法：三處都包成 `asyncio.create_task` + 每 5 秒 yield 帶秒數的 status event；前端 `STREAM_TIMEOUT_MS` 從 180s 拉到 300s 當雙重保險
96. ✅ 修復「完整分析」期間 SSE timeout 誤報（深層）：上一輪心跳修了「await 沒回應」，但實測仍 300s timeout。
    根因：function execution 內部呼叫 `run_quant_research` / `run_walk_forward` / `run_monte_carlo` 等都是 sync NumPy/pandas/sklearn 計算，**直接阻塞 asyncio event loop**（無 `await` 切換點）。即使 chat.py heartbeat loop 有 `await asyncio.sleep(2)`，event loop 被卡住期間 sleep 也排不上時程，心跳事件實際間隔變成 60-300+ 秒，前端誤判超時斷線。
    解法：新增 `_execute_function_calls_in_thread()` helper，把 `execute_function_calls` 透過 `asyncio.to_thread` 丟到 worker thread 執行（thread 內開新 event loop 跑 async 內容）。主 event loop 完全不被 ML 計算佔用 → 心跳每 2-5 秒準時觸發 → 前端持續收到進度 → 不再誤判超時。三個 call site（預回測 / 主函式執行 / 補齊缺失函式）統一改用 thread 模式
97. ✅ Regime-aware 回測 + 退化偵測 + 自我記錄（建立可信賴的投資分析系統 — Phase 1+2.1+2.2+2.3+3.1+3.2+L1）：
    **Phase 1 A**（regime filter）：新增 `backend/app/core/regime_filter.py`，6 種 regime 分類（trending_up/down, ranging, high/low_vol, unknown），用規則式 + 自算 ADX（修了 _detect_market_regime 取 ADX 永遠拿 None 的 bug）；策略相容 regime 對應表（趨勢類→trending、均值回歸→ranging/low_vol、動量突破→trending+high_vol）
    **Phase 1 B**（放寬 SL/TP）：[chat.py](backend/app/api/routes/chat.py) 預設 6 策略 SL 5%→8%、TP 15%→12% 等（給加密波動空間）+ 加 compatible_regimes 標籤
    **Phase 1 C**（滾動視窗）：[executor.py](backend/app/core/llm/executor.py) `_exec_compare_strategies` 加 `lookback_months=12` 預設，跑「全歷史 + 近 1 年」兩組對照
    **Phase 2.1**（風險揭露強化）：[engine.py](backend/app/core/backtest/engine.py) `_compute_metrics` 新增 Wilson 95% 信心區間（小樣本下勝率真實信心）+ 連續虧損次數 + MDD 持續期間 + CVAR 5% + Calmar Ratio
    **Phase 2.2**（per-regime 績效拆解）：每筆 trade 按 entry_idx 對應 regime，輸出 `per_regime_metrics: {trending_up: {n_trades, win_rate, avg_return}, ...}` 給 LLM 看「策略在哪個 regime 才有效」
    **Phase 2.3**（unknown regime 警告）：[chat.py](backend/app/api/routes/chat.py) 自動注入 `chart_state.currentRegime` + confidence < 0.5 時注入 `regimeWarning`（auto_position_multiplier=0.5），系統 prompt 強制 LLM 在低信心時警示 + 縮小建議倉位 + 不用「強烈建議」措辭
    **Phase 3.1+3.2**（auto audit + launchd）：新增 `backend/scripts/audit_system_health.py` 跑 30天 vs 90天命中率對比 + 退化判定 (drop > 15pp) + 行動建議；輸出 `system_health.json` + append `health_history.log`；`install_audit_launchd.sh` 排程每天 0:30 自動跑（不依賴系統運行）。新增 `/api/system/health` endpoint 給前端啟動時讀取
    **Level 1**（unknown regime 自動記錄）：新增 `backend/app/core/unknown_regime_logger.py`，confidence < 0.5 時自動寫進 `unknown_regime_log.db`，累積樣本給未來 Level 2 auto-classify 用
    LLM 解讀指引（function_defs.py）加 currentRegime / regimeWarning / per_regime_metrics 的解讀範例，教 LLM「依當前 regime 過濾哪些策略結果可信」「低信心時主動警示」「跨 regime 比較策略 robustness」

98. ✅ 「全部分析」按鈕去重重構 + 三階段真序列執行（系統 prompt 強化第 4 波）：
    **A1**（[chat.py:1620](backend/app/api/routes/chat.py#L1620) `_PRE_BT_INTENTS`）：加 `deep_phase3` + `comprehensive_analysis` 進預回測白名單，讓「全部分析」/「完整分析三」也跑 8 策略對比。原本只有 phase1/2 跑，phase3 與 comprehensive 直接跳過 → 缺乏對比基礎
    **A2**（[ChatInterface.tsx:90](frontend/src/components/ChatInterface/ChatInterface.tsx#L90) + 點擊處理）：「完整分析三階段」按鈕改成 sequence 欄位（含 phase 1/2/3 三條 prompt），按下後第一條直接送、第 2-3 條進 messageQueue 自動接力。原本按鈕文字是「完整分析三階段」但 prompt 只送「完整分析一」，第二、三階段從沒真的跑
    **A3**（[function_defs.py](backend/app/core/llm/function_defs.py) CORE 第 250 行附近）：加「回測結果引用規則」段落，強制 LLM 必須引用 `per_regime_metrics`（拆 regime 勝率）+ `wilson_ci_lower/upper`（信心區間），策略對比按 Wilson 下界排名，避免被高勝率少樣本誤導
    **B1+B3+B4**（comprehensive_analysis 新意圖）：function_defs.py 新增 `comprehensive_analysis` intent 關鍵字（"全部分析", "完整全面分析" 等），在 detect_intents 內自動剝除 deep_analysis/factor_validation/momentum_analysis 等低階意圖避免重複；新增 `_DOMINANCE` 字典在 assemble_system_prompt_split 內套用模組支配關係（高階模組存在時自動移除被支配的低階 prompt）；chat.py `_REQUIRED_ANALYSIS_FUNCS` 加 comprehensive_analysis 對應的 6 個必要函式（detect_smc_structure / generate_scenarios / analyze_momentum / compare_strategies / scan_conditional_probability / run_quant_research）
    **B2**（[function_defs.py](backend/app/core/llm/function_defs.py) `_PROMPT_MODULES["comprehensive_analysis"]`）：新增統合報告 prompt，定義 6 段固定報告結構（市場環境 / 結構 / 動能 / 8 策略回測 / 量化研究 / 跨維度結論）+ 4 條禁止重複規則（動能只寫 #3、因子只寫 #5、regime 只寫 #1、結論段不重述）+ 函式呼叫順序 + 每段字數預算（總計 1500-2000 字 vs 改造前 4000+ 字）
    **B5**（[chat.py:1639](backend/app/api/routes/chat.py#L1639) 預回測策略）：6 策略加 2 個 ROC 動量策略（多/空）變 8 策略，避免動能分析另外跑回測；status 文案同步更新（"6 個策略" → "8 個策略"，預估時間 60-180s → 80-240s）
    預期改善：函式呼叫 6-8 → 5；時間 15-30 分 → 8-15 分；token 輸入 40K → 25-30K；報告 10K+ 字 → 6-8K 字；Anthropic 成本省 50%

99. ✅ 分批進場（Laddered Entries）— 後端計算 + Regime 動態配比 + 完整回測驗證（Phase 1-5）：
    **Phase 1**（[backend/app/core/laddered_entries.py](backend/app/core/laddered_entries.py)，新建）：`compute_laddered_entries()` 後端算分批價，所有 price 直接從 BB / EMA / Donchian / ATR 取，禁止任何算術推估。Regime 動態配比表：trending → 50/30/20 金字塔加碼（current → ema_20 → bb_middle）；ranging/low_vol → 25/35/40 倒金字塔接刀（bb_middle → bb_lower → donchian_low）；high_vol → 33/33/34 對稱平均（current → −1×ATR → −2×ATR）；regime confidence < 0.5 → enabled = false 並重用既有 regimeWarning。SL = min(entries) − 1.5×ATR；TP = weighted_avg + max(2×ATR, risk × 2)，保證 RR ≥ 2
    **Phase 2**（[engine.py](backend/app/core/backtest/engine.py)）：加 `ladder_config` 參數 + 新建 `_run_ladder_loop()` 助手（不動既有單進場主迴圈，避免回歸風險）。每筆信號排掛 N 個 limit orders（首檔市價成交、其餘為限價單）；`max_wait_bars` 內未填者過期取消；SL/TP 基於加權均價；單筆 Trade 含 `entry_legs` metadata + `fill_count`。新增 `metrics["ladder"]`：`avg_fills_per_trade` + `full_fill_rate_pct` + `ratios` + `price_offsets_pct`。新增 [tests/test_backtest_ladder.py](backend/tests/test_backtest_ladder.py) 6 個測試（Phase 1 + Phase 2 + 向下相容）
    **Phase 3**（[walk_forward.py](backend/app/core/backtest/walk_forward.py)）：`run_walk_forward()` 加 `ladder_config` 參數透傳到內部 `run_backtest` 呼叫；MC 不需改動（pnl 列表已包含 ladder 結果）
    **Phase 4**（function 註冊 + LLM prompt 整合）：[function_defs.py](backend/app/core/llm/function_defs.py) 註冊 `compute_laddered_entries` schema + 加入 `ALLOWED_FUNCTIONS` + `_PARALLEL_FUNCS` + CORE prompt 加「分批進場價位引用規則」（強制引用 price/size_pct/source，禁止推算 SL/TP）；[chat.py](backend/app/api/routes/chat.py) `_REQUIRED_ANALYSIS_FUNCS["comprehensive_analysis"]` 加 `compute_laddered_entries`（自動補齊缺失函式）；`_PROMPT_MODULES["comprehensive_analysis"]` #6 結論段加「必須含分批進場建議表格」+ 函式呼叫順序加第 6 步；[executor.py](backend/app/core/llm/executor.py) 加 `_exec_compute_laddered_entries()` handler（自動從 df 算 regime + confidence，不依賴 chart_state）
    **Phase 5**（圖表視覺化）：executor.py 處理 `compute_laddered_entries` 結果時，自動把 ladder 多空各檔 + SL + TP 轉成 horizontal_line annotations 推到 `chart_updates.annotations`（綠線多、紅線空、SL/TP 各一條）
    端到端驗證（BTC/USDT 1d，2026-04-27）：regime 自動分類 trending_up（conf 0.726）→ 3 檔 long entries（current + EMA20 + BB middle）→ weighted_avg $76,074 / SL $72,299 / TP $83,623 / RR 2.0
    預期效益：投資建議從「30% 倉位」抽象變「$X 進 50% / $Y 進 30% / $Z 進 20%」具體；ladder vs 單進場有完整回測對照（含 WF + MC）；LLM 幻覺風險 = 0（強制引用後端計算）

100. ✅ 預測驗證閉環 — 結構化結論卡 + Calibration 面板 + 圖表標註過往預測 + recent_accuracy 結構化注入（v100）：
    **問題**：系統提出投資分析後，使用者無法驗證準確度。3 個關鍵缺口：① 隱藏 ---PREDICTIONS--- block 跟正文結論可能漂移 ② 後端有 calibration（Brier/ECE/Bayesian）但前端看不到 ③ 圖表上沒有歷史預測痕跡
    **Phase 1**（[function_defs.py](backend/app/core/llm/function_defs.py) CORE + [prediction_tracker.py](backend/app/core/prediction_tracker.py) + [chat.py](backend/app/api/routes/chat.py)）：CORE prompt PREDICTIONS 段改成「結構化結論卡」（emoji + 表格 + 用戶可見），含方向/進場/目標/止損/時間框/信心/指標/regime/失效條件 9 欄位。新增 `_VISIBLE_CARD_PATTERN` regex 從可見卡解析 + 「建議觀望」格式偵測（不存 prediction）；舊隱藏 block 保留 fallback 向下相容。chat.py post-process 加低信心過濾（confidence="low" 不存 DB）+ `_inject_recent_accuracy()` 自動替換「📈 系統參考：」佔位行為實際歷史命中率（從 prediction_tracker.get_stats）；SSE 新事件 `accuracy_inject` 讓前端即時替換顯示
    **Phase 2**（[predictions.py](backend/app/api/routes/predictions.py) + [PredictionDashboard.tsx](frontend/src/components/PredictionDashboard/PredictionDashboard.tsx)）：新 `/api/predictions/calibration` endpoint 暴露 Brier/ECE/信心桶/Bayesian/regime/indicator 拆分；PredictionDashboard 新增「📊 命中率」tab 顯示整體命中率（含 Bayesian 95% CI）、Brier/ECE 解讀、信心校準表（看高信心是否真更準）、按 regime 拆分、可靠性曲線。頂部加免責橫幅
    **Phase 3**（[predictions.py](backend/app/api/routes/predictions.py) + [ChartView.tsx](frontend/src/components/ChartView/ChartView.tsx)）：新 `/api/predictions/by_symbol` endpoint 拉某 symbol 過去 90 天預測（含 entry/target/stop/status/MFE/MAE）；ChartView 加「📊 歷史預測」toggle 按鈕（預設關閉），開啟後自動載入 + 用既有 horizontal_line annotation 系統畫進場價/SL/TP 線（多綠空紅 + ✓/✗/⏳ 結果標籤），歸到 `past_predictions` group 方便管理
    **Phase 4**（[chat.py:631](backend/app/api/routes/chat.py#L631)）：把 prediction_feedback 的純文字注入升級為 `chart_state.recent_accuracy` 結構化欄位（30d/90d 命中率、樣本數、Bayesian CI、Brier、best/worst indicators）；CORE prompt 加引用規則：win_rate < 50% 且 n >= 10 必須警示 + 強制降信心；best_indicators 優先引用 worst_indicators 標警告
    **Phase 5**：PredictionDashboard 頂部 + Calibration tab 加免責聲明「歷史命中率不保證未來表現，僅供參考」
    端到端閉環：使用者按「全部分析」→ LLM 產生可見結構化結論卡 → 後端 regex 解析存進 predictions.db → 後續 K 線到期自動驗證（既有 prediction_validator.validate_all_active）→ 累積 30 天後 Calibration 面板顯示真實命中率 → chart_state.recent_accuracy 注回下次分析讓 LLM 校準信心
    預期效益：使用者首次能驗證系統準確度（不再黑盒）；模糊建議無法量化的問題被「低信心 → 建議觀望，不存 prediction」解決；圖表上直觀看到「上次預測對在哪錯在哪」

101. ✅ v101 模仿學習 + 動態 Blend Ensemble + 9 層防護「永不變壞」（Phase 0 + Phase 2.0-2.6）：
    **問題**：v100 累積的 verified predictions 沒被任何模型學習，系統無法「越用越聰明」。但加 ML 有過擬合 / drift / 越用越差等風險，使用者擔心改完系統爛掉。
    **設計鐵律**：v100 結論卡格式永遠不變；v101 段落是 ADD ONLY，從不 REPLACE；v101 沒被量化證明 ≥ v100 之前，使用者看到 100% v100 體驗（Quality Gate 7 硬閾值守衛）
    **9 層防護**：① Feature Flags 1 秒回退 ② Database 純加法 ③ Git tag + DB backup ④ Shadow Mode 4-8 週 ⑤ Quality Gate 7 硬閾值 ⑥ Canary 1→10→25→50→100% 漸進 ⑦ Auto-rollback Watchdog ⑧ Regression Tests ⑨ Champion-Challenger + Stable Fallback
    **Phase 0 安全基礎建設**（[settings.py](backend/app/core/config/settings.py) + [shadow_runner.py](backend/app/core/shadow_runner.py) + [canary.py](backend/app/core/canary.py) + [auto_rollback.py](backend/app/core/auto_rollback.py) + [v101_self_validator.py](backend/app/core/v101_self_validator.py) + [test_v100_regression.py](backend/tests/test_v100_regression.py) + [v101_emergency_rollback.md](backend/docs/v101_emergency_rollback.md)）：9 個 v101 feature flags 預設全 OFF / SHADOW；4 層 use_v101() 守衛（learning_enabled + shadow_off + quality_gate + canary）；自動 rollback 每天 03:00 跑；8 個 v100 regression tests；5 分鐘三層回退 SOP
    **Phase 2.0 特徵記錄**（[prediction_tracker.py](backend/app/core/prediction_tracker.py) `_ensure_schema_v101` + [feature_extractor.py](backend/app/core/feature_extractor.py)）：純加法新增 4 表（prediction_features / shadow_predictions / imitation_model_metrics / quality_gate_log）；39 個特徵分層設計 — 規則用結構性（smc_bias / regime / FVG），LightGBM 用統計性（連續指標）→ 誤差相關性 0.7 → 0.4，ensemble 收益增加 100-200%；[chat.py:1380](backend/app/api/routes/chat.py#L1380) 即時記錄；[backfill_prediction_features.py](backend/scripts/backfill_prediction_features.py) 對 239 筆既有 verified predictions 全部成功回填
    **Phase 2.1 訓練 pipeline**（[imitation_trainer.py](backend/app/core/imitation_trainer.py)）：LightGBM + Platt scaling 校準；動態模型容量（n_estimators / max_depth / min_child_samples 隨樣本量自動調整）；強正則化（reg_alpha=0.5 + Bagging subsample=0.7）；Walk-forward TimeSeriesSplit OOF；Lockbox 最近 20% 永不訓練（**主要 OOS 訊號** — 比 OOF 在小樣本更可靠）；拒絕條件：overfit_gap > 0.20 / lockbox_auc < 0.55 / 沒比現役 +0.02。**首版 v4 model 訓練成功 — Lockbox AUC 0.81、Brier 0.21、n=157**
    **Phase 2.2 推論層**（[imitation_predictor.py](backend/app/core/imitation_predictor.py)）：動態 blend α=sigmoid(0.05*(n-50))（n=30→0.27 / n=50→0.50 / n=100→0.92 / n=200→0.99）；軟性 veto（regime conf < 0.3 或 Wilson CI < 30% → cap p_ml at 0.30，不完全否決）；分歧偵測 |p_ml - p_rule| > 0.3 → conflicts + position_multiplier=0.5；SHAP top 3 + KNN 路徑類比 3 筆。輸出 JSON 對應使用者提的 prompt 格式（policy_distribution / q_value / top_features / similar_paths / conflicts）
    **Phase 2.3 LLM 整合**（[chat.py:677](backend/app/api/routes/chat.py#L677) + [function_defs.py](backend/app/core/llm/function_defs.py) `_PROMPT_CORE` + comprehensive_analysis #6.5）：4 層守衛包圍 chart_state.rl_strategic_insight 注入；CORE prompt 加 v101 引用規則（mode 自然語言改寫 / SHAP top 3 必引 / conflicts 強制警示 / position_multiplier 必套 / 絕對禁止編造）
    **Phase 2.4 監控 + Champion-Challenger**（[champion_challenger.py](backend/app/core/champion_challenger.py) + [predictions.py](backend/app/api/routes/predictions.py) 5 個新 endpoints + [PredictionDashboard.tsx](frontend/src/components/PredictionDashboard/PredictionDashboard.tsx) 「🤖 模型狀態」tab）：3 個模型版本維護（Champion / Challenger / Stable Fallback）+ 1-click 回退 / 重訓 / 停用按鈕；Quality Gate 進度視覺化
    **Phase 2.5 Drift 偵測**（[drift_monitor.py](backend/app/core/drift_monitor.py)）：Adversarial Validation（AUC > 0.65 顯著漂移強制重訓）+ 每特徵 PSI 監控（≥ 0.25 重大漂移）
    **Phase 2.6 自動重訓**（[retrain_imitation.py](backend/scripts/retrain_imitation.py) + launchd 每週日 02:00）：drift 檢查 → 樣本檢查 → 訓練 → auto-rollback → quality gate → canary progression
    端到端驗證：v4 model trained, lockbox AUC 0.81; Quality Gate 6/7 通過（剩 shadow_4w_hit_rate 等累積 4 週 shadow 數據）；37 個測試全通過（29 既有 + 8 新 v100 regression）；frontend TS clean
    **「永不變壞」量化保證**：使用者目前看到 100% v100 體驗（Quality Gate 嚴格守衛）；累積 4 週 shadow + 第二次重訓後若所有 gate 通過 → 自動啟動 Canary 1%；任何時候 5 分鐘可手動回退到 v100.0 git tag

102. ✅ v102 Subprocess 隔離 — 真正解決 macOS native lib 衝突 segfault（保留 ML 推論能力）：
    **問題**：v101 一啟用，主 process 同時 import lightgbm + shap + statsmodels + pandas/numpy，macOS 原生 OMP/BLAS lib 衝突導致 chat 期間整個 backend segfault crash。原本以為要回退到 v100 純穩定狀態（喪失 ML 價值）
    **解法**：[ml_worker.py](backend/app/core/ml_worker.py) + [ml_client.py](backend/app/core/ml_client.py) — 主 process 完全不 import lightgbm/shap，改透過 `subprocess.run` spawn 隔離 worker 跑推論；JSON stdin/stdout 溝通；OMP_NUM_THREADS=1 強制單線程避免內部並發衝突；timeout 30s 保護
    **v102.1 修正**：subprocess 收到的 features 只有 4 個（其他 35 個被填 0），SHAP 解釋全是「當前 0.0」。改用 [feature_extractor.py](backend/app/core/feature_extractor.py) `extract_features_at` 抽全 39 特徵；prediction-related 欄位用合理 placeholder（entry +5%、stop -3%）給推論
    主 process 永遠純穩定（不 segfault），ML 推論能力保留 100%；額外成本只有 spawn subprocess 的 ~50ms

103. ✅ v103 完整優化（不付費版，13 項全做）：
    **動機**：v102 後仍存在多個 user-visible bugs（雙向計劃缺、KNN similarity 全 0、SHAP 跟 LLM 決策脫節）+ 模型品質根本問題（8 策略全 0 trades 訓練資料貧乏）。User 要求「不付費前提下完成全部優化」
    **Phase 1 報告品質修復（commit 59d1e36）**：
      - 雙向計劃格式（[function_defs.py](backend/app/core/llm/function_defs.py) CORE 4 種結論卡選擇規則 + [prediction_tracker.py](backend/app/core/prediction_tracker.py) `_BILATERAL_CARD_PATTERN`）：ranging 環境吃掉 90% 觀望場景；schema 加 `is_bilateral` / `bilateral_pair_id`（純加法）；`parse_predictions` 改回傳 list（雙向 = 2 筆）
      - 「📈 系統參考」placeholder regex 寬鬆化 + accuracy_inject SSE 補打 fallback（[chat.py](backend/app/api/routes/chat.py) + [api.ts](frontend/src/services/api.ts)）
      - KNN 特徵 StandardScaler 正規化（[imitation_predictor.py](backend/app/core/imitation_predictor.py) `_knn_similar`）：similarity 從全 0 → 0.24-0.30 合理範圍
      - 30 秒結論段強制第一行（comprehensive_analysis prompt #6.5）
    **Phase 2 模型品質根本解（commit 23bca68）**：
      - 8 策略改 dict 格式 + cross_above/cross_below operators + 小數百分比（原本 stop_loss_pct=8.0 被當 800%）→ 每 symbol 86-228 trades（原本全 0），ML 訓練資料品質回穩
      - bilateral validator（[prediction_validator.py](backend/app/core/prediction_validator.py) `_validate_bilateral`）：先觸碰 entry 的方向走正常 target/stop 驗證；對向標 cancelled_by_pair；兩邊都未觸碰 → 過期一起 expire；race condition + pair 不完整 fallback 防護
    **Phase 3 模型架構升級（commit ee36c30）**：
      - per-regime 子模型（[imitation_trainer.py](backend/app/core/imitation_trainer.py) `train_per_regime_models`）：分別訓 trending_up / trending_down / ranging / high_vol；模型存檔加 `_<regime>` suffix；Champion-Challenger 只跟同 regime 比較；imitation_model_metrics 加 `regime` 欄位（NULL = all-in-one）
      - predictor 依 chart_state.currentRegime 自動切子模型，找不到 fallback all-in-one
      - 每日 quick drift check（[daily_drift_check.py](backend/scripts/daily_drift_check.py) + launchd 每天 03:00）：短窗 PSI（5d vs 30d），max PSI > 0.20 立刻 force=True retrain，regime shift 反應從 7 天縮到 24h
      - 新增 shap_log 表（給 Phase 4 用）
    **Phase 4 Dashboard 增強（commit e7b9bbd）**：
      - 3 個新 endpoint（`/imitation/auc_history` / `/shap_top_features` / `/divergence_stats`）
      - PredictionDashboard「🤖 模型狀態」tab 加「📊 模型觀察」區塊：純 SVG sparkline AUC 趨勢（不引 recharts）+ SHAP top features bar + 分歧次數 7d/30d/全期三格卡
    **Phase 5 策略 + 跨 symbol 視圖（commit f70bc90）**：
      - `/strategy_performance` 從 regime_stats 推算策略類型勝率（趨勢 / 均值回歸 / 動量突破），標記最強 / 最弱
      - `/cross_symbol_rs` 用本地 OHLCV 算 BTC-relative return（預設 4h × 30d）
      - 新 tab「📊 策略績效」雙向 RS bar（中線分多空）
    **Phase 6 事件 + 資料品質（commit dfc83ff）**：
      - 經濟日曆（不付費）：[backend/data/calendar/events.json](backend/data/calendar/events.json) 手動維護 + [event_injector.py](backend/app/core/event_injector.py) 注入 chart_state.upcoming_events（72h 內 ≥ medium）；CORE prompt 加事件警示規則（severity=high 在 24h 內 → 強制倉位上限 50%）
      - OHLCV 資料品質監控（[data_quality_monitor.py](backend/app/core/data_quality_monitor.py)）：5σ outlier 偵測 + 自動去重，寫 data_quality.log；每日跟 drift check 一起跑
    **Phase 7 收尾（this commit）**：系統prompt.md 加第 102/103 項；37 pytest 全綠；frontend tsc 零錯
    **設計鐵律**：每個 Phase 獨立 Stop-Safe / schema 純加法 / 不付費（無 Bloomberg / 無 cloud / 無 GPU）/ v100 結論卡格式不變
    **完成後系統能力**：每份「全部分析」具備真實 SHAP + 真相似路徑 + 雙向計劃 + 30 秒結論 + 事件警示 + per-regime 模型 + Dashboard 完整觀察工具

104. ✅ v104 系列（2026-04-28 ~ 04-29）：fact-checker + ranging 子類型 + lean 卡 + bias 9 分量 + 雙向方向機率
    **動機**：v103 後，使用者實測發現多個品質瑕疵：(1) LLM 寫具體數字無對照（fabrication 風險）(2) ranging 場景 90% 都被無腦套雙向計劃，未細分子類型 (3) ATR 倍數固定不分 timeframe / regime / 信心 (4) bias_score 只有 5 分量，訊號粒度不夠
    **Q1（commit 7a20f6d）**：免費資料源整合（衍生品 funding/OI/多空比 + Fear&Greed + FRED 總體 + 批次掃描），注入 chart_state.external_signals
    **Q3（commit ae023b0）**：[fact_checker.py](backend/app/core/fact_checker.py) 新建 — 掃 LLM 文本中具體數值（RSI/MACD/funding/多空比/F&G/命中率），與 chart_state 對照，超容忍度標 mismatch；CORE prompt 加「數值引用鐵律」（v104 Q3）
    **Q4（commit 5596dae）**：prediction_features 加 8 個 lag column（rsi_14_lag5、close_return_5/20 等），給 ML 訓練序列性特徵
    **Fix A（commit 2784fb5）**：timeframe + regime + 信心三維度自適應 ATR 倍數（[laddered_entries.py](backend/app/core/laddered_entries.py) `_get_atr_mults`），含 cap + feature flag
    **Fix B（commit 0dcadfe）**：[regime_subtype.py](backend/app/core/regime_subtype.py) 新建 — ranging 拆 5 個子類型（true_ranging / lean_long / lean_short / breakout_pending / neutral_ranging），後端先算標籤注入 chart_state，LLM 看標籤直接選結論卡
    **Fix C（commit 1d03582）**：6 規則卡片選擇表 + 偏多/偏空獨立結論卡（格式 D/E）+ parser 識別（[prediction_tracker.py](backend/app/core/prediction_tracker.py) `_LEAN_CARD_PATTERN`）；倉位 ×0.7 縮減
    **Fix D + E + F（commit 363ab23）**：每張卡開頭加「🎯 結論卡選擇 + 理由 ≤30 字」+ SL/TP 後標 ATR 倍數 + CORE prompt 重組分區
    **Fix G（commit 8c15380）**：量化驗證腳本 + 補規則涵蓋率
    **4.1（commit cb036fc）**：bias_score 5→9 分量擴展（加均值回歸 IC、breadth、RS、funding squeeze、leadership 等），動態 threshold 依激活分量數調整
    **4.2（commit 39b3de9）**：修 LLM 對 unknown subtype 措辭錯誤 + ladder fallback 加 SL/TP 倍數提示（即使 ladder disabled 也給 LLM 建議倍數）
    **4.3（commit 5355289）**：雙向卡顯示方向機率行（🧭 行）— 從 metrics.bias_score 線性 clip 算 P_long / P_short，倉位按機率不對稱拆分（不再強制對稱）
    **4.4（commit 8798f70）**：修 parser 對 markdown bold 的失敗（解 ETH/ADA 沒存進 DB 根因）
    **設計鐵律**：fact-checker 不阻擋（事後標註）、ranging 子類型純加法、ATR 倍數 cap + flag 可關
    **完成後系統能力**：報告數值有後端 fact-check 護網；ranging 場景按子類型走對應卡（90% 不再無腦雙向）；bias_score 訊號粒度從 5 → 9，雙向卡有方向機率不對稱倉位

105. ✅ v105 系列（2026-04-29 ~ 05-02）：horizon 拆分 + 因子權重學習 + per-regime 校準 + 多個投資邏輯 bug 修復
    **動機**：v104 後實測 ADA 案例發現 ladder 邏輯多個 bug；模型對不同 horizon 樣本混為一談；regime 間機率校準偏差大
    **Phase A（commit 2f5d588）**：predictions 加 `horizon_class`（short/medium/long 持倉分類）+ `regime_std`（標準化 6 種 regime label）+ Coinglass 嘗試式 liq 抓取
    **Phase B（commit f5a9c37）**：[factor_weight_learner.py](backend/app/core/factor_weight_learner.py) 新建 — 因子權重學習器 + PCA 正交化（含 quality gate，PSI / IC / 樣本門檻）
    **Phase C（commit 623f987）**：per-regime isotonic 校準（[per_regime_calibrator.py](backend/app/core/per_regime_calibrator.py)）+ walk-forward + 32 unit tests
    **5.1（commit 94b4d5a）**：bias_score 動態 threshold（依 9 分量激活數調整：n≤2→0.20、n=3→0.30、n≥4→0.40）+ ladder 最小 spacing 守衛（< 0.5×ATR 強制拉開避免重合）
    **5.2（commit fa161c8）**：修 ladder 兩個邏輯 bug（Bug A：long entries 高於 current_price → clamp、Bug B：依最終位置重新分配 ratios）
    **5.3（commit 9d4e43c）**：投資邏輯 bug 全面審查 + 19 個新 unit tests 補強
    **5.4（commit e093a70）**：修經濟日曆 NFP 日期錯誤 + 加過期警示流程（calendar_meta 注入 chart_state）
    **5.5（commit 13eeec0）**：條件機率掃描加 Wilson CI + Bayesian shrinkage（解決小樣本誤導）
    **5.6（commit 5d657c0）**：修「全部分析後續查詢無回應」前端佇列卡死 bug
    **5.7（commit ce1b0de）**：修 `_format_function_results` 3000 字截斷導致 LLM 誤判工具失敗
    **設計鐵律**：因子權重學習過 quality gate 才上線；horizon / regime 純加法欄位
    **完成後系統能力**：模型分 horizon 訓練；regime 間機率校準準確；ladder 邏輯經實戰驗證；條件機率有信心區間

106. ✅ v106 系列（2026-05-04，5 階段）：策略品質升級 + 精準護網 + 體感優化 + 安全效率 + CI 防 regression
    **動機**：累積多輪優化後需系統性品質提升 + 防止 regression，分 5 階段做完整套裝
    **階段 1（commit e74a8fc）— A1+A2+A3 策略品質升級**：策略生成器強化 / 多策略對比改進 / 回測指標精細化
    **階段 2（commit 5a50d47）— B1+B3+B4 精準護網 + 體感優化**：護網規則補強 / UI 響應優化 / 訊息表述精準
    **階段 3（commit 4aaafae）— C1+C2+C3+C4 精準性深化 + 穩定性**：訊號精準度提升 / 穩定性護網 / 邊界情況處理
    **階段 4（commit ca7cbd0）— D1+D2+D3 安全 + 效率**：安全防護強化 / 效率優化 / 資源管理
    **階段 5（commit b61157a）— CI 防 regression**：[test_v106.py](backend/tests/test_v106.py) 12 個 smoke tests，每階段 commit 必過
    **設計鐵律**：每階段獨立 Stop-Safe + smoke test 自動防 regression
    **完成後系統能力**：策略品質、護網、效率、安全四大維度系統性升級，CI 自動把關

107. ✅ v107（commit b992056，2026-05-04）：公正性 + 可靠性 + 資料誠實度
    **動機**：v106 後使用者反饋系統有時「對使用者立場有偏向」、「指標來源不誠實揭露」、「可靠性表述需更精準」
    **v107.1 公正性**：修正 LLM 對使用者持倉產生的偏向（注入 portfolio_summary 為**客觀組合風控**而非「使用者立場」），避免基於使用者多空比反向推論方向；CORE prompt 加引用規則（不可用組合推論方向）
    **v107.2 可靠性**：[chat.py](backend/app/api/routes/chat.py) `chart_state["portfolio_summary"]` 注入時嚴格區分「組合層風控」vs「單一標的方向」；風控警示精準化（多空比 > 3 / freshness 警示 / 集中度 > 30%）
    **v107.3 資料誠實度**：CORE prompt 強化「資料來源透明」原則，每個指標、訊號、命中率都需明確標註來源（chart_state 欄位 / 工具回傳 / 計算式）；fact-checker 警示優先順序提升
    **設計鐵律**：誠實表述、不偏向、揭露邊界
    **完成後系統能力**：報告中所有客觀數字、訊號、判斷都有可追溯來源；組合層資訊不再誤導單標的方向判斷

108. ✅ v108（2026-05-06）：投資分析報告 3 點優化（不對稱降階 + 客觀數值強制 + 失效條件 watcher）
    **動機**：使用者實測 ranging 雙向計劃報告，發現 3 個直接影響投資決策品質的問題：(1) 客觀數值（區間位置 %、雙向計劃進場分批價）由 LLM 自編而非系統算 → 用戶可能基於虛構數字下單 (2) 失效條件（如「跌破 $X 切換做空」「FOMC 12h 取消」）純文字裝飾、系統不處理 → 用戶以為自動處理但實際全手動 (3) 「資料不足」hardcode 走「視為對稱 50/50」雙向 = 反 Kelly 設計 → 必有一邊先打 SL，期望值為負
    **Phase 1（[function_defs.py](backend/app/core/llm/function_defs.py) 改路由）**：把 `bias_score 缺失 → 對稱 50/50` hardcode 改為三分支決策樹：(1) `chart_state.donchian_position_pct` 可用 → 走新格式 B-asymm「🔀 雙向計劃（不對稱·依區間位置加權）」（`short_ratio = clamp(0.5 + (pct/100 - 0.5), 0.30, 0.70)`，倉位按比例不對稱拆）(2) `metrics.bias_reasons` 仍有方向 → 走 lean 卡（格式 D / E，倉位 ×0.7）(3) 完全無訊號 → 走觀望卡（格式 C，不寫 prediction）；新格式 B-asymm 標題「🔀 雙向計劃（不對稱·...）」刻意以「🔀 雙向計劃」開頭，相容既有 `_BILATERAL_CARD_PATTERN` regex
    **Phase 2（chart_state 注入 + 強制插值 + fact_check 收緊）**：
      - [chat.py](backend/app/api/routes/chat.py) regime_subtype 後注入 `donchian_position_pct`（標準 Donchian-20 算）+ `donchian_upper`/`donchian_lower` + `bilateral_plan`（ranging/unknown 時呼叫既有 `compute_laddered_entries(direction="both")`）
      - [function_defs.py](backend/app/core/llm/function_defs.py) v104 Fix E 後加「v108 Fix」強制規則段：「位於區間 X% 位置」必抄 `chart_state.donchian_position_pct`、雙向進場分批價/SL/TP/RR 必抄 `chart_state.bilateral_plan`，禁止 LLM 自估自編
      - [fact_checker.py](backend/app/core/fact_checker.py) 容忍度從統一 ±10% 收緊為依指標分組：振盪類（RSI/Stoch/MFI/ADX）±2 absolute、趨勢類（MACD/ATR/CCI）±5% relative、通道類（BB）±2% relative；新增 `donchian_position_pct` 校驗（±1 absolute）
      - [chat.py](backend/app/api/routes/chat.py) fact_check 結果從只發 SSE event 升級為串流可見區塊到報告底部（「═══ ⚠️ 數值校驗異常 ═══」+ mismatch 列表，最多 8 條）
    **Phase 3（失效條件 watcher）**：
      - [prediction_tracker.py](backend/app/core/prediction_tracker.py) schema 加 3 欄位（`invalidation_json` / `invalidated_at` / `invalidation_trigger`）+ `parse_invalidation_to_json()` 把 LLM 寫的失效條件文字解析為結構化 conditions（價格條件 + 量過濾），支援「突破/跌破/高於/低於/上方/下方/超過/⬆/⬇/`> $`/`< $`」+「放量 > N×」+「；/ → / ;」分段；hit-rate 計算所有 query 從 `WHERE status != 'active'` 改為 `WHERE status NOT IN ('active', 'invalidated')`
      - 新建 [watcher.py](backend/app/core/watcher.py) — `sweep_symbol(symbol, tf)` 對 active 預測逐根 K 線比對失效條件；風險控制：60 分鐘最小存活時間（防剛開單就被 wick）+ 2 根 K 線連續確認（防單根 wick 誤觸）；shadow mode（`settings.imitation_shadow_mode=True` 時只 log 不改 status）；`startup_sweep()` 啟動冷啟動掃所有 active
      - [data_sync.py](backend/app/api/routes/data_sync.py) 每個 symbol+tf 同步完成後 hook 觸發 `sweep_symbol(symbol, tf)`；[main.py](backend/app/main.py) lifespan `_background_init` 加 `startup_sweep()` 不阻塞主流程
      - [chat.py](backend/app/api/routes/chat.py) `_inject_recent_accuracy` 另查 invalidated 筆數，> 0 時報告底加「📛 另 N 筆因失效條件觸發已排除（不計入命中率）」
    **設計鐵律**：用既有設施（lean 卡 / laddered_entries / fact_checker / function-call schema 都已存在但用得不夠），只新建必要的 watcher；watcher shadow mode 預設 ON 等實證觀察 1-2 週（觸發 ≥ 10 次、假陽性率 < 20% 才正式啟用，假陽性定義：觸發後 4h 內價格回到觸發前 ±0.5×ATR 範圍）；schema 純加法（既有 query 不影響）
    **完成後系統能力**：報告中所有客觀數值（區間位置、雙向進場分批價、TP/SL/RR、ATR 倍數、技術指標數值）都由系統算後注入並強制 LLM 抄，編造後系統會 fact-check 標警告區塊；失效條件由系統實際監控、自動標 invalidated 不污染 hit-rate 統計、報告中可見「📛」提示；資料不足時保留多級降階可用性（不對稱雙向 / lean / 觀望），不再無腦對稱 50/50 反 Kelly

109. ✅ v109（2026-05-06）：經濟事件日曆改造 — 從手動維護改為純自動同步（只收可驗證事件）
    **動機**：v108 完成後使用者實測報告，發現事件警示出現 (1) FOMC `2026-05-07` 是不存在的會議（2026 年 5 月根本沒 FOMC 會議，4/29 後直接跳到 6/17）(2) NFP `2026-05-08` 日期錯（5 月 NFP 實際是 5/1 已過，下一場是 6/5）(3) `2026-06-18 FOMC` 也錯（Fed 官方 6/16-17，會議結束日是 6/17）。根因是 [events.json](backend/data/calendar/events.json) 純手動維護易出錯，且 v105.4 加的 `unverified: true` flag 雖然標警示但 LLM 仍會引用 → 警示稀釋。**v107 誠實表述路線延伸：不講 > 講錯但加警示**
    **事件分類策略**（按可行性決定處理方式）：
      - **A 類（央行類）**：HTML 結構爬蟲可行 — Fed `class="fomc-meeting"` 結構穩定，預先公告全年會議
      - **B 類（固定規則類）**：規則計算比爬蟲還可靠（NFP=每月第一週五、ISM=第一/三工作日），BLS anti-bot 擋住爬蟲反而是好事
      - **C 類（爬不到 + 規則粗糙）**：CPI / PPI / Retail Sales / GDP / PCE — BLS/BEA 有 anti-bot、規則只能逼近 → **不收錄**，改 system prompt 通用提醒
      - **D 類（不規律）**：OPEC / 地緣政治 → 不在 events.json 範疇（既有 social_sentiment + news 處理）
    **Step 1 立即修 events.json**：刪假 May FOMC entry、移除過期錯日 NFP、修 June FOMC 日期 6/18 → 6/17、加正確 6/5 NFP；schema 加 `source_strategy` 欄位區分自動/手動來源
    **Step 2 重寫 [event_calendar_sync.py](backend/app/core/event_calendar_sync.py) `fetch_fomc_dates()`**：舊策略用 `fomcpresconf[YYYYMMDD].htm` URL pattern 只能抓**已發生**會議（Fed 在會議結束後才上傳 press conference URL）→ 改用 `soup.find_all(class_="fomc-meeting")` 解析 HTML 結構（`fomc-meeting__month` + `fomc-meeting__date`），取結束日為利率公告日；含 SEP/Dot Plot 場次（`*` 標記）自動加「+ Dot Plot」名稱；實測抓到 8 筆未來會議含 2026 6/17、9/16、12/9 等含 Dot Plot 場次
    **Step 3 新增 `compute_nfp_dates(months_ahead=6)` 規則計算**：每月第一個週五 8:30 ET (12:30 UTC)；含美國聯邦假日推遲邏輯（如 2026/7/3 因 7/4 國慶假日推遲到 7/6）；NFP 名稱用「上一個月」（5/1 公布 April NFP）；舊 `fetch_nfp_dates()` BLS 爬蟲保留作 fallback（雖 anti-bot 擋住）
    **Step 4 新增 `sync_verifiable_events()`**：跑 fetch_fomc + compute_nfp + 過期清理（`date < today` 自動移除）+ 合併保留 `source_strategy="manual"` entries（自動同步不覆寫使用者手動加的）+ 寫回 events.json 加 `_meta.last_auto_sync` 時間戳；全失敗時 graceful fallback 保留現有 entries 並標 unverified
    **Step 5 [main.py](backend/app/main.py) lifespan startup hook**：`_background_init` 加 `sync_verifiable_events()`（與 v108 watcher.startup_sweep 並列），啟動時自動跑事件同步，不阻塞主流程
    **Step 6 [function_defs.py](backend/app/core/llm/function_defs.py) CORE prompt 加「v109 未收錄事件提醒」**：在既有「事件警示規則」段（line ~316）後加段落，明列 CPI/PPI/Retail Sales/GDP/PCE 不在 events.json；觸發條件 3 種（使用者問事件 / 月中 10-25 日 / BTC-ETH macro 分析）需提醒「⚠️ 系統未收錄通膨/總體數據，下單前請自行查 BLS/BEA 行事曆」；**絕對禁止**自編這些事件具體日期（會被 fact-checker 標 mismatch）
    **設計鐵律**：A+B 類自動化覆蓋 ~80% 高頻重要事件（FOMC 月會 + NFP 月發 + ISM 月發 + Jobless 週發），C 類維持手動模式但改通用提醒避免誤觸；既有 [event_injector.py](backend/app/core/event_injector.py) 過期過濾邏輯（line 75 `if ev_dt < now_utc: continue`）已充分，雙重保護不需新增；HTML 結構雖穩定但不是合約，scraper 失敗 → 保留舊 events + log warning，不阻塞主流程
    **完成後系統能力**：FOMC 從手動維護升級為 Fed 官方來源自動同步（HTML 結構穩定可靠）；NFP 用規則計算比爬蟲還準（含假日推遲）；CPI/PPI/GDP 等不再出現「unverified 但 LLM 仍引用」的假事件警示，改由通用提醒讓使用者主動核對；events.json 自動清過期、保留手動 entries、可追溯來源；報告中事件警示再無不存在會議或日期錯誤情況

110. ✅ v110（2026-05-07）：streaming 中斷防護 + executor 並行結果對應修復 + 事件 prompt index mismatch 修復
    **動機**：v109 後使用者實測「全部分析」遇到多種 streaming 中斷症狀：(1)「請繼續」連按沒反應 (2) 「⚠️ AI 分析完成但未能產生文字報告」(3) `[client_disconnect]` log 頻繁出現。深入排查發現多個獨立 bug 互相疊加。
    **問題 1：前端切換 symbol/timeframe 無聲打斷分析**
      - 使用者在 streaming 進行中切換標的 → 前端某條路徑觸發 SSE 連線 abort → 後端看到 `Cancelled by RequestResponseCycle.run_asgi` → LLM 二輪被打斷沒輸出 → 走 fallback「未能產生文字報告」
      - **修法**：[TopBar.tsx](frontend/src/components/TopBar.tsx) 新增 `confirmAndSwitch` helper，4 個切換點（symbol selector / timeframe button / 起訖日期 input）都包進保護；[ChatInterface.tsx](frontend/src/components/ChatInterface/ChatInterface.tsx) useEffect 監聽 `force-abort-streaming` window event 走既有 `handleAbort` 完整清理路徑（重用 v105.6 兜底邏輯）
    **問題 2：streamChatMessage 連線中斷靜默結束**
      - reader 異常結束時前端只 onDone 不 onError → 使用者看不到任何錯誤提示，只看到「未能產生文字報告」誤以為 LLM 不會回
      - **修法**：[api.ts](frontend/src/services/api.ts) `streamChatMessage` 區分「reader 自然結束」與「timeout 觸發」（用 `timeoutFired` flag）；natural close 時主動 `onError("⚠️ 分析中斷：連線提早斷開（可能原因：切換標的/週期、網路波動、browser tab 斷連）")`，使用者明確知道發生什麼
    **問題 3：「尚無數據」UX 配套**
      - 使用者切到無資料 timeframe 時只看到「前往同步數據」按鈕，但常常該標的其他週期已有資料
      - **修法**：[ChartView.tsx](frontend/src/components/ChartView/ChartView.tsx) useEffect 拉 `/api/chart/available/list`，過濾該 symbol 其他可用 timeframe；空狀態加「💡 此標的其他週期已有資料：[切到 4h]」快速切換按鈕（重用既有 API 不新增後端）
    **問題 4：executor results 順序錯位導致誤標未授權函式**
      - [executor.py](backend/app/core/llm/executor.py) `execute_function_calls` 的 `results` 用 `.append()` 累積，sequential / parallel / reject 三條路混合 push；但 [`_format_function_results`](backend/app/api/routes/chat.py#L1084) 用 `enumerate(function_calls)` 對 `results[i]` 配對 → **index 完全錯位**
      - 使用者看到 log「函式 1: detect_smc_structure 錯誤: 未授權的函式呼叫」是**誤報**：實際被 reject 的是別的 function（LLM 可能寫了 case mismatch 或 typo），但 error 訊息被掛到 detect_smc_structure 上
      - **修法**：[executor.py](backend/app/core/llm/executor.py) 改 by-index 寫入 — `results = [{} for _ in function_calls]` 預先分配；sequential / parallel / reject 都用 `results[idx] = ...` 而非 `append`；parallel 用 `asyncio.gather` 保證順序對應，`zip(parallel_calls, parallel_results)` 寫回原始 idx；[`_format_function_results`](backend/app/api/routes/chat.py#L1084) 改用 `result.get("function")` 顯示真實 name，不一致時 log warning
    **問題 5：function name validate 不容忍 LLM 偶爾的 trailing/leading 空白**
      - LLM 偶爾輸出 `"detect_smc_structure "`（含尾隨空白）→ `name not in ALLOWED_FUNCTIONS` → reject
      - **修法**：[executor.py:validate_function_call](backend/app/core/llm/executor.py#L122) 加 `name.strip()` 容忍空白後再比對；caller 端把 stripped name 寫回 fc dict 確保所有後續比對（`_PARALLEL_FUNCS` 等）都用乾淨 name；case mismatch 仍拒絕（屬於 API contract 應該嚴格）
    **問題 6：HMM 分析模組缺失**
      - log 顯示 `HMM 分析失敗: No module named 'hmmlearn'` — 量化研究跳過 HMM 狀態轉移層
      - **修法**：`.venv/bin/pip install hmmlearn` (0.3.3)，補上 HMM 模組讓 quant_research 完整跑
    **問題 7：function call 階段 progress 心跳頻率太低**
      - quant_research 等重型 function call 可能跑數分鐘，期間每 6 秒一次 progress event，client 端某層偶爾誤判 idle
      - **修法**：[chat.py](backend/app/api/routes/chat.py) function call progress 迴圈每 2 秒強制 yield SSE event（從 6s 提高），確保 SSE 連線在長 function call 期間頻繁有 byte 流動
    **問題 8（已知殘留，未根治）：LLM 二輪 streaming 期間 client 真斷線**
      - log 確認某些情境下 client 真的關了 HTTP 連線（`Cancelled by RequestResponseCycle.run_asgi`，anyio cancel scope from ASGI）
      - 嘗試過 `_stream_with_heartbeat` 包裝（含 `asyncio.shield`），但 anyio cancel scope 是 task tree 整體 cancel，shield 擋不住 → 已**回退**該 helper（保留定義但 3 個 caller 都改回 raw `async for adapter.chat_stream_events()`）
      - **替代措施**：[chat.py CancelledError handler](backend/app/api/routes/chat.py#L2781) 加詳細狀態 log（`round2_started` / `r2_text_len` / `r2_function_calls` / `has_response`），下次中斷可精準定位斷點；[adapter.py:1591](backend/app/core/llm/adapter.py#L1591) `[client_disconnect]` 訊息中性化避免誤導（CancelledError 來源除了 client 真斷還可能是 wait_for cancel 或正常 generator close）
    **設計鐵律**：問題 1-7 是真 bug 必修；問題 8 真因仍待精準診斷（可能是前端切換保護沒生效、prompt 過大導致 TTFT 超時、browser fetch 內部行為等），加 log 等下次重現有更完整資訊
    **完成後系統能力**：分析 streaming 不再被切換 symbol 等動作無聲打斷；連線異常斷開有明確錯誤訊息給使用者；executor function call 結果按原始順序對應，不再誤標未授權；validate 容忍空白；HMM 分析補回；function call 階段 SSE 流量更密集

### 待開發功能

- ⬜ CI/CD 流程建立
- ⬜ 鏈上數據整合（地址分析、大戶持倉等）
- ⬜ Google 趨勢整合

---

## v110 → v122 變更紀錄

> 自 2026-05-07 起新增的修復與功能。原本章節未動，僅在末尾追加紀錄，避免破壞既有結構。

### v118 — 修「看漲說漲」bias（三道 prompt 防線）

- `regime_warning`：trending_up + RSI 超買 + StochRSI 過熱時注入「警戒回調」訊號，避免追高
- `direction_balance`：統計近 30 日方向預測的多空比例，過度偏多時提示重新檢視
- 三道 prompt 防線（[chat.py:817-866](backend/app/api/routes/chat.py#L817-L866)）：注入到 chart_state，LLM 必須在輸出前檢視
- 對應 commit：`4fdb966 v118`

### v119 系列 — external_signals 永遠注入

- v119.1：external_signals 加 retry + stale cache fallback（修「無衍生品快照」）
- v119.2：external_signals **永遠注入**（[chat.py:395-415](backend/app/api/routes/chat.py#L395-L415)）+ prompt 強制 LLM 引用
- v119.3：止損位風險評估 prompt（流動性獵取警告）
- v119.4：Coinbase Premium fetcher（美國機構承接訊號）

### v120 系列 — signal bucket 分類 + 訊號歷史回填

- v120.1：predictions schema 加 `signal_at_entry` 欄位 + JSON 彈性容器
- v120.2：signal bucket classifier（funding / OI / premium / long_short / ob / fear_greed / etf）
- v120.3：prediction_tracker.store 改寫 capture external_signals 進新欄位
- v120.4：歷史回填腳本（fear_greed + funding，免費全歷史 API）
- v120.5：`get_signal_combo_stats` + chat.py 注入 `signal_history`（每 request 多 10-24KB）
- v120.6：function_defs.py 加 v120 訊號組合命中率警告規則（最後一道防線）

### v111 — 指標視覺權重 + 動態 ATR + 偏好追蹤

- 指標 registry 加 `visual_weight` 欄位（primary / secondary / minor）
- 前端按權重渲染：minor 細透明、secondary 普通、primary 粗
- `_dedupe_close_annotations` 加動態 ATR threshold（取代寫死 1%，依標的波動性調整）
- 前端 chartStore 加 `getPreferredIndicatorTypes()`，localStorage 追蹤 7 天指標使用偏好
- 對應 commit：`b157e2f` / `d21a6e0`

### v122 系列 — Stream 中斷終結 + 自動分段 + 4 策略附錄

#### v122.1 修復 v118-v120 chart_state 疊加導致的 round2 stream 中斷
- 真因：v118-v120 累積把 `regime_warning` / `direction_balance` / `external_signals` / `signal_history` 都塞進 chart_state，round2 仍完整重傳，prompt 從 100KB 膨脹到 200-300KB，Claude CLI TTFT 過長前端中斷
- 修法：`_minimal_r2_chart_state()`（[adapter.py](backend/app/core/llm/adapter.py)）只保留 symbol / timeframe / currentPrice / currentRegime / recent_accuracy 摘要，移除 signal_history 等 round1 已用過的細節
- 對應 commit：`36e0e65`

#### v122.2 加分段自動接續輸出
- `comprehensive_analysis_seg1` prompt → 先輸出「30 秒結論」（方向 / 信心 / 倉位 / 進場 / 止損 / 失效）
- `comprehensive_analysis_seg2` prompt → 接著輸出「完整詳細分析」（10 個段落）
- 後端內部自動接續第 2 段，不關閉 stream，前端收到 `segment_complete` SSE 事件顯示分隔線
- 對應 commit：`78f9237`

#### v122.3 4 種高機率進場策略附錄
- 策略 A：SMC Demand Zone 回測進場（trending_up 已超買時的回調）
- 策略 B：RSI 雙背離 + 爆量確認（反轉訊號 + 真實買盤）
- 策略 C：三重確認突破（突破 + RSI > 50 + 量放大）
- 策略 D：均值回歸（BB 下軌 + RSI < 30 + StochRSI 反轉）
- 每策略含：適用條件 / 進場價 / 止損 / 停利 / Wilson 勝率參考
- 對應 commit：`78f9237`

#### v122.4 partial-save 失效 bug 修復
- 真因：[chat.py:2974](backend/app/api/routes/chat.py#L2974) 內層 `except (Exception, asyncio.CancelledError)` 把 CancelledError 一起吃掉，外層 partial-save 從未生效
- 修法：拆成 `except asyncio.CancelledError: raise` + `except Exception` 兩個獨立 handler
- 同時修 [chat.py:3097](backend/app/api/routes/chat.py#L3097) partial-save 合併 `final_text` + `_r2_text_buf`（round2 內容也存進 DB）
- 對應 commit：`78f9237`

#### v122.5 `_stream_with_heartbeat` 重新啟用
- v110 因 anyio cancel scope 問題回退此 helper、改用 raw `async for`
- v122 重新啟用、加 `asyncio.shield` 包裹 round2 stream loop（[chat.py:2629](backend/app/api/routes/chat.py#L2629)）
- 同時 [adapter.py](backend/app/core/llm/adapter.py) 3 處 `_line_timeout = 300` → `600`，與前端 STREAM_TIMEOUT_MS 600s 對齊
- 對應 commit：`704b354`

#### v122.6 HARD_TIMEOUT_MS 630s → 1800s
- 真因：[ChatInterface.tsx:828](frontend/src/components/ChatInterface/ChatInterface.tsx#L828) `HARD_TIMEOUT_MS = 630_000`（10.5 分鐘）寫死，「全部分析」實際需 15-20 分鐘（round1 LLM 8m + function calls 2m + round2 兩段 5-10m）
- 之前所有 round2 / SSE / 分段修復都救不了，因為這是「整個 stream 最多 10.5 分鐘」的硬性上限
- 修法：改成 1800_000（30 分鐘），真正 idle 仍由 [api.ts STREAM_TIMEOUT_MS 600s](frontend/src/services/api.ts#L273) 兜底
- 對應 commit：`90c9d16`

#### v122.7 `_auto_calc_indicator_values` 內 chart_symbol → symbol（長期 NameError）
- 真因：函式內 10 處用 `chart_symbol`（不存在於 scope）而非 local `symbol`，每次 NameError 被 try/except 吞掉
- 影響：`external_signals` / `social_sentiment` / `regime_subtype` / `historical_insights` 四個 chart_state 欄位**從未成功注入**到 LLM
- 修法：line 292-628 範圍內 10 處 `chart_symbol` → `symbol`
- 對應 commit：`3bdf6f8`

#### v122.8 `_exec_quant_research` 加細粒度 log + timeout
- 加 GMM / GARCH / HMM / `calculate_dynamic_positions` / `_generate_conclusion` 個別計時與 30 秒 timeout 保護
- function call 階段心跳訊息附耗時，讓使用者看到「量化研究進行中... 5s / 10s / 15s」
- 對應 commit：`1219de1`

### 治理層（v122 同期建立）— 防止技術債再次累積

- **[CLAUDE.md](CLAUDE.md)**：專案根目錄的 governance rule（包含 refactor 配額、shadow flag 規範、chart_state schema 變更流程）
- **[backend/docs/SHADOW_FEATURES.md](backend/docs/SHADOW_FEATURES.md)**：「flag=False ≠ dead code」教訓記錄（避免再次誤刪 shadow mode 中的 imitation_learning）
- **[backend/docs/CHART_STATE_SCHEMA.md](backend/docs/CHART_STATE_SCHEMA.md)**：27 個 chart_state 注入欄位的 schema 文件（產生位置 / 消費位置 / 是否進 round2 / 大小級別）
- **[backend/scripts/check_repo_health.py](backend/scripts/check_repo_health.py)**：pre-commit hook 檢查項目
  - 檔案 > 4000 行 block / > 3000 行警告
  - chart_state 注入欄位 > 30 警告
  - `_PROMPT_MODULES` > 35 警告
- **[backend/scripts/install_git_hooks.sh](backend/scripts/install_git_hooks.sh)**：一鍵安裝 pre-commit hook
- **[backend/scripts/shadow_mode.py](backend/scripts/shadow_mode.py)**：alpha-touching 改動的 pre-flight check 工具（跑 shadow mode 比對 CPCV PF baseline）
- **[.gitignore](.gitignore)**：補完規則涵蓋 `*.db` / `*.pkl` 衍生物 / `events.json` 自動同步
