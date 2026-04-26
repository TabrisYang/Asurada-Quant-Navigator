# 阿斯拉量化系統 — 完整系統 Prompt 與架構規格書

> **最後更新：2026-04-25** — 與實際程式碼同步

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
│  分析模式：基礎分析 / 因子驗證 / 策略回測 /              │
│           市場體制 / 基本面 / 動能分析 / 完整分析         │
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
│  │  + 動態 Intent 偵測（18 種意圖）                │      │
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

### Function Calling 定義（21 個函式）

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
| `error` | 錯誤訊息 |
| `done` | 串流結束 |

### SYSTEM_PROMPT 核心指引

- **多維度交叉分析框架**：8 個維度（趨勢方向、動量超買超賣、量能驗證、波動率關鍵價位、市場情緒與微觀結構、風險管理、市場結構與機構行為、市場體制 GMM/HMM/GARCH），每個維度至少用 2 個指標交叉驗證
- **智慧繪圖策略**：精選原則（每次最多 3~5 組標記）、優先順序、顏色規範、主動繪圖行為
- **批量繪圖範例**：annotate_chart 用 annotations 陣列一次繪多條線
- **draw_pattern 範例**：諧波型態只需 5 個點即可自動連線
- **回測策略多元化**：5 種策略模板（含不同嚴格度 + 必含至少 1 個做空策略），未指定時自動比較至少 4 種
- **指標必須同步添加**：文字提到的指標必須同時呼叫 manage_indicator 添加到圖表
- **知識萃取**：每次分析後自動附加 `---KEY_INSIGHTS---` 結構化知識碎片

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

| 模式 | 內容 | 耗時 |
|------|------|------|
| **基礎分析** | 市場環境 + 八維度技術 + SMC + 情境預測 + GMM/GARCH/HMM | ~15 秒 |
| **因子驗證** | 因子 IC 排名 + 組合 IC + Bucket 評分 + 條件機率 | ~20 秒 |
| **策略回測** | 多策略比較 + MC + WF + CPCV + SHAP | ~30 秒 |
| **市場體制** | GMM Regime + GARCH 波動率 + HMM 狀態轉移 + 事件型態 | ~15 秒 |
| **基本面** | 月營收趨勢 + 法人買賣超 + 財報指標 + 綜合評分（台股限定） | ~10 秒 |
| **動能分析** | 多週期動量 + 加速度 + 相對強弱 + 反轉偵測 + 策略回測 | ~15 秒 |
| **完整分析** | 三階段全跑（最完整） | ~2-3 分鐘 |

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

### 待開發功能

- ⬜ CI/CD 流程建立
- ⬜ 鏈上數據整合（地址分析、大戶持倉等）
- ⬜ Google 趨勢整合
