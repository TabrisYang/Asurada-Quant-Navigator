# 阿斯拉量化系統 Asurada Quant Navigator

一個加密貨幣 + 台股的 AI 量化分析平台。透過自然語言與 AI 助手互動，在 K 線圖上進行專業級技術分析和機構級量化研究。

> 使用者不需要學習 Pine Script，只需用自然語言描述需求，AI 就能在圖表上呈現結果並提供數據驅動的投資建議。

---

## 功能特色

### 8 種分析模式
| 模式 | 內容 |
|------|------|
| 基礎分析 | 市場環境 + 八維度技術分析 + SMC 智慧資金 + 情境預測 |
| 因子驗證 | 因子 IC 排名 + 組合 IC + Bucket 評分 + 條件機率 |
| 策略回測 | 多策略比較 + Monte Carlo + Walk Forward + CPCV |
| 市場體制 | GMM Regime + GARCH 波動率 + HMM 狀態轉移 |
| 基本面 | 月營收 + 法人買賣超 + 財報指標（台股限定） |
| 動能分析 | 多週期動量 + 加速度 + 相對強弱 + 反轉偵測 |
| 完整分析 | 三階段全面分析 |

### 核心能力
- **30+ 技術指標**：RSI、MACD、BB、ADX、SMC 訂單流等
- **六大量化模組**：衍生品數據、精細化標籤、GMM/HMM/GARCH 市場體制、VIF+SHAP 因子分析、Brier/ECE 機率校準、CPCV 交叉驗證
- **多資產支援**：加密貨幣（10+ 幣種，五交易所投票）+ 台股（所有上市櫃，TWSE 官方資料）
- **台股基本面**：月營收、三大法人買賣超、外資持股、EPS/本益比/殖利率
- **21 個 Function Tools**：AI 助手可操作圖表、回測、下載數據、分析族群
- **對話觸發下載**：直接在對話中說「下載國巨的數據」即可抓取

### 技術架構
- **前端**：React 18 + TypeScript + TradingView Lightweight Charts v5
- **後端**：FastAPI + Python + SSE Streaming
- **LLM**：支援 OpenAI / Google Gemini / Anthropic Claude / Claude 訂閱制 / Ollama 本地
- **ML**：LightGBM + XGBoost + Random Forest + GMM + HMM + GARCH
- **數據**：五交易所投票機制 + TWSE/TPEx 官方 API + yfinance

---

## 快速開始

### 環境需求
- Python 3.10+
- Node.js 18+
- macOS / Linux

### 安裝步驟

```bash
# 1. 下載
git clone https://github.com/TabrisYang/Asurada-Quant-Navigator.git
cd Asurada-Quant-Navigator

# 2. 一鍵啟動（macOS）
chmod +x 阿斯拉量化系統.command
open 阿斯拉量化系統.command
```

或手動啟動：

```bash
# 後端
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py

# 前端（另一個終端）
cd frontend
npm install
npm run build
```

### 設定 LLM
1. 開啟瀏覽器 `http://localhost:5173`（dev）或 `http://localhost:8000`（build）
2. 點右上角「設定」
3. 選擇 LLM 供應商 → 輸入 API Key → 偵測模型 → 儲存

---

## 截圖

系統介面包含：
- 左側：K 線圖表 + 30 種技術指標（可調參數）
- 右側：AI 分析助手對話框 + 8 種分析模式
- 上方：標的選擇 + 時間框架切換 + 同步數據

---

## 授權

本專案採用 [CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/) 授權。

- ✅ 可以自由查看、學習、非商業使用
- ✅ 可以修改和分享（需標註來源）
- ❌ 不可用於商業目的（需取得授權）

商業使用請聯繫作者。

---

## 免責聲明

本系統僅供學習和研究使用，不構成任何投資建議。使用者應自行承擔投資風險。
