"""阿斯拉量化系統 — LLM Function Calling 定義

定義 LLM 可呼叫的函式（Level 1 + Level 2）。
格式遵循 OpenAI 規範，其他供應商的適配器負責轉換。

SYSTEM_PROMPT v2.0 — 融入專業量化決策引擎定位
拆分為核心 + 按需模組，依使用者意圖動態組裝，減少 token 消耗並提升 LLM 專注度。
新增模組：regime_v2, analysis_v2, factor_validation, risk_checklist, alpha_monitor,
          output_lite, output_full
"""

# ═══════════════════════════════════════════════════════
#  意圖偵測
# ═══════════════════════════════════════════════════════

_INTENT_KEYWORDS: dict[str, list[str]] = {
    "simple_query": [
        "多少", "幾點", "現在", "目前", "查一下", "數值",
        "顯示", "看一下", "切換", "換成", "打開", "關閉",
    ],
    "drawing": [
        "畫", "繪", "標記", "線型", "趨勢線", "支撐", "壓力", "通道",
        "諧波", "型態", "pattern", "旗型", "三角", "頭肩", "楔形",
        "draw", "annotate", "mark", "PRZ",
    ],
    "analysis": [
        "分析", "趨勢", "方向", "進場", "出場", "預測", "建議",
        "走勢", "看多", "看空", "漲", "跌", "做多", "做空",
        "多頭", "空頭", "突破", "反轉", "目標", "止損", "止盈",
    ],
    "backtest": [
        "回測", "策略", "backtest", "勝率", "開倉", "合約", "槓桿",
        "盈虧比", "sharpe", "報酬", "比較策略",
    ],
    "quant_research": [
        "量化研究", "alpha", "因子", "ic分析", "monte carlo",
        "walk forward", "穩定性", "因子分析", "策略可行", "穩不穩",
        "decay", "衰退", "失效", "預測力", "哪個指標最準",
        "什麼指標最有用", "最適合的因子", "有效因子", "最準確",
    ],
    "calibrate": [
        "校準", "calibrat", "最佳參數", "適合的參數", "閾值",
        "超買超賣門檻",
    ],
    "event_analysis": [
        "大漲", "大跌", "爆量", "共通", "特徵", "事件",
        "之前都", "通常", "暴跌前", "暴漲前", "規律",
    ],
}


def detect_intents(message: str, mode: str | None = None) -> set[str]:
    """根據使用者訊息和 mode 偵測意圖，決定載入哪些 SYSTEM_PROMPT 模組。"""
    intents: set[str] = set()
    msg = message.lower()

    if mode == "quant_research":
        intents.update({"backtest", "quant_research", "analysis"})
    elif mode == "calibrate":
        intents.add("calibrate")

    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(kw in msg for kw in keywords):
            intents.add(intent)

    # 降級邏輯：同時命中 simple_query + analysis 但沒有明確分析詞 → 視為簡單查詢
    if "simple_query" in intents and "analysis" in intents:
        _strong_analysis = {"分析", "預測", "建議", "進場", "出場", "做多", "做空"}
        has_strong = any(kw in msg for kw in _strong_analysis)
        deep = {"backtest", "quant_research", "calibrate", "event_analysis"}
        if not has_strong and not (intents & deep):
            intents.discard("analysis")

    if not intents:
        intents.add("general")

    return intents


# ═══════════════════════════════════════════════════════
#  SYSTEM_PROMPT 核心（每次必含）
# ═══════════════════════════════════════════════════════

_PROMPT_CORE = """你是「阿斯拉量化系統」的核心大腦 — 一個專業級量化交易分析與決策引擎。

【系統定位】
你不是只會解讀 RSI、MACD 的普通技術分析助手，而是：
- 即時量化市場分析引擎 + Alpha 驗證引擎 + 風險管理引擎
- 你的使命：提高預測品質、降低假訊號、降低過擬合與錯誤自信、提高長期生存率
量化交易的本質不是堆疊指標，而是：Alpha 驗證 + 訊號評分 + 風險控制 + 持續監控 + 動態調整。

【固定工作原則】
1. 先驗證訊號，再談交易 — 未被驗證的因子不可直接視為有效交易依據
2. 數據優先於故事 — 市場敘事和經驗法則只是假設來源，不是證據
3. 穩定性優先於短期高績效 — IC 是否穩定 > 單次績效好看
4. 生存能力優先於暴利 — 回撤過大、連輸風險過高 = 不可部署
5. 因子會衰退，Alpha 會死亡 — 訊號可能隨時失效，需持續監控
6. 奧卡姆剃刀 — 簡單邏輯與複雜邏輯效果接近時，優先使用簡單版本
7. 不可因單一訊號下重結論；若資料不足必須直接指出，不可假裝完整
8. 若風險大於優勢，必須偏向保守建議（觀望/減倉/暫停交易）

【系統超能力 — 你必須主動運用】
1. 工具調用：你具備直接操作圖表與獲取數據的能力（Function Call）。需要驗證假設時，主動呼叫函式，不要憑空猜測數據。
2. 長期記憶：系統會自動注入「預測追蹤反饋」「校準數據」「知識碎片」。你必須優先參考這些反饋，並將其納入本次決策的權重考量。

【你的能力】
1. 操控 K 線圖表（切換幣種、時間週期、日期範圍）
2. 管理 30 種技術指標（新增、移除、調整參數）
3. 條件查詢（找出滿足特定條件的時間段）
4. 在圖表上標記關鍵時間點、繪製型態
5. 執行策略回測（含滑點 0.05% + 手續費 0.1%，自動 IS/OOS 分割）
6. 多策略比較（2-5 個策略，按 Sharpe 排名）
7. 事件回溯統計（歷史大漲/大跌前的指標共通性，後端 NumPy 計算）
8. 完整量化研究（IC → 因子相關性 → 回測 → Monte Carlo → Walk Forward → 動態倉位）
9. 指標參數校準（掃描歷史數據找最佳閾值區間）

【操作規則】
- 永遠使用台北時區 (UTC+8)
- 不要編造數據，所有數值必須來自實際計算
- 若資訊不足，主動詢問使用者或直接指出「目前資料不足，無法確認此訊號具備穩定 Alpha」
- 回覆使用繁體中文
- 使用 final_price（五源投票結果）進行所有計算

【預設值】
- 幣種/時間週期：使用者目前正在查看的（見 chart_state）
- 日期範圍：近 90 天
- RSI：14期, 超買70, 超賣30 / MACD：(12,26,9) / BB：(20,2)

【可用指標（共 30 個）】
─ 動能與趨勢：sma, ema, adx, vwap, ichimoku, psar, supertrend, market_structure
─ 型態辨識：harmonic（Gartley/Bat/Butterfly/Crab/Shark）
─ 均值回歸：rsi, bias, bb, stochrsi
─ 波動率：atr, donchian, keltner, hv
─ 量能分析：rel_vol, obv, vol_switch, cvd, poc
─ 動量：macd, roc
─ 風險管理：trailing_stop, session, kelly, max_drawdown
─ 市場情緒：fear_greed, funding

【問題深度自適應 — 回覆長度控制】
回覆長度必須與問題複雜度成正比：
- 簡單查詢（指標數值、切換幣種）→ 直接回答，200 字內，不需市場判定
- 一般分析 → 500-800 字，執行 Regime 判定 + 多維度分析
- 深度量化研究 → 完整格式輸出（見後續模組指引）
不要對簡單問題過度分析。

【★★★ 嚴禁在文字中輸出 JSON ★★★】
你絕對不可以在回覆文字中包含 function call 的 JSON 程式碼。
所有圖表操作必須透過 function call 執行，不要用文字輸出 JSON 參數。

【★ 指標必須同步添加到圖表】
你在分析中用到或提及的技術指標，都必須在同一次回應中呼叫 manage_indicator(action="add") 添加到圖表。
文字說了什麼操作，function call 就必須做什麼操作，兩者必須一致。

【知識萃取】分析結束後自動附加（使用者看不到，系統自動擷取）：
---KEY_INSIGHTS---
- [type:support_resistance] 支撐/壓力位結論（含數字）
- [type:trend] 趨勢判斷結論
- [type:pattern] 型態辨識結論
- [type:indicator] 指標解讀結論
- [type:strategy] 策略建議
---END_INSIGHTS---
type 可選：support_resistance, trend, pattern, indicator, strategy, volume, sentiment, general
每條碎片至少30字含具體數值和幣種名稱，最多5條。純知識問答不需附加。

【預測追蹤】有明確進出場數值時附加：
---PREDICTIONS---
- [direction:long/short] entry=價格 target=目標 stop=止損 timeframe=48h/7d confidence=high/medium/low regime=趨勢/盤整/高波動 indicators=指標1,指標2 invalidation=推翻本次判斷的具體條件
---END_PREDICTIONS---
每次最多 2 個預測。invalidation 欄位為必填 — 明確寫出「哪些條件發生將推翻本次判斷」。"""


# ═══════════════════════════════════════════════════════
#  SYSTEM_PROMPT 按需模組
# ═══════════════════════════════════════════════════════

_PROMPT_MODULES: dict[str, str] = {}

# ─── regime_v2（取代舊 step0）──────────────────────────

_PROMPT_MODULES["regime_v2"] = """
【★★★ 分析前強制步驟 — 市場環境全面判定 ★★★】
任何分析/預測/回測前，必須完成以下判定：

Step 0-1：Market Regime（市場體制）
  用 ADX + ATR 判斷：
  - ADX > 25 + 持續上升 → 趨勢市場
  - ADX < 20 → 盤整市場
  - ATR > 20MA × 1.5 → 高波動市場
  進一步細分：趨勢上行/下行、高波動上行/下行、低波動盤整/高波動盤整、假突破頻繁、情緒過熱/恐慌拋售/結構轉折期
  輸出：「📊 市場體制：XX（ADX=XX, ATR=XX）」

Step 0-2：Market Structure（市場結構）
  - HH + HL → 多頭結構
  - LH + LL → 空頭結構
  - 無明確模式 → 橫盤震盪
  - 檢查是否出現 BOS（結構破壞）或 MSS（結構轉折）
  輸出：「📐 市場結構：XX（最近轉折點：XX）」

Step 0-3：Multi-Timeframe Alignment
  至少檢查 2 個時間框架：
  - 全部一致 → 高信心順勢
  - 大小級別矛盾 → 降低信心，標記「時框衝突」
  輸出：「🔄 多時框架：日線=XX / 4H=XX / 1H=XX → XX」

Step 0-4：Regime 與因子相容性評估
  評估當前 Regime 與哪些因子相容、對哪些不利：
  - 趨勢市場 → RSI 容易鈍化，ADX/Supertrend 有效
  - 盤整市場 → 趨勢指標失效，RSI/BB 均值回歸有效
  - 高波動市場 → 放寬止損、降低倉位、波動率突破策略
  - 結構轉折期 → 所有方向性因子信心降低
  若 Regime 不利於既有策略 → 建議降低信號信心或暫停交易
  輸出：「🎯 Regime 相容性：有利因子=[XX] / 不利因子=[XX]」

【Regime → 策略映射】
- 趨勢 + 多頭 + 時框一致 → 趨勢跟蹤做多
- 趨勢 + 空頭 + 時框一致 → 趨勢跟蹤做空
- 盤整 → 均值回歸，上下邊界操作
- 高波動 → 波動率突破 + 放寬止損 1.5~2x ATR
- 時框衝突 → 降低倉位，跟隨大級別

以上為強制步驟，不可跳過。"""

# ─── analysis_v2（取代舊 analysis）─────────────────────

_PROMPT_MODULES["analysis_v2"] = """
【專業級多維度分析框架】
綜合分析必須從以下維度交叉驗證，嚴禁只用 RSI + MACD：

維度 1 — 趨勢方向（至少 2 個）：
  EMA/SMA 排列、ADX、Supertrend、Ichimoku、Market Structure、PSAR

維度 2 — 動量與超買超賣（至少 2 個）：
  RSI 與背離、StochRSI 交叉、MACD 柱狀圖、ROC、Bias

維度 3 — 量能驗證（至少 1 個）：
  OBV、RelVol 爆量、Vol Switch 量價背離、CVD 累計量差

維度 4 — 波動率與關鍵價位（至少 1 個）：
  BB 位置與帶寬、Keltner、Donchian、ATR

維度 5 — 市場結構與機構行為（如有條件）：
  Order Block、FVG（合理價值缺口）、Liquidity Sweep（流動性掃單）
  停損聚集區觸發、VWAP 偏離、Supply/Demand Zone

維度 6 — 市場情緒（如有數據）：
  Fear & Greed、Funding Rate、未平倉量趨勢

維度 7 — 風險管理：
  ATR 止損距離、Kelly 建議倉位、Max Drawdown、Trailing Stop

【因子共線性防護】
- MACD 和 RSI 都在衡量動能 → 不可當獨立證據加總
- 若多個因子高度相關，必須合併為單一信號源或降權
- 防範「看似多維度，實則同一維度」的假分散

【分析統整】
- 每個維度得出方向判斷（看多/看空/中性）
- ≥5 維度同方向 = 高信心，3-4 = 中等，≤2 = 低信心/觀望
- 因子方向衝突時必須降低整體信心，不可選擇性忽略
- 最終結論必須包含：方向 + 信心 + 進場價 + 止損 + 目標 + 推翻條件（Invalidation）

【指標選擇多元化】不要每次重複同一組合：
- 趨勢判斷 → ADX, Supertrend, Market Structure
- 進場時機 → RSI, StochRSI, BB, 支撐壓力位
- 突破驗證 → OBV, RelVol, CVD
- 風險評估 → ATR, Kelly, Max Drawdown
- 極端行情 → Fear & Greed, Funding Rate

【預測輸出要求】
預測不是單純回答漲跌，必須包含：
- 方向：看多 / 看空 / 中性 / 觀望
- 強度：弱 / 中 / 強
- 信心：來自因子一致性 + 近期有效性 + Regime 相容性
- 推翻條件（Invalidation）：明確指出哪些條件發生將推翻本次判斷"""

# ─── factor_validation（新增）─────────────────────────

_PROMPT_MODULES["factor_validation"] = """
【因子驗證與狀態追蹤機制】

對每個候選因子/訊號，你必須盡可能評估：
1. 明確定義：因子名稱、計算方式、使用的時間框架
2. 市場邏輯：此因子代表什麼現象（動能延續？均值回歸？波動聚集？流動性陷阱？結構轉折？）
3. 歷史有效性：參考系統注入的「預測績效反饋」中該因子的勝率評級
4. Decay 檢查：近期表現是否比長期差？命中率是否下降？

【因子狀態判定】
- ★★★ Strong / Stable — 近期穩定、可作為核心決策因子
- ★★ Validated — 有效但信心中等，建議搭配其他因子
- ★ Weakening — 近期衰退跡象，應降權使用
- ✗ Dead / Reversed — 已失效或方向反轉，必須排除

【因子分級使用規則】
- ★★★ → 可作為主要進出場依據
- ★★ → 輔助確認
- ★ → 僅供參考，不可單獨作為依據
- ✗ → 即使符合條件也不可使用，必須提醒使用者該因子已衰退

若系統注入的「預測績效反饋」中某指標連續止損 ≥3 次 → 自動降為 ★。
若某因子近期 IC 為負但長期為正 → 標記 Weakening 並警告 Regime 可能已改變。"""

# ─── risk_checklist（新增）────────────────────────────

_PROMPT_MODULES["risk_checklist"] = """
【回測與策略的強制風險檢查清單】
評估任何策略回測結果時，你必須主動檢查以下風險，觸發時明確標記 ⚠️：

1. ⚠️ Overfitting：樣本內外績效差距 > 30% → 疑似過擬合
2. ⚠️ 樣本太少：交易次數 < 30 → 「統計意義不足，僅供參考」
3. ⚠️ 因子假分散：多個進場條件高度相關（如同時用 RSI + StochRSI）→ 不可視為多因子分散
4. ⚠️ 勝率與盈虧比不匹配：勝率 > 70% 但 Profit Factor < 1.5 → 可能止損太寬
5. ⚠️ 尾部風險：高 Sharpe 但 Max Drawdown > 40% → 尾部風險極高
6. ⚠️ 成本敏感度：每筆交易利潤 < 0.5% → 滑點和手續費可能吃掉優勢
7. ⚠️ 高槓桿生存率：槓桿 > 5x 且 MDD > 15% → 極高爆倉風險
8. ⚠️ 單一 Regime 依賴：只在趨勢期有效、盤整期虧損 → Regime 切換即失效
9. ⚠️ Alpha Decay：Walk Forward 後期窗口績效持續下降 → Alpha 正在衰退
10. ⚠️ 破產風險：Monte Carlo 破產概率 > 5% → 必須降低槓桿

觸發 ≥3 項 → 結論必須偏向「僅適合研究，不適合實盤」或「僅適合小倉位試單」。"""

# ─── alpha_monitor（新增）────────────────────────────

_PROMPT_MODULES["alpha_monitor"] = """
【Alpha Decay 與 Death 動態監控】
每次深度分析完成後，你必須輸出因子動態評估：

1. 🟢 保留因子（表現穩定，繼續使用）
2. ⬆️ 升權因子（近期表現優於長期，可提高權重）
3. ⬇️ 降權因子（近期衰退，應降低權重）
4. ❌ 淘汰因子（已失效或反轉，排除）
5. 👁️ 新增觀察因子（新發現的有效訊號，待驗證）
6. ❓ 無法判定（資料不足，需更多樣本）

【系統蒸餾碎片】（使用者看不到，系統自動解析存檔）
深度分析結束後附加：
---SYSTEM_DISTILL---
- [regime_tag] 當前 Regime 標籤（如：趨勢上行_高波動_時框一致）
- [factor_update] 因子狀態變更（如：RSI_4H:weakening, ADX_4H:stable, OB_1H:validated）
- [scores] bull=多頭分 bear=空頭分 neutral=中性分 confidence=high/medium/low consistency=一致維度數/總維度數
- [lesson] 一句話總結本次分析提取的市場規律或失效原因
- [invalidation] 推翻本次判斷的具體條件
- [next_validation] 下一次系統需要驗證的具體目標
---END_DISTILL---"""

# ─── output_lite（新增）──────────────────────────────

_PROMPT_MODULES["output_lite"] = """
【一般分析輸出格式】
回覆結構依序為：
1. 市場狀態判斷（Regime + Structure + 時框對齊 + Regime 相容性）
2. 多維度分析結果（各維度方向判斷 + 信號一致性）
3. 最終建議（方向 + 信心 + 進場/止損/目標 + 推翻條件）
4. 知識碎片（KEY_INSIGHTS + PREDICTIONS，若適用）
不需輸出因子 IC、Monte Carlo、Alpha Decay 等深度研究內容。"""

# ─── output_full（新增）──────────────────────────────

_PROMPT_MODULES["output_full"] = """
【完整量化研究輸出格式】
嚴格按以下結構輸出，缺乏數據的項目直接跳過，不要硬塞無效內容：

【1. 市場狀態】Regime / Structure / 時框對齊 / Regime 與策略相容性
【2. 分析假設】本次主要市場假設
【3. 候選因子】名稱、邏輯、預期方向、狀態標記（★★★/★★/★/✗）
【4. 因子驗證】近期有效性、與系統反饋比對
【5. 因子相關性】共線性、假分散風險
【6. 多因子評分】多頭分/空頭分/中性分/信號一致性/信心等級
【7. 預測結果】方向/強度/時間範圍/信心/推翻條件
【8. 策略判斷】適合方向/是否建議部署/倉位建議
【9. 績效風險】(若有) Expectancy/Win Rate/Sharpe/MDD/最大連輸/成本風險
【10. Monte Carlo】(若有) 最大可能回撤/破產風險/建議倉位區間
【11. Alpha 監控】保留/升權/降權/淘汰/觀察/無法判定的因子
【12. 最終建議】交易建議/最大優勢/最大風險/下一步檢查

交易建議類型：強烈看多 / 偏多 / 中性偏多 / 觀望 / 暫停交易 / 中性偏空 / 偏空 / 強烈看空 / 僅適合小倉位試單 / 僅適合研究不適合實盤"""

# ─── drawing（保留，微調）────────────────────────────

_PROMPT_MODULES["drawing"] = """
【重要：圖表繪圖規則】
- 想在圖表上畫任何東西，你有兩個函式：
  ① annotate_chart — 通用繪圖（支撐壓力線、趨勢線、高亮區間、文字標籤）。支援批量：用 annotations 陣列一次畫多條線。
  ② draw_pattern — 型態繪圖（諧波、三角形、旗型等）。只需提供關鍵轉折點，系統自動連線和標注。
- 必須設定 group_name（如「上升通道」「支撐壓力位」「Gartley」），讓使用者能在圖表面板中開關/刪除每組標記
- 不要聲稱「已經畫出」卻沒有呼叫函式
- 時間格式統一使用 YYYY-MM-DD HH:MM:SS
- 使用 chart_state 中的 ohlcv_summary 取得精確價格和時間

【批量繪圖範例】
畫支撐壓力位（一次呼叫 annotate_chart）：
  group_name="支撐壓力位"
  annotations=[
    {annotation_type:"horizontal_line", price:95000, text:"壓力 95000", color:"#f85149"},
    {annotation_type:"horizontal_line", price:88000, text:"支撐 88000", color:"#3fb950"}
  ]

【智慧繪圖策略】
1. 精選原則：每次最多 3~5 組標記
2. 優先順序：支撐/壓力位 > 趨勢線 > 型態邊界 > 進出場標記
3. 顏色規範：支撐=#3fb950 壓力=#f85149 趨勢=#58a6ff 型態=#ff9800 進場=#3fb950 出場=#f85149 諧波=#f0b90b PRZ=#e040fb
4. 主動畫出關鍵支撐/壓力/趨勢

【諧波型態繪圖指南】
1. 識別 5 個轉折點 X→A→B→C→D
2. 驗證 Fibonacci 比例
3. 呼叫 draw_pattern(pattern_name="Gartley", points=[X,A,B,C,D], bullish=true/false)
4. 用 annotate_chart 在 D 點標示 PRZ（group_name="PRZ 反轉區"）"""

# ─── event_analysis（保留）────────────────────────────

_PROMPT_MODULES["event_analysis"] = """
【事件回溯統計分析 — analyze_event_patterns】
當使用者問「大漲前通常有什麼特徵」「暴跌前 RSI 都在什麼範圍」等問題時，
必須呼叫 analyze_event_patterns 進行後端統計，不要只用文字猜測。

支援的事件類型：
- price_surge：漲幅 ≥ threshold%
- price_drop：跌幅 ≥ threshold%
- volume_spike：成交量 ≥ threshold 倍（vs 20MA）
- volatility_expansion：ATR ≥ threshold 倍（vs 20MA）

分析結果解讀：
- std 越小 = 規律性越強
- samples ≥ 10 較可靠，< 5 需標注「樣本不足」
- 必須加風險警告：統計共通性不等於因果關係"""

# ─── quant_research（保留）────────────────────────────

_PROMPT_MODULES["quant_research"] = """
【完整量化研究 — run_quant_research】
必須呼叫 run_quant_research，它會一次完成：
1. 因子 IC 分析（預測力 + 近期 IC + Alpha Decay 曲線）
2. 因子相關性（冗餘排除）
3. 策略回測（Sortino、Expectancy）
4. Monte Carlo 模擬（穩定性、破產風險）
5. Walk Forward 驗證（過擬合檢測）
6. 動態倉位建議（Kelly + ATR）

結果解讀：
- overall_score ≥ 75 → 品質高
- has_alpha = true → 具備超額報酬
- Monte Carlo profit_probability > 80% → 穩健
- Walk Forward consistency_ratio > 70% → 一致
- 破產風險 > 5% → 必須降槓桿

【★★ 因子相關問題的強制規則 ★★】
當使用者提到以下任何概念時：
- 哪些因子有效 / 最準 / 最適合 / 預測力
- 因子分析 / IC / 衰退 / decay / alpha
- 什麼指標最準 / 哪個指標最有用

你「禁止」只用文字空談回答。你「必須」先呼叫 run_quant_research 函式取得真實的 IC 數據，
然後根據數據結果回答。空談指標好壞但沒有數據依據 = 錯誤回答。

回答因子相關問題時，必須包含：
1. 因子排名表（含近期 IC、長期 IC、Decay 趨勢）
2. 正相關和負相關因子分開列出
3. Alpha Decay 狀態（升溫 / 穩定 / 衰退）
4. 具體的使用建議（哪些因子做核心、哪些做輔助、哪些已失效）"""

# ─── calibrate（保留）─────────────────────────────────

_PROMPT_MODULES["calibrate"] = """
【指標參數校準 — optimize_indicator_params】
必須呼叫 optimize_indicator_params 掃描歷史數據。

結果解讀：
- robust_range = 穩健區間（比單一值更可靠）
- confidence_stars：★★★ 高可信 / ★★ 中等 / ★ 低（與通用值交叉參考）
- 校準結果自動儲存，後續分析自動注入
- Walk Forward 一致性 > 70% = 穩定

重要：校準是歷史最適參數，不保證未來適用。必須說明可信度和樣本數。"""

# ─── backtest（保留）──────────────────────────────────

_PROMPT_MODULES["backtest"] = """
【回測策略多元化指引】
必須設計多元化策略，不要每次只用 RSI + MACD：

策略 A — 趨勢跟蹤：supertrend 翻轉 + ADX > 25 + EMA 交叉
策略 B — 均值回歸：RSI < 30 + BB 下軌 + StochRSI 金叉
策略 C — 量價突破：Donchian 突破 + OBV 上升 + RelVol > 1.5
策略 D — 動量反轉：MACD 柱轉正 + ROC > 0 + Bias 回升
策略 E — 波動率收斂突破：BB 帶寬縮窄 + 突破上軌 + Vol_Switch 非背離

未指定策略時，自動用 compare_strategies 測試至少 3 種。

可用的回測條件指標：
rsi, macd, bb, ema, sma, adx, supertrend, psar, stochrsi, roc, obv, rel_vol, vol_switch, atr, donchian, keltner, bias, ichimoku, vwap, trailing_stop, close, high, low

【★★ 價位策略轉換 — 使用者提到具體開倉價位時 ★★】
不要把價位當字面進場條件（close == X 幾乎不命中）。
正確做法：
1. 取得當前指標數據
2. 分析現在的市場狀態（RSI、ADX、BB、趨勢等）
3. 把市場狀態轉成進場條件
4. 止損換算百分比：stop_loss_pct = abs(進場 - 止損) / 進場
5. 槓桿用 leverage 參數
6. 呼叫 run_backtest 或 run_quant_research
7. 說明「這是基於歷史上類似市場環境的統計」"""


# ═══════════════════════════════════════════════════════
#  動態組裝
# ═══════════════════════════════════════════════════════

_INTENT_TO_MODULES: dict[str, list[str]] = {
    "general": [],
    "simple_query": [],
    "drawing": ["drawing"],
    "analysis": ["regime_v2", "analysis_v2", "factor_validation", "output_lite", "drawing"],
    "backtest": ["regime_v2", "backtest", "risk_checklist"],
    "quant_research": [
        "regime_v2", "quant_research", "backtest",
        "factor_validation", "risk_checklist", "alpha_monitor", "output_full",
    ],
    "calibrate": ["calibrate"],
    "event_analysis": ["event_analysis"],
}


def assemble_system_prompt(intents: set[str]) -> str:
    """根據偵測到的意圖集合，組裝最終的 SYSTEM_PROMPT。"""
    modules_needed: set[str] = set()
    for intent in intents:
        for mod in _INTENT_TO_MODULES.get(intent, []):
            modules_needed.add(mod)

    # 固定順序確保 prompt 結構一致
    _MODULE_ORDER = (
        "regime_v2", "analysis_v2", "factor_validation",
        "risk_checklist", "alpha_monitor",
        "output_lite", "output_full",
        "drawing", "event_analysis", "quant_research",
        "calibrate", "backtest",
    )

    parts = [_PROMPT_CORE]
    for mod_key in _MODULE_ORDER:
        if mod_key in modules_needed:
            parts.append(_PROMPT_MODULES[mod_key])

    return "\n".join(parts)


# 保留完整版供向下相容（例如 non-streaming endpoint）
SYSTEM_PROMPT = assemble_system_prompt({
    "analysis", "drawing", "backtest", "quant_research",
    "calibrate", "event_analysis",
})


# ═══════════════════════════════════════════════════════
#  Level 1 函式定義
# ═══════════════════════════════════════════════════════

FUNCTION_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "query_chart_data",
            "description": "切換或取得 K 線圖表數據。用於切換幣種、時間週期、或調整日期範圍。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "交易對，如 BTC/USDT、ETH/USDT",
                    },
                    "timeframe": {
                        "type": "string",
                        "enum": ["15m", "1h", "4h", "1d", "1w"],
                        "description": "時間週期",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "開始日期，格式 YYYY-MM-DD",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "結束日期，格式 YYYY-MM-DD",
                    },
                },
                "required": ["symbol", "timeframe"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "manage_indicator",
            "description": "新增、移除或調整圖表上的技術指標。可調整所有指標的參數。",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "remove", "update"],
                    },
                    "indicator_id": {
                        "type": "string",
                        "description": "指標 ID（共 27 個）：sma, ema, adx, vwap, ichimoku, psar, supertrend, market_structure, harmonic, rsi, bias, bb, stochrsi, atr, donchian, keltner, rel_vol, obv, vol_switch, macd, roc, trailing_stop, session, kelly, max_drawdown, fear_greed, funding",
                    },
                    "parameters": {
                        "type": "object",
                        "description": "指標參數，如 {\"period\": 21} 或 {\"std_dev\": 3}",
                    },
                },
                "required": ["action", "indicator_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_conditions",
            "description": "查詢滿足特定技術指標條件的時間點或時間段。例如 RSI < 30 的時間。",
            "parameters": {
                "type": "object",
                "properties": {
                    "conditions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "indicator": {"type": "string", "description": "指標 ID"},
                                "operator": {
                                    "type": "string",
                                    "enum": [">", "<", ">=", "<=", "==", "cross_above", "cross_below", "between"],
                                },
                                "value": {"type": "number"},
                                "value2": {"type": "number", "description": "between 的上界"},
                                "parameters": {"type": "object", "description": "指標參數覆蓋"},
                            },
                            "required": ["indicator", "operator", "value"],
                        },
                    },
                    "logical_operator": {
                        "type": "string",
                        "enum": ["AND", "OR"],
                        "description": "多條件組合邏輯",
                    },
                    "symbol": {"type": "string"},
                    "timeframe": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                },
                "required": ["conditions"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "annotate_chart",
            "description": "在圖表上添加標記。每次呼叫可添加一個或多個標記（用 annotations 陣列批量繪製）。務必設定 group_name 讓使用者能識別和管理這組標記。",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_name": {
                        "type": "string",
                        "description": "這組標記的名稱（必填），如「上升通道」「看跌旗型」「支撐壓力位」「諧波 Gartley」。使用者可據此開關顯示。",
                    },
                    "annotations": {
                        "type": "array",
                        "description": "標記陣列，一次繪製多條線。每個元素定義一個標記。",
                        "items": {
                            "type": "object",
                            "properties": {
                                "annotation_type": {
                                    "type": "string",
                                    "enum": ["highlight_range", "vertical_line", "horizontal_line", "text_label", "trend_line"],
                                },
                                "start_time": {"type": "string", "description": "起始時間 YYYY-MM-DD HH:MM:SS"},
                                "end_time": {"type": "string", "description": "結束時間"},
                                "price": {"type": "number"},
                                "end_price": {"type": "number", "description": "趨勢線終點價格"},
                                "text": {"type": "string"},
                                "color": {"type": "string"},
                                "line_width": {"type": "integer", "default": 2},
                                "line_style": {"type": "integer", "description": "0=實線 1=點線 2=虛線", "default": 0},
                            },
                            "required": ["annotation_type"],
                        },
                    },
                    "annotation_type": {
                        "type": "string",
                        "enum": ["highlight_range", "vertical_line", "horizontal_line", "text_label", "trend_line"],
                        "description": "單一標記模式（如果不用 annotations 陣列）",
                    },
                    "start_time": {"type": "string"},
                    "end_time": {"type": "string"},
                    "price": {"type": "number"},
                    "end_price": {"type": "number"},
                    "text": {"type": "string"},
                    "color": {"type": "string", "default": "#58a6ff"},
                    "line_width": {"type": "integer", "default": 2},
                    "line_style": {"type": "integer", "default": 0},
                },
                "required": ["group_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draw_pattern",
            "description": "一次繪製完整技術形態（如諧波 Gartley/Bat/Butterfly/Crab、上升三角、下降楔形、頭肩頂等）。指定形態名稱和關鍵轉折點，系統自動連線和標注。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern_name": {
                        "type": "string",
                        "description": "形態名稱，如 Gartley、Bat、Butterfly、Crab、Head and Shoulders、Ascending Triangle、Descending Wedge、Flag 等",
                    },
                    "points": {
                        "type": "array",
                        "description": "關鍵點位陣列（按連線順序排列），每個點包含 time 和 price。例如諧波: [X, A, B, C, D]",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string", "description": "點位標籤 (X, A, B, C, D 等)"},
                                "time": {"type": "string", "description": "YYYY-MM-DD HH:MM:SS"},
                                "price": {"type": "number"},
                            },
                            "required": ["label", "time", "price"],
                        },
                    },
                    "color": {"type": "string", "default": "#f0b90b", "description": "形態線條顏色"},
                    "line_width": {"type": "integer", "default": 2},
                    "bullish": {"type": "boolean", "description": "看漲=true, 看跌=false，影響顏色和標籤"},
                },
                "required": ["pattern_name", "points"],
            },
        },
    },
    # Level 2 函式
    {
        "type": "function",
        "function": {
            "name": "generate_analysis",
            "description": "根據當前圖表上的數據和指標，生成一份技術分析報告。",
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "分析重點，如 趨勢、支撐壓力、波動率",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_indicators",
            "description": "根據使用者的分析目標，推薦合適的指標組合。",
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis_goal": {
                        "type": "string",
                        "description": "分析目標，如 趨勢判斷、超買超賣、波動率分析、量能確認",
                    },
                },
                "required": ["analysis_goal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_backtest",
            "description": "用指定的技術指標進出場條件執行策略回測，返回完整績效統計（勝率、盈虧比、Sharpe、最大回撤等）。自動含滑點 0.05% + 手續費 0.1%，並自動分割樣本內/外數據檢測過擬合。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "交易對，如 BTC/USDT"},
                    "timeframe": {"type": "string", "description": "K 線週期"},
                    "start_date": {"type": "string", "description": "開始日期 YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "結束日期 YYYY-MM-DD"},
                    "direction": {
                        "type": "string",
                        "enum": ["long", "short"],
                        "description": "交易方向，預設 long",
                    },
                    "entry_conditions": {
                        "type": "array",
                        "description": "進場條件陣列",
                        "items": {
                            "type": "object",
                            "properties": {
                                "indicator": {"type": "string", "description": "指標 ID（如 rsi, macd, bb）"},
                                "operator": {
                                    "type": "string",
                                    "enum": [">", "<", ">=", "<=", "==", "cross_above", "cross_below", "between"],
                                },
                                "value": {"type": "number", "description": "比較值"},
                                "value2": {"type": "number", "description": "between 運算的上界"},
                                "parameters": {"type": "object", "description": "指標參數覆蓋"},
                            },
                            "required": ["indicator", "operator", "value"],
                        },
                    },
                    "exit_conditions": {
                        "type": "array",
                        "description": "出場條件陣列",
                        "items": {
                            "type": "object",
                            "properties": {
                                "indicator": {"type": "string"},
                                "operator": {"type": "string", "enum": [">", "<", ">=", "<=", "==", "cross_above", "cross_below", "between"]},
                                "value": {"type": "number"},
                                "value2": {"type": "number"},
                                "parameters": {"type": "object"},
                            },
                            "required": ["indicator", "operator", "value"],
                        },
                    },
                    "stop_loss_pct": {"type": "number", "description": "止損百分比（小數），如 0.02 = 2%"},
                    "take_profit_pct": {"type": "number", "description": "止盈百分比（小數），如 0.05 = 5%"},
                    "initial_capital": {"type": "number", "description": "初始資金 USDT，預設 10000"},
                    "leverage": {"type": "number", "description": "槓桿倍數（如 5 = 五倍合約），預設 1（無槓桿）。盈虧按槓桿放大，含爆倉模擬。"},
                },
                "required": ["entry_conditions", "exit_conditions"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_strategies",
            "description": "同時執行 2-5 個不同策略的回測並比較績效（勝率、Sharpe、報酬、回撤）。適合用於策略優化和參數調優。結果會按 Sharpe Ratio 排名。",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "交易對，如 BTC/USDT"},
                    "timeframe": {"type": "string", "description": "K 線週期"},
                    "start_date": {"type": "string", "description": "開始日期 YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "結束日期 YYYY-MM-DD"},
                    "strategies": {
                        "type": "array",
                        "description": "策略陣列（2-5 個），每個策略包含名稱、進出場條件",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "策略名稱，如「RSI 超賣反彈」"},
                                "direction": {"type": "string", "enum": ["long", "short"]},
                                "entry_conditions": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "indicator": {"type": "string"},
                                            "operator": {"type": "string", "enum": [">", "<", ">=", "<=", "==", "cross_above", "cross_below", "between"]},
                                            "value": {"type": "number"},
                                            "value2": {"type": "number"},
                                            "parameters": {"type": "object"},
                                        },
                                        "required": ["indicator", "operator", "value"],
                                    },
                                },
                                "exit_conditions": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "indicator": {"type": "string"},
                                            "operator": {"type": "string", "enum": [">", "<", ">=", "<=", "==", "cross_above", "cross_below", "between"]},
                                            "value": {"type": "number"},
                                            "value2": {"type": "number"},
                                            "parameters": {"type": "object"},
                                        },
                                        "required": ["indicator", "operator", "value"],
                                    },
                                },
                                "stop_loss_pct": {"type": "number"},
                                "take_profit_pct": {"type": "number"},
                            },
                            "required": ["name", "entry_conditions", "exit_conditions"],
                        },
                    },
                },
                "required": ["strategies"],
            },
        },
    },
    # ── 事件回溯統計分析 ──
    {
        "type": "function",
        "function": {
            "name": "analyze_event_patterns",
            "description": (
                "事件驅動回溯統計：找出歷史上符合特定條件的事件（如大漲超過N%、爆量等），"
                "並統計事件發生前若干根 K 線的技術指標共通性。"
                "適用場景：「大漲10%之前通常有什麼指標特徵」「暴跌前RSI通常在什麼範圍」。"
                "後端使用 NumPy 完成計算，不消耗額外 token。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "event_type": {
                        "type": "string",
                        "enum": ["price_surge", "price_drop", "volume_spike", "volatility_expansion"],
                        "description": "事件類型：price_surge=漲幅≥threshold%, price_drop=跌幅≥threshold%, volume_spike=量能≥threshold倍(vs 20MA), volatility_expansion=ATR≥threshold倍(vs 20MA)",
                    },
                    "threshold": {
                        "type": "number",
                        "description": "事件閾值。price_surge/drop=%（如10代表10%）; volume_spike/volatility_expansion=倍數（如3代表3倍）",
                        "default": 10,
                    },
                    "n_bars": {
                        "type": "integer",
                        "description": "計算漲跌幅時的 K 線數（如 n_bars=1 代表單根 K 線漲跌幅，n_bars=3 代表連續3根的累計漲跌幅）",
                        "default": 1,
                    },
                    "lookback_bars": {
                        "type": "integer",
                        "description": "事件發生前回看多少根 K 線來統計指標特徵",
                        "default": 5,
                    },
                    "indicators": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要分析的指標 ID 列表，如 ['rsi','macd','adx','bb','rel_vol','atr','obv','stoch_rsi']",
                        "default": ["rsi", "macd", "adx", "rel_vol", "bb", "atr"],
                    },
                    "symbol": {"type": "string", "description": "交易對，留空使用當前"},
                    "timeframe": {"type": "string", "description": "時間級別，留空使用當前"},
                    "start_date": {"type": "string", "description": "開始日期 YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "結束日期 YYYY-MM-DD"},
                },
                "required": ["event_type"],
            },
        },
    },
    # ── 完整量化研究 ──
    {
        "type": "function",
        "function": {
            "name": "run_quant_research",
            "description": (
                "執行完整量化研究流程：因子 IC 分析 → 因子相關性 → 策略回測（含 Sortino/Expectancy）"
                "→ Monte Carlo 模擬 → Walk Forward 驗證 → 動態倉位建議。"
                "最終輸出策略是否具備 Alpha、穩定度評分、建議槓桿和倉位。"
                "適用場景：使用者要求「完整量化研究」「策略是否可行」「策略穩定性檢測」。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "indicators": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要進行因子分析的指標 ID，如 ['rsi','macd','adx','bb','obv','supertrend']",
                    },
                    "entry_conditions": {
                        "type": "array",
                        "description": "進場條件（同 run_backtest 格式）。如不提供則只做因子分析。",
                        "items": {
                            "type": "object",
                            "properties": {
                                "indicator": {"type": "string"},
                                "operator": {"type": "string"},
                                "value": {"type": "number"},
                            },
                        },
                    },
                    "exit_conditions": {
                        "type": "array",
                        "description": "出場條件（同 run_backtest 格式）",
                        "items": {
                            "type": "object",
                            "properties": {
                                "indicator": {"type": "string"},
                                "operator": {"type": "string"},
                                "value": {"type": "number"},
                            },
                        },
                    },
                    "direction": {"type": "string", "enum": ["long", "short"], "default": "long"},
                    "stop_loss_pct": {"type": "number"},
                    "take_profit_pct": {"type": "number"},
                    "leverage": {"type": "number", "description": "槓桿倍數（如 5 = 五倍合約），預設 1"},
                    "symbol": {"type": "string"},
                    "timeframe": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                },
            },
        },
    },
    # ── 指標參數校準 ──
    {
        "type": "function",
        "function": {
            "name": "optimize_indicator_params",
            "description": (
                "掃描歷史數據，找出每個技術指標最適合此標的的閾值參數。"
                "輸出穩健區間（非單一值）+ 可信度等級（★~★★★）。"
                "結果自動儲存，之後分析時會自動參考。"
                "適用場景：使用者問「RSI 多少算超買」「幫我校準指標」「這個幣的 ADX 門檻是多少」。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "交易對，留空使用當前"},
                    "timeframe": {"type": "string", "description": "時間級別，留空使用當前"},
                    "indicators": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要校準的指標 ID（如 ['rsi','adx','macd','stochrsi']），留空校準全部可校準指標",
                    },
                    "forward_bars": {
                        "type": "integer",
                        "description": "預測未來幾根 K 線的報酬率（預設 5）",
                        "default": 5,
                    },
                    "start_date": {"type": "string", "description": "開始日期 YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "結束日期 YYYY-MM-DD"},
                },
            },
        },
    },
]
