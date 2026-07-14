"""課程資料 — 「理論 → 圖表 → 回測驗證 → 本質」四段式教學。

布林通道完整知識體系（四課系列）：
  1. bollinger_basics   構造與均值回歸 — 是什麼、為什麼、何時失效
  2. bollinger_squeeze  Squeeze — 波動率循環與突破時機
  3. bollinger_trend    趨勢市 — walk the band 與順勢用法
  4. bollinger_advanced 進階 — 趨勢過濾、W 底 M 頭、Bollinger 法則與誤區

每一課的結構：
- theory_sections: 理論卡（漸進式），卡上可帶 chart_action（套指標 / 跑回測）
- backtest_template + tunable_params: 一鍵回測與可調參數（條件由本檔的
  builder 產生，格式與 run_backtest 完全一致）
- experiments: 引導實驗 — 帶著問題調參重跑，這是學到「本質」的核心環節
- ask_ai_prompts: 預填給 AI 導師的問題範本（配合 teaching_mode prompt 模組）

指標的 description / pro_tip 由 get_lesson() 從 registry 即時帶入，
避免與 indicators/technical.py 的教學文案重複維護。
新增課程 = 加一筆 LESSONS + 一個 builder，前端零改動。
"""

from typing import Any, Callable, Optional

from app.core.indicators import registry


# ─────────────────────────────────────────────────────────────
# 條件產生器（每課一個；輸入已 clamp 的參數，輸出 run_backtest 條件）
# ─────────────────────────────────────────────────────────────

def _bb_spec(period: int = 20, std_dev: float = 2.0) -> dict:
    return {"indicator": "bb", "parameters": {"period": period, "std_dev": std_dev}}


def _build_basics(p: dict) -> dict:
    """第 1 課：均值回歸 — 收盤下穿下軌買入，回中軌賣出。"""
    bb = _bb_spec(int(p["period"]), float(p["std_dev"]))
    return {
        "entry_conditions": [
            {"indicator": "close", "operator": "cross_below", "compare_to": {**bb, "series": "BB_Lower"}},
        ],
        "exit_conditions": [
            {"indicator": "close", "operator": "cross_above", "compare_to": {**bb, "series": "BB_Middle"}},
        ],
    }


def _build_squeeze(p: dict) -> dict:
    """第 2 課：Squeeze 突破 — 通道壓縮（Width 百分位低）時的上軌突破。"""
    bb = _bb_spec(20, 2.0)
    return {
        "entry_conditions": [
            {
                "indicator": "vol_squeeze",
                "series": "BB_Width_Pctile",
                "parameters": {"bb_period": 20, "lookback": int(p["lookback"])},
                "operator": "<",
                "value": float(p["squeeze_pctile"]),
            },
            {"indicator": "close", "operator": "cross_above", "compare_to": {**bb, "series": "BB_Upper"}},
        ],
        "exit_conditions": [
            {"indicator": "close", "operator": "cross_below", "compare_to": {**bb, "series": "BB_Middle"}},
        ],
    }


def _build_trend(p: dict) -> dict:
    """第 3 課：順勢 — 突破上軌進場，跌破中軌出場（與第 1 課完全相反）。"""
    bb = _bb_spec(int(p["period"]), float(p["std_dev"]))
    return {
        "entry_conditions": [
            {"indicator": "close", "operator": "cross_above", "compare_to": {**bb, "series": "BB_Upper"}},
        ],
        "exit_conditions": [
            {"indicator": "close", "operator": "cross_below", "compare_to": {**bb, "series": "BB_Middle"}},
        ],
    }


def _build_advanced(p: dict) -> dict:
    """第 4 課：趨勢過濾版均值回歸 — 只在價格站上長期均線時做多下軌反彈。"""
    bb = _bb_spec(20, float(p["std_dev"]))
    return {
        "entry_conditions": [
            {"indicator": "close", "operator": "cross_below", "compare_to": {**bb, "series": "BB_Lower"}},
            {
                "indicator": "close",
                "operator": ">",
                "compare_to": {"indicator": "sma", "parameters": {"period": int(p["trend_period"])}},
            },
        ],
        "exit_conditions": [
            {"indicator": "close", "operator": "cross_above", "compare_to": {**bb, "series": "BB_Middle"}},
        ],
    }


_CONDITION_BUILDERS: dict[str, Callable[[dict], dict]] = {
    "bollinger_basics": _build_basics,
    "bollinger_squeeze": _build_squeeze,
    "bollinger_trend": _build_trend,
    "bollinger_advanced": _build_advanced,
}


# ─────────────────────────────────────────────────────────────
# 課程內容
# ─────────────────────────────────────────────────────────────

LESSONS: list[dict[str, Any]] = [
    # ═══════════ 第 1 課：構造與均值回歸 ═══════════
    {
        "id": "bollinger_basics",
        "title": "布林通道 ①：構造與均值回歸",
        "subtitle": "從「±2 個標準差」的統計含義，到策略何時有效、何時失效",
        "indicator_id": "bb",
        "difficulty": "入門",
        "estimated_minutes": 15,
        "theory_sections": [
            {
                "title": "① 布林通道是什麼",
                "body": (
                    "布林通道（Bollinger Bands）由 John Bollinger 在 1980 年代提出，"
                    "由三條線組成：中軌是 N 期簡單移動平均（預設 20），上下軌是"
                    "中軌 ± k 倍標準差（預設 k=2）。標準差衡量「價格最近有多躁動」"
                    "——波動大時通道自動變寬，盤整時自動收窄。\n\n"
                    "這是它與固定百分比通道最大的差異：布林通道會呼吸。"
                    "由此衍生兩個重要讀數：BB Width（通道寬度，波動率量尺）和 "
                    "%B（價格在通道內的相對位置，0=下軌、0.5=中軌、1=上軌）。"
                ),
                "key_points": [
                    "中軌 = SMA(N)，代表近期共識價",
                    "上下軌 = 中軌 ± k×σ，σ 是近 N 期收盤價標準差",
                    "BB Width = 通道寬度 ÷ 中軌，本身就是波動率指標（第 2 課主角）",
                    "%B = (價格 − 下軌) ÷ (上軌 − 下軌)，把位置標準化成 0~1",
                ],
                "chart_action": "apply_indicator",
                "chart_action_label": "把布林通道疊到 K 線圖上看看",
            },
            {
                "title": "② 為什麼「碰下軌買入」有統計依據",
                "body": (
                    "若價格分佈接近常態，收盤價落在 ±2σ 之外的機率理論上只有約 5%。"
                    "所以價格跌破下軌，統計上是「罕見的過度延伸」，之後向均值（中軌）"
                    "回歸的機率較高——這就是均值回歸策略的核心假設。\n\n"
                    "下面的回測範本就是這個最經典的版本：收盤價下穿下軌買入，"
                    "回到中軌賣出。先跑一次，看看真實歷史數據給出的勝率和獲利因子，"
                    "再回來讀下一段。"
                ),
                "key_points": [
                    "±2σ ≈ 95% 涵蓋率，是「罕見程度」的量尺",
                    "進場邏輯：賭「過度延伸會被修正」",
                    "出場設在中軌而非上軌：均值回歸只賭「回到平均」，不賭反轉",
                ],
                "chart_action": "run_backtest",
                "chart_action_label": "馬上回測這個策略",
            },
            {
                "title": "③ 本質：它何時失效（最重要的一段）",
                "body": (
                    "價格並不是常態分佈——報酬分佈是「肥尾」的，極端走勢比常態假設"
                    "頻繁得多。強趨勢中，價格可以「沿著下軌走」（walk the band）"
                    "連跌數週，每一根都像「超跌」，每一次抄底都被套。"
                    "均值回歸策略的虧損幾乎都發生在趨勢行情。\n\n"
                    "反過來，通道收窄（Squeeze）代表波動率被壓縮，往往是大行情的前奏"
                    "——此時該用的是突破策略而不是均值回歸。\n\n"
                    "所以布林通道的本質不是「訊號機」，而是「波動率的量尺」：同一組線，"
                    "盤整市要反著做（碰軌反向），趨勢市要順著做（沿軌持有）。"
                    "先判斷市場狀態，再決定用哪套邏輯——這正是量化系統 regime filter "
                    "存在的原因。第 2~4 課會把這三條路各自展開。"
                ),
                "key_points": [
                    "常態假設在趨勢中失效：價格會 walk the band（第 3 課展開）",
                    "Squeeze（通道收窄）→ 醞釀突破，均值回歸勝率驟降（第 2 課展開）",
                    "修復方法：加趨勢過濾（第 4 課展開）",
                ],
                "chart_action": None,
                "chart_action_label": None,
            },
        ],
        "backtest_template": {"timeframe": "4h", "direction": "long", "entry_logic": "AND", "exit_logic": "AND"},
        "tunable_params": [
            {"name": "period", "label": "均線週期 N", "min": 5, "max": 200, "step": 1, "default": 20,
             "hint": "越大通道越平滑、訊號越少"},
            {"name": "std_dev", "label": "標準差倍數 k", "min": 0.5, "max": 4.0, "step": 0.1, "default": 2.0,
             "hint": "越大代表要求越極端的偏離才進場"},
        ],
        "experiments": [
            {"id": "wider_band",
             "question": "把 k 從 2 調成 3 再跑一次：交易次數和勝率各會怎麼變？為什麼會有這種取捨？",
             "override": {"std_dev": 3.0},
             "insight_hint": "更極端的偏離更罕見 → 訊號變少；但在趨勢型標的上，「更超跌」往往代表趨勢更兇，不一定更安全。"},
            {"id": "slower_ma",
             "question": "把 N 從 20 調成 50：通道變慢之後，這個策略更像在抓什麼樣的行情？",
             "override": {"period": 50},
             "insight_hint": "慢通道過濾掉短線噪音，但出場（回中軌）也需要更久，持倉時間拉長。"},
            {"id": "tighter_band",
             "question": "把 k 調成 1.0：訊號會暴增，但勝率和獲利因子撐得住嗎？觀察「訊號多」和「訊號好」的差別。",
             "override": {"std_dev": 1.0},
             "insight_hint": "±1σ 內是常態波動，根本不算過度延伸——這是「過度交易」的經典陷阱。"},
        ],
        "ask_ai_prompts": [
            "我剛在教學模式跑了布林通道均值回歸回測（碰下軌買、回中軌賣），請幫我解讀這組結果的勝率和獲利因子代表什麼，以及樣本外表現為什麼重要。",
            "布林通道策略在什麼樣的市場狀態下會連續虧損？請用目前圖表上的標的舉例說明 walk the band 現象。",
            "%B 指標是什麼？它跟直接看價格碰沒碰到軌道差在哪裡？",
        ],
    },

    # ═══════════ 第 2 課：Squeeze 與波動率循環 ═══════════
    {
        "id": "bollinger_squeeze",
        "title": "布林通道 ②：Squeeze — 波動率循環與突破",
        "subtitle": "「低波動孕育高波動」——通道收窄是大行情的前奏，但方向要用突破確認",
        "indicator_id": "vol_squeeze",
        "difficulty": "中階",
        "estimated_minutes": 15,
        "theory_sections": [
            {
                "title": "① 波動率會循環",
                "body": (
                    "John Bollinger 的名言：「低波動孕育高波動，高波動孕育低波動。」"
                    "市場不會永遠躁動，也不會永遠安靜——盤整壓縮能量，趨勢釋放能量，"
                    "然後再回到盤整。\n\n"
                    "BB Width（通道寬度 ÷ 中軌）就是這個循環的溫度計。"
                    "當 Width 收縮到近期極低水位，代表多空分歧極小、市場在「屏息」，"
                    "統計上往往離劇烈波動不遠了。這個狀態就叫 Squeeze（擠壓）。"
                ),
                "key_points": [
                    "波動率有均值回歸性：極端安靜之後常是極端躁動",
                    "BB Width 收縮 = 能量壓縮；擴張 = 能量釋放",
                    "Squeeze 的嚴謹定義用「百分位」：目前 Width 在近 N 期裡排多低",
                ],
                "chart_action": "apply_indicator",
                "chart_action_label": "把 Squeeze 指標疊上圖表",
            },
            {
                "title": "② Squeeze 的量化定義",
                "body": (
                    "「通道看起來變窄」是主觀的；量化的做法是算滾動百分位：目前 BB Width "
                    "在最近 120 根 K 線中的排名。低於 20% 算壓縮中，低於 10% 算強壓縮。\n\n"
                    "本系統的 vol_squeeze 指標輸出的就是這個百分位（BB_Width_Pctile），"
                    "而頂部工具列的「台股掃描」功能，本質就是把這個判斷一次掃過全市場"
                    "約 1900 檔，找出正在屏息的股票。\n\n"
                    "另一個常見定義（TTM Squeeze）是「布林通道整條縮進 Keltner 通道內」，"
                    "殊途同歸：都在量化「波動率被壓縮到不尋常的程度」。"
                ),
                "key_points": [
                    "百分位定義讓不同標的、不同時期可以比較（相對性，不是絕對寬度）",
                    "台股 BB 掃描 = 全市場批次跑同一個 Squeeze 判斷",
                    "TTM Squeeze（BB vs Keltner）是另一種等價定義",
                ],
                "chart_action": None,
                "chart_action_label": None,
            },
            {
                "title": "③ 本質：Squeeze 是時機篩選器，不是方向訊號",
                "body": (
                    "Squeeze 只告訴你「快了」，不告訴你「往哪」。方向必須由突破確認"
                    "——例如收盤價站上上軌。下面的範本就是這個組合：Width 百分位低於"
                    "門檻（還在壓縮）＋ 收盤突破上軌（方向出現），進場做多，跌回中軌出場。\n\n"
                    "要小心 Bollinger 說的 head fake（假突破）：擠壓末端常常先往一個方向"
                    "假動作、再反向走真行情。實務上會用「等收盤確認」「量能放大」或"
                    "「突破後回測不破」來過濾。跑跑看範本，特別注意勝率和單筆盈虧比的"
                    "組合——突破策略的典型樣貌是勝率不高、但賺的時候賺得多，"
                    "和第 1 課的均值回歸剛好是兩種相反的損益結構。"
                ),
                "key_points": [
                    "Squeeze（時機）× 突破（方向）= 完整訊號，缺一不可",
                    "head fake：擠壓末端的假突破是這個策略最大的敵人",
                    "突破策略靠「賺大賠小」而非高勝率——評估時看 PF 和盈虧比，別只看勝率",
                ],
                "chart_action": "run_backtest",
                "chart_action_label": "回測 Squeeze 突破策略",
            },
        ],
        "backtest_template": {"timeframe": "4h", "direction": "long", "entry_logic": "AND", "exit_logic": "AND"},
        "tunable_params": [
            {"name": "squeeze_pctile", "label": "壓縮門檻（Width 百分位 <）", "min": 5, "max": 50, "step": 1, "default": 25,
             "hint": "越低要求壓得越緊，訊號越少"},
            {"name": "lookback", "label": "百分位回看期", "min": 60, "max": 240, "step": 10, "default": 120,
             "hint": "拿最近幾根 K 線當比較基準"},
        ],
        "experiments": [
            {"id": "strong_squeeze",
             "question": "把門檻收緊到 10（強壓縮才出手）：訊號少了多少？留下來的訊號品質有變好嗎？",
             "override": {"squeeze_pctile": 10},
             "insight_hint": "壓得越緊、爆發統計上越猛，但樣本也越少——注意交易次數太少時，勝率數字本身就不可靠。"},
            {"id": "no_squeeze",
             "question": "把門檻放寬到 50（幾乎不篩選壓縮）：這就接近「純突破策略」了，跟預設比差在哪？",
             "override": {"squeeze_pctile": 50},
             "insight_hint": "少了 Squeeze 篩選，會多接到很多「波動已高檔時的追高突破」——對比兩者的 PF，就能量化 Squeeze 篩選的價值。"},
        ],
        "ask_ai_prompts": [
            "我剛回測了 Squeeze 突破策略，請幫我解讀結果，特別是勝率和盈虧比的組合跟均值回歸策略有什麼結構性差異。",
            "什麼是 Bollinger 說的 head fake（假突破）？實務上有哪些過濾方法？",
            "TTM Squeeze（布林通道縮進 Keltner 通道）跟 BB Width 百分位定義的 Squeeze 有什麼差異？",
        ],
    },

    # ═══════════ 第 3 課：趨勢市與 walk the band ═══════════
    {
        "id": "bollinger_trend",
        "title": "布林通道 ③：趨勢市 — walk the band 與順勢用法",
        "subtitle": "觸碰上軌不是超買——同一組線，反過來用",
        "indicator_id": "bb",
        "difficulty": "中階",
        "estimated_minutes": 12,
        "theory_sections": [
            {
                "title": "① 觸軌不是訊號",
                "body": (
                    "John Bollinger 親自強調過：「觸碰上軌本身不是賣出訊號，"
                    "觸碰下軌本身也不是買入訊號。」\n\n"
                    "強趨勢中，價格會貼著軌道走（walk the band）：一根根 K 線收在"
                    "上軌附近，通道跟著價格一路上移。這時把觸上軌當「超買」去做空，"
                    "等於逆著整條趨勢——第 1 課回測裡的大虧損，多數就是這樣來的"
                    "（只是方向相反：在下跌趨勢中不斷接刀）。"
                ),
                "key_points": [
                    "walk the band：強趨勢中價格沿軌運行，是「強勢」不是「超買」",
                    "%B 持續 > 0.8 或 < 0.2，反而是趨勢確認",
                    "把觸軌當反轉訊號，是布林通道最常見的誤用",
                ],
                "chart_action": "apply_indicator",
                "chart_action_label": "疊上布林通道，找一段 walk the band",
            },
            {
                "title": "② 順勢版策略：突破進場、中軌出場",
                "body": (
                    "把第 1 課的邏輯整個反過來：收盤突破上軌（強勢確認）進場做多，"
                    "跌破中軌（趨勢動能衰竭）出場。\n\n"
                    "這裡中軌的角色從「獲利目標」變成「移動停損」：趨勢行情中價格"
                    "多數時間在上半通道活動，跌破中軌通常代表這段趨勢至少要休息了。"
                    "先猜猜看：在你目前圖表的標的上，這個順勢版和第 1 課的逆勢版，"
                    "誰的績效好？猜完再按回測。"
                ),
                "key_points": [
                    "同一組線：逆勢版把軌道當「邊界」，順勢版把軌道當「方向確認」",
                    "中軌 = 趨勢的移動支撐，跌破即出場",
                    "典型損益結構：勝率中低、靠少數大趨勢貢獻主要獲利",
                ],
                "chart_action": "run_backtest",
                "chart_action_label": "回測順勢版（跟第 1 課對比）",
            },
            {
                "title": "③ 本質：指標沒有立場，市場狀態才有",
                "body": (
                    "跑完你會發現：同一組布林通道，順勢用法和逆勢用法在同一個標的上"
                    "的結果可能天差地遠。這不是哪個用法「比較對」，而是**市場狀態**"
                    "決定了哪套邏輯當下有效：趨勢市獎勵順勢、懲罰逆勢；盤整市相反。\n\n"
                    "所以成熟的量化系統不會直接問「布林通道說什麼」，而是先分類"
                    "市場狀態（regime：趨勢/盤整/高波動…），再決定啟用哪套規則。"
                    "本系統的 regime filter 就是在做這件事。你在這兩課親手跑出來的"
                    "對比，就是它存在的理由。"
                ),
                "key_points": [
                    "指標輸出的是「事實」（價格在哪），策略賦予它「立場」（該做什麼）",
                    "regime 分類先於策略選擇——這是從散戶到量化的關鍵一步",
                    "評估兩套用法時，注意它們賺錢的時段幾乎不重疊",
                ],
                "chart_action": None,
                "chart_action_label": None,
            },
        ],
        "backtest_template": {"timeframe": "4h", "direction": "long", "entry_logic": "AND", "exit_logic": "AND"},
        "tunable_params": [
            {"name": "period", "label": "均線週期 N", "min": 5, "max": 200, "step": 1, "default": 20,
             "hint": "慢通道抓大趨勢、快通道抓小波段"},
            {"name": "std_dev", "label": "標準差倍數 k", "min": 0.5, "max": 4.0, "step": 0.1, "default": 2.0,
             "hint": "越小越早突破進場，但假突破越多"},
        ],
        "experiments": [
            {"id": "early_entry",
             "question": "把 k 調成 1.5：進場更早，但被假突破洗出場的次數會多多少？",
             "override": {"std_dev": 1.5},
             "insight_hint": "早進場 vs 假訊號是突破策略永恆的取捨——看勝率降多少、單筆平均獲利升多少。"},
            {"id": "big_trend",
             "question": "把 N 調成 50：慢通道只抓大趨勢，交易次數和單筆持倉時間怎麼變？",
             "override": {"period": 50},
             "insight_hint": "訊號更少、持倉更久、單筆振幅更大——適合沒時間盯盤的節奏，但回撤也更深。"},
        ],
        "ask_ai_prompts": [
            "我剛對比了布林通道的順勢用法（突破上軌買）和逆勢用法（跌破下軌買）的回測結果，請幫我分析為什麼同一指標兩種用法差異這麼大。",
            "目前圖表上這個標的，最近是趨勢市還是盤整市？用什麼指標組合可以量化判斷（regime 分類）？",
            "順勢突破策略有哪些常見的出場改良？例如移動停損、分批出場，各自的取捨是什麼？",
        ],
    },

    # ═══════════ 第 4 課：進階 — 過濾、形態與法則 ═══════════
    {
        "id": "bollinger_advanced",
        "title": "布林通道 ④：進階 — 趨勢過濾、W 底 M 頭與 Bollinger 法則",
        "subtitle": "檢驗「教科書療法」能否修復第 1 課、認識經典形態、避開最常見的誤區",
        "indicator_id": "bb",
        "difficulty": "進階",
        "estimated_minutes": 18,
        "theory_sections": [
            {
                "title": "① 用趨勢過濾修復均值回歸",
                "body": (
                    "第 1 課的策略虧在「趨勢中接刀」。修復思路很直觀：**順大勢、逆小勢**"
                    "——只在長期趨勢向上時（收盤價站上 200 期均線），才去接短線的下軌"
                    "超跌反彈。多頭趨勢裡的回檔碰下軌，和空頭趨勢裡的下跌中繼碰下軌，"
                    "是完全不同的兩件事。\n\n"
                    "下面的範本就是第 1 課 + 一條 SMA 趨勢過濾。跑之前先預測：交易次數"
                    "一定會變少（過濾嘛），但勝率和獲利因子會變好嗎？好多少？\n\n"
                    "跑完把結果跟第 1 課對比。如果過濾「有改善但沒救回來」，別失望"
                    "——這正是本課最重要的一堂：教科書療法不保證適用於每個標的和週期。"
                    "在慣性極強的標的上（例如加密貨幣），跌破下軌後繼續跌可能是常態，"
                    "均值回歸整個框架都不適合，換過濾器也只是五十步笑百步。"
                    "「先驗證、再上倉位」，這就是回測存在的意義。"
                ),
                "key_points": [
                    "順大勢逆小勢：大週期定方向，小週期找進場點",
                    "過濾器的評估：犧牲多少訊號 vs 換到多少品質（期望值）",
                    "改善 ≠ 修復：如果框架不適合標的，換過濾器救不了它",
                    "這是「多指標組合」的正確姿勢：互補資訊，而不是同源指標互相背書",
                ],
                "chart_action": "run_backtest",
                "chart_action_label": "回測趨勢過濾版（跟第 1 課對比）",
            },
            {
                "title": "② 經典形態：W 底與 M 頭",
                "body": (
                    "Bollinger 本人最重視的用法其實是形態確認。以 W 底（雙重底）為例：\n\n"
                    "第一隻腳恐慌下殺，**跌破下軌**；反彈後第二隻腳再探——價格可能創新低，"
                    "但只要它**收在下軌之內**，就代表相對於波動率而言，賣壓已經衰竭"
                    "（%B 的第二個低點高於第一個），這是比「價格不破前低」更早、更靈敏的"
                    "反轉線索。M 頭剛好相反：第二個高點價格更高，但 %B 更低。\n\n"
                    "這就是 %B 的價值：它把「價格 vs 通道」的相對位置標準化，"
                    "讓兩個不同時點的低點可以公平比較——本質上是一種內建的背離偵測。"
                ),
                "key_points": [
                    "W 底：第二腳價格可以更低，但 %B 必須更高（不破下軌）",
                    "M 頭：第二頭價格更高，但 %B 更低（碰不到上軌）",
                    "%B 背離比裸價格背離更早，因為它考慮了波動率的變化",
                ],
                "chart_action": "apply_indicator",
                "chart_action_label": "疊上通道，在圖上找一個 W 底",
            },
            {
                "title": "③ Bollinger 法則精選與常見誤區",
                "body": (
                    "John Bollinger 自己整理過 22 條使用法則，最重要的幾條：\n"
                    "• 通道只提供「相對」的高低定義：貴不貴要看相對位置，不是絕對價格\n"
                    "• 觸軌本身不構成訊號（第 3 課的核心）\n"
                    "• 要搭配「非同源」指標確認：用成交量指標配 BB 可以，"
                    "用另一個「同樣拿價格算的擺盪指標」等於同一份資訊背書兩次\n"
                    "• 預設參數（20, 2）只是起點，但也別針對單一標的過度優化\n\n"
                    "最危險的三個誤區：\n"
                    "1. **把 ±2σ 當機率保證**——報酬分佈是肥尾的，「理論上 5%」的事件"
                    "實際發生率高得多，倉位若按常態假設押注會被極端行情消滅\n"
                    "2. **同源指標互相確認**——BB + RSI + KD 三個都看價格動能，"
                    "三個一起亮不代表訊號變強，只代表你把同一句話聽了三遍\n"
                    "3. **參數過擬合**——把 N 和 k 調到歷史回測最漂亮，通常只是"
                    "記住了雜訊。用樣本外數據檢驗（回測結果卡的 OOS 區塊）才算數。"
                ),
                "key_points": [
                    "相對性：BB 定義的是「相對於近期波動」的高低",
                    "確認要用非同源資訊（量、市場寬度、跨市場），不是更多價格指標",
                    "樣本外驗證是對抗過擬合的最低標準——調參後永遠看 OOS 那行",
                ],
                "chart_action": None,
                "chart_action_label": None,
            },
        ],
        "backtest_template": {"timeframe": "4h", "direction": "long", "entry_logic": "AND", "exit_logic": "AND"},
        "tunable_params": [
            {"name": "std_dev", "label": "標準差倍數 k", "min": 0.5, "max": 4.0, "step": 0.1, "default": 2.0,
             "hint": "進場要求的超跌程度"},
            {"name": "trend_period", "label": "趨勢過濾均線週期", "min": 50, "max": 200, "step": 10, "default": 200,
             "hint": "越長過濾越嚴格（200 = 經典牛熊分界）"},
        ],
        "experiments": [
            {"id": "loose_filter",
             "question": "把趨勢均線從 200 放寬到 50：過濾變鬆之後，是「多賺到訊號」還是「放進來更多接刀」？",
             "override": {"trend_period": 50},
             "insight_hint": "SMA50 之上也可能是空頭中的反彈段——對比勝率變化，量化「過濾嚴格度」的價值。"},
            {"id": "extreme_dip",
             "question": "在 200 均線過濾下，把 k 調成 3（只接極端超跌）：這是「牛市裡撿大恐慌」的配置，樣本夠嗎？",
             "override": {"std_dev": 3.0, "trend_period": 200},
             "insight_hint": "條件越多訊號越稀有。注意：交易次數 < 30 時，任何勝率數字都只是故事，不是統計。"},
        ],
        "ask_ai_prompts": [
            "我剛回測了「200均線趨勢過濾 + 布林下軌反彈」策略，請對比第 1 課無過濾版本，幫我分析過濾器帶來的期望值變化是否值得犧牲的訊號量。",
            "請在目前圖表的標的上找一個布林通道 W 底的實例，說明兩隻腳的 %B 差異怎麼看。",
            "為什麼說 BB、RSI、KD 互相確認是「同源指標」陷阱？哪些指標才算非同源的獨立確認？",
        ],
    },
]

_LESSONS_BY_ID = {lesson["id"]: lesson for lesson in LESSONS}


def get_lesson_summaries() -> list[dict[str, Any]]:
    """課程列表（輕量欄位，供列表頁）。"""
    return [
        {
            "id": lesson["id"],
            "title": lesson["title"],
            "subtitle": lesson["subtitle"],
            "indicator_id": lesson["indicator_id"],
            "difficulty": lesson["difficulty"],
            "estimated_minutes": lesson["estimated_minutes"],
        }
        for lesson in LESSONS
    ]


def get_lesson(lesson_id: str) -> Optional[dict[str, Any]]:
    """完整課程內容，並從 registry 即時帶入指標元資料（description / pro_tip）。"""
    lesson = _LESSONS_BY_ID.get(lesson_id)
    if lesson is None:
        return None
    out = dict(lesson)
    info = next(
        (i for i in registry.to_info_list() if i.get("id") == lesson["indicator_id"]),
        None,
    )
    if info:
        out["indicator_info"] = {
            "name": info.get("name"),
            "description": info.get("description"),
            "pro_tip": info.get("pro_tip"),
            "parameters": info.get("parameters"),
            "display_mode": info.get("display_mode"),
        }
    return out


def build_conditions(lesson_id: str, params: Optional[dict] = None) -> Optional[dict]:
    """把課程範本 + 使用者調整的參數組成 run_backtest 可用的條件。

    只允許 tunable_params 宣告過的參數並 clamp 到合法範圍，防止任意注入。
    """
    lesson = _LESSONS_BY_ID.get(lesson_id)
    builder = _CONDITION_BUILDERS.get(lesson_id)
    if lesson is None or builder is None:
        return None
    allowed = {p["name"]: p for p in lesson.get("tunable_params", [])}
    merged: dict[str, Any] = {p["name"]: p["default"] for p in lesson.get("tunable_params", [])}
    for key, value in (params or {}).items():
        spec = allowed.get(key)
        if spec is None:
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        v = max(float(spec["min"]), min(float(spec["max"]), v))
        merged[key] = int(v) if float(spec.get("step", 1)).is_integer() else v

    conds = builder(merged)
    template = lesson["backtest_template"]
    return {
        **conds,
        "direction": template.get("direction", "long"),
        "entry_logic": template.get("entry_logic", "AND"),
        "exit_logic": template.get("exit_logic", "AND"),
        "timeframe": template.get("timeframe", "4h"),
        "resolved_params": merged,
    }
