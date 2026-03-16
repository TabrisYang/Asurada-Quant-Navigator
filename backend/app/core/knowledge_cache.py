"""阿斯拉量化系統 — 知識問答快取

常見技術分析知識問題的本地快取，完全不需呼叫 LLM。
- 使用關鍵字匹配判斷是否命中
- 命中率高的常見知識類問題直接回傳
- 不處理涉及具體數據的問題（那些需要 LLM）
"""

import re
from typing import Optional

from loguru import logger


class KnowledgeCache:
    """技術分析知識快取"""

    def __init__(self):
        self._entries = _build_knowledge_base()
        logger.info(f"知識快取已載入 {len(self._entries)} 筆常見問答")

    def try_answer(self, question: str) -> Optional[str]:
        """
        嘗試用知識快取回答問題。

        Returns:
            匹配到的答案（帶 ⚡ 標記為快取回答），
            如果不匹配返回 None（交給 LLM 處理）。
        """
        q = question.strip().lower()

        # 太短或太長的問題不處理（可能是複合問題）
        if len(q) < 4 or len(q) > 100:
            return None

        # 如果包含具體數據查詢的關鍵字，交給 LLM
        data_keywords = [
            "現在", "目前", "最新", "今天", "昨天", "幾月", "幾號",
            "多少", "數值", "幾次", "計算", "顯示", "畫出", "標記",
            "比較", "回測", "分析", "報告",
        ]
        if any(kw in q for kw in data_keywords):
            return None

        # 嘗試匹配知識庫
        for entry in self._entries:
            if entry["match"](q):
                return f"{entry['answer']}\n\n_⚡ 來自知識快取（未消耗 token）_"

        return None


def _build_knowledge_base() -> list[dict]:
    """建立知識庫（關鍵字 + 答案）"""
    entries = []

    def _add(keywords: list[str], answer: str):
        """加入一條知識，keywords 中任意一個匹配即命中"""
        pattern = "|".join(re.escape(kw) for kw in keywords)
        compiled = re.compile(pattern)
        entries.append({
            "keywords": keywords,
            "match": lambda q, p=compiled: bool(p.search(q)),
            "answer": answer,
        })

    # ─── 基礎指標知識 ─────────────────────────

    _add(
        ["什麼是rsi", "rsi是什麼", "rsi 是什麼", "rsi定義", "rsi 定義", "何謂rsi"],
        "**RSI（Relative Strength Index，相對強弱指標）**\n\n"
        "由 J. Welles Wilder 於 1978 年提出的動量指標，衡量價格變動的速度和幅度。\n\n"
        "**計算方式**：RSI = 100 - (100 / (1 + RS))，其中 RS = 平均漲幅 / 平均跌幅\n\n"
        "**解讀**：\n"
        "- RSI > 70：超買區域，可能面臨回調壓力\n"
        "- RSI < 30：超賣區域，可能出現反彈機會\n"
        "- RSI = 50：多空分界線\n\n"
        "**常見參數**：週期 14（預設），短線可用 7，長線可用 21"
    )

    _add(
        ["什麼是macd", "macd是什麼", "macd 是什麼", "macd定義", "何謂macd"],
        "**MACD（Moving Average Convergence Divergence，指數平滑異同移動平均線）**\n\n"
        "由 Gerald Appel 於 1979 年提出，追蹤趨勢的動量指標。\n\n"
        "**計算方式**：\n"
        "- MACD 線 = 12 日 EMA - 26 日 EMA\n"
        "- 信號線 = MACD 線的 9 日 EMA\n"
        "- 柱狀圖 = MACD 線 - 信號線\n\n"
        "**交易信號**：\n"
        "- 黃金交叉（MACD 上穿信號線）：看多信號\n"
        "- 死亡交叉（MACD 下穿信號線）：看空信號\n"
        "- 零軸上方：多頭市場；零軸下方：空頭市場"
    )

    _add(
        ["什麼是布林", "布林帶是什麼", "布林通道", "bollinger", "什麼是bb", "bb是什麼"],
        "**布林帶（Bollinger Bands）**\n\n"
        "由 John Bollinger 於 1980 年代提出的波動率指標。\n\n"
        "**組成**：\n"
        "- 中軌：20 日簡單移動平均線（SMA）\n"
        "- 上軌：中軌 + 2 倍標準差\n"
        "- 下軌：中軌 - 2 倍標準差\n\n"
        "**解讀**：\n"
        "- 價格觸及上軌：可能超買\n"
        "- 價格觸及下軌：可能超賣\n"
        "- 帶寬收窄（擠壓）：即將出現大幅波動\n"
        "- 帶寬擴張：波動正在增大"
    )

    _add(
        ["什麼是ema", "ema是什麼", "ema 是什麼", "指數移動平均"],
        "**EMA（Exponential Moving Average，指數移動平均線）**\n\n"
        "一種加權移動平均線，對近期數據給予更高權重，比 SMA 更敏感。\n\n"
        "**常見週期**：\n"
        "- EMA 9 / 12：短線趨勢\n"
        "- EMA 26 / 50：中線趨勢\n"
        "- EMA 200：長線趨勢（機構常用）\n\n"
        "**使用方式**：價格在 EMA 上方為多頭，下方為空頭。多條 EMA 交叉可作為買賣信號。"
    )

    _add(
        ["什麼是sma", "sma是什麼", "sma 是什麼", "簡單移動平均"],
        "**SMA（Simple Moving Average，簡單移動平均線）**\n\n"
        "最基礎的移動平均線，計算 N 期收盤價的算術平均值。\n\n"
        "**常見週期**：SMA 20（短線）、SMA 50（中線）、SMA 200（長線）\n\n"
        "**特點**：\n"
        "- 比 EMA 反應慢，但更穩定\n"
        "- 適合判斷長期趨勢方向\n"
        "- SMA 200 是機構投資者常用的多空分界線"
    )

    _add(
        ["什麼是atr", "atr是什麼", "atr 是什麼", "真實波幅", "真實波動"],
        "**ATR（Average True Range，平均真實波幅）**\n\n"
        "衡量市場波動性的指標，不指示方向，只衡量波動幅度。\n\n"
        "**計算方式**：取以下三者的最大值再做平均：\n"
        "1. 當日最高 - 當日最低\n"
        "2. |當日最高 - 昨日收盤|\n"
        "3. |當日最低 - 昨日收盤|\n\n"
        "**用途**：\n"
        "- 設定止損位（如 2x ATR）\n"
        "- 評估市場波動狀態\n"
        "- ATR 擴大 = 波動加劇；ATR 收縮 = 波動減小"
    )

    _add(
        ["什麼是obv", "obv是什麼", "obv 是什麼", "能量潮"],
        "**OBV（On-Balance Volume，能量潮）**\n\n"
        "由 Joseph Granville 提出的量能指標，結合價格和成交量判斷趨勢。\n\n"
        "**計算方式**：\n"
        "- 收盤價上漲：OBV = 前日 OBV + 當日成交量\n"
        "- 收盤價下跌：OBV = 前日 OBV - 當日成交量\n\n"
        "**解讀**：\n"
        "- OBV 上升 + 價格上升：上漲趨勢確認\n"
        "- OBV 下降 + 價格上升：量價背離，警戒信號"
    )

    _add(
        ["什麼是adx", "adx是什麼", "adx 是什麼", "平均趨向"],
        "**ADX（Average Directional Index，平均趨向指數）**\n\n"
        "衡量趨勢強度的指標，數值越高代表趨勢越強。\n\n"
        "**解讀**：\n"
        "- ADX < 20：無趨勢（盤整）\n"
        "- ADX 20-40：趨勢形成中\n"
        "- ADX > 40：強趨勢\n"
        "- ADX > 60：極端強趨勢\n\n"
        "**注意**：ADX 只衡量趨勢強度，不指示方向。需搭配 +DI/-DI 判斷多空。"
    )

    _add(
        ["什麼是vwap", "vwap是什麼", "vwap 是什麼", "成交量加權"],
        "**VWAP（Volume Weighted Average Price，成交量加權均價）**\n\n"
        "考慮成交量的平均價格，常被機構投資者用作基準價格。\n\n"
        "**解讀**：\n"
        "- 價格 > VWAP：偏多\n"
        "- 價格 < VWAP：偏空\n"
        "- 每日重置，適合日內交易\n\n"
        "**用途**：判斷當日多空力量分佈，機構常用來評估執行品質。"
    )

    # ─── 通用問題 ─────────────────────────────

    _add(
        ["什麼是k線", "k線是什麼", "k線圖", "蠟燭圖", "candlestick"],
        "**K 線圖（Candlestick Chart，蠟燭圖）**\n\n"
        "源自日本的價格圖表，每根 K 線包含四個價格：\n"
        "- **開盤價（Open）**：該時段開始的價格\n"
        "- **最高價（High）**：該時段的最高價格\n"
        "- **最低價（Low）**：該時段的最低價格\n"
        "- **收盤價（Close）**：該時段結束的價格\n\n"
        "**顏色**：\n"
        "- 綠色/紅色：收盤 > 開盤（上漲）\n"
        "- 紅色/綠色：收盤 < 開盤（下跌）\n"
        "（東西方慣例顏色相反）"
    )

    _add(
        ["什麼是恐懼貪婪", "恐懼貪婪指數", "fear greed", "fear & greed"],
        "**恐懼與貪婪指數（Fear & Greed Index）**\n\n"
        "衡量加密貨幣市場情緒的綜合指標，範圍 0-100。\n\n"
        "**分級**：\n"
        "- 0-25：極度恐懼（市場恐慌，可能是買入機會）\n"
        "- 25-45：恐懼\n"
        "- 45-55：中性\n"
        "- 55-75：貪婪\n"
        "- 75-100：極度貪婪（市場過熱，需警惕回調）\n\n"
        "**數據來源**：綜合波動率、交易量、社群情緒、市佔率等因素計算。"
    )

    _add(
        ["什麼是資金費率", "funding rate", "資金費率是什麼"],
        "**資金費率（Funding Rate）**\n\n"
        "永續合約中多空雙方定期交換的費用，用於使合約價格錨定現貨價格。\n\n"
        "**解讀**：\n"
        "- 正費率：多頭付給空頭（市場偏多）\n"
        "- 負費率：空頭付給多頭（市場偏空）\n"
        "- 極端正費率：市場過度樂觀，可能回調\n"
        "- 極端負費率：市場過度悲觀，可能反彈\n\n"
        "**結算頻率**：通常每 8 小時一次（交易所規定）。"
    )

    _add(
        ["什麼是凱利公式", "kelly", "凱利", "kelly criterion"],
        "**凱利公式（Kelly Criterion）**\n\n"
        "一種資金管理公式，計算最佳下注比例以最大化長期收益。\n\n"
        "**公式**：f* = (bp - q) / b\n"
        "- f*：最佳資金比例\n"
        "- b：賠率\n"
        "- p：勝率\n"
        "- q：敗率（1-p）\n\n"
        "**實務建議**：通常使用半凱利（f*/2）以降低風險。"
    )

    _add(
        ["什麼是最大回撤", "max drawdown", "最大回撤", "mdd"],
        "**最大回撤（Max Drawdown，MDD）**\n\n"
        "從歷史最高點到最低點的最大跌幅，衡量投資的最大風險。\n\n"
        "**計算**：MDD = (最高點 - 之後的最低點) / 最高點 × 100%\n\n"
        "**意義**：\n"
        "- MDD 20% = 你的帳戶最多曾經從頂部下跌 20%\n"
        "- 一般投資人心理承受極限約在 20-30%\n"
        "- 優秀的量化策略通常控制在 15% 以內"
    )

    return entries


# 全域單例
knowledge_cache = KnowledgeCache()
