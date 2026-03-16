"""阿斯拉量化系統 — 預置種子知識碎片

系統啟動時自動載入，提供基礎分析經驗，
讓新使用者第一次使用就有 RAG 參考可用。
種子碎片不會被自動淘汰（is_seed=1）。
"""

from loguru import logger

SEED_FRAGMENTS = [
    {
        "content": (
            "BTC/USDT 週線 RSI 低於 30 在歷史上極為罕見"
            "（2018底、2020.3、2022底），通常對應大級別底部區域，"
            "回報率極高但需耐心持有"
        ),
        "type": "indicator",
        "symbol": "BTC/USDT",
    },
    {
        "content": (
            "BTC/USDT 四小時線在強勢上漲趨勢中，RSI 80 以上"
            "並不代表立即反轉，常出現高位鈍化持續上漲的現象，"
            "應搭配量能和支撐位綜合判斷"
        ),
        "type": "indicator",
        "symbol": "BTC/USDT",
    },
    {
        "content": (
            "BTC/USDT 日線布林帶收窄（帶寬低於歷史 20 分位）"
            "通常預示大幅波動即將到來，但不指示方向，"
            "需配合趨勢指標確認"
        ),
        "type": "pattern",
        "symbol": "BTC/USDT",
    },
    {
        "content": (
            "BTC/USDT 日線 MACD 零軸上方金叉的勝率"
            "顯著高於零軸下方金叉，配合成交量放大確認效果更佳"
        ),
        "type": "indicator",
        "symbol": "BTC/USDT",
    },
    {
        "content": (
            "BTC/USDT 資金費率持續高於 0.05% 時通常代表"
            "市場過度槓桿做多，有回調清算風險；"
            "持續為負則可能是底部機會"
        ),
        "type": "sentiment",
        "symbol": "BTC/USDT",
    },
    {
        "content": (
            "ETH/USDT 通常跟隨 BTC 走勢但波動更大（beta > 1），"
            "在 BTC 橫盤期 ETH 常出現獨立行情，"
            "關注 ETH/BTC 匯率變化"
        ),
        "type": "trend",
        "symbol": "ETH/USDT",
    },
    {
        "content": (
            "ETH/USDT 日線 SuperTrend 翻多（紅轉綠）配合 OBV 上升"
            "是較可靠的趨勢反轉信號，歷史勝率約 65%"
        ),
        "type": "indicator",
        "symbol": "ETH/USDT",
    },
    {
        "content": (
            "多時間週期分析（MTF）建議從大到小：先看週線確認大趨勢方向，"
            "再用日線找結構位（支撐/壓力），最後用 4H 或 1H 找精確入場點"
        ),
        "type": "strategy",
        "symbol": "",
    },
    {
        "content": (
            "Market Structure 判斷趨勢：連續 HH+HL 為上升趨勢，"
            "連續 LH+LL 為下降趨勢；出現 CHoCH（趨勢轉變）時"
            "是潛在的趨勢反轉信號"
        ),
        "type": "strategy",
        "symbol": "",
    },
    {
        "content": (
            "凱利公式建議的倉位通常偏高，實務上建議使用半凱利（Kelly/2）"
            "或四分之一凱利（Kelly/4）來控制風險，避免單筆虧損過大"
        ),
        "type": "strategy",
        "symbol": "",
    },
    {
        "content": (
            "恐懼貪婪指數低於 20 時屬於極度恐懼，歷史上在此區間"
            "逆勢分批建倉的長期回報率顯著優於追高；"
            "指數超過 80 則應考慮減倉"
        ),
        "type": "sentiment",
        "symbol": "",
    },
    {
        "content": (
            "追蹤止損（Trailing Stop）使用 ATR 倍數設定時，"
            "日線建議 2-3 倍 ATR，4H 建議 1.5-2 倍 ATR，"
            "可有效保護利潤又不會被正常波動觸發"
        ),
        "type": "strategy",
        "symbol": "",
    },
    {
        "content": (
            "成交量分析中，上漲放量+下跌縮量是健康的上升趨勢特徵；"
            "若上漲縮量+下跌放量則預示趨勢可能反轉，"
            "OBV 背離是重要的領先指標"
        ),
        "type": "volume",
        "symbol": "",
    },
    {
        "content": (
            "一目均衡表（Ichimoku）雲層在日線以上週期參考價值較高，"
            "價格在雲層上方且前置線高於基準線為多頭排列，"
            "三役好轉是強勢做多信號"
        ),
        "type": "indicator",
        "symbol": "",
    },
]


def initialize_seed_knowledge():
    """系統啟動時載入種子知識（僅在碎片庫為空時）"""
    try:
        from app.core.knowledge_fragments import fragment_store
        import app.core.embedding_service as _emb

        stats = fragment_store.get_stats()
        if stats["seed_count"] > 0:
            logger.debug(f"種子知識已存在（{stats['seed_count']} 筆），跳過初始化")
            return

        if not _emb.is_available():
            logger.info("嵌入模型尚未就緒，延遲種子知識初始化")
            return

        loaded = 0
        for frag in SEED_FRAGMENTS:
            ok = fragment_store.store_fragment(
                content=frag["content"],
                fragment_type=frag["type"],
                symbol=frag.get("symbol", ""),
                source_question="system_seed",
                is_seed=True,
            )
            if ok:
                loaded += 1

        logger.info(f"種子知識初始化完成：載入 {loaded}/{len(SEED_FRAGMENTS)} 筆")
    except Exception as e:
        logger.warning(f"種子知識初始化失敗: {e}")
