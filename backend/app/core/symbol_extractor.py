"""阿斯拉量化系統 — 從問題文字中提取幣種

用於快取層的幣種交叉比對：
- 若問題中明確提到的幣種 ≠ chart_state 的幣種 → 跳過快取
- 避免 BTC 的答案被返回給 ADA 的問題
"""

import re
from typing import Optional

_KNOWN_BASES = {
    "BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "DOT",
    "MATIC", "BNB", "LTC", "PEPE", "SHIB", "ARB", "OP", "APT", "SUI",
    "FIL", "ATOM", "NEAR", "ICP", "UNI", "AAVE", "MKR", "SAND", "MANA",
    "TRX", "ETC", "BCH", "XLM", "ALGO", "VET", "FTM", "HBAR", "EGLD",
    "比特幣", "以太坊", "以太幣", "狗狗幣", "艾達幣", "瑞波幣", "索拉納",
}

_CHINESE_TO_SYMBOL = {
    "比特幣": "BTC", "以太坊": "ETH", "以太幣": "ETH",
    "狗狗幣": "DOGE", "艾達幣": "ADA", "瑞波幣": "XRP",
    "索拉納": "SOL",
}

_PAIR_PATTERN = re.compile(
    r'\b([A-Z]{2,10})\s*/\s*(USDT?|BUSD|USD)\b',
    re.IGNORECASE,
)

_BASE_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(b) for b in _KNOWN_BASES if b.isascii()) + r')\b',
    re.IGNORECASE,
)


def extract_symbol_from_text(text: str) -> Optional[str]:
    """從問題文字中提取幣種交易對（如 BTC/USDT）。

    Returns:
        正規化的交易對（如 "BTC/USDT"），或 None（無法識別）
    """
    if not text:
        return None

    upper = text.upper()

    m = _PAIR_PATTERN.search(upper)
    if m:
        base = m.group(1).upper()
        quote = m.group(2).upper()
        if quote == "USD":
            quote = "USDT"
        return f"{base}/{quote}"

    for cn, sym in _CHINESE_TO_SYMBOL.items():
        if cn in text:
            return f"{sym}/USDT"

    m = _BASE_PATTERN.search(upper)
    if m:
        return f"{m.group(1).upper()}/USDT"

    return None


def should_bypass_cache(question: str, chart_state_symbol: str) -> bool:
    """判斷是否應該跳過快取。

    當問題文字中明確提到的幣種 ≠ 圖表上顯示的幣種時，返回 True。
    """
    mentioned = extract_symbol_from_text(question)
    if not mentioned or not chart_state_symbol:
        return False

    norm_chart = chart_state_symbol.upper().replace(" ", "")
    norm_mentioned = mentioned.upper().replace(" ", "")

    return norm_chart != norm_mentioned
