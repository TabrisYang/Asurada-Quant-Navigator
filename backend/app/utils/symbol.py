"""阿斯拉量化系統 — 交易對標準化工具"""

import re


# Coinbase USDT → USD 映射（完整 20 組）
COINBASE_USD_MAP = {
    "BTC/USDT": "BTC/USD",
    "ETH/USDT": "ETH/USD",
    "SOL/USDT": "SOL/USD",
    "XRP/USDT": "XRP/USD",
    "DOGE/USDT": "DOGE/USD",
    "ADA/USDT": "ADA/USD",
    "AVAX/USDT": "AVAX/USD",
    "LINK/USDT": "LINK/USD",
    "DOT/USDT": "DOT/USD",
    "MATIC/USDT": "MATIC/USD",
    "LTC/USDT": "LTC/USD",
    "UNI/USDT": "UNI/USD",
    "ATOM/USDT": "ATOM/USD",
    "ETC/USDT": "ETC/USD",
    "XLM/USDT": "XLM/USD",
    "BCH/USDT": "BCH/USD",
    "FIL/USDT": "FIL/USD",
    "APT/USDT": "APT/USD",
    "ARB/USDT": "ARB/USD",
    "OP/USDT": "OP/USD",
}

SUPPORTED_QUOTE_CURRENCIES = {"USDT", "USD", "BUSD", "USDC", "BTC", "ETH", "TWD"}

# 台股指數代碼映射（內部格式 → yfinance ticker）
TW_INDEX_MAP: dict[str, str] = {
    "TWII/TWD": "^TWII",       # 加權指數
    "TWOII/TWD": "^TWOII",     # 櫃買指數
}

# 中文別名 → 內部格式
_TW_INDEX_ALIASES: dict[str, str] = {
    "加權指數": "TWII/TWD",
    "櫃買指數": "TWOII/TWD",
    "大盤": "TWII/TWD",
    "台股大盤": "TWII/TWD",
}


def normalize_symbol(raw: str) -> str:
    """
    標準化交易對格式
    BTCUSDT → BTC/USDT
    btcusdt → BTC/USDT
    btc/usdt → BTC/USDT
    BTC/USDT → BTC/USDT
    2330.TW → 2330/TWD
    2330.TWO → 2330/TWD
    ^TWII → TWII/TWD
    加權指數 → TWII/TWD
    """
    raw = raw.strip()

    # 中文別名（大小寫敏感，先處理再轉大寫）
    if raw in _TW_INDEX_ALIASES:
        return _TW_INDEX_ALIASES[raw]

    raw = raw.upper()

    # 指數格式：^TWII → TWII/TWD
    if raw.startswith("^TW"):
        return f"{raw[1:]}/TWD"

    # 台股格式：2330.TW / 2330.TWO → 2330/TWD
    tw_match = re.match(r"^(\d{4,6})\.(TW|TWO)$", raw)
    if tw_match:
        return f"{tw_match.group(1)}/TWD"

    # 已經是標準格式
    if "/" in raw:
        parts = raw.split("/")
        if len(parts) == 2 and parts[1] in SUPPORTED_QUOTE_CURRENCIES:
            return raw
        return raw

    # 嘗試分離 base 和 quote
    for quote in sorted(SUPPORTED_QUOTE_CURRENCIES, key=len, reverse=True):
        if raw.endswith(quote):
            base = raw[: -len(quote)]
            if base:
                return f"{base}/{quote}"

    return raw


def get_coinbase_symbol(symbol: str) -> str:
    """取得 Coinbase 對應的交易對（USDT → USD）"""
    return COINBASE_USD_MAP.get(symbol, symbol)


def is_tw_stock(symbol: str) -> bool:
    """判斷是否為台股代碼（/TWD 結尾）"""
    return normalize_symbol(symbol).endswith("/TWD")


# 常見上櫃股票代碼前綴（6 開頭多為上櫃）
# 簡化判斷：4 碼且 6 開頭視為上櫃，其餘為上市
def symbol_to_yf_ticker(symbol: str) -> str:
    """將內部格式轉為 yfinance ticker

    2330/TWD → 2330.TW（上市）
    6547/TWD → 6547.TWO（上櫃）
    TWII/TWD → ^TWII（加權指數）
    """
    symbol = normalize_symbol(symbol)
    # 指數優先查映射表
    if symbol in TW_INDEX_MAP:
        return TW_INDEX_MAP[symbol]
    code = symbol.split("/")[0]
    # 4 碼且以 6 開頭 → 上櫃 (.TWO)，否則 → 上市 (.TW)
    suffix = ".TWO" if len(code) == 4 and code.startswith("6") else ".TW"
    return f"{code}{suffix}"


def symbol_to_filename(symbol: str, timeframe: str) -> str:
    """
    交易對轉 CSV 檔名
    BTC/USDT + 1h → BTC_USDT_1h.csv
    """
    safe = symbol.replace("/", "_")
    return f"{safe}_{timeframe}.csv"
