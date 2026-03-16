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

SUPPORTED_QUOTE_CURRENCIES = {"USDT", "USD", "BUSD", "USDC", "BTC", "ETH"}


def normalize_symbol(raw: str) -> str:
    """
    標準化交易對格式
    BTCUSDT → BTC/USDT
    btcusdt → BTC/USDT
    btc/usdt → BTC/USDT
    BTC/USDT → BTC/USDT
    """
    raw = raw.strip().upper()

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


def symbol_to_filename(symbol: str, timeframe: str) -> str:
    """
    交易對轉 CSV 檔名
    BTC/USDT + 1h → BTC_USDT_1h.csv
    """
    safe = symbol.replace("/", "_")
    return f"{safe}_{timeframe}.csv"
