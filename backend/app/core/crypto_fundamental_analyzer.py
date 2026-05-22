"""阿斯拉量化系統 — 加密貨幣基本面分析引擎

整合代幣經濟學（tokenomics）、開發/社群活躍度、TVL，產出結構化基本面摘要。
資料全部來自免費無 key API（CoinGecko / CoinPaprika / CoinCap / DeFiLlama）。

注意：技術路線圖（roadmap）無可靠的結構化即時 API，本引擎不抓 roadmap，
僅回傳免責說明 + 官方連結，由 LLM 以既有知識補充並明確標示非即時。
"""

import asyncio
from loguru import logger

from app.data.fetchers.crypto_fundamental import crypto_fundamental
from app.utils.symbol import normalize_symbol, symbol_to_coingecko_id


ROADMAP_DISCLAIMER = (
    "技術路線圖無可靠的結構化即時 API，以下路線圖內容為模型既有知識推估，"
    "可能已過時或不準確，請以專案官方 whitepaper / homepage 為準。"
)


def _assess_dilution(mcap_to_fdv_pct):
    if mcap_to_fdv_pct is None:
        return None
    if mcap_to_fdv_pct >= 90:
        return "流通近乎全額，無重大解鎖稀釋壓力"
    if mcap_to_fdv_pct >= 60:
        return "中度未來解鎖（FDV 高於市值，留意解鎖排程）"
    return "高度稀釋風險（大量代幣未流通，FDV 遠高於市值）"


def _assess_supply_maturity(circ_pct):
    if circ_pct is None:
        return None
    if circ_pct >= 90:
        return "供給成熟（流通量接近上限）"
    if circ_pct >= 60:
        return "供給釋出中段"
    return "早期釋出（多數供給尚未流通）"


def _assess_dev_activity(commit_4w):
    if commit_4w is None:
        return None
    if commit_4w == 0:
        return "⚠️ 近 4 週零提交，開發疑似停滯"
    if commit_4w >= 50:
        return "開發活躍"
    if commit_4w >= 10:
        return "開發中度活躍"
    return "開發低度活躍"


def _assess_tvl_trend(change_30d):
    if change_30d is None:
        return None
    if change_30d >= 10:
        return "TVL 近 30 天明顯流入"
    if change_30d <= -10:
        return "TVL 近 30 天明顯流出"
    return "TVL 近 30 天大致持平"


async def analyze_crypto_fundamentals(symbol: str) -> dict:
    """加密貨幣基本面分析（免費無 key 多源）。

    Args:
        symbol: 如 "BTC/USDT"

    Returns:
        dict: status / tokenomics / ecosystem / links / data_status / roadmap_disclaimer
    """
    symbol = normalize_symbol(symbol)
    logger.info(f"加密基本面分析 [{symbol}]: 開始抓取...")

    results = await asyncio.gather(
        crypto_fundamental.fetch_tokenomics(symbol),
        crypto_fundamental.fetch_dev_community(symbol),
        crypto_fundamental.fetch_tvl(symbol),
        return_exceptions=True,
    )

    def _coerce(r):
        return r if isinstance(r, dict) else {"available": False, "error": str(r)}

    tok = _coerce(results[0])
    dev = _coerce(results[1])
    tvl = _coerce(results[2])

    tokenomics = dict(tok)
    if tok.get("available"):
        tokenomics["dilution_assessment"] = _assess_dilution(tok.get("mcap_to_fdv_pct"))
        tokenomics["supply_maturity"] = _assess_supply_maturity(tok.get("circ_pct"))

    ecosystem = {"dev_community": dev, "tvl": tvl}
    if dev.get("available"):
        ecosystem["dev_activity_assessment"] = _assess_dev_activity(
            dev.get("commit_count_4_weeks"))
    if tvl.get("available"):
        ecosystem["tvl_trend"] = _assess_tvl_trend(tvl.get("tvl_change_30d_pct"))

    links = tok.get("links") if isinstance(tok.get("links"), dict) else None

    data_status = {
        "tokenomics": f"live:{tok.get('source')}" if tok.get("available") else "unavailable",
        "ecosystem_dev": "live:coingecko" if dev.get("available") else "unavailable",
        "tvl": ("live:defillama" if tvl.get("available")
                else tvl.get("reason", "unavailable")),
        "roadmap": "not_live_llm_knowledge_only",
    }

    any_live = bool(tok.get("available") or dev.get("available") or tvl.get("available"))
    out = {
        "status": "success" if any_live else "partial",
        "symbol": symbol,
        "coingecko_id": tok.get("coingecko_id") or symbol_to_coingecko_id(symbol),
        "tokenomics": tokenomics,
        "ecosystem": ecosystem,
        "links": links,
        "data_status": data_status,
        "roadmap_disclaimer": ROADMAP_DISCLAIMER,
    }
    if not any_live:
        out["hint"] = (
            "即時基本面資料暫不可得（API 失敗或限流），"
            "請勿編造數字，明確告知使用者資料暫缺。"
        )
    logger.info(f"加密基本面分析 [{symbol}]: 完成 status={out['status']} "
                f"tok={data_status['tokenomics']} tvl={data_status['tvl']}")
    return out
