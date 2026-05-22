"""阿斯拉量化系統 — 加密貨幣基本面抓取器（全部免費無 key）

提供無法從 OHLCV 計算的協議基本面數據：
- 代幣經濟學 (tokenomics)：供給量 / 市值 / FDV / ATH —— 多源 fallback
- 開發 / 社群活躍度 (dev/community)：CoinGecko developer_data + community_data
- TVL：DeFiLlama（協議級 → 鏈級）

tokenomics 多源 fallback 鏈：CoinGecko → CoinPaprika → CoinCap，任一成功即回（標 source）。
所有方法失敗回 {"available": False, ...}，永不 raise、永不退回 LLM 編造。
"""

from datetime import datetime
from typing import Optional

import httpx
from loguru import logger

from app.utils.symbol import (
    symbol_to_coingecko_id,
    symbol_to_coinpaprika_id,
    symbol_to_coincap_id,
    symbol_to_defillama_protocol,
    symbol_to_defillama_chain,
    _base_of,
)


def _now_iso() -> str:
    return datetime.now().isoformat()


def _pct(a, b) -> Optional[float]:
    """a/b*100，安全處理 None / 0。"""
    try:
        if a is None or b is None or b == 0:
            return None
        return round(a / b * 100, 2)
    except Exception:
        return None


def _to_float(x) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


class CryptoFundamentalEngine:
    """加密基本面抓取器（免費無 key、多源 fallback、真實 TTL 快取）。"""

    _TTL_TOKENOMICS = 3600     # 1h
    _TTL_DEV = 21600           # 6h
    _TTL_TVL = 3600            # 1h
    _TTL_COINS_LIST = 86400    # 24h

    def __init__(self):
        self._cache: dict[str, tuple[float, dict]] = {}  # key -> (expiry_ts, data)

    def _get_cache(self, key: str) -> Optional[dict]:
        if key in self._cache:
            expiry, data = self._cache[key]
            if datetime.now().timestamp() < expiry:
                return dict(data)
        return None

    def _set_cache(self, key: str, data: dict, ttl: int):
        self._cache[key] = (datetime.now().timestamp() + ttl, data)

    # ──────────────────────────────────────────────
    # 代幣經濟學（多源 fallback：CoinGecko → CoinPaprika → CoinCap）
    # ──────────────────────────────────────────────
    async def fetch_tokenomics(self, symbol: str) -> dict:
        cache_key = f"tokenomics_{symbol}"
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached

        for fetch in (
            self._tokenomics_coingecko,
            self._tokenomics_coinpaprika,
            self._tokenomics_coincap,
        ):
            try:
                data = await fetch(symbol)
                if data and data.get("available"):
                    self._set_cache(cache_key, data, self._TTL_TOKENOMICS)
                    return data
            except Exception as e:
                logger.debug(f"tokenomics 源失敗 {fetch.__name__} {symbol}: {e}")
                continue

        return {
            "available": False, "source": None,
            "error": "all_sources_failed", "fetched_at": _now_iso(),
        }

    async def _tokenomics_coingecko(self, symbol: str) -> dict:
        cid = symbol_to_coingecko_id(symbol) or await self._resolve_via_coins_list(symbol)
        if not cid:
            return {"available": False, "source": "coingecko", "error": "no_id"}
        url = (
            f"https://api.coingecko.com/api/v3/coins/{cid}"
            "?localization=false&tickers=false&market_data=true"
            "&community_data=false&developer_data=false&sparkline=false"
        )
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            d = resp.json()
        md = d.get("market_data") or {}
        links = d.get("links") or {}
        circ = md.get("circulating_supply")
        total = md.get("total_supply")
        mx = md.get("max_supply")
        mcap = (md.get("market_cap") or {}).get("usd")
        fdv = (md.get("fully_diluted_valuation") or {}).get("usd")
        return {
            "available": True, "source": "coingecko", "fetched_at": _now_iso(),
            "coingecko_id": cid,
            "price_usd": (md.get("current_price") or {}).get("usd"),
            "market_cap_usd": mcap,
            "fdv_usd": fdv,
            "circulating_supply": circ,
            "total_supply": total,
            "max_supply": mx,
            "circ_pct": _pct(circ, mx or total),
            "mcap_to_fdv_pct": _pct(mcap, fdv),
            "ath_usd": (md.get("ath") or {}).get("usd"),
            "ath_change_pct": (md.get("ath_change_percentage") or {}).get("usd"),
            "atl_usd": (md.get("atl") or {}).get("usd"),
            "links": {
                "homepage": (links.get("homepage") or [None])[0],
                "whitepaper": links.get("whitepaper") or None,
                "github": ((links.get("repos_url") or {}).get("github") or [None])[0],
            },
        }

    async def _tokenomics_coinpaprika(self, symbol: str) -> dict:
        pid = symbol_to_coinpaprika_id(symbol)
        if not pid:
            return {"available": False, "source": "coinpaprika", "error": "no_id"}
        url = f"https://api.coinpaprika.com/v1/tickers/{pid}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            d = resp.json()
        q = (d.get("quotes") or {}).get("USD") or {}
        circ = d.get("circulating_supply")
        total = d.get("total_supply")
        mx = d.get("max_supply")
        mcap = q.get("market_cap")
        fdv = q.get("fully_diluted_market_cap")
        return {
            "available": True, "source": "coinpaprika", "fetched_at": _now_iso(),
            "price_usd": q.get("price"),
            "market_cap_usd": mcap,
            "fdv_usd": fdv,
            "circulating_supply": circ,
            "total_supply": total,
            "max_supply": mx,
            "circ_pct": _pct(circ, mx or total),
            "mcap_to_fdv_pct": _pct(mcap, fdv),
            "ath_usd": q.get("ath_price"),
            "ath_change_pct": q.get("percent_from_price_ath"),
            "atl_usd": None,
            "links": None,  # fallback 源不提供 links
        }

    async def _tokenomics_coincap(self, symbol: str) -> dict:
        ccid = symbol_to_coincap_id(symbol)
        if not ccid:
            return {"available": False, "source": "coincap", "error": "no_id"}
        url = f"https://api.coincap.io/v2/assets/{ccid}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            d = (resp.json() or {}).get("data") or {}
        circ = _to_float(d.get("supply"))
        mx = _to_float(d.get("maxSupply"))
        mcap = _to_float(d.get("marketCapUsd"))
        return {
            "available": True, "source": "coincap", "fetched_at": _now_iso(),
            "price_usd": _to_float(d.get("priceUsd")),
            "market_cap_usd": mcap,
            "fdv_usd": None,
            "circulating_supply": circ,
            "total_supply": None,
            "max_supply": mx,
            "circ_pct": _pct(circ, mx),
            "mcap_to_fdv_pct": None,
            "ath_usd": None,
            "ath_change_pct": None,
            "atl_usd": None,
            "links": None,
        }

    # ──────────────────────────────────────────────
    # 開發 / 社群活躍度（僅 CoinGecko，無對等免費備援）
    # ──────────────────────────────────────────────
    async def fetch_dev_community(self, symbol: str) -> dict:
        cache_key = f"dev_{symbol}"
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached
        cid = symbol_to_coingecko_id(symbol) or await self._resolve_via_coins_list(symbol)
        if not cid:
            return {"available": False, "source": "coingecko", "error": "no_id",
                    "fetched_at": _now_iso()}
        try:
            url = (
                f"https://api.coingecko.com/api/v3/coins/{cid}"
                "?localization=false&tickers=false&market_data=false"
                "&community_data=true&developer_data=true&sparkline=false"
            )
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                d = resp.json()
            dev = d.get("developer_data") or {}
            com = d.get("community_data") or {}
            total_issues = dev.get("total_issues")
            closed_issues = dev.get("closed_issues")
            data = {
                "available": True, "source": "coingecko", "fetched_at": _now_iso(),
                "commit_count_4_weeks": dev.get("commit_count_4_weeks"),
                "stars": dev.get("stars"),
                "forks": dev.get("forks"),
                "subscribers": dev.get("subscribers"),
                "total_issues": total_issues,
                "closed_issues": closed_issues,
                "issue_close_ratio_pct": _pct(closed_issues, total_issues),
                "pull_requests_merged": dev.get("pull_requests_merged"),
                "pr_contributors": dev.get("pull_request_contributors"),
                "twitter_followers": com.get("twitter_followers"),
                "reddit_subscribers": com.get("reddit_subscribers"),
                "reddit_active_48h": com.get("reddit_accounts_active_48h"),
                "telegram_users": com.get("telegram_channel_user_count"),
            }
            self._set_cache(cache_key, data, self._TTL_DEV)
            return data
        except Exception as e:
            logger.debug(f"dev/community 抓取失敗 {symbol}: {e}")
            return {"available": False, "source": "coingecko", "error": str(e),
                    "fetched_at": _now_iso()}

    # ──────────────────────────────────────────────
    # TVL（DeFiLlama：協議級 → 鏈級，免費無 key）
    # ──────────────────────────────────────────────
    async def fetch_tvl(self, symbol: str) -> dict:
        cache_key = f"tvl_{symbol}"
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached
        slug = symbol_to_defillama_protocol(symbol)
        chain = symbol_to_defillama_chain(symbol)
        if not slug and not chain:
            data = {"available": False, "reason": "not_a_defi_protocol",
                    "source": "defillama", "fetched_at": _now_iso()}
            self._set_cache(cache_key, data, self._TTL_TVL)
            return data
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                if slug:
                    resp = await client.get(f"https://api.llama.fi/protocol/{slug}")
                    resp.raise_for_status()
                    d = resp.json()
                    series = d.get("tvl") or []  # [{date, totalLiquidityUSD}]
                    tvl_now = series[-1].get("totalLiquidityUSD") if series else None
                    tvl_30d = series[-30].get("totalLiquidityUSD") if len(series) >= 30 else None
                    scope, name = "protocol", (d.get("name") or slug)
                else:
                    resp = await client.get(
                        f"https://api.llama.fi/v2/historicalChainTvl/{chain}")
                    resp.raise_for_status()
                    series = resp.json() or []  # [{date, tvl}]
                    tvl_now = series[-1].get("tvl") if series else None
                    tvl_30d = series[-30].get("tvl") if len(series) >= 30 else None
                    scope, name = "chain", chain
            change_30d = _pct((tvl_now - tvl_30d), tvl_30d) if (tvl_now and tvl_30d) else None
            data = {
                "available": tvl_now is not None, "source": "defillama",
                "fetched_at": _now_iso(), "scope": scope, "name": name,
                "tvl_usd": tvl_now, "tvl_change_30d_pct": change_30d,
            }
            self._set_cache(cache_key, data, self._TTL_TVL)
            return data
        except Exception as e:
            logger.debug(f"TVL 抓取失敗 {symbol}: {e}")
            return {"available": False, "source": "defillama", "error": str(e),
                    "fetched_at": _now_iso()}

    # ──────────────────────────────────────────────
    # CoinGecko id 解析 fallback（靜態 map 沒命中時）
    # ──────────────────────────────────────────────
    async def _resolve_via_coins_list(self, symbol: str) -> Optional[str]:
        base = _base_of(symbol).lower()
        cache_key = "coingecko_coins_list"
        cached = self._get_cache(cache_key)
        if cached is None:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.get("https://api.coingecko.com/api/v3/coins/list")
                    resp.raise_for_status()
                    cached = {"list": resp.json()}
                self._set_cache(cache_key, cached, self._TTL_COINS_LIST)
            except Exception as e:
                logger.debug(f"coins/list 解析失敗: {e}")
                return None
        for item in cached.get("list", []):
            if str(item.get("symbol", "")).lower() == base:
                return item.get("id")
        return None


crypto_fundamental = CryptoFundamentalEngine()
