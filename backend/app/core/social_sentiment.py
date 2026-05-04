"""阿斯拉量化系統 — v106 A3：免費社群情緒聚合（Reddit + Alternative.me + RSS）。

不付費、不需 API key 的社群情緒指標：
1. Reddit r/CryptoCurrency / r/Bitcoin / r/<symbol> 熱門貼文（免費 JSON）
2. Alternative.me Fear & Greed Index（免費，已在 external_signals 用）
3. CryptoPanic 免費 RSS（聚合多個新聞源）

策略：
- 多源 cross-check（避免單源誤導）
- 30 分鐘 cache（節省 rate limit）
- graceful fallback（任一源失敗其他照常）
- 不做精細 NLP（用關鍵字計分 + 直接給 LLM 摘要）

注入到 chart_state.social_sentiment：
{
  "fear_greed": {value, label},  # 來自 Alternative.me
  "reddit_buzz": {top_posts: [...], net_sentiment: -1~+1},
  "news_recent": [{title, url, source, time_ago}],
  "stale_warning": "..."
}
"""

from __future__ import annotations

import re
import time
from typing import Optional

import httpx
from loguru import logger

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}
_HTTP_TIMEOUT = 5.0
_CACHE_TTL = 7200  # v107.3：2 小時（情緒變化沒這麼快、避免常常 stale）
_cache: dict[str, tuple[float, dict]] = {}


def _cached(key: str) -> Optional[dict]:
    if key in _cache:
        ts, payload = _cache[key]
        if time.time() - ts < _CACHE_TTL:
            return payload
    return None


def _store(key: str, payload: dict) -> None:
    _cache[key] = (time.time(), payload)


# ─── 簡易情緒關鍵字計分 ──────────────────────────────


_BULLISH_KEYWORDS = [
    "bullish", "moon", "pump", "buy", "long", "rally", "breakout",
    "ATH", "all-time high", "surge", "rocket", "🚀", "📈", "to the moon",
    "breakout", "support holds", "accumulate",
]
_BEARISH_KEYWORDS = [
    "bearish", "dump", "crash", "sell", "short", "drop", "breakdown",
    "support broken", "panic", "capitulation", "📉", "🔻",
    "rejection", "rejected", "rug", "exit liquidity",
]


def _score_text_sentiment(text: str) -> float:
    """簡單關鍵字計分：+1 強看多 / -1 強看空 / 0 中性。"""
    if not text:
        return 0.0
    t = text.lower()
    bull = sum(1 for kw in _BULLISH_KEYWORDS if kw in t)
    bear = sum(1 for kw in _BEARISH_KEYWORDS if kw in t)
    if bull == 0 and bear == 0:
        return 0.0
    return (bull - bear) / max(bull + bear, 1)


# ─── Reddit fetcher（免費 JSON，無需 key）──────────────────────────────


def _symbol_to_subreddit(symbol: str) -> Optional[str]:
    """BTC/USDT → bitcoin / ETH/USDT → ethereum / 其他 → 通用 cryptocurrency。"""
    if "/" not in symbol:
        return "CryptoCurrency"
    base = symbol.split("/")[0].upper()
    mapping = {
        "BTC": "Bitcoin",
        "ETH": "ethereum",
        "ADA": "cardano",
        "SOL": "solana",
        "DOGE": "dogecoin",
        "DOT": "dot",
        "LINK": "Chainlink",
        "AVAX": "Avax",
        "MATIC": "0xPolygon",
    }
    return mapping.get(base, "CryptoCurrency")


def _fetch_reddit_posts(subreddit: str, limit: int = 10) -> list[dict]:
    """抓 subreddit hot posts。免費 JSON endpoint，無需 OAuth。"""
    url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
    try:
        with httpx.Client(headers=_HEADERS, timeout=_HTTP_TIMEOUT, follow_redirects=True) as c:
            r = c.get(url)
            r.raise_for_status()
        data = r.json()
        out = []
        for child in data.get("data", {}).get("children", [])[:limit]:
            p = child.get("data", {})
            title = p.get("title", "")
            score = p.get("score", 0)
            num_comments = p.get("num_comments", 0)
            created_utc = p.get("created_utc", 0)
            if not title or score < 5:  # 篩低熱度
                continue
            sentiment = _score_text_sentiment(title)
            out.append({
                "title": title[:120],
                "score": score,
                "comments": num_comments,
                "sentiment": round(sentiment, 2),
                "hours_ago": round((time.time() - created_utc) / 3600, 1) if created_utc else None,
            })
        return out
    except Exception as e:
        logger.debug(f"[social] reddit r/{subreddit} 失敗: {e}")
        return []


# ─── CryptoPanic free RSS ──────────────────────────────


def _fetch_cryptopanic_rss(symbol_filter: Optional[str] = None) -> list[dict]:
    """CryptoPanic 免費 RSS — 聚合多個 crypto 新聞源。

    URL: https://cryptopanic.com/news/rss/
    可加 ?currencies=BTC 篩特定幣種。
    """
    base_url = "https://cryptopanic.com/news/rss/"
    if symbol_filter:
        base = symbol_filter.split("/")[0].upper() if "/" in symbol_filter else symbol_filter.upper()
        url = f"{base_url}?currencies={base}"
    else:
        url = base_url

    try:
        with httpx.Client(headers=_HEADERS, timeout=_HTTP_TIMEOUT, follow_redirects=True) as c:
            r = c.get(url)
            r.raise_for_status()

        # 簡易 RSS 解析（不引入 feedparser 套件）
        text = r.text
        items = re.findall(r"<item>(.*?)</item>", text, re.DOTALL)
        out = []
        for item in items[:8]:
            title_m = re.search(r"<title><!\[CDATA\[(.*?)\]\]></title>", item, re.DOTALL)
            link_m = re.search(r"<link>(.*?)</link>", item, re.DOTALL)
            pubdate_m = re.search(r"<pubDate>(.*?)</pubDate>", item, re.DOTALL)
            if not title_m:
                title_m = re.search(r"<title>(.*?)</title>", item, re.DOTALL)
            if not title_m:
                continue
            title = title_m.group(1).strip()
            sentiment = _score_text_sentiment(title)
            out.append({
                "title": title[:140],
                "url": (link_m.group(1).strip() if link_m else ""),
                "published": (pubdate_m.group(1).strip() if pubdate_m else ""),
                "sentiment": round(sentiment, 2),
            })
        return out
    except Exception as e:
        logger.debug(f"[social] cryptopanic RSS 失敗: {e}")
        return []


# ─── 主聚合函式 ──────────────────────────────


def get_social_sentiment(symbol: str) -> dict:
    """整合社群情緒給 chart_state 用。30 分 cache 共享。"""
    cache_key = f"sentiment|{symbol}"
    cached = _cached(cache_key)
    if cached:
        return {**cached, "cached": True}

    out: dict = {"symbol": symbol, "sources": {}, "cached": False}

    # 1. Reddit
    sub = _symbol_to_subreddit(symbol)
    if sub:
        posts = _fetch_reddit_posts(sub, limit=10)
        if posts:
            sentiments = [p["sentiment"] for p in posts if p["sentiment"] != 0]
            net = sum(sentiments) / len(sentiments) if sentiments else 0.0
            out["reddit_buzz"] = {
                "subreddit": sub,
                "n_posts": len(posts),
                "net_sentiment": round(net, 2),
                "top_3_titles": [p["title"] for p in posts[:3]],
            }
            out["sources"]["reddit"] = "ok"
        else:
            out["sources"]["reddit"] = "empty"

    # 2. CryptoPanic
    news = _fetch_cryptopanic_rss(symbol_filter=symbol)
    if news:
        net_news = sum(n["sentiment"] for n in news) / len(news) if news else 0.0
        out["news_recent"] = news[:5]
        out["news_net_sentiment"] = round(net_news, 2)
        out["sources"]["cryptopanic"] = "ok"
    else:
        out["sources"]["cryptopanic"] = "empty"

    # 3. 整體 score（reddit + news 平均，若都有）
    component_scores = []
    if "reddit_buzz" in out:
        component_scores.append(out["reddit_buzz"]["net_sentiment"])
    if "news_net_sentiment" in out:
        component_scores.append(out["news_net_sentiment"])
    if component_scores:
        overall = sum(component_scores) / len(component_scores)
        out["overall_sentiment"] = round(overall, 2)
        out["overall_label"] = (
            "強看多" if overall > 0.4
            else "偏多" if overall > 0.15
            else "中性" if abs(overall) <= 0.15
            else "偏空" if overall > -0.4
            else "強看空"
        )

    # 失敗警示
    n_ok_sources = sum(1 for v in out["sources"].values() if v == "ok")
    if n_ok_sources == 0:
        out["stale_warning"] = "所有社群情緒來源都失敗，本次無社群訊號可用"

    _store(cache_key, out)
    return out


def format_sentiment_summary(sentiment: dict) -> str:
    """LLM prompt 友善字串。"""
    if not sentiment or sentiment.get("stale_warning"):
        return ""
    lines = []
    if "overall_label" in sentiment:
        lines.append(f"📱 社群情緒：{sentiment['overall_label']}（score={sentiment.get('overall_sentiment', 0):+.2f}）")
    if "reddit_buzz" in sentiment:
        rb = sentiment["reddit_buzz"]
        lines.append(f"  Reddit r/{rb['subreddit']}: {rb['n_posts']} 熱門貼文，net={rb['net_sentiment']:+.2f}")
    if "news_recent" in sentiment:
        lines.append(f"  新聞 ({len(sentiment['news_recent'])} 則)：net={sentiment.get('news_net_sentiment', 0):+.2f}")
    return "\n".join(lines)
