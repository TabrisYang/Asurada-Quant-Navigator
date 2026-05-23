"""阿斯拉量化系統 — v104 Q1：外部訊號快照（免費資料源）。

統一 fetcher，把以下三類資料即時注入 chart_state：
1. **衍生品快照**（Binance public API，無 key）
   - funding_rate: 最新 funding rate（8h 一次）
   - open_interest: 當前 + 24h 變化
   - long_short_ratio: 多空持倉比
2. **總體經濟**（FRED API + alternative.me）
   - DXY: 美元指數
   - US 10Y yield: 10 年期國債殖利率
   - Fear & Greed Index: 加密恐懼貪婪指數
3. **CP 值**：每次「全部分析」前 1 次，用 30 分鐘 cache 控成本

設計原則：
- 任何單一 API 失敗 → graceful fallback（其他資料照樣注入）
- 全部用 httpx 同步 client（簡單，避免 chat.py 改 async flow）
- timeout 5s，總耗時上限 ~15s（即使全失敗）
- 30 分鐘 cache key=symbol，跨多次分析共用
"""

from __future__ import annotations

import time
from typing import Any, Optional

import httpx
from loguru import logger

# 30 分鐘新鮮快取（外部訊號變動不快，省 API 額度）
_CACHE_TTL = 1800
# v119.1：6 小時 stale limit — 新 fetch 全失敗時用過期 cache 比沒有好
# （funding rate 8h 才換一次 / OI / order_book 6h 內變化通常 < 5%，仍 informative）
_STALE_LIMIT = 6 * 3600
_cache: dict[str, tuple[float, dict]] = {}


def _cached(key: str) -> Optional[dict]:
    """新鮮 cache（< _CACHE_TTL = 30 分）"""
    if key in _cache:
        ts, payload = _cache[key]
        if time.time() - ts < _CACHE_TTL:
            return payload
    return None


def _stale_cached(key: str) -> Optional[tuple[float, dict]]:
    """過期但仍在 _STALE_LIMIT（6h）內的 cache，fallback 用。

    Returns (age_seconds, payload) 或 None。
    """
    if key in _cache:
        ts, payload = _cache[key]
        age = time.time() - ts
        if age < _STALE_LIMIT:
            return age, payload
    return None


def _store(key: str, payload: dict) -> None:
    _cache[key] = (time.time(), payload)


def _fetch_with_retry(fn, *args, max_retries: int = 3, base_backoff: float = 0.5):
    """v119.1：重試包裝 — 失敗時 0.5s / 1.0s / 2.0s 指數 backoff 重試。

    Binance / Coinbase / SoSoValue 等 API 偶爾 rate limit 或瞬斷，
    重試一兩次幾乎都能成功。
    """
    last_err: Optional[Exception] = None
    last_falsy_attempt = 0
    for attempt in range(max_retries):
        try:
            result = fn(*args)
            if result:
                return result
            last_falsy_attempt = attempt + 1  # fn 回 None/empty 但沒 raise
        except Exception as e:
            last_err = e
        if attempt < max_retries - 1:
            time.sleep(base_backoff * (2 ** attempt))
    # v141.2：debug → warning（升級 log level，讓 production 看得到全失敗）
    if last_err:
        logger.warning(f"[external] retry exhausted ({getattr(fn, '__name__', repr(fn))}): {last_err}")
    elif last_falsy_attempt:
        logger.warning(f"[external] retry exhausted ({getattr(fn, '__name__', repr(fn))}): all {max_retries} attempts returned falsy")
    return None


def _to_binance_symbol(symbol: str) -> str:
    """BTC/USDT → BTCUSDT"""
    return symbol.replace("/", "").replace("-", "").upper()


# ─── 衍生品（Binance public API） ─────────────────────────

def _fetch_funding_rate(client: httpx.Client, sym: str) -> Optional[dict]:
    """最新 funding rate（每 8 小時更新一次）。"""
    try:
        r = client.get(
            "https://fapi.binance.com/fapi/v1/premiumIndex",
            params={"symbol": sym},
            timeout=5.0,
        )
        r.raise_for_status()
        data = r.json()
        rate = float(data.get("lastFundingRate") or 0)
        return {
            "funding_rate": rate,
            "funding_rate_pct": round(rate * 100, 4),  # %
            "next_funding_time": data.get("nextFundingTime"),
        }
    except Exception as e:
        logger.debug(f"[external] funding_rate fetch 失敗 ({sym}): {e}")
        return None


def _fetch_open_interest(client: httpx.Client, sym: str) -> Optional[dict]:
    """當前 OI 名目 + 24h 變化（用 5m 區間粗算）。"""
    try:
        # 當前 OI
        r1 = client.get(
            "https://fapi.binance.com/fapi/v1/openInterest",
            params={"symbol": sym},
            timeout=5.0,
        )
        r1.raise_for_status()
        cur_oi = float(r1.json().get("openInterest") or 0)

        # 歷史 OI（5m × ~288 根 = 24h）
        r2 = client.get(
            "https://fapi.binance.com/futures/data/openInterestHist",
            params={"symbol": sym, "period": "5m", "limit": 288},
            timeout=5.0,
        )
        r2.raise_for_status()
        hist = r2.json()
        if hist and len(hist) >= 2:
            old_oi = float(hist[0].get("sumOpenInterest") or 0)
            change_pct = ((cur_oi - old_oi) / old_oi * 100) if old_oi > 0 else 0
        else:
            change_pct = 0
        return {
            "open_interest": cur_oi,
            "open_interest_24h_change_pct": round(change_pct, 2),
        }
    except Exception as e:
        logger.debug(f"[external] OI fetch 失敗 ({sym}): {e}")
        return None


def _fetch_order_book(client: httpx.Client, sym: str, limit: int = 20) -> Optional[dict]:
    """v106 C2：Binance L2 order book snapshot（無 key，免費）。

    抓 spot /api/v3/depth，取 top N bids/asks，算：
    - top_5_bids / top_5_asks（給 LLM 引用「大單在 $X 堆疊」）
    - total_bid_depth_usd / total_ask_depth_usd（top N 累積）
    - imbalance_ratio = bid / ask（>1.5 偏多買壓 / <0.67 偏空賣壓）
    - spread_bps = (best_ask - best_bid) / mid * 10000
    - mid_price
    """
    try:
        r = client.get(
            "https://fapi.binance.com/fapi/v1/depth",
            params={"symbol": sym, "limit": limit},
            timeout=5.0,
        )
        r.raise_for_status()
        data = r.json()
        bids = [(float(p), float(q)) for p, q in data.get("bids", [])][:limit]
        asks = [(float(p), float(q)) for p, q in data.get("asks", [])][:limit]
        if not bids or not asks:
            return None
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid = (best_bid + best_ask) / 2 if (best_bid and best_ask) else 0
        spread_bps = ((best_ask - best_bid) / mid * 10000) if mid > 0 else 0
        bid_depth_usd = sum(p * q for p, q in bids)
        ask_depth_usd = sum(p * q for p, q in asks)
        imbalance = (bid_depth_usd / ask_depth_usd) if ask_depth_usd > 0 else 0
        return {
            "ob_mid_price": round(mid, 6),
            "ob_spread_bps": round(spread_bps, 2),
            "ob_top_5_bids": [{"price": round(p, 6), "qty": round(q, 4)} for p, q in bids[:5]],
            "ob_top_5_asks": [{"price": round(p, 6), "qty": round(q, 4)} for p, q in asks[:5]],
            "ob_bid_depth_usd": round(bid_depth_usd, 0),
            "ob_ask_depth_usd": round(ask_depth_usd, 0),
            "ob_imbalance_ratio": round(imbalance, 3),
            "ob_imbalance_label": (
                "強買壓" if imbalance > 1.5
                else "偏買壓" if imbalance > 1.15
                else "強賣壓" if imbalance < 0.67
                else "偏賣壓" if imbalance < 0.85
                else "平衡"
            ),
        }
    except Exception as e:
        logger.debug(f"[external] order_book fetch 失敗 ({sym}): {e}")
        return None


def _fetch_long_short_ratio(client: httpx.Client, sym: str) -> Optional[dict]:
    """多空持倉比（top traders + global）。"""
    try:
        out: dict = {}
        # 全網多空比
        r1 = client.get(
            "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
            params={"symbol": sym, "period": "5m", "limit": 1},
            timeout=5.0,
        )
        r1.raise_for_status()
        data = r1.json()
        if data:
            out["global_long_short_ratio"] = float(data[0].get("longShortRatio") or 0)

        # 大戶多空比
        r2 = client.get(
            "https://fapi.binance.com/futures/data/topLongShortPositionRatio",
            params={"symbol": sym, "period": "5m", "limit": 1},
            timeout=5.0,
        )
        r2.raise_for_status()
        data = r2.json()
        if data:
            out["top_traders_long_short_ratio"] = float(data[0].get("longShortRatio") or 0)
        return out if out else None
    except Exception as e:
        logger.debug(f"[external] long_short fetch 失敗 ({sym}): {e}")
        return None


# 註：Binance allForceOrders 已要求認證（2024 後）。
# v105 A2：嘗試 Coinglass 公開 endpoint 取 24h liquidation；多數情況會失敗（anti-bot），
#         graceful fallback → derivatives 不含 liquidation 欄位。
def _fetch_coinglass_liquidation(client: httpx.Client, sym: str) -> Optional[dict]:
    """嘗試 Coinglass 公開 endpoint 抓 24h 清算（多空合計 + 拆分）。

    Coinglass free tier 政策變動頻繁，2026 起多數 endpoint 已封閉。
    這個 fetcher 採嘗試式 + graceful fallback，全失敗回 None。
    """
    try:
        # 嘗試 open API（若使用者後續設 COINGLASS_API_KEY env，可用付費 tier）
        headers = {"User-Agent": "Mozilla/5.0", "accept": "application/json"}
        api_key = os.environ.get("COINGLASS_API_KEY")
        if api_key:
            headers["coinglassSecret"] = api_key
            url = "https://open-api-v3.coinglass.com/api/futures/liquidation/v2/aggregated-history"
            r = client.get(url, params={"symbol": sym.replace("USDT", ""), "interval": "1d"},
                           headers=headers, timeout=5.0)
            r.raise_for_status()
            data = r.json().get("data") or []
            if data:
                latest = data[-1]
                return {
                    "liq_24h_long_usd": float(latest.get("longLiquidationUsd") or 0),
                    "liq_24h_short_usd": float(latest.get("shortLiquidationUsd") or 0),
                    "liq_24h_total_usd": float(latest.get("longLiquidationUsd") or 0)
                                          + float(latest.get("shortLiquidationUsd") or 0),
                    "liq_source": "coinglass_paid",
                }
        # 沒 API key 直接 skip（公開 endpoint 全擋了）
        return None
    except Exception as e:
        logger.debug(f"[external] coinglass liquidation 失敗 ({sym}): {e}")
        return None


# ─── v119.4：Coinbase Premium ─────────────────

def _fetch_coinbase_premium(client: httpx.Client, sym: str) -> Optional[dict]:
    """v119.4：Coinbase（美國機構主場）vs Binance（全球散戶）即時價差。

    正溢價 → 美國機構主動承接 → 偏多訊號
    負溢價 → 美國散戶或原生加密拋售 → 中性 / 短空訊號（但若同時 ETF 流入仍偏多）

    需要主流幣才有意義（Coinbase Spot 上架的）：BTC / ETH / SOL / XRP / ADA / DOGE /
    AVAX / DOT / LINK / LTC / BCH / UNI / ATOM 等。

    sym 格式：BTCUSDT（Binance perp 格式），自動 derive base = BTC → coinbase 用 BTC-USD
    """
    if not sym.endswith("USDT"):
        return None
    base = sym[:-4]  # ETHUSDT → ETH

    try:
        # Coinbase spot price（無 key，公開 API）
        r1 = client.get(
            f"https://api.coinbase.com/v2/prices/{base}-USD/spot",
            timeout=5.0,
        )
        r1.raise_for_status()
        cb_data = r1.json().get("data") or {}
        cb_price = float(cb_data.get("amount") or 0)
        if cb_price <= 0:
            return None

        # Binance perp price（fapi）— 跟 fund/OI 同個 source 保持一致
        r2 = client.get(
            "https://fapi.binance.com/fapi/v1/ticker/price",
            params={"symbol": sym},
            timeout=5.0,
        )
        r2.raise_for_status()
        bn_price = float(r2.json().get("price") or 0)
        if bn_price <= 0:
            return None

        premium_pct = round((cb_price - bn_price) / bn_price * 100, 4)

        # Classify label（用百分位門檻，後續 v120 bucket 用同樣門檻）
        if premium_pct >= 0.05:
            label = "positive_high"  # 美國機構強烈承接
        elif premium_pct >= 0.01:
            label = "positive"
        elif premium_pct >= -0.01:
            label = "neutral"
        elif premium_pct >= -0.05:
            label = "negative"
        else:
            label = "negative_high"  # 美國拋售壓力強

        return {
            "coinbase_price": cb_price,
            "binance_perp_price": bn_price,
            "coinbase_premium_pct": premium_pct,
            "coinbase_premium_label": label,
        }
    except Exception as e:
        logger.debug(f"[external] coinbase_premium fetch 失敗 ({sym}): {e}")
        return None


# ─── 情緒指標 ─────────────────────────

def _fetch_fear_greed(client: httpx.Client) -> Optional[dict]:
    """alternative.me Fear & Greed Index（免費，每日一筆）。"""
    try:
        r = client.get("https://api.alternative.me/fng/", params={"limit": 1}, timeout=5.0)
        r.raise_for_status()
        data = r.json().get("data") or []
        if not data:
            return None
        item = data[0]
        return {
            "fear_greed_value": int(item.get("value") or 50),
            "fear_greed_label": item.get("value_classification", "Neutral"),
        }
    except Exception as e:
        logger.debug(f"[external] fear_greed fetch 失敗: {e}")
        return None


# ─── 總體經濟（FRED API） ─────────────────────────

# FRED API 100% 免費，但需要 API key（註冊即得，無付費）。
# 這裡用 series_id 直接抓最新值。若沒設 key，跳過 macro 區塊。
import os

_FRED_KEY = os.environ.get("FRED_API_KEY", "")

# 常用 series：
#   DTWEXBGS = Trade-Weighted USD Index (Broad)
#   DGS10 = 10-Year Treasury Constant Maturity Rate
#   M2SL = M2 Money Stock
_FRED_SERIES = {
    "dxy": "DTWEXBGS",
    "us10y": "DGS10",
    "m2": "M2SL",
}


def _fetch_fred_value(client: httpx.Client, series_id: str) -> Optional[dict]:
    """抓 FRED 單一 series 最新觀測值 + 前一個觀測值（算變化）。"""
    if not _FRED_KEY:
        return None
    try:
        r = client.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": series_id,
                "api_key": _FRED_KEY,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 5,
            },
            timeout=5.0,
        )
        r.raise_for_status()
        obs = r.json().get("observations") or []
        # 過濾掉 "." (FRED 表示無資料)
        valid = [o for o in obs if o.get("value") not in (".", None, "")]
        if len(valid) < 2:
            return None
        latest = float(valid[0]["value"])
        prev = float(valid[1]["value"])
        change_pct = ((latest - prev) / prev * 100) if prev != 0 else 0
        return {
            "value": round(latest, 4),
            "change_pct": round(change_pct, 3),
            "as_of": valid[0].get("date"),
        }
    except Exception as e:
        logger.debug(f"[external] FRED {series_id} fetch 失敗: {e}")
        return None


def _fetch_macro_yfinance() -> Optional[dict]:
    """免費無 key 補總體經濟（DXY + 美十年債殖利率）。M2 yfinance 無 → 略過。

    yfinance 為專案既有依賴（見 tw_fundamental.py）；macro 在 30 分快取後才重抓，
    2 個 ticker 成本可接受。FRED key 缺席時的免費 fallback。
    """
    try:
        import yfinance as yf
    except Exception:
        return None
    out: dict = {}
    for name, ticker in (("dxy", "DX-Y.NYB"), ("us10y", "^TNX")):
        try:
            hist = yf.Ticker(ticker).history(period="7d")
            if hist is None or hist.empty:
                continue
            closes = hist["Close"].dropna()
            if len(closes) < 2:
                continue
            latest = float(closes.iloc[-1])
            prev = float(closes.iloc[-2])
            change_pct = ((latest - prev) / prev * 100) if prev != 0 else 0
            out[name] = {
                "value": round(latest, 4),
                "change_pct": round(change_pct, 3),
                "as_of": str(closes.index[-1].date()),
                "source": "yfinance",
            }
        except Exception as e:
            logger.debug(f"[external] yfinance macro {ticker} 失敗: {e}")
    return out if out else None


def _fetch_macro_snapshot(client: httpx.Client) -> Optional[dict]:
    """抓 DXY + 10Y yield + M2。有 FRED key 走 FRED（含 M2）；否則 yfinance 免費補 DXY/10Y。"""
    if _FRED_KEY:
        out: dict = {}
        for name, sid in _FRED_SERIES.items():
            v = _fetch_fred_value(client, sid)
            if v:
                v["source"] = "fred"
                out[name] = v
        if out:
            return out
        # FRED key 有設但全失敗 → 退 yfinance 免費源
    return _fetch_macro_yfinance()


# ─── 公開 API ─────────────────────────

def get_signals_snapshot(symbol: str, include_macro: bool = True) -> dict:
    """v104 Q1：取得當前外部訊號快照（衍生品 + 情緒 + 總體經濟）。

    用 30 分鐘 cache 避免重複打 API。即使部分 API 失敗也回傳可用部分。

    Returns dict 結構：
    {
      "derivatives": {funding_rate / OI / long_short_ratio / liquidations / ...},
      "sentiment": {fear_greed_value, fear_greed_label},
      "macro": {dxy, us10y, m2},
      "fetched_at": "2026-04-28T...",
      "cached": False,  # True if from cache
    }
    """
    cache_key = f"{symbol}|{include_macro}"
    cached = _cached(cache_key)
    if cached:
        # v141.2：cache hit 也 log，幫忙抓「cache 卡 empty」這類 poisoning 問題
        _c_deriv = len(cached.get("derivatives") or {})
        _c_sent = len(cached.get("sentiment") or {})
        _c_macro = len(cached.get("macro") or {})
        if _c_deriv == 0 and _c_sent == 0 and _c_macro == 0:
            logger.warning(
                f"[external] CACHE HIT EMPTY {cache_key} — cache 卡 empty，"
                f"重啟 backend 或呼叫 clear_cache() 可清除"
            )
        else:
            logger.info(f"[external] cache HIT {cache_key} deriv={_c_deriv} sent={_c_sent} macro={_c_macro}")
        return {**cached, "cached": True}

    sym = _to_binance_symbol(symbol)
    out: dict[str, Any] = {
        "derivatives": {},
        "sentiment": {},
        "macro": {},
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cached": False,
    }

    with httpx.Client() as client:
        # 衍生品（USDT 永續才有意義；台股 / 非 perp 跳過）
        # v119.1：每個 fetcher 用 retry 包，三次失敗才放棄
        if "USDT" in sym or "USD" in sym:
            for fn in (
                _fetch_funding_rate,
                _fetch_open_interest,
                _fetch_long_short_ratio,
                _fetch_coinglass_liquidation,
                _fetch_order_book,
                _fetch_coinbase_premium,  # v119.4
            ):
                res = _fetch_with_retry(fn, client, sym)
                if res:
                    out["derivatives"].update(res)

        # 情緒（與 symbol 無關，全市場一個值）
        fg = _fetch_with_retry(_fetch_fear_greed, client)
        if fg:
            out["sentiment"].update(fg)

        # 總體（有 FRED key 走 FRED 含 M2；否則 yfinance 免費補 DXY/10Y）
        if include_macro:
            m = _fetch_with_retry(_fetch_macro_snapshot, client)
            if m:
                out["macro"].update(m)

    # v119.1：若整個 fetch 都失敗（derivatives + sentiment 都空）→ 用 stale cache fallback
    # 比「無衍生品快照」好太多，明確標 stale_seconds 讓 LLM 知道資料新鮮度
    if not out["derivatives"] and not out["sentiment"] and not out["macro"]:
        stale = _stale_cached(cache_key)
        if stale:
            age, payload = stale
            logger.warning(
                f"[external] 全部 fetch 失敗，用 stale cache（age={int(age)}s, key={cache_key}）"
            )
            return {**payload, "cached": True, "stale": True, "stale_seconds": int(age)}

    # v141.2：fresh fetch 結束時 log，diagnose 多 API 失敗模式
    _deriv = len(out["derivatives"])
    _sent = len(out["sentiment"])
    _macro = len(out["macro"])
    if _deriv == 0 and _sent == 0 and _macro == 0:
        logger.error(
            f"[external] FRESH FETCH ALL FAILED {cache_key} — "
            f"derivatives/sentiment/macro 全空、cache 也無 stale fallback 可用。"
            f"檢查網路 / API 限制 / SSL；後續 30 分鐘 cache 會回 empty。"
        )
    else:
        logger.info(
            f"[external] fresh fetch {cache_key} "
            f"deriv={_deriv} sent={_sent} macro={_macro}"
        )
    _store(cache_key, out)
    return out


def format_signals_summary(signals: dict) -> str:
    """把 snapshot 整理成 LLM prompt 友善字串（給 chart_state 注入用）。"""
    lines: list[str] = []

    deriv = signals.get("derivatives") or {}
    if deriv:
        bits = []
        if "funding_rate_pct" in deriv:
            fr = deriv["funding_rate_pct"]
            tag = ""
            if abs(fr) > 0.05:
                tag = "（多方過熱）" if fr > 0 else "（空方過熱）"
            bits.append(f"funding={fr:+.4f}%{tag}")
        if "open_interest_24h_change_pct" in deriv:
            oc = deriv["open_interest_24h_change_pct"]
            bits.append(f"OI 24h {oc:+.1f}%")
        if "global_long_short_ratio" in deriv:
            r = deriv["global_long_short_ratio"]
            tag = "（極端多）" if r > 2.5 else "（極端空）" if r < 0.4 else ""
            bits.append(f"全網多空比={r:.2f}{tag}")
        if "top_traders_long_short_ratio" in deriv:
            bits.append(f"大戶多空比={deriv['top_traders_long_short_ratio']:.2f}")
        if "liq_24h_total_usd" in deriv:
            tot = deriv["liq_24h_total_usd"]
            long_l = deriv.get("liq_24h_long_usd", 0)
            short_l = deriv.get("liq_24h_short_usd", 0)
            bits.append(f"24h 清算 ${tot/1e6:.1f}M（多 ${long_l/1e6:.1f}M / 空 ${short_l/1e6:.1f}M）")
        if bits:
            lines.append("📉 衍生品：" + " | ".join(bits))

        # v106 C2：order book L2 摘要
        if "ob_imbalance_ratio" in deriv:
            ob_bits = [
                f"imbalance={deriv['ob_imbalance_ratio']:.2f}（{deriv.get('ob_imbalance_label','?')}）",
                f"買單深度 ${deriv.get('ob_bid_depth_usd', 0)/1e6:.2f}M",
                f"賣單深度 ${deriv.get('ob_ask_depth_usd', 0)/1e6:.2f}M",
                f"spread {deriv.get('ob_spread_bps', 0):.1f}bps",
            ]
            lines.append("📚 L2 訂單簿：" + " | ".join(ob_bits))
            top_bids = deriv.get("ob_top_5_bids") or []
            top_asks = deriv.get("ob_top_5_asks") or []
            if top_bids:
                lines.append("  Top 5 買單: " + ", ".join(
                    f"${b['price']}×{b['qty']}" for b in top_bids[:5]
                ))
            if top_asks:
                lines.append("  Top 5 賣單: " + ", ".join(
                    f"${a['price']}×{a['qty']}" for a in top_asks[:5]
                ))

    sent = signals.get("sentiment") or {}
    if sent.get("fear_greed_value") is not None:
        lines.append(f"😱 Fear&Greed：{sent['fear_greed_value']}（{sent.get('fear_greed_label','?')}）")

    macro = signals.get("macro") or {}
    if macro:
        bits = []
        if "dxy" in macro:
            bits.append(f"DXY {macro['dxy']['value']}（{macro['dxy']['change_pct']:+.2f}%）")
        if "us10y" in macro:
            bits.append(f"10Y {macro['us10y']['value']}%（{macro['us10y']['change_pct']:+.2f}%）")
        if "m2" in macro:
            bits.append(f"M2 ${macro['m2']['value']:.0f}B")
        if bits:
            lines.append("🌍 總體：" + " | ".join(bits))

    return "\n".join(lines) if lines else ""


def clear_cache() -> int:
    """清除快取（測試 / 強制重抓用）。"""
    n = len(_cache)
    _cache.clear()
    return n
