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

# 30 分鐘快取（外部訊號變動不快，省 API 額度）
_CACHE_TTL = 1800
_cache: dict[str, tuple[float, dict]] = {}


def _cached(key: str) -> Optional[dict]:
    if key in _cache:
        ts, payload = _cache[key]
        if time.time() - ts < _CACHE_TTL:
            return payload
    return None


def _store(key: str, payload: dict) -> None:
    _cache[key] = (time.time(), payload)


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


def _fetch_macro_snapshot(client: httpx.Client) -> Optional[dict]:
    """抓 DXY + 10Y yield + M2 最新值。"""
    if not _FRED_KEY:
        return None
    out: dict = {}
    for name, sid in _FRED_SERIES.items():
        v = _fetch_fred_value(client, sid)
        if v:
            out[name] = v
    return out if out else None


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
        if "USDT" in sym or "USD" in sym:
            for fn, key in [
                (_fetch_funding_rate, None),
                (_fetch_open_interest, None),
                (_fetch_long_short_ratio, None),
                (_fetch_coinglass_liquidation, None),
            ]:
                try:
                    res = fn(client, sym)
                    if res:
                        out["derivatives"].update(res)
                except Exception:
                    continue

        # 情緒（與 symbol 無關，全市場一個值）
        try:
            fg = _fetch_fear_greed(client)
            if fg:
                out["sentiment"].update(fg)
        except Exception:
            pass

        # 總體（FRED key 有設才抓）
        if include_macro and _FRED_KEY:
            try:
                m = _fetch_macro_snapshot(client)
                if m:
                    out["macro"].update(m)
            except Exception:
                pass

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
