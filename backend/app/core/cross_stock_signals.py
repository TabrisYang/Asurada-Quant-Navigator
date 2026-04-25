"""跨股票群體訊號：給 LLM 看到「個股 / 個幣 + 所屬集合 + 龍頭」的整體圖。

整合方式：在 chat.py 的 chart_state 組裝點，當意圖是「分析」時自動跑，
結果塞進 chart_state.crossStockSignals 給 LLM。

支援兩種市場：
  - 台股：用 TW_SECTORS 表找族群成員、龍頭、breadth
  - 加密：用「本地有資料的所有 USDT 對」（排除穩定幣）算 market breadth

訊號欄位：
  共通：
    - market_type           : "tw" / "crypto"
    - breadth_pct_advancing : 集合中「最近一根 K 線收漲」的 %
    - leadership_signal     : 龍頭股 / BTC 趨勢方向
    - leader_change_5d      : 龍頭最近 5 K 線漲跌幅
    - self_change_5d        : 自身最近 5 K 線漲跌幅
    - relative_strength     : 自身 vs 集合平均
    - interpretation        : 文字解讀（給 LLM 直接引用）
    - note_timeframe        : 提醒「最近一根 K 線」的時間單位
  台股獨有：
    - sector                : 所屬族群名稱
  加密獨有：
    - basket_size           : 動態 basket 成員數量
    - basket_members        : basket 成員清單
    - btc_dominance_proxy_pct : BTC 在 basket 中的 volume 佔比（市值代理）
    - alt_outperform_count  : 跑贏 BTC 的非 BTC 幣數量
    - market_regime         : "btc_led" / "alt_season" / "bearish" / "mixed"
"""

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from app.data.tw_sectors import TW_SECTORS
from app.data.fetchers.tw_stock_engine import tw_stock_engine
from app.data.fetchers.crypto_engine import crypto_engine
from app.utils.symbol import is_tw_stock


# ─── 加密市場常數 ────────────────────────────
_STABLECOINS = {"USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "FDUSD", "USDE"}
_CRYPTO_LEADER = "BTC/USDT"   # BTC 永遠是加密市場定錨
_MIN_BASKET_SIZE = 3          # 至少需要 3 個成員才算 breadth


def _find_sector_for_stock(stock_code: str) -> Optional[str]:
    """反查：給股票代號找所屬族群。回傳第一個匹配的族群名稱（一檔可能屬於多個族群）。"""
    for sector_name, codes in TW_SECTORS.items():
        if stock_code in codes:
            return sector_name
    return None


def _last_n_change_pct(df: pd.DataFrame, n: int = 5) -> Optional[float]:
    """計算最近 n 根 K 線的漲跌幅（%）。"""
    if df is None or len(df) < n + 1 or "close" not in df.columns:
        return None
    close = df["close"].dropna()
    if len(close) < n + 1:
        return None
    end_price = float(close.iloc[-1])
    start_price = float(close.iloc[-(n + 1)])
    if start_price <= 0:
        return None
    return round((end_price - start_price) / start_price * 100, 2)


def _last_bar_advanced(df: pd.DataFrame) -> Optional[bool]:
    """最近一根 K 線是否收漲（close > prev_close）。"""
    if df is None or len(df) < 2 or "close" not in df.columns:
        return None
    close = df["close"].dropna()
    if len(close) < 2:
        return None
    return float(close.iloc[-1]) > float(close.iloc[-2])


def compute_signals(symbol: str, timeframe: str = "1d") -> dict:
    """主函式：給 symbol 算出跨股票群體訊號摘要。"""
    if not is_tw_stock(symbol):
        return {"sector": None, "note": "非台股或加密貨幣（族群分析有限）"}

    stock_code = symbol.split("/")[0]
    sector = _find_sector_for_stock(stock_code)
    if not sector:
        return {"sector": None, "note": f"{stock_code} 未歸類至任何族群"}

    # 取族群成員
    peer_codes = [c for c in TW_SECTORS[sector] if c != stock_code]
    if not peer_codes:
        return {
            "sector": sector,
            "note": f"族群「{sector}」只有 {stock_code} 一檔成員，無法計算群體訊號",
        }

    # 載入族群成員 OHLCV（從本地 CSV）
    peer_data: dict[str, pd.DataFrame] = {}
    for code in peer_codes:
        try:
            df = tw_stock_engine.load_local_data(f"{code}/TWD", timeframe)
            if df is not None and not df.empty and len(df) >= 6:
                peer_data[code] = df
        except Exception:
            continue  # 該成員資料缺失就跳過

    if len(peer_data) < 2:
        return {
            "sector": sector,
            "note": f"族群「{sector}」本地有效資料 < 2 檔（{len(peer_data)} 檔），訊號無法計算。請先到「同步」面板下載族群所有成分股",
            "peers_loaded": len(peer_data),
            "peers_total": len(peer_codes),
        }

    # 1. Breadth：% 族群成員「最近一根收漲」
    advanced = sum(1 for df in peer_data.values() if _last_bar_advanced(df))
    breadth_pct = round(advanced / len(peer_data) * 100, 1)

    # 2. 族群 5 日動量（成員平均漲跌幅）
    changes = [_last_n_change_pct(df, 5) for df in peer_data.values()]
    changes = [c for c in changes if c is not None]
    sector_momentum_5d = round(sum(changes) / len(changes), 2) if changes else 0.0

    # 3. 龍頭股訊號：用第一個成員作為「龍頭」（TW_SECTORS 通常以權值排序）
    leader_code = TW_SECTORS[sector][0]
    leader_df = peer_data.get(leader_code)
    if leader_df is None:
        # 自己（stock_code）就是龍頭，從本地讀
        try:
            leader_df = tw_stock_engine.load_local_data(f"{leader_code}/TWD", timeframe)
        except Exception:
            leader_df = None
    leader_change_5d = _last_n_change_pct(leader_df, 5) if leader_df is not None else None

    if leader_change_5d is None:
        leadership_signal = "未知"
    elif leader_change_5d > 2:
        leadership_signal = "強勢"
    elif leader_change_5d < -2:
        leadership_signal = "弱勢"
    else:
        leadership_signal = "中性"

    # 4. 個股相對族群強弱
    self_df = tw_stock_engine.load_local_data(symbol, timeframe)
    self_change_5d = _last_n_change_pct(self_df, 5)
    if self_change_5d is None or sector_momentum_5d == 0:
        rs = None
    else:
        # RS 簡化版：個股 5 日漲跌 / 族群平均 5 日漲跌（避免除零）
        if abs(sector_momentum_5d) < 0.1:
            # 族群幾乎不動，用差值代替
            rs = round(self_change_5d - sector_momentum_5d, 2)
            rs_unit = "abs_diff_pct"
        else:
            rs = round(self_change_5d / sector_momentum_5d, 2)
            rs_unit = "ratio"

    # 5. 文字解讀（給 LLM 直接引用）
    interp_parts = [f"所屬族群：{sector}（{len(peer_data)} 檔有效對照）"]
    if breadth_pct >= 60:
        interp_parts.append(f"族群廣度強（{breadth_pct}% 成員上漲）")
    elif breadth_pct <= 40:
        interp_parts.append(f"族群廣度弱（僅 {breadth_pct}% 成員上漲）")
    else:
        interp_parts.append(f"族群廣度中性（{breadth_pct}% 成員上漲）")

    if leadership_signal != "未知":
        interp_parts.append(f"龍頭股 {leader_code} 5 日 {leader_change_5d:+.2f}%（{leadership_signal}）")

    if rs is not None and self_change_5d is not None:
        if abs(sector_momentum_5d) >= 0.1:
            if rs > 1.5:
                interp_parts.append(f"個股 5 日 {self_change_5d:+.2f}% 顯著強於族群（RS={rs}）")
            elif rs < 0.5:
                interp_parts.append(f"個股 5 日 {self_change_5d:+.2f}% 弱於族群（RS={rs}）")
            else:
                interp_parts.append(f"個股 5 日 {self_change_5d:+.2f}% 同步族群（RS={rs}）")

    return {
        "market_type": "tw",
        "sector": sector,
        "breadth_pct_advancing": breadth_pct,
        "sector_momentum_5d": sector_momentum_5d,
        "leadership_signal": leadership_signal,
        "leader_code": leader_code,
        "leader_change_5d": leader_change_5d,
        "self_change_5d": self_change_5d,
        "relative_strength_vs_sector": rs,
        "peers_loaded": len(peer_data),
        "peers_total": len(peer_codes),
        "interpretation": "；".join(interp_parts),
    }


# ─── 加密貨幣版 ────────────────────────────────

def _discover_crypto_basket(timeframe: str) -> list[str]:
    """掃描本地 crypto 資料，回傳可用的 USDT 對清單（排除穩定幣、衍生品）。"""
    try:
        all_data = crypto_engine.list_available_data()
    except Exception as e:
        logger.debug(f"list_available_data 失敗: {e}")
        return []

    basket: list[str] = []
    for entry in all_data:
        sym = entry.get("symbol", "")
        tf = entry.get("timeframe", "")
        # 對應 timeframe
        if tf != timeframe:
            continue
        # 含「derivatives」的衍生品跳過
        if "derivatives" in tf or "derivatives" in sym:
            continue
        # 必須是 X/USDT 對
        parts = sym.split("/")
        if len(parts) != 2 or parts[1] != "USDT":
            continue
        # 排除穩定幣 base（USDC/USDT 等）
        if parts[0] in _STABLECOINS:
            continue
        basket.append(sym)
    return sorted(set(basket))


def compute_signals_crypto(symbol: str, timeframe: str = "1d") -> dict:
    """加密貨幣版：用本地有資料的 USDT 對算 market breadth + BTC dominance + alt strength。"""
    basket = _discover_crypto_basket(timeframe)
    if len(basket) < _MIN_BASKET_SIZE:
        return {
            "market_type": "crypto",
            "basket_size": len(basket),
            "basket_members": basket,
            "note": f"本地僅 {len(basket)} 個加密貨幣對（{timeframe}），需 ≥ {_MIN_BASKET_SIZE} 個才能計算 breadth。請先到「同步」面板下載更多主流幣資料",
        }

    # 載入 basket OHLCV
    coin_data: dict[str, pd.DataFrame] = {}
    for c in basket:
        try:
            df = crypto_engine.load_local_data(c, timeframe)
            if df is not None and not df.empty and len(df) >= 6:
                coin_data[c] = df
        except Exception:
            continue

    if len(coin_data) < _MIN_BASKET_SIZE:
        return {
            "market_type": "crypto",
            "basket_size": len(coin_data),
            "basket_members": list(coin_data.keys()),
            "note": f"本地有 {len(basket)} 個 USDT 對但實際載入 {len(coin_data)} 個成功，需 ≥ {_MIN_BASKET_SIZE}",
        }

    # 1. Breadth：% 最近一根 K 線收漲（K 線方法 — 解決缺點 3）
    advanced = sum(1 for df in coin_data.values() if _last_bar_advanced(df))
    breadth_pct = round(advanced / len(coin_data) * 100, 1)

    # 2. Basket 5 K 線平均漲跌幅
    changes = [_last_n_change_pct(df, 5) for df in coin_data.values()]
    valid_changes = [c for c in changes if c is not None]
    basket_avg_change_5d = round(float(np.mean(valid_changes)), 2) if valid_changes else 0.0

    # 3. BTC 龍頭信號
    btc_df = coin_data.get(_CRYPTO_LEADER)
    btc_change_5d = _last_n_change_pct(btc_df, 5) if btc_df is not None else None
    if btc_change_5d is None:
        leadership_signal = "未知"
    elif btc_change_5d > 3:
        leadership_signal = "強勢"
    elif btc_change_5d < -3:
        leadership_signal = "弱勢"
    else:
        leadership_signal = "中性"

    # 4. 自身相對強弱
    self_df = coin_data.get(symbol)
    if self_df is None:
        try:
            self_df = crypto_engine.load_local_data(symbol, timeframe)
        except Exception:
            self_df = None
    self_change_5d = _last_n_change_pct(self_df, 5) if self_df is not None else None
    if self_change_5d is not None and abs(basket_avg_change_5d) >= 0.1:
        rs = round(self_change_5d / basket_avg_change_5d, 2)
    else:
        rs = None

    # 5. ★ BTC dominance proxy（USD 成交額 — volume × price，解決缺點 4 一部分）
    btc_dominance_proxy = None
    if btc_df is not None and "volume" in btc_df.columns and "close" in btc_df.columns:
        try:
            def _usd_turnover_5(df: pd.DataFrame) -> float:
                v = df["volume"].tail(5)
                p = df["close"].tail(5)
                if len(v) == 0 or len(p) == 0:
                    return 0.0
                return float((v.values * p.values).mean())
            btc_turnover = _usd_turnover_5(btc_df)
            total_turnover = sum(_usd_turnover_5(df) for df in coin_data.values())
            if total_turnover > 0:
                btc_dominance_proxy = round(btc_turnover / total_turnover * 100, 1)
        except Exception as _e:
            logger.debug(f"BTC dominance proxy 計算失敗: {_e}")

    # 6. ★ Alt outperform count（解決缺點 4 另一半）
    alt_outperform_count = 0
    alt_total = 0
    if btc_change_5d is not None:
        for c, df in coin_data.items():
            if c == _CRYPTO_LEADER:
                continue
            alt_total += 1
            chg = _last_n_change_pct(df, 5)
            if chg is not None and chg > btc_change_5d:
                alt_outperform_count += 1

    # 7. Market regime 推斷
    if breadth_pct >= 60 and (btc_change_5d or 0) > 0:
        market_regime = "btc_led"
    elif alt_total > 0 and alt_outperform_count >= alt_total * 0.6:
        market_regime = "alt_season"
    elif breadth_pct < 30 and (btc_change_5d or 0) < -3:
        market_regime = "bearish"
    else:
        market_regime = "mixed"

    # 8. 文字解讀（給 LLM 直接引用）
    interp_parts = [f"加密 basket {len(coin_data)} 檔（{', '.join(sorted(coin_data.keys())[:5])}{'...' if len(coin_data) > 5 else ''}）"]

    if breadth_pct >= 60:
        interp_parts.append(f"breadth 強（{breadth_pct}% 上漲）")
    elif breadth_pct <= 40:
        interp_parts.append(f"breadth 弱（僅 {breadth_pct}% 上漲）")
    else:
        interp_parts.append(f"breadth 中性（{breadth_pct}% 上漲）")

    if leadership_signal != "未知" and btc_change_5d is not None:
        interp_parts.append(f"BTC 5K 線 {btc_change_5d:+.2f}%（{leadership_signal}）")

    if btc_dominance_proxy is not None:
        interp_parts.append(f"BTC dominance proxy {btc_dominance_proxy}%")

    if alt_total > 0:
        interp_parts.append(f"{alt_outperform_count}/{alt_total} alts 跑贏 BTC")

    interp_parts.append(f"市場 regime: {market_regime}")

    if rs is not None and self_change_5d is not None:
        if rs > 1.5:
            interp_parts.append(f"自身 5K 線 {self_change_5d:+.2f}% 強於 basket（RS={rs}）")
        elif rs < 0.5 and rs > -0.5:
            interp_parts.append(f"自身 5K 線 {self_change_5d:+.2f}% 弱於 basket（RS={rs}）")

    return {
        "market_type": "crypto",
        "basket_size": len(coin_data),
        "basket_members": sorted(coin_data.keys()),
        "breadth_pct_advancing": breadth_pct,
        "basket_avg_change_5d": basket_avg_change_5d,
        "leadership_signal": leadership_signal,
        "leader": _CRYPTO_LEADER,
        "leader_change_5d": btc_change_5d,
        "self_change_5d": self_change_5d,
        "relative_strength_vs_basket": rs,
        "btc_dominance_proxy_pct": btc_dominance_proxy,
        "alt_outperform_count": alt_outperform_count,
        "alt_outperform_total": alt_total,
        "market_regime": market_regime,
        "interpretation": "；".join(interp_parts),
        "note_timeframe": f"基於 {timeframe} 最近一根 K 線（非 UTC 自然日）",
    }
