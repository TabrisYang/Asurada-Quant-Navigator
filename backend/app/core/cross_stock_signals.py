"""跨股票群體訊號：給 LLM 看到「個股 + 所屬族群 + 龍頭」的整體圖。

整合方式：在 chat.py 的 chart_state 組裝點，當意圖是「分析」時自動跑，
結果塞進 chart_state.crossStockSignals 給 LLM。

提供的訊號：
  1. sector              : 該股所屬族群（若有歸類）
  2. breadth_pct_advancing : 族群中「最近一根 K 線收漲」的成員 %
  3. sector_momentum_5d  : 族群成員 5 日漲跌幅平均（%）
  4. leadership_signal   : 龍頭股（族群中第一檔）最近的趨勢方向
  5. relative_strength_vs_sector : 個股 5 日漲跌 vs 族群平均（>1=強）
  6. interpretation      : 一句話文字解讀（給 LLM 直接引用）

注意：此模組為「輕量」實作 — 不做完整 sector_analyzer 的指標分析，
只算「群體共動」相關的快速指標。需要更深入分析仍可呼叫 analyze_sector function。
"""

from typing import Optional

import pandas as pd
from loguru import logger

from app.data.tw_sectors import TW_SECTORS
from app.data.fetchers.tw_stock_engine import tw_stock_engine
from app.utils.symbol import is_tw_stock


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
