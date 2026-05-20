"""價格精度工具 — 與前端 K 線圖顯示位數完全一致。

前端 ChartView.tsx 的 getAutoPriceFormat 依價格量級決定小數位：
  - >= 1000  → 2 位（如 60000.00）
  - >= 1     → 4 位（如 2105.0200）
  - >= 0.01  → 6 位（如 0.251700）  ← ADA 等低價幣
  - <  0.01  → 8 位（如 0.00001234）

報告中所有價位（SMC / entry / SL / TP / 分批 / 指標）都應走此函式，
確保「報告寫的價位 == K 線圖上看到的價位」，避免 round(x,2) 把 0.2517 毀成 0.25。
"""

from __future__ import annotations

from typing import Optional


def price_precision(price: float) -> int:
    """回傳該價格量級對應的小數位數（對齊前端 getAutoPriceFormat）。"""
    ap = abs(price)
    if ap >= 1000:
        return 2
    if ap >= 1:
        return 4
    if ap >= 0.01:
        return 6
    return 8


def round_price(price: Optional[float]) -> Optional[float]:
    """依價格量級四捨五入到與 K 線圖一致的小數位。None 原樣回傳。"""
    if price is None:
        return None
    try:
        p = float(price)
    except (TypeError, ValueError):
        return price
    return round(p, price_precision(p))
