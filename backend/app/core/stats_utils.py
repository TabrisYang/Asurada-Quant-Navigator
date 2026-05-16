"""統計工具函式集中處。

集中 Wilson score 信賴區間計算，避免 executor.py / auto_scanner.py 等多處
重複實作邏輯一致但邊界處理不一致的問題。
"""
from __future__ import annotations

import math


def wilson_ci(hits: int, count: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 信賴區間（百分點，回傳 (lo, hi) tuple）。

    比 normal approximation 在小樣本/極端機率時更準。z=1.96 對應 95% CI。

    n=0 時回 (0.0, 100.0)：沒有資料 = 沒有任何關於機率的訊息，CI 是全範圍。
    （不採 (0.0, 0.0)，因為那會被誤讀為「機率很低」。）
    """
    if count <= 0:
        return (0.0, 100.0)
    p = hits / count
    z2 = z * z
    denom = 1.0 + z2 / count
    center = (p + z2 / (2 * count)) / denom
    margin = z * math.sqrt(p * (1 - p) / count + z2 / (4 * count * count)) / denom
    lo = max(0.0, (center - margin) * 100)
    hi = min(100.0, (center + margin) * 100)
    return (round(lo, 1), round(hi, 1))


def wilson_ci_lower(hits: int, n: int, z: float = 1.96) -> float:
    """Wilson 信賴區間下界（0-1 浮點，便利函式給只關心 lower bound 的場景用）。"""
    if n <= 0:
        return 0.0
    lo_pct, _ = wilson_ci(hits, n, z)
    return lo_pct / 100.0
