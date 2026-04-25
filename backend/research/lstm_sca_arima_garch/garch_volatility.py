"""GARCH(1,1) 對 returns 建模波動率。

注意：GARCH 不直接預測「價格方向」，而是預測「未來變異數 / 條件標準差」。
用途：給混合模型提供「波動率 regime」資訊。
"""

import warnings
from typing import Optional

import pandas as pd

try:
    from arch import arch_model
    _ARCH_AVAILABLE = True
except ImportError:
    _ARCH_AVAILABLE = False


def fit_predict(
    returns: pd.Series,
    n_forecast: int = 5,
    p: int = 1,
    q: int = 1,
) -> Optional[pd.Series]:
    """GARCH(p,q) 預測未來 n_forecast 步的條件變異數。

    Returns:
        pd.Series: 預測的條件標準差（不是變異數，方便跟價格比較）
        None：arch 未安裝
    """
    if not _ARCH_AVAILABLE:
        print("⚠ arch 未安裝，跳過 GARCH。執行：pip install arch")
        return None

    # GARCH 通常用 returns × 100（百分比化），數值穩定性較好
    r = returns.dropna() * 100

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = arch_model(r.values, vol="Garch", p=p, q=q, mean="constant")
        res = model.fit(disp="off", show_warning=False)

    forecast = res.forecast(horizon=n_forecast, reindex=False)
    # 取條件變異數的平方根（標準差）
    cond_std = (forecast.variance.values[-1] ** 0.5) / 100  # 還原回原 scale
    return pd.Series(cond_std, name="garch_cond_std")


def is_available() -> bool:
    return _ARCH_AVAILABLE
