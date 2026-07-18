"""測試共用配置

契約測試設計原則（v154）：
- CI 環境沒有 backend/data/ohlcv/*.csv（整個目錄在 .gitignore）→
  所有契約測試「必須」用 make_ohlcv 合成資料＋patch_executor_data monkeypatch，
  真實 CSV 只能用 HAS_LOCAL_OHLCV skipif 做本地選配冒煙。
- monkeypatch 會自動還原；「禁止」像舊測試那樣永久覆蓋 _load_local_data。
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_ROOT))

# 本地是否有真實 OHLCV（CI 上恆為 False → 相關測試自動 skip）
HAS_LOCAL_OHLCV = (_BACKEND_ROOT / "data" / "ohlcv" / "BTC_USDT_1d.csv").exists()


@pytest.fixture(scope="session", autouse=True)
def _chdir_backend():
    """settings.data_dir='./data' 是相對路徑 — 統一 chdir 到 backend 根，
    讓 pytest 從 repo root 或 backend 跑都一致。"""
    prev = os.getcwd()
    os.chdir(_BACKEND_ROOT)
    yield
    os.chdir(prev)


def make_ohlcv(start: str = "2022-01-01", n: int = 600, seed: int = 7,
               base_price: float = 100.0) -> pd.DataFrame:
    """合成日線 OHLCV（固定 seed 保證可重現）。"""
    rng = np.random.RandomState(seed)
    close = base_price + np.cumsum(rng.randn(n))
    close = np.maximum(close, 1.0)
    ts = pd.date_range(start, periods=n, freq="1D")
    return pd.DataFrame({
        "timestamp": ts,
        "open": close,
        "high": close + rng.uniform(0.2, 1.0, n),
        "low": np.maximum(close - rng.uniform(0.2, 1.0, n), 0.01),
        "close": close,
        "volume": rng.uniform(1000, 10000, n),
    })


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    return make_ohlcv()


@pytest.fixture
def patch_executor_data(monkeypatch, synthetic_ohlcv):
    """把 executor 的本地資料 loader 換成合成資料（monkeypatch 自動還原）。"""
    import app.core.llm.executor as ex

    def _fake_load(symbol, timeframe, start=None, end=None):
        return synthetic_ohlcv.copy()

    monkeypatch.setattr(ex, "_load_local_data", _fake_load)
    return synthetic_ohlcv
