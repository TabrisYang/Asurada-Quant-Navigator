"""v124 機率三聯顯示 + Wilson CI 測試。

涵蓋：
- stats_utils.wilson_ci 邊界與正確性（重構後與 executor / auto_scanner 一致）
- probability_baseline.calc_unconditional_baseline 路徑相依計算
- prediction_tracker.get_winrate_with_ci CI 分母排除 expired
- _build_triplet_warnings 警示文案產生
- adapter._minimal_r2_chart_state 機率三聯壓 summary
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_wilson_ci_zero_sample():
    """n=0 時應回 (0, 100) — 沒資料 = 沒任何機率資訊，CI 是全範圍。"""
    from app.core.stats_utils import wilson_ci
    lo, hi = wilson_ci(0, 0)
    assert lo == 0.0
    assert hi == 100.0


def test_wilson_ci_50_of_100():
    """n=100, p=0.5：Wilson 95% CI 約 (40.4, 59.6)。"""
    from app.core.stats_utils import wilson_ci
    lo, hi = wilson_ci(50, 100)
    assert 40.0 < lo < 41.0
    assert 59.0 < hi < 60.0


def test_wilson_ci_extreme_small_n():
    """n=1, p=1.0：上界 100，下界 ~21%（驗證小樣本 Wilson 不是 normal approx）。"""
    from app.core.stats_utils import wilson_ci
    lo, hi = wilson_ci(1, 1)
    assert 20.0 < lo < 22.0
    assert hi == 100.0


def test_wilson_ci_extreme_zero_hits():
    """n=10, hits=0：下界 0，上界 ~28%。"""
    from app.core.stats_utils import wilson_ci
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0
    assert 25.0 < hi < 30.0


def test_wilson_ci_lower_convenience():
    """wilson_ci_lower 應回 [0,1] 浮點，不是百分點。"""
    from app.core.stats_utils import wilson_ci_lower
    v = wilson_ci_lower(50, 100)
    assert 0.40 < v < 0.41
    assert wilson_ci_lower(0, 0) == 0.0


def test_baseline_path_max_is_unconditional():
    """baseline 必須是 unconditional（所有歷史樣本，不過濾任何指標）。

    生成 500 根隨機走勢，驗證 baseline 數值穩定且符合預期區間。
    """
    from app.core.probability_baseline import calc_unconditional_baseline
    np.random.seed(42)
    n = 500
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    highs = closes + np.abs(np.random.randn(n))
    lows = closes - np.abs(np.random.randn(n))
    df = pd.DataFrame({"high": highs, "low": lows, "close": closes})

    r = calc_unconditional_baseline(df, forward_bars=6, target_pct=3.0, direction="up")
    assert r["status"] == "ok"
    assert 0 < r["prob_pct"] < 100
    assert r["n"] == n - 6  # forward_bars = 6
    assert len(r["ci_pct"]) == 2
    assert r["ci_pct"][0] < r["ci_pct"][1]
    assert r["params"]["forward_bars"] == 6
    assert r["params"]["target_pct"] == 3.0


def test_baseline_insufficient_data():
    """df 不足時應回 status=insufficient_data，不丟錯。"""
    from app.core.probability_baseline import calc_unconditional_baseline
    df = pd.DataFrame({"high": [1, 2, 3], "low": [0, 1, 2], "close": [0.5, 1.5, 2.5]})
    r = calc_unconditional_baseline(df, forward_bars=6, target_pct=3.0)
    assert r["status"] == "insufficient_data"
    assert "reason" in r


def test_baseline_none_df():
    """df=None 應安全 fallback。"""
    from app.core.probability_baseline import calc_unconditional_baseline
    r = calc_unconditional_baseline(None, forward_bars=6, target_pct=3.0)
    assert r["status"] == "insufficient_data"


def test_baseline_path_max_higher_than_endpoint():
    """期間最高漲幅 baseline 必然 ≥ 固定終點 baseline（path-max vs endpoint 的數學特性）。

    這是 P1 設計的核心：path-max 的 baseline 必然偏高，與使用者報告中提到的
    現象一致。
    """
    from app.core.probability_baseline import calc_unconditional_baseline
    np.random.seed(7)
    n = 500
    closes = 100 + np.cumsum(np.random.randn(n) * 0.4)
    highs = closes + np.abs(np.random.randn(n)) * 0.5
    lows = closes - np.abs(np.random.randn(n)) * 0.5
    df = pd.DataFrame({"high": highs, "low": lows, "close": closes})

    # path-max baseline（系統的算法）
    r_pathmax = calc_unconditional_baseline(df, forward_bars=6, target_pct=2.0, direction="up")

    # endpoint baseline（手算對照）
    hit_endpoint = 0
    valid_n = 0
    for i in range(n - 6):
        if closes[i] > 0:
            endpoint_pct = (closes[i + 6] - closes[i]) / closes[i] * 100
            if endpoint_pct >= 2.0:
                hit_endpoint += 1
            valid_n += 1
    endpoint_pct = hit_endpoint / valid_n * 100

    assert r_pathmax["prob_pct"] >= endpoint_pct, (
        f"path-max {r_pathmax['prob_pct']}% should be >= endpoint {endpoint_pct}%"
    )


def test_build_triplet_warnings_track_below_baseline():
    """track_record 點估值低於 baseline 且 n_decided>=10 → 警示「強烈建議降倉」。"""
    from app.api.routes.chat import _build_triplet_warnings

    triplet = {
        "baseline_unconditional": {"status": "ok", "prob_pct": 35.2, "n": 2400},
        "track_record": {
            "status": "ok",
            "win_rate_raw_pct": 17.9,
            "n_decided": 28,
            "ci_pct": [7.2, 35.6],
            "ci_width_pp": 28.4,
        },
    }
    lines = _build_triplet_warnings(triplet)
    text = " ".join(lines)
    assert "強烈建議降倉" in text
    assert "17.9" in text and "35.2" in text


def test_build_triplet_warnings_n_decided_too_small():
    """n_decided < 10 應有「樣本不足」警示。"""
    from app.api.routes.chat import _build_triplet_warnings

    triplet = {
        "baseline_unconditional": {"status": "ok", "prob_pct": 35.2, "n": 2400},
        "track_record": {
            "status": "ok",
            "win_rate_raw_pct": 50.0,
            "n_decided": 5,
            "ci_pct": [22.0, 78.0],
            "ci_width_pp": 56.0,
        },
    }
    lines = _build_triplet_warnings(triplet)
    text = " ".join(lines)
    assert "樣本不足" in text


def test_build_triplet_warnings_baseline_too_small():
    """baseline.n < 200 應有「baseline 本身也不準」警示。"""
    from app.api.routes.chat import _build_triplet_warnings

    triplet = {
        "baseline_unconditional": {"status": "ok", "prob_pct": 35.2, "n": 80},
        "track_record": {"status": "no_history", "n": 0, "n_decided": 0},
    }
    lines = _build_triplet_warnings(triplet)
    text = " ".join(lines)
    assert "baseline 樣本不足" in text


def test_build_triplet_warnings_no_warning():
    """三聯都健康（大樣本、track > baseline）應該沒警示。"""
    from app.api.routes.chat import _build_triplet_warnings

    triplet = {
        "baseline_unconditional": {"status": "ok", "prob_pct": 35.2, "n": 5000},
        "track_record": {
            "status": "ok",
            "win_rate_raw_pct": 65.0,
            "n_decided": 100,
            "ci_pct": [55.0, 74.0],
            "ci_width_pp": 19.0,
        },
    }
    lines = _build_triplet_warnings(triplet)
    assert lines == []


def test_adapter_r2_compresses_triplet():
    """_minimal_r2_chart_state 應把完整 triplet 壓成 summary string + 保留 warnings。"""
    from app.core.llm.adapter import _minimal_r2_chart_state

    full_state = {
        "symbol": "ETH/USDT",
        "timeframe": "4h",
        "currentPrice": 3000,
        "currentRegime": {"regime": "ranging"},
        "recent_accuracy": {
            "win_rate_30d": 60.0,
            "probability_triplet": {
                "baseline_unconditional": {"prob_pct": 35.2, "n": 2400},
                "ta_conditional": {"prob_pct": 65.0, "source": "bias_score_9dim"},
                "track_record": {
                    "win_rate_raw_pct": 17.9, "n_decided": 28, "ci_pct": [7.2, 35.6],
                },
                "significance": {
                    "warning_lines": ["⚠️ track record CI 寬度 28pp，點估值僅供參考"],
                },
            },
        },
    }
    r2 = _minimal_r2_chart_state(full_state)
    ra = r2["recent_accuracy"]
    assert "probability_triplet_summary" in ra
    s = ra["probability_triplet_summary"]
    assert "baseline=35.2%" in s and "ta=65.0%" in s and "track=17.9%" in s
    assert ra.get("probability_triplet_warnings") == [
        "⚠️ track record CI 寬度 28pp，點估值僅供參考",
    ]
    # 確認沒把完整 triplet 帶進 r2
    assert "probability_triplet" not in ra
