"""阿斯拉量化系統 — v102 ML 推論主進程客戶端

主進程**永不載 lightgbm/shap**，所有推論透過 subprocess 隔離。

效能：
  - 啟動 subprocess: ~0.5-1 秒（Python 啟動 + lightgbm 載入）
  - 推論: < 100ms
  - 總每次: 0.6-1.1 秒（「全部分析」原本 8-15 分，可接受）

設計原則：
  - 任何錯誤永不冒泡（log + 回 None）
  - timeout 30 秒（防 subprocess 卡住）
  - 環境變數 OMP/MKL=1（subprocess 內也限制 thread 數）
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

_BACKEND_DIR = Path(__file__).resolve().parents[2]
_VENV_PYTHON = _BACKEND_DIR / ".venv" / "bin" / "python3"

DEFAULT_TIMEOUT = 30


def predict_via_subprocess(
    features: dict,
    chart_state: Optional[dict] = None,
    timeout_sec: int = DEFAULT_TIMEOUT,
) -> Optional[dict]:
    """主進程呼叫 — spawn subprocess 跑 lightgbm 推論。

    Args:
        features: dict — 39 個特徵（dict 形式，subprocess 內會轉 array）
        chart_state: dict | None — 完整 chart_state（給 _compute_rule_score 用）
        timeout_sec: 超時秒數，預設 30s

    Returns:
        rl_strategic_insight dict / None（失敗時）
    """
    if not _VENV_PYTHON.exists():
        logger.warning(f"找不到 .venv python: {_VENV_PYTHON}")
        return None

    # 序列化輸入（注意：chart_state 可能含複雜結構，default=str 兜底）
    try:
        # 移除可能的 _cached_df（DataFrame 不能 JSON 序列化）
        if isinstance(chart_state, dict):
            chart_state = {k: v for k, v in chart_state.items() if not k.startswith("_cached_")}
        payload_json = json.dumps(
            {"features": features, "chart_state": chart_state or {}},
            default=str,
            ensure_ascii=False,
        )
    except Exception as e:
        logger.warning(f"ml_client 序列化失敗: {e}")
        return None

    env = {
        **os.environ,
        # ★ subprocess 內也限制 thread（雙重保險）
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    try:
        result = subprocess.run(
            [str(_VENV_PYTHON), "-m", "app.core.ml_worker"],
            input=payload_json,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env,
            cwd=str(_BACKEND_DIR),
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"ml_worker timeout {timeout_sec}s")
        return None
    except Exception as e:
        logger.warning(f"ml_worker spawn error: {e}")
        return None

    if result.returncode != 0:
        logger.debug(
            f"ml_worker exit {result.returncode}: stderr={result.stderr[:300]}"
        )
        return None

    if not result.stdout.strip():
        return None

    try:
        insight = json.loads(result.stdout)
        if isinstance(insight, dict) and "error" in insight:
            logger.debug(f"ml_worker reported error: {insight['error']}")
            return None
        return insight
    except json.JSONDecodeError:
        logger.warning(f"ml_worker output not JSON: {result.stdout[:200]}")
        return None


def health_check() -> dict:
    """快速測試 subprocess 是否能正常 spawn + return（給 /health endpoint 或啟動驗證）。"""
    test_features = {"direction_long": 1, "regime_confidence": 0.5}
    test_state = {"currentRegime": {"regime": "ranging", "confidence": 0.5}}
    insight = predict_via_subprocess(test_features, test_state, timeout_sec=15)
    if insight is None:
        return {"healthy": False, "reason": "subprocess returned None"}
    return {
        "healthy": True,
        "mode": insight.get("mode"),
        "model_version": insight.get("model_version"),
    }
