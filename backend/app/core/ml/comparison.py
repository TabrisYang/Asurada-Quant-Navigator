"""阿斯拉量化系統 — 多模型比較引擎

同時跑多個模型，比較 OOS 表現，並提供特徵重要性交叉驗證。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from app.core.ml.registry import model_registry
from app.core.ml.feature_engineer import build_features, normalize_features
from app.core.ml.base import TrainResult


def compare_models(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    model_ids: list[str] | None = None,
    feature_set: str = "standard",
    forward_period: int = 5,
    train_ratio: float = 0.75,
    min_samples: int = 300,
    target_direction: str = "up",
    target_threshold: float = 0.03,
    lookback_window: int = 7,
) -> dict:
    """多模型比較

    Args:
        df: OHLCV DataFrame
        symbol: 交易對
        timeframe: K 線級別
        model_ids: 要比較的模型 ID（None = 所有可用模型）
        feature_set: 特徵集
        forward_period: 預測期數
        train_ratio: 訓練/驗證分割比例
        min_samples: 最低樣本數
        target_direction: 預測方向 "up"/"down"
        target_threshold: 幅度門檻
        lookback_window: 回看窗口

    Returns:
        比較結果 dict
    """
    X, y, feature_names, valid_mask, label_stats = build_features(
        df,
        feature_set=feature_set,
        forward_period=forward_period,
        target_direction=target_direction,
        target_threshold=target_threshold,
        lookback_window=lookback_window,
    )

    if len(X) < min_samples:
        return {
            "status": "insufficient_data",
            "message": f"有效樣本 {len(X)} 不足（需 {min_samples}）",
            "label_stats": label_stats,
        }

    if label_stats.get("insufficient_positives"):
        return {
            "status": "insufficient_positives",
            "message": (
                f"正樣本僅 {label_stats['positive_count']} 個，不足以比較模型。"
                f"建議門檻 ≤ {label_stats['suggested_threshold']:.1%}"
            ),
            "label_stats": label_stats,
        }

    X_norm, scaler_params = normalize_features(X)
    split = int(len(X_norm) * train_ratio)
    X_train, y_train = X_norm[:split], y[:split]
    X_val, y_val = X_norm[split:], y[split:]

    pos_count = int(np.sum(y_train == 1.0))
    neg_count = len(y_train) - pos_count
    pos_weight = neg_count / max(pos_count, 1)

    if model_ids is None:
        available = model_registry.list_available()
        model_ids = [m.id for m in available]

    results: list[dict] = []
    feature_consensus: dict[str, list[float]] = {fn: [] for fn in feature_names}

    for mid in model_ids:
        defn = model_registry.get(mid)
        if not defn:
            continue

        if len(X_train) < defn.min_samples:
            results.append({
                "model_id": mid,
                "model_name": defn.name,
                "status": "skipped",
                "reason": f"樣本不足（需 {defn.min_samples}）",
            })
            continue

        predictor = model_registry.create_predictor(mid)
        if predictor is None:
            results.append({
                "model_id": mid,
                "model_name": defn.name,
                "status": "unavailable",
                "reason": f"依賴未安裝: {defn.requires}",
            })
            continue

        try:
            train_result = predictor.train(
                X_train, y_train, X_val, y_val, feature_names,
                {"pos_weight": round(pos_weight, 4)},
            )

            importance = predictor.get_feature_importance()
            for fn in feature_names:
                if fn in importance:
                    feature_consensus[fn].append(importance[fn])

            overfit_gap = train_result.train_accuracy - train_result.oos_accuracy
            n_features = len(feature_names)
            sfr = len(X_train) / max(n_features, 1)
            model_warnings = []
            if overfit_gap > 0.25:
                model_warnings.append(f"過擬合風險高（落差 {overfit_gap:.1%}）")
            elif overfit_gap > 0.15:
                model_warnings.append(f"過擬合風險中（落差 {overfit_gap:.1%}）")
            if sfr < 15:
                model_warnings.append(f"樣本/特徵比偏低（{sfr:.1f}:1）")

            results.append({
                "model_id": mid,
                "model_name": defn.name,
                "category": defn.category,
                "status": "success",
                "metrics": train_result.to_dict(),
                "training_speed": defn.training_speed,
                "diagnostics": {
                    "overfit_gap": round(overfit_gap, 4),
                    "sample_feature_ratio": round(sfr, 1),
                },
                "warnings": model_warnings,
            })

        except Exception as e:
            logger.error(f"模型 {mid} 比較時失敗: {e}")
            results.append({
                "model_id": mid,
                "model_name": defn.name,
                "status": "error",
                "reason": str(e),
            })

    # 排序：按 OOS AUC 降序
    successful = [r for r in results if r["status"] == "success"]
    successful.sort(
        key=lambda r: r["metrics"]["oos_auc"], reverse=True,
    )
    other = [r for r in results if r["status"] != "success"]

    # 特徵共識分析
    consensus = _build_feature_consensus(feature_consensus, len(successful))

    best = successful[0] if successful else None

    return {
        "status": "success",
        "symbol": symbol,
        "timeframe": timeframe,
        "feature_set": feature_set,
        "forward_period": forward_period,
        "total_samples": len(X),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "models_compared": len(model_ids),
        "models_successful": len(successful),
        "ranking": successful + other,
        "feature_consensus": consensus,
        "label_stats": label_stats,
        "recommendation": {
            "model_id": best["model_id"] if best else None,
            "model_name": best["model_name"] if best else None,
            "reason": _generate_recommendation(successful) if successful else "無可用模型",
        },
    }


def consensus_predict(
    predictions: list[dict],
    mode: str = "best",
    weights: dict[str, float] | None = None,
) -> dict:
    """多模型共識預測

    Args:
        predictions: 各模型的預測結果列表
        mode: "best" / "vote" / "weighted_avg"
        weights: 模型權重（用於 weighted_avg）

    Returns:
        共識預測結果
    """
    valid = [p for p in predictions if p.get("status") == "success"]
    if not valid:
        return {"status": "no_valid_predictions"}

    if mode == "best":
        best = max(
            valid,
            key=lambda p: p["prediction"].get("model_quality", {}).get("oos_auc", 0),
        )
        result = dict(best["prediction"])
        result["consensus_mode"] = "best"
        result["models_considered"] = len(valid)
        return {"status": "success", "prediction": result}

    probs = [p["prediction"]["probability"] for p in valid]

    if mode == "vote":
        directions = [p["prediction"]["direction"] for p in valid]
        long_count = directions.count("long")
        short_count = directions.count("short")
        if long_count > short_count:
            direction = "long"
        elif short_count > long_count:
            direction = "short"
        else:
            direction = "neutral"

        avg_prob = float(np.mean(probs))
        return {
            "status": "success",
            "prediction": {
                "probability": round(avg_prob, 4),
                "direction": direction,
                "confidence": _prob_to_confidence(avg_prob),
                "consensus_mode": "vote",
                "models_considered": len(valid),
                "vote_detail": {
                    "long": long_count,
                    "short": short_count,
                    "neutral": len(directions) - long_count - short_count,
                },
            },
        }

    if mode == "weighted_avg":
        w = weights or {}
        weighted_sum = 0.0
        total_weight = 0.0
        for p in valid:
            mid = p["prediction"].get("model_id", "")
            weight = w.get(mid, 1.0)
            weighted_sum += p["prediction"]["probability"] * weight
            total_weight += weight
        avg_prob = weighted_sum / total_weight if total_weight > 0 else 0.5
        direction = "long" if avg_prob >= 0.6 else ("short" if avg_prob <= 0.4 else "neutral")
        return {
            "status": "success",
            "prediction": {
                "probability": round(avg_prob, 4),
                "direction": direction,
                "confidence": _prob_to_confidence(avg_prob),
                "consensus_mode": "weighted_avg",
                "models_considered": len(valid),
            },
        }

    return {"status": "error", "message": f"未知的共識模式: {mode}"}


# ─── 輔助函式 ──────────────────────────────────

def _build_feature_consensus(
    feature_scores: dict[str, list[float]],
    n_models: int,
) -> list[dict]:
    """特徵共識排名"""
    if n_models == 0:
        return []

    consensus = []
    for fname, scores in feature_scores.items():
        if not scores:
            continue
        avg = float(np.mean(scores))
        consensus.append({
            "feature": fname,
            "avg_importance": round(avg, 4),
            "models_agree": len(scores),
            "agreement_ratio": round(len(scores) / n_models, 2),
            "consistency": round(1.0 - float(np.std(scores)) if len(scores) > 1 else 1.0, 3),
        })

    consensus.sort(key=lambda x: x["avg_importance"], reverse=True)

    for i, c in enumerate(consensus[:20]):
        if c["agreement_ratio"] >= 0.8:
            c["reliability"] = "★★★"
        elif c["agreement_ratio"] >= 0.5:
            c["reliability"] = "★★"
        else:
            c["reliability"] = "★"

    return consensus[:20]


def _generate_recommendation(successful: list[dict]) -> str:
    if not successful:
        return "無可用模型"
    best = successful[0]
    acc = best["metrics"]["oos_accuracy"]
    auc = best["metrics"]["oos_auc"]
    name = best["model_name"]
    parts = [f"{name} 表現最佳 (OOS AUC={auc:.3f}, Acc={acc:.1%})"]
    if len(successful) > 1:
        second = successful[1]
        gap = auc - second["metrics"]["oos_auc"]
        if gap < 0.02:
            parts.append(f"與 {second['model_name']} 差距僅 {gap:.3f}，建議搭配使用")
    return "；".join(parts)


def _prob_to_confidence(prob: float) -> str:
    diff = abs(prob - 0.5)
    if diff >= 0.2:
        return "high"
    elif diff >= 0.1:
        return "medium"
    return "low"
