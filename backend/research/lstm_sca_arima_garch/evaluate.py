"""OOS 評估指標：RMSE、MAE、方向命中率。"""

import numpy as np
import pandas as pd


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float((y_true - y_pred).abs().mean())


def directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """方向命中率：預測值跟實際值同向變化的比例。"""
    y_true_diff = y_true.diff().dropna()
    y_pred_diff = y_pred.diff().dropna()
    n = min(len(y_true_diff), len(y_pred_diff))
    if n == 0:
        return 0.0
    same_sign = ((y_true_diff.iloc[:n].values * y_pred_diff.iloc[:n].values) > 0).sum()
    return float(same_sign / n)


def evaluate_all(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """一次算所有指標。"""
    aligned = pd.concat([y_true, y_pred], axis=1).dropna()
    if len(aligned) == 0:
        return {"error": "y_true / y_pred 沒有共同有效樣本"}
    y_t, y_p = aligned.iloc[:, 0], aligned.iloc[:, 1]
    return {
        "rmse": round(rmse(y_t, y_p), 4),
        "mae": round(mae(y_t, y_p), 4),
        "directional_accuracy_pct": round(directional_accuracy(y_t, y_p) * 100, 2),
        "n_samples": len(aligned),
    }
