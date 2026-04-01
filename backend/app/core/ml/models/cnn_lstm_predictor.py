"""阿斯拉量化系統 — CNN-LSTM 預測模型

CNN 提取局部模式 + LSTM 捕捉時序依賴，融合兩者優勢。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from loguru import logger

from app.core.ml.base import BasePredictor, TrainResult
from app.core.ml.registry import model_registry

_DEFAULT_CONFIG = {
    "cnn_filters": 32,
    "cnn_kernel_size": 3,
    "hidden_size": 64,
    "num_layers": 1,
    "dropout": 0.3,
    "sequence_length": 15,
    "batch_size": 32,
    "epochs": 60,
    "learning_rate": 0.0008,
    "patience": 10,
    "random_state": 42,
}


@model_registry.register(
    id="cnn_lstm",
    name="CNN-LSTM",
    category="神經網路",
    description="CNN 提取局部模式 + LSTM 捕捉時序依賴，融合兩者優勢，適合複雜市場結構",
    requires=["torch"],
    default_config=_DEFAULT_CONFIG,
    min_samples=600,
    supports_gpu=True,
    training_speed="slow",
)
class CNNLSTMPredictor(BasePredictor):

    def __init__(self):
        self._model = None
        self._feature_names: list[str] = []
        self._config: dict = {}
        self._importance_scores: dict[str, float] = {}

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str],
        config: dict | None = None,
    ) -> TrainResult:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
        )

        t0 = time.time()
        cfg = {**_DEFAULT_CONFIG, **(config or {})}
        self._config = cfg
        self._feature_names = list(feature_names)
        seq_len = cfg["sequence_length"]
        n_features = X_train.shape[1]

        torch.manual_seed(cfg["random_state"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_train_seq, y_train_seq = _make_sequences(X_train, y_train, seq_len)
        X_val_seq, y_val_seq = _make_sequences(X_val, y_val, seq_len)

        if len(X_train_seq) < 30 or len(X_val_seq) < 10:
            logger.warning("CNN-LSTM: 序列化後樣本不足")
            return TrainResult(n_train_samples=len(X_train_seq), n_oos_samples=len(X_val_seq))

        train_ds = TensorDataset(
            torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train_seq),
        )
        val_ds = TensorDataset(
            torch.FloatTensor(X_val_seq), torch.FloatTensor(y_val_seq),
        )
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])

        model = _CNNLSTMNet(
            n_features=n_features,
            cnn_filters=cfg["cnn_filters"],
            cnn_kernel_size=cfg["cnn_kernel_size"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        pw = cfg.get("pos_weight")
        pw_tensor = torch.tensor([pw], device=device) if pw and pw > 1.5 else None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(cfg["epochs"]):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb).squeeze(-1)
                loss = criterion(out, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb).squeeze(-1)
                    val_loss += criterion(out, yb).item() * len(yb)
            val_loss /= len(y_val_seq)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= cfg["patience"]:
                    break

        if best_state:
            model.load_state_dict(best_state)
        self._model = model.cpu()

        all_probs_val = self._predict_sequences(X_val_seq, device)
        all_probs_train = self._predict_sequences(X_train_seq, device)

        y_pred_val = (all_probs_val >= 0.5).astype(int)
        y_pred_train = (all_probs_train >= 0.5).astype(int)

        self._importance_scores = self._compute_permutation_importance(
            X_val_seq, y_val_seq, device,
        )

        elapsed_ms = int((time.time() - t0) * 1000)
        result = TrainResult(
            oos_accuracy=float(accuracy_score(y_val_seq, y_pred_val)),
            oos_precision=float(precision_score(y_val_seq, y_pred_val, zero_division=0)),
            oos_recall=float(recall_score(y_val_seq, y_pred_val, zero_division=0)),
            oos_f1=float(f1_score(y_val_seq, y_pred_val, zero_division=0)),
            oos_auc=float(roc_auc_score(y_val_seq, all_probs_val)) if len(set(y_val_seq)) > 1 else 0.5,
            train_accuracy=float(accuracy_score(y_train_seq, y_pred_train)),
            feature_importance=self._importance_scores,
            n_train_samples=len(y_train_seq),
            n_oos_samples=len(y_val_seq),
            train_time_ms=elapsed_ms,
        )

        logger.info(
            f"CNN-LSTM 訓練完成: OOS acc={result.oos_accuracy:.3f} "
            f"AUC={result.oos_auc:.3f} ({elapsed_ms}ms, {epoch+1} epochs)"
        )
        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("模型尚未訓練")
        import torch
        self._model.eval()
        if X.ndim == 2:
            X = X.reshape(1, 1, X.shape[-1])
        with torch.no_grad():
            out = self._model(torch.FloatTensor(X))
            probs = torch.sigmoid(out).squeeze(-1).numpy()
        return probs

    def get_feature_importance(self) -> dict[str, float]:
        return dict(sorted(self._importance_scores.items(), key=lambda x: x[1], reverse=True))

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("模型尚未訓練，無法儲存")
        import torch
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), path / "model.pt")
        with open(path / "meta.json", "w") as f:
            json.dump({
                "feature_names": self._feature_names,
                "config": self._config,
                "importance": self._importance_scores,
            }, f)

    def load(self, path: Path) -> None:
        import torch
        meta_path = path / "meta.json"
        model_path = path / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"找不到模型檔案: {model_path}")
        with open(meta_path) as f:
            meta = json.load(f)
        self._feature_names = meta["feature_names"]
        self._config = meta["config"]
        self._importance_scores = meta.get("importance", {})
        n_features = len(self._feature_names)
        cfg = self._config
        self._model = _CNNLSTMNet(
            n_features=n_features,
            cnn_filters=cfg["cnn_filters"],
            cnn_kernel_size=cfg["cnn_kernel_size"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        )
        self._model.load_state_dict(torch.load(model_path, weights_only=True))
        self._model.eval()

    def is_fitted(self) -> bool:
        return self._model is not None

    def _predict_sequences(self, X_seq: np.ndarray, device) -> np.ndarray:
        import torch
        self._model.eval()
        self._model.to(device)
        with torch.no_grad():
            out = self._model(torch.FloatTensor(X_seq).to(device))
            probs = torch.sigmoid(out).squeeze(-1).cpu().numpy()
        self._model.cpu()
        return probs

    def _compute_permutation_importance(
        self, X_seq: np.ndarray, y_seq: np.ndarray, device,
    ) -> dict[str, float]:
        from sklearn.metrics import accuracy_score

        base_probs = self._predict_sequences(X_seq, device)
        base_acc = accuracy_score(y_seq, (base_probs >= 0.5).astype(int))

        importance = {}
        n_features = X_seq.shape[2]
        rng = np.random.RandomState(42)

        for i in range(min(n_features, len(self._feature_names))):
            X_perm = X_seq.copy()
            perm_idx = rng.permutation(len(X_seq))
            X_perm[:, :, i] = X_seq[perm_idx, :, i]
            perm_probs = self._predict_sequences(X_perm, device)
            perm_acc = accuracy_score(y_seq, (perm_probs >= 0.5).astype(int))
            drop = max(base_acc - perm_acc, 0.0)
            importance[self._feature_names[i]] = round(drop, 4)

        total = sum(importance.values()) or 1.0
        return {k: round(v / total, 4) for k, v in importance.items()}


# ─── 輔助 ────────────────────────────

def _make_sequences(
    X: np.ndarray, y: np.ndarray, seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(X)
    if n < seq_len:
        return np.empty((0, seq_len, X.shape[1])), np.empty(0)
    seqs = np.array([X[i:i + seq_len] for i in range(n - seq_len + 1)])
    labels = y[seq_len - 1:]
    return seqs, labels


class _CNNLSTMNet:
    def __new__(cls, n_features, cnn_filters, cnn_kernel_size, hidden_size, num_layers, dropout):
        import torch.nn as nn

        class CNNLSTMModel(nn.Module):
            def __init__(self):
                super().__init__()
                padding = cnn_kernel_size // 2
                self.cnn = nn.Sequential(
                    nn.Conv1d(n_features, cnn_filters, kernel_size=cnn_kernel_size, padding=padding),
                    nn.BatchNorm1d(cnn_filters),
                    nn.ReLU(),
                    nn.Dropout(dropout * 0.5),
                )
                self.lstm = nn.LSTM(
                    input_size=cnn_filters,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True,
                )
                self.fc = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, 1),
                )

            def forward(self, x):
                # x: (batch, seq_len, features)
                # Conv1d expects (batch, features/channels, seq_len)
                cnn_in = x.permute(0, 2, 1)
                cnn_out = self.cnn(cnn_in)
                # Back to (batch, seq_len, cnn_filters)
                lstm_in = cnn_out.permute(0, 2, 1)
                lstm_out, _ = self.lstm(lstm_in)
                return self.fc(lstm_out[:, -1, :])

        return CNNLSTMModel()
