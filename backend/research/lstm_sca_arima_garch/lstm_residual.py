"""LSTM 學習 STL 分解後的「殘差」非線性結構。

殘差 = 原始價格 - 趨勢 - 季節性，理論上是 ARIMA/GARCH 抓不到的非線性訊號。
LSTM 強項是序列的長短期依賴，適合處理這種剩餘結構。

注意：實作 minimal version，不依賴 SCA 優化超參（先用固定值），
完整 SCA 優化版見 sca_optimizer.py + hybrid_pipeline.py。
"""

from typing import Optional

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None
    nn = None


class LSTMResidual:
    """簡單 LSTM regressor for 殘差預測。"""

    def __init__(
        self,
        seq_length: int = 10,
        hidden_size: int = 32,
        num_layers: int = 2,
        epochs: int = 50,
        learning_rate: float = 1e-3,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("torch 未安裝。執行：pip install torch")
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = learning_rate
        self.model: Optional[nn.Module] = None
        self.mean_: float = 0.0
        self.std_: float = 1.0

    class _LSTMNet(nn.Module):
        def __init__(self, hidden_size, num_layers):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1, hidden_size=hidden_size,
                num_layers=num_layers, batch_first=True,
            )
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    def _make_sequences(self, series: np.ndarray):
        X, y = [], []
        for i in range(len(series) - self.seq_length):
            X.append(series[i:i + self.seq_length])
            y.append(series[i + self.seq_length])
        X = np.array(X).reshape(-1, self.seq_length, 1)
        y = np.array(y)
        return X, y

    def fit(self, residual: pd.Series):
        s = residual.dropna().values
        # 標準化
        self.mean_ = float(s.mean())
        self.std_ = float(s.std()) or 1.0
        s_norm = (s - self.mean_) / self.std_

        X, y = self._make_sequences(s_norm)
        if len(X) == 0:
            raise ValueError(f"資料太短（{len(s)}），無法形成訓練序列")

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(-1)

        self.model = self._LSTMNet(self.hidden_size, self.num_layers)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            out = self.model(X_tensor)
            loss = loss_fn(out, y_tensor)
            loss.backward()
            optimizer.step()

    def predict(self, residual: pd.Series, n_forecast: int = 5) -> pd.Series:
        if self.model is None:
            raise RuntimeError("模型未訓練。請先呼叫 fit()")
        s = residual.dropna().values
        s_norm = (s - self.mean_) / self.std_

        # 用最後 seq_length 根遞迴預測
        seq = list(s_norm[-self.seq_length:])
        forecasts = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(n_forecast):
                x = torch.FloatTensor(seq[-self.seq_length:]).reshape(1, self.seq_length, 1)
                pred_norm = float(self.model(x).item())
                forecasts.append(pred_norm)
                seq.append(pred_norm)

        # 反標準化
        forecasts_real = [p * self.std_ + self.mean_ for p in forecasts]
        return pd.Series(forecasts_real, name="residual_forecast")


def is_available() -> bool:
    return _TORCH_AVAILABLE
