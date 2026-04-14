"""阿斯拉量化系統 — GMM/HMM 市場體制分類器

使用 GMM 對對數報酬率 + 波動率進行無監督分類，
動態識別市場處於哪種 regime（強多/弱多/盤整/強空等）。
"""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.mixture import GaussianMixture
from typing import Optional


class RegimeModel:
    """GMM 市場體制分類器"""

    # Regime 標籤映射（根據 GMM 聚類中心的報酬+波動排序）
    _REGIME_LABELS = {
        (True, False): "強多（高報酬低波動）",
        (True, True): "弱多（高報酬高波動）",
        (False, False): "低波盤整",
        (False, True): "強空（負報酬高波動）",
    }

    def __init__(self):
        self._gmm: Optional[GaussianMixture] = None
        self._regime_map: dict[int, str] = {}

    def fit_predict(
        self,
        df: pd.DataFrame,
        n_regimes: int = 4,
        min_samples: int = 100,
    ) -> dict:
        """用對數報酬率 + 波動率擬合 GMM，回傳各 regime 機率。

        Args:
            df: OHLCV DataFrame
            n_regimes: 分類數（預設 4：強多/弱多/盤整/強空）
            min_samples: 最少需要的 K 線數

        Returns:
            dict: 當前 regime、各 regime 機率、歷史分類
        """
        close = df["close"].values.astype(float)
        if len(close) < min_samples:
            return {
                "status": "insufficient_data",
                "message": f"數據不足（{len(close)} 根），至少需要 {min_samples} 根",
            }

        # 計算對數報酬率和滾動波動率
        log_returns = np.diff(np.log(close))
        volatility = pd.Series(log_returns).rolling(14, min_periods=5).std().values

        # 去除 NaN
        valid_mask = ~np.isnan(volatility)
        log_ret_valid = log_returns[valid_mask]
        vol_valid = volatility[valid_mask]

        if len(log_ret_valid) < min_samples:
            return {"status": "insufficient_data", "message": "有效數據不足"}

        X = np.column_stack([log_ret_valid, vol_valid])

        # 擬合 GMM（多次初始化取最佳）
        gmm = GaussianMixture(
            n_components=n_regimes,
            covariance_type="full",
            n_init=5,
            random_state=42,
        )
        gmm.fit(X)
        self._gmm = gmm

        # 分類所有歷史數據點
        labels = gmm.predict(X)
        probs = gmm.predict_proba(X)

        # 根據聚類中心的報酬和波動率為每個 regime 命名
        self._regime_map = self._label_regimes(gmm.means_)

        # 當前 regime（最後一個數據點）
        current_label_idx = labels[-1]
        current_regime = self._regime_map.get(current_label_idx, f"regime_{current_label_idx}")
        current_probs = probs[-1]

        # 各 regime 的統計
        regime_stats = {}
        for i in range(n_regimes):
            mask = labels == i
            regime_name = self._regime_map.get(i, f"regime_{i}")
            regime_stats[regime_name] = {
                "count": int(mask.sum()),
                "pct": round(mask.sum() / len(labels) * 100, 1),
                "avg_return": round(float(np.mean(log_ret_valid[mask])) * 100, 4),
                "avg_volatility": round(float(np.mean(vol_valid[mask])) * 100, 4),
            }

        # Regime 轉移矩陣（粗略版：從前一天的 regime 到今天的 regime）
        transitions = np.zeros((n_regimes, n_regimes))
        for i in range(1, len(labels)):
            transitions[labels[i - 1], labels[i]] += 1
        # 正規化
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_matrix = (transitions / row_sums).round(3).tolist()

        logger.info(
            f"GMM Regime 分類完成: {n_regimes} regimes, "
            f"當前={current_regime} ({current_probs[current_label_idx]:.1%})"
        )

        return {
            "status": "success",
            "n_regimes": n_regimes,
            "current_regime": current_regime,
            "current_regime_idx": int(current_label_idx),
            "regime_probabilities": {
                self._regime_map.get(i, f"regime_{i}"): round(float(current_probs[i]) * 100, 1)
                for i in range(n_regimes)
            },
            "regime_stats": regime_stats,
            "transition_matrix": transition_matrix,
            "bic": round(float(gmm.bic(X)), 2),
            "total_samples": len(labels),
        }

    def get_regime_history(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """回傳每根 K 線的 regime 標籤（用於 regime 條件回測）。"""
        if self._gmm is None:
            return None

        close = df["close"].values.astype(float)
        log_returns = np.diff(np.log(close))
        volatility = pd.Series(log_returns).rolling(14, min_periods=5).std().values

        # 完整陣列（第一根填 0）
        labels = np.zeros(len(close), dtype=int)
        valid_mask = ~np.isnan(volatility)
        if valid_mask.sum() < 2:
            return labels

        X = np.column_stack([log_returns[valid_mask], volatility[valid_mask]])
        pred = self._gmm.predict(X)

        # 映射回完整索引
        valid_indices = np.where(valid_mask)[0] + 1  # +1 因為 diff 少一根
        for i, idx in enumerate(valid_indices):
            if idx < len(labels):
                labels[idx] = pred[i]

        return labels

    def _label_regimes(self, means: np.ndarray) -> dict[int, str]:
        """根據 GMM 聚類中心為每個 regime 命名。"""
        n = len(means)
        # means[:, 0] = 平均報酬率, means[:, 1] = 平均波動率
        ret_median = np.median(means[:, 0])
        vol_median = np.median(means[:, 1])

        regime_map = {}
        for i in range(n):
            is_positive = means[i, 0] > ret_median
            is_high_vol = means[i, 1] > vol_median

            label = self._REGIME_LABELS.get(
                (is_positive, is_high_vol),
                f"regime_{i}",
            )
            regime_map[i] = label

        return regime_map


    # ═══════════════════════════════════════════════════
    #  GARCH(1,1) 波動率預測
    # ═══════════════════════════════════════════════════

    def predict_volatility(self, df: pd.DataFrame, horizon: int = 5) -> dict:
        """GARCH(1,1) 預測未來波動率的擴張或收斂。

        Args:
            df: OHLCV DataFrame
            horizon: 預測未來幾根 K 線

        Returns:
            dict: 當前波動率、預測波動率、建議止損倍率
        """
        from arch import arch_model

        close = df["close"].values.astype(float)
        if len(close) < 100:
            return {"status": "insufficient_data", "message": "GARCH 需要至少 100 根 K 線"}

        returns = np.diff(np.log(close)) * 100  # 百分比對數報酬

        try:
            model = arch_model(returns, vol="Garch", p=1, q=1, dist="t", rescale=False)
            result = model.fit(disp="off", show_warning=False)

            forecast = result.forecast(horizon=horizon)
            predicted_var = forecast.variance.values[-1]
            predicted_vol = np.sqrt(predicted_var)

            current_vol = float(pd.Series(returns).rolling(14).std().iloc[-1])
            avg_predicted = float(np.mean(predicted_vol))
            vol_expanding = avg_predicted > current_vol * 1.1

            # 動態止損倍率：波動率擴張時放寬
            if vol_expanding:
                sl_mult = min(2.0, avg_predicted / current_vol) if current_vol > 0 else 1.5
            else:
                sl_mult = 1.0

            return {
                "status": "success",
                "current_volatility": round(current_vol, 4),
                "predicted_volatility": [round(float(v), 4) for v in predicted_vol],
                "avg_predicted": round(avg_predicted, 4),
                "vol_direction": "expanding" if vol_expanding else "contracting",
                "suggested_sl_multiplier": round(sl_mult, 2),
                "interpretation": (
                    f"波動率預測{'擴張' if vol_expanding else '收斂'}，"
                    f"建議止損倍率 {sl_mult:.1f}x"
                ),
            }
        except Exception as e:
            logger.warning(f"GARCH 預測失敗: {e}")
            return {"status": "error", "message": str(e)}

    # ═══════════════════════════════════════════════════
    #  HMM 狀態轉移模型
    # ═══════════════════════════════════════════════════

    def fit_hmm(
        self,
        df: pd.DataFrame,
        n_states: int = 4,
        min_samples: int = 200,
    ) -> dict:
        """Hidden Markov Model：加入狀態轉移機率，預測 regime 持續時間。

        Args:
            df: OHLCV DataFrame
            n_states: 隱含狀態數
            min_samples: 最少需要的 K 線數

        Returns:
            dict: 當前狀態、轉移矩陣、各狀態預期持續時間
        """
        from hmmlearn.hmm import GaussianHMM

        close = df["close"].values.astype(float)
        if len(close) < min_samples:
            return {"status": "insufficient_data", "message": f"HMM 需要至少 {min_samples} 根"}

        log_returns = np.diff(np.log(close))
        volatility = pd.Series(log_returns).rolling(14, min_periods=5).std().values

        valid = ~np.isnan(volatility)
        X = np.column_stack([log_returns[valid], volatility[valid]])

        if len(X) < min_samples:
            return {"status": "insufficient_data"}

        try:
            hmm = GaussianHMM(
                n_components=n_states,
                covariance_type="diag",  # diag 比 full 更穩定
                n_iter=200,
                random_state=42,
            )
            hmm.fit(X)

            states = hmm.predict(X)
            current_state = int(states[-1])

            # 轉移矩陣
            trans_mat = hmm.transmat_.round(3).tolist()

            # 各狀態預期持續時間：1 / (1 - P(stay))
            expected_duration = []
            for i in range(n_states):
                p_stay = hmm.transmat_[i, i]
                duration = 1 / (1 - p_stay) if p_stay < 1 else float("inf")
                expected_duration.append(round(duration, 1))

            # 狀態命名（根據均值）
            state_names = self._label_regimes(hmm.means_)

            return {
                "status": "success",
                "n_states": n_states,
                "current_state": state_names.get(current_state, f"state_{current_state}"),
                "current_state_idx": current_state,
                "transition_matrix": trans_mat,
                "expected_duration_bars": {
                    state_names.get(i, f"state_{i}"): expected_duration[i]
                    for i in range(n_states)
                },
                "state_names": state_names,
                "interpretation": (
                    f"當前處於「{state_names.get(current_state, '?')}」，"
                    f"預期持續 {expected_duration[current_state]:.0f} 根 K 線"
                ),
            }
        except Exception as e:
            logger.warning(f"HMM 擬合失敗: {e}")
            return {"status": "error", "message": str(e)}


# Singleton
regime_model = RegimeModel()
