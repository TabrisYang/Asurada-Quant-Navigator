"""阿斯拉量化系統 — v101 模仿學習推論器（Phase 2.2）

從 chart_state + 預測 metadata 抽特徵，結合：
  - LightGBM 模型推論 P(hit_target | features) → p_ml
  - 規則融合 → p_rule（從既有訊號加權合成）
  - 動態 blend：α = sigmoid(0.05*(n-50))
  - 軟性 veto：regime conf < 0.3 或 Wilson CI < 30% → cap p_ml at 0.30
  - 分歧偵測：|p_ml - p_rule| > 0.3 → conflicts + position_multiplier=0.5

Output schema 對應你 prompt 範例的 rl_strategic_insight。
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger

from app.core.feature_extractor import FEATURE_COLUMNS, extract_features_at
from app.core.imitation_trainer import get_active_model, get_plain_model_for_shap


def _alpha(n: int) -> float:
    """動態 blend 權重 — 樣本量自動調整 ML 占比。

    n=30  → 0.27 (rule 主導)
    n=50  → 0.50 (對等)
    n=100 → 0.92 (ML 主導)
    n=200 → 0.99 (幾乎純 ML)
    """
    return 1.0 / (1.0 + math.exp(-0.05 * (n - 50)))


class ImitationPredictor:
    """單例 — 在 module level 載入模型，省每次推論的 IO。"""

    def __init__(self):
        self._loaded_version: Optional[int] = None
        self._model = None
        self._plain_model = None
        self._shap_explainer = None
        self._metrics: dict = {}
        self._reload()

    def _reload(self) -> None:
        active = get_active_model()
        if not active:
            self._loaded_version = None
            self._model = None
            return

        version = active.get("version")
        if version == self._loaded_version:
            return

        self._model = active.get("model")
        self._metrics = active.get("metrics") or {}
        self._loaded_version = version
        # ★ Lazy load SHAP — 不在 _reload 觸發，避免 native lib 同時 init 觸發 segfault
        self._plain_model = None
        self._shap_explainer = None

    def _ensure_shap(self) -> None:
        """SHAP 第一次需要時才載入（避免 module import 時 native 衝突）。"""
        if self._shap_explainer is not None or self._loaded_version is None:
            return
        try:
            self._plain_model = get_plain_model_for_shap(self._loaded_version)
            if self._plain_model is not None:
                import shap
                self._shap_explainer = shap.TreeExplainer(self._plain_model)
        except Exception as e:
            logger.debug(f"SHAP 載入失敗（不影響推論）：{e}")
            self._plain_model = None
            self._shap_explainer = None

    def predict(self, current_features: dict, chart_state: Optional[dict] = None) -> dict:
        """推論 P(hit_target) + policy + Q-value + SHAP + 衝突。"""
        # 每次推論前檢查模型有沒有更新（可能剛 retrain 過）
        try:
            self._reload()
        except Exception:
            pass

        n = self._metrics.get("trainset_n", 0)

        # ─── Layer 1：規則融合（永遠可用，cold-start fallback）───
        p_rule = self._compute_rule_score(current_features, chart_state)

        # ─── Layer 1b：LightGBM 推論（n >= 30 才嘗試）───
        p_ml: Optional[float] = None
        mode = "cold_start"
        blend_alpha = 0.0

        if self._model is not None and n >= 30:
            try:
                X = self._features_to_array(current_features)
                p_ml = float(self._model.predict_proba(X)[0, 1])
                blend_alpha = _alpha(n)
                mode = "blended" if n < 150 else "ml_dominant"
            except Exception as e:
                logger.warning(f"LightGBM 推論失敗（fallback to rule）：{e}")
                p_ml = None
                mode = "cold_start"

        # ─── Layer 2：動態 blend ───
        if p_ml is not None:
            p_blend = blend_alpha * p_ml + (1 - blend_alpha) * p_rule
        else:
            p_blend = p_rule

        # ─── Layer 3：軟性 veto（規則 hard signal cap p_ml at 0.30）───
        veto_active = False
        veto_reasons = []
        regime_conf = 0.0
        wilson_ci_lo = 1.0

        if chart_state:
            regime_info = chart_state.get("currentRegime") or {}
            regime_conf = float(regime_info.get("confidence", 0))

        wilson_ci_lo = current_features.get("wilson_ci_lower", 1.0) or 1.0

        if regime_conf < 0.3:
            veto_reasons.append(f"regime 信心 {regime_conf:.2f} < 0.3")
            veto_active = True
        if wilson_ci_lo < 0.3:
            veto_reasons.append(f"Wilson CI 下界 {wilson_ci_lo:.0%} < 30%")
            veto_active = True

        p_blend_capped = min(p_blend, 0.30) if veto_active else p_blend

        # ─── Layer 4：分歧偵測 + position multiplier ───
        conflicts = []
        position_multiplier = 1.0

        if p_ml is not None and abs(p_ml - p_rule) > 0.3:
            conflicts.append(
                f"⚠️ ML 估 {p_ml:.0%} vs 規則估 {p_rule:.0%}（差距 {abs(p_ml - p_rule):.0%}）→ 信號不一致，倉位 ×0.5"
            )
            position_multiplier = 0.5

        if veto_active:
            conflicts.append(f"⚠️ 規則 hard veto：{' + '.join(veto_reasons)} → ML 估值上限 30%")
            position_multiplier = min(position_multiplier, 0.5)

        # ─── Layer 5：policy 三元映射 ───
        direction_long = current_features.get("direction_long", 1)
        if direction_long == 1:
            p_buy = p_blend_capped
            p_sell = (1 - p_blend_capped) * 0.3
        else:
            p_sell = p_blend_capped
            p_buy = (1 - p_blend_capped) * 0.3
        p_wait = max(0.0, 1.0 - p_buy - p_sell)

        # Q-value
        target_dist = current_features.get("target_distance_pct", 0) or 0
        stop_dist = current_features.get("stop_distance_pct", 0) or 0
        q_value = p_blend_capped * target_dist - (1 - p_blend_capped) * abs(stop_dist)

        # SHAP top 3 + 路徑類比（只在 mode != cold_start 才有）
        top_features = self._top_shap(current_features) if p_ml is not None else []
        similar_paths = self._knn_similar(current_features) if p_ml is not None else []

        return {
            "policy_distribution": {
                "buy": round(p_buy, 3),
                "wait": round(p_wait, 3),
                "sell": round(p_sell, 3),
            },
            "q_value": round(q_value, 4),
            "p_hit_target": round(p_blend_capped, 3),
            "p_ml": round(p_ml, 3) if p_ml is not None else None,
            "p_rule": round(p_rule, 3),
            "blend_alpha": round(blend_alpha, 2),
            "position_multiplier": position_multiplier,
            "mode": mode,
            "model_version": self._loaded_version,
            "model_metrics": {
                "trainset_n": n,
                "auc": self._metrics.get("auc"),
                "lockbox_auc": self._metrics.get("lockbox_auc"),
                "brier": self._metrics.get("brier"),
                "trained_at": self._metrics.get("trained_at"),
            },
            "top_features": top_features,
            "similar_paths": similar_paths,
            "conflicts": conflicts,
            "veto_active": veto_active,
            "is_cold_start": mode == "cold_start",
        }

    # ─── 內部 helper ─────────────────────────────────

    def _features_to_array(self, current_features: dict) -> np.ndarray:
        """把 dict 轉 1×39 array（順序按 FEATURE_COLUMNS）。"""
        vec = []
        for col in FEATURE_COLUMNS:
            v = current_features.get(col)
            try:
                f = float(v) if v is not None else 0.0
            except (TypeError, ValueError):
                f = 0.0
            if f != f:  # NaN
                f = 0.0
            vec.append(f)
        return np.array([vec], dtype=float)

    def _compute_rule_score(self, features: dict, chart_state: Optional[dict]) -> float:
        """規則融合：用既有 v100 訊號加權合成 P(hit_target)。

        分量（每個 0-1）：
          - regime_confidence × regime 跟方向是否一致
          - Wilson CI 下界
          - SMC bias 跟方向是否一致
          - 動量訊號（RSI / MACD）跟方向是否一致
          - 波動率合理性
        """
        direction_long = features.get("direction_long", 1) == 1
        score_parts = []

        # 1. Regime alignment
        regime_conf = float(features.get("regime_confidence", 0) or 0)
        is_up_regime = features.get("regime_trending_up", 0) == 1
        is_down_regime = features.get("regime_trending_down", 0) == 1
        if direction_long and is_up_regime:
            score_parts.append(regime_conf)
        elif (not direction_long) and is_down_regime:
            score_parts.append(regime_conf)
        else:
            score_parts.append((1 - regime_conf) * 0.4)  # 中性偏低

        # 2. Wilson CI 下界（若有）
        wilson = features.get("wilson_ci_lower")
        if wilson is not None:
            try:
                score_parts.append(float(wilson))
            except Exception:
                pass

        # 3. SMC bias alignment
        smc_bias = features.get("smc_bias", 0) or 0
        if direction_long and smc_bias > 0:
            score_parts.append(0.7)
        elif (not direction_long) and smc_bias < 0:
            score_parts.append(0.7)
        elif smc_bias == 0:
            score_parts.append(0.5)
        else:
            score_parts.append(0.3)  # 反向 → 低分

        # 4. 動量訊號 alignment（RSI 35-65 = 中性、>70 + long = 過熱、<30 + long = 接刀）
        rsi = features.get("rsi_14")
        if rsi is not None:
            try:
                rsi = float(rsi)
                if direction_long:
                    if 40 <= rsi <= 65:
                        score_parts.append(0.7)
                    elif rsi < 30:
                        score_parts.append(0.55)  # 接刀略風險
                    elif rsi > 75:
                        score_parts.append(0.3)
                    else:
                        score_parts.append(0.5)
                else:
                    if 35 <= rsi <= 60:
                        score_parts.append(0.7)
                    elif rsi > 70:
                        score_parts.append(0.55)
                    elif rsi < 25:
                        score_parts.append(0.3)
                    else:
                        score_parts.append(0.5)
            except Exception:
                pass

        # 5. 歷史校準
        recent_winrate = features.get("recent_30d_winrate")
        if recent_winrate is not None:
            try:
                wr = float(recent_winrate) / 100  # 已是百分比
                score_parts.append(max(0.3, min(0.85, wr)))
            except Exception:
                pass

        return float(np.mean(score_parts)) if score_parts else 0.5

    def _top_shap(self, current_features: dict, k: int = 3) -> list[dict]:
        """SHAP top k 驅動因子。失敗回 []。"""
        # 第一次需要時才載入 SHAP（lazy）— 避免 native lib 衝突
        self._ensure_shap()
        if self._shap_explainer is None or self._plain_model is None:
            return []
        try:
            X = self._features_to_array(current_features)
            shap_values = self._shap_explainer.shap_values(X)
            # binary classifier 可能回 list of 2 arrays（對應 class 0/1）；新版回單一 array
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]
            sv = shap_values[0] if shap_values.ndim == 2 else shap_values

            pairs = sorted(
                zip(FEATURE_COLUMNS, sv, X[0]),
                key=lambda p: abs(p[1]),
                reverse=True,
            )[:k]
            return [
                {
                    "name": name,
                    "shap": round(float(v), 4),
                    "value": round(float(val), 4) if val == val else None,  # NaN check
                    "direction": "long" if v > 0 else "short",
                }
                for name, v, val in pairs
            ]
        except Exception as e:
            logger.debug(f"SHAP top_features 計算失敗：{e}")
            return []

    def _knn_similar(self, current_features: dict, k: int = 3) -> list[dict]:
        """K-NN 找歷史最相似 k 筆 verified prediction（給 prompt 路徑類比用）。

        v103 1C：用 StandardScaler 正規化所有特徵後才算 L2 距離。
        原本 RSI(0-100) 跟 atr_pct(0-0.1) 直接算距離 → 大尺度特徵主導 → similarity 全 0。
        """
        from app.core.prediction_tracker import prediction_tracker
        from sklearn.preprocessing import StandardScaler

        if not prediction_tracker._conn:
            return []

        try:
            cursor = prediction_tracker._conn.execute(
                """SELECT pf.*, p.status, p.actual_outcome_pct, p.id
                   FROM prediction_features pf
                   JOIN predictions p ON pf.prediction_id = p.id
                   WHERE p.status IN ('hit_target', 'hit_stop')
                   ORDER BY p.created_at DESC LIMIT 200"""
            )
            cols = [d[0] for d in cursor.description]
            rows = cursor.fetchall()
            if len(rows) < k:
                return []

            df = pd.DataFrame(rows, columns=cols)
            for col in FEATURE_COLUMNS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df_features = df[FEATURE_COLUMNS].fillna(0).astype(float).values

            current_vec = self._features_to_array(current_features)
            # ★ v103 1C：合併 fit_transform 確保 scaler 用一致統計
            combined = np.vstack([df_features, current_vec])
            scaler = StandardScaler()
            scaled = scaler.fit_transform(combined)
            df_scaled = scaled[:-1]
            current_scaled = scaled[-1:]

            distances = np.linalg.norm(df_scaled - current_scaled, axis=1)
            # 取最近 k 筆
            top_k_idx = distances.argsort()[:k]
            return [
                {
                    "prediction_id": int(df.iloc[i]["id"]),
                    "outcome": df.iloc[i]["status"],
                    "actual_return_pct": float(df.iloc[i].get("actual_outcome_pct") or 0),
                    "similarity": round(1.0 / (1.0 + float(distances[i])), 3),
                }
                for i in top_k_idx
            ]
        except Exception as e:
            logger.debug(f"_knn_similar 失敗：{e}")
            return []


# Module-level singleton
imitation_predictor = ImitationPredictor()
