"""阿斯拉量化系統 — ML 模型版本管理器

管理模型的訓練、序列化、再訓練排程、品質監控。
提供 Walk Forward 驗證和使用者偏好設定的持久化。
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from app.core.config.settings import settings
from app.core.ml.base import BasePredictor, PredictResult, TrainResult
from app.core.ml.registry import model_registry
from app.core.ml.feature_engineer import (
    build_features, build_latest_features, normalize_features, apply_normalization,
)


# ─── 使用者可調參數的預設值和護欄 ────────────────────

DEFAULT_ML_SETTINGS = {
    "enabled": "auto",              # "auto" / "on" / "off"
    "model_id": "lightgbm",
    "feature_set": "standard",      # "compact" / "standard" / "full"
    "forward_period": 5,            # 預測期數: 1~100
    "threshold": 0.60,              # 模型信心採用門檻: 0.52~0.85
    "show_explanation": True,
    "train_window": 500,            # 訓練窗口: 200~5000
    "retrain_interval": 50,         # 再訓練頻率: 10~500
    "walk_forward": True,
    "wf_windows": 5,                # WF 窗口數: 2~15
    "min_samples": 500,             # 最低樣本量: 100~5000
    "consensus_mode": "best",       # "best" / "vote" / "weighted_avg"
    # ── 新增：前兆模式辨識參數 ──
    "target_direction": "up",       # "up" / "down"
    "target_threshold": 0.03,       # 幅度門檻 0.5%~20%
    "lookback_window": 7,           # 回看窗口 3~30
}

_SETTINGS_LIMITS = {
    "forward_period": (1, 100),
    "threshold": (0.52, 0.85),
    "train_window": (200, 5000),
    "retrain_interval": (10, 500),
    "wf_windows": (2, 15),
    "min_samples": (100, 5000),
    "target_threshold": (0.005, 0.20),
    "lookback_window": (3, 30),
}


class ModelManager:
    """ML 模型管理中心"""

    def __init__(self):
        self._db_path = settings.db_path / "ml.db"
        self._models_dir = settings.db_path / "ml_models"
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        try:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=3000")

            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS ml_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    feature_set TEXT NOT NULL,
                    forward_period INTEGER NOT NULL,
                    train_window INTEGER NOT NULL,
                    oos_accuracy REAL,
                    oos_auc REAL,
                    oos_f1 REAL,
                    train_accuracy REAL,
                    n_train_samples INTEGER,
                    n_oos_samples INTEGER,
                    train_time_ms INTEGER,
                    feature_importance TEXT,
                    scaler_params TEXT,
                    is_reliable INTEGER DEFAULT 0,
                    data_range TEXT,
                    model_path TEXT,
                    created_at TEXT NOT NULL,
                    bars_since_train INTEGER DEFAULT 0
                )
            """)

            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS ml_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS ml_predictions_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    probability REAL,
                    direction TEXT,
                    confidence TEXT,
                    top_features TEXT,
                    predicted_at TEXT NOT NULL,
                    actual_outcome REAL,
                    validated_at TEXT
                )
            """)
            self._conn.commit()
        except Exception as e:
            logger.error(f"ML DB 初始化失敗: {e}")

    # ─── 設定管理 ──────────────────────────────

    def get_settings(self) -> dict:
        """取得使用者 ML 設定"""
        result = dict(DEFAULT_ML_SETTINGS)
        try:
            rows = self._conn.execute("SELECT key, value FROM ml_settings").fetchall()
            for row in rows:
                key = row["key"]
                val = row["value"]
                if key in result:
                    orig_type = type(DEFAULT_ML_SETTINGS[key])
                    if orig_type is bool:
                        result[key] = val.lower() in ("true", "1", "yes")
                    elif orig_type is int:
                        result[key] = int(float(val))
                    elif orig_type is float:
                        result[key] = float(val)
                    else:
                        result[key] = val
        except Exception as e:
            logger.debug(f"讀取 ML 設定失敗: {e}")
        return result

    def update_settings(self, updates: dict) -> dict:
        """更新使用者 ML 設定（含護欄驗證）"""
        now = datetime.now().isoformat()
        warnings = []

        for key, val in updates.items():
            if key not in DEFAULT_ML_SETTINGS:
                continue

            if key in _SETTINGS_LIMITS:
                lo, hi = _SETTINGS_LIMITS[key]
                if isinstance(val, (int, float)):
                    if val < lo:
                        val = lo
                        warnings.append(f"{key} 已調整至最小值 {lo}")
                    elif val > hi:
                        val = hi
                        warnings.append(f"{key} 已調整至最大值 {hi}")

            self._conn.execute(
                "INSERT OR REPLACE INTO ml_settings (key, value, updated_at) VALUES (?, ?, ?)",
                (key, str(val), now),
            )

        self._conn.commit()

        current = self.get_settings()
        if warnings:
            current["_warnings"] = warnings
        return current

    # ─── 訓練 & 儲存 ──────────────────────────

    def train_model(
        self,
        df,
        symbol: str,
        timeframe: str,
        model_id: str | None = None,
        config_override: dict | None = None,
    ) -> dict:
        """訓練單一模型

        Args:
            df: OHLCV DataFrame
            symbol: 交易對
            timeframe: K 線級別
            model_id: 模型 ID（None 則用設定中的預設模型）
            config_override: 覆蓋模型超參數

        Returns:
            訓練結果 dict（含 status, train_result, model_quality 等）
        """
        user_settings = self.get_settings()
        mid = model_id or user_settings["model_id"]
        feature_set = user_settings["feature_set"]
        forward_period = user_settings["forward_period"]
        train_window = user_settings["train_window"]
        use_wf = user_settings["walk_forward"]
        wf_windows = user_settings["wf_windows"]
        min_samples = user_settings["min_samples"]
        target_direction = user_settings["target_direction"]
        target_threshold = user_settings["target_threshold"]
        lookback_window = user_settings["lookback_window"]

        # ── 冷卻期檢查（同模型 + 同交易對 10 分鐘內不重複訓練）──
        COOLDOWN_MINUTES = 10
        recent = self._conn.execute(
            """SELECT created_at FROM ml_models
               WHERE model_id = ? AND symbol = ? AND timeframe = ?
               ORDER BY id DESC LIMIT 1""",
            (mid, symbol, timeframe),
        ).fetchone()
        if recent:
            last_train = datetime.fromisoformat(recent["created_at"])
            if datetime.now() - last_train < timedelta(minutes=COOLDOWN_MINUTES):
                remaining = COOLDOWN_MINUTES - int(
                    (datetime.now() - last_train).total_seconds() / 60
                )
                return {
                    "status": "cooldown",
                    "message": (
                        f"此模型於 {last_train.strftime('%H:%M')} 剛完成訓練，"
                        f"請等待約 {max(remaining, 1)} 分鐘後再試。"
                        f"頻繁重訓會增加過擬合風險。"
                    ),
                }

        defn = model_registry.get(mid)
        if not defn:
            return {"status": "error", "message": f"未知模型: {mid}"}

        # 特徵工程（含窗口特徵 + 幅度門檻標籤）
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
                "message": f"有效樣本 {len(X)} 不足（需至少 {min_samples}）",
                "available_samples": len(X),
                "required_samples": min_samples,
                "label_stats": label_stats,
            }

        # ── 正樣本不足檢查 ──
        if label_stats.get("insufficient_positives"):
            return {
                "status": "insufficient_positives",
                "message": (
                    f"正樣本僅 {label_stats['positive_count']} 個（佔 {label_stats['positive_ratio']:.1%}），"
                    f"不足以訓練可靠模型。建議降低幅度門檻"
                    f"（目前 {target_threshold:.1%}，建議 ≤ {label_stats['suggested_threshold']:.1%}）"
                    f"或增加歷史數據。"
                ),
                "label_stats": label_stats,
            }

        # 限制訓練窗口
        if len(X) > train_window:
            X = X[-train_window:]
            y = y[-train_window:]

        # 標準化
        X_norm, scaler_params = normalize_features(X)

        # 計算正/負樣本權重，自動傳給模型處理類別不平衡
        pos_count = int(np.sum(y == 1.0))
        neg_count = len(y) - pos_count
        pos_weight = neg_count / max(pos_count, 1)
        imbalance_config = {"pos_weight": round(pos_weight, 4)}
        merged_config = {**(config_override or {}), **imbalance_config}

        if use_wf:
            result = self._walk_forward_train(
                mid, X_norm, y, feature_names, wf_windows, merged_config,
            )
        else:
            split = int(len(X_norm) * 0.8)
            predictor = model_registry.create_predictor(mid)
            if predictor is None:
                return {"status": "error", "message": f"模型 {mid} 依賴未安裝"}
            result = predictor.train(
                X_norm[:split], y[:split],
                X_norm[split:], y[split:],
                feature_names, merged_config,
            )

        # 儲存模型
        model_dir = self._models_dir / f"{symbol.replace('/', '_')}_{timeframe}_{mid}"
        if model_dir.exists():
            shutil.rmtree(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        if use_wf and result.extra.get("best_predictor"):
            best: BasePredictor = result.extra["best_predictor"]
            best.save(model_dir)
        elif not use_wf and predictor is not None:
            predictor.save(model_dir)

        # 儲存 scaler
        with open(model_dir / "scaler.json", "w") as f:
            json.dump(scaler_params, f)

        # 記錄到 DB
        ts_col = df["timestamp"] if "timestamp" in df.columns else None
        data_range = ""
        if ts_col is not None and len(ts_col) > 0:
            data_range = f"{ts_col.iloc[0]} ~ {ts_col.iloc[-1]}"

        self._conn.execute(
            """INSERT INTO ml_models
               (model_id, symbol, timeframe, feature_set, forward_period,
                train_window, oos_accuracy, oos_auc, oos_f1, train_accuracy,
                n_train_samples, n_oos_samples, train_time_ms,
                feature_importance, scaler_params, is_reliable,
                data_range, model_path, created_at, bars_since_train)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)""",
            (
                mid, symbol, timeframe, feature_set, forward_period,
                len(X), result.oos_accuracy, result.oos_auc, result.oos_f1,
                result.train_accuracy, result.n_train_samples, result.n_oos_samples,
                result.train_time_ms,
                json.dumps(result.feature_importance),
                json.dumps(scaler_params),
                1 if result.is_reliable else 0,
                data_range, str(model_dir),
                datetime.now().isoformat(),
            ),
        )
        self._conn.commit()

        # ── 過擬合偵測 & 警告 ──
        warnings = []

        # 1. Train/OOS 準確率落差
        overfit_gap = result.train_accuracy - result.oos_accuracy
        if overfit_gap > 0.25:
            warnings.append(
                f"過擬合風險高：訓練準確率 {result.train_accuracy:.1%} vs "
                f"測試準確率 {result.oos_accuracy:.1%}（落差 {overfit_gap:.1%}）。"
                f"建議增加訓練窗口或降低特徵集規模。"
            )
        elif overfit_gap > 0.15:
            warnings.append(
                f"過擬合風險中：訓練準確率 {result.train_accuracy:.1%} vs "
                f"測試準確率 {result.oos_accuracy:.1%}（落差 {overfit_gap:.1%}）。"
            )

        # 2. 樣本/特徵比率
        n_features = len(feature_names)
        sample_feature_ratio = len(X) / max(n_features, 1)
        if sample_feature_ratio < 15:
            warnings.append(
                f"特徵過多風險：{len(X)} 樣本 / {n_features} 特徵 = "
                f"{sample_feature_ratio:.1f}:1（建議 > 20:1）。"
                f"建議切換至精簡特徵集。"
            )

        # 3. Walk Forward 穩定性
        wf_accs = result.extra.get("per_window_accuracy", [])
        if len(wf_accs) >= 3:
            acc_std = float(np.std(wf_accs))
            if acc_std > 0.08:
                acc_str = ", ".join(f"{a:.1%}" for a in wf_accs)
                warnings.append(
                    f"模型不穩定：各 WF 窗口準確率波動大（{acc_str}，"
                    f"標準差 {acc_std:.1%}）。預測可靠度存疑。"
                )

        # 4. 類別不平衡警告
        if label_stats.get("imbalance_warning"):
            warnings.append(
                f"類別不平衡：正樣本比例 {label_stats['positive_ratio']:.1%}。"
                f"模型已自動加權補償（pos_weight={pos_weight:.1f}），但結果可能偏差。"
            )

        return {
            "status": "success",
            "model_id": mid,
            "symbol": symbol,
            "timeframe": timeframe,
            "train_result": result.to_dict(),
            "model_path": str(model_dir),
            "warnings": warnings,
            "label_stats": label_stats,
            "diagnostics": {
                "overfit_gap": round(overfit_gap, 4),
                "sample_feature_ratio": round(sample_feature_ratio, 1),
                "wf_stability_std": round(float(np.std(wf_accs)), 4) if wf_accs else None,
                "pos_weight": round(pos_weight, 4),
            },
        }

    def predict(
        self,
        df,
        symbol: str,
        timeframe: str,
        model_id: str | None = None,
    ) -> dict:
        """用已訓練的模型預測最新數據"""
        user_settings = self.get_settings()
        mid = model_id or user_settings["model_id"]
        threshold = user_settings["threshold"]
        feature_set = user_settings["feature_set"]
        lookback_window = user_settings["lookback_window"]

        model_dir = self._models_dir / f"{symbol.replace('/', '_')}_{timeframe}_{mid}"
        scaler_path = model_dir / "scaler.json"

        if not model_dir.exists():
            return {"status": "no_model", "message": f"尚未訓練 {mid} 模型"}

        predictor = model_registry.create_predictor(mid)
        if predictor is None:
            return {"status": "error", "message": f"模型 {mid} 依賴未安裝"}

        try:
            predictor.load(model_dir)
        except Exception as e:
            return {"status": "error", "message": f"載入模型失敗: {e}"}

        X_latest, feature_names = build_latest_features(
            df, feature_set=feature_set, lookback_window=lookback_window,
        )
        if len(X_latest) == 0:
            return {"status": "error", "message": "無法計算最新特徵"}

        if scaler_path.exists():
            with open(scaler_path) as f:
                scaler_params = json.load(f)
            X_latest = apply_normalization(X_latest, scaler_params)

        result = predictor.predict_single(X_latest, feature_names, threshold)
        result.model_id = mid

        model_info = self.get_model_status(symbol, timeframe, mid)
        if model_info:
            result.model_quality = {
                "oos_accuracy": model_info.get("oos_accuracy"),
                "oos_auc": model_info.get("oos_auc"),
                "is_reliable": bool(model_info.get("is_reliable")),
                "trained_at": model_info.get("created_at"),
            }

        # 記錄預測
        self._conn.execute(
            """INSERT INTO ml_predictions_log
               (symbol, timeframe, model_id, probability, direction,
                confidence, top_features, predicted_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                symbol, timeframe, mid, result.probability,
                result.direction, result.confidence,
                json.dumps(result.top_features),
                datetime.now().isoformat(),
            ),
        )
        self._conn.commit()

        return {"status": "success", "prediction": result.to_dict()}

    def predict_consensus(
        self,
        df,
        symbol: str,
        timeframe: str,
    ) -> dict:
        """根據 consensus_mode 設定執行多模型共識預測

        - best: 從該 symbol/timeframe 所有已訓練模型中選 OOS AUC 最高的
        - vote / weighted_avg: 載入所有已訓練模型各自預測，再彙整
        """
        user_settings = self.get_settings()
        mode = user_settings["consensus_mode"]

        trained = self._get_distinct_trained_model_ids(symbol, timeframe)
        if not trained:
            return {"status": "no_model", "message": "尚無已訓練模型"}

        if mode == "best" or len(trained) == 1:
            best_mid = trained[0]
            result = self.predict(df, symbol, timeframe, model_id=best_mid)
            if result.get("status") == "success":
                result["prediction"]["consensus_mode"] = "best"
                result["prediction"]["models_considered"] = len(trained)
            return result

        individual_results = []
        for mid in trained:
            r = self.predict(df, symbol, timeframe, model_id=mid)
            individual_results.append(r)

        from app.core.ml.comparison import consensus_predict
        weights = {}
        for mid in trained:
            info = self.get_model_status(symbol, timeframe, mid)
            if info:
                weights[mid] = info.get("oos_auc", 0.5)

        consensus = consensus_predict(individual_results, mode=mode, weights=weights)
        if consensus.get("status") == "success" and consensus.get("prediction"):
            consensus["prediction"]["individual_models"] = [
                {
                    "model_id": r.get("prediction", {}).get("model_id", ""),
                    "probability": r.get("prediction", {}).get("probability"),
                    "direction": r.get("prediction", {}).get("direction"),
                    "confidence": r.get("prediction", {}).get("confidence"),
                }
                for r in individual_results
                if r.get("status") == "success"
            ]
        return consensus

    def _get_distinct_trained_model_ids(
        self, symbol: str, timeframe: str,
    ) -> list[str]:
        """取得該 symbol/timeframe 所有不重複的已訓練 model_id（按 OOS AUC 降序）"""
        rows = self._conn.execute(
            """SELECT model_id, MAX(oos_auc) as best_auc
               FROM ml_models
               WHERE symbol=? AND timeframe=?
               GROUP BY model_id
               ORDER BY best_auc DESC""",
            (symbol, timeframe),
        ).fetchall()
        return [r["model_id"] for r in rows]

    def should_retrain(self, symbol: str, timeframe: str, model_id: str) -> bool:
        """判斷是否需要再訓練"""
        user_settings = self.get_settings()
        interval = user_settings["retrain_interval"]

        row = self._conn.execute(
            """SELECT bars_since_train FROM ml_models
               WHERE symbol=? AND timeframe=? AND model_id=?
               ORDER BY created_at DESC LIMIT 1""",
            (symbol, timeframe, model_id),
        ).fetchone()

        if not row:
            return True
        return row["bars_since_train"] >= interval

    def increment_bars(self, symbol: str, timeframe: str):
        """新增 K 線時遞增計數器"""
        self._conn.execute(
            """UPDATE ml_models SET bars_since_train = bars_since_train + 1
               WHERE symbol=? AND timeframe=?""",
            (symbol, timeframe),
        )
        self._conn.commit()

    def get_model_status(
        self, symbol: str, timeframe: str, model_id: str | None = None,
    ) -> Optional[dict]:
        """查詢模型最新狀態"""
        if model_id:
            row = self._conn.execute(
                """SELECT * FROM ml_models
                   WHERE symbol=? AND timeframe=? AND model_id=?
                   ORDER BY created_at DESC LIMIT 1""",
                (symbol, timeframe, model_id),
            ).fetchone()
        else:
            row = self._conn.execute(
                """SELECT * FROM ml_models
                   WHERE symbol=? AND timeframe=?
                   ORDER BY created_at DESC LIMIT 1""",
                (symbol, timeframe),
            ).fetchone()
        return dict(row) if row else None

    def list_trained_models(self, symbol: str | None = None) -> list[dict]:
        """列出所有已訓練的模型"""
        cols = """id, model_id, symbol, timeframe, feature_set, forward_period,
                  oos_accuracy, oos_auc, oos_f1, train_accuracy,
                  n_train_samples, n_oos_samples, train_time_ms,
                  is_reliable, created_at, bars_since_train"""
        if symbol:
            rows = self._conn.execute(
                f"SELECT {cols} FROM ml_models WHERE symbol=? ORDER BY created_at DESC",
                (symbol,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"SELECT {cols} FROM ml_models ORDER BY created_at DESC",
            ).fetchall()
        return [dict(r) for r in rows]

    def should_enable_ml(self, symbol: str, timeframe: str) -> tuple[bool, str]:
        """自動模式判斷：是否應該啟用 ML 信號"""
        user_settings = self.get_settings()
        mode = user_settings["enabled"]

        if mode == "off":
            return False, "使用者已關閉 ML"
        if mode == "on":
            return True, "使用者強制啟用 ML"

        # auto 模式
        status = self.get_model_status(symbol, timeframe)
        if not status:
            return False, "尚未訓練模型"
        if not status.get("is_reliable"):
            return False, f"模型品質不足 (OOS={status.get('oos_accuracy', 0):.1%})"
        return True, "模型品質足夠"

    # ─── Walk Forward 訓練 ──────────────────

    def _walk_forward_train(
        self,
        model_id: str,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        n_windows: int,
        config: dict | None,
    ) -> TrainResult:
        """Walk Forward 交叉驗證訓練"""
        n = len(X)
        window_size = n // (n_windows + 1)

        all_results: list[TrainResult] = []
        best_result: Optional[TrainResult] = None
        best_predictor: Optional[BasePredictor] = None

        for i in range(n_windows):
            train_end = window_size * (i + 1)
            val_end = min(train_end + window_size, n)

            if val_end - train_end < 20:
                continue

            predictor = model_registry.create_predictor(model_id)
            if predictor is None:
                break

            result = predictor.train(
                X[:train_end], y[:train_end],
                X[train_end:val_end], y[train_end:val_end],
                feature_names, config,
            )
            all_results.append(result)

            if best_result is None or result.oos_auc > best_result.oos_auc:
                best_result = result
                best_predictor = predictor

        if not all_results:
            return TrainResult()

        avg_result = TrainResult(
            oos_accuracy=float(np.mean([r.oos_accuracy for r in all_results])),
            oos_precision=float(np.mean([r.oos_precision for r in all_results])),
            oos_recall=float(np.mean([r.oos_recall for r in all_results])),
            oos_f1=float(np.mean([r.oos_f1 for r in all_results])),
            oos_auc=float(np.mean([r.oos_auc for r in all_results])),
            train_accuracy=float(np.mean([r.train_accuracy for r in all_results])),
            feature_importance=best_result.feature_importance if best_result else {},
            n_train_samples=sum(r.n_train_samples for r in all_results),
            n_oos_samples=sum(r.n_oos_samples for r in all_results),
            train_time_ms=sum(r.train_time_ms for r in all_results),
            extra={
                "wf_windows": len(all_results),
                "per_window_accuracy": [r.oos_accuracy for r in all_results],
                "per_window_auc": [r.oos_auc for r in all_results],
                "accuracy_std": float(np.std([r.oos_accuracy for r in all_results])),
                "best_predictor": best_predictor,
            },
        )

        return avg_result

    # ─── 模型監控 ──────────────────

    def get_performance_history(
        self,
        symbol: str | None = None,
        model_id: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """取得模型各版本的 OOS 效能指標趨勢"""
        self._ensure_db()
        query = """SELECT id, model_id, symbol, timeframe,
                          oos_accuracy, oos_auc, oos_f1, train_accuracy,
                          n_oos_samples, is_reliable, created_at
                   FROM ml_models WHERE 1=1"""
        params: list = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)
        query += " ORDER BY created_at ASC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def check_model_health(
        self,
        symbol: str,
        timeframe: str,
        threshold: float = 0.55,
    ) -> dict:
        """檢查模型健康度，低於閾值標記需要再訓練"""
        self._ensure_db()
        latest = self.get_model_status(symbol, timeframe)
        if not latest:
            return {"status": "no_model", "needs_retrain": False}

        oos_acc = latest.get("oos_accuracy") or 0
        is_reliable = latest.get("is_reliable", 0)
        needs_retrain = oos_acc < threshold or not is_reliable

        return {
            "status": "degraded" if needs_retrain else "healthy",
            "oos_accuracy": round(oos_acc, 4),
            "threshold": threshold,
            "needs_retrain": needs_retrain,
            "is_reliable": bool(is_reliable),
            "model_id": latest.get("model_id", ""),
            "created_at": latest.get("created_at", ""),
        }


# 全域單例
model_manager = ModelManager()
