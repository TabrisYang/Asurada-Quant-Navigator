"""阿斯拉量化系統 — 全量歷史特徵分析引擎

掃描所有幣種的全部歷史 K 線，為每個特徵計算：
- IC (Information Coefficient)：Spearman 相關性 vs 未來波動
- 最佳門檻：使 lift（條件機率/基線機率）最大化的閾值
- 數據驅動權重：基於 IC × 跨幣種一致性

三層架構中的底層：
  Feature Profiler（全量歷史）→ Scanner Calibrator（預警回饋）→ 硬編碼（fallback）
"""

import json
import sqlite3
import threading
from typing import Optional

import numpy as np
from loguru import logger
from scipy.stats import spearmanr

from app.core.config.settings import settings
from app.utils.timezone import taipei_now


# ─── 常量 ──────────────────────────────

_LOOKBACK = 6           # 前兆窗口
_FORWARD = 6            # 未來觀察窗口
_WARMUP = 50            # 指標預熱
_STEP = _LOOKBACK       # 不重疊
_MIN_SAMPLES_PER_THRESHOLD = 30   # 門檻搜索最低樣本
_IC_SIGNIFICANT = 0.03            # |IC| > 此值才算顯著
_IC_STRONG = 0.06                 # |IC| > 此值給全額權重
_MIN_SYMBOLS_CONSISTENT = 3      # 至少這麼多幣種方向一致
_WEIGHT_SCALE = 20.0              # IC→權重的縮放因子
_WEIGHT_MAX = 5.0
_PROFILE_EXPIRY_DAYS = 7          # 超過 N 天視為過期

# 要分析的特徵名（全部 _compute_features 產出的數值特徵）
_SCORING_FEATURES = [
    "vol_range_desync", "direction_consistency", "price_efficiency",
    "ma5_cross_count", "vol_momentum_pressure", "max_streak",
    "vol_concentration", "accumulation_signal", "vol_acceleration",
    "body_range_ratio", "atr_relative", "vol_ratio_20ma",
    "range_contraction", "max_vol_spike", "long_wick_count",
    "bullish_ratio", "vol_price_divergence",
]

# 方向性特徵（不計入評分權重，但仍分析 IC 供參考）
_DIRECTIONAL_FEATURES = [
    "rsi", "bb_position", "bb_width", "adx", "di_spread",
    "price_vs_ma20", "pct_6bar",
]

_ALL_FEATURES = _SCORING_FEATURES + _DIRECTIONAL_FEATURES


class ScannerFeatureProfiler:
    """全量歷史特徵分析引擎。"""

    def __init__(self):
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._profiles_cache: Optional[dict] = None

    def _ensure_db(self):
        if self._conn:
            return
        db_file = settings.db_path / "predictions.db"
        from app.core.db_utils import open_sqlite
        self._conn = open_sqlite(db_file, check_same_thread=False, row_factory=True)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS scanner_feature_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_name TEXT NOT NULL UNIQUE,
                mean REAL, std REAL,
                p10 REAL, p25 REAL, p50 REAL, p75 REAL, p90 REAL,
                ic REAL NOT NULL,
                ic_pvalue REAL,
                ic_per_symbol TEXT,
                ic_consistency REAL,
                symbols_significant INTEGER,
                optimal_threshold REAL,
                threshold_direction TEXT,
                conditional_prob REAL,
                baseline_prob REAL,
                lift REAL,
                weight REAL NOT NULL,
                total_samples INTEGER,
                symbols_used INTEGER,
                computed_at TEXT NOT NULL
            )
        """)
        # meta 行存全局統計
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS scanner_profile_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        self._conn.commit()

    # ─── 核心計算 ──────────────────────────────

    def compute_all_profiles(
        self,
        symbols: list[str] | None = None,
        timeframe: str = "4h",
        move_threshold: float = 3.0,
    ) -> dict:
        """掃描全量歷史數據，計算每個特徵的統計檔案。"""
        from app.core.auto_scanner import _precompute_indicators, _compute_features
        from app.data.fetchers.crypto_engine import crypto_engine

        if symbols is None:
            from app.api.routes.config import load_system_settings
            cfg = load_system_settings()
            symbols = cfg.get("scan_symbols", [
                "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT",
                "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", "MATIC/USDT",
            ])

        logger.info(f"[特徵分析] 開始全量歷史分析：{len(symbols)} 幣種, {timeframe}, 門檻={move_threshold}%")

        # 步驟 1：建構特徵矩陣
        all_rows: list[dict] = []  # {symbol, features, max_abs_move, hit}
        symbols_used = []

        for symbol in symbols:
            try:
                df = crypto_engine.load_local_data(symbol, timeframe)
                if df.empty or len(df) < _WARMUP + _LOOKBACK + _FORWARD + 10:
                    logger.debug(f"[特徵分析] {symbol} 數據不足，跳過")
                    continue

                df = df.reset_index(drop=True)
                ind = _precompute_indicators(df)
                closes = ind["closes"]
                n = len(df)

                symbol_count = 0
                for i in range(_WARMUP, n - _LOOKBACK - _FORWARD, _STEP):
                    idx = list(range(i, i + _LOOKBACK))
                    features = _compute_features(
                        closes, ind["opens"], ind["highs"], ind["lows"], ind["volumes"],
                        ind["ma5"], ind["ma20"], ind["vol_ma20"],
                        ind["rsi14"], ind["bb_pos"], ind["bb_width"],
                        ind["atr14"], ind["adx"], ind["plus_di"], ind["minus_di"],
                        idx,
                    )

                    # 未來結果：next 1~6 bars 最大波動
                    entry_price = closes[i + _LOOKBACK - 1]
                    if entry_price <= 0:
                        continue
                    forward_start = i + _LOOKBACK
                    forward_end = min(forward_start + _FORWARD, n)
                    forward_prices = closes[forward_start:forward_end]

                    if len(forward_prices) < 3:
                        continue

                    max_up = float(np.max((forward_prices - entry_price) / entry_price * 100))
                    max_down = float(np.min((forward_prices - entry_price) / entry_price * 100))
                    max_abs_move = max(max_up, abs(max_down))
                    hit = 1 if max_abs_move >= move_threshold else 0

                    all_rows.append({
                        "symbol": symbol,
                        "features": features,
                        "max_abs_move": max_abs_move,
                        "hit": hit,
                    })
                    symbol_count += 1

                if symbol_count > 0:
                    symbols_used.append(symbol)
                    logger.debug(f"[特徵分析] {symbol}: {symbol_count} 窗口")

            except Exception as e:
                logger.warning(f"[特徵分析] {symbol} 失敗: {e}")

        total_samples = len(all_rows)
        if total_samples < 100:
            logger.warning(f"[特徵分析] 樣本不足（{total_samples}），跳過")
            return {"status": "insufficient_data", "total_samples": total_samples}

        total_hits = sum(r["hit"] for r in all_rows)
        baseline_prob = total_hits / total_samples

        logger.info(
            f"[特徵分析] 特徵矩陣：{total_samples} 窗口, "
            f"{len(symbols_used)} 幣種, 基線命中率={baseline_prob*100:.1f}%"
        )

        # 步驟 2-4：Per-feature 分析
        profiles: list[dict] = []

        for feat_name in _ALL_FEATURES:
            profile = self._analyze_feature(
                feat_name, all_rows, symbols_used, baseline_prob, move_threshold,
            )
            if profile:
                profiles.append(profile)

        # 步驟 5：校準觸發門檻和信心分界
        meta = self._compute_score_calibration(
            profiles, all_rows, baseline_prob, total_samples, len(symbols_used),
        )

        # 存儲
        self._store_profiles(profiles, meta)
        self._profiles_cache = None  # 清除快取

        adjusted_count = sum(1 for p in profiles if p["weight"] > 0.1)
        logger.info(
            f"[特徵分析] 完成：{len(profiles)} 特徵已分析, "
            f"{adjusted_count} 個有效權重, "
            f"建議門檻={meta.get('score_threshold', 'N/A')}"
        )

        return {
            "status": "computed",
            "total_samples": total_samples,
            "symbols_used": len(symbols_used),
            "baseline_prob": round(baseline_prob * 100, 1),
            "features_analyzed": len(profiles),
            "features_with_weight": adjusted_count,
            "meta": meta,
        }

    def _analyze_feature(
        self,
        feat_name: str,
        all_rows: list[dict],
        symbols_used: list[str],
        baseline_prob: float,
        move_threshold: float,
    ) -> Optional[dict]:
        """分析單一特徵的預測力。"""
        # 提取數值
        values = []
        outcomes = []
        hits = []
        sym_values: dict[str, list] = {s: [] for s in symbols_used}
        sym_outcomes: dict[str, list] = {s: [] for s in symbols_used}

        for row in all_rows:
            v = row["features"].get(feat_name)
            if v is None or not np.isfinite(v):
                continue
            values.append(v)
            outcomes.append(row["max_abs_move"])
            hits.append(row["hit"])
            s = row["symbol"]
            if s in sym_values:
                sym_values[s].append(v)
                sym_outcomes[s].append(row["max_abs_move"])

        if len(values) < 100:
            return None

        values_arr = np.array(values)
        outcomes_arr = np.array(outcomes)
        hits_arr = np.array(hits)

        # 分佈統計
        mean_val = float(np.mean(values_arr))
        std_val = float(np.std(values_arr))
        percentiles = {
            f"p{p}": float(np.percentile(values_arr, p))
            for p in [10, 25, 50, 75, 90]
        }

        # 全局 IC
        try:
            ic, ic_pval = spearmanr(values_arr, outcomes_arr)
            if not np.isfinite(ic):
                ic, ic_pval = 0.0, 1.0
        except Exception:
            ic, ic_pval = 0.0, 1.0

        # Per-symbol IC
        ic_per_symbol: dict[str, float] = {}
        for sym in symbols_used:
            sv = sym_values[sym]
            so = sym_outcomes[sym]
            if len(sv) < 30:
                continue
            try:
                s_ic, _ = spearmanr(sv, so)
                if np.isfinite(s_ic):
                    ic_per_symbol[sym] = round(s_ic, 4)
            except Exception:
                pass

        # IC 一致性
        if ic_per_symbol:
            same_sign = sum(
                1 for s_ic in ic_per_symbol.values()
                if (s_ic > 0) == (ic > 0)
            )
            ic_consistency = same_sign / len(ic_per_symbol)
            symbols_significant = sum(
                1 for s_ic in ic_per_symbol.values()
                if abs(s_ic) > _IC_SIGNIFICANT
            )
        else:
            ic_consistency = 0.0
            symbols_significant = 0

        # 最佳門檻搜索
        threshold_direction = "above" if ic > 0 else "below"
        best_lift = 0.0
        best_threshold = percentiles["p50"]
        best_cond_prob = baseline_prob

        for pct in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
            threshold_val = float(np.percentile(values_arr, pct))
            if threshold_direction == "above":
                mask = values_arr > threshold_val
            else:
                mask = values_arr < threshold_val

            count_in = int(np.sum(mask))
            if count_in < _MIN_SAMPLES_PER_THRESHOLD:
                continue

            cond_prob = float(np.mean(hits_arr[mask]))
            lift = cond_prob / baseline_prob if baseline_prob > 0 else 0

            if lift > best_lift:
                best_lift = lift
                best_threshold = threshold_val
                best_cond_prob = cond_prob

        # 權重推導
        is_scoring = feat_name in _SCORING_FEATURES
        abs_ic = abs(ic)

        if not is_scoring:
            weight = 0.0  # 方向性特徵不參與評分
        elif abs_ic < _IC_SIGNIFICANT or symbols_significant < _MIN_SYMBOLS_CONSISTENT:
            weight = abs_ic * ic_consistency * _WEIGHT_SCALE * 0.1
        elif abs_ic < _IC_STRONG:
            weight = abs_ic * ic_consistency * _WEIGHT_SCALE * 0.5
        else:
            weight = abs_ic * ic_consistency * _WEIGHT_SCALE

        weight = round(min(weight, _WEIGHT_MAX), 2)

        return {
            "feature_name": feat_name,
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            **percentiles,
            "ic": round(ic, 4),
            "ic_pvalue": round(ic_pval, 6),
            "ic_per_symbol": ic_per_symbol,
            "ic_consistency": round(ic_consistency, 3),
            "symbols_significant": symbols_significant,
            "optimal_threshold": round(best_threshold, 4),
            "threshold_direction": threshold_direction,
            "conditional_prob": round(best_cond_prob, 4),
            "baseline_prob": round(baseline_prob, 4),
            "lift": round(best_lift, 3),
            "weight": weight,
            "total_samples": len(values),
            "symbols_used": len(ic_per_symbol),
        }

    def _compute_score_calibration(
        self,
        profiles: list[dict],
        all_rows: list[dict],
        baseline_prob: float,
        total_samples: int,
        symbols_count: int,
    ) -> dict:
        """用數據驅動權重模擬評分分佈，校準門檻和信心分界。"""
        scoring_profiles = [p for p in profiles if p["weight"] > 0.01]

        if not scoring_profiles:
            return {
                "score_threshold": 5.5,
                "high_boundary": 12.0,
                "medium_boundary": 8.0,
                "baseline_prob": round(baseline_prob * 100, 1),
                "total_samples": total_samples,
                "symbols_used": symbols_count,
            }

        # 模擬所有歷史窗口的評分
        scores = []
        hits = []
        for row in all_rows:
            score = 0.0
            for p in scoring_profiles:
                val = row["features"].get(p["feature_name"])
                if val is None:
                    continue
                if p["threshold_direction"] == "above":
                    if val > p["optimal_threshold"]:
                        score += p["weight"]
                    elif val > p["p50"]:
                        score += p["weight"] * 0.33
                else:
                    if val < p["optimal_threshold"]:
                        score += p["weight"]
                    elif val < p["p50"]:
                        score += p["weight"] * 0.33
            scores.append(score)
            hits.append(row["hit"])

        scores_arr = np.array(scores)
        hits_arr = np.array(hits)

        # 找最佳觸發門檻（precision 40-60%）
        best_threshold = float(np.percentile(scores_arr, 75))
        best_precision = 0.0

        for pct in range(60, 96, 2):
            t = float(np.percentile(scores_arr, pct))
            mask = scores_arr >= t
            count_above = int(np.sum(mask))
            if count_above < 20:
                continue
            precision = float(np.mean(hits_arr[mask]))
            if 0.35 <= precision <= 0.70:
                if abs(precision - 0.50) < abs(best_precision - 0.50):
                    best_threshold = t
                    best_precision = precision

        # 信心分界：在門檻之上分 high/medium/low
        above_mask = scores_arr >= best_threshold
        above_scores = scores_arr[above_mask]

        if len(above_scores) > 10:
            p66 = float(np.percentile(above_scores, 66))
            p33 = float(np.percentile(above_scores, 33))

            # 驗證：high 的命中率應 > medium
            high_boundary = round(p66, 2)
            medium_boundary = round(p33, 2)

            # 確保合理間距
            if high_boundary - medium_boundary < 1.0:
                high_boundary = medium_boundary + 1.0
        else:
            high_boundary = best_threshold * 1.8
            medium_boundary = best_threshold * 1.2

        now = taipei_now().isoformat()

        return {
            "score_threshold": round(best_threshold, 2),
            "high_boundary": round(high_boundary, 2),
            "medium_boundary": round(medium_boundary, 2),
            "baseline_prob": round(baseline_prob * 100, 1),
            "total_samples": total_samples,
            "symbols_used": symbols_count,
            "computed_at": now,
        }

    # ─── 存儲 ──────────────────────────────

    def _store_profiles(self, profiles: list[dict], meta: dict):
        """存入 DB。"""
        self._ensure_db()
        now = taipei_now().isoformat()

        with self._lock:
            for p in profiles:
                self._conn.execute(
                    """INSERT INTO scanner_feature_profiles
                       (feature_name, mean, std, p10, p25, p50, p75, p90,
                        ic, ic_pvalue, ic_per_symbol, ic_consistency, symbols_significant,
                        optimal_threshold, threshold_direction, conditional_prob,
                        baseline_prob, lift, weight,
                        total_samples, symbols_used, computed_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(feature_name) DO UPDATE SET
                       mean=excluded.mean, std=excluded.std,
                       p10=excluded.p10, p25=excluded.p25, p50=excluded.p50,
                       p75=excluded.p75, p90=excluded.p90,
                       ic=excluded.ic, ic_pvalue=excluded.ic_pvalue,
                       ic_per_symbol=excluded.ic_per_symbol,
                       ic_consistency=excluded.ic_consistency,
                       symbols_significant=excluded.symbols_significant,
                       optimal_threshold=excluded.optimal_threshold,
                       threshold_direction=excluded.threshold_direction,
                       conditional_prob=excluded.conditional_prob,
                       baseline_prob=excluded.baseline_prob,
                       lift=excluded.lift, weight=excluded.weight,
                       total_samples=excluded.total_samples,
                       symbols_used=excluded.symbols_used,
                       computed_at=excluded.computed_at""",
                    (
                        p["feature_name"], p["mean"], p["std"],
                        p["p10"], p["p25"], p["p50"], p["p75"], p["p90"],
                        p["ic"], p["ic_pvalue"],
                        json.dumps(p["ic_per_symbol"], ensure_ascii=False),
                        p["ic_consistency"], p["symbols_significant"],
                        p["optimal_threshold"], p["threshold_direction"],
                        p["conditional_prob"], p["baseline_prob"], p["lift"],
                        p["weight"], p["total_samples"], p["symbols_used"], now,
                    ),
                )

            # 存 meta
            for k, v in meta.items():
                self._conn.execute(
                    "INSERT INTO scanner_profile_meta (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (k, json.dumps(v) if not isinstance(v, str) else v),
                )

            self._conn.commit()

    # ─── 公開介面 ──────────────────────────────

    def get_feature_profiles(self) -> dict:
        """返回結構化 dict 供 _detect_signals() 消費。"""
        if self._profiles_cache:
            return self._profiles_cache

        self._ensure_db()

        rows = self._conn.execute(
            "SELECT * FROM scanner_feature_profiles ORDER BY weight DESC"
        ).fetchall()

        if not rows:
            return {}

        result: dict = {}
        for row in rows:
            result[row["feature_name"]] = {
                "weight": row["weight"],
                "optimal_threshold": row["optimal_threshold"],
                "threshold_direction": row["threshold_direction"],
                "p50": row["p50"],
                "ic": row["ic"],
                "ic_pvalue": row["ic_pvalue"],
                "ic_per_symbol": json.loads(row["ic_per_symbol"]) if row["ic_per_symbol"] else {},
                "ic_consistency": row["ic_consistency"],
                "symbols_significant": row["symbols_significant"],
                "lift": row["lift"],
                "conditional_prob": row["conditional_prob"],
                "baseline_prob": row["baseline_prob"],
                "mean": row["mean"],
                "std": row["std"],
                "p10": row["p10"],
                "p25": row["p25"],
                "p75": row["p75"],
                "p90": row["p90"],
                "total_samples": row["total_samples"],
                "symbols_used": row["symbols_used"],
            }

        # 讀取 meta
        meta_rows = self._conn.execute(
            "SELECT key, value FROM scanner_profile_meta"
        ).fetchall()
        meta = {}
        for mr in meta_rows:
            try:
                meta[mr["key"]] = json.loads(mr["value"])
            except (json.JSONDecodeError, TypeError):
                meta[mr["key"]] = mr["value"]

        result["_meta"] = meta
        self._profiles_cache = result
        return result

    def ensure_profiles(self):
        """確保 profiles 存在且未過期。過期或不存在時自動計算。"""
        self._ensure_db()

        meta_row = self._conn.execute(
            "SELECT value FROM scanner_profile_meta WHERE key='computed_at'"
        ).fetchone()

        if meta_row:
            try:
                from datetime import datetime
                computed_at = datetime.fromisoformat(meta_row["value"].strip('"'))
                now = taipei_now()
                if computed_at.tzinfo is None:
                    computed_at = computed_at.replace(tzinfo=now.tzinfo)
                age_days = (now - computed_at).total_seconds() / 86400
                if age_days < _PROFILE_EXPIRY_DAYS:
                    logger.debug(f"[特徵分析] Profiles 尚未過期（{age_days:.1f} 天前計算）")
                    return
            except Exception:
                pass

        logger.info("[特徵分析] Profiles 不存在或已過期，開始計算...")
        self.compute_all_profiles()

    def reset(self):
        """清空所有 profiles（下次掃描時會重新計算）。"""
        self._ensure_db()
        with self._lock:
            self._conn.execute("DELETE FROM scanner_feature_profiles")
            self._conn.execute("DELETE FROM scanner_profile_meta")
            self._conn.commit()
        self._profiles_cache = None
        logger.info("[特徵分析] 已清空所有特徵分析資料")


scanner_feature_profiler = ScannerFeatureProfiler()
