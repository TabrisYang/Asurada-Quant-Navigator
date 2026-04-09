"""阿斯拉量化系統 — 掃描器自適應校準引擎

基於已驗證預警的歷史結果，自動校準掃描器參數：
- feature_weight: 特徵權重乘數（0.5x ~ 2.0x）
- score_threshold: 觸發門檻（4.0 ~ 8.0）
- confidence_boundary: 信心等級分界（high / medium）
- similarity_threshold: 餘弦相似度門檻（0.75 ~ 0.95）

設計模式仿 auto_adjuster.py：計算 → 存儲 → 查詢，搭載於 validate_expired_alerts() 之後執行。
"""

import json
import math
import sqlite3
from typing import Optional

from loguru import logger

from app.core.config.settings import settings
from app.utils.timezone import taipei_now


# ─── 安全邊界常數 ──────────────────────────────

_MIN_TOTAL_SAMPLES = 20       # 最低已驗證預警數，不足則不校準
_MIN_FEATURE_SAMPLES = 10     # 每特徵最低出現次數
_SMOOTHING = 0.3              # 每次校準只移動 30%（~7-8 次收斂）
_DECAY_LAMBDA = 0.02          # 指數衰減：半衰期 ~35 天

# 特徵權重邊界
_WEIGHT_MULT_MIN = 0.5
_WEIGHT_MULT_MAX = 2.0

# 門檻邊界
_THRESHOLD_MIN = 4.0
_THRESHOLD_MAX = 8.0
_THRESHOLD_STEP = 0.25        # 每次最大調整量

# 信心分界邊界
_HIGH_BOUNDARY_MIN = 10
_HIGH_BOUNDARY_MAX = 16
_MEDIUM_BOUNDARY_MIN = 6
_MEDIUM_BOUNDARY_MAX = 12
_BOUNDARY_MIN_GAP = 2         # high - medium 至少差 2

# 相似度門檻邊界
_SIM_MIN = 0.75
_SIM_MAX = 0.95
_SIM_STEP = 0.02

# Precision 目標區間
_PRECISION_LOW = 40            # < 40% → 提高門檻
_PRECISION_HIGH = 70           # > 70% → 降低門檻

# 硬編碼後備權重（對應 auto_scanner._detect_signals 中的 triggered 名稱）
_HARDCODED_FEATURE_WEIGHTS: dict[str, float] = {
    "vol_range_desync": 3.0,
    "direction_consistency_high": 2.5,
    "direction_consistency": 1.5,
    "ma5_no_cross": 2.5,
    "ma5_cross_low": 1.5,
    "price_efficiency": 1.5,
    "vol_momentum_pressure": 1.0,
    "max_streak": 1.5,
    "vol_even_distribution": 1.0,
    "accumulation_signal": 1.0,
    "vol_accelerating": 1.0,
    "atr_expanding": 1.0,
    "strong_body_ratio": 1.0,
    "vol_above_ma20": 1.0,
}


def _get_base_weights() -> dict[str, float]:
    """從 profiler 讀取數據驅動基礎權重，fallback 為硬編碼。"""
    try:
        from app.core.scanner_feature_profiler import scanner_feature_profiler
        profiles = scanner_feature_profiler.get_feature_profiles()
        if profiles.get("_meta"):
            return {
                k: v["weight"] for k, v in profiles.items()
                if not k.startswith("_") and v.get("weight", 0) > 0.01
            }
    except Exception:
        pass
    return _HARDCODED_FEATURE_WEIGHTS


# 向後相容的別名
_DEFAULT_FEATURE_WEIGHTS = _HARDCODED_FEATURE_WEIGHTS


class ScannerCalibrator:
    """掃描器自適應校準引擎。"""

    def __init__(self):
        self._conn: Optional[sqlite3.Connection] = None

    def _ensure_db(self):
        if self._conn:
            return
        db_file = settings.db_path / "predictions.db"
        self._conn = sqlite3.connect(str(db_file), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS scanner_calibrations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                calibration_type TEXT NOT NULL,
                key TEXT NOT NULL,
                default_value REAL NOT NULL,
                adjusted_value REAL NOT NULL,
                sample_count INTEGER NOT NULL,
                hit_rate REAL,
                reason TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                user_override INTEGER DEFAULT 0
            )
        """)
        self._conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_cal_type_key
            ON scanner_calibrations(calibration_type, key)
        """)
        self._conn.commit()

    # ─── 校準計算 ──────────────────────────────

    def _load_validated_alerts(self) -> list[dict]:
        """從 alerts 表載入已驗證的預警。"""
        self._ensure_db()
        rows = self._conn.execute(
            "SELECT * FROM alerts WHERE status IN ('triggered', 'expired') "
            "AND validated_at IS NOT NULL"
        ).fetchall()
        return [dict(r) for r in rows]

    def _calibrate_feature_weights(
        self, alerts: list[dict], baseline_hit_rate: float,
    ) -> list[dict]:
        """根據每個特徵在成功/失敗預警中的出現率校準權重。"""
        now_ts = taipei_now()
        feature_stats: dict[str, dict] = {}  # {name: {appearances, hits, weighted_app, weighted_hits}}

        for alert in alerts:
            tc = alert.get("trigger_conditions")
            if not tc:
                continue
            try:
                cond = json.loads(tc) if isinstance(tc, str) else tc
            except (json.JSONDecodeError, TypeError):
                continue

            triggered = cond.get("triggered", [])
            is_hit = alert["status"] == "triggered"

            # 時間衰減
            days_ago = 0.0
            if alert.get("validated_at"):
                try:
                    from datetime import datetime
                    val_time = datetime.fromisoformat(alert["validated_at"])
                    if val_time.tzinfo is None:
                        val_time = val_time.replace(tzinfo=now_ts.tzinfo)
                    days_ago = (now_ts - val_time).total_seconds() / 86400
                except Exception:
                    pass
            weight = math.exp(-_DECAY_LAMBDA * days_ago)

            for feat_name in triggered:
                # 只校準我們追蹤的特徵（忽略方向性觸發如 RSI=72>70）
                if feat_name not in _DEFAULT_FEATURE_WEIGHTS:
                    continue
                if feat_name not in feature_stats:
                    feature_stats[feat_name] = {
                        "appearances": 0, "hits": 0,
                        "weighted_app": 0.0, "weighted_hits": 0.0,
                    }
                s = feature_stats[feat_name]
                s["appearances"] += 1
                s["weighted_app"] += weight
                if is_hit:
                    s["hits"] += 1
                    s["weighted_hits"] += weight

        calibrations = []
        base_weights = _get_base_weights()
        for feat_name, default_w in base_weights.items():
            s = feature_stats.get(feat_name)
            if not s or s["appearances"] < _MIN_FEATURE_SAMPLES:
                continue

            hit_rate = (s["weighted_hits"] / s["weighted_app"] * 100) if s["weighted_app"] > 0 else 0
            relative = hit_rate / baseline_hit_rate if baseline_hit_rate > 0 else 1.0
            multiplier = max(_WEIGHT_MULT_MIN, min(_WEIGHT_MULT_MAX, relative))

            # 平滑：只移動 30% 向目標
            # 讀取上一次的 adjusted_value
            prev = self._get_current_value("feature_weight", feat_name)
            prev_mult = prev / default_w if prev and default_w > 0 else 1.0
            smoothed_mult = prev_mult + _SMOOTHING * (multiplier - prev_mult)
            smoothed_mult = max(_WEIGHT_MULT_MIN, min(_WEIGHT_MULT_MAX, smoothed_mult))
            adjusted = round(default_w * smoothed_mult, 2)

            direction = "⬆" if smoothed_mult > 1.05 else ("⬇" if smoothed_mult < 0.95 else "─")
            reason = (
                f"{feat_name} 命中率{hit_rate:.1f}%/{s['appearances']}筆 "
                f"(基準{baseline_hit_rate:.1f}%) → {smoothed_mult:.2f}x {direction}"
            )

            calibrations.append({
                "calibration_type": "feature_weight",
                "key": feat_name,
                "default_value": default_w,
                "adjusted_value": adjusted,
                "sample_count": s["appearances"],
                "hit_rate": round(hit_rate, 1),
                "reason": reason,
            })

        return calibrations

    def _calibrate_score_threshold(self, alerts: list[dict]) -> list[dict]:
        """根據 precision 校準觸發門檻。"""
        total = len(alerts)
        if total < _MIN_TOTAL_SAMPLES:
            return []

        triggered = sum(1 for a in alerts if a["status"] == "triggered")
        precision = triggered / total * 100

        prev = self._get_current_value("score_threshold", "trigger_threshold")
        current = prev if prev is not None else 5.5

        if precision < _PRECISION_LOW:
            target = current + _THRESHOLD_STEP
        elif precision > _PRECISION_HIGH:
            target = current - _THRESHOLD_STEP
        else:
            target = current  # 在目標區間內，不調整

        # 平滑
        smoothed = current + _SMOOTHING * (target - current)
        smoothed = round(max(_THRESHOLD_MIN, min(_THRESHOLD_MAX, smoothed)), 2)

        if abs(smoothed - 5.5) < 0.01 and abs(current - 5.5) < 0.01:
            return []  # 無變化

        reason = f"precision={precision:.1f}% ({triggered}/{total}) → 門檻 {smoothed}"
        return [{
            "calibration_type": "score_threshold",
            "key": "trigger_threshold",
            "default_value": 5.5,
            "adjusted_value": smoothed,
            "sample_count": total,
            "hit_rate": round(precision, 1),
            "reason": reason,
        }]

    def _calibrate_confidence_boundaries(self, alerts: list[dict]) -> list[dict]:
        """校準信心等級分界。"""
        conf_groups: dict[str, dict] = {}
        for a in alerts:
            c = a.get("confidence", "low")
            if c not in conf_groups:
                conf_groups[c] = {"total": 0, "hits": 0}
            conf_groups[c]["total"] += 1
            if a["status"] == "triggered":
                conf_groups[c]["hits"] += 1

        # 需要至少兩個等級都有足夠數據
        for level in ("high", "medium", "low"):
            g = conf_groups.get(level, {"total": 0})
            if g["total"] < 5:
                return []

        high_hr = conf_groups["high"]["hits"] / conf_groups["high"]["total"] * 100
        med_hr = conf_groups["medium"]["hits"] / conf_groups["medium"]["total"] * 100
        low_hr = conf_groups["low"]["hits"] / conf_groups["low"]["total"] * 100

        prev_high = self._get_current_value("confidence_boundary", "high")
        prev_med = self._get_current_value("confidence_boundary", "medium")
        cur_high = prev_high if prev_high is not None else 12.0
        cur_med = prev_med if prev_med is not None else 8.0

        new_high = cur_high
        new_med = cur_med

        # 若 high 命中率 ≤ medium → 拉高 high 分界
        if high_hr <= med_hr:
            new_high = cur_high + 1.0
        # 若 high 命中率遠高於 medium → 可以放寬
        elif high_hr > med_hr + 20:
            new_high = cur_high - 1.0

        # 若 low 命中率 ≥ medium → 降低 medium 分界
        if low_hr >= med_hr:
            new_med = cur_med - 1.0
        # 若 medium 命中率遠高於 low → 可以放寬
        elif med_hr > low_hr + 20:
            new_med = cur_med + 1.0

        # 邊界與間距約束
        new_high = max(_HIGH_BOUNDARY_MIN, min(_HIGH_BOUNDARY_MAX, new_high))
        new_med = max(_MEDIUM_BOUNDARY_MIN, min(_MEDIUM_BOUNDARY_MAX, new_med))
        if new_high - new_med < _BOUNDARY_MIN_GAP:
            new_med = new_high - _BOUNDARY_MIN_GAP

        calibrations = []
        if abs(new_high - 12.0) > 0.01 or prev_high is not None:
            calibrations.append({
                "calibration_type": "confidence_boundary",
                "key": "high",
                "default_value": 12.0,
                "adjusted_value": round(new_high, 1),
                "sample_count": conf_groups["high"]["total"],
                "hit_rate": round(high_hr, 1),
                "reason": f"high信心命中率{high_hr:.1f}% vs medium{med_hr:.1f}% → 分界={new_high}",
            })
        if abs(new_med - 8.0) > 0.01 or prev_med is not None:
            calibrations.append({
                "calibration_type": "confidence_boundary",
                "key": "medium",
                "default_value": 8.0,
                "adjusted_value": round(new_med, 1),
                "sample_count": conf_groups["medium"]["total"],
                "hit_rate": round(med_hr, 1),
                "reason": f"medium信心命中率{med_hr:.1f}% vs low{low_hr:.1f}% → 分界={new_med}",
            })

        return calibrations

    def _calibrate_similarity_threshold(self, alerts: list[dict]) -> list[dict]:
        """根據機率預測準確度校準相似度門檻。"""
        errors = []
        for a in alerts:
            tc = a.get("trigger_conditions")
            if not tc:
                continue
            try:
                cond = json.loads(tc) if isinstance(tc, str) else tc
            except (json.JSONDecodeError, TypeError):
                continue

            prob_data = cond.get("probability")
            if not prob_data or not isinstance(prob_data, dict):
                continue

            predicted = prob_data.get("headline_probability")
            actual_pct = a.get("outcome_pct")
            if predicted is None or actual_pct is None:
                continue

            # 將 outcome_pct 轉為 0/1（是否達到 move_threshold）
            # 預設 move_threshold=3.0，outcome > 3% = hit
            actual_hit = 1.0 if actual_pct > 3.0 else 0.0
            # predicted 是百分比
            errors.append(predicted / 100.0 - actual_hit)

        if len(errors) < _MIN_TOTAL_SAMPLES:
            return []

        # 平均誤差：正 = 高估，負 = 低估
        mean_error = sum(errors) / len(errors)
        mae = sum(abs(e) for e in errors) / len(errors)

        prev = self._get_current_value("similarity_threshold", "similarity")
        current = prev if prev is not None else 0.85

        if mae > 0.15:  # MAE > 15pp
            if mean_error > 0:
                # 持續高估 → 收緊門檻
                target = current + _SIM_STEP
            else:
                # 持續低估 → 放寬門檻
                target = current - _SIM_STEP
        else:
            target = current  # 準確度可接受

        smoothed = current + _SMOOTHING * (target - current)
        smoothed = round(max(_SIM_MIN, min(_SIM_MAX, smoothed)), 3)

        if abs(smoothed - 0.85) < 0.001 and prev is None:
            return []

        reason = f"MAE={mae*100:.1f}pp, bias={mean_error*100:+.1f}pp ({len(errors)}筆) → 門檻={smoothed}"
        return [{
            "calibration_type": "similarity_threshold",
            "key": "similarity",
            "default_value": 0.85,
            "adjusted_value": smoothed,
            "sample_count": len(errors),
            "hit_rate": None,
            "reason": reason,
        }]

    # ─── 輔助 ──────────────────────────────

    def _get_current_value(self, cal_type: str, key: str) -> Optional[float]:
        """取得當前校準值（若存在）。"""
        self._ensure_db()
        row = self._conn.execute(
            "SELECT adjusted_value FROM scanner_calibrations "
            "WHERE calibration_type=? AND key=?",
            (cal_type, key),
        ).fetchone()
        return row["adjusted_value"] if row else None

    # ─── 存儲 ──────────────────────────────

    def _store_calibrations(self, calibrations: list[dict]):
        """Upsert 校準值（跳過 user_override=1）。"""
        self._ensure_db()
        now = taipei_now().isoformat()

        for cal in calibrations:
            existing = self._conn.execute(
                "SELECT user_override FROM scanner_calibrations "
                "WHERE calibration_type=? AND key=?",
                (cal["calibration_type"], cal["key"]),
            ).fetchone()

            if existing and existing["user_override"] == 1:
                continue

            self._conn.execute(
                """INSERT INTO scanner_calibrations
                   (calibration_type, key, default_value, adjusted_value,
                    sample_count, hit_rate, reason, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(calibration_type, key) DO UPDATE SET
                   adjusted_value=excluded.adjusted_value,
                   sample_count=excluded.sample_count,
                   hit_rate=excluded.hit_rate,
                   reason=excluded.reason,
                   updated_at=excluded.updated_at""",
                (
                    cal["calibration_type"], cal["key"],
                    cal["default_value"], cal["adjusted_value"],
                    cal["sample_count"], cal["hit_rate"],
                    cal["reason"], now,
                ),
            )
        self._conn.commit()

    # ─── 公開介面 ──────────────────────────────

    def run_calibration_cycle(self) -> dict:
        """執行完整校準流程：查詢 → 計算 → 存儲 → 返回摘要。"""
        self._ensure_db()
        alerts = self._load_validated_alerts()

        total = len(alerts)
        if total < _MIN_TOTAL_SAMPLES:
            logger.debug(
                f"[校準] 已驗證預警不足（{total}/{_MIN_TOTAL_SAMPLES}），跳過校準"
            )
            return {"status": "insufficient_data", "total_samples": total}

        triggered = sum(1 for a in alerts if a["status"] == "triggered")
        baseline_hit_rate = triggered / total * 100

        all_calibrations: list[dict] = []
        all_calibrations.extend(self._calibrate_feature_weights(alerts, baseline_hit_rate))
        all_calibrations.extend(self._calibrate_score_threshold(alerts))
        all_calibrations.extend(self._calibrate_confidence_boundaries(alerts))
        all_calibrations.extend(self._calibrate_similarity_threshold(alerts))

        if all_calibrations:
            self._store_calibrations(all_calibrations)
            types = list({c["calibration_type"] for c in all_calibrations})
            logger.info(
                f"[校準] 完成校準：{len(all_calibrations)} 項調整 "
                f"(類型: {', '.join(types)}, 基準命中率: {baseline_hit_rate:.1f}%, 樣本: {total})"
            )
        else:
            logger.debug(f"[校準] 無需調整（樣本: {total}, 命中率: {baseline_hit_rate:.1f}%）")

        return {
            "status": "calibrated",
            "total_samples": total,
            "baseline_hit_rate": round(baseline_hit_rate, 1),
            "calibrations_count": len(all_calibrations),
            "types": list({c["calibration_type"] for c in all_calibrations}),
        }

    def get_active_calibrations(self) -> dict:
        """返回結構化校準值，供 _detect_signals() 直接消費。

        不足 20 筆已驗證預警時返回全部預設值。
        """
        self._ensure_db()

        # 先檢查是否有足夠數據
        total_row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM alerts "
            "WHERE status IN ('triggered', 'expired') AND validated_at IS NOT NULL"
        ).fetchone()
        total_samples = total_row["cnt"] if total_row else 0

        rows = self._conn.execute(
            "SELECT * FROM scanner_calibrations ORDER BY calibration_type, key"
        ).fetchall()

        result: dict = {
            "feature_weights": {},
            "score_threshold": {"default": 5.5, "adjusted": 5.5},
            "confidence_boundaries": {"high": 12.0, "medium": 8.0},
            "similarity_threshold": {"default": 0.85, "adjusted": 0.85},
            "last_updated": None,
            "total_samples": total_samples,
            "overall_hit_rate": None,
            "calibration_active": total_samples >= _MIN_TOTAL_SAMPLES,
        }

        if total_samples < _MIN_TOTAL_SAMPLES:
            # 返回全部預設值
            for feat, default_w in _get_base_weights().items():
                result["feature_weights"][feat] = {
                    "default": default_w, "adjusted": default_w,
                    "hit_rate": None, "samples": 0,
                }
            return result

        # 計算 overall_hit_rate
        triggered_row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM alerts WHERE status='triggered' AND validated_at IS NOT NULL"
        ).fetchone()
        triggered_cnt = triggered_row["cnt"] if triggered_row else 0
        result["overall_hit_rate"] = round(triggered_cnt / total_samples * 100, 1) if total_samples > 0 else 0

        # 填入校準值
        for row in rows:
            ct = row["calibration_type"]
            key = row["key"]
            if row["updated_at"] and (
                result["last_updated"] is None or row["updated_at"] > result["last_updated"]
            ):
                result["last_updated"] = row["updated_at"]

            if ct == "feature_weight":
                result["feature_weights"][key] = {
                    "default": row["default_value"],
                    "adjusted": row["adjusted_value"],
                    "hit_rate": row["hit_rate"],
                    "samples": row["sample_count"],
                    "id": row["id"],
                    "user_override": bool(row["user_override"]),
                }
            elif ct == "score_threshold":
                result["score_threshold"] = {
                    "default": row["default_value"],
                    "adjusted": row["adjusted_value"],
                    "id": row["id"],
                    "precision": row["hit_rate"],
                    "samples": row["sample_count"],
                }
            elif ct == "confidence_boundary":
                result["confidence_boundaries"][key] = row["adjusted_value"]
                result["confidence_boundaries"][f"{key}_detail"] = {
                    "default": row["default_value"],
                    "adjusted": row["adjusted_value"],
                    "hit_rate": row["hit_rate"],
                    "samples": row["sample_count"],
                    "id": row["id"],
                }
            elif ct == "similarity_threshold":
                result["similarity_threshold"] = {
                    "default": row["default_value"],
                    "adjusted": row["adjusted_value"],
                    "id": row["id"],
                    "samples": row["sample_count"],
                    "reason": row["reason"],
                }

        # 補齊未校準的特徵
        for feat, default_w in _get_base_weights().items():
            if feat not in result["feature_weights"]:
                result["feature_weights"][feat] = {
                    "default": default_w, "adjusted": default_w,
                    "hit_rate": None, "samples": 0,
                }

        return result

    def reset_all(self):
        """重置所有校準值為預設。"""
        self._ensure_db()
        self._conn.execute("DELETE FROM scanner_calibrations")
        self._conn.commit()
        logger.info("[校準] 已重置所有校準值為預設")

    def set_override(self, cal_id: int, value: Optional[float] = None) -> dict:
        """用戶手動覆寫校準值。"""
        self._ensure_db()
        if value is not None:
            self._conn.execute(
                "UPDATE scanner_calibrations SET adjusted_value=?, user_override=1 WHERE id=?",
                (value, cal_id),
            )
        else:
            self._conn.execute(
                "UPDATE scanner_calibrations SET user_override=1 WHERE id=?",
                (cal_id,),
            )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT * FROM scanner_calibrations WHERE id=?", (cal_id,),
        ).fetchone()
        return dict(row) if row else {}


scanner_calibrator = ScannerCalibrator()
