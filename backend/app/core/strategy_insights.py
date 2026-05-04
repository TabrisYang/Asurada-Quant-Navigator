"""阿斯拉量化系統 — v106 C3：策略洞察庫。

從 predictions + prediction_features 表中萃取「成功 vs 失敗」patterns，
注入 chart_state.historical_insights 給 LLM 引用。

設計原則：
- 不做硬性 override（只給 LLM 參考；明確標 confidence + 樣本量）
- 用 Wilson CI 下界排序避免少樣本誤導
- per-(symbol, timeframe, regime) 切分 → 高解析度 insight
- 每週自動重建（離線批次），運行時讀 JSON
- 樣本量 < 5 的組合直接跳過

注入結構：
chart_state.historical_insights = {
  "symbol": "ADA/USDT",
  "timeframe": "4h",
  "regime": "ranging",
  "insights": [
    {"pattern": "regime=ranging + RSI>70", "n": 8, "winrate": 0.62, "ci_lo": 0.35, "verdict": "弱訊號"},
    ...
  ],
  "overall_winrate_for_segment": 0.55,
  "n_samples": 12,
  "as_of": "2026-04-28"
}
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "db" / "predictions.db"
_INSIGHTS_PATH = Path(__file__).resolve().parents[2] / "data" / "db" / "strategy_insights.json"
_MIN_N_PER_SEGMENT = 5
_MIN_N_PER_PATTERN = 4
_INSIGHT_CACHE_TTL_DAYS = 14  # 14 天沒重建會在 prompt 標警示


def _wilson_lower(wins: int, n: int, z: float = 1.96) -> float:
    """Wilson 95% CI 下界。"""
    if n == 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    pm = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (center - pm) / denom)


def _regime_from_features(row: dict) -> str:
    """從 prediction_features 的 regime_* one-hot 還原 regime label。"""
    for r in ("trending_up", "trending_down", "ranging", "high_vol", "low_vol", "unknown"):
        if row.get(f"regime_{r}"):
            return r
    return "unknown"


def _load_joined_rows() -> list[dict]:
    """JOIN predictions + prediction_features，只取已驗證樣本。"""
    if not _DB_PATH.exists():
        return []
    with sqlite3.connect(_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("""
            SELECT p.id, p.symbol, p.timeframe, p.direction, p.status, p.confidence,
                   p.regime, p.created_at, p.actual_outcome_pct,
                   f.*
            FROM predictions p
            LEFT JOIN prediction_features f ON f.prediction_id = p.id
            WHERE p.status IN ('hit_target', 'hit_stop')
        """)
        return [dict(r) for r in cur.fetchall()]


def _segment_winrate(rows: list[dict]) -> tuple[int, int, float, float]:
    """傳一組 rows 算 (n, wins, winrate, wilson_lo)。"""
    n = len(rows)
    wins = sum(1 for r in rows if r["status"] == "hit_target")
    wr = wins / n if n > 0 else 0
    return n, wins, wr, _wilson_lower(wins, n)


def _bucket(value: Optional[float], thresholds: list[float]) -> Optional[str]:
    """數值分桶（None / NaN 跳過）。"""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    if v < thresholds[0]:
        return f"<{thresholds[0]}"
    for i, t in enumerate(thresholds[:-1]):
        if v < thresholds[i + 1]:
            return f"{t}-{thresholds[i+1]}"
    return f">={thresholds[-1]}"


def _extract_patterns(rows: list[dict]) -> list[dict]:
    """從 segment 內找有意義的條件 → 統計勝率。"""
    if len(rows) < _MIN_N_PER_PATTERN:
        return []

    patterns: list[dict] = []

    # RSI 分桶
    for bucket_name in ["<30", "30-50", "50-70", ">=70"]:
        sub = [r for r in rows if _bucket(r.get("rsi_14"), [30, 50, 70]) == bucket_name]
        if len(sub) >= _MIN_N_PER_PATTERN:
            n, w, wr, lo = _segment_winrate(sub)
            patterns.append({
                "pattern": f"RSI {bucket_name}",
                "n": n, "winrate": round(wr, 3), "ci_lower": round(lo, 3),
            })

    # ADX 分桶
    for bucket_name in ["<20", "20-30", ">=30"]:
        sub = [r for r in rows if _bucket(r.get("adx_14"), [20, 30]) == bucket_name]
        if len(sub) >= _MIN_N_PER_PATTERN:
            n, w, wr, lo = _segment_winrate(sub)
            patterns.append({
                "pattern": f"ADX {bucket_name}",
                "n": n, "winrate": round(wr, 3), "ci_lower": round(lo, 3),
            })

    # BB position（位置）
    for label, lo_b, hi_b in [("upper", 0.8, 1.0), ("middle", 0.4, 0.6), ("lower", 0.0, 0.2)]:
        sub = [r for r in rows
               if r.get("bb_position") is not None
               and lo_b <= r["bb_position"] <= hi_b]
        if len(sub) >= _MIN_N_PER_PATTERN:
            n, w, wr, lo = _segment_winrate(sub)
            patterns.append({
                "pattern": f"BB position={label}",
                "n": n, "winrate": round(wr, 3), "ci_lower": round(lo, 3),
            })

    # Breadth（市場廣度，僅 crypto）
    for bucket_name in ["<25", "25-50", "50-75", ">=75"]:
        sub = [r for r in rows if _bucket(r.get("breadth_pct"), [25, 50, 75]) == bucket_name]
        if len(sub) >= _MIN_N_PER_PATTERN:
            n, w, wr, lo = _segment_winrate(sub)
            patterns.append({
                "pattern": f"Breadth {bucket_name}",
                "n": n, "winrate": round(wr, 3), "ci_lower": round(lo, 3),
            })

    # 方向 × confidence
    for conf_name in ["high", "medium", "low"]:
        sub = [r for r in rows if (r.get("confidence") or "").lower() == conf_name]
        if len(sub) >= _MIN_N_PER_PATTERN:
            n, w, wr, lo = _segment_winrate(sub)
            patterns.append({
                "pattern": f"原始信心={conf_name}",
                "n": n, "winrate": round(wr, 3), "ci_lower": round(lo, 3),
            })

    # 按 ci_lower 排序，取最有意義的 top 6（避免 prompt 太肥）
    patterns.sort(key=lambda p: p["ci_lower"], reverse=True)
    return patterns[:6]


def _verdict_for(winrate: float, ci_lo: float, n: int) -> str:
    """把數據翻成口語結論。"""
    if n < _MIN_N_PER_PATTERN:
        return "樣本不足"
    if ci_lo >= 0.6:
        return "強訊號（高信心可採信）"
    if ci_lo >= 0.5:
        return "中等訊號"
    if winrate >= 0.5 and ci_lo < 0.4:
        return "弱訊號（樣本少需更多驗證）"
    if winrate < 0.4:
        return "歷史多失敗（避免進場）"
    return "中性"


def build_insights() -> dict:
    """掃整個 predictions DB，per-(symbol, timeframe, regime) 切片計算 insights。

    結果同時：
    1. 寫入 _INSIGHTS_PATH（離線快取）
    2. 回傳完整 dict 供 caller 使用
    """
    rows = _load_joined_rows()
    segments: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (
            r.get("symbol", "?"),
            r.get("timeframe", "?"),
            _regime_from_features(r) if r.get("regime_unknown") is not None else (r.get("regime") or "unknown"),
        )
        segments.setdefault(key, []).append(r)

    by_segment: dict[str, dict] = {}
    overall_n = 0
    overall_w = 0
    for (sym, tf, regime), seg_rows in segments.items():
        n, w, wr, lo = _segment_winrate(seg_rows)
        overall_n += n
        overall_w += w
        if n < _MIN_N_PER_SEGMENT:
            continue
        patterns = _extract_patterns(seg_rows)
        for p in patterns:
            p["verdict"] = _verdict_for(p["winrate"], p["ci_lower"], p["n"])
        by_segment[f"{sym}|{tf}|{regime}"] = {
            "symbol": sym,
            "timeframe": tf,
            "regime": regime,
            "n_samples": n,
            "wins": w,
            "winrate": round(wr, 3),
            "winrate_ci_lower": round(lo, 3),
            "patterns": patterns,
        }

    out = {
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "total_validated": overall_n,
        "overall_winrate": round(overall_w / overall_n, 3) if overall_n > 0 else 0,
        "n_segments_with_insight": len(by_segment),
        "by_segment": by_segment,
    }

    try:
        _INSIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _INSIGHTS_PATH.write_text(
            json.dumps(out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass

    return out


def _load_cached_insights() -> Optional[dict]:
    if not _INSIGHTS_PATH.exists():
        return None
    try:
        return json.loads(_INSIGHTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_insights_for(symbol: str, timeframe: str, regime: str) -> Optional[dict]:
    """注入 chart_state 用：給定當前 (symbol, tf, regime)，回傳對應 insight 或 None。

    若快取超過 14 天會在回傳值加 stale_warning。
    """
    cached = _load_cached_insights()
    if not cached:
        return None

    key = f"{symbol}|{timeframe}|{regime}"
    seg = cached.get("by_segment", {}).get(key)
    if not seg:
        return None

    # 檢查新鮮度
    try:
        as_of = datetime.strptime(cached["as_of"], "%Y-%m-%d %H:%M UTC").replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - as_of).days
        is_stale = age_days > _INSIGHT_CACHE_TTL_DAYS
    except Exception:
        is_stale = True

    out = {
        **seg,
        "as_of": cached["as_of"],
        "stale_warning": (
            f"此 insight 已 {age_days} 天沒重建，pattern 可能反映舊市況"
            if is_stale else None
        ),
    }
    return out


def format_insights_summary(insight: Optional[dict]) -> str:
    """LLM prompt 友善字串。"""
    if not insight:
        return ""
    lines = [
        f"📚 歷史洞察（{insight['symbol']} {insight['timeframe']} regime={insight['regime']}）：",
        f"  樣本 {insight['n_samples']} 筆，整體勝率 {insight['winrate']*100:.1f}%（Wilson 下界 {insight['winrate_ci_lower']*100:.1f}%）",
    ]
    for p in insight.get("patterns", [])[:5]:
        lines.append(
            f"  • {p['pattern']}: n={p['n']}, "
            f"winrate={p['winrate']*100:.0f}% (CI≥{p['ci_lower']*100:.0f}%) — {p['verdict']}"
        )
    if insight.get("stale_warning"):
        lines.append(f"  ⚠️ {insight['stale_warning']}")
    return "\n".join(lines)


if __name__ == "__main__":
    out = build_insights()
    print(f"已驗證樣本: {out['total_validated']}")
    print(f"整體勝率: {out['overall_winrate']*100:.1f}%")
    print(f"有效 insight 區段: {out['n_segments_with_insight']}")
    print(f"\n寫入: {_INSIGHTS_PATH}")
    for key, seg in list(out["by_segment"].items())[:5]:
        print(f"\n{key}: n={seg['n_samples']}, winrate={seg['winrate']*100:.1f}%")
        for p in seg["patterns"][:3]:
            print(f"  • {p['pattern']}: n={p['n']}, wr={p['winrate']*100:.0f}%, ci≥{p['ci_lower']*100:.0f}% → {p['verdict']}")
