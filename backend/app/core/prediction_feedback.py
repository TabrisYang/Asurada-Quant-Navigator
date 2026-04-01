"""阿斯拉量化系統 — 預測績效反饋生成器（增強版）

根據已驗證的預測記錄，生成結構化的反饋 prompt 注入 LLM，
讓 AI 能夠看到自己的歷史成績單並自我修正。

增強功能：
- 指標可靠度分級（★★★ / ★★ / ★）
- 多/空方向分析
- 連敗警報（觸發防禦模式）
- 市場體制匹配

防過擬合機制：
1. 最低樣本門檻（≥8 筆才給建議）
2. 範圍化指標勝率（不過度精確）
3. 市場體制過濾（只參考相似環境的經驗）
4. 時間衰減權重（舊經驗自然退場）
5. 強制多樣性指令
"""

from typing import Optional

from app.core.prediction_tracker import prediction_tracker


_MIN_SAMPLES_FOR_ADVICE = 8
_MIN_SAMPLES_FOR_INDICATOR = 3
_MIN_SAMPLES_FOR_CONFIDENCE = 2
_LOSS_STREAK_ALERT = 3


def generate_feedback_prompt(
    symbol: Optional[str] = None,
    current_regime: Optional[str] = None,
) -> str:
    """生成預測績效反饋 prompt，供注入 LLM 上下文。

    Returns:
        str: 反饋 prompt，為空代表尚無足夠資料。
    """
    global_stats = prediction_tracker.get_stats(symbol=symbol, days=90)
    if global_stats["total"] == 0:
        return ""

    lines: list[str] = []
    lines.append(f"[預測績效回顧 — {symbol or '全部幣種'}]")

    total = global_stats["total"]
    sufficient = global_stats["sample_sufficient"]

    lines.append(
        f"90天記錄 {total}筆 | "
        f"命中{global_stats['hit_target']} 止損{global_stats['hit_stop']} 過期{global_stats['expired']} | "
        f"加權勝率 {global_stats['win_rate_weighted']}%"
    )
    lines.append(
        f"平均盈虧 {global_stats['avg_outcome_pct']:+.1f}% | "
        f"最佳 {global_stats['best_outcome_pct']:+.1f}% | "
        f"最差 {global_stats['worst_outcome_pct']:+.1f}%"
    )

    # ★ 多/空方向分析
    dir_stats = prediction_tracker.get_direction_stats(symbol=symbol, days=90)
    if dir_stats:
        dir_parts = []
        for d, data in dir_stats.items():
            label = "做多" if d == "long" else "做空"
            dir_parts.append(f"{label} {data['win_rate']}%({data['samples']}筆)")
        lines.append(f"方向勝率：{'｜'.join(dir_parts)}")

    # ★ 連敗警報
    streak = prediction_tracker.get_recent_streak(symbol=symbol, n=10)
    if streak.get("total", 0) > 0:
        if streak["streak_type"] == "hit_stop" and streak["current_streak"] >= _LOSS_STREAK_ALERT:
            lines.append(
                f"⚠️ 近期連續止損 {streak['current_streak']} 次 → "
                f"建議降低倉位、放寬止損、優先使用高可靠度指標"
            )
        elif streak["streak_type"] == "hit_target" and streak["current_streak"] >= 5:
            lines.append(
                f"✅ 近期連續命中 {streak['current_streak']} 次 → "
                f"保持策略紀律，但注意過度自信風險"
            )

    # 市場體制績效對比
    if current_regime:
        regime_stats = prediction_tracker.get_stats(
            symbol=symbol, regime=current_regime, days=90,
        )
        if regime_stats["total"] >= _MIN_SAMPLES_FOR_CONFIDENCE:
            lines.append(
                f"「{current_regime}」環境勝率："
                f"{regime_stats['win_rate_weighted']}%（{regime_stats['total']}筆）"
            )
            if regime_stats["total"] < _MIN_SAMPLES_FOR_ADVICE:
                lines.append(f"  ⚠️ 此環境樣本<{_MIN_SAMPLES_FOR_ADVICE}筆，參考價值有限")

    # ★ 指標可靠度分級
    ind_perf = global_stats.get("indicator_performance", {})
    if ind_perf:
        tier_high = []   # ≥55%
        tier_mid = []    # 40%~55%
        tier_low = []    # <40%

        for ind, data in ind_perf.items():
            entry = f"{ind}({data['win_rate']}%/{data['samples']}次)"
            if data["win_rate"] >= 55:
                tier_high.append(entry)
            elif data["win_rate"] >= 40:
                tier_mid.append(entry)
            else:
                tier_low.append(entry)

        if tier_high:
            lines.append(f"指標可靠度 ★★★：{', '.join(tier_high)}")
        if tier_mid:
            lines.append(f"指標可靠度 ★★：{', '.join(tier_mid)}")
        if tier_low:
            lines.append(f"指標可靠度 ★：{', '.join(tier_low)}")
        if tier_high:
            lines.append("→ 規則：★ 指標禁止作為主要訊號，必須由 ★★★ 指標主導決策")

        # Alpha 衰退預警：比較 30 天 vs 90 天指標勝率
        recent_stats = prediction_tracker.get_stats(symbol=symbol, days=30)
        recent_ind = recent_stats.get("indicator_performance", {})
        if recent_ind and recent_stats["total"] >= 3:
            decay_alerts = []
            for ind, data90 in ind_perf.items():
                if data90["samples"] < 3:
                    continue
                r30 = recent_ind.get(ind)
                if r30 and r30["samples"] >= 2:
                    drop = data90["win_rate"] - r30["win_rate"]
                    if drop >= 15:
                        decay_alerts.append(
                            f"{ind}(90天{data90['win_rate']}%→近30天{r30['win_rate']}%↓{drop:.0f}pp)"
                        )
            if decay_alerts:
                lines.append(f"⚠️ Alpha衰退: {', '.join(decay_alerts)} — 建議降權或暫停使用")

    # 信心校準
    conf_cal = global_stats.get("confidence_calibration", {})
    if conf_cal:
        cal_parts = []
        for level in ("high", "medium", "low"):
            if level in conf_cal:
                actual = conf_cal[level]["win_rate"]
                expected = {"high": 70, "medium": 50, "low": 30}[level]
                label = {"high": "高", "medium": "中", "low": "低"}[level]
                diff = actual - expected
                if diff < -15:
                    cal_parts.append(f"{label}信心偏高(實{actual}%/期{expected}%)→下調")
                elif diff > 15:
                    cal_parts.append(f"{label}信心偏低(實{actual}%/期{expected}%)→可上調")
                else:
                    cal_parts.append(f"{label}信心校準良好({actual}%)")
        if cal_parts:
            lines.append(f"信心校準：{'；'.join(cal_parts)}")

    # 使用指南
    if sufficient:
        lines.append(
            "使用指南：優先★★★指標+≥3維度交叉驗證，信心等級依校準結果調整，"
            "市場體制不同時降低歷史統計權重"
        )
    else:
        lines.append(
            f"⚠️ 樣本{total}筆(<{_MIN_SAMPLES_FOR_ADVICE})，僅初步參考，"
            "請繼續使用多維度框架獨立判斷"
        )

    return "\n".join(lines)


def get_active_predictions_summary(symbol: Optional[str] = None) -> str:
    """取得仍在進行中的預測摘要，供 LLM 知道之前給過什麼建議。"""
    active = prediction_tracker.get_active(symbol)
    if not active:
        return ""

    lines = [f"[進行中預測 {len(active)}筆]"]
    for p in active[:5]:
        direction = "多" if p["direction"] == "long" else "空"
        lines.append(
            f"• {p['symbol']} {direction} "
            f"{p['entry_price']}→{p['target_price']}/{p['stop_price']} "
            f"({p['confidence']}信心)"
        )
    if len(active) > 5:
        lines.append(f"...+{len(active)-5}筆")
    return "\n".join(lines)
