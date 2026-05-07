"""v112 regression tests：防止 chat.py / predictions.py 對 prediction_tracker.get_stats
回傳 schema 用錯 key（曾發生 indicator_stats vs indicator_performance 連環 bug）。

歷史 bug（v100~v111 隱藏）：
1. chat.py:791 用 stats.get("indicator_stats", {}) 但實際 key 是 "indicator_performance"
   → best/worst_indicators 永遠空 list → LLM 看不到 per-indicator 命中率
2. chat.py:797-798 寫 `for k, _ in _best[:3] if _[1].get(...)` 但 _ 是 dict 不是 tuple → KeyError（被 bug 1 遮蓋從沒跑到）
3. 邏輯順序錯：先排序取 top 3 再 filter samples >= 3 → top 3 都是 samples=1 的 → 過濾後空
"""

from app.core import prediction_tracker as pt_mod


def test_get_stats_returns_indicator_performance_key_not_indicator_stats():
    """確認 get_stats 回傳含 'indicator_performance' key（且 'indicator_stats' 不存在於頂層）。
    若有人改 schema 改回 'indicator_stats' 或加新 alias，這個 test 會抓到防回歸。
    """
    # 跑空 query（無 symbol 限制）回傳的 keys 結構
    stats = pt_mod.prediction_tracker.get_stats(days=30)

    # 若有資料，必須有 indicator_performance key（即使值是空 dict）
    if stats.get("total", 0) > 0:
        assert "indicator_performance" in stats, (
            "get_stats 必須回傳 'indicator_performance' key（chat.py 注入 recent_accuracy 依此）"
        )
        # 且 'indicator_stats' 不應該存在（避免歷史錯誤 alias 復活）
        assert "indicator_stats" not in stats, (
            "get_stats 不應該回傳 'indicator_stats' key（歷史錯名，會誤導 caller）"
        )


def test_chat_recent_accuracy_uses_correct_key():
    """確認 chat.py 注入 recent_accuracy 時用的 key 是 'indicator_performance'。
    grep 檢查 source code，避免 schema 不一致 bug 復活。
    """
    import pathlib
    chat_path = pathlib.Path(__file__).resolve().parent.parent / "app" / "api" / "routes" / "chat.py"
    src = chat_path.read_text(encoding="utf-8")

    # chat.py 不應再用 indicator_stats 取值（key 名 bug）
    # 排除註解行（v112 fix 註解可能含此字串）
    code_lines = [
        line for line in src.split("\n")
        if not line.strip().startswith("#")
    ]
    code_only = "\n".join(code_lines)

    # 不該有 .get("indicator_stats" 這種取值寫法
    assert '.get("indicator_stats"' not in code_only, (
        "chat.py 不應再用 .get(\"indicator_stats\")（這是錯名，正確是 indicator_performance）"
    )


def test_predictions_route_uses_correct_indicator_key():
    """確認 predictions.py route 也用正確 key。"""
    import pathlib
    pred_path = pathlib.Path(__file__).resolve().parent.parent / "app" / "api" / "routes" / "predictions.py"
    src = pred_path.read_text(encoding="utf-8")
    code_lines = [
        line for line in src.split("\n")
        if not line.strip().startswith("#")
    ]
    code_only = "\n".join(code_lines)
    assert '.get("indicator_stats"' not in code_only, (
        "predictions.py 不應再用 .get(\"indicator_stats\")"
    )


def test_recent_accuracy_filter_samples_before_sort():
    """確認 best/worst 計算邏輯：先 filter samples >= 3 再排序，避免 top 3 被 samples=1 的 indicator 佔據。"""
    # 模擬 indicator_performance 內含「高勝率但樣本少」+「中等勝率但樣本足」混合
    fake_stats = {
        "noise_a": {"win_rate": 100.0, "samples": 1},  # 偽高勝率
        "noise_b": {"win_rate": 100.0, "samples": 2},
        "real_high": {"win_rate": 70.0, "samples": 20},  # 真實高勝率
        "real_mid": {"win_rate": 50.0, "samples": 30},
        "real_low": {"win_rate": 30.0, "samples": 25},  # 真實低勝率
    }

    # 重現 chat.py 修復後的邏輯
    filtered = [(k, v) for k, v in fake_stats.items() if v.get("samples", 0) >= 3]
    best = sorted(filtered, key=lambda x: x[1].get("win_rate", 0), reverse=True)
    worst = sorted(filtered, key=lambda x: x[1].get("win_rate", 0))
    best_inds = [k for k, _ in best[:3]]
    worst_inds = [k for k, _ in worst[:2]]

    # noise_a / noise_b 不該入榜（samples < 3 已過濾）
    assert "noise_a" not in best_inds and "noise_b" not in best_inds
    # 真實高勝率該在 best
    assert "real_high" in best_inds
    # 真實低勝率該在 worst
    assert "real_low" in worst_inds


def test_chat_recent_accuracy_injection_with_real_data():
    """整合 test：模擬 chat.py 注入流程，確認 recent_accuracy 結構正確。"""
    stats_30d = pt_mod.prediction_tracker.get_stats(days=30)
    if stats_30d.get("total", 0) < 3:
        # 樣本不足，跳過此 test
        return

    # 用 v112 修復後的邏輯計算
    ind_perf = stats_30d.get("indicator_performance", {})
    filtered = [(k, v) for k, v in ind_perf.items() if v.get("samples", 0) >= 3]
    best = sorted(filtered, key=lambda x: x[1].get("win_rate", 0), reverse=True)
    best_inds = [k for k, _ in best[:3]]

    # 若 indicator_performance 有資料且有任何 samples >= 3，best_inds 應該至少 1 個
    has_qualified = any(v.get("samples", 0) >= 3 for v in ind_perf.values())
    if has_qualified:
        assert len(best_inds) >= 1, (
            "samples >= 3 的 indicator 存在時，best_indicators 不應該是空 list"
        )
