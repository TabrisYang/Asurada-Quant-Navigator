"""v118 regression tests：修「看漲說漲」bias 的三道防線。

Diagnose 發現（從 predictions.db 245 筆已驗證 prediction）：
- BULLISH regime 100% 看多（23 long : 0 short）但命中率僅 21.7%（賠錢領域）
- 整體多空 2:1 偏多（160:81）
- 信心 high/medium/low 命中率沒區別（calibration 失效）

修法：注入 regime_warning + direction_balance + prompt 三道強規則。
"""

import pathlib

from app.core.prediction_tracker import prediction_tracker


# ─── classify_regime ────────────────────────────────────

def test_classify_regime_bullish_keywords():
    cls = prediction_tracker.classify_regime
    for s in ["trending_up", "趨勢上行", "強趨勢上行+時框衝突", "BULLISH", "上升"]:
        assert cls(s) == "BULLISH", f"{s!r} 應分類為 BULLISH"


def test_classify_regime_bearish_keywords():
    cls = prediction_tracker.classify_regime
    for s in ["trending_down", "趨勢下行_動能衰減", "空頭趨勢", "弱趨勢偏空"]:
        assert cls(s) == "BEARISH", f"{s!r} 應分類為 BEARISH"


def test_classify_regime_bearish_priority_over_bullish():
    """『趨勢下行』裡也含『下』，但不該被『上』優先 match — 確認 BEARISH 在 BULLISH 之前判。"""
    cls = prediction_tracker.classify_regime
    assert cls("趨勢下行") == "BEARISH"


def test_classify_regime_range_keywords():
    cls = prediction_tracker.classify_regime
    for s in ["ranging", "盤整", "盤整偏多", "低波盤整", "均值回歸"]:
        assert cls(s) == "RANGE", f"{s!r} 應分類為 RANGE"


def test_classify_regime_unknown_to_other():
    cls = prediction_tracker.classify_regime
    assert cls("unknown") == "OTHER"
    assert cls("") == "OTHER"
    assert cls(None) == "OTHER"


# ─── get_regime_class_stats ─────────────────────────────

def test_get_regime_class_stats_returns_dict():
    """有歷史資料時，回傳結構正確。"""
    result = prediction_tracker.get_regime_class_stats(days=90)
    # 至少有一個 class 有資料（系統有歷史 prediction）
    assert isinstance(result, dict)
    if result:
        for cls_name, data in result.items():
            assert cls_name in {"BULLISH", "BEARISH", "RANGE", "OTHER"}
            assert "samples" in data
            assert "win_rate" in data
            assert "long" in data
            assert "short" in data
            assert "long_pct" in data
            assert data["samples"] > 0


def test_get_regime_class_stats_long_pct_consistent():
    """long + short 加總應該 ≤ samples（hit_target 之外的 status 也算）。"""
    result = prediction_tracker.get_regime_class_stats(days=90)
    for cls_name, data in result.items():
        # long + short 不會超過 samples（中間有 hold/None direction）
        assert data["long"] + data["short"] <= data["samples"], (
            f"{cls_name}: long({data['long']}) + short({data['short']}) > samples({data['samples']})"
        )


# ─── chat.py 注入邏輯 ──────────────────────────────────

# v157：注入邏輯已從 chat.py 搬到 chat_context.py，改讀整個 chat*.py 家族的原始碼
# （只掃 chat.py 會因為搬家而假性紅燈 / 未來假性綠燈）。
_ROUTES = pathlib.Path(__file__).resolve().parent.parent / "app" / "api" / "routes"


def _chat_family_src() -> str:
    return "\n".join(
        p.read_text(encoding="utf-8") for p in sorted(_ROUTES.glob("chat*.py"))
    )


def test_chat_py_injects_regime_warning():
    """chat.py 必須在 recent_accuracy 注入後加 regime_warning（防線 1）。"""
    src = _chat_family_src()
    assert 'regime_warning' in src and 'get_regime_class_stats' in src, (
        "chat.py 必須注入 regime_warning（用 get_regime_class_stats 取資料）"
    )
    # 警告觸發條件：n >= 10 且 win_rate < 50
    assert 'samples", 0) >= 10' in src, "regime_warning 必須要求 samples >= 10"
    assert 'win_rate", 0) < 50' in src, "regime_warning 必須要求 win_rate < 50"


def test_chat_py_injects_direction_balance():
    """chat.py 必須注入 direction_balance（防線 3）。"""
    src = _chat_family_src()
    assert 'direction_balance' in src, "chat.py 必須注入 direction_balance"
    assert 'biased_long' in src and 'biased_short' in src, (
        "direction_balance 必須含 biased_long / biased_short flag"
    )


# ─── function_defs.py prompt 規則 ──────────────────────

FUNC_DEFS_PY = (
    pathlib.Path(__file__).resolve().parent.parent / "app" / "core" / "llm" / "function_defs.py"
)
# prompt 規則內容已抽至 prompt_modules.py（repo health 行數護欄），兩檔串接檢查
PROMPT_MODULES_PY = FUNC_DEFS_PY.parent / "prompt_modules.py"


def _prompt_source() -> str:
    return FUNC_DEFS_PY.read_text(encoding="utf-8") + PROMPT_MODULES_PY.read_text(encoding="utf-8")


def test_function_defs_has_regime_warning_rule():
    """function_defs 必須含 regime_warning 強規則（防線 1）。"""
    src = _prompt_source()
    assert 'regime_warning' in src, "function_defs 必須含 regime_warning 規則"
    assert 'contrarian' in src.lower() or '逆向' in src or '逆勢' in src, (
        "regime_warning 規則必須提到 contrarian / 逆向視角"
    )


def test_function_defs_has_high_confidence_threshold():
    """function_defs 必須含信心 high 樣本門檻規則（防線 2）。"""
    src = _prompt_source()
    # 應該有「禁止 high」+ 樣本相關門檻
    has_block_high = any(
        kw in src for kw in ['禁止」使用「高」', '禁止使用「高」', '禁止給「高」', '禁止**使用「高」']
    )
    assert has_block_high, (
        "function_defs 必須含『禁止使用「高」信心』之類規則（防線 2）"
    )


def test_function_defs_has_direction_balance_rule():
    """function_defs 必須含 direction_balance 規則（防線 3）。"""
    src = _prompt_source()
    assert 'direction_balance' in src, "function_defs 必須含 direction_balance 規則"
    assert 'biased_long' in src or '看多' in src and 'biased' in src.lower(), (
        "direction_balance 規則必須提到 biased_long / 連續看多"
    )
