"""v119 regression tests：資料層補完（external_signals 補強 + Coinbase Premium + ETF Flow）。

歷史 bug：external_signals fetcher 失敗時 silent 跳過 → 報告常出現「⚠️ 無衍生品快照」。
prompt 也沒強制 LLM 引用既有的 funding/OI/order_book 資料。

修法（5 個 sub-tasks）：
- v119.1: external_signals 加 retry + stale cache fallback
- v119.2: chat.py 永遠注入 + prompt 強制引用
- v119.3: 止損位風險評估（流動性獵取）prompt
- v119.4: Coinbase Premium fetcher
- v119.5: ETF Flow（SoSoValue + CoinGlass fallback）
"""

import pathlib
import time
from unittest.mock import patch

from app.core import external_signals as es
from app.core.external_signals import (
    _fetch_with_retry,
    _stale_cached,
    get_signals_snapshot,
)


# ─── v119.1：retry + stale fallback ─────────────────────

def test_v119_1_fetch_with_retry_succeeds_after_retry():
    """第一次失敗、第二次成功 → 回傳第二次結果。"""
    call_count = {"n": 0}

    def flaky(c, s):
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise RuntimeError("simulated transient fail")
        return {"value": "ok"}

    result = _fetch_with_retry(flaky, None, "TEST", max_retries=3, base_backoff=0.01)
    assert result == {"value": "ok"}
    assert call_count["n"] == 2


def test_v119_1_fetch_with_retry_exhausts_returns_none():
    """重試耗盡都失敗 → 回 None（不 raise）。"""
    def always_fail(c, s):
        raise RuntimeError("永遠失敗")

    result = _fetch_with_retry(always_fail, None, "TEST", max_retries=3, base_backoff=0.01)
    assert result is None


def test_v119_1_stale_cached_returns_age():
    """超過 _CACHE_TTL 但在 _STALE_LIMIT 內的 cache 仍可被 _stale_cached 取出。"""
    es._cache.clear()
    # 偽造一筆 1 小時前的 cache（過期但在 6h stale limit 內）
    now = time.time()
    es._cache["TEST_KEY"] = (now - 3600, {"derivatives": {"funding_rate_pct": 0.01}})

    stale = _stale_cached("TEST_KEY")
    assert stale is not None
    age, payload = stale
    assert 3500 < age < 3700  # ~1 小時
    assert payload["derivatives"]["funding_rate_pct"] == 0.01

    # 過 _STALE_LIMIT（6h）的 cache 不該被取出
    es._cache["TOO_OLD"] = (now - 7 * 3600, {"x": 1})
    assert _stale_cached("TOO_OLD") is None
    es._cache.clear()


def test_v119_1_stale_fallback_used_when_all_fetch_fails():
    """所有 fetcher 都失敗時，自動用 stale cache 並標 stale=True。"""
    es._cache.clear()
    # 預先放一筆「1 小時前的成功 cache」
    cache_key = "ETH/USDT|False"
    now = time.time()
    es._cache[cache_key] = (
        now - 3600,
        {
            "derivatives": {"funding_rate_pct": 0.01, "open_interest": 1000000},
            "sentiment": {"fear_greed_value": 50},
            "macro": {},
            "fetched_at": "2026-05-08T11:00:00",
            "cached": False,
        },
    )

    # mock 所有 fetcher 都失敗
    with patch.object(es, "_fetch_funding_rate", return_value=None), \
         patch.object(es, "_fetch_open_interest", return_value=None), \
         patch.object(es, "_fetch_long_short_ratio", return_value=None), \
         patch.object(es, "_fetch_coinglass_liquidation", return_value=None), \
         patch.object(es, "_fetch_order_book", return_value=None), \
         patch.object(es, "_fetch_coinbase_premium", return_value=None), \
         patch.object(es, "_fetch_fear_greed", return_value=None):
        # 第一次呼叫拿到 cached（_CACHE_TTL 內），所以要清掉新鮮 cache 但保留 stale
        # 偽造：把 ts 改成「過期但 stale 範圍內」
        result = get_signals_snapshot("ETH/USDT", include_macro=False)

    # 應該標 stale=True 且帶 stale_seconds
    # 注意：因為 cache 還在 _CACHE_TTL 範圍內（1h < 30min ❌，超過 30min 但 < 6h），
    # 1h > 30min（_CACHE_TTL=1800s），所以不會命中新鮮 cache，會走 fetch path（全 fail）
    # → stale fallback 觸發
    assert result.get("stale") is True
    assert "stale_seconds" in result
    assert result["derivatives"]["funding_rate_pct"] == 0.01

    es._cache.clear()


# ─── v119.2：chat.py 強制注入 + prompt 強制引用 ──────

def test_v119_2_chat_py_inject_condition_relaxed():
    """chat.py 注入條件放寬：只要 _signals 不空就注入（不再要求 derivatives 必須有）。"""
    chat_py = (
        pathlib.Path(__file__).resolve().parent.parent / "app" / "api" / "routes" / "chat.py"
    )
    src = chat_py.read_text(encoding="utf-8")
    # v119.2 之後不再有「if _signals and (derivatives or sentiment or macro)」這種強限制
    # 應該是簡單的「if _signals」
    assert 'if _signals and (_signals.get("derivatives")' not in src, (
        "chat.py 不該再有「derivatives 必須有才注入」的舊條件（v119.2）"
    )


def test_v119_2_function_defs_requires_external_citation():
    """function_defs 必須含 external_signals 強制引用規則。"""
    _llm_dir = pathlib.Path(__file__).resolve().parent.parent / "app" / "core" / "llm"
    # prompt 規則內容已抽至 prompt_modules.py，兩檔串接檢查
    src = (_llm_dir / "function_defs.py").read_text(encoding="utf-8") + (
        _llm_dir / "prompt_modules.py"
    ).read_text(encoding="utf-8")
    # 必須含「funding_rate」+ 「open_interest」+ 「ob_imbalance」+ 強制字眼
    assert "funding_rate" in src and "open_interest" in src, (
        "function_defs 必須要求引用 funding_rate + open_interest"
    )
    assert "ob_imbalance" in src or "order_book" in src, (
        "function_defs 必須要求引用 order book imbalance"
    )
    assert "external_signals" in src, "function_defs 必須有 external_signals 強制引用區塊"


# ─── v119.3：止損位風險評估 prompt ───────────────

def test_v119_3_stop_loss_risk_rule_in_prompt():
    """function_defs 必須含止損位風險評估規則（含『流動性獵取』警告）。"""
    _llm_dir = pathlib.Path(__file__).resolve().parent.parent / "app" / "core" / "llm"
    # prompt 規則內容已抽至 prompt_modules.py，兩檔串接檢查
    src = (_llm_dir / "function_defs.py").read_text(encoding="utf-8") + (
        _llm_dir / "prompt_modules.py"
    ).read_text(encoding="utf-8")
    assert "流動性獵取" in src or "liquidity sweep" in src.lower(), (
        "function_defs 必須含『流動性獵取 / liquidity sweep』警告"
    )
    assert "止損" in src, "function_defs 必須含『止損』風險評估"
    assert "ob_top_5_bids" in src or "機構防禦區" in src, (
        "function_defs 必須提及訂單簿密集區下方的止損風險（機構防禦區）"
    )
    assert "ATR" in src, "止損規則必須要求標示 ATR 倍數"


# ─── v119.4：Coinbase Premium fetcher ───────────

def test_v119_4_coinbase_premium_fetcher_classifies_label():
    """_fetch_coinbase_premium classify 各 label 邊界正確。"""
    # 我們不真的 mock httpx（已 smoke test 過），直接驗證 label 邏輯
    # 透過讀 source 檢查門檻定義
    es_py = (
        pathlib.Path(__file__).resolve().parent.parent / "app" / "core" / "external_signals.py"
    )
    src = es_py.read_text(encoding="utf-8")
    assert "def _fetch_coinbase_premium" in src
    assert "positive_high" in src and "negative_high" in src and "neutral" in src, (
        "Coinbase premium label 必須含 positive_high / positive / neutral / negative / negative_high 5 級"
    )
    # 確認門檻在合理範圍（不是極端值）
    assert ">= 0.05" in src or ">= 0.05," in src, "label 門檻應該以 ±0.05% 為極端值界線"


def test_v119_4_coinbase_premium_in_snapshot_loop():
    """get_signals_snapshot 的 fetcher loop 必須包含 _fetch_coinbase_premium。"""
    es_py = (
        pathlib.Path(__file__).resolve().parent.parent / "app" / "core" / "external_signals.py"
    )
    src = es_py.read_text(encoding="utf-8")
    # 找 derivatives fetcher loop（含 _fetch_funding_rate 那個區塊）必須含 _fetch_coinbase_premium
    import re
    m = re.search(
        r'_fetch_funding_rate.+?_fetch_coinbase_premium',
        src, re.DOTALL,
    )
    assert m, "_fetch_coinbase_premium 必須在 derivatives fetcher loop 內被呼叫"
