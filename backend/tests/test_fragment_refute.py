"""碎片證偽淘汰契約（v155）

驗證：penalize_by_source 降權（refuted_count+1）、累計 2 次刪除、
種子碎片不受影響、檢索品質乘 0.5^refuted 懲罰公式。
用 in-memory SQLite 直插資料，不依賴 embedding 模型。
"""

import sqlite3

import numpy as np

from app.core.knowledge_fragments import (
    KnowledgeFragmentStore,
    _compute_fragment_quality,
)


def _make_store():
    """建一個接 in-memory DB 的 store（繞過 __init__ 的檔案路徑與模型依賴）"""
    store = KnowledgeFragmentStore.__new__(KnowledgeFragmentStore)
    store._conn = sqlite3.connect(":memory:", check_same_thread=False)
    store._conn.execute("""
        CREATE TABLE fragments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL,
            fragment_type TEXT NOT NULL DEFAULT 'general',
            symbol TEXT NOT NULL DEFAULT '',
            source_question TEXT,
            created_at REAL NOT NULL,
            hit_count INTEGER NOT NULL DEFAULT 0,
            last_hit_at REAL,
            is_seed INTEGER NOT NULL DEFAULT 0,
            refuted_count INTEGER NOT NULL DEFAULT 0
        )
    """)
    return store


def _insert(store, content, src, symbol="BTC/USDT", is_seed=0):
    store._conn.execute(
        "INSERT INTO fragments (content, embedding, fragment_type, symbol, "
        "source_question, created_at, is_seed) VALUES (?, ?, 'insight', ?, ?, 1.0, ?)",
        (content, np.zeros(4, dtype=np.float32).tobytes(), symbol, src, is_seed),
    )
    store._conn.commit()


def _refuted(store, content):
    row = store._conn.execute(
        "SELECT refuted_count FROM fragments WHERE content=?", (content,)
    ).fetchone()
    return row[0] if row else None  # None = 已被刪除


class TestPenalizeBySource:
    def test_penalize_increments_and_second_deletes(self):
        store = _make_store()
        _insert(store, "frag_a", "分析 BTC 壓縮")
        _insert(store, "frag_b", "分析 BTC 壓縮")
        _insert(store, "frag_other", "別的問題")

        n = store.penalize_by_source("分析 BTC 壓縮", "BTC/USDT")
        assert n == 2
        assert _refuted(store, "frag_a") == 1
        assert _refuted(store, "frag_other") == 0  # 不同來源不受影響

        store.penalize_by_source("分析 BTC 壓縮", "BTC/USDT")
        assert _refuted(store, "frag_a") is None, "累計 2 次應被刪除"
        assert _refuted(store, "frag_other") == 0

    def test_seed_fragments_immune(self):
        store = _make_store()
        _insert(store, "seed_frag", "分析 BTC 壓縮", is_seed=1)
        n = store.penalize_by_source("分析 BTC 壓縮", "BTC/USDT")
        assert n == 0
        assert _refuted(store, "seed_frag") == 0

    def test_symbol_scoped(self):
        store = _make_store()
        _insert(store, "frag_eth", "同一個問題", symbol="ETH/USDT")
        n = store.penalize_by_source("同一個問題", "BTC/USDT")
        assert n == 0
        assert _refuted(store, "frag_eth") == 0

    def test_empty_source_noop(self):
        store = _make_store()
        assert store.penalize_by_source("", "BTC/USDT") == 0


class TestQualityPenaltyFormula:
    def test_refuted_halves_quality(self):
        base = _compute_fragment_quality("insight", hit_count=5)
        assert base * (0.5 ** 1) == base / 2
        assert base * (0.5 ** 2) == base / 4
        # 命中加成無法翻越證偽懲罰：refuted 一次後即使 hit 滿載也低於未 refuted 基準
        maxed = _compute_fragment_quality("insight", hit_count=100)
        assert maxed * 0.5 < _compute_fragment_quality("insight", hit_count=0)
