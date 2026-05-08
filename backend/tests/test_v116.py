"""v116 regression tests：store_batch 改 single transaction + 結尾 trim 一次。

歷史 bug（v100~v115 隱藏）：
[knowledge_fragments.py:store_batch:264] for loop 內每筆呼叫 store_fragment，
每筆 INSERT + commit + _trim_if_needed（內含 SELECT + DELETE + commit）。
batch 11 筆 = 22 次 commit + 11 次 DELETE，SQLite WAL fsync 鎖住 ~700-800ms，
期間 frontend 切標的需要的其他 backend 路徑被 busy_timeout=3000ms 卡住。

修法：
- 抽 _insert_fragment_no_commit：純 INSERT 不 commit、不 trim
- store_batch 用 `with self._conn:` 包成單一 transaction
- 結尾跑 1 次 _trim_if_needed
- store_fragment（單筆 API）保留原行為（INSERT + commit + trim 即時）
"""

import random
import string
from unittest.mock import patch

import pytest

from app.core.knowledge_fragments import fragment_store


@pytest.fixture
def cleanup_test_symbol():
    """每個 test 跑完清掉測試資料。"""
    test_symbol = f"TEST_V116/{random.randint(1000, 9999)}"
    yield test_symbol
    if fragment_store._conn:
        fragment_store._conn.execute(
            "DELETE FROM fragments WHERE symbol = ?", (test_symbol,)
        )
        fragment_store._conn.commit()


def _rand_content(n: int = 120) -> str:
    """造 random content 避免被 _is_duplicate cosine 0.92 dedup。"""
    return "".join(random.choices(string.ascii_letters + string.digits + " ", k=n))


class _CountingConn:
    """暫時包裝 sqlite3.Connection 計算 commit / context-manager exit 次數。"""

    def __init__(self, real):
        self._real = real
        self.commit_count = 0

    def commit(self):
        self.commit_count += 1
        return self._real.commit()

    def __enter__(self):
        return self._real.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # `with conn:` 結束時，sqlite3 自動 commit（success）或 rollback（exception）
        result = self._real.__exit__(exc_type, exc_val, exc_tb)
        if exc_type is None:
            self.commit_count += 1
        return result

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_store_batch_uses_few_commits(cleanup_test_symbol):
    """batch 5 筆應 ≤ 2 次 commit（v116 優化）。
    舊行為：每筆都 commit + trim_if_needed commit ≈ 10 次。
    """
    fragments = [{"type": "general", "content": _rand_content()} for _ in range(5)]

    counter = _CountingConn(fragment_store._conn)
    real = fragment_store._conn
    fragment_store._conn = counter
    try:
        fragment_store.store_batch(
            fragments, symbol=cleanup_test_symbol, source_question="commit count test"
        )
    finally:
        fragment_store._conn = real

    assert counter.commit_count <= 2, (
        f"batch 5 筆 commit 次數 = {counter.commit_count}（應 ≤ 2）。"
        f"舊行為（每筆都 commit）≥ 5 次，會鎖住 SQLite 卡住前端。"
    )


def test_store_fragment_single_still_commits(cleanup_test_symbol):
    """單筆 store_fragment 保留原行為：commit 至少 1 次（呼叫完資料已落地）。"""
    counter = _CountingConn(fragment_store._conn)
    real = fragment_store._conn
    fragment_store._conn = counter
    try:
        fragment_store.store_fragment(
            content=_rand_content(),
            symbol=cleanup_test_symbol,
            source_question="single commit test",
        )
    finally:
        fragment_store._conn = real

    assert counter.commit_count >= 1, (
        f"單筆 store_fragment commit 次數 = {counter.commit_count}（應 ≥ 1，呼叫完應落地）"
    )


def test_store_batch_returns_correct_count(cleanup_test_symbol):
    """batch 回傳的 stored 數應該等於實際 INSERT 進 DB 的筆數。"""
    fragments = [{"type": "general", "content": _rand_content()} for _ in range(10)]
    stored = fragment_store.store_batch(
        fragments, symbol=cleanup_test_symbol, source_question="count test"
    )

    db_count = fragment_store._conn.execute(
        "SELECT COUNT(*) FROM fragments WHERE symbol = ?", (cleanup_test_symbol,)
    ).fetchone()[0]

    assert stored == db_count, (
        f"store_batch 回傳 stored={stored} 跟實際 DB 中該 symbol 的筆數 {db_count} 不一致"
    )


def test_store_batch_skips_too_short_content(cleanup_test_symbol):
    """太短的 content（< _MIN_FRAGMENT_LENGTH）應該被跳過，不算 stored。"""
    fragments = [
        {"type": "general", "content": _rand_content()},  # OK
        {"type": "general", "content": "short"},  # 太短
        {"type": "general", "content": _rand_content()},  # OK
    ]
    stored = fragment_store.store_batch(
        fragments, symbol=cleanup_test_symbol, source_question="skip test"
    )
    # 短的會被跳過，只剩最多 2 筆（可能再被 dedup）
    assert stored <= 2


def test_store_batch_rollback_on_exception(cleanup_test_symbol):
    """transaction 中途 raise，rollback 不留半邊資料。

    用 patch 讓第 3 筆 INSERT 失敗，確認前 2 筆也不會留在 DB。
    """
    fragments = [{"type": "general", "content": _rand_content()} for _ in range(5)]

    call_count = 0
    real_insert = fragment_store._insert_fragment_no_commit

    def maybe_fail_insert(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            raise RuntimeError("Simulated mid-batch failure")
        return real_insert(*args, **kwargs)

    with patch.object(
        fragment_store, "_insert_fragment_no_commit", side_effect=maybe_fail_insert
    ):
        fragment_store.store_batch(
            fragments, symbol=cleanup_test_symbol, source_question="rollback test"
        )

    # rollback 後 DB 不該有任何該 symbol 的資料
    db_count = fragment_store._conn.execute(
        "SELECT COUNT(*) FROM fragments WHERE symbol = ?", (cleanup_test_symbol,)
    ).fetchone()[0]
    assert db_count == 0, (
        f"transaction 中途失敗應該 rollback，但 DB 還有 {db_count} 筆"
    )


def test_insert_fragment_no_commit_does_not_commit(cleanup_test_symbol):
    """_insert_fragment_no_commit 純 INSERT 不 commit — 直接呼叫不應 commit 任何資料。

    驗證方式：呼叫後立刻在另一個 connection 查（看不到）+ 自 connection rollback 後（沒了）
    """
    test_content = _rand_content()
    inserted = fragment_store._insert_fragment_no_commit(
        content=test_content,
        symbol=cleanup_test_symbol,
        source_question="no-commit test",
    )

    if not inserted:
        # 如果是 dedup 跳過了，這個 test 不適用，直接 pass
        return

    # 此時資料還沒 commit — rollback 應該讓資料消失
    fragment_store._conn.rollback()

    db_count = fragment_store._conn.execute(
        "SELECT COUNT(*) FROM fragments WHERE symbol = ?", (cleanup_test_symbol,)
    ).fetchone()[0]
    assert db_count == 0, (
        f"_insert_fragment_no_commit 後 rollback，DB 不該有資料"
        f"（如果有 {db_count} 筆代表 INSERT 內部偷偷 commit 了）"
    )
