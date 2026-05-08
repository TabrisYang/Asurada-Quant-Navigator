"""v115 part 1：chat_history 增量更新 API（producer/consumer 解耦的基礎）。

新增 API：
- create_streaming_message：建立 status='streaming' 訊息，回傳 message_id
- append_message_content：增量 append（producer 用）
- update_message_status：streaming → completed / error
- get_message_content_after：tail 讀指定 offset 後的 content + status（consumer 用）
- get_conversation_messages：回傳值加 id + status（讓 frontend 重整能看到 streaming 中的訊息）

Schema migration：messages 表加 `status` 欄位（既有訊息 default 'completed'）。
"""

import uuid

import pytest

from app.core.chat_history import chat_history


@pytest.fixture
def conv_id():
    """每個 test 用獨立 conversation_id，跑完清掉避免汙染。"""
    cid = f"test_v115_{uuid.uuid4().hex[:8]}"
    yield cid
    chat_history.delete_conversation(cid)


def test_create_streaming_message_returns_valid_id(conv_id):
    mid = chat_history.create_streaming_message(conv_id, role="assistant")
    assert mid is not None and isinstance(mid, int) and mid > 0


def test_streaming_message_starts_with_streaming_status(conv_id):
    mid = chat_history.create_streaming_message(conv_id, role="assistant")
    _, status = chat_history.get_message_content_after(mid, offset=0)
    assert status == "streaming", f"新訊息 status 應為 'streaming'，實際 {status!r}"


def test_append_message_content_accumulates(conv_id):
    mid = chat_history.create_streaming_message(conv_id, role="assistant")
    chat_history.append_message_content(mid, "Hello ")
    chat_history.append_message_content(mid, "World")
    content, _ = chat_history.get_message_content_after(mid, offset=0)
    assert content == "Hello World"


def test_get_message_content_after_offset_returns_only_new(conv_id):
    """模擬 consumer tail：每次帶 offset 只拿到新增部分。"""
    mid = chat_history.create_streaming_message(conv_id, role="assistant")
    chat_history.append_message_content(mid, "ABCDE")  # offset 0-4
    chat_history.append_message_content(mid, "FGH")    # offset 5-7

    full, _ = chat_history.get_message_content_after(mid, offset=0)
    assert full == "ABCDEFGH"

    new_only, _ = chat_history.get_message_content_after(mid, offset=5)
    assert new_only == "FGH", "offset=5 應該只拿 'FGH'"

    nothing_new, _ = chat_history.get_message_content_after(mid, offset=8)
    assert nothing_new == "", "offset 等於 content 長度應回空字串（不是 error）"


def test_update_message_status_lifecycle(conv_id):
    """status 從 streaming → completed（正常路徑）"""
    mid = chat_history.create_streaming_message(conv_id, role="assistant")
    chat_history.append_message_content(mid, "partial content")

    _, status_before = chat_history.get_message_content_after(mid, offset=0)
    assert status_before == "streaming"

    chat_history.update_message_status(mid, "completed")
    _, status_after = chat_history.get_message_content_after(mid, offset=0)
    assert status_after == "completed"


def test_update_message_status_with_final_content_overwrites(conv_id):
    """update_message_status 帶 final_content 應 overwrite content（producer 結束時清理用）。"""
    mid = chat_history.create_streaming_message(conv_id, role="assistant")
    chat_history.append_message_content(mid, "raw text with ---KEY_INSIGHTS--- markup")

    chat_history.update_message_status(
        mid, "completed", final_content="cleaned text"
    )
    content, status = chat_history.get_message_content_after(mid, offset=0)
    assert content == "cleaned text"
    assert status == "completed"


def test_update_message_status_error_path(conv_id):
    """status 'error' 也合法（producer 異常時用）"""
    mid = chat_history.create_streaming_message(conv_id, role="assistant")
    chat_history.update_message_status(mid, "error")
    _, status = chat_history.get_message_content_after(mid, offset=0)
    assert status == "error"


def test_get_conversation_messages_returns_id_and_status(conv_id):
    """get_conversation_messages 回傳值必須含 id + status（讓 frontend 重整能看 streaming 訊息）。"""
    mid = chat_history.create_streaming_message(conv_id, role="assistant")
    chat_history.append_message_content(mid, "test")

    messages = chat_history.get_conversation_messages(conv_id)
    assert len(messages) >= 1
    msg = next(m for m in messages if m.get("id") == mid)
    assert msg["id"] == mid
    assert msg["status"] == "streaming"
    assert msg["content"] == "test"


def test_message_not_found_returns_error_status(conv_id):
    """get_message_content_after 找不到 message 時回 ('', 'error')，不該 raise。"""
    content, status = chat_history.get_message_content_after(message_id=99999999, offset=0)
    assert content == ""
    assert status == "error"


def test_existing_save_message_defaults_to_completed_status(conv_id):
    """既有 save_message 流程不寫 status，schema migration 後 default 'completed'。"""
    chat_history.save_message(
        conversation_id=conv_id, role="user", content="hi",
    )
    messages = chat_history.get_conversation_messages(conv_id)
    assert len(messages) == 1
    assert messages[0]["status"] == "completed", (
        "既有 save_message 寫入的訊息應預設 status='completed'，避免被 consumer 誤判為 streaming"
    )
