"""Content hash on Message + history_hash."""

from __future__ import annotations

from convopack import (
    ImageBlock,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    history_hash,
)


def test_identical_text_messages_hash_same() -> None:
    a = Message(role="user", content=[TextBlock(text="hello")])
    b = Message(role="user", content=[TextBlock(text="hello")])
    assert a.content_hash == b.content_hash


def test_different_text_hashes_differ() -> None:
    a = Message(role="user", content=[TextBlock(text="hello")])
    b = Message(role="user", content=[TextBlock(text="world")])
    assert a.content_hash != b.content_hash


def test_role_change_changes_hash() -> None:
    a = Message(role="user", content=[TextBlock(text="x")])
    b = Message(role="assistant", content=[TextBlock(text="x")])
    assert a.content_hash != b.content_hash


def test_tool_use_id_ignored_by_hash() -> None:
    """tool_use_id is runtime metadata; identical name + input -> same hash."""
    a = Message(
        role="assistant",
        content=[ToolUseBlock(id="t1", name="weather", input={"city": "oslo"})],
    )
    b = Message(
        role="assistant",
        content=[ToolUseBlock(id="t999", name="weather", input={"city": "oslo"})],
    )
    assert a.content_hash == b.content_hash


def test_tool_use_input_difference_changes_hash() -> None:
    a = Message(
        role="assistant",
        content=[ToolUseBlock(id="t1", name="weather", input={"city": "oslo"})],
    )
    b = Message(
        role="assistant",
        content=[ToolUseBlock(id="t1", name="weather", input={"city": "bergen"})],
    )
    assert a.content_hash != b.content_hash


def test_image_block_hash_includes_source_and_media_type() -> None:
    a = Message(role="user", content=[ImageBlock(source="abc", media_type="image/png")])
    b = Message(role="user", content=[ImageBlock(source="abc", media_type="image/jpeg")])
    c = Message(role="user", content=[ImageBlock(source="xyz", media_type="image/png")])
    assert a.content_hash != b.content_hash
    assert a.content_hash != c.content_hash


def test_tool_result_string_and_list_distinguishable() -> None:
    a = Message(role="tool", content=[ToolResultBlock(tool_use_id="t1", content="ok")])
    b = Message(
        role="tool",
        content=[ToolResultBlock(tool_use_id="t1", content=[TextBlock(text="ok")])],
    )
    assert a.content_hash != b.content_hash


def test_history_hash_order_sensitive() -> None:
    m1 = Message(role="user", content=[TextBlock(text="a")])
    m2 = Message(role="user", content=[TextBlock(text="b")])
    assert history_hash([m1, m2]) != history_hash([m2, m1])


def test_history_hash_identical_inputs_match() -> None:
    msgs = [
        Message(role="system", content=[TextBlock(text="be brief")]),
        Message(role="user", content=[TextBlock(text="hi")]),
    ]
    assert history_hash(msgs) == history_hash(list(msgs))


def test_history_hash_empty_input() -> None:
    h = history_hash([])
    assert len(h) == 64  # sha256 hex


def test_hash_is_64_char_hex() -> None:
    m = Message(role="user", content=[TextBlock(text="hi")])
    h = m.content_hash
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)
