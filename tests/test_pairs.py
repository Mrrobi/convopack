"""tool_use / tool_result pair grouping."""

from __future__ import annotations

from convopack import Message, TextBlock, ToolResultBlock, ToolUseBlock
from convopack._pairs import group_pairs, validate_pairs


def test_no_tools_yields_per_message_chunks() -> None:
    msgs = [
        Message(role="user", content=[TextBlock(text="hi")]),
        Message(role="assistant", content=[TextBlock(text="hello")]),
    ]
    chunks = group_pairs(msgs)
    assert len(chunks) == 2
    assert [len(c.messages) for c in chunks] == [1, 1]


def test_paired_messages_form_one_chunk() -> None:
    msgs = [
        Message(role="assistant", content=[ToolUseBlock(id="t1", name="x", input={})]),
        Message(role="tool", content=[ToolResultBlock(tool_use_id="t1", content="ok")]),
    ]
    chunks = group_pairs(msgs)
    assert len(chunks) == 1
    assert len(chunks[0].messages) == 2


def test_dangling_tool_use_detected() -> None:
    msgs = [
        Message(role="assistant", content=[ToolUseBlock(id="t1", name="x", input={})]),
    ]
    assert validate_pairs(msgs) == ["t1"]


def test_satisfied_pair_has_no_dangling() -> None:
    msgs = [
        Message(role="assistant", content=[ToolUseBlock(id="t1", name="x", input={})]),
        Message(role="tool", content=[ToolResultBlock(tool_use_id="t1", content="ok")]),
    ]
    assert validate_pairs(msgs) == []


def test_parallel_tool_calls_in_one_chunk() -> None:
    msgs = [
        Message(
            role="assistant",
            content=[
                ToolUseBlock(id="t1", name="x", input={}),
                ToolUseBlock(id="t2", name="y", input={}),
            ],
        ),
        Message(role="tool", content=[ToolResultBlock(tool_use_id="t1", content="a")]),
        Message(role="tool", content=[ToolResultBlock(tool_use_id="t2", content="b")]),
    ]
    chunks = group_pairs(msgs)
    assert len(chunks) == 1
    assert len(chunks[0].messages) == 3
