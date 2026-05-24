"""ApproxTokenizer behaviour."""

from __future__ import annotations

import pytest

from convopack import ApproxTokenizer, Message, TextBlock, ToolResultBlock, ToolUseBlock


def test_empty_text_is_zero() -> None:
    tok = ApproxTokenizer()
    assert tok.count("") == 0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a", 1),
        ("abcd", 1),
        ("abcde", 2),
        ("a" * 8, 2),
        ("a" * 9, 3),
    ],
)
def test_text_counts(text: str, expected: int) -> None:
    assert ApproxTokenizer().count(text) == expected


def test_message_overhead_added() -> None:
    tok = ApproxTokenizer()
    m = Message(role="user", content=[TextBlock(text="abcd")])
    assert tok.count_message(m) > tok.count("abcd")


def test_tool_blocks_count(approx: ApproxTokenizer) -> None:
    m = Message(
        role="assistant",
        content=[ToolUseBlock(id="t1", name="weather", input={"city": "oslo"})],
    )
    n = approx.count_message(m)
    assert n > 0


def test_tool_result_string_counts(approx: ApproxTokenizer) -> None:
    m = Message(role="tool", content=[ToolResultBlock(tool_use_id="t1", content="rainy, 8C")])
    assert approx.count_message(m) > 0
