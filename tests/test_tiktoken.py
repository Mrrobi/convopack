"""TiktokenAdapter integration. Skips if tiktoken is not installed."""

from __future__ import annotations

import pytest

from convopack import Message, Packer, Recency, TextBlock

tiktoken = pytest.importorskip("tiktoken")


def test_tiktoken_counts_text() -> None:
    from convopack.tokenizers.tiktoken_adapter import TiktokenAdapter

    tok = TiktokenAdapter("gpt-4o")
    assert tok.count("hello world") > 0
    assert tok.count("") == 0
    assert tok.name.startswith("tiktoken:")


def test_packer_with_tiktoken() -> None:
    packer = Packer(budget=10_000, tokenizer="tiktoken:gpt-4o", strategy=Recency())
    msgs = [
        Message(role="system", content=[TextBlock(text="be brief")]),
        Message(role="user", content=[TextBlock(text="hi")]),
    ]
    result = packer.pack(msgs)
    assert result.fits
    assert len(result.kept) == 2


def test_unknown_model_falls_back_to_encoding() -> None:
    from convopack.tokenizers.tiktoken_adapter import TiktokenAdapter

    tok = TiktokenAdapter("cl100k_base")
    assert tok.count("hello") > 0


def test_tiktoken_counts_tool_use_block() -> None:
    from convopack import Message, ToolUseBlock
    from convopack.tokenizers.tiktoken_adapter import TiktokenAdapter

    tok = TiktokenAdapter("gpt-4o")
    msg = Message(
        role="assistant",
        content=[ToolUseBlock(id="call_1", name="weather", input={"city": "oslo"})],
    )
    assert tok.count_message(msg) > 0


def test_tiktoken_counts_tool_result_string_and_list() -> None:
    from convopack import Message, TextBlock, ToolResultBlock
    from convopack.tokenizers.tiktoken_adapter import TiktokenAdapter

    tok = TiktokenAdapter("gpt-4o")
    str_msg = Message(role="tool", content=[ToolResultBlock(tool_use_id="t1", content="ok")])
    list_msg = Message(
        role="tool",
        content=[ToolResultBlock(tool_use_id="t2", content=[TextBlock(text="inner")])],
    )
    assert tok.count_message(str_msg) > 0
    assert tok.count_message(list_msg) > 0


def test_tiktoken_counts_image_block() -> None:
    from convopack import ImageBlock, Message
    from convopack.tokenizers.tiktoken_adapter import TiktokenAdapter

    tok = TiktokenAdapter("gpt-4o")
    msg = Message(role="user", content=[ImageBlock(source="x.png", media_type="image/png")])
    assert tok.count_message(msg) > 100


def test_tiktoken_message_with_name() -> None:
    from convopack import Message, TextBlock
    from convopack.tokenizers.tiktoken_adapter import TiktokenAdapter

    tok = TiktokenAdapter("gpt-4o")
    msg = Message(role="user", content=[TextBlock(text="hi")], name="alice")
    assert tok.count_message(msg) > 0
