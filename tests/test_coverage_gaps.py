"""Targeted tests filling coverage gaps in tokenizers and provider adapters."""

from __future__ import annotations

import pytest

from convopack import ApproxTokenizer, ImageBlock, Message
from convopack.providers import from_anthropic, from_openai, to_anthropic, to_openai
from convopack.tokenizers import get_tokenizer
from convopack.tokenizers.anthropic_adapter import AnthropicAdapter


def test_get_tokenizer_approx_string() -> None:
    tok = get_tokenizer("approx")
    assert isinstance(tok, ApproxTokenizer)


def test_get_tokenizer_passthrough() -> None:
    existing = ApproxTokenizer()
    assert get_tokenizer(existing) is existing


def test_get_tokenizer_anthropic_spec() -> None:
    tok = get_tokenizer("anthropic:claude-haiku-4-5")
    assert tok.name == "anthropic:claude-haiku-4-5"
    assert tok.count("hello") > 0


def test_get_tokenizer_unknown_spec_raises() -> None:
    with pytest.raises(ValueError, match="Unknown tokenizer spec"):
        get_tokenizer("nonsense:thing")


def test_anthropic_adapter_offline_counts() -> None:
    adapter = AnthropicAdapter("claude-haiku-4-5", offline=True)
    assert adapter.count("hello") > 0
    msg = Message(role="user", content=[])
    assert adapter.count_message(msg) > 0
    assert adapter.count_messages([msg, msg]) >= 2 * adapter.count_message(msg) - 1


def test_anthropic_adapter_name() -> None:
    adapter = AnthropicAdapter("claude-sonnet-4-6")
    assert adapter.name == "anthropic:claude-sonnet-4-6"


def test_openai_image_block_roundtrip() -> None:
    raw = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
            ],
        }
    ]
    msgs = from_openai(raw)
    assert any(isinstance(b, ImageBlock) for b in msgs[0].content)
    out = to_openai(msgs)
    assert isinstance(out[0]["content"], list)


def test_openai_name_field_preserved() -> None:
    raw = [{"role": "user", "content": "hi", "name": "alice"}]
    msgs = from_openai(raw)
    assert msgs[0].name == "alice"
    out = to_openai(msgs)
    assert out[0]["name"] == "alice"


def test_openai_invalid_tool_arguments_falls_back() -> None:
    raw = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "x",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{not-json"},
                }
            ],
        }
    ]
    msgs = from_openai(raw)
    tool_blocks = [b for b in msgs[0].content if hasattr(b, "input")]
    assert tool_blocks
    assert tool_blocks[0].input == {"_raw": "{not-json"}


def test_anthropic_image_block_roundtrip() -> None:
    raw = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0",
                    },
                },
                {"type": "text", "text": "describe"},
            ],
        }
    ]
    msgs = from_anthropic(raw)
    assert any(isinstance(b, ImageBlock) for b in msgs[0].content)
    payload = to_anthropic(msgs)
    assert payload.messages[0]["content"][0]["type"] == "image"


def test_anthropic_tool_result_with_inner_blocks() -> None:
    raw = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [
                        {"type": "text", "text": "ok"},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": "abc"},
                        },
                    ],
                }
            ],
        }
    ]
    msgs = from_anthropic(raw)
    payload = to_anthropic(msgs)
    inner = payload.messages[0]["content"][0]["content"]
    assert isinstance(inner, list)
    assert inner[0]["type"] == "text"


def test_approx_image_block_has_estimate() -> None:
    tok = ApproxTokenizer()
    msg = Message(role="user", content=[ImageBlock(source="x.png", media_type="image/png")])
    assert tok.count_message(msg) > 100


def test_packer_pin_out_of_range_int_ignored() -> None:
    from convopack import Packer

    packer = Packer(budget=1000, tokenizer="approx", pin=(99,))
    result = packer.pack([Message(role="user", content=[])])
    assert len(result.kept) == 1


def test_packer_pin_negative_int() -> None:
    from convopack import Message, Packer, TextBlock

    msgs = [
        Message(role="user", content=[TextBlock(text="a" * 80)]),
        Message(role="user", content=[TextBlock(text="b" * 80)]),
        Message(role="user", content=[TextBlock(text="c" * 80)]),
    ]
    packer = Packer(budget=30, tokenizer="approx", pin=(-1,))
    result = packer.pack(msgs)
    assert msgs[-1] in result.kept
