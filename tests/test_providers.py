"""Provider message-shape adapters: round-trip and feature coverage."""

from __future__ import annotations

import json

from convopack import Message, TextBlock, ToolResultBlock, ToolUseBlock
from convopack.providers import from_anthropic, from_openai, to_anthropic, to_openai


def test_openai_text_roundtrip() -> None:
    raw = [
        {"role": "system", "content": "hi"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    msgs = from_openai(raw)
    out = to_openai(msgs)
    assert out[0]["role"] == "system"
    assert out[1]["content"] == "hello"
    assert out[2]["content"] == "world"


def test_openai_tool_call_roundtrip() -> None:
    raw = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city":"oslo"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "rainy"},
    ]
    msgs = from_openai(raw)
    assert msgs[0].has_tool_use()
    assert msgs[1].has_tool_result()
    out = to_openai(msgs)
    assert out[0]["tool_calls"][0]["function"]["name"] == "weather"
    assert json.loads(out[0]["tool_calls"][0]["function"]["arguments"]) == {"city": "oslo"}
    assert out[1]["tool_call_id"] == "call_1"


def test_anthropic_system_extracted_to_payload() -> None:
    msgs = [
        Message(role="system", content=[TextBlock(text="be brief")]),
        Message(role="user", content=[TextBlock(text="hi")]),
    ]
    payload = to_anthropic(msgs)
    assert payload.system == "be brief"
    assert payload.messages[0]["role"] == "user"


def test_anthropic_tool_use_roundtrip() -> None:
    raw = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "weather", "input": {"city": "oslo"}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "rainy"}],
        },
    ]
    msgs = from_anthropic(raw, system="be brief")
    assert msgs[0].role == "system"
    assert any(isinstance(b, ToolUseBlock) for b in msgs[1].content)
    assert any(isinstance(b, ToolResultBlock) for b in msgs[2].content)
    payload = to_anthropic(msgs)
    assert payload.system == "be brief"
    assert payload.messages[0]["content"][0]["type"] == "tool_use"
    assert payload.messages[1]["content"][0]["type"] == "tool_result"
