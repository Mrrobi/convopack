"""Gemini provider adapter."""

from __future__ import annotations

from convopack import ImageBlock, Message, Packer, TextBlock, ToolResultBlock, ToolUseBlock
from convopack.providers import from_gemini, to_gemini


def test_text_roundtrip() -> None:
    raw = [
        {"role": "user", "parts": [{"text": "hello"}]},
        {"role": "model", "parts": [{"text": "hi"}]},
    ]
    msgs = from_gemini(raw)
    assert msgs[0].role == "user"
    assert msgs[1].role == "assistant"
    payload = to_gemini(msgs)
    assert payload.contents[0]["role"] == "user"
    assert payload.contents[1]["role"] == "model"


def test_system_instruction_extracted() -> None:
    msgs = [
        Message(role="system", content=[TextBlock(text="be brief")]),
        Message(role="user", content=[TextBlock(text="hi")]),
    ]
    payload = to_gemini(msgs)
    assert payload.system_instruction == "be brief"
    assert payload.contents[0]["role"] == "user"


def test_function_call_roundtrip() -> None:
    raw = [
        {"role": "user", "parts": [{"text": "weather?"}]},
        {
            "role": "model",
            "parts": [{"function_call": {"name": "weather", "args": {"city": "oslo"}}}],
        },
        {
            "role": "user",
            "parts": [{"function_response": {"name": "weather", "response": {"content": "rainy"}}}],
        },
    ]
    msgs = from_gemini(raw)
    assert any(isinstance(b, ToolUseBlock) for b in msgs[1].content)
    assert any(isinstance(b, ToolResultBlock) for b in msgs[2].content)
    payload = to_gemini(msgs)
    fc = payload.contents[1]["parts"][0]["function_call"]
    fr = payload.contents[2]["parts"][0]["function_response"]
    assert fc["name"] == "weather"
    assert fc["args"] == {"city": "oslo"}
    assert fr["name"] == "weather"


def test_image_part_roundtrip() -> None:
    raw = [
        {
            "role": "user",
            "parts": [
                {"text": "describe"},
                {"inline_data": {"mime_type": "image/png", "data": "abc123"}},
            ],
        }
    ]
    msgs = from_gemini(raw)
    assert any(isinstance(b, ImageBlock) for b in msgs[0].content)
    payload = to_gemini(msgs)
    parts = payload.contents[0]["parts"]
    assert any("inline_data" in p for p in parts)


def test_pack_gemini_smokes() -> None:
    raw = [
        {"role": "user", "parts": [{"text": "what is rain"}]},
        {"role": "model", "parts": [{"text": "water from sky"}]},
        {"role": "user", "parts": [{"text": "and snow"}]},
    ]
    packer = Packer(budget=10_000, tokenizer="approx")
    payload = packer.pack_gemini(raw, system_instruction="be helpful")
    assert payload.system_instruction == "be helpful"
    assert len(payload.contents) == 3
