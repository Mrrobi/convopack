"""Packer facade: pinning resolution, provider convenience paths."""

from __future__ import annotations

import pytest

from convopack import Message, Packer, Recency, TextBlock


def test_packer_requires_positive_budget() -> None:
    with pytest.raises(ValueError, match="budget must be positive"):
        Packer(budget=0)


def test_pack_openai_roundtrip() -> None:
    raw = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    packed = Packer(budget=10_000, tokenizer="approx").pack_openai(raw)
    assert packed[0]["role"] == "system"
    assert packed[-1]["content"] == "hello"


def test_pack_anthropic_returns_payload() -> None:
    raw = [{"role": "user", "content": "hi"}]
    payload = Packer(budget=10_000, tokenizer="approx").pack_anthropic(raw, system="be brief")
    assert payload.system == "be brief"
    assert payload.messages[0]["role"] == "user"


def test_pin_integer_index() -> None:
    msgs = [
        Message(role="user", content=[TextBlock(text="x" * 60)]),
        Message(role="user", content=[TextBlock(text="y" * 60)]),
        Message(role="user", content=[TextBlock(text="z" * 60)]),
    ]
    packer = Packer(budget=20, tokenizer="approx", strategy=Recency(), pin=(0,))
    result = packer.pack(msgs)
    assert msgs[0] in result.kept


def test_pin_first_and_last_user() -> None:
    msgs = [
        Message(role="user", content=[TextBlock(text="first " * 50)]),
        Message(role="assistant", content=[TextBlock(text="mid")]),
        Message(role="user", content=[TextBlock(text="middle " * 50)]),
        Message(role="user", content=[TextBlock(text="last " * 50)]),
    ]
    packer = Packer(
        budget=80,
        tokenizer="approx",
        strategy=Recency(),
        pin=("first_user", "last_user"),
    )
    result = packer.pack(msgs)
    assert msgs[0] in result.kept
    assert msgs[-1] in result.kept


async def test_pack_async_runs() -> None:
    msgs = [Message(role="user", content=[TextBlock(text="hi")])]
    result = await Packer(budget=100, tokenizer="approx").pack_async(msgs)
    assert result.fits
