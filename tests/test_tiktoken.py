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
