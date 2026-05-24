"""HFTokenizerAdapter integration. Skips if transformers not installed."""

from __future__ import annotations

import pytest

from convopack import Message, Packer, Recency, TextBlock

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def small_tokenizer_model() -> str:
    """Use a tiny tokenizer that loads with just `transformers` (no sentencepiece)."""
    return "gpt2"


def test_hf_counts_text(small_tokenizer_model: str) -> None:
    from convopack.tokenizers.huggingface_adapter import HFTokenizerAdapter

    tok = HFTokenizerAdapter(small_tokenizer_model)
    assert tok.count("") == 0
    assert tok.count("hello world") > 0


def test_hf_name_format(small_tokenizer_model: str) -> None:
    from convopack.tokenizers.huggingface_adapter import HFTokenizerAdapter

    tok = HFTokenizerAdapter(small_tokenizer_model)
    assert tok.name == f"huggingface:{small_tokenizer_model}"


def test_packer_with_hf_spec(small_tokenizer_model: str) -> None:
    packer = Packer(
        budget=10_000,
        tokenizer=f"huggingface:{small_tokenizer_model}",
        strategy=Recency(),
    )
    msgs = [Message(role="user", content=[TextBlock(text="hi")])]
    result = packer.pack(msgs)
    assert result.fits
