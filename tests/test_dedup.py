"""SemanticDedup strategy and Embedder protocol."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import pytest

from convopack import Message, Packer, SemanticDedup, TextBlock
from convopack.embedders.base import cosine


class CharBigramEmbedder:
    """Deterministic, dependency-free embedder for tests.

    Maps text to a normalised bigram-frequency vector over [a-z0-9 ]. Two
    identical strings yield cosine 1.0; very different strings yield ~0.
    """

    _VOCAB = "abcdefghijklmnopqrstuvwxyz0123456789 "

    def embed(self, text: str) -> list[float]:
        text = text.lower()
        bigrams = [text[i : i + 2] for i in range(len(text) - 1)]
        counts: Counter[str] = Counter(bigrams)
        vec: list[float] = []
        for a in self._VOCAB:
            for b in self._VOCAB:
                vec.append(float(counts.get(a + b, 0)))
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec] if norm else vec

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def test_cosine_identical() -> None:
    a = [1.0, 0.0, 0.0]
    assert cosine(a, a) == pytest.approx(1.0)


def test_cosine_orthogonal() -> None:
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_zero_vector_returns_zero() -> None:
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        cosine([1.0], [1.0, 1.0])


def test_threshold_must_be_in_range() -> None:
    with pytest.raises(ValueError, match="threshold"):
        SemanticDedup(CharBigramEmbedder(), threshold=0.0)
    with pytest.raises(ValueError, match="threshold"):
        SemanticDedup(CharBigramEmbedder(), threshold=1.5)


def test_dedup_drops_exact_duplicate() -> None:
    msgs = [
        Message(role="user", content=[TextBlock(text="hello world")]),
        Message(role="assistant", content=[TextBlock(text="hi there")]),
        Message(role="user", content=[TextBlock(text="hello world")]),
    ]
    packer = Packer(
        budget=10_000,
        tokenizer="approx",
        strategy=SemanticDedup(CharBigramEmbedder(), threshold=0.99),
        pin=(),
    )
    result = packer.pack(msgs)
    kept_texts = [m.text() for m in result.kept]
    assert kept_texts.count("hello world") == 1


def test_dedup_keeps_distinct() -> None:
    msgs = [
        Message(role="user", content=[TextBlock(text="weather in oslo")]),
        Message(role="assistant", content=[TextBlock(text="quantum entanglement explained")]),
        Message(role="user", content=[TextBlock(text="best norwegian cheeses")]),
    ]
    packer = Packer(
        budget=10_000,
        tokenizer="approx",
        strategy=SemanticDedup(CharBigramEmbedder(), threshold=0.9),
        pin=(),
    )
    result = packer.pack(msgs)
    assert len(result.kept) == len(msgs)


def test_dedup_preserves_tool_call_chunks() -> None:
    from convopack import ToolResultBlock, ToolUseBlock

    msgs = [
        Message(role="user", content=[TextBlock(text="weather")]),
        Message(
            role="assistant",
            content=[ToolUseBlock(id="t1", name="weather", input={"city": "oslo"})],
        ),
        Message(role="tool", content=[ToolResultBlock(tool_use_id="t1", content="rainy")]),
    ]
    packer = Packer(
        budget=10_000,
        tokenizer="approx",
        strategy=SemanticDedup(CharBigramEmbedder()),
        pin=(),
    )
    result = packer.pack(msgs)
    assert len(result.kept) == 3


def test_dedup_respects_budget_via_fallback() -> None:
    msgs = [
        Message(role="user", content=[TextBlock(text="x" * 80)]),
        Message(role="user", content=[TextBlock(text="y" * 80)]),
        Message(role="user", content=[TextBlock(text="z" * 80)]),
    ]
    packer = Packer(
        budget=30,
        tokenizer="approx",
        strategy=SemanticDedup(CharBigramEmbedder()),
        pin=(),
    )
    result = packer.pack(msgs)
    assert len(result.kept) < len(msgs)
