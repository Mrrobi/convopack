"""Recency strategy."""

from __future__ import annotations

from convopack import Message, Packer, Recency, TextBlock
from convopack._pairs import validate_pairs


def test_under_budget_keeps_everything(simple_history: list[Message]) -> None:
    packer = Packer(budget=10_000, tokenizer="approx", strategy=Recency())
    result = packer.pack(simple_history)
    assert result.dropped == []
    assert len(result.kept) == len(simple_history)
    assert result.fits


def test_tiny_budget_keeps_pinned_system_and_min(simple_history: list[Message]) -> None:
    packer = Packer(budget=1, tokenizer="approx", strategy=Recency(min_keep=1))
    result = packer.pack(simple_history)
    assert any(m.role == "system" for m in result.kept)
    assert any(m.role != "system" for m in result.kept)


def test_recency_keeps_tail(simple_history: list[Message]) -> None:
    packer = Packer(budget=30, tokenizer="approx", strategy=Recency())
    result = packer.pack(simple_history)
    assert simple_history[-1] in result.kept


def test_tool_pair_not_split(tool_history: list[Message]) -> None:
    packer = Packer(budget=40, tokenizer="approx", strategy=Recency())
    result = packer.pack(tool_history)
    assert validate_pairs(result.kept) == []


def test_pinned_first_user_kept(simple_history: list[Message]) -> None:
    packer = Packer(
        budget=20,
        tokenizer="approx",
        strategy=Recency(),
        pin=("system", "first_user"),
    )
    result = packer.pack(simple_history)
    first_user_text = "hello, my name is robi"
    assert any(m.text() == first_user_text for m in result.kept)


def test_kept_preserves_order(simple_history: list[Message]) -> None:
    packer = Packer(budget=10_000, tokenizer="approx", strategy=Recency())
    result = packer.pack(simple_history)
    indices = [simple_history.index(m) for m in result.kept]
    assert indices == sorted(indices)


def test_empty_input() -> None:
    packer = Packer(budget=100, tokenizer="approx", strategy=Recency())
    result = packer.pack([])
    assert result.kept == []
    assert result.dropped == []
    assert result.token_count == 0


def test_summary_field_is_none_for_recency() -> None:
    packer = Packer(budget=100, tokenizer="approx", strategy=Recency())
    result = packer.pack([Message(role="user", content=[TextBlock(text="hi")])])
    assert result.summary is None
