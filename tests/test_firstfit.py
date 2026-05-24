"""FirstFit strategy: keep oldest chunks that fit."""

from __future__ import annotations

from convopack import FirstFit, Message, Packer, TextBlock
from convopack._pairs import validate_pairs


def test_keeps_oldest_when_over_budget(simple_history: list[Message]) -> None:
    packer = Packer(budget=30, tokenizer="approx", strategy=FirstFit())
    result = packer.pack(simple_history)
    assert simple_history[0] in result.kept


def test_drops_tail_first() -> None:
    msgs = [Message(role="user", content=[TextBlock(text=f"turn {i} " * 20)]) for i in range(5)]
    packer = Packer(budget=40, tokenizer="approx", strategy=FirstFit(), pin=())
    result = packer.pack(msgs)
    if result.dropped:
        kept_idx = [msgs.index(m) for m in result.kept]
        dropped_idx = [msgs.index(m) for m in result.dropped]
        assert max(kept_idx) < min(dropped_idx)


def test_under_budget_keeps_all(simple_history: list[Message]) -> None:
    packer = Packer(budget=10_000, tokenizer="approx", strategy=FirstFit())
    result = packer.pack(simple_history)
    assert result.dropped == []


def test_tool_pair_preserved(tool_history: list[Message]) -> None:
    packer = Packer(budget=40, tokenizer="approx", strategy=FirstFit())
    result = packer.pack(tool_history)
    assert validate_pairs(result.kept) == []


def test_empty_input_returns_empty() -> None:
    packer = Packer(budget=100, tokenizer="approx", strategy=FirstFit())
    result = packer.pack([])
    assert result.kept == []
    assert result.dropped == []
