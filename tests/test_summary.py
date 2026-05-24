"""SummaryEvict strategy."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from convopack import Message, Packer, SummaryEvict

if TYPE_CHECKING:
    pass


def _fake_summarizer(messages: list[Message]) -> str:
    return f"({len(messages)} earlier turns: {messages[0].text()[:20]}...)"


async def _async_fake_summarizer(messages: list[Message]) -> str:
    await asyncio.sleep(0)
    return f"async-({len(messages)} turns)"


def test_no_summary_when_under_budget(simple_history: list[Message]) -> None:
    packer = Packer(
        budget=10_000,
        tokenizer="approx",
        strategy=SummaryEvict(_fake_summarizer),
    )
    result = packer.pack(simple_history)
    assert result.summary is None
    assert result.dropped == []


def test_summary_inserted_when_over_budget(simple_history: list[Message]) -> None:
    packer = Packer(
        budget=30,
        tokenizer="approx",
        strategy=SummaryEvict(_fake_summarizer),
    )
    result = packer.pack(simple_history)
    if result.dropped:
        assert result.summary is not None
        assert result.kept[0] is result.summary
        assert "earlier turns" in result.summary.text()


def test_sync_summarizer_returns_immediately(simple_history: list[Message]) -> None:
    s = SummaryEvict(_fake_summarizer)
    result = s.pack(
        simple_history,
        budget=20,
        tokenizer=Packer(budget=1, tokenizer="approx").tokenizer,
        pinned_indices=set(),
    )
    assert isinstance(result.token_count, int)


async def test_async_summarizer_via_pack_async(simple_history: list[Message]) -> None:
    packer = Packer(
        budget=20,
        tokenizer="approx",
        strategy=SummaryEvict(_async_fake_summarizer),
    )
    with pytest.raises(RuntimeError, match="async summariser"):
        packer.pack(simple_history)
