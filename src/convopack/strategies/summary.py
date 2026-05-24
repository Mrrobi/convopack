"""SummaryEvict strategy: keep recent + a summary of evicted older turns."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, cast

from convopack._types import Message, PackResult, TextBlock
from convopack.strategies.recency import Recency

if TYPE_CHECKING:
    from convopack.tokenizers.base import Tokenizer


SummarizerSync = Callable[[list[Message]], str]
SummarizerAsync = Callable[[list[Message]], Awaitable[str]]
Summarizer = SummarizerSync | SummarizerAsync


class SummaryEvict:
    """Drop oldest chunks like :class:`Recency`, but replace them with a summary message.

    The summariser is any callable that takes a list of messages and returns a
    string (sync or async). The summary is inserted as a system message at the
    head of the kept list, with the prefix ``[summary]`` to make it
    identifiable.
    """

    name = "summary_evict"

    def __init__(
        self,
        summarizer: Summarizer,
        *,
        prefix: str = "[summary] ",
        reserve: int | None = None,
    ) -> None:
        self._summarizer = summarizer
        self._prefix = prefix
        self._reserve = reserve

    def pack(
        self,
        messages: list[Message],
        *,
        budget: int,
        tokenizer: Tokenizer,
        pinned_indices: set[int],
    ) -> PackResult:
        reserve = self._reserve if self._reserve is not None else max(budget // 10, 128)
        inner_budget = max(budget - reserve, 1)
        inner = Recency().pack(
            messages,
            budget=inner_budget,
            tokenizer=tokenizer,
            pinned_indices=pinned_indices,
        )

        if not inner.dropped:
            return PackResult(
                kept=inner.kept,
                dropped=inner.dropped,
                summary=None,
                token_count=inner.token_count,
                budget=budget,
            )

        summary_text = self._run_summarizer(inner.dropped)
        summary_msg = Message(
            role="system",
            content=[TextBlock(text=f"{self._prefix}{summary_text}")],
            metadata={"convopack.summary": True, "dropped_count": len(inner.dropped)},
        )
        summary_cost = tokenizer.count_message(summary_msg)
        kept = [summary_msg, *inner.kept]
        return PackResult(
            kept=kept,
            dropped=inner.dropped,
            summary=summary_msg,
            token_count=inner.token_count + summary_cost,
            budget=budget,
        )

    def _run_summarizer(self, messages: list[Message]) -> str:
        result = self._summarizer(messages)
        if asyncio.iscoroutine(result):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return cast(str, asyncio.run(result))
            result.close()
            raise RuntimeError(
                "SummaryEvict.pack() called from inside a running event loop with an "
                "async summariser. Use `await packer.pack_async(...)` instead."
            )
        return cast(str, result)
