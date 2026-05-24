"""High-level :class:`Packer` facade."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

from convopack._types import Message, PackResult
from convopack.events import PackEvent
from convopack.providers.anthropic import (
    AnthropicPayload,
    from_anthropic,
    to_anthropic,
)
from convopack.providers.gemini import GeminiPayload, from_gemini, to_gemini
from convopack.providers.openai import from_openai, to_openai
from convopack.strategies import Recency, Strategy
from convopack.tokenizers import Tokenizer, get_tokenizer

if TYPE_CHECKING:
    from convopack.strategies.base import PinSpec


class Packer:
    """Coordinates a tokenizer, a strategy, and pinning rules.

    Example
    -------
    >>> from convopack import Packer, Recency
    >>> packer = Packer(budget=4000, tokenizer="approx", strategy=Recency())
    >>> packed = packer.pack(messages)
    """

    def __init__(
        self,
        *,
        budget: int,
        tokenizer: str | Tokenizer = "approx",
        strategy: Strategy | None = None,
        pin: Sequence[PinSpec] = ("system",),
    ) -> None:
        if budget <= 0:
            raise ValueError("budget must be positive")
        self.budget = budget
        self.tokenizer = get_tokenizer(tokenizer)
        self.strategy = strategy or Recency()
        self.pin = tuple(pin)

    def pack(self, messages: Iterable[Message]) -> PackResult:
        """Pack a sequence of internal :class:`Message` objects."""
        msgs = list(messages)
        pinned = self._resolve_pinned(msgs)
        return self.strategy.pack(
            msgs, budget=self.budget, tokenizer=self.tokenizer, pinned_indices=pinned
        )

    async def pack_async(self, messages: Iterable[Message]) -> PackResult:
        """Async variant. Runs sync strategies in a thread; awaits async summarisers."""
        msgs = list(messages)
        pinned = self._resolve_pinned(msgs)

        strategy = self.strategy
        if hasattr(strategy, "pack_async"):
            return await strategy.pack_async(  # type: ignore[no-any-return]
                msgs,
                budget=self.budget,
                tokenizer=self.tokenizer,
                pinned_indices=pinned,
            )
        return await asyncio.to_thread(
            strategy.pack,
            msgs,
            budget=self.budget,
            tokenizer=self.tokenizer,
            pinned_indices=pinned,
        )

    def pack_stream(self, messages: Iterable[Message]) -> Iterator[PackEvent]:
        """Yield :class:`PackEvent` items reflecting the strategy's decisions.

        Useful for progress bars, audit logs, or piping pack telemetry to a
        debugger. The terminal ``done`` event carries the total token count.
        """
        msgs = list(messages)
        result = self.pack(msgs)
        positions = {id(m): i for i, m in enumerate(msgs)}
        summary_id = id(result.summary) if result.summary is not None else None
        for m in result.kept:
            cost = self.tokenizer.count_message(m)
            if summary_id is not None and id(m) == summary_id:
                yield PackEvent(kind="summarized", index=-1, message=m, token_cost=cost)
            else:
                yield PackEvent(
                    kind="kept",
                    index=positions.get(id(m), -1),
                    message=m,
                    token_cost=cost,
                )
        for m in result.dropped:
            cost = self.tokenizer.count_message(m)
            yield PackEvent(
                kind="dropped",
                index=positions.get(id(m), -1),
                message=m,
                token_cost=cost,
            )
        yield PackEvent(kind="done", index=-1, message=None, token_cost=result.token_count)

    def pack_openai(self, raw: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convenience: accept OpenAI Chat dicts, return packed OpenAI Chat dicts."""
        return to_openai(self.pack(from_openai(raw)).kept)

    def pack_anthropic(
        self, raw: Iterable[dict[str, Any]], *, system: str | None = None
    ) -> AnthropicPayload:
        """Convenience: accept Anthropic Messages dicts, return a packed payload."""
        result = self.pack(from_anthropic(raw, system=system))
        return to_anthropic(result.kept)

    def pack_gemini(
        self,
        raw: Iterable[dict[str, Any]],
        *,
        system_instruction: str | None = None,
    ) -> GeminiPayload:
        """Convenience: accept Gemini ``Content`` dicts, return a packed payload."""
        result = self.pack(from_gemini(raw, system_instruction=system_instruction))
        return to_gemini(result.kept)

    def _resolve_pinned(self, msgs: list[Message]) -> set[int]:
        if not msgs:
            return set()
        pinned: set[int] = set()
        first_user: int | None = None
        last_user: int | None = None
        for idx, msg in enumerate(msgs):
            if msg.role == "user":
                if first_user is None:
                    first_user = idx
                last_user = idx
        for spec in self.pin:
            if isinstance(spec, int):
                if 0 <= spec < len(msgs) or -len(msgs) <= spec < 0:
                    pinned.add(spec % len(msgs))
                continue
            if spec == "system":
                for idx, msg in enumerate(msgs):
                    if msg.role == "system":
                        pinned.add(idx)
            elif spec == "first_user" and first_user is not None:
                pinned.add(first_user)
            elif spec == "last_user" and last_user is not None:
                pinned.add(last_user)
            elif spec == "tool_results":
                for idx, msg in enumerate(msgs):
                    if msg.has_tool_use() or msg.has_tool_result():
                        pinned.add(idx)
        return pinned
