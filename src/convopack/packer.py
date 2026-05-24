"""High-level :class:`Packer` facade."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

from convopack._types import Message, PackResult
from convopack.caching import stable_marker_indices
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
        cache: bool = False,
    ) -> None:
        if budget <= 0:
            raise ValueError("budget must be positive")
        self.budget = budget
        self.tokenizer = get_tokenizer(tokenizer)
        self.strategy = strategy or Recency()
        self.pin = tuple(pin)
        self.cache = cache

    def pack(self, messages: Iterable[Message]) -> PackResult:
        """Pack a sequence of internal :class:`Message` objects."""
        msgs = list(messages)
        pinned = self._resolve_pinned(msgs)
        result = self.strategy.pack(
            msgs, budget=self.budget, tokenizer=self.tokenizer, pinned_indices=pinned
        )
        if self.cache:
            result.cache_markers = stable_marker_indices(result.kept, pin_specs=self.pin)
        return result

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
        result = await asyncio.to_thread(
            strategy.pack,
            msgs,
            budget=self.budget,
            tokenizer=self.tokenizer,
            pinned_indices=pinned,
        )
        if self.cache:
            result.cache_markers = stable_marker_indices(result.kept, pin_specs=self.pin)
        return result

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
        """Convenience: accept Anthropic Messages dicts, return a packed payload.

        Cache markers (from ``Packer(cache=True)``) are forwarded so the
        resulting payload's content blocks carry the appropriate
        ``cache_control`` markers for Anthropic prompt caching.
        """
        result = self.pack(from_anthropic(raw, system=system))
        return to_anthropic(result.kept, cache_markers=result.cache_markers)

    def pack_gemini(
        self,
        raw: Iterable[dict[str, Any]],
        *,
        system_instruction: str | None = None,
    ) -> GeminiPayload:
        """Convenience: accept Gemini ``Content`` dicts, return a packed payload."""
        result = self.pack(from_gemini(raw, system_instruction=system_instruction))
        return to_gemini(result.kept)

    def cache_prefix_signature(self, messages: Iterable[Message]) -> str:
        """Return a sha256 of the *stable* prefix that should drive cache hits.

        For OpenAI's automatic prefix caching to work, the leading slice of
        every call must be byte-identical to prior calls. This signature
        covers exactly the kept messages flagged as cache-stable -- the same
        ones that would receive an Anthropic ``cache_control`` marker. If the
        signature differs between two calls, your cache prefix has drifted.
        """
        from convopack._types import history_hash

        msgs = list(messages)
        result = self.pack(msgs)
        markers = stable_marker_indices(result.kept, pin_specs=self.pin)
        if not markers:
            return history_hash([])
        prefix_end = max(markers) + 1
        return history_hash(result.kept[:prefix_end])

    def cache_info(self, messages: Iterable[Message]) -> dict[str, Any]:
        """Report what would be cached for a given history.

        Returns a dict with:

        * ``markers`` -- indices into ``kept`` that get a cache marker.
        * ``marked_messages`` -- count of those messages.
        * ``marked_tokens`` -- tokens covered by the marked prefix.
        * ``total_tokens`` -- tokens in the entire packed output.
        * ``hit_ratio`` -- ``marked_tokens / total_tokens`` (estimated hit
          ratio on the *next* call if the prefix doesn't drift).
        * ``prefix_signature`` -- ``cache_prefix_signature`` for convenience.
        """
        msgs = list(messages)
        result = self.pack(msgs)
        markers = stable_marker_indices(result.kept, pin_specs=self.pin)
        marked_tokens = sum(
            self.tokenizer.count_message(result.kept[i])
            for i in markers
            if 0 <= i < len(result.kept)
        )
        total = result.token_count or 1
        return {
            "markers": markers,
            "marked_messages": len(markers),
            "marked_tokens": marked_tokens,
            "total_tokens": result.token_count,
            "hit_ratio": marked_tokens / total,
            "prefix_signature": self.cache_prefix_signature(msgs),
        }

    def pack_litellm(self, raw: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convenience: accept LiteLLM messages (OpenAI Chat shape), return packed dicts."""
        from convopack.providers.litellm import from_litellm, to_litellm

        return to_litellm(self.pack(from_litellm(raw)).kept)

    def pack_dspy(self, raw: Any, *, as_history: bool = False) -> Any:
        """Convenience: accept dspy.History or message list, return packed equivalent."""
        from convopack.providers.dspy import from_dspy, to_dspy

        return to_dspy(self.pack(from_dspy(raw)).kept, as_history=as_history)

    def pack_anthropic_managed(
        self,
        raw: Iterable[dict[str, Any]],
        *,
        system: str | None = None,
        trigger_tokens: int = 30_000,
        keep_tool_uses: int = 3,
    ) -> tuple[AnthropicPayload, dict[str, Any]]:
        """Pack AND build an Anthropic ``context_management`` config in one call.

        The returned tuple is ``(payload, context_management_dict)``. Pass
        the dict as the ``context_management`` argument of
        ``client.messages.create``. ``trigger_tokens`` controls when the
        server compresses; ``keep_tool_uses`` is how many recent tool
        exchanges survive its compression.
        """
        from convopack.anthropic_managed import ContextManagementConfig

        payload = self.pack_anthropic(raw, system=system)
        config = ContextManagementConfig.clear_tool_uses(
            trigger_tokens=trigger_tokens, keep_n=keep_tool_uses
        )
        return payload, config.to_dict()

    def pack_langchain(self, raw: Iterable[Any]) -> list[Any]:
        """Convenience: accept LangChain ``BaseMessage`` objects, return packed objects.

        Drop-in replacement for ``langchain_core.messages.trim_messages`` with
        the added guarantee that ``tool_use`` / ``tool_result`` pairs stay
        atomic.
        """
        from convopack.providers.langchain import from_langchain, to_langchain

        result = self.pack(from_langchain(raw))
        return to_langchain(result.kept)

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
