"""Importance strategy: user-scored eviction."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from convopack._pairs import group_pairs
from convopack._types import Message, PackResult

if TYPE_CHECKING:
    from convopack.tokenizers.base import Tokenizer


Scorer = Callable[[Message], float]


def default_scorer(msg: Message) -> float:
    """A reasonable default: tool exchanges and the latest turns score highest."""
    base = 1.0
    if msg.role == "system":
        base = 5.0
    elif msg.has_tool_use() or msg.has_tool_result():
        base = 3.0
    elif msg.role == "user":
        base = 2.0
    return base


class Importance:
    """Drop the lowest-scoring chunks until under budget.

    A chunk's score is the *maximum* message score it contains, so a chunk that
    holds a tool_use/tool_result pair inherits the tool score.
    """

    name = "importance"

    def __init__(self, scorer: Scorer | None = None) -> None:
        self._scorer = scorer or default_scorer

    def pack(
        self,
        messages: list[Message],
        *,
        budget: int,
        tokenizer: Tokenizer,
        pinned_indices: set[int],
    ) -> PackResult:
        if not messages:
            return PackResult(kept=[], dropped=[], summary=None, token_count=0, budget=budget)

        chunks = group_pairs(messages)
        scored: list[tuple[float, int, int]] = []
        for ci, chunk in enumerate(chunks):
            pinned = bool(pinned_indices.intersection(chunk.indices))
            chunk.pinned = pinned
            score = max((self._scorer(m) for m in chunk.messages), default=0.0)
            if pinned:
                score = float("inf")
            cost = tokenizer.count_messages(chunk.messages)
            scored.append((score, ci, cost))

        scored.sort(key=lambda t: (-t[0], -t[1]))

        kept_ids: set[int] = set()
        used = 0
        for score, ci, cost in scored:
            if score == float("inf") or used + cost <= budget:
                kept_ids.add(ci)
                used += cost

        kept: list[Message] = []
        dropped: list[Message] = []
        for ci, chunk in enumerate(chunks):
            (kept if ci in kept_ids else dropped).extend(chunk.messages)

        return PackResult(
            kept=kept,
            dropped=dropped,
            summary=None,
            token_count=used,
            budget=budget,
        )
