"""SemanticDedup strategy: remove near-duplicate turns before falling back."""

from __future__ import annotations

from typing import TYPE_CHECKING

from convopack._pairs import group_pairs
from convopack._types import Message, PackResult
from convopack.embedders.base import cosine
from convopack.strategies.recency import Recency

if TYPE_CHECKING:
    from convopack.embedders.base import Embedder
    from convopack.strategies.base import Strategy
    from convopack.tokenizers.base import Tokenizer


class SemanticDedup:
    """Drop near-duplicate messages (by embedding similarity), then defer to a fallback strategy.

    Chunks containing tool calls or pinned messages are never deduped — they're
    handed straight to the fallback. Among the remaining message-chunks, only
    the *first* occurrence of each near-duplicate group survives. Removing
    duplicates is order-preserving.

    Parameters
    ----------
    embedder
        An object with ``embed(text) -> list[float]``.
    threshold
        Cosine similarity above which two messages are considered duplicates.
    fallback
        Strategy to enforce the budget after dedup. Defaults to :class:`Recency`.
    """

    name = "semantic_dedup"

    def __init__(
        self,
        embedder: Embedder,
        *,
        threshold: float = 0.95,
        fallback: Strategy | None = None,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self._embedder = embedder
        self._threshold = threshold
        self._fallback = fallback or Recency()

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
        survivors: list[Message] = []
        dedup_dropped: list[Message] = []
        seen_embeddings: list[list[float]] = []

        for chunk in chunks:
            chunk_indices = set(chunk.indices)
            untouchable = bool(chunk_indices & pinned_indices) or any(
                m.has_tool_use() or m.has_tool_result() for m in chunk.messages
            )
            if untouchable:
                survivors.extend(chunk.messages)
                for m in chunk.messages:
                    text = m.text()
                    if text:
                        seen_embeddings.append(self._embedder.embed(text))
                continue

            for m in chunk.messages:
                text = m.text()
                if not text:
                    survivors.append(m)
                    continue
                vec = self._embedder.embed(text)
                if any(cosine(vec, prev) >= self._threshold for prev in seen_embeddings):
                    dedup_dropped.append(m)
                else:
                    survivors.append(m)
                    seen_embeddings.append(vec)

        fallback_pinned = self._reindex_pinned(messages, survivors, pinned_indices)
        fallback_result = self._fallback.pack(
            survivors,
            budget=budget,
            tokenizer=tokenizer,
            pinned_indices=fallback_pinned,
        )
        return PackResult(
            kept=fallback_result.kept,
            dropped=dedup_dropped + fallback_result.dropped,
            summary=fallback_result.summary,
            token_count=fallback_result.token_count,
            budget=budget,
        )

    @staticmethod
    def _reindex_pinned(
        original: list[Message],
        survivors: list[Message],
        pinned: set[int],
    ) -> set[int]:
        pinned_ids = {id(original[i]) for i in pinned if 0 <= i < len(original)}
        return {i for i, m in enumerate(survivors) if id(m) in pinned_ids}
