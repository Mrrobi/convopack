"""Recency strategy: keep the newest chunks that fit in budget."""

from __future__ import annotations

from typing import TYPE_CHECKING

from convopack._pairs import group_pairs
from convopack._types import PackResult

if TYPE_CHECKING:
    from convopack._types import Message
    from convopack.tokenizers.base import Tokenizer


class Recency:
    """Keep the most recent chunks that still fit under ``budget``.

    Pinned messages are always kept; if pinned messages alone exceed budget,
    they are still kept (the result will report ``fits=False``).
    """

    name = "recency"

    def __init__(self, *, min_keep: int = 1) -> None:
        if min_keep < 0:
            raise ValueError("min_keep must be >= 0")
        self._min_keep = min_keep

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
        pinned_chunk_ids: set[int] = set()
        for ci, chunk in enumerate(chunks):
            if pinned_indices.intersection(chunk.indices):
                chunk.pinned = True
                pinned_chunk_ids.add(ci)

        pinned_msgs: list[Message] = []
        for ci in pinned_chunk_ids:
            pinned_msgs.extend(chunks[ci].messages)
        used = tokenizer.count_messages(pinned_msgs) if pinned_msgs else 0

        kept_ids: set[int] = set(pinned_chunk_ids)
        non_pinned_kept = 0
        for ci in range(len(chunks) - 1, -1, -1):
            if ci in kept_ids:
                continue
            cost = tokenizer.count_messages(chunks[ci].messages)
            if used + cost <= budget or non_pinned_kept < self._min_keep:
                kept_ids.add(ci)
                used += cost
                non_pinned_kept += 1

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
