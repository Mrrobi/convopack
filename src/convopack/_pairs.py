"""Tool-call pair detection.

A ``tool_use`` block in an assistant message must be followed by a matching
``tool_result`` block in a subsequent user (or tool) message. If we evict only
half of a pair, the next provider call rejects the conversation.

The helpers here group messages into atomic *chunks* that the strategy layer
can keep or drop as one unit.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from convopack._types import Message, ToolResultBlock, ToolUseBlock


@dataclass(slots=True)
class Chunk:
    """One indivisible group of messages, plus their original positions."""

    messages: list[Message]
    indices: list[int] = field(default_factory=list)
    pinned: bool = False

    @property
    def first_index(self) -> int:
        return self.indices[0] if self.indices else -1


def group_pairs(messages: Iterable[Message]) -> list[Chunk]:
    """Group messages so that no chunk splits a tool_use / tool_result pair.

    Walks forward, tracking unsatisfied tool_use IDs. While any are open, the
    current chunk keeps growing; the chunk closes when every open ID has been
    satisfied.
    """
    msgs = list(messages)
    chunks: list[Chunk] = []
    open_ids: set[str] = set()
    buf_msgs: list[Message] = []
    buf_idx: list[int] = []

    for idx, msg in enumerate(msgs):
        buf_msgs.append(msg)
        buf_idx.append(idx)
        for block in msg.content:
            if isinstance(block, ToolUseBlock):
                open_ids.add(block.id)
            elif isinstance(block, ToolResultBlock):
                open_ids.discard(block.tool_use_id)
        if not open_ids:
            chunks.append(Chunk(messages=buf_msgs, indices=buf_idx))
            buf_msgs = []
            buf_idx = []

    if buf_msgs:
        chunks.append(Chunk(messages=buf_msgs, indices=buf_idx))
    return chunks


def validate_pairs(messages: Iterable[Message]) -> list[str]:
    """Return a list of dangling tool_use IDs that have no matching tool_result."""
    open_ids: set[str] = set()
    for msg in messages:
        for block in msg.content:
            if isinstance(block, ToolUseBlock):
                open_ids.add(block.id)
            elif isinstance(block, ToolResultBlock):
                open_ids.discard(block.tool_use_id)
    return sorted(open_ids)
