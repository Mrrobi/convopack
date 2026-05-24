"""Core type definitions: provider-agnostic Message and content blocks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class TextBlock:
    text: str
    kind: Literal["text"] = "text"


@dataclass(frozen=True, slots=True)
class ImageBlock:
    """Reference to an image. `source` is opaque (URL, base64, file_id) and provider-specific."""

    source: str
    media_type: str | None = None
    kind: Literal["image"] = "image"


@dataclass(frozen=True, slots=True)
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    kind: Literal["tool_use"] = "tool_use"


@dataclass(frozen=True, slots=True)
class ToolResultBlock:
    tool_use_id: str
    content: str | list[TextBlock | ImageBlock]
    is_error: bool = False
    kind: Literal["tool_result"] = "tool_result"


ContentBlock = TextBlock | ImageBlock | ToolUseBlock | ToolResultBlock


@dataclass(slots=True)
class Message:
    """Provider-agnostic message.

    `content` is always a list of blocks internally. Provider adapters convert to
    and from string / dict forms when crossing the public boundary.
    """

    role: Role
    content: list[ContentBlock] = field(default_factory=list)
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_tool_use(self) -> bool:
        return any(b.kind == "tool_use" for b in self.content)

    def has_tool_result(self) -> bool:
        return any(b.kind == "tool_result" for b in self.content)

    def tool_use_ids(self) -> list[str]:
        return [b.id for b in self.content if isinstance(b, ToolUseBlock)]

    def tool_result_ids(self) -> list[str]:
        return [b.tool_use_id for b in self.content if isinstance(b, ToolResultBlock)]

    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def content_hash(self) -> str:
        """Stable sha256 hex digest of this message's *semantic* content.

        The hash deliberately excludes runtime-generated tool-call IDs so that
        two messages with identical text, tool name, and input arguments hash
        the same regardless of which provider generated the wire IDs. Useful
        as a cache key and to detect prompt-prefix drift across packs.
        """
        return hashlib.sha256(
            json.dumps(_canonical(self), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def _canonical_block(block: ContentBlock) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"k": "text", "t": block.text}
    if isinstance(block, ImageBlock):
        return {"k": "image", "m": block.media_type, "s": block.source}
    if isinstance(block, ToolUseBlock):
        return {"k": "tool_use", "n": block.name, "i": block.input}
    if isinstance(block, ToolResultBlock):
        content_payload: Any
        if isinstance(block.content, str):
            content_payload = block.content
        else:
            content_payload = [_canonical_block(b) for b in block.content]
        return {"k": "tool_result", "c": content_payload, "e": block.is_error}
    raise TypeError(f"Unknown block type: {type(block).__name__}")


def _canonical(message: Message) -> dict[str, Any]:
    return {
        "r": message.role,
        "n": message.name,
        "c": [_canonical_block(b) for b in message.content],
    }


def history_hash(messages: list[Message]) -> str:
    """Stable sha256 of an ordered list of messages.

    Useful as a cache key for "give me the same packed output for the same
    input history" and to detect whether your prompt prefix has drifted
    between turns (e.g. for OpenAI's automatic prefix caching).
    """
    payload = [_canonical(m) for m in messages]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(slots=True)
class PackResult:
    """Result of a single pack operation."""

    kept: list[Message]
    dropped: list[Message]
    summary: Message | None
    token_count: int
    budget: int
    cache_markers: list[int] = field(default_factory=list)

    @property
    def fits(self) -> bool:
        return self.token_count <= self.budget

    @property
    def headroom(self) -> int:
        return self.budget - self.token_count
