"""Core type definitions: provider-agnostic Message and content blocks."""

from __future__ import annotations

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


@dataclass(slots=True)
class PackResult:
    """Result of a single pack operation."""

    kept: list[Message]
    dropped: list[Message]
    summary: Message | None
    token_count: int
    budget: int

    @property
    def fits(self) -> bool:
        return self.token_count <= self.budget

    @property
    def headroom(self) -> int:
        return self.budget - self.token_count
