"""Anthropic Messages API shape adapter."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, cast

from convopack._types import (
    ContentBlock,
    ImageBlock,
    Message,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)


@dataclass(slots=True)
class AnthropicPayload:
    """The two pieces an Anthropic ``messages.create`` call wants."""

    system: str
    messages: list[dict[str, Any]] = field(default_factory=list)


def from_anthropic(messages: Iterable[dict[str, Any]], system: str | None = None) -> list[Message]:
    """Convert Anthropic-shape messages (and optional system prompt) to internal form."""
    out: list[Message] = []
    if system:
        out.append(Message(role="system", content=[TextBlock(text=system)]))

    for raw in messages:
        role = cast(Role, raw.get("role", "user"))
        blocks: list[ContentBlock] = []
        content = raw.get("content")
        if isinstance(content, str):
            if content:
                blocks.append(TextBlock(text=content))
        elif isinstance(content, list):
            for part in content:
                ptype = part.get("type")
                if ptype == "text":
                    blocks.append(TextBlock(text=part.get("text", "")))
                elif ptype == "image":
                    src = part.get("source", {})
                    blocks.append(
                        ImageBlock(
                            source=src.get("data", src.get("url", "")),
                            media_type=src.get("media_type"),
                        )
                    )
                elif ptype == "tool_use":
                    blocks.append(
                        ToolUseBlock(
                            id=part.get("id", ""),
                            name=part.get("name", ""),
                            input=part.get("input", {}),
                        )
                    )
                elif ptype == "tool_result":
                    inner = part.get("content", "")
                    if isinstance(inner, list):
                        inner_blocks: list[TextBlock | ImageBlock] = []
                        for ip in inner:
                            if ip.get("type") == "text":
                                inner_blocks.append(TextBlock(text=ip.get("text", "")))
                            elif ip.get("type") == "image":
                                isrc = ip.get("source", {})
                                inner_blocks.append(
                                    ImageBlock(
                                        source=isrc.get("data", ""),
                                        media_type=isrc.get("media_type"),
                                    )
                                )
                        blocks.append(
                            ToolResultBlock(
                                tool_use_id=part.get("tool_use_id", ""),
                                content=inner_blocks,
                                is_error=bool(part.get("is_error", False)),
                            )
                        )
                    else:
                        blocks.append(
                            ToolResultBlock(
                                tool_use_id=part.get("tool_use_id", ""),
                                content=str(inner),
                                is_error=bool(part.get("is_error", False)),
                            )
                        )
        out.append(Message(role=role, content=blocks))
    return out


def to_anthropic(messages: Iterable[Message]) -> AnthropicPayload:
    """Convert internal messages to an :class:`AnthropicPayload`.

    System messages are concatenated into a single ``system`` string. The
    remaining messages are emitted in Anthropic content-block shape.
    """
    msgs = list(messages)
    system_parts: list[str] = []
    api_msgs: list[dict[str, Any]] = []
    for msg in msgs:
        if msg.role == "system":
            system_parts.append(msg.text())
            continue
        if msg.role == "tool":
            api_msgs.append(
                {"role": "user", "content": [_block_to_anthropic(b) for b in msg.content]}
            )
            continue
        api_msgs.append(
            {"role": msg.role, "content": [_block_to_anthropic(b) for b in msg.content]}
        )
    return AnthropicPayload(system="\n\n".join(p for p in system_parts if p), messages=api_msgs)


def _block_to_anthropic(block: ContentBlock) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ImageBlock):
        return {
            "type": "image",
            "source": {
                "type": "base64" if block.media_type else "url",
                "media_type": block.media_type or "image/png",
                "data": block.source,
            },
        }
    if isinstance(block, ToolUseBlock):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    if isinstance(block, ToolResultBlock):
        if isinstance(block.content, str):
            content: Any = block.content
        else:
            content = [_block_to_anthropic(b) for b in block.content]
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": content,
            "is_error": block.is_error,
        }
    raise TypeError(f"Unknown block type: {type(block).__name__}")
