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
    """The two pieces an Anthropic ``messages.create`` call wants.

    ``system`` is a plain string by default; if any cache marker was applied
    it becomes a list of text-block dicts so ``cache_control`` can be
    attached. The Anthropic SDK accepts both forms.
    """

    system: str | list[dict[str, Any]]
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


def to_anthropic(
    messages: Iterable[Message],
    *,
    cache_markers: list[int] | None = None,
) -> AnthropicPayload:
    """Convert internal messages to an :class:`AnthropicPayload`.

    System messages are concatenated into a single ``system`` string. The
    remaining messages are emitted in Anthropic content-block shape.

    If ``cache_markers`` is provided, it is interpreted as indices into the
    *input* ``messages`` list; the **last content block** of each marked
    message receives ``{"cache_control": {"type": "ephemeral"}}``. Markers
    pointing at system messages are attached to the system string fragment
    instead, since Anthropic's system field is just a list of text blocks
    under the hood.
    """
    msgs = list(messages)
    markers = set(cache_markers or [])
    system_parts: list[dict[str, Any]] = []
    api_msgs: list[dict[str, Any]] = []
    for i, msg in enumerate(msgs):
        is_marked = i in markers
        if msg.role == "system":
            entry: dict[str, Any] = {"type": "text", "text": msg.text()}
            if is_marked:
                entry["cache_control"] = {"type": "ephemeral"}
            system_parts.append(entry)
            continue

        blocks = [_block_to_anthropic(b) for b in msg.content]
        if is_marked and blocks:
            blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}

        if msg.role == "tool":
            api_msgs.append({"role": "user", "content": blocks})
            continue
        api_msgs.append({"role": msg.role, "content": blocks})

    return AnthropicPayload(system=_render_system(system_parts), messages=api_msgs)


def _render_system(parts: list[dict[str, Any]]) -> Any:
    """Return Anthropic's preferred system shape.

    If no cache markers are present, a plain string keeps the payload
    compatible with older Anthropic SDK versions. As soon as a marker shows
    up we must use the list-of-text-blocks form.
    """
    has_marker = any("cache_control" in p for p in parts)
    text = "\n\n".join(p["text"] for p in parts if p["text"])
    if not has_marker:
        return text
    return parts


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
