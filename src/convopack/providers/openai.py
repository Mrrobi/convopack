"""OpenAI Chat Completions message-shape adapter."""

from __future__ import annotations

import json
from collections.abc import Iterable
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


def from_openai(messages: Iterable[dict[str, Any]]) -> list[Message]:
    """Convert OpenAI Chat Completions messages into :class:`Message` objects."""
    out: list[Message] = []
    for raw in messages:
        role = cast(Role, raw.get("role", "user"))
        if role == "tool":
            out.append(
                Message(
                    role="tool",
                    content=[
                        ToolResultBlock(
                            tool_use_id=raw.get("tool_call_id", ""),
                            content=str(raw.get("content", "")),
                        )
                    ],
                )
            )
            continue

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
                elif ptype in {"image_url", "input_image"}:
                    url = part.get("image_url", {})
                    src = url.get("url") if isinstance(url, dict) else url
                    blocks.append(ImageBlock(source=src or part.get("image_url", "")))

        for call in raw.get("tool_calls", []) or []:
            fn = call.get("function", {})
            args_raw = fn.get("arguments", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                args = {"_raw": args_raw}
            blocks.append(ToolUseBlock(id=call.get("id", ""), name=fn.get("name", ""), input=args))

        out.append(Message(role=role, content=blocks, name=raw.get("name")))
    return out


def to_openai(messages: Iterable[Message]) -> list[dict[str, Any]]:
    """Convert :class:`Message` objects into OpenAI Chat Completions shape."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "tool":
            tool_results = [b for b in msg.content if isinstance(b, ToolResultBlock)]
            for tr in tool_results:
                payload = (
                    tr.content
                    if isinstance(tr.content, str)
                    else "".join(b.text for b in tr.content if isinstance(b, TextBlock))
                )
                out.append({"role": "tool", "tool_call_id": tr.tool_use_id, "content": payload})
            continue

        text_parts: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        for block in msg.content:
            if isinstance(block, TextBlock):
                text_parts.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageBlock):
                text_parts.append({"type": "image_url", "image_url": {"url": block.source}})
            elif isinstance(block, ToolUseBlock):
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        },
                    }
                )

        entry: dict[str, Any] = {"role": msg.role}
        if msg.name:
            entry["name"] = msg.name
        if len(text_parts) == 1 and text_parts[0]["type"] == "text":
            entry["content"] = text_parts[0]["text"]
        elif text_parts:
            entry["content"] = text_parts
        else:
            entry["content"] = None
        if tool_calls:
            entry["tool_calls"] = tool_calls
        out.append(entry)
    return out
