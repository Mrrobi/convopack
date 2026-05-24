"""Google Gemini ``Content`` shape adapter.

Gemini uses different role names (``"model"`` instead of ``"assistant"``) and
nests payloads under ``parts``. Tool calls and tool results are surfaced as
``function_call`` and ``function_response`` parts respectively, and there is no
explicit ``tool_use_id`` -- function responses are matched by name only.
This adapter generates synthetic IDs on the way in and drops them on the way
out so the rest of convopack can use its uniform tool-pair invariant.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from convopack._types import (
    ContentBlock,
    ImageBlock,
    Message,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

_GEMINI_TO_ROLE: dict[str, Role] = {"user": "user", "model": "assistant"}
_ROLE_TO_GEMINI: dict[Role, str] = {
    "user": "user",
    "assistant": "model",
    "tool": "user",
    "system": "user",
}
_SYNTH_ID_PREFIX = "gemini-synthetic-"


@dataclass(slots=True)
class GeminiPayload:
    """What ``GenerativeModel.generate_content`` wants."""

    system_instruction: str
    contents: list[dict[str, Any]] = field(default_factory=list)


def from_gemini(
    contents: Iterable[dict[str, Any]],
    system_instruction: str | None = None,
) -> list[Message]:
    """Convert Gemini ``Content`` dicts (and optional system instruction)."""
    out: list[Message] = []
    if system_instruction:
        out.append(Message(role="system", content=[TextBlock(text=system_instruction)]))

    pending_call_ids: list[str] = []
    synth_seq = 0
    for raw in contents:
        gem_role = raw.get("role", "user")
        role: Role = _GEMINI_TO_ROLE.get(gem_role, "user")
        blocks: list[ContentBlock] = []
        for part in raw.get("parts", []):
            if "text" in part:
                blocks.append(TextBlock(text=part["text"]))
            elif "inline_data" in part:
                data = part["inline_data"]
                blocks.append(
                    ImageBlock(
                        source=data.get("data", ""),
                        media_type=data.get("mime_type"),
                    )
                )
            elif "function_call" in part:
                fc = part["function_call"]
                synth_seq += 1
                synth_id = f"{_SYNTH_ID_PREFIX}{synth_seq}"
                pending_call_ids.append(synth_id)
                blocks.append(
                    ToolUseBlock(
                        id=synth_id,
                        name=fc.get("name", ""),
                        input=dict(fc.get("args", {})),
                    )
                )
            elif "function_response" in part:
                fr = part["function_response"]
                tool_id = (
                    pending_call_ids.pop(0) if pending_call_ids else f"{_SYNTH_ID_PREFIX}orphan"
                )
                response = fr.get("response", {})
                content = (
                    response.get("content")
                    if isinstance(response, dict) and "content" in response
                    else response
                )
                blocks.append(
                    ToolResultBlock(
                        tool_use_id=tool_id,
                        content=str(content) if not isinstance(content, str) else content,
                    )
                )
        out.append(Message(role=role, content=blocks))
    return out


def to_gemini(messages: Iterable[Message]) -> GeminiPayload:
    """Convert internal messages to a :class:`GeminiPayload`.

    System messages are concatenated into ``system_instruction``. Tool calls
    become ``function_call`` parts; tool results become ``function_response``
    parts on a synthetic ``user`` turn. Internal ``tool_use_id`` values are
    stripped because Gemini does not carry them.
    """
    msgs = list(messages)
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    pending_names: list[str] = []
    for msg in msgs:
        if msg.role == "system":
            system_parts.append(msg.text())
            continue
        parts: list[dict[str, Any]] = []
        for block in msg.content:
            if isinstance(block, TextBlock):
                parts.append({"text": block.text})
            elif isinstance(block, ImageBlock):
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": block.media_type or "image/png",
                            "data": block.source,
                        }
                    }
                )
            elif isinstance(block, ToolUseBlock):
                pending_names.append(block.name)
                parts.append({"function_call": {"name": block.name, "args": dict(block.input)}})
            elif isinstance(block, ToolResultBlock):
                name = pending_names.pop(0) if pending_names else block.tool_use_id
                parts.append(
                    {
                        "function_response": {
                            "name": name,
                            "response": {"content": block.content}
                            if isinstance(block.content, str)
                            else {"content": _blocks_to_text(block.content)},
                        }
                    }
                )
        if parts:
            contents.append({"role": _ROLE_TO_GEMINI[msg.role], "parts": parts})
    return GeminiPayload(
        system_instruction="\n\n".join(p for p in system_parts if p),
        contents=contents,
    )


def _blocks_to_text(blocks: list[TextBlock | ImageBlock]) -> str:
    return "".join(b.text for b in blocks if isinstance(b, TextBlock))
