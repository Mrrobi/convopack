"""LangChain ``BaseMessage`` adapter.

Lets you drop-in replace ``langchain_core.messages.trim_messages`` with
``convopack.Packer.pack_langchain()`` while keeping every LangChain integration
you already wrote. Requires ``langchain_core`` to be installed; lazy-imported.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

from convopack._types import (
    ContentBlock,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


def _import_lc() -> Any:
    try:
        import langchain_core.messages as lc_messages
    except ImportError as exc:
        raise ImportError(
            "langchain_core is required for the LangChain adapter; "
            "install with `pip install langchain-core`."
        ) from exc
    return lc_messages


def from_langchain(messages: Iterable[BaseMessage]) -> list[Message]:
    """Convert ``langchain_core.messages.BaseMessage`` objects to internal form."""
    lc = _import_lc()
    out: list[Message] = []
    for m in messages:
        content = _flatten_lc_content(getattr(m, "content", ""))
        if isinstance(m, lc.SystemMessage):
            out.append(Message(role="system", content=[TextBlock(text=content)]))
        elif isinstance(m, lc.HumanMessage):
            out.append(Message(role="user", content=[TextBlock(text=content)]))
        elif isinstance(m, lc.AIMessage):
            blocks: list[ContentBlock] = []
            if content:
                blocks.append(TextBlock(text=content))
            for call in getattr(m, "tool_calls", []) or []:
                blocks.append(
                    ToolUseBlock(
                        id=str(call.get("id") or ""),
                        name=str(call.get("name", "")),
                        input=dict(call.get("args", {}) or {}),
                    )
                )
            out.append(Message(role="assistant", content=blocks))
        elif isinstance(m, lc.ToolMessage):
            out.append(
                Message(
                    role="tool",
                    content=[
                        ToolResultBlock(
                            tool_use_id=str(getattr(m, "tool_call_id", "")),
                            content=content,
                        )
                    ],
                )
            )
        else:
            role = getattr(m, "type", "user")
            out.append(
                Message(
                    role=cast(
                        Any, role if role in {"system", "user", "assistant", "tool"} else "user"
                    ),
                    content=[TextBlock(text=content)],
                )
            )
    return out


def to_langchain(messages: Iterable[Message]) -> list[BaseMessage]:
    """Convert internal :class:`Message` objects to LangChain ``BaseMessage``."""
    lc = _import_lc()
    out: list[BaseMessage] = []
    for msg in messages:
        text = msg.text()
        tool_calls = [
            {"id": b.id, "name": b.name, "args": dict(b.input)}
            for b in msg.content
            if isinstance(b, ToolUseBlock)
        ]
        if msg.role == "system":
            out.append(lc.SystemMessage(content=text))
        elif msg.role == "user":
            out.append(lc.HumanMessage(content=text))
        elif msg.role == "assistant":
            if tool_calls:
                out.append(lc.AIMessage(content=text or "", tool_calls=tool_calls))
            else:
                out.append(lc.AIMessage(content=text))
        elif msg.role == "tool":
            tool_results = [b for b in msg.content if isinstance(b, ToolResultBlock)]
            for tr in tool_results:
                payload = (
                    tr.content
                    if isinstance(tr.content, str)
                    else "".join(b.text for b in tr.content if isinstance(b, TextBlock))
                )
                out.append(lc.ToolMessage(content=payload, tool_call_id=tr.tool_use_id))
    return out


def _flatten_lc_content(content: Any) -> str:
    """LangChain ``content`` can be a string or a list of content parts; flatten to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and "text" in p:
                parts.append(str(p["text"]))
        return "".join(parts)
    return str(content)
