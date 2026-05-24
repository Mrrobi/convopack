"""Wrapper around Anthropic's server-side context-management beta.

Anthropic ships a ``context_management`` field that lets the *server* prune
old tool exchanges from a long conversation. The two are complementary:

* **convopack** decides client-side what to send: budgets, strategies,
  cache markers, multi-provider shape.
* **context_management** lets the server further compact what it sees, with
  no client-side compute.

``ContextManagementConfig`` is a thin builder for the field shape. Combine
with :meth:`Packer.pack_anthropic` to ship a fully-managed payload:

>>> payload = packer.pack_anthropic(history, system="...")
>>> config = ContextManagementConfig.clear_tool_uses(trigger_tokens=30_000, keep_n=3)
>>> client.messages.create(
...     model="claude-sonnet-4-6",
...     max_tokens=1024,
...     system=payload.system,
...     messages=payload.messages,
...     context_management=config.to_dict(),
... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ContextManagementConfig:
    """Build the ``context_management`` field for ``messages.create``."""

    edits: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"edits": list(self.edits)}

    @classmethod
    def clear_tool_uses(
        cls,
        *,
        trigger_tokens: int = 30_000,
        keep_n: int = 3,
        edit_type: str = "clear_tool_uses_20250919",
    ) -> ContextManagementConfig:
        """When the conversation crosses ``trigger_tokens``, drop all but the
        ``keep_n`` most recent ``tool_use`` / ``tool_result`` pairs.
        """
        return cls(
            edits=[
                {
                    "type": edit_type,
                    "trigger": {"type": "input_tokens", "value": trigger_tokens},
                    "keep": {"type": "tool_uses", "value": keep_n},
                }
            ]
        )

    @classmethod
    def empty(cls) -> ContextManagementConfig:
        return cls(edits=[])

    def with_edit(self, edit: dict[str, Any]) -> ContextManagementConfig:
        """Return a new config with ``edit`` appended. Useful for stacking
        custom edit types we don't model yet.
        """
        return ContextManagementConfig(edits=[*self.edits, edit])
