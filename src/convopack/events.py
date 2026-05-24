"""Pack events for observability of strategy decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from convopack._types import Message


PackEventKind = Literal["kept", "dropped", "summarized", "done"]


@dataclass(slots=True)
class PackEvent:
    """One step in a pack operation.

    Attributes
    ----------
    kind
        ``kept`` -- message survived the pack.
        ``dropped`` -- message was evicted.
        ``summarized`` -- a synthetic summary message replaces some dropped block.
        ``done`` -- terminal event carrying the final token count.
    index
        Original position in the input list, or ``-1`` for synthetic events.
    message
        The message itself, or ``None`` for ``done``.
    token_cost
        Tokens contributed by this message. For ``done``, the total kept tokens.
    """

    kind: PackEventKind
    index: int
    message: Message | None
    token_cost: int
