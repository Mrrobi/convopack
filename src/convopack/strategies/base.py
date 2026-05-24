"""Strategy protocol and helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from convopack._types import Message, PackResult
    from convopack.tokenizers.base import Tokenizer


PinSpec = Literal["system", "first_user", "last_user", "tool_results"] | int


@runtime_checkable
class Strategy(Protocol):
    """A packing strategy decides which messages survive a budget cut.

    Implementations must:

      * keep pinned messages,
      * never split a tool_use / tool_result pair,
      * return a :class:`PackResult` whose ``kept`` list has the same
        relative order as the input.
    """

    name: str

    def pack(
        self,
        messages: list[Message],
        *,
        budget: int,
        tokenizer: Tokenizer,
        pinned_indices: set[int],
    ) -> PackResult: ...
