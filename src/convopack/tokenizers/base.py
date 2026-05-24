"""Tokenizer protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable

    from convopack._types import Message


@runtime_checkable
class Tokenizer(Protocol):
    """Counts tokens for text and messages.

    Implementations should be cheap to call repeatedly. Counting a message must
    include any per-message overhead the target provider charges (role tags,
    separators, tool framing).
    """

    name: str

    def count(self, text: str) -> int: ...

    def count_message(self, message: Message) -> int: ...

    def count_messages(self, messages: Iterable[Message]) -> int: ...
