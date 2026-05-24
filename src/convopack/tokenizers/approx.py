"""Zero-dependency approximate tokenizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from convopack._types import ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock

if TYPE_CHECKING:
    from collections.abc import Iterable

    from convopack._types import Message


_CHARS_PER_TOKEN = 4
_PER_MESSAGE_OVERHEAD = 4
_IMAGE_TOKEN_ESTIMATE = 765


class ApproxTokenizer:
    """Estimates tokens as ``ceil(char_count / 4)``.

    Use when you don't want to pull in tiktoken or anthropic. Numbers will be off
    by 10-30% depending on language; fine for a soft budget but not for billing.
    """

    name = "approx"

    def count(self, text: str) -> int:
        if not text:
            return 0
        return (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN

    def count_message(self, message: Message) -> int:
        total = _PER_MESSAGE_OVERHEAD + self.count(message.role) + self.count(message.name or "")
        for block in message.content:
            total += self._block_tokens(block)
        return total

    def count_messages(self, messages: Iterable[Message]) -> int:
        return sum(self.count_message(m) for m in messages)

    def _block_tokens(self, block: object) -> int:
        if isinstance(block, TextBlock):
            return self.count(block.text)
        if isinstance(block, ImageBlock):
            return _IMAGE_TOKEN_ESTIMATE
        if isinstance(block, ToolUseBlock):
            payload = f"{block.id}{block.name}{block.input!r}"
            return self.count(payload) + _PER_MESSAGE_OVERHEAD
        if isinstance(block, ToolResultBlock):
            if isinstance(block.content, str):
                return self.count(block.content) + self.count(block.tool_use_id)
            inner = sum(self._block_tokens(b) for b in block.content)
            return inner + self.count(block.tool_use_id) + _PER_MESSAGE_OVERHEAD
        return 0
