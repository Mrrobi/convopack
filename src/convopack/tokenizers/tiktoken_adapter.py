"""tiktoken-backed tokenizer (OpenAI BPE)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from convopack._types import ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock

if TYPE_CHECKING:
    from collections.abc import Iterable

    from convopack._types import Message


_PER_MESSAGE_OVERHEAD = 4
_PER_REPLY_OVERHEAD = 3
_IMAGE_TOKEN_ESTIMATE = 765


class TiktokenAdapter:
    """Wraps a tiktoken encoding. Specify either a model name or an encoding name."""

    name: str

    def __init__(self, model_or_encoding: str) -> None:
        try:
            import tiktoken
        except ImportError as exc:
            raise ImportError(
                "tiktoken is required for TiktokenAdapter; "
                "install with `pip install convopack[tiktoken]`."
            ) from exc

        self._model = model_or_encoding
        self.name = f"tiktoken:{model_or_encoding}"
        try:
            self._enc = tiktoken.encoding_for_model(model_or_encoding)
        except KeyError:
            self._enc = tiktoken.get_encoding(model_or_encoding)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._enc.encode(text, disallowed_special=()))

    def count_message(self, message: Message) -> int:
        total = _PER_MESSAGE_OVERHEAD + self.count(message.role)
        if message.name:
            total += self.count(message.name)
        for block in message.content:
            total += self._block_tokens(block)
        return total

    def count_messages(self, messages: Iterable[Message]) -> int:
        return sum(self.count_message(m) for m in messages) + _PER_REPLY_OVERHEAD

    def _block_tokens(self, block: object) -> int:
        if isinstance(block, TextBlock):
            return self.count(block.text)
        if isinstance(block, ImageBlock):
            return _IMAGE_TOKEN_ESTIMATE
        if isinstance(block, ToolUseBlock):
            return self.count(f"{block.name}{block.input!r}") + self.count(block.id)
        if isinstance(block, ToolResultBlock):
            if isinstance(block.content, str):
                return self.count(block.content) + self.count(block.tool_use_id)
            return sum(self._block_tokens(b) for b in block.content) + self.count(block.tool_use_id)
        return 0
