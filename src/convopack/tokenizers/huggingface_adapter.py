"""HuggingFace ``transformers``-backed tokenizer adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from convopack._types import ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock

if TYPE_CHECKING:
    from collections.abc import Iterable

    from convopack._types import Message


_PER_MESSAGE_OVERHEAD = 4
_PER_REPLY_OVERHEAD = 3
_IMAGE_TOKEN_ESTIMATE = 765


class HFTokenizerAdapter:
    """Wraps a HuggingFace ``AutoTokenizer`` so any open-weights model can be used.

    Pass either a HuggingFace model id (e.g. ``meta-llama/Llama-3.1-8B``) or a
    local path. Tokeniser files are downloaded on first use unless
    ``local_files_only=True``.
    """

    name: str

    def __init__(
        self,
        model: str,
        *,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for HFTokenizerAdapter; "
                "install with `pip install convopack[huggingface]`."
            ) from exc

        self._model = model
        self.name = f"huggingface:{model}"
        self._tok = AutoTokenizer.from_pretrained(
            model, revision=revision, local_files_only=local_files_only
        )

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tok.encode(text, add_special_tokens=False))

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
