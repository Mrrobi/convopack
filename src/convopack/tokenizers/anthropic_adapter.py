"""Anthropic tokenizer adapter.

Anthropic does not ship a local tokenizer; the official method is to call
``messages.count_tokens`` on the server. That is a network call, so we offer
two modes:

* ``offline=True`` (default) -- delegate to :class:`ApproxTokenizer`. Fast and
  free, off by ~10-15%.
* ``offline=False`` -- if the ``anthropic`` SDK is installed and an API key is
  configured, call ``client.messages.count_tokens`` for exact counts. Slow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from convopack.tokenizers.approx import ApproxTokenizer

if TYPE_CHECKING:
    from collections.abc import Iterable

    from convopack._types import Message


class AnthropicAdapter:
    name: str

    def __init__(self, model: str, *, offline: bool = True) -> None:
        self._model = model
        self._offline = offline
        self._approx = ApproxTokenizer()
        self.name = f"anthropic:{model}"
        self._client = None
        if not offline:
            try:
                import anthropic

                self._client = anthropic.Anthropic()
            except ImportError as exc:
                raise ImportError(
                    "anthropic SDK is required for online AnthropicAdapter; "
                    "install with `pip install convopack[anthropic]`."
                ) from exc

    def count(self, text: str) -> int:
        return self._approx.count(text)

    def count_message(self, message: Message) -> int:
        if self._offline or self._client is None:
            return self._approx.count_message(message)
        return self.count_messages([message])

    def count_messages(self, messages: Iterable[Message]) -> int:
        msgs = list(messages)
        if self._offline or self._client is None:
            return self._approx.count_messages(msgs)
        from convopack.providers.anthropic import to_anthropic

        payload = to_anthropic(msgs)
        kwargs: dict[str, object] = {"model": self._model, "messages": payload.messages}
        if payload.system:
            kwargs["system"] = payload.system
        result = self._client.messages.count_tokens(**kwargs)  # type: ignore[arg-type]
        return int(result.input_tokens)
