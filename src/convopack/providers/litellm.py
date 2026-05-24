"""LiteLLM adapter.

LiteLLM normalises ~100 providers behind a single OpenAI-Chat-compatible
``messages=[...]`` shape. The wire format is the OpenAI Chat Completions
shape, so this adapter mostly forwards to the OpenAI adapter. The reason it
exists as a separate import path is so users can write::

    from convopack.providers.litellm import from_litellm, to_litellm

and signal intent in their code without thinking about the OpenAI link.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from convopack._types import Message
from convopack.providers.openai import from_openai, to_openai


def from_litellm(messages: Iterable[dict[str, Any]]) -> list[Message]:
    """Convert LiteLLM messages (OpenAI Chat shape) to :class:`Message` objects."""
    return from_openai(messages)


def to_litellm(messages: Iterable[Message]) -> list[dict[str, Any]]:
    """Convert :class:`Message` objects to LiteLLM messages (OpenAI Chat shape)."""
    return to_openai(messages)
