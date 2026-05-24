"""Tokenizer adapters."""

from __future__ import annotations

from convopack.tokenizers.approx import ApproxTokenizer
from convopack.tokenizers.base import Tokenizer

__all__ = ["ApproxTokenizer", "Tokenizer", "get_tokenizer"]


def get_tokenizer(spec: str | Tokenizer) -> Tokenizer:
    """Resolve a tokenizer from a string spec or pass through an instance.

    Spec formats:
      - ``"tiktoken:<encoding-or-model>"`` -- e.g. ``"tiktoken:gpt-4o"``.
      - ``"anthropic:<model>"`` -- e.g. ``"anthropic:claude-sonnet-4-6"``.
      - ``"huggingface:<model-id>"`` -- any HuggingFace tokenizer id.
      - ``"approx"`` -- char-length / 4 estimator (zero deps).
    """
    if not isinstance(spec, str):
        return spec
    if spec == "approx":
        return ApproxTokenizer()
    if spec.startswith("tiktoken:"):
        from convopack.tokenizers.tiktoken_adapter import TiktokenAdapter

        return TiktokenAdapter(spec.split(":", 1)[1])
    if spec.startswith("anthropic:"):
        from convopack.tokenizers.anthropic_adapter import AnthropicAdapter

        return AnthropicAdapter(spec.split(":", 1)[1])
    if spec.startswith("huggingface:"):
        from convopack.tokenizers.huggingface_adapter import HFTokenizerAdapter

        return HFTokenizerAdapter(spec.split(":", 1)[1])
    raise ValueError(f"Unknown tokenizer spec: {spec!r}")
