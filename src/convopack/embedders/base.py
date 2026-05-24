"""Embedder protocol and helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Turns text into a dense vector.

    Implementations may optionally provide :meth:`embed_batch` for efficiency;
    a default implementation that loops over ``embed`` is acceptable.
    """

    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]: ...


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Returns 0 if either vector is zero."""
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    return dot / denom if denom else 0.0
