"""Prompt-cache annotation system.

LLM providers offer prompt caching to reduce cost on stable prefixes:

* **Anthropic** -- explicit ``cache_control: {"type": "ephemeral"}`` markers
  on selected content blocks. Maximum of 4 markers per request.
* **OpenAI** -- automatic caching of identical 1024-token prefixes. No
  marker needed; alignment matters.

``convopack`` knows which chunks are pinned, so it's the natural place to
decide where caches should go. Markers are computed by :class:`Packer` when
``cache=True`` and surfaced on :class:`PackResult.cache_markers`. Providers
emit them; the rest of the library doesn't care.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from convopack._types import Message


ANTHROPIC_MAX_MARKERS = 4
STABLE_PIN_SPECS: frozenset[str] = frozenset({"system", "first_user"})


def stable_marker_indices(
    kept: list[Message],
    *,
    pin_specs: tuple[str | int, ...],
    max_markers: int = ANTHROPIC_MAX_MARKERS,
) -> list[int]:
    """Compute indices into ``kept`` that should receive a cache marker.

    Only *stable* pin specs (``"system"`` and ``"first_user"``) and explicit
    integer-index pins are eligible -- volatile pins like ``"last_user"`` are
    never marked because the last user turn changes every call and would
    invalidate the cache.

    The result is capped at ``max_markers`` (default 4 for Anthropic) and
    sorted ascending so the longest stable prefix benefits.
    """
    if max_markers <= 0:
        return []

    stable_indices: list[int] = []
    first_user_seen = False
    for i, msg in enumerate(kept):
        for spec in pin_specs:
            if isinstance(spec, int):
                if spec >= 0 and spec == i:
                    stable_indices.append(i)
                    break
            elif spec == "system" and msg.role == "system":
                stable_indices.append(i)
                break
            elif spec == "first_user" and msg.role == "user" and not first_user_seen:
                stable_indices.append(i)
                first_user_seen = True
                break

    deduped: list[int] = []
    for idx in sorted(set(stable_indices)):
        if idx not in deduped:
            deduped.append(idx)
        if len(deduped) >= max_markers:
            break
    return deduped
