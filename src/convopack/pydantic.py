"""Optional Pydantic helpers for typed ``Message.metadata``.

``Message.metadata`` is a plain ``dict[str, Any]`` in the core type so the
library has no required dependencies. When you want type-checked access to
metadata fields (e.g. inside a scorer for :class:`Importance`), this module
provides two small helpers that lazy-import Pydantic.

Requires the ``pydantic`` package (``pip install convopack[pydantic]``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from pydantic import BaseModel

    from convopack._types import Message


T = TypeVar("T", bound="BaseModel")


def _require_pydantic() -> None:
    try:
        import pydantic  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pydantic is required for convopack.pydantic; "
            "install with `pip install convopack[pydantic]`."
        ) from exc


def validate_metadata(messages: Iterable[Message], schema: type[T]) -> list[T]:
    """Validate every message's metadata against ``schema`` and return the parsed models.

    Raises ``pydantic.ValidationError`` on the first message whose metadata
    fails validation. Use when you want a hard wall between the dict-typed
    core and your application's typed view.
    """
    _require_pydantic()
    return [schema.model_validate(m.metadata) for m in messages]


def typed_scorer(
    schema: type[T],
    score_fn: Callable[[T, Message], float],
) -> Callable[[Message], float]:
    """Wrap a scorer so it receives a parsed Pydantic model instead of a raw dict.

    Example
    -------
    >>> from pydantic import BaseModel
    >>> class TurnMeta(BaseModel):
    ...     starred: bool = False
    ...     source: str = "user"
    >>> def score(meta: TurnMeta, msg) -> float:
    ...     return 100.0 if meta.starred else 1.0
    >>> from convopack import Importance
    >>> from convopack.pydantic import typed_scorer
    >>> packer_strategy = Importance(scorer=typed_scorer(TurnMeta, score))
    """
    _require_pydantic()

    def wrapped(msg: Message) -> float:
        parsed = schema.model_validate(msg.metadata)
        return score_fn(parsed, msg)

    return wrapped


def metadata_pin(
    history: Iterable[Message],
    schema: type[T],
    predicate: Callable[[T], bool],
) -> tuple[int, ...]:
    """Compute pin indices for messages whose validated metadata satisfies ``predicate``.

    Useful for ``pin=metadata_pin(history, TurnMeta, lambda m: m.starred)``.
    Messages whose metadata fails to validate are silently skipped (the
    invariant is "starred messages always survive", not "validate everything").
    """
    _require_pydantic()
    from pydantic import ValidationError

    pinned: list[int] = []
    for i, m in enumerate(history):
        try:
            parsed = schema.model_validate(m.metadata)
        except ValidationError:
            continue
        if predicate(parsed):
            pinned.append(i)
    return tuple(pinned)


def _wrap_attr(obj: Any) -> Any:
    return obj
