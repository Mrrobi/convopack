"""DSPy adapter.

DSPy stores chat history as a :class:`dspy.History` (or a plain ``list[dict]``
in OpenAI Chat shape under the hood). We accept either and convert to our
internal :class:`Message` form.

Requires the ``dspy`` package only at call time; this module never imports
dspy at module load.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from convopack._types import Message
from convopack.providers.openai import from_openai, to_openai

if TYPE_CHECKING:
    pass


def from_dspy(history: Any) -> list[Message]:
    """Convert a ``dspy.History`` (or a plain message list) to internal messages.

    DSPy's ``History`` is backed by a list of OpenAI-shape dicts via its
    ``.messages`` attribute. We dig for that first and fall back to treating
    the input as the list itself.
    """
    msgs: Iterable[dict[str, Any]]
    if hasattr(history, "messages"):
        msgs = history.messages
    elif isinstance(history, list):
        msgs = history
    else:
        raise TypeError(
            f"from_dspy expects a dspy.History or list of dicts, got {type(history).__name__}"
        )
    return from_openai(msgs)


def to_dspy(messages: Iterable[Message], *, as_history: bool = False) -> Any:
    """Convert internal messages back to a DSPy-friendly shape.

    By default returns a list of OpenAI-shape dicts (works with DSPy's
    chat-LM interface directly). Pass ``as_history=True`` to wrap the result
    in ``dspy.History`` -- requires ``dspy`` to be installed.
    """
    msgs = to_openai(messages)
    if not as_history:
        return msgs
    try:
        import dspy
    except ImportError as exc:
        raise ImportError(
            "dspy is required when as_history=True; install with `pip install dspy`."
        ) from exc
    return dspy.History(messages=msgs)
