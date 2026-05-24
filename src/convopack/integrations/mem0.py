"""mem0 integration: forward evicted turns into long-term semantic memory.

`SummaryEvict` already replaces dropped turns with a short summary message at
the head of the kept list. With `Mem0Summarizer`, the same evicted turns are
also added to a mem0 `Memory` instance so they can be semantically retrieved
later -- giving you per-turn budget enforcement and persistent memory in one
loop.

Requires the ``mem0ai`` package (``pip install convopack[mem0]``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from convopack._types import Message


class Mem0Summarizer:
    """Summariser that also stores evicted turns in mem0.

    Pass an instance to :class:`convopack.SummaryEvict`:

    >>> from mem0 import Memory
    >>> from convopack import Packer, SummaryEvict
    >>> from convopack.integrations.mem0 import Mem0Summarizer
    >>> memory = Memory()
    >>> packer = Packer(
    ...     budget=8000,
    ...     strategy=SummaryEvict(Mem0Summarizer(memory, user_id="alice")),
    ... )

    Parameters
    ----------
    memory
        A ``mem0.Memory`` instance (or anything with the same ``.add`` and
        ``.search`` interface).
    user_id, agent_id, session_id
        Scope identifiers forwarded to ``memory.add``.
    summary_template
        Format string with ``{n}`` (evicted count) and ``{retrieved}`` (mem0
        results joined by newline). Used to build the in-context summary
        message that the packer inserts at the head of the kept list.
    llm_summariser
        Optional callable that takes the evicted messages and returns a short
        text. If given, that text becomes ``{summary}`` in ``summary_template``
        (which then accepts an extra ``{summary}`` slot).
    retrieve_query
        Optional callable that returns a query string used to retrieve
        relevant prior memories from mem0 to include in the summary. Defaults
        to "everything we've discussed in this session".
    """

    def __init__(
        self,
        memory: Any,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        summary_template: str = "[memory] {n} earlier turns archived. Recent context: {retrieved}",
        llm_summariser: Any = None,
        retrieve_query: Any = None,
    ) -> None:
        if memory is None:
            raise ValueError("Mem0Summarizer requires a memory instance")
        self._memory = memory
        self._user_id = user_id
        self._agent_id = agent_id
        self._session_id = session_id
        self._template = summary_template
        self._llm = llm_summariser
        self._retrieve_query = retrieve_query

    def __call__(self, messages: list[Message]) -> str:
        text = _messages_to_text(messages)
        self._memory.add(
            text,
            **{
                k: v
                for k, v in {
                    "user_id": self._user_id,
                    "agent_id": self._agent_id,
                    "session_id": self._session_id,
                }.items()
                if v is not None
            },
        )

        query = self._retrieve_query() if callable(self._retrieve_query) else None
        retrieved = self._search(query) if query else ""
        summary = self._llm(messages) if callable(self._llm) else ""
        return self._template.format(
            n=len(messages),
            retrieved=retrieved or "(none)",
            summary=summary,
        )

    def _search(self, query: str) -> str:
        try:
            results = self._memory.search(
                query,
                **{
                    k: v
                    for k, v in {
                        "user_id": self._user_id,
                        "agent_id": self._agent_id,
                    }.items()
                    if v is not None
                },
            )
        except Exception:
            return ""
        items: list[str] = []
        for r in results.get("results", []) if isinstance(results, dict) else results:
            if isinstance(r, dict) and "memory" in r:
                items.append(str(r["memory"]))
            else:
                items.append(str(r))
        return " | ".join(items[:5])


def _messages_to_text(messages: list[Message]) -> str:
    return "\n\n".join(f"{m.role}: {m.text()}" for m in messages if m.text())
