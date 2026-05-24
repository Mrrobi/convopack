"""Mem0Summarizer with a fake mem0 Memory.

We don't require the real mem0 install for tests; the contract is just
``.add(text, **scope)`` and ``.search(query, **scope)``.
"""

from __future__ import annotations

from typing import Any

from convopack import Message, Packer, SummaryEvict, TextBlock
from convopack.integrations.mem0 import Mem0Summarizer


class FakeMemory:
    def __init__(self) -> None:
        self.added: list[tuple[str, dict[str, Any]]] = []
        self.searched: list[tuple[str, dict[str, Any]]] = []

    def add(self, text: str, **scope: Any) -> None:
        self.added.append((text, scope))

    def search(self, query: str, **scope: Any) -> dict[str, Any]:
        self.searched.append((query, scope))
        return {"results": [{"memory": "prior conversation about Oslo weather"}]}


def test_stores_evicted_messages() -> None:
    memory = FakeMemory()
    summariser = Mem0Summarizer(memory, user_id="alice")
    msgs = [
        Message(role="user", content=[TextBlock(text="earlier topic A")]),
        Message(role="assistant", content=[TextBlock(text="response A")]),
    ]
    out = summariser(msgs)
    assert "2 earlier turns archived" in out
    assert memory.added
    text, scope = memory.added[0]
    assert "earlier topic A" in text
    assert scope == {"user_id": "alice"}


def test_passes_all_scope_ids() -> None:
    memory = FakeMemory()
    summariser = Mem0Summarizer(memory, user_id="alice", agent_id="agent1", session_id="s1")
    summariser([Message(role="user", content=[TextBlock(text="x")])])
    assert memory.added[0][1] == {
        "user_id": "alice",
        "agent_id": "agent1",
        "session_id": "s1",
    }


def test_retrieve_query_searches_and_formats() -> None:
    memory = FakeMemory()
    summariser = Mem0Summarizer(
        memory,
        user_id="alice",
        retrieve_query=lambda: "weather",
        summary_template="[mem] {n} turns. recall: {retrieved}",
    )
    out = summariser([Message(role="user", content=[TextBlock(text="hi")])])
    assert memory.searched
    assert memory.searched[0][0] == "weather"
    assert "prior conversation about Oslo weather" in out


def test_works_as_summary_evict_summariser() -> None:
    memory = FakeMemory()
    packer = Packer(
        budget=20,
        tokenizer="approx",
        strategy=SummaryEvict(Mem0Summarizer(memory, user_id="alice")),
        pin=("system", "last_user"),
    )
    history = [
        Message(role="system", content=[TextBlock(text="be helpful")]),
        *[Message(role="user", content=[TextBlock(text=f"q{i}: " + "x" * 30)]) for i in range(5)],
        Message(role="user", content=[TextBlock(text="latest question")]),
    ]
    result = packer.pack(history)
    if result.dropped:
        assert memory.added
        assert result.summary is not None
        assert "archived" in result.summary.text()


def test_requires_memory_argument() -> None:
    import pytest

    with pytest.raises(ValueError, match="requires a memory"):
        Mem0Summarizer(None, user_id="alice")
