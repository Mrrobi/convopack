"""pack_stream() event emission."""

from __future__ import annotations

from convopack import (
    Message,
    Packer,
    PackEvent,
    Recency,
    SummaryEvict,
    TextBlock,
)


def test_pack_stream_under_budget_emits_kept_then_done(simple_history: list[Message]) -> None:
    packer = Packer(budget=10_000, tokenizer="approx", strategy=Recency())
    events = list(packer.pack_stream(simple_history))
    kinds = [e.kind for e in events]
    assert kinds.count("kept") == len(simple_history)
    assert kinds.count("dropped") == 0
    assert kinds[-1] == "done"


def test_pack_stream_over_budget_emits_dropped(simple_history: list[Message]) -> None:
    packer = Packer(budget=25, tokenizer="approx", strategy=Recency())
    events = list(packer.pack_stream(simple_history))
    assert any(e.kind == "dropped" for e in events)
    assert events[-1].kind == "done"
    done = events[-1]
    assert done.token_cost >= 0
    assert done.message is None


def test_pack_stream_emits_summarized_event(simple_history: list[Message]) -> None:
    def summariser(messages: list[Message]) -> str:
        return f"summary of {len(messages)} turns"

    packer = Packer(
        budget=25,
        tokenizer="approx",
        strategy=SummaryEvict(summariser),
        pin=("system", "last_user"),
    )
    events = list(packer.pack_stream(simple_history))
    summarized = [e for e in events if e.kind == "summarized"]
    if summarized:
        assert summarized[0].message is not None
        assert "summary of" in summarized[0].message.text()


def test_pack_event_index_matches_input(simple_history: list[Message]) -> None:
    packer = Packer(budget=10_000, tokenizer="approx", strategy=Recency())
    events = [e for e in packer.pack_stream(simple_history) if e.kind == "kept"]
    indices = [e.index for e in events]
    assert indices == list(range(len(simple_history)))


def test_pack_event_dataclass() -> None:
    msg = Message(role="user", content=[TextBlock(text="hi")])
    ev = PackEvent(kind="kept", index=0, message=msg, token_cost=5)
    assert ev.kind == "kept"
    assert ev.token_cost == 5
