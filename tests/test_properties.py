"""Hypothesis property tests for invariants of pair grouping and packing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hypothesis import given, settings
from hypothesis import strategies as st

from convopack import Message, Packer, Recency, TextBlock, ToolResultBlock, ToolUseBlock
from convopack._pairs import group_pairs, validate_pairs
from convopack.providers import from_anthropic, from_openai, to_anthropic, to_openai

if TYPE_CHECKING:
    pass


ascii_text = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1, max_size=40
)
short_id = st.text(
    alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=2, max_size=6
)
roles_no_tool = st.sampled_from(["system", "user", "assistant"])


@st.composite
def message_strategy(draw: st.DrawFn) -> Message:
    role = draw(roles_no_tool)
    return Message(role=role, content=[TextBlock(text=draw(ascii_text))])


@st.composite
def history_with_tools(draw: st.DrawFn) -> list[Message]:
    """Generate a history with optional well-formed tool_use/tool_result pairs."""
    msgs: list[Message] = []
    n = draw(st.integers(min_value=0, max_value=8))
    used_ids: set[str] = set()
    for _ in range(n):
        kind = draw(st.sampled_from(["text", "tool_pair"]))
        if kind == "text":
            msgs.append(draw(message_strategy()))
        else:
            tid = f"t_{draw(short_id)}_{len(used_ids)}"
            used_ids.add(tid)
            msgs.append(
                Message(
                    role="assistant",
                    content=[ToolUseBlock(id=tid, name="x", input={"k": "v"})],
                )
            )
            msgs.append(
                Message(
                    role="tool",
                    content=[ToolResultBlock(tool_use_id=tid, content="ok")],
                )
            )
    return msgs


@given(history_with_tools())
@settings(max_examples=200, deadline=None)
def test_group_pairs_preserves_order(history: list[Message]) -> None:
    chunks = group_pairs(history)
    flat = [m for c in chunks for m in c.messages]
    assert flat == history


@given(history_with_tools())
@settings(max_examples=200, deadline=None)
def test_well_formed_history_has_no_dangling_tools(history: list[Message]) -> None:
    """If we only generate matched pairs, validate_pairs returns empty."""
    assert validate_pairs(history) == []


@given(history_with_tools())
@settings(max_examples=200, deadline=None)
def test_chunk_indices_are_contiguous(history: list[Message]) -> None:
    chunks = group_pairs(history)
    flat_idx = [i for c in chunks for i in c.indices]
    assert flat_idx == list(range(len(history)))


@given(history_with_tools(), st.integers(min_value=1, max_value=50_000))
@settings(max_examples=150, deadline=None)
def test_pack_kept_has_no_dangling_pairs(history: list[Message], budget: int) -> None:
    """Recency must never emit a kept list with a dangling tool_use."""
    packer = Packer(budget=budget, tokenizer="approx", strategy=Recency())
    result = packer.pack(history)
    assert validate_pairs(result.kept) == []


@given(history_with_tools(), st.integers(min_value=1, max_value=50_000))
@settings(max_examples=150, deadline=None)
def test_pack_preserves_relative_order(history: list[Message], budget: int) -> None:
    packer = Packer(budget=budget, tokenizer="approx", strategy=Recency())
    result = packer.pack(history)
    original_positions = {id(m): i for i, m in enumerate(history)}
    kept_positions = [original_positions[id(m)] for m in result.kept]
    assert kept_positions == sorted(kept_positions)


@given(history_with_tools(), st.integers(min_value=1, max_value=50_000))
@settings(max_examples=150, deadline=None)
def test_pack_kept_plus_dropped_equals_input(history: list[Message], budget: int) -> None:
    packer = Packer(budget=budget, tokenizer="approx", strategy=Recency())
    result = packer.pack(history)
    assert sorted(map(id, result.kept + result.dropped)) == sorted(map(id, history))


@given(history_with_tools())
@settings(max_examples=100, deadline=None)
def test_openai_roundtrip_invariant(history: list[Message]) -> None:
    out = to_openai(history)
    back = from_openai(out)
    assert len(back) == len(history)
    for orig, rt in zip(history, back, strict=True):
        assert orig.role == rt.role
        assert orig.text() == rt.text()


@given(history_with_tools())
@settings(max_examples=100, deadline=None)
def test_anthropic_roundtrip_invariant(history: list[Message]) -> None:
    payload = to_anthropic(history)
    back = from_anthropic(payload.messages, system=payload.system or None)
    text_history = [m for m in history if m.role != "system"]
    text_back = [m for m in back if m.role != "system"]
    assert len(text_back) == len(text_history)
