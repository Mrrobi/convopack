"""Prompt-cache annotation system + Anthropic cache_control emission."""

from __future__ import annotations

from convopack import (
    Message,
    Packer,
    TextBlock,
    history_hash,
)
from convopack.caching import stable_marker_indices
from convopack.providers import to_anthropic


def _hist() -> list[Message]:
    return [
        Message(role="system", content=[TextBlock(text="You are a helpful assistant.")]),
        Message(role="user", content=[TextBlock(text="What is Norway?")]),
        Message(role="assistant", content=[TextBlock(text="A Nordic country.")]),
        Message(role="user", content=[TextBlock(text="And Sweden?")]),
    ]


def test_cache_markers_empty_when_disabled() -> None:
    packer = Packer(budget=10_000, tokenizer="approx")
    result = packer.pack(_hist())
    assert result.cache_markers == []


def test_cache_markers_target_system_and_first_user() -> None:
    packer = Packer(
        budget=10_000,
        tokenizer="approx",
        pin=("system", "first_user"),
        cache=True,
    )
    result = packer.pack(_hist())
    assert 0 in result.cache_markers  # system
    assert 1 in result.cache_markers  # first user


def test_cache_markers_ignore_last_user() -> None:
    """last_user is volatile -- never marked even if pinned, to keep cache stable."""
    packer = Packer(
        budget=10_000,
        tokenizer="approx",
        pin=("system", "last_user"),
        cache=True,
    )
    result = packer.pack(_hist())
    last_user_idx = len(result.kept) - 1
    assert last_user_idx not in result.cache_markers


def test_cache_markers_capped_at_four() -> None:
    msgs = [Message(role="system", content=[TextBlock(text=f"s{i}")]) for i in range(8)]
    msgs.append(Message(role="user", content=[TextBlock(text="hi")]))
    indices = stable_marker_indices(msgs, pin_specs=("system", "first_user"))
    assert len(indices) <= 4


def test_anthropic_emits_cache_control_on_marked_messages() -> None:
    packer = Packer(
        budget=10_000,
        tokenizer="approx",
        pin=("system", "first_user"),
        cache=True,
    )
    result = packer.pack(_hist())
    payload = to_anthropic(result.kept, cache_markers=result.cache_markers)

    # System should be list-of-blocks because a marker was applied
    assert isinstance(payload.system, list)
    assert any(p.get("cache_control") for p in payload.system)
    # First user message should have cache_control on its last content block
    user_msg = payload.messages[0]
    assert user_msg["content"][-1].get("cache_control") == {"type": "ephemeral"}


def test_anthropic_no_markers_keeps_system_as_string() -> None:
    packer = Packer(budget=10_000, tokenizer="approx", pin=("system",), cache=False)
    result = packer.pack(_hist())
    payload = to_anthropic(result.kept, cache_markers=result.cache_markers)
    assert isinstance(payload.system, str)
    assert payload.system == "You are a helpful assistant."


def test_pack_anthropic_threads_cache_markers() -> None:
    raw = [
        {"role": "user", "content": "What is Norway?"},
        {"role": "assistant", "content": "A Nordic country."},
        {"role": "user", "content": "And Sweden?"},
    ]
    packer = Packer(
        budget=10_000,
        tokenizer="approx",
        pin=("system", "first_user"),
        cache=True,
    )
    payload = packer.pack_anthropic(raw, system="be brief")
    # System with marker becomes list form
    assert isinstance(payload.system, list)
    sys_block = payload.system[0]
    assert sys_block["cache_control"] == {"type": "ephemeral"}


def test_history_hash_stable_across_call_with_appended_turn() -> None:
    """Prefix hash should match if the prefix hasn't changed."""
    base = _hist()
    extended = [*base, Message(role="assistant", content=[TextBlock(text="A Scandinavian one.")])]
    # Hash of the prefix is the same regardless of what comes after
    assert history_hash(base) == history_hash(extended[: len(base)])


def test_cache_prefix_signature_stable_across_calls() -> None:
    """Adding turns after the prefix must not change the signature."""
    packer = Packer(
        budget=10_000,
        tokenizer="approx",
        pin=("system", "first_user"),
        cache=True,
    )
    base = _hist()
    extended = [*base, Message(role="assistant", content=[TextBlock(text="extra")])]
    assert packer.cache_prefix_signature(base) == packer.cache_prefix_signature(extended)


def test_cache_prefix_signature_changes_when_system_changes() -> None:
    packer = Packer(
        budget=10_000,
        tokenizer="approx",
        pin=("system", "first_user"),
        cache=True,
    )
    a = _hist()
    b = list(a)
    b[0] = Message(role="system", content=[TextBlock(text="A DIFFERENT system prompt.")])
    assert packer.cache_prefix_signature(a) != packer.cache_prefix_signature(b)


def test_cache_info_returns_expected_shape() -> None:
    packer = Packer(
        budget=10_000,
        tokenizer="approx",
        pin=("system", "first_user"),
        cache=True,
    )
    info = packer.cache_info(_hist())
    assert set(info.keys()) == {
        "markers",
        "marked_messages",
        "marked_tokens",
        "total_tokens",
        "hit_ratio",
        "prefix_signature",
    }
    assert info["marked_messages"] >= 1
    assert 0 <= info["hit_ratio"] <= 1
    assert len(info["prefix_signature"]) == 64
