"""Importance strategy."""

from __future__ import annotations

from convopack import Importance, Message, Packer, TextBlock


def test_high_score_message_kept() -> None:
    msgs = [
        Message(role="user", content=[TextBlock(text="a" * 80)]),
        Message(role="assistant", content=[TextBlock(text="b" * 80)]),
        Message(role="user", content=[TextBlock(text="c" * 80)]),
    ]

    def scorer(m: Message) -> float:
        return 100.0 if "c" in m.text() else 1.0

    packer = Packer(
        budget=40,
        tokenizer="approx",
        strategy=Importance(scorer=scorer),
        pin=(),
    )
    result = packer.pack(msgs)
    assert any(m.text() == "c" * 80 for m in result.kept)


def test_default_scorer_keeps_system(simple_history: list[Message]) -> None:
    packer = Packer(budget=20, tokenizer="approx", strategy=Importance(), pin=("system",))
    result = packer.pack(simple_history)
    assert any(m.role == "system" for m in result.kept)
