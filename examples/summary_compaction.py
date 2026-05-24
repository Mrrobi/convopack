"""Summarise evicted turns instead of dropping them silently."""

from __future__ import annotations

from convopack import Message, Packer, SummaryEvict


def fake_summariser(messages: list[Message]) -> str:
    """In real code: call an LLM to produce a short paragraph summary."""
    return f"{len(messages)} earlier turns covered weather, geography, and small talk."


def main() -> None:
    history: list[dict[str, object]] = [
        {"role": "system", "content": "You are helpful."},
        *[
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn #{i}: " + "x" * 40}
            for i in range(20)
        ],
        {"role": "user", "content": "What did we discuss?"},
    ]

    packer = Packer(
        budget=200,
        tokenizer="approx",
        strategy=SummaryEvict(fake_summariser),
        pin=("system", "last_user"),
    )
    packed = packer.pack_openai(history)
    for msg in packed:
        print(f"[{msg['role']:>10}] {str(msg.get('content', ''))[:70]}")


if __name__ == "__main__":
    main()
