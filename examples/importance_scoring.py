"""Custom importance scorer: keep messages tagged as 'memorable' first."""

from __future__ import annotations

from convopack import Importance, Message, Packer, TextBlock


def memorable_first(msg: Message) -> float:
    if msg.metadata.get("memorable"):
        return 100.0
    if msg.role == "system":
        return 50.0
    if msg.role == "user":
        return 2.0
    return 1.0


def main() -> None:
    history = [
        Message(role="system", content=[TextBlock(text="Be concise.")]),
        Message(
            role="user", content=[TextBlock(text="my name is robi")], metadata={"memorable": True}
        ),
        Message(role="assistant", content=[TextBlock(text="hi robi")]),
        *[
            Message(
                role="user" if i % 2 == 0 else "assistant", content=[TextBlock(text="filler " * 20)]
            )
            for i in range(20)
        ],
        Message(role="user", content=[TextBlock(text="what is my name?")]),
    ]

    packer = Packer(
        budget=300,
        tokenizer="approx",
        strategy=Importance(scorer=memorable_first),
        pin=(),
    )
    result = packer.pack(history)
    print(
        f"kept {len(result.kept)} of {len(history)} (budget={result.budget}, used={result.token_count})"
    )
    for msg in result.kept:
        print(f"  [{msg.role:>10}] {msg.text()[:60]}")


if __name__ == "__main__":
    main()
