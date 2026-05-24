"""Use convopack inside an OpenAI Chat Completions loop.

Pass raw OpenAI message dicts straight into ``Packer.pack_openai`` and get back
a list of dicts you can hand to ``client.chat.completions.create``.
"""

from __future__ import annotations

from convopack import Packer, Recency


def main() -> None:
    history: list[dict[str, object]] = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Tell me about Norway."},
        {"role": "assistant", "content": "Norway is a Nordic country..."},
        {"role": "user", "content": "Tell me about Sweden."},
        {"role": "assistant", "content": "Sweden is a Scandinavian country..."},
        {"role": "user", "content": "And Denmark?"},
    ]

    packer = Packer(
        budget=200,
        tokenizer="approx",
        strategy=Recency(),
        pin=("system", "last_user"),
    )
    packed = packer.pack_openai(history)
    for msg in packed:
        print(f"[{msg['role']:>10}] {str(msg.get('content', ''))[:60]}")


if __name__ == "__main__":
    main()
