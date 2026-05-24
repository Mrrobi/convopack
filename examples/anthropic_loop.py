"""Use convopack inside an Anthropic Messages API loop.

``Packer.pack_anthropic`` returns an :class:`AnthropicPayload` with a ``system``
string and a ``messages`` list, ready to pass to ``client.messages.create``.
"""

from __future__ import annotations

from convopack import Packer, Recency


def main() -> None:
    history: list[dict[str, object]] = [
        {"role": "user", "content": "weather in oslo?"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "weather", "input": {"city": "oslo"}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "rainy, 8C"}],
        },
        {"role": "assistant", "content": "It's rainy and 8C."},
        {"role": "user", "content": "and bergen?"},
    ]

    packer = Packer(
        budget=120,
        tokenizer="approx",
        strategy=Recency(),
        pin=("last_user",),
    )
    payload = packer.pack_anthropic(history, system="Be concise.")
    print("system:", payload.system)
    for msg in payload.messages:
        kind = msg["content"][0].get("type", "text") if isinstance(msg["content"], list) else "text"
        print(f"  [{msg['role']:>9}] ({kind})")


if __name__ == "__main__":
    main()
