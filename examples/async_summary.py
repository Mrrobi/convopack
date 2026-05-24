"""Async summariser used from inside a running event loop."""

from __future__ import annotations

import asyncio

from convopack import Message, Packer, SummaryEvict, TextBlock


async def async_summariser(messages: list[Message]) -> str:
    await asyncio.sleep(0)
    return f"({len(messages)} earlier turns about random topics)"


async def main() -> None:
    history = [
        Message(role="system", content=[TextBlock(text="Be helpful.")]),
        *[
            Message(role="user" if i % 2 == 0 else "assistant", content=[TextBlock(text="x " * 30)])
            for i in range(30)
        ],
        Message(role="user", content=[TextBlock(text="summary please?")]),
    ]
    packer = Packer(
        budget=300,
        tokenizer="approx",
        strategy=SummaryEvict(async_summariser),
        pin=("system", "last_user"),
    )
    result = await packer.pack_async(history)
    print(
        f"kept {len(result.kept)} of {len(history)}; summary present: {result.summary is not None}"
    )
    if result.summary:
        print(f"summary: {result.summary.text()}")


if __name__ == "__main__":
    asyncio.run(main())
