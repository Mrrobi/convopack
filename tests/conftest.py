"""Shared fixtures."""

from __future__ import annotations

import pytest

from convopack import (
    ApproxTokenizer,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)


@pytest.fixture
def approx() -> ApproxTokenizer:
    return ApproxTokenizer()


@pytest.fixture
def simple_history() -> list[Message]:
    return [
        Message(role="system", content=[TextBlock(text="You are a helpful assistant.")]),
        Message(role="user", content=[TextBlock(text="hello, my name is robi")]),
        Message(role="assistant", content=[TextBlock(text="hi robi, how can i help?")]),
        Message(role="user", content=[TextBlock(text="what is 2 + 2?")]),
        Message(role="assistant", content=[TextBlock(text="2 + 2 is 4.")]),
        Message(role="user", content=[TextBlock(text="thanks!")]),
    ]


@pytest.fixture
def tool_history() -> list[Message]:
    return [
        Message(role="system", content=[TextBlock(text="You can use tools.")]),
        Message(role="user", content=[TextBlock(text="weather in oslo")]),
        Message(
            role="assistant",
            content=[
                TextBlock(text="Looking up..."),
                ToolUseBlock(id="t1", name="weather", input={"city": "oslo"}),
            ],
        ),
        Message(
            role="tool",
            content=[ToolResultBlock(tool_use_id="t1", content="rainy, 8C")],
        ),
        Message(role="assistant", content=[TextBlock(text="It's rainy and 8C in Oslo.")]),
        Message(role="user", content=[TextBlock(text="and bergen?")]),
        Message(
            role="assistant",
            content=[ToolUseBlock(id="t2", name="weather", input={"city": "bergen"})],
        ),
        Message(
            role="tool",
            content=[ToolResultBlock(tool_use_id="t2", content="cloudy, 10C")],
        ),
        Message(role="assistant", content=[TextBlock(text="Bergen is cloudy, 10C.")]),
    ]
