"""Synthetic conversation fixtures for benchmarking."""

from __future__ import annotations

import random

from convopack import Message, TextBlock, ToolResultBlock, ToolUseBlock


def make_history(
    turns: int,
    *,
    tool_density: float = 0.2,
    seed: int = 0,
) -> list[Message]:
    """Generate a conversation with `turns` exchanges.

    Each exchange is one user message + one assistant message. Some assistant
    messages are tool_use followed by tool_result (next user turn).
    """
    rng = random.Random(seed)
    msgs: list[Message] = [
        Message(role="system", content=[TextBlock(text="You are a helpful assistant.")])
    ]
    for i in range(turns):
        if rng.random() < tool_density:
            msgs.append(
                Message(
                    role="user",
                    content=[
                        TextBlock(text=f"Tool turn {i}: " + "lorem ipsum " * rng.randint(3, 15))
                    ],
                )
            )
            msgs.append(
                Message(
                    role="assistant",
                    content=[ToolUseBlock(id=f"t{i}", name="search", input={"q": f"query {i}"})],
                )
            )
            msgs.append(
                Message(
                    role="tool",
                    content=[
                        ToolResultBlock(tool_use_id=f"t{i}", content="result " * rng.randint(5, 30))
                    ],
                )
            )
            msgs.append(
                Message(
                    role="assistant",
                    content=[TextBlock(text=f"Answer {i}: " + "explanation " * rng.randint(5, 25))],
                )
            )
        else:
            msgs.append(
                Message(
                    role="user",
                    content=[TextBlock(text=f"Question {i}: " + "context " * rng.randint(3, 20))],
                )
            )
            msgs.append(
                Message(
                    role="assistant",
                    content=[TextBlock(text=f"Answer {i}: " + "response " * rng.randint(5, 30))],
                )
            )
    return msgs
