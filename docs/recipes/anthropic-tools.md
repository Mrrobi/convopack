# Anthropic agent with tools

This is the test case where most context-packing libraries break: a multi-turn agent that issues `tool_use` blocks and receives `tool_result` blocks. If the packer evicts only half of a pair, the next `messages.create` call returns a 400.

`convopack` makes that impossible.

```python
import anthropic
from convopack import Packer, Recency
from convopack.providers import from_anthropic, to_anthropic

client = anthropic.Anthropic()
packer = Packer(
    budget=20_000,
    tokenizer="anthropic:claude-sonnet-4-6",
    strategy=Recency(),
    pin=("system", "last_user"),
)

TOOLS = [
    {
        "name": "weather",
        "description": "Get the weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
]

history: list[dict] = []
system = "You are a helpful assistant with weather tools."

def call_tool(name: str, payload: dict) -> str:
    if name == "weather":
        return f"It's rainy in {payload['city']}, 9 °C."
    return f"Unknown tool: {name}"

def run_turn(user_text: str) -> str:
    history.append({"role": "user", "content": user_text})

    while True:
        payload = packer.pack_anthropic(history, system=system)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=payload.system,
            messages=payload.messages,
            tools=TOOLS,
        )

        # Record the assistant turn in our local history (full, unpacked).
        history.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "tool_use":
            # Run every tool_use block, then append the matching tool_result message.
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = call_tool(block.name, block.input)
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": result}
                    )
            history.append({"role": "user", "content": tool_results})
            continue

        # Final response.
        return "".join(b.text for b in response.content if b.type == "text")

print(run_turn("What's the weather in Oslo and Bergen?"))
```

## Why this is safe

After many turns, `history` may contain a sprawl of `tool_use` and `tool_result` pairs. `packer.pack_anthropic(history, system=...)` runs through the chunker, which guarantees:

- Every `tool_use` survives if and only if its matching `tool_result` survives.
- Parallel tool calls in a single assistant turn (multiple `tool_use` blocks) are all kept together with their matching results.

You never have to manually scan for orphaned IDs.

## Validating in tests

When you write integration tests for your agent, assert the invariant directly:

```python
from convopack._pairs import validate_pairs

result = packer.pack(from_anthropic(history, system=system))
assert validate_pairs(result.kept) == []  # no dangling tool_use IDs
```

## Combining with `SummaryEvict`

Long-running agents accumulate enormous tool histories. Stack `SummaryEvict` on top to compress the older tool exchanges into a single summary while keeping the recent ones intact:

```python
from convopack import SummaryEvict

async def summarise(messages):
    response = await async_client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=400,
        messages=[{"role": "user", "content": "Summarise: " + str(messages)}],
    )
    return response.content[0].text

packer = Packer(
    budget=20_000,
    tokenizer="anthropic:claude-sonnet-4-6",
    strategy=SummaryEvict(summarise),
    pin=("system", "last_user"),
)

payload = await packer.pack_async(from_anthropic(history, system=system))
```
