# Async summarisation

When your application is async-first (FastAPI, asyncio agents, etc.), `convopack` accepts an `async def` summariser and runs it without spinning up a separate event loop.

```python
import asyncio
import anthropic
from convopack import Packer, SummaryEvict, Message
from convopack.providers import from_anthropic, to_openai

async_client = anthropic.AsyncAnthropic()

async def summarise(messages: list[Message]) -> str:
    response = await async_client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": "Summarise the following turns in two sentences:\n\n"
                + "\n".join(f"{m.role}: {m.text()}" for m in messages),
            }
        ],
    )
    return response.content[0].text

packer = Packer(
    budget=8000,
    tokenizer="approx",
    strategy=SummaryEvict(summarise),
    pin=("system", "last_user"),
)

async def main():
    history = [...]  # any list[Message] or use from_openai/from_anthropic on a dict list
    result = await packer.pack_async(history)
    print(f"kept {len(result.kept)} of {len(history)}")
    if result.summary:
        print(f"summary: {result.summary.text()}")

asyncio.run(main())
```

## Sync and async on the same `SummaryEvict`

The same `SummaryEvict(summarise)` instance can be called from either `pack()` or `pack_async()`. The strategy detects whether the summariser is a coroutine and routes accordingly:

| Call site | Summariser shape | What happens |
|---|---|---|
| `packer.pack(...)` (sync) | `def summarise(msgs) -> str` | Direct call |
| `packer.pack(...)` (sync) | `async def summarise(msgs)` outside an event loop | `asyncio.run(...)` is used |
| `packer.pack(...)` (sync) | `async def summarise(msgs)` inside an event loop | `RuntimeError` — switch to `pack_async` |
| `await packer.pack_async(...)` | either | Works |

## Cost tip

The summariser is called only when the budget is exceeded, so a chatty user costs a summary call once every few turns instead of every turn. You can stack `SemanticDedup` *before* `SummaryEvict` to compress before summarising:

```python
from convopack import SemanticDedup, SummaryEvict

packer = Packer(
    budget=8000,
    strategy=SemanticDedup(embedder, fallback=SummaryEvict(summarise)),
)
```
