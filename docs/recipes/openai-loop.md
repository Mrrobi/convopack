# OpenAI chat loop

A complete conversational loop with `convopack` in front of every model call.

```python
import openai
from convopack import Packer, Recency

client = openai.OpenAI()
packer = Packer(
    budget=8000,
    tokenizer="tiktoken:gpt-4o",
    strategy=Recency(),
    pin=("system", "last_user"),
)

history: list[dict] = [
    {"role": "system", "content": "You are a helpful, concise assistant."},
]

while True:
    user_input = input("you> ").strip()
    if not user_input:
        break
    history.append({"role": "user", "content": user_input})

    packed = packer.pack_openai(history)
    response = client.chat.completions.create(model="gpt-4o", messages=packed)
    reply = response.choices[0].message.content

    print(f"bot> {reply}\n")
    history.append({"role": "assistant", "content": reply})
```

## What's happening on every turn

1. The full `history` keeps growing in memory; we never delete from it.
2. Before each model call, `packer.pack_openai(history)` produces a list that fits the 8 K budget.
3. The system prompt and the latest user turn are guaranteed to survive thanks to `pin=("system", "last_user")`.
4. The newest other turns are kept until the budget runs out, and the older ones are dropped silently.

You can swap in any other strategy without changing the loop:

```python
packer = Packer(budget=8000, strategy=FirstFit(), pin=("system",))
```

## Stopping silent eviction

If you want to know when something gets dropped, swap `pack_openai` for the lower-level path and inspect the result:

```python
from convopack.providers import from_openai, to_openai

result = packer.pack(from_openai(history))
if result.dropped:
    print(f"warning: dropped {len(result.dropped)} older turns")
packed = to_openai(result.kept)
```

## Combining with summarisation

When silent eviction is unacceptable, switch to `SummaryEvict`:

```python
from convopack import SummaryEvict

def summarise(messages):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarise these turns in two sentences."},
            *to_openai(messages),
        ],
    )
    return response.choices[0].message.content

packer = Packer(
    budget=8000,
    strategy=SummaryEvict(summarise, reserve=500),
    pin=("system", "last_user"),
)
```

The summariser is called only when the budget is exceeded, so most turns add nothing to your bill.
