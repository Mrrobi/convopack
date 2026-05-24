# LiteLLM loop

LiteLLM normalises ~100 providers behind a single OpenAI-Chat-compatible `messages=[...]` interface. `convopack` is the layer above it that decides what to send.

```python
from litellm import completion
from convopack import Packer, Recency

packer = Packer(
    budget=8000,
    tokenizer="tiktoken:gpt-4o",
    strategy=Recency(),
    pin=("system", "last_user"),
    cache=True,
)

history = []

def respond(user_input: str) -> str:
    history.append({"role": "user", "content": user_input})
    packed = packer.pack_litellm(history)
    response = completion(model="claude-sonnet-4-6", messages=packed)
    reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    return reply
```

`pack_litellm` is essentially `pack_openai` because LiteLLM's wire format is the OpenAI Chat shape. The reason it exists as a separate method is so your code says what it means.

## Choosing the model at call time

LiteLLM routes by the `model=` parameter, not by the message shape, so you can keep one `Packer` and switch models freely:

```python
for model in ("gpt-4o", "claude-sonnet-4-6", "gemini-2.5-pro"):
    packed = packer.pack_litellm(history)
    response = completion(model=model, messages=packed)
    print(model, "->", response.choices[0].message.content[:60])
```

## Caching across providers

Both Anthropic and OpenAI honour their respective prompt caches when called through LiteLLM. The `cache=True` flag on `Packer` will still emit `cache_control` markers — LiteLLM forwards them to Anthropic, and the same flag also keeps prefixes byte-stable for OpenAI's automatic caching.

## Tool calls

LiteLLM passes through the OpenAI `tool_calls` / `tool` message shape unchanged for most providers. `convopack`'s tool-pair atomicity invariant holds end-to-end: if a `tool_use` survives the pack, its matching `tool` response survives with it.
