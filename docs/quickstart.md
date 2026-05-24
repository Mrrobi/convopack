# Quickstart

## 1. Install

```bash
pip install "convopack[tiktoken]"
```

The base package is zero-dependency. Tokenizer extras are optional.

## 2. Build a packer

```python
from convopack import Packer, Recency

packer = Packer(
    budget=4000,
    tokenizer="tiktoken:gpt-4o",
    strategy=Recency(),
    pin=("system", "first_user"),
)
```

The four knobs:

| Argument | Purpose |
|---|---|
| `budget` | Maximum tokens you want the packed conversation to occupy. |
| `tokenizer` | A string spec or a `Tokenizer` instance. See [tokenizers guide](guide/tokenizers.md). |
| `strategy` | How to decide which messages survive. See [strategies guide](guide/strategies.md). |
| `pin` | Messages that must always be kept (`"system"`, `"first_user"`, `"last_user"`, `"tool_results"`, or an integer index). |

## 3. Pack

Convert your provider's messages to convopack's internal form, pack, and convert back:

=== "OpenAI"

    ```python
    history = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Norway?"},
        {"role": "assistant", "content": "A Nordic country..."},
        # ... 50 more turns ...
        {"role": "user", "content": "What did we discuss?"},
    ]

    packed = packer.pack_openai(history)
    # `packed` is a list of OpenAI Chat Completion dicts, fits the budget.

    response = openai.chat.completions.create(model="gpt-4o", messages=packed)
    ```

=== "Anthropic"

    ```python
    history = [
        {"role": "user", "content": "Norway?"},
        {"role": "assistant", "content": "A Nordic country..."},
        # ...
    ]

    payload = packer.pack_anthropic(history, system="You are concise.")
    # `payload.system` is a string; `payload.messages` is the list.

    response = anthropic.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=payload.system,
        messages=payload.messages,
    )
    ```

=== "Gemini"

    ```python
    contents = [
        {"role": "user", "parts": [{"text": "Norway?"}]},
        {"role": "model", "parts": [{"text": "A Nordic country..."}]},
        # ...
    ]

    payload = packer.pack_gemini(contents, system_instruction="Be concise.")

    response = model.generate_content(
        payload.contents,
        system_instruction=payload.system_instruction,
    )
    ```

## 4. Inspect what happened

`pack()` returns a `PackResult`:

```python
result = packer.pack_messages(history)  # internal form
print(f"kept {len(result.kept)} of {len(history)}")
print(f"tokens used: {result.token_count} / {result.budget}")
print(f"fits: {result.fits}")
```

Need a stream of events for a progress bar or audit log? Use `pack_stream`:

```python
for event in packer.pack_stream(history):
    print(event.kind, event.index, event.token_cost)
# kept 0 12
# kept 1 8
# dropped 2 24
# done -1 1873
```

## 5. Tool calls are safe by default

If your assistant emits `tool_use` and the next message is a `tool_result`, convopack keeps or drops them together — never splits a pair. This is enforced for every strategy:

```python
from convopack._pairs import validate_pairs

assert validate_pairs(result.kept) == []  # never has dangling tool_use IDs
```

That guarantee is what lets you point your existing agent loop at `Packer.pack_openai()` and stop worrying about 400s from half-evicted tool exchanges.

## Next

- [Strategies](guide/strategies.md) — pick the right one for your app.
- [Pinning](guide/pinning.md) — keep critical messages anchored.
- [Recipes](recipes/openai-loop.md) — full-loop examples.
