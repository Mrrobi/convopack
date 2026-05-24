# Prompt caching

Most providers will discount your bill by 75-90% if the leading slice of every request is identical between calls. `convopack` knows which chunks are pinned, so it's the natural place to decide where the cache breakpoints go.

## Anthropic — explicit markers

Anthropic supports up to **four** `cache_control` markers per request. Each marker says "everything from the start of the conversation up to and including this block is cacheable."

Enable it with one flag:

```python
from convopack import Packer

packer = Packer(
    budget=8000,
    tokenizer="anthropic:claude-sonnet-4-6",
    pin=("system", "first_user", "last_user"),
    cache=True,  # << emit cache_control on stable pinned chunks
)

payload = packer.pack_anthropic(history, system="You are concise.")
```

What happens:

- `system` and `first_user` are marked stable → they get `cache_control`.
- `last_user` is **never** marked — it changes every turn and would bust the cache.
- The marker count is capped at 4 automatically, in prefix order.

Result: from turn 2 onward, the prefix that includes your system prompt and original question is served from Anthropic's cache, at ~10% of the input cost.

## OpenAI — automatic prefix caching

OpenAI doesn't take markers. Instead, identical prefixes ≥ 1024 tokens are cached automatically. The trick: make sure your prefix is byte-identical between calls.

`convopack`'s pinning + stable chunk ordering does the work. Use `cache_prefix_signature` to assert the prefix hasn't drifted:

```python
packer = Packer(budget=8000, tokenizer="tiktoken:gpt-4o", cache=True)

sig = packer.cache_prefix_signature(history)
# Persist `sig` in your request log. If it changes, your prefix drifted
# and the next call won't hit OpenAI's cache.
```

You can also inspect what would be cached at any time:

```python
info = packer.cache_info(history)
# {
#   "markers": [0, 1],
#   "marked_messages": 2,
#   "marked_tokens": 412,
#   "total_tokens": 7821,
#   "hit_ratio": 0.053,
#   "prefix_signature": "a4f3e..."
# }
```

`hit_ratio` is a rough estimate of what fraction of the *next* call's prompt tokens will hit the cache.

## What makes a prefix "stable"

`convopack`'s rule for emitting a marker is conservative on purpose:

| Pin spec | Marked? | Why |
|---|---|---|
| `"system"` | yes | System prompts don't change between turns. |
| `"first_user"` | yes | The original question is the most reused token block. |
| `"tool_results"` | no | Tool results vary per call. |
| `"last_user"` | no | Changes every turn — would invalidate the cache. |
| `int` index | yes | You opted in explicitly. |

If you have a long, stable few-shot block, pin it with an explicit index:

```python
Packer(pin=("system", 1, 2, "first_user"), cache=True)
# Indices 1 and 2 (whatever they are) get markers too, up to the 4-marker cap.
```

## Combining with other features

Caching composes with every strategy and tokenizer. The packer first decides which messages survive the budget, then independently decides which of the survivors get a cache marker. You can stack `SummaryEvict` and caching — the summary message is treated as an ordinary message (not marked).

## Gotchas

- **Caching is per-account, per-model.** Switching models invalidates the cache. Stay on one model for the cached endpoint.
- **System message edits break the cache.** If you A/B test prompts, expect each variant to start cold.
- **Tool definitions count toward the prefix.** Changing your tool list invalidates the cache for `tool_use` exchanges.
- **Anthropic 4-marker cap is hard.** Don't pin more than 4 things and expect them all to be cached.
