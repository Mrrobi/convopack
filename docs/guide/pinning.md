# Pinning

A *pinned* message is one the packer is forbidden to evict, even if it pushes the result over budget. Pinning is how you say "no matter what, keep this".

## Pin specs

The `pin` argument on `Packer` accepts a sequence of any of:

| Spec | Meaning |
|---|---|
| `"system"` | All messages with role `system`. |
| `"first_user"` | The first message with role `user`. |
| `"last_user"` | The last message with role `user`. |
| `"tool_results"` | Any message containing a `tool_use` or `tool_result` block. |
| `0`, `1`, `-1`, ... | A specific index in the input list (negative wraps). |

Examples:

```python
from convopack import Packer

# Keep the system prompt and the most recent user turn (typical chat assistant).
Packer(budget=4000, pin=("system", "last_user"))

# Keep the system prompt, the original question, and the latest one.
Packer(budget=4000, pin=("system", "first_user", "last_user"))

# Anchor message at index 2 explicitly.
Packer(budget=4000, pin=(2,))

# No pinning at all.
Packer(budget=4000, pin=())
```

## Pinning vs strategy

Pinning is orthogonal to strategy. Every strategy honours the pinned set; the strategy decides what happens with the *remaining* messages.

So `Recency` with `pin=("system",)` packs: "always keep the system prompt + the newest chunks that fit". `FirstFit` with the same pin packs: "always keep the system prompt + the oldest chunks that fit".

## Pinning and tool pairs

If you pin an index that lands inside a `tool_use` / `tool_result` chunk, the entire chunk is kept — pairs are atomic. You don't have to think about ID matching when pinning.

## Pinning vs `min_keep`

`Recency` and `FirstFit` both accept a `min_keep` parameter that says "keep at least N non-pinned chunks even if they overflow the budget". The two work together: pinning is an absolute floor; `min_keep` is a relative floor for the strategy's own decisions.

```python
# Always keep system + at least 2 non-system chunks even if it overflows.
Packer(budget=200, strategy=Recency(min_keep=2), pin=("system",))
```

## Adding your own pin signals

You can compute pin indices yourself and pass them in:

```python
critical = {i for i, m in enumerate(history) if m.metadata.get("starred")}
Packer(budget=4000, pin=tuple(critical))
```

Combine with named specs:

```python
Packer(budget=4000, pin=("system", *critical))
```
