# Anthropic server-side managed context

Anthropic's `context_management` field lets the server compact a long conversation before generating, with no client-side compute. `convopack` is still useful as the client-side budget enforcer; the server handles whatever survives.

```python
import anthropic
from convopack import Packer, Recency

client = anthropic.Anthropic()
packer = Packer(
    budget=80_000,                       # generous client-side cap
    tokenizer="anthropic:claude-sonnet-4-6",
    strategy=Recency(),
    pin=("system", "last_user"),
    cache=True,                          # mark stable prefix
)

payload, context_mgmt = packer.pack_anthropic_managed(
    history,
    system="Be concise.",
    trigger_tokens=30_000,               # server compresses if input > 30k
    keep_tool_uses=3,                    # keep the 3 most recent tool exchanges
)

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system=payload.system,
    messages=payload.messages,
    context_management=context_mgmt,
)
```

## Roles of the two layers

| Layer | What it does | Where it runs |
|---|---|---|
| `Packer.pack_anthropic_managed` | Decide which messages to send. Pin system + last user. Emit cache markers. | Client (your machine). |
| `context_management` (Anthropic) | Drop oldest tool exchanges once input crosses a threshold. | Server (Anthropic). |

You can use both, either, or neither. The library default is to use neither (`cache=False`, no `context_management`), so existing code keeps working.

## Custom edits

`ContextManagementConfig.with_edit(...)` lets you add edits that aren't modelled yet:

```python
from convopack.anthropic_managed import ContextManagementConfig

cm = (
    ContextManagementConfig.empty()
    .with_edit({"type": "future_edit_type", "trigger": {"type": "input_tokens", "value": 50_000}})
)
```

That keeps the wrapper useful as Anthropic ships new edit types without requiring a `convopack` release.
