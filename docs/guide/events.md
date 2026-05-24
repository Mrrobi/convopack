# Streaming pack events

When a pack happens you usually only see the final result. `Packer.pack_stream()` turns the operation into a stream of `PackEvent` objects you can render, log, or attach to a span.

## Event kinds

| Kind | Meaning |
|---|---|
| `kept` | This message survived the pack. |
| `dropped` | This message was evicted. |
| `summarized` | A synthetic summary message replaces some dropped block (only from `SummaryEvict`). |
| `done` | Terminal event with the total token count (`message=None`). |

Each event carries:

- `index` — the message's position in the original input, or `-1` for synthetic events.
- `message` — the message itself, or `None` on `done`.
- `token_cost` — tokens contributed by this message. For `done`, the total kept tokens.

## Drawing a progress bar

```python
import rich.progress

with rich.progress.Progress() as bar:
    task = bar.add_task("packing", total=len(history))
    for ev in packer.pack_stream(history):
        if ev.kind in ("kept", "dropped"):
            bar.update(task, advance=1)
        if ev.kind == "done":
            print(f"used {ev.token_cost} tokens")
```

## Logging eviction decisions

```python
import logging

logger = logging.getLogger("convopack")

for ev in packer.pack_stream(history):
    if ev.kind == "dropped":
        logger.info("evicted msg %d (%d tokens): %r", ev.index, ev.token_cost, ev.message.text()[:60])
    if ev.kind == "summarized":
        logger.info("inserted summary (%d tokens)", ev.token_cost)
```

## Auditing in a server

```python
audit_log.write({
    "request_id": req_id,
    "events": [
        {"kind": e.kind, "index": e.index, "tokens": e.token_cost}
        for e in packer.pack_stream(history)
    ],
})
```

## A note on ordering

`pack_stream` yields all `kept` events in input-order first (including any inserted `summarized` event at its position in the kept list), then all `dropped` events in input-order, then a single `done`. This makes it cheap to feed a UI that wants the "current state" to render before showing the diff against what was lost.
