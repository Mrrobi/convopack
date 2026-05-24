# Custom importance scoring

When recency and tool-pair atomicity aren't enough — when *this particular turn matters* — supply a scorer.

## Star-aware scoring

A user clicks a star on certain turns in your UI. Those turns must survive every pack.

```python
from convopack import Importance, Message, Packer

def starred_first(msg: Message) -> float:
    if msg.metadata.get("starred"):
        return 1_000.0
    if msg.role == "system":
        return 100.0
    if msg.has_tool_use() or msg.has_tool_result():
        return 50.0
    if msg.role == "user":
        return 5.0
    return 1.0  # plain assistant text

packer = Packer(
    budget=4000,
    strategy=Importance(scorer=starred_first),
    pin=("system",),
)
```

`Importance` drops chunks in ascending score order until the result fits. A chunk inherits the maximum score of its messages, so a `tool_use` + `tool_result` pair takes the max of both.

## Topic-aware scoring with embeddings

You can run a quick embedding distance from the latest user turn and score recent topical hits higher:

```python
from convopack import Message

class TopicScorer:
    def __init__(self, embedder, anchor_text: str):
        self.embedder = embedder
        self.anchor = embedder.embed(anchor_text)

    def __call__(self, msg: Message) -> float:
        base = 1.0
        if msg.role == "system":
            base = 100.0
        if msg.has_tool_use() or msg.has_tool_result():
            base = 50.0
        text = msg.text()
        if not text:
            return base
        from convopack.embedders import cosine
        return base + cosine(self.embedder.embed(text), self.anchor) * 10.0

scorer = TopicScorer(my_embedder, anchor_text=latest_user_question)
packer = Packer(budget=4000, strategy=Importance(scorer=scorer), pin=("system",))
```

Be careful with cost: scoring is called once per message per pack. Cache embeddings on the message metadata if turns are long-lived.

## Application-defined signals

Anything you can read off a `Message` is fair game. Examples:

- A turn known to come from an authenticated admin.
- A turn produced by a specific tool call (e.g. `web_search` results worth keeping).
- A user "remember this" command that adds `metadata["pinned_by_user"] = True`.
- A turn within N seconds of a "good answer" thumbs-up.

```python
def importance(msg):
    score = 1.0
    if msg.metadata.get("auth_admin"):
        score += 50.0
    if msg.metadata.get("source") == "web_search":
        score += 25.0
    if msg.metadata.get("pinned_by_user"):
        score = 1000.0
    return score
```

## Combining with pinning

`Importance` and pinning aren't mutually exclusive. Use pinning for "no matter what, never drop" and the scorer for relative ranking among non-pinned chunks. The cleanest pattern is:

```python
Packer(
    budget=4000,
    strategy=Importance(scorer=my_scorer),
    pin=("system", "last_user"),
)
```

Now `last_user` and `system` are absolute floors; everything else competes on score.
