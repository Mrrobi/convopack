# Strategies

A *strategy* decides which messages survive when the conversation exceeds the budget. All built-in strategies are framework-free and respect tool-pair atomicity.

## Recency

Keep the most recent chunks that fit. The default and cheapest option.

```python
from convopack import Packer, Recency

packer = Packer(budget=4000, strategy=Recency())
```

`Recency` walks from the newest chunk backward, keeping each one that still fits. Pinned chunks are always kept. The `min_keep` parameter forces at least N non-pinned chunks to survive even if it pushes the result over budget:

```python
Recency(min_keep=2)  # always keep the last two chunks even when over budget
```

Use when:

- A chat assistant whose latest user question is what matters.
- You don't have an LLM call budget for summarisation.

## FirstFit

The mirror of `Recency`: keep the *oldest* chunks that fit, drop the tail.

```python
from convopack import FirstFit

Packer(budget=4000, strategy=FirstFit())
```

Use when:

- Your prompt is heavy on early context — a long system prompt, few-shot examples, a research brief — and the trailing turns are less essential.
- You're building a retrieval-augmented loop where the retrieved documents come first and the chat tail is disposable.

## SummaryEvict

Drop older chunks like `Recency` would, but replace them with a single summary message at the head of the kept list.

```python
from convopack import SummaryEvict

def my_summariser(messages):
    # Call an LLM, return a short string.
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarise these turns in two sentences."},
            *to_openai(messages),
        ],
    )
    return response.choices[0].message.content

packer = Packer(
    budget=4000,
    strategy=SummaryEvict(my_summariser, reserve=500),
)
```

`reserve` tells the strategy how many tokens to leave for the summary itself; defaults to `budget // 10`.

`SummaryEvict` accepts an `async def` summariser too. Use `Packer.pack_async` from inside an event loop:

```python
async def my_async_summariser(messages):
    response = await async_client.messages.create(...)
    return response.content[0].text

packer = Packer(budget=4000, strategy=SummaryEvict(my_async_summariser))

result = await packer.pack_async(history)
```

Use when:

- You care about preserving information from older turns more than about cost.
- Your app is long-running (hours) and the early context matters semantically.

## Importance

You supply a score function; `convopack` drops the lowest-scoring chunks first.

```python
from convopack import Importance, Message

def my_scorer(msg: Message) -> float:
    if msg.metadata.get("starred"):
        return 100.0
    if msg.role == "system":
        return 50.0
    if msg.has_tool_use() or msg.has_tool_result():
        return 3.0
    return 1.0

packer = Packer(budget=4000, strategy=Importance(scorer=my_scorer))
```

A chunk's score is the *maximum* score of any message it contains, so a tool exchange inherits its highest-scored member.

Use when:

- You have application-specific signals (user starred a turn, this turn caused a tool call, this turn is from an admin) that should override pure recency.

A `default_scorer` is supplied if you omit `scorer`: system > tool exchanges > user > assistant.

## SemanticDedup

Drop near-duplicate messages by embedding cosine similarity, then defer to a fallback strategy for the remaining budget enforcement.

```python
from convopack import SemanticDedup

class OpenAIEmbedder:
    def embed(self, text: str) -> list[float]:
        return openai_client.embeddings.create(
            input=text, model="text-embedding-3-small"
        ).data[0].embedding

packer = Packer(
    budget=4000,
    strategy=SemanticDedup(
        OpenAIEmbedder(),
        threshold=0.92,
        fallback=Recency(),
    ),
)
```

`SemanticDedup` never deduplicates chunks containing tool calls or pinned messages — those flow through untouched. Among ordinary message chunks, only the *first* occurrence of each near-duplicate group survives.

Use when:

- An agent loop repeats itself ("Let me think... Let me check again...").
- Users paraphrase the same question multiple times.
- You're piping live transcript into the model and want to compact it.

## Writing your own

Any object satisfying the `Strategy` protocol works:

```python
class Strategy(Protocol):
    name: str

    def pack(
        self,
        messages: list[Message],
        *,
        budget: int,
        tokenizer: Tokenizer,
        pinned_indices: set[int],
    ) -> PackResult: ...
```

Just return a `PackResult`. Use `convopack._pairs.group_pairs(messages)` to get tool-pair-safe chunks so you don't have to reimplement pair tracking.
