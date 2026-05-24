# Why convopack

Context packing isn't a new problem, and `convopack` isn't the first attempt. This page is an honest comparison with the alternatives and a clear statement of when *not* to use it.

## The problem in one paragraph

An LLM application keeps a conversation history. Each turn appends a message. Eventually the conversation exceeds the model's context window and either the API rejects the call, or the SDK silently truncates it, or — worst of all — your tool calls get half-evicted and the next model response is a 400. Context packing is the layer that decides what stays and what goes.

## What context packing must get right

Three things, in priority order:

1. **Tool-pair atomicity.** A `tool_use` block and its matching `tool_result` are either both kept or both dropped. Splitting them produces a malformed conversation that providers reject.
2. **Pinning.** The system prompt, the original question, the latest user turn — these are non-negotiable.
3. **A reasonable eviction policy** for the messages that don't fall into either of the above.

The third is where everyone differs.

## Comparison

| Aspect | convopack | LangChain `trim_messages` | mem0 | Anthropic `clear_tool_uses_*` |
|---|---|---|---|---|
| Framework-free | yes | no (LangChain Core) | no (own runtime) | yes |
| Multi-provider message shapes | OpenAI, Anthropic, Gemini | LangChain BaseMessage only | n/a (own memory model) | Anthropic only |
| Tool-pair atomicity guarantee | yes, enforced for every strategy | no (default trimmer is per-message) | n/a | yes, but only for one strategy |
| Pluggable strategies | yes (`Strategy` protocol) | no | n/a | server-side, one strategy |
| Pluggable tokenizers | yes (tiktoken, anthropic, HF, char) | yes | n/a | server-side |
| Async summariser hook | yes | partial | n/a | n/a |
| Streaming observability | yes (`PackEvent`) | no | n/a | no |
| Long-term semantic memory | no (intentional) | partial via VectorStoreRetrieverMemory | yes (its core feature) | no |

## When to use convopack

- You're building an LLM app and don't want to take a framework dependency just to manage the context window.
- You use multiple providers and want a unified packing layer.
- You have a tool-using agent and need correctness, not just truncation.
- You want to plug in your own scoring or summarisation logic.
- You want a small library you can read end-to-end in an afternoon.

## When not to use convopack

- You want long-term semantic memory across sessions — use [mem0](https://github.com/mem0ai/mem0). You can stack convopack in front for per-call budgeting.
- You're already deeply invested in LangGraph and your agents use its checkpointers — `trim_messages` is good enough and fits your existing graph.
- You need server-side context editing on Anthropic — their `clear_tool_uses_*` API is excellent and free of round-trip cost. `convopack` is for client-side decisions.

## The tool-pair issue in detail

Here's the failure mode that motivates the whole library. Take a history with an outstanding tool call:

```
[0]  system:     "You can call get_weather()..."
[1]  user:       "What's the weather in Oslo?"
[2]  assistant:  <tool_use id=t1 name=get_weather input={city: "oslo"}>
[3]  tool:       <tool_result tool_use_id=t1 content="rainy">
[4]  assistant:  "It's rainy in Oslo."
[5]  user:       "And Bergen?"
[6]  assistant:  <tool_use id=t2 name=get_weather input={city: "bergen"}>
[7]  tool:       <tool_result tool_use_id=t2 content="cloudy">
[8]  assistant:  "Bergen is cloudy."
```

Suppose the budget allows only messages 4 through 8. A per-message truncator that doesn't know about tool pairs will:

- keep `[5]` (user)
- keep `[6]` (tool_use t2)
- keep `[7]` (tool_result t2)
- keep `[8]` (assistant)
- drop everything before `[4]`

That's fine here. But move the budget down one and the trimmer drops `[5]` while keeping `[6]`/`[7]`/`[8]`. Now the assistant message containing `tool_use t2` is the first message — providers reject this because there's no preceding user turn establishing context, *or* the trimmer drops `[6]` and keeps `[7]`, which gets rejected because the `tool_result` has no preceding `tool_use`.

`convopack` solves this by grouping `[6]` and `[7]` into a single atomic chunk (and any preceding setup) so eviction decisions happen at the chunk level. The invariant — `validate_pairs(packed.kept) == []` — is property-tested across thousands of randomised histories.

## Benchmarks

See [`bench/RESULTS.md`](https://github.com/Mrrobi/convopack/blob/main/bench/RESULTS.md) for the v0.2.0 numbers. Headline: submillisecond on a 6-turn conversation, ~2 ms on a 645-turn worst case with the `approx` tokenizer, 100 % tool-pair correctness across every test case.

You can run the benchmarks yourself:

```bash
git clone https://github.com/Mrrobi/convopack
cd convopack
pip install -e ".[dev]"
python -m bench.run
```
