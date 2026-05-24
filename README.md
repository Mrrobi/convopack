# convopack

[![PyPI](https://img.shields.io/pypi/v/convopack.svg)](https://pypi.org/project/convopack/)
[![Python](https://img.shields.io/pypi/pyversions/convopack.svg)](https://pypi.org/project/convopack/)
[![CI](https://github.com/Mrrobi/convopack/actions/workflows/ci.yml/badge.svg)](https://github.com/Mrrobi/convopack/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Mrrobi/convopack/branch/main/graph/badge.svg)](https://codecov.io/gh/Mrrobi/convopack)
[![License](https://img.shields.io/pypi/l/convopack.svg)](https://github.com/Mrrobi/convopack/blob/main/LICENSE)

Framework-agnostic, provider-agnostic context-window packer for LLM chat history.

## Why

LLM apps accumulate messages until they overflow the model's context window. Existing fixes are framework-locked (LangChain, LangGraph), provider-specific (Anthropic context-management beta), or designed for long-term semantic memory rather than turn-by-turn packing (mem0).

`convopack` is a small, focused library that takes a conversation history and a token budget and returns the largest tail that fits, while:

- preserving `tool_use` / `tool_result` pairs atomically,
- normalising message shapes across OpenAI, Anthropic, and Gemini,
- letting you plug in any tokenizer or summariser,
- staying async-friendly and zero-dependency at the core.

## Install

```bash
pip install convopack                 # core only
pip install "convopack[tiktoken]"     # + OpenAI tokenizer
pip install "convopack[anthropic]"    # + Anthropic tokenizer
pip install "convopack[all]"          # everything
```

## Quickstart

```python
from convopack import Packer, Recency

packer = Packer(
    budget=8000,
    tokenizer="tiktoken:gpt-4o",
    strategy=Recency(),
    pin=["system", "first_user"],
)

packed = packer.pack(messages)        # list[dict] in, list[dict] out
```

## Strategies

| Strategy        | When to use                                              |
| --------------- | -------------------------------------------------------- |
| `Recency`       | Keep the tail that fits. Cheapest, no LLM call.          |
| `SummaryEvict`  | Summarise evicted turns into a single system message.    |
| `Importance`    | Score each turn yourself; drop the lowest until it fits. |
| `SemanticDedup` | Remove near-duplicate turns first.                       |

## Comparison

| Feature                       | convopack | LangChain `trim_messages` | mem0          | Anthropic native |
| ----------------------------- | --------- | ------------------------- | ------------- | ---------------- |
| Framework-free                | yes       | no                        | no            | yes              |
| Multi-provider message shapes | yes       | partial                   | n/a           | no               |
| Tool-call pair safety         | yes       | no                        | n/a           | yes (one type)   |
| Pluggable strategy            | yes       | no                        | n/a           | no               |
| Async summariser              | yes       | no                        | n/a           | n/a              |
| Scope                         | per-turn  | per-turn                  | cross-session | per-turn         |

## License

MIT
