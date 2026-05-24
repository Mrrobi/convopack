---
hide:
  - navigation
---

# convopack

[![PyPI](https://img.shields.io/pypi/v/convopack.svg)](https://pypi.org/project/convopack/)
[![Python](https://img.shields.io/pypi/pyversions/convopack.svg)](https://pypi.org/project/convopack/)
[![CI](https://github.com/Mrrobi/convopack/actions/workflows/ci.yml/badge.svg)](https://github.com/Mrrobi/convopack/actions/workflows/ci.yml)
[![License](https://img.shields.io/pypi/l/convopack.svg)](https://github.com/Mrrobi/convopack/blob/main/LICENSE)

> **Framework-agnostic, provider-agnostic context-window packer for LLM chat history.**

LLM applications accumulate conversation messages until they overflow the model's context window. Existing solutions are framework-locked, provider-specific, or designed for a different problem entirely. `convopack` is a small library that does one thing well: take a conversation and a token budget, return the largest subset that fits — without breaking your tool calls.

## Install

```bash
pip install convopack
```

Optional tokenizers:

```bash
pip install "convopack[tiktoken]"      # OpenAI BPE
pip install "convopack[anthropic]"     # Anthropic offline + online counts
pip install "convopack[huggingface]"   # transformers AutoTokenizer
pip install "convopack[all]"           # everything
```

## At a glance

```python
from convopack import Packer, Recency

packer = Packer(
    budget=8000,
    tokenizer="tiktoken:gpt-4o",
    strategy=Recency(),
    pin=("system", "first_user"),
)

# Internal Message[] in, packed Message[] out
packed = packer.pack(history).kept

# Or work directly with OpenAI Chat Completion dicts
packed_dicts = packer.pack_openai(openai_history)
```

## Why a new library

- **Tool-pair atomicity.** A `tool_use` and its matching `tool_result` are evicted together or kept together; you never get a 400 from the provider because half of a pair survived.
- **Pluggable strategies.** Recency, FirstFit, SummaryEvict, Importance, SemanticDedup — and your own.
- **Provider-agnostic.** OpenAI Chat Completions, Anthropic Messages, and Google Gemini shapes are all first-class; the internal `Message` type is the bridge.
- **Tokenizer-agnostic.** tiktoken, Anthropic, HuggingFace, or a char-based approximation with zero deps.
- **Async-friendly.** `pack_async` + an async summariser is a one-liner.
- **No framework lock-in.** No LangChain, no LlamaIndex, no required runtime. ~700 lines of pure Python.

## Next steps

- [Quickstart](quickstart.md) — five-minute tour.
- [Strategies guide](guide/strategies.md) — when to use which.
- [Recipes](recipes/openai-loop.md) — drop-in code for common loops.
- [Why convopack](comparison.md) — what's different from LangChain `trim_messages`, mem0, Anthropic's native context-management, and others.
