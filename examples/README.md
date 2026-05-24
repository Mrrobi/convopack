# Examples

Runnable scripts demonstrating each strategy and provider integration.

| File                        | Strategy        | Provider  | Tokenizer            |
| --------------------------- | --------------- | --------- | -------------------- |
| `openai_loop.py`            | `Recency`       | OpenAI    | `tiktoken:gpt-4o`    |
| `anthropic_loop.py`         | `Recency`       | Anthropic | `anthropic:...` offline |
| `summary_compaction.py`     | `SummaryEvict`  | OpenAI    | `approx`             |
| `importance_scoring.py`     | `Importance`    | -         | `approx`             |
| `async_summary.py`          | `SummaryEvict`  | -         | `approx`             |

Each example is self-contained and runs without an LLM call (LLM call is stubbed).
