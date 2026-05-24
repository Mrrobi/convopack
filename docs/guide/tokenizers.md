# Tokenizers

Token counting is provider-specific. `convopack` accepts any `Tokenizer` implementation and ships adapters for the common ones.

## Spec strings

The fastest way to choose a tokenizer is by passing a string to `Packer`:

```python
Packer(tokenizer="approx", ...)
Packer(tokenizer="tiktoken:gpt-4o", ...)
Packer(tokenizer="anthropic:claude-sonnet-4-6", ...)
Packer(tokenizer="huggingface:meta-llama/Llama-3.1-8B", ...)
```

| Spec | Backend | Deps |
|---|---|---|
| `approx` | `char_count / 4` estimate | none (zero-dep default) |
| `tiktoken:<model-or-encoding>` | OpenAI BPE | `pip install "convopack[tiktoken]"` |
| `anthropic:<model>` | Approx + optional online count | `pip install "convopack[anthropic]"` |
| `huggingface:<model-id>` | HuggingFace `AutoTokenizer` | `pip install "convopack[huggingface]"` |

## `ApproxTokenizer`

Char count divided by four, with a small per-message overhead. Off by 10–30 % depending on language; fine for a soft budget but never use it for billing.

```python
from convopack import ApproxTokenizer

tok = ApproxTokenizer()
tok.count("hello world")          # 3
tok.count_message(message)        # includes per-message overhead
tok.count_messages([m1, m2, m3])  # sums + per-reply overhead
```

## `TiktokenAdapter`

Wraps the official `tiktoken` encoder. Pass either a model name or an encoding name:

```python
from convopack.tokenizers.tiktoken_adapter import TiktokenAdapter

TiktokenAdapter("gpt-4o")
TiktokenAdapter("o200k_base")
```

Unknown model names fall back to `tiktoken.get_encoding`.

## `AnthropicAdapter`

Anthropic doesn't ship a local BPE; the canonical method is `client.messages.count_tokens`, which is a network call. Two modes:

```python
from convopack.tokenizers.anthropic_adapter import AnthropicAdapter

# default: offline approximation, fast
AnthropicAdapter("claude-sonnet-4-6")

# online: per-call network round-trip, exact
AnthropicAdapter("claude-sonnet-4-6", offline=False)
```

Online mode requires the `anthropic` SDK and a configured API key. Use it for billing-grade counts; otherwise the offline mode is fine.

## `HFTokenizerAdapter`

Wraps `transformers.AutoTokenizer` so any open-weights model can be tokenised correctly:

```python
from convopack.tokenizers.huggingface_adapter import HFTokenizerAdapter

HFTokenizerAdapter("meta-llama/Llama-3.1-8B")
HFTokenizerAdapter("/local/path/to/tokenizer")
HFTokenizerAdapter("meta-llama/Llama-3.1-8B", local_files_only=True)
```

## Writing your own

Implement the protocol:

```python
from typing import Iterable
from convopack import Message

class MyTokenizer:
    name = "my:counter"

    def count(self, text: str) -> int: ...
    def count_message(self, message: Message) -> int: ...
    def count_messages(self, messages: Iterable[Message]) -> int: ...
```

Pass an instance to `Packer(tokenizer=...)` and it works everywhere a spec string would.

## A note on accuracy

`count_message` adds a small per-message overhead to cover role tags and separators. The exact number varies by provider and is best-effort — use online counting modes when overshoot matters more than throughput.
