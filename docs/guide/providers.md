# Providers

`convopack` keeps its own provider-agnostic `Message` type internally and offers `from_X` / `to_X` adapters for each major provider's message shape. Use the convenience methods on `Packer` to round-trip in a single call.

## OpenAI Chat Completions

```python
from convopack.providers import from_openai, to_openai

# raw is list[dict] in OpenAI Chat shape
msgs = from_openai(raw)
out = to_openai(msgs)
```

Supported shapes:

- `{"role": "system" | "user" | "assistant", "content": str}`
- `{"role": "user", "content": [{"type": "text", ...}, {"type": "image_url", ...}]}`
- `{"role": "assistant", "tool_calls": [{"id", "type", "function": {"name", "arguments"}}]}`
- `{"role": "tool", "tool_call_id": "...", "content": str}`
- `name` field is round-tripped.

Convenience on `Packer`:

```python
packed = Packer(budget=4000, tokenizer="tiktoken:gpt-4o").pack_openai(history)
```

## Anthropic Messages

```python
from convopack.providers import from_anthropic, to_anthropic

msgs = from_anthropic(raw, system="Be concise.")
payload = to_anthropic(msgs)
# payload.system: str
# payload.messages: list[dict] (no system message inside)
```

Supported content blocks:

- `text`
- `image` with `base64` or `url` sources
- `tool_use` with id/name/input
- `tool_result` with `tool_use_id`, string or list content, `is_error` flag

Convenience on `Packer`:

```python
packed = Packer(budget=4000, tokenizer="anthropic:claude-sonnet-4-6").pack_anthropic(
    history, system="Be concise."
)
```

## Google Gemini

Gemini differs in two ways:

- The assistant role is named `"model"` instead of `"assistant"`.
- Tool calls (`function_call`) and tool results (`function_response`) match by name, not by ID.

`convopack` generates synthetic IDs on the way in so the rest of the library keeps its uniform tool-pair invariant, and drops them on the way out.

```python
from convopack.providers import from_gemini, to_gemini

msgs = from_gemini(contents, system_instruction="Be concise.")
payload = to_gemini(msgs)
# payload.system_instruction: str
# payload.contents: list[dict]
```

Convenience on `Packer`:

```python
packed = Packer(budget=4000, tokenizer="approx").pack_gemini(
    contents, system_instruction="Be concise."
)
```

## Cross-provider conversions

Because everything goes through the internal `Message` type, you can read in one shape and write out another:

```python
from convopack.providers import from_openai, to_anthropic

anthropic_payload = to_anthropic(from_openai(openai_history))
```

This is useful for migrating an agent loop from one provider to another, or for libraries that want to test parity across providers without rewriting their fixtures.

## Tool-pair atomicity, across providers

Every provider models tool exchanges slightly differently — OpenAI uses `tool_calls`/`tool` messages with shared `id`, Anthropic uses content blocks with `tool_use_id`, Gemini uses parts with no ID. `convopack` reconciles these into one internal model: a `ToolUseBlock` always pairs with a matching `ToolResultBlock`. Strategies see only that pairing, so the invariant — "never split a pair" — holds regardless of which provider you came from.
