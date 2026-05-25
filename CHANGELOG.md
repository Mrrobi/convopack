# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] - 2026-05-24

### Added
- Four executable Jupyter notebooks under `docs/notebooks/`:
  - `01_quickstart.ipynb` — first pack walkthrough.
  - `02_tool_pair_atomicity.ipynb` — the killer feature with a 200-iteration
    property check.
  - `03_strategies.ipynb` — every built-in strategy side by side.
  - `04_prompt_caching.ipynb` — Anthropic `cache_control` markers, OpenAI
    prefix-stability signature, cost back-of-envelope.
- `mkdocs-jupyter` plugin and a new "Notebooks" section in the docs nav so
  notebooks render with outputs on https://mrrobi.github.io/convopack/.
- Typed-package and uv-supported badges in README.

### Changed
- README now documents `uv add convopack` alongside `pip install`, the
  typed-package guarantee, and links to the four example notebooks. This
  release exists primarily to refresh the README rendered on PyPI.

[0.3.1]: https://github.com/Mrrobi/convopack/releases/tag/v0.3.1

### Added
- Prompt caching layer. `Packer(cache=True)` populates
  `PackResult.cache_markers` with indices into `kept` that should receive a
  cache breakpoint. `to_anthropic()` emits Anthropic
  `{"cache_control": {"type": "ephemeral"}}` on those messages, respecting
  the 4-marker cap and never marking the volatile `last_user`.
- `Packer.cache_prefix_signature(history)` returns a sha256 of the stable
  prefix -- use it to detect drift that would invalidate OpenAI's automatic
  prompt caching.
- `Packer.cache_info(history)` reports markers, marked-token count, and an
  estimated hit ratio.
- Stable `Message.content_hash` (sha256 over canonical semantic form,
  excludes runtime tool-call IDs) and `history_hash()` helper.
- LiteLLM provider adapter (`from_litellm` / `to_litellm`,
  `Packer.pack_litellm`), the OpenAI-shape passthrough for the LiteLLM
  ecosystem.
- DSPy adapter (`from_dspy` / `to_dspy`, `Packer.pack_dspy`) that accepts a
  `dspy.History` or a plain dict list and optionally returns a
  `dspy.History`.
- Anthropic server-side managed-context wrapper. New
  `convopack.anthropic_managed.ContextManagementConfig` builder plus
  `Packer.pack_anthropic_managed(history, system, trigger_tokens,
  keep_tool_uses)` returning `(payload, context_management_dict)` ready to
  pass to `messages.create`.
- Docs: `guide/caching.md`, `recipes/litellm.md`, `recipes/dspy.md`,
  `recipes/anthropic-managed.md`.

### Changed
- `AnthropicPayload.system` is now `str | list[dict]`. A plain string is
  returned when no cache markers are applied (compatible with all SDK
  versions); the list-of-text-blocks form is used when markers exist.

[0.3.0]: https://github.com/Mrrobi/convopack/releases/tag/v0.3.0

### Added
- `FirstFit` strategy: counterpart to `Recency` that keeps the oldest chunks
  that fit and drops the tail.
- `SemanticDedup` strategy: drops near-duplicate messages via cosine similarity
  on an `Embedder` callback, then defers to a fallback strategy for budget
  enforcement.
- `Embedder` protocol in `convopack.embedders` with a `cosine()` helper.
- `Packer.pack_stream()` returning a stream of `PackEvent`s (`kept`,
  `dropped`, `summarized`, `done`) for observability.
- `HFTokenizerAdapter` and the `huggingface:<model-id>` tokenizer spec, plus
  a new `convopack[huggingface]` extra.
- Gemini provider adapter (`from_gemini` / `to_gemini`, `Packer.pack_gemini`),
  with synthetic tool-use-id reconciliation so the rest of the library can
  enforce its uniform tool-pair invariant.
- LangChain `BaseMessage` adapter (`from_langchain` / `to_langchain`,
  `Packer.pack_langchain`) -- drop-in replacement for
  `langchain_core.messages.trim_messages` with tool-pair safety.
- `convopack.pydantic` helpers (`validate_metadata`, `typed_scorer`,
  `metadata_pin`) for typed-metadata access; `convopack[pydantic]` extra.
- `convopack.integrations.mem0.Mem0Summarizer`: a `SummaryEvict` summariser
  that also stores evicted turns in mem0 for long-term semantic recall.
- `convopack` CLI with subcommands `pack`, `inspect`, `estimate`. Reads JSON
  from a file or stdin, writes JSON to stdout.
- Benchmark harness in `bench/`; `python -m bench.run` prints a strategy x
  history-size table with median/p95 timing and tool-pair correctness.
- mkdocs Material documentation site published to GitHub Pages
  (https://mrrobi.github.io/convopack/).
- Hypothesis property tests covering pair-grouping and pack invariants.
- Coverage gate in CI (`fail_under=85`), codecov upload, README badges.
- Release workflow scaffold with PyPI Trusted Publisher (OIDC).

### Changed
- `Packer` constructor and helper signatures unchanged; v0.2 is purely additive.

[0.2.0]: https://github.com/Mrrobi/convopack/releases/tag/v0.2.0

## [0.1.0] - 2026-05-24

### Added
- Initial public release.
- Core `Packer` class with `Strategy` and `Tokenizer` protocols.
- Strategies: `Recency`, `SummaryEvict` (sync + async summariser), `Importance`.
- Tokenizer adapters: `ApproxTokenizer` (zero-dep), `TiktokenAdapter`,
  `AnthropicAdapter` (offline approximation + optional online count).
- Provider message-shape adapters for OpenAI Chat Completions and Anthropic
  Messages, with `tool_use` / `tool_result` pair atomicity.
- Pinning rules: `system`, `first_user`, `last_user`, `tool_results`, integer index.
- 41 tests, ruff clean, mypy strict clean.
- Five runnable examples.

[0.1.0]: https://github.com/Mrrobi/convopack/releases/tag/v0.1.0
