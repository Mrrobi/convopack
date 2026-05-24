# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-05-24

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
