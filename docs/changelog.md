# Changelog

The authoritative changelog lives at [`CHANGELOG.md`](https://github.com/Mrrobi/convopack/blob/main/CHANGELOG.md) in the repository, in [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

## v0.1.0 — 2026-05-24

Initial public release.

- Core `Packer` class with `Strategy` and `Tokenizer` protocols.
- Strategies: `Recency`, `SummaryEvict`, `Importance`.
- Tokenizers: `ApproxTokenizer`, `TiktokenAdapter`, `AnthropicAdapter`.
- Providers: OpenAI Chat Completions, Anthropic Messages.
- Pinning rules: `system`, `first_user`, `last_user`, `tool_results`, integer index.
- `tool_use` / `tool_result` pair atomicity enforced everywhere.

## Unreleased (v0.2.0 in progress)

- `FirstFit` strategy: counterpart to `Recency`.
- `SemanticDedup` strategy: drop near-duplicates via cosine similarity.
- `Embedder` protocol in `convopack.embedders`.
- `Packer.pack_stream()` returning `PackEvent`s.
- `HFTokenizerAdapter` and `huggingface:<id>` spec.
- Gemini provider adapter with synthetic `tool_use_id` reconciliation.
- Benchmark harness in `bench/`.
- mkdocs Material documentation site.
- Hypothesis property tests for pair-grouping invariants.
- Coverage gate in CI (fail_under=85), README badges.
- Release workflow with PyPI Trusted Publisher (OIDC).
