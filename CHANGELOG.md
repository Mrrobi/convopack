# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
