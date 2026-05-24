# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial scaffold: pyproject, src layout, MIT license.
- Core `Packer` class with `Strategy` and `Tokenizer` protocols.
- `Recency` strategy.
- `tiktoken` and approximate char-based tokenizer adapters.
- Message normalisation for OpenAI Chat and Anthropic Messages shapes.
- `tool_use` / `tool_result` pair atomicity.
