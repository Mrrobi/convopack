# API reference

Generated automatically from docstrings via [mkdocstrings](https://mkdocstrings.github.io/).

## Top-level

::: convopack
    options:
      members:
        - Packer
        - PackResult
        - PackEvent
        - PackEventKind
        - Message
        - Role
        - ContentBlock
        - TextBlock
        - ImageBlock
        - ToolUseBlock
        - ToolResultBlock

## Strategies

::: convopack.strategies
    options:
      members:
        - Strategy
        - Recency
        - FirstFit
        - SummaryEvict
        - Importance
        - SemanticDedup

## Tokenizers

::: convopack.tokenizers
    options:
      members:
        - Tokenizer
        - ApproxTokenizer
        - get_tokenizer

::: convopack.tokenizers.tiktoken_adapter

::: convopack.tokenizers.anthropic_adapter

::: convopack.tokenizers.huggingface_adapter

## Embedders

::: convopack.embedders
    options:
      members:
        - Embedder
        - cosine

## Providers

::: convopack.providers
    options:
      members:
        - from_openai
        - to_openai
        - from_anthropic
        - to_anthropic
        - GeminiPayload
        - from_gemini
        - to_gemini
