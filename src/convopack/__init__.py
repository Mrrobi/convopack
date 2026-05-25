"""convopack: framework-agnostic context-window packer for LLM chat history."""

from convopack._types import (
    ContentBlock,
    ImageBlock,
    Message,
    PackResult,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    history_hash,
)
from convopack.embedders import Embedder
from convopack.events import PackEvent, PackEventKind
from convopack.packer import Packer
from convopack.strategies import (
    FirstFit,
    Importance,
    Recency,
    SemanticDedup,
    Strategy,
    SummaryEvict,
)
from convopack.tokenizers import ApproxTokenizer, Tokenizer, get_tokenizer

__all__ = [
    "ApproxTokenizer",
    "ContentBlock",
    "Embedder",
    "FirstFit",
    "ImageBlock",
    "Importance",
    "Message",
    "PackEvent",
    "PackEventKind",
    "PackResult",
    "Packer",
    "Recency",
    "Role",
    "SemanticDedup",
    "Strategy",
    "SummaryEvict",
    "TextBlock",
    "Tokenizer",
    "ToolResultBlock",
    "ToolUseBlock",
    "get_tokenizer",
    "history_hash",
]

__version__ = "0.3.1"
