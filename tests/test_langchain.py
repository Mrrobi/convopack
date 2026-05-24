"""LangChain BaseMessage adapter. Skips if langchain_core not installed."""

from __future__ import annotations

import pytest

from convopack import Packer, Recency
from convopack.providers.langchain import from_langchain, to_langchain

pytest.importorskip("langchain_core")

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


def test_system_human_ai_roundtrip() -> None:
    lc_msgs = [
        SystemMessage(content="Be concise."),
        HumanMessage(content="hi"),
        AIMessage(content="hello"),
    ]
    msgs = from_langchain(lc_msgs)
    assert [m.role for m in msgs] == ["system", "user", "assistant"]
    back = to_langchain(msgs)
    assert isinstance(back[0], SystemMessage)
    assert isinstance(back[1], HumanMessage)
    assert isinstance(back[2], AIMessage)
    assert back[2].content == "hello"


def test_tool_call_and_result_roundtrip() -> None:
    lc_msgs = [
        AIMessage(
            content="",
            tool_calls=[{"name": "weather", "args": {"city": "oslo"}, "id": "t1"}],
        ),
        ToolMessage(content="rainy", tool_call_id="t1"),
    ]
    msgs = from_langchain(lc_msgs)
    assert msgs[0].has_tool_use()
    assert msgs[1].has_tool_result()
    back = to_langchain(msgs)
    assert isinstance(back[0], AIMessage)
    assert back[0].tool_calls[0]["name"] == "weather"
    assert back[0].tool_calls[0]["args"] == {"city": "oslo"}
    assert isinstance(back[1], ToolMessage)
    assert back[1].tool_call_id == "t1"


def test_pack_langchain_smokes() -> None:
    lc_msgs = [
        SystemMessage(content="hi"),
        HumanMessage(content="first"),
        AIMessage(content="reply"),
        HumanMessage(content="second"),
    ]
    packer = Packer(budget=10_000, tokenizer="approx", strategy=Recency())
    out = packer.pack_langchain(lc_msgs)
    assert len(out) == 4


def test_list_content_flattened() -> None:
    msg = HumanMessage(
        content=[{"type": "text", "text": "hello"}, {"type": "text", "text": " world"}]
    )
    msgs = from_langchain([msg])
    assert msgs[0].text() == "hello world"
