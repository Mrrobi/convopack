"""LiteLLM, DSPy, and Anthropic managed-context wrappers."""

from __future__ import annotations

from typing import Any

import pytest

from convopack import Message, Packer, TextBlock
from convopack.anthropic_managed import ContextManagementConfig
from convopack.providers.dspy import from_dspy, to_dspy
from convopack.providers.litellm import from_litellm, to_litellm

# --- LiteLLM ----------------------------------------------------------------


def test_litellm_roundtrip_matches_openai_shape() -> None:
    raw = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    msgs = from_litellm(raw)
    out = to_litellm(msgs)
    assert out[0]["role"] == "system"
    assert out[1]["content"] == "hi"


def test_packer_pack_litellm() -> None:
    raw = [{"role": "user", "content": "hello"}]
    packer = Packer(budget=10_000, tokenizer="approx")
    packed = packer.pack_litellm(raw)
    assert isinstance(packed, list)
    assert packed[0]["role"] == "user"


# --- DSPy ------------------------------------------------------------------


class FakeHistory:
    """Stand-in for ``dspy.History`` -- only needs ``.messages``."""

    def __init__(self, messages: list[dict[str, Any]]):
        self.messages = messages


def test_dspy_from_history_object() -> None:
    h = FakeHistory(
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    )
    msgs = from_dspy(h)
    assert len(msgs) == 2
    assert msgs[0].role == "user"


def test_dspy_from_plain_list() -> None:
    msgs = from_dspy([{"role": "user", "content": "hi"}])
    assert msgs[0].role == "user"


def test_dspy_rejects_unknown_input() -> None:
    with pytest.raises(TypeError, match=r"expects a dspy\.History"):
        from_dspy(object())


def test_dspy_to_returns_list_by_default() -> None:
    msgs = [Message(role="user", content=[TextBlock(text="hi")])]
    out = to_dspy(msgs)
    assert isinstance(out, list)
    assert out[0]["role"] == "user"


def test_dspy_as_history_raises_without_dspy_installed() -> None:
    msgs = [Message(role="user", content=[TextBlock(text="hi")])]
    try:
        import dspy  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="dspy is required"):
            to_dspy(msgs, as_history=True)


def test_packer_pack_dspy() -> None:
    h = FakeHistory(messages=[{"role": "user", "content": "hi"}])
    packer = Packer(budget=10_000, tokenizer="approx")
    out = packer.pack_dspy(h)
    assert isinstance(out, list)


# --- Anthropic managed-context --------------------------------------------


def test_clear_tool_uses_default_config() -> None:
    cfg = ContextManagementConfig.clear_tool_uses()
    payload = cfg.to_dict()
    assert payload["edits"][0]["type"] == "clear_tool_uses_20250919"
    assert payload["edits"][0]["trigger"]["value"] == 30_000
    assert payload["edits"][0]["keep"]["value"] == 3


def test_with_edit_appends() -> None:
    cfg = ContextManagementConfig.empty().with_edit({"type": "custom"})
    assert cfg.to_dict()["edits"] == [{"type": "custom"}]


def test_pack_anthropic_managed_returns_payload_and_config() -> None:
    raw = [{"role": "user", "content": "hi"}]
    packer = Packer(budget=10_000, tokenizer="approx")
    payload, cm = packer.pack_anthropic_managed(
        raw, system="be brief", trigger_tokens=20_000, keep_tool_uses=2
    )
    assert payload.system == "be brief"
    assert cm["edits"][0]["trigger"]["value"] == 20_000
    assert cm["edits"][0]["keep"]["value"] == 2
