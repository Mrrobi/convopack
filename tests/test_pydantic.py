"""Optional Pydantic helpers for typed metadata access. Skipped without pydantic."""

from __future__ import annotations

import pytest

from convopack import Importance, Message, Packer, TextBlock

pydantic = pytest.importorskip("pydantic")

from pydantic import BaseModel, ValidationError  # noqa: E402

from convopack.pydantic import (  # noqa: E402
    metadata_pin,
    typed_scorer,
    validate_metadata,
)


class TurnMeta(BaseModel):
    starred: bool = False
    source: str = "user"


def test_validate_metadata_returns_models() -> None:
    msgs = [
        Message(role="user", content=[], metadata={"starred": True, "source": "admin"}),
        Message(role="user", content=[], metadata={}),
    ]
    parsed = validate_metadata(msgs, TurnMeta)
    assert parsed[0].starred is True
    assert parsed[0].source == "admin"
    assert parsed[1].starred is False
    assert parsed[1].source == "user"


def test_validate_metadata_raises_on_bad_field() -> None:
    msgs = [Message(role="user", content=[], metadata={"starred": "not-a-bool"})]
    with pytest.raises(ValidationError):
        validate_metadata(msgs, TurnMeta)


def test_typed_scorer_receives_parsed_model() -> None:
    def score(meta: TurnMeta, _msg: Message) -> float:
        return 100.0 if meta.starred else 1.0

    msgs = [
        Message(role="user", content=[TextBlock(text="a" * 80)], metadata={"starred": True}),
        Message(role="user", content=[TextBlock(text="b" * 80)], metadata={"starred": False}),
        Message(role="user", content=[TextBlock(text="c" * 80)], metadata={"starred": False}),
    ]
    packer = Packer(
        budget=40,
        tokenizer="approx",
        strategy=Importance(scorer=typed_scorer(TurnMeta, score)),
        pin=(),
    )
    result = packer.pack(msgs)
    starred_kept = [m for m in result.kept if m.metadata.get("starred")]
    assert starred_kept


def test_metadata_pin_returns_matching_indices() -> None:
    msgs = [
        Message(role="user", content=[], metadata={"starred": True}),
        Message(role="user", content=[], metadata={"starred": False}),
        Message(role="user", content=[], metadata={"starred": True}),
    ]
    pinned = metadata_pin(msgs, TurnMeta, lambda m: m.starred)
    assert pinned == (0, 2)


def test_metadata_pin_skips_invalid_silently() -> None:
    msgs = [
        Message(role="user", content=[], metadata={"starred": True}),
        Message(role="user", content=[], metadata={"starred": "broken"}),
    ]
    pinned = metadata_pin(msgs, TurnMeta, lambda m: m.starred)
    assert pinned == (0,)
