"""CLI: pack / inspect / estimate subcommands."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from convopack.cli import main


@pytest.fixture
def history_path(tmp_path: Path) -> Path:
    p = tmp_path / "history.json"
    p.write_text(
        json.dumps(
            [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "what is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        )
    )
    return p


def _run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    rc = main(argv)
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


def test_pack_writes_packed_history(history_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc, out, _ = _run(
        ["pack", str(history_path), "--budget", "100", "--tokenizer", "approx"],
        capsys,
    )
    assert rc == 0
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert parsed[0]["role"] == "system"


def test_pack_pretty_indents(history_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _, out, _ = _run(["pack", str(history_path), "--budget", "100", "--pretty"], capsys)
    assert "\n  " in out  # pretty-printed


def test_pack_verbose_writes_summary_to_stderr(
    history_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, _, err = _run(["pack", str(history_path), "--budget", "1000", "--verbose"], capsys)
    assert "kept" in err


def test_inspect_lists_each_message(history_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc, out, _ = _run(["inspect", str(history_path)], capsys)
    assert rc == 0
    assert "total:" in out
    assert "system" in out
    assert "user" in out


def test_estimate_shows_budget_table(
    history_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc, out, _ = _run(["estimate", str(history_path)], capsys)
    assert rc == 0
    assert "total:" in out
    assert "budget" in out
    assert "fits" in out or "OVERFLOW" in out


def test_stdin_input(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    history = [{"role": "user", "content": "hi"}]
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(history)))
    rc, out, _ = _run(["pack", "-", "--budget", "100"], capsys)
    assert rc == 0
    parsed = json.loads(out)
    assert parsed[0]["role"] == "user"


def test_anthropic_provider_handles_system(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    p = tmp_path / "anth.json"
    p.write_text(
        json.dumps(
            {
                "system": "be brief",
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
    )
    rc, out, _ = _run(["pack", str(p), "--budget", "1000", "--provider", "anthropic"], capsys)
    assert rc == 0
    parsed = json.loads(out)
    assert parsed["system"] == "be brief"


def test_unknown_strategy_rejected(history_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        _run(
            ["pack", str(history_path), "--budget", "100", "--strategy", "nope"],
            capsys,
        )


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        _run(["--version"], capsys)
    assert exc.value.code == 0


def test_main_returns_int(history_path: Path) -> None:
    rc = main(["estimate", str(history_path)])
    assert rc == 0
