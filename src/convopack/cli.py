"""Command-line entry point.

Three subcommands:

* ``pack`` -- read a conversation, write back the packed conversation that
  fits the budget.
* ``inspect`` -- print per-message token counts.
* ``estimate`` -- print the total token count and which sample budgets it fits.

Reads JSON from a path or stdin; writes JSON to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from convopack import (
    FirstFit,
    Importance,
    Packer,
    Recency,
    Strategy,
    __version__,
)
from convopack.providers import (
    from_anthropic,
    from_gemini,
    from_openai,
    to_anthropic,
    to_gemini,
    to_openai,
)
from convopack.tokenizers import get_tokenizer

PROVIDERS = ("openai", "anthropic", "gemini")
STRATEGIES = ("recency", "firstfit", "importance")


def _build_strategy(name: str) -> Strategy:
    if name == "recency":
        return Recency()
    if name == "firstfit":
        return FirstFit()
    if name == "importance":
        return Importance()
    raise ValueError(f"Unknown strategy: {name}")


def _load_input(path: str) -> Any:
    if path == "-":
        return json.load(sys.stdin)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _normalize(provider: str, data: Any) -> tuple[list[Any], dict[str, Any]]:
    """Return (internal messages, extras dict for provider context like system)."""
    if provider == "openai":
        return from_openai(data), {}
    if provider == "anthropic":
        if isinstance(data, dict):
            return from_anthropic(data.get("messages", []), system=data.get("system")), {
                "system": data.get("system")
            }
        return from_anthropic(data), {}
    if provider == "gemini":
        if isinstance(data, dict):
            return from_gemini(
                data.get("contents", []),
                system_instruction=data.get("system_instruction"),
            ), {"system_instruction": data.get("system_instruction")}
        return from_gemini(data), {}
    raise ValueError(f"Unknown provider: {provider}")


def _denormalize(provider: str, messages: list[Any], extras: dict[str, Any]) -> Any:
    if provider == "openai":
        return to_openai(messages)
    if provider == "anthropic":
        ant_payload = to_anthropic(messages)
        return {"system": ant_payload.system, "messages": ant_payload.messages}
    if provider == "gemini":
        gem_payload = to_gemini(messages)
        return {
            "system_instruction": gem_payload.system_instruction,
            "contents": gem_payload.contents,
        }
    raise ValueError(f"Unknown provider: {provider}")


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", help='path to JSON history, or "-" for stdin')
    p.add_argument(
        "--provider",
        choices=PROVIDERS,
        default="openai",
        help="message shape of the input/output (default: openai)",
    )
    p.add_argument(
        "--tokenizer",
        default="approx",
        help='tokenizer spec, e.g. "tiktoken:gpt-4o" or "approx" (default: approx)',
    )


def cmd_pack(args: argparse.Namespace) -> int:
    data = _load_input(args.input)
    msgs, _ = _normalize(args.provider, data)
    strategy = _build_strategy(args.strategy)
    packer = Packer(
        budget=args.budget,
        tokenizer=args.tokenizer,
        strategy=strategy,
        pin=tuple(args.pin.split(",")) if args.pin else ("system",),
    )
    result = packer.pack(msgs)
    out = _denormalize(args.provider, result.kept, {})
    json.dump(out, sys.stdout, indent=2 if args.pretty else None)
    sys.stdout.write("\n")
    if args.verbose:
        print(
            f"# kept {len(result.kept)} of {len(msgs)} messages, "
            f"{result.token_count}/{result.budget} tokens",
            file=sys.stderr,
        )
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    data = _load_input(args.input)
    msgs, _ = _normalize(args.provider, data)
    tok = get_tokenizer(args.tokenizer)
    total = 0
    print(f"{'idx':>4}  {'role':<10}  {'tokens':>7}  preview")
    print("-" * 70)
    for i, m in enumerate(msgs):
        n = tok.count_message(m)
        total += n
        preview = m.text().replace("\n", " ")[:40]
        print(f"{i:>4}  {m.role:<10}  {n:>7}  {preview}")
    print("-" * 70)
    print(f"total: {total} tokens, {len(msgs)} messages")
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    data = _load_input(args.input)
    msgs, _ = _normalize(args.provider, data)
    tok = get_tokenizer(args.tokenizer)
    total = tok.count_messages(msgs)
    print(f"total: {total} tokens across {len(msgs)} messages")
    for budget in (4_000, 8_000, 16_000, 32_000, 128_000, 1_000_000):
        verdict = "fits" if total <= budget else "OVERFLOW"
        delta = budget - total
        print(f"  budget {budget:>8}: {verdict:<8}  headroom {delta:>+8}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="convopack",
        description="Pack LLM chat histories under a token budget.",
    )
    p.add_argument("--version", action="version", version=f"convopack {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("pack", help="pack a history under a token budget")
    _add_common_args(pp)
    pp.add_argument("--budget", type=int, required=True, help="token budget for the packed output")
    pp.add_argument(
        "--strategy",
        choices=STRATEGIES,
        default="recency",
        help="packing strategy (default: recency)",
    )
    pp.add_argument(
        "--pin",
        default="system",
        help='comma-separated pin specs (default: "system"). Pass empty string for no pins.',
    )
    pp.add_argument("--pretty", action="store_true", help="indent the output JSON")
    pp.add_argument("--verbose", action="store_true", help="print summary to stderr")
    pp.set_defaults(func=cmd_pack)

    pi = sub.add_parser("inspect", help="show per-message token counts")
    _add_common_args(pi)
    pi.set_defaults(func=cmd_inspect)

    pe = sub.add_parser("estimate", help="estimate total tokens and budget fit")
    _add_common_args(pe)
    pe.set_defaults(func=cmd_estimate)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
