"""Run the benchmark suite.

Usage: ``python -m bench.run``.

Measures, for each strategy and history size:

  * wall-clock pack time (median of N runs)
  * kept token count vs budget
  * tool-pair correctness on the kept output (must always be 100%)

If ``langchain_core`` is installed, also runs ``trim_messages`` from LangChain
as a comparison baseline. Tool-pair correctness is the key differentiator
because LangChain's default trimmer doesn't preserve ``tool_use`` /
``tool_result`` pairing.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

from bench.fixtures import make_history
from convopack import FirstFit, Importance, Message, Packer, Recency, Strategy
from convopack._pairs import validate_pairs

HISTORY_SIZES = [5, 50, 200, 500]
BUDGETS = {5: 500, 50: 2_000, 200: 8_000, 500: 20_000}
RUNS = 30


def _time_ms(fn: Callable[[], object]) -> float:
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000


def bench_strategy(
    name: str,
    strategy: Strategy,
    history: list[Message],
    budget: int,
) -> dict[str, Any]:
    packer = Packer(budget=budget, tokenizer="approx", strategy=strategy)
    samples = [_time_ms(lambda: packer.pack(history)) for _ in range(RUNS)]
    result = packer.pack(history)
    dangling = validate_pairs(result.kept)
    return {
        "strategy": name,
        "turns": (len(history) - 1) // 2,
        "budget": budget,
        "kept_msgs": len(result.kept),
        "kept_tokens": result.token_count,
        "ms_median": round(statistics.median(samples), 3),
        "ms_p95": round(sorted(samples)[int(len(samples) * 0.95)], 3),
        "pair_correct": len(dangling) == 0,
    }


def bench_langchain(history: list[Message], budget: int) -> dict[str, Any] | None:
    """Compare with LangChain's trim_messages, if installed."""
    try:
        from langchain_core.messages import (  # type: ignore[import-not-found]
            AIMessage,
            HumanMessage,
            SystemMessage,
            ToolMessage,
            trim_messages,
        )
    except ImportError:
        return None

    lc_msgs: list[Any] = []
    for m in history:
        text = m.text() or ""
        if m.role == "system":
            lc_msgs.append(SystemMessage(content=text))
        elif m.role == "user":
            lc_msgs.append(HumanMessage(content=text))
        elif m.role == "assistant":
            lc_msgs.append(AIMessage(content=text))
        elif m.role == "tool":
            tool_ids = m.tool_result_ids()
            lc_msgs.append(ToolMessage(content=text, tool_call_id=tool_ids[0] if tool_ids else ""))

    def token_counter(messages: list[Any]) -> int:
        return sum(len(str(getattr(m, "content", "")) or "") // 4 for m in messages)

    samples = [
        _time_ms(
            lambda: trim_messages(
                lc_msgs, max_tokens=budget, token_counter=token_counter, strategy="last"
            )
        )
        for _ in range(RUNS)
    ]
    return {
        "strategy": "langchain.trim_messages(last)",
        "turns": (len(history) - 1) // 2,
        "budget": budget,
        "ms_median": round(statistics.median(samples), 3),
        "ms_p95": round(sorted(samples)[int(len(samples) * 0.95)], 3),
        "pair_correct": "n/a (LangChain doesn't model tool pairs uniformly)",
    }


def main() -> None:
    print(
        f"{'strategy':<32} {'turns':>6} {'budget':>8} {'kept':>5} {'tok':>6} {'p50':>8} {'p95':>8} pair_ok"
    )
    print("-" * 100)
    for n in HISTORY_SIZES:
        history = make_history(n, tool_density=0.3, seed=n)
        budget = BUDGETS[n]
        rows = [
            bench_strategy("convopack.Recency", Recency(), history, budget),
            bench_strategy("convopack.FirstFit", FirstFit(), history, budget),
            bench_strategy("convopack.Importance", Importance(), history, budget),
        ]
        lc = bench_langchain(history, budget)
        if lc is not None:
            rows.append(lc)
        for r in rows:
            kept = r.get("kept_msgs", "-")
            tok = r.get("kept_tokens", "-")
            print(
                f"{r['strategy']:<32} {r['turns']:>6} {r['budget']:>8} "
                f"{kept!s:>5} {tok!s:>6} {r['ms_median']:>8} {r['ms_p95']:>8} {r['pair_correct']}"
            )
        print()


if __name__ == "__main__":
    main()
