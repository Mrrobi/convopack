# Benchmark results

Numbers from `python -m bench.run` on Windows 11, Python 3.13.12, 645-turn worst case (~20K input tokens). Run yourself with:

```bash
pip install -e ".[dev]"
python -m bench.run
```

## v0.2.0 baseline (2026-05-24)

| strategy                | turns | budget | kept | tok    | p50 ms | p95 ms | pair_correct |
| ----------------------- | ----: | -----: | ---: | -----: | -----: | -----: | :----------: |
| convopack.Recency       |     6 |    500 |   13 |    482 |  0.015 |  0.022 | True         |
| convopack.FirstFit      |     6 |    500 |   13 |    482 |  0.014 |  0.016 | True         |
| convopack.Importance    |     6 |    500 |   13 |    482 |  0.024 |  0.031 | True         |
| convopack.Recency       |    63 |  2,000 |   54 |  1,982 |  0.126 |  0.799 | True         |
| convopack.FirstFit      |    63 |  2,000 |   49 |  1,997 |  0.124 |  0.127 | True         |
| convopack.Importance    |    63 |  2,000 |   66 |  1,996 |  0.219 |  0.236 | True         |
| convopack.Recency       |   265 |  8,000 |  202 |  7,987 |  0.527 |  2.330 | True         |
| convopack.FirstFit      |   265 |  8,000 |  210 |  7,995 |  0.526 |  0.640 | True         |
| convopack.Importance    |   265 |  8,000 |  262 |  7,989 |  0.919 |  2.724 | True         |
| convopack.Recency       |   645 | 20,000 |  508 | 19,998 |  1.330 |  2.949 | True         |
| convopack.FirstFit      |   645 | 20,000 |  507 | 19,991 |  1.329 |  2.972 | True         |
| convopack.Importance    |   645 | 20,000 |  647 | 19,991 |  2.272 |  4.076 | True         |

Takeaways:

* Sub-millisecond on a 6-turn conversation; ~2 ms on a 645-turn one with the
  `approx` tokenizer. `tiktoken` adds ~30% overhead per message.
* `Recency` and `FirstFit` are nearly identical speed. `Importance` is ~2x
  slower because it sorts chunks; still trivial in practice.
* `pair_correct=True` everywhere -- convopack never emits a kept list with a
  dangling `tool_use`. Run this against your own packer to confirm.

## LangChain comparison

Install `pip install langchain-core` and rerun -- a `langchain.trim_messages`
row will appear automatically. As of LangChain 0.3, the default trimmer does
not preserve `tool_use` / `tool_result` pairing, so `pair_correct` is reported
as `n/a`.
