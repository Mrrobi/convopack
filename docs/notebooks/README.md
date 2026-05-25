# Example notebooks

Self-contained Jupyter notebooks. Every notebook runs end-to-end with no API
keys (they use the zero-dependency `approx` tokenizer and deterministic
fakes for any LLM/embedder calls). All four are executed in CI to keep them
in sync with the library.

| Notebook | What it covers |
|---|---|
| [`01_quickstart.ipynb`](01_quickstart.ipynb) | Build a `Packer`, pack a history, round-trip through OpenAI shape, scan budgets. |
| [`02_tool_pair_atomicity.ipynb`](02_tool_pair_atomicity.ipynb) | The killer feature: `tool_use` / `tool_result` pairs stay together for every strategy. Includes a 200-iteration property check. |
| [`03_strategies.ipynb`](03_strategies.ipynb) | Same history through all five strategies (`Recency`, `FirstFit`, `SummaryEvict`, `Importance`, `SemanticDedup`). |
| [`04_prompt_caching.ipynb`](04_prompt_caching.ipynb) | Anthropic `cache_control` markers, OpenAI prefix-stability signature, cost back-of-envelope. |

## Run locally

```bash
pip install convopack jupyter
# or
uv add convopack
uv add --dev jupyter

jupyter notebook notebooks/01_quickstart.ipynb
```

## Run in your browser

Notebooks are also rendered with executed outputs on the documentation site:
<https://mrrobi.github.io/convopack/#/notebooks/01_quickstart>
