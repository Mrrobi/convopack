# DSPy integration

DSPy programs accumulate `dspy.History` over many turns. Plug `convopack` in front of the LM call to keep that history under the model's context window.

```python
import dspy
from convopack import Packer, SummaryEvict

dspy.configure(lm=dspy.LM("openai/gpt-4o"))

packer = Packer(
    budget=8000,
    tokenizer="tiktoken:gpt-4o",
    strategy=SummaryEvict(summarise),
    pin=("system", "last_user"),
    cache=True,
)

class Chat(dspy.Signature):
    """Be a helpful assistant."""
    history: dspy.History = dspy.InputField()
    user_input: str = dspy.InputField()
    response: str = dspy.OutputField()

predict = dspy.Predict(Chat)

history = dspy.History(messages=[])

def turn(user_input: str) -> str:
    # convopack reads the dspy.History and writes a fresh, budget-trimmed one.
    trimmed = packer.pack_dspy(history, as_history=True)
    out = predict(history=trimmed, user_input=user_input)
    # Append the new exchange to the long history we keep on the side.
    history.messages.append({"role": "user", "content": user_input})
    history.messages.append({"role": "assistant", "content": out.response})
    return out.response
```

## What `pack_dspy` does

- Reads `dspy.History.messages` (a list of OpenAI-shape dicts).
- Runs the configured strategy and tokenizer.
- Returns either a plain `list[dict]` (default) or a fresh `dspy.History` when `as_history=True`.

## Compose with everything else

`pack_dspy` is just a thin layer over the OpenAI provider adapter; cache markers, tool-pair atomicity, async summarisation, and pinning all work the same way as in any other recipe.
