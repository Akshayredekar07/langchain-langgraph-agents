# LangChain — `ainvoke` (Async Invoke)

---

## What is `ainvoke`?

`ainvoke` is the **async version** of `invoke`.  
It does the exact same thing — sends a message, waits for the full response, returns an `AIMessage` — but it does it **without blocking** the event loop.

Use `ainvoke` when:

- You are inside an `async def` function
- You want to run multiple model calls concurrently with `asyncio.gather`
- You are building a FastAPI / async web server
- You are in a Jupyter notebook (which has its own event loop)

The return value is identical to `invoke` — an `AIMessage` object with the same `.content`, `.usage_metadata`, `.response_metadata` etc.

---

## Basic Usage

```python
import asyncio
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai")

# ainvoke must be called with await inside an async function
async def main():
    response = await model.ainvoke("What is Python?")
    print(response.content)

asyncio.run(main())
```

In a Jupyter notebook, use `await` directly (no `asyncio.run` needed):

```python
response = await model.ainvoke("What is Python?")
print(response.content)
```

---

## All Input Formats (same as `invoke`)

```python
# 1. Plain string
response = await model.ainvoke("What is Python?")

# 2. List of dicts
response = await model.ainvoke([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "What is Python?"}
])

# 3. Message objects
from langchain_core.messages import SystemMessage, HumanMessage

response = await model.ainvoke([
    SystemMessage("You are a helpful assistant."),
    HumanMessage("What is Python?")
])

print(response.content)
```

---

## Run Multiple Calls in Parallel

This is the main reason to use `ainvoke` — running many calls at the same time.

```python
import asyncio
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai")

async def main():
    prompts = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
    ]

    # All three calls happen at the same time — not one after another
    responses = await asyncio.gather(*[
        model.ainvoke(p) for p in prompts
    ])

    for r in responses:
        print(r.content[:80])
        print("---")

asyncio.run(main())
```

Sequential `invoke` for 3 calls might take 9 seconds.  
Parallel `ainvoke` with `asyncio.gather` typically takes ~3 seconds.

---

## With Different Providers

`ainvoke` works with every provider exactly like `invoke`:

```python
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

claude = ChatAnthropic(model="claude-sonnet-4-6")
groq   = ChatGroq(model="llama-3.3-70b-versatile")

async def compare():
    r1 = await claude.ainvoke("Explain recursion in one sentence.")
    r2 = await groq.ainvoke("Explain recursion in one sentence.")
    print("Claude :", r1.content)
    print("Groq   :", r2.content)

await compare()   # in notebook
```

---

## Accessing the Response

The response object is the same `AIMessage` as `invoke`:

```python
response = await model.ainvoke("What is 2 + 2?")

print(response.content)            # text answer
print(response.usage_metadata)     # token counts
print(response.response_metadata)  # model name, finish_reason, etc.
```

---

## `invoke` vs `ainvoke` — Quick Comparison


|                  | `invoke`                | `ainvoke`                         |
| ---------------- | ----------------------- | --------------------------------- |
| Syntax           | `model.invoke(...)`     | `await model.ainvoke(...)`        |
| Blocking         | Yes — blocks until done | No — yields control to event loop |
| Return type      | `AIMessage`             | `AIMessage` (identical)           |
| Use in async def | Works but not ideal     | Preferred                         |
| Parallel calls   | Use `batch()`           | Use `asyncio.gather`              |
| Notebook         | Works directly          | Use `await` directly              |


---

## Note on Jupyter Notebooks

Jupyter already runs an event loop, so you do **not** need `asyncio.run()`.  
Just use `await` at the top level of a cell:

```python
# This works fine in a notebook cell
response = await model.ainvoke("Hello")
print(response.content)
```

