# **LangChain `stream` and `astream`**


## What is `stream`?

`stream()` calls the model and **returns tokens one by one** as they are generated, instead of waiting for the full response.

Each token arrives as an `AIMessageChunk` object — not a full `AIMessage`.  
You loop over chunks and process each one as it comes in.

**When to use stream:**
- Chat UIs where you want text to appear progressively (like ChatGPT)
- Long responses where the user should not wait for everything
- Showing reasoning steps or tool call progress in real time
- Any situation where perceived speed matters


## `invoke` vs `stream` — Core Difference

```
invoke  ->  waits for everything -> returns one AIMessage
stream  ->  yields piece by piece -> returns many AIMessageChunks
```

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai")

# invoke: one call, one result
response = model.invoke("Write a poem about the ocean.")
print(response.content)   # prints all at once after waiting

# stream: many chunks, printed as they arrive
for chunk in model.stream("Write a poem about the ocean."):
    print(chunk.content, end="", flush=True)
```

## What is `AIMessageChunk`?

Each item yielded by `stream()` is an `AIMessageChunk`.  
It is similar to `AIMessage` but designed to be **partial and incremental**.

```python
from langchain_core.messages import AIMessageChunk

# AIMessageChunk fields
# .content           -> partial text (a few words or even one character)
# .type              -> "AIMessageChunk"
# .id                -> message ID (set from first chunk only)
# .tool_call_chunks  -> partial tool call data (when model is calling a tool)
# .tool_calls        -> partially assembled tool calls
# .usage_metadata    -> token counts (usually only on the LAST chunk)
# .response_metadata -> finish reason etc. (usually only on the LAST chunk)
# .content_blocks    -> typed structured content (text, reasoning, tool_call_chunk)
```

Key rules:
- `content` is empty string `""` on most chunks — only some chunks carry text
- `usage_metadata` is `None` on most chunks — only the **last chunk** has token counts
- `response_metadata` `finish_reason` appears only on the **last chunk**
- Chunks can be **added together** with `+` to build the full `AIMessage`


## Basic Stream — Print Tokens as They Arrive

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai")

# Basic streaming — simplest form
for chunk in model.stream("Explain gradient descent step by step."):
    print(chunk.content, end="", flush=True)

print()   # newline after streaming finishes
```

The `end=""` prevents newlines between chunks.  
The `flush=True` forces the output to display immediately without buffering.


## Accumulating Chunks into a Full AIMessage

You can add chunks together using `+` to build the complete message.  
The result behaves exactly like a regular `AIMessage`.

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai")

full = None   # will be AIMessageChunk or None

for chunk in model.stream("What is machine learning?"):
    # Add chunks together to build the full message
    full = chunk if full is None else full + chunk
    print(chunk.content, end="", flush=True)

print()

# After the loop, full is a complete assembled AIMessageChunk
# It behaves exactly like an AIMessage
print("\n--- ASSEMBLED MESSAGE ---")
print("type           :", type(full))
print("content        :", full.content[:80])
print("usage_metadata :", full.usage_metadata)
print("response_meta  :", full.response_metadata)
```


## Streaming with Conversation History

```python
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

model = init_chat_model("gpt-4o-mini", model_provider="openai")

conversation = [
    SystemMessage("You are a Python tutor. Be concise."),
    HumanMessage("What is a decorator?"),
    AIMessage("A decorator wraps a function to modify its behavior without changing its code."),
    HumanMessage("Can you show me a simple example?")
]

for chunk in model.stream(conversation):
    print(chunk.content, end="", flush=True)

print()
```



## Content Blocks in Stream — The Right Way to Read Chunks

`content_blocks` gives you **typed structured access** to what each chunk contains.  
This is the correct way to handle text, reasoning, and tool calls in a single loop.

Each block has a `type` field:
- `"text"` — regular text content
- `"reasoning"` — thinking/reasoning tokens (Claude extended thinking, o-series)
- `"tool_call_chunk"` — partial tool call being built up

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai")

for chunk in model.stream("What color is the sky?"):
    for block in chunk.content_blocks:

        if block["type"] == "text":
            # Regular response text — print as it arrives
            print(block["text"], end="", flush=True)

        elif block["type"] == "reasoning":
            # Thinking tokens (Claude extended thinking / o-series reasoning)
            reasoning_text = block.get("reasoning", "")
            if reasoning_text:
                print(f"[THINKING] {reasoning_text}", end="", flush=True)

        elif block["type"] == "tool_call_chunk":
            # Partial tool call arriving piece by piece
            print(f"\n[TOOL CHUNK] name={block.get('name')} args={block.get('args')}")

print()
```


## Streaming Tool Call Chunks

When the model decides to call a tool, the tool call arguments stream in piece by piece.

```python
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny in {city}."

model = init_chat_model("gpt-4o-mini", model_provider="openai")
model_with_tools = model.bind_tools([get_weather])

for chunk in model_with_tools.stream("What is the weather in Paris and London?"):
    # tool_call_chunks arrive one fragment at a time
    for tc_chunk in chunk.tool_call_chunks:
        if tc_chunk.get("name"):
            print(f"\n[TOOL NAME ] {tc_chunk['name']}")
        if tc_chunk.get("id"):
            print(f"[TOOL ID   ] {tc_chunk['id']}")
        if tc_chunk.get("args"):
            print(f"[TOOL ARGS ] {tc_chunk['args']}", end="")

    # Regular text also comes through separately
    if chunk.content:
        print(chunk.content, end="", flush=True)

print()
```

Accumulate tool_call_chunks to get the complete tool calls:

```python
full = None

for chunk in model_with_tools.stream("What is the weather in Paris?"):
    full = chunk if full is None else full + chunk

# After accumulation, full.tool_calls contains the complete parsed tool calls
print("tool_calls:", full.tool_calls)
# [{'name': 'get_weather', 'args': {'city': 'Paris'}, 'id': 'call_abc...', 'type': 'tool_call'}]
```


## Streaming Reasoning Tokens (Claude Extended Thinking)

```python
from langchain_anthropic import ChatAnthropic

# Enable extended thinking on Anthropic
model = ChatAnthropic(
    model="claude-sonnet-4-6",
    thinking={"type": "enabled", "budget_tokens": 5000}
)

reasoning_parts = []
text_parts = []

for chunk in model.stream("Solve: if x^2 = 16 and x > 0, what is x?"):
    for block in chunk.content_blocks:
        if block["type"] == "reasoning":
            r = block.get("reasoning", "")
            if r:
                reasoning_parts.append(r)
                print(f"[THINK] {r}", end="", flush=True)
        elif block["type"] == "text":
            t = block.get("text", "")
            if t:
                text_parts.append(t)
                print(t, end="", flush=True)

print()
print("\n--- SUMMARY ---")
print("Reasoning chars:", sum(len(r) for r in reasoning_parts))
print("Answer         :", "".join(text_parts))
```



## `astream` — Async Streaming

`astream` is the async version of `stream`. Use it inside `async def` functions.  
Each iteration yields the same `AIMessageChunk` objects.

```python
import asyncio
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai")

async def stream_async():
    async for chunk in model.astream("Write a haiku about rain."):
        print(chunk.content, end="", flush=True)
    print()

asyncio.run(stream_async())
```

In a Jupyter notebook:

```python
async for chunk in model.astream("Write a haiku about rain."):
    print(chunk.content, end="", flush=True)
print()
```


## `astream` with Content Blocks (Async)

```python
async def stream_with_blocks():
    async for chunk in model.astream("Explain what a neural network is."):
        for block in chunk.content_blocks:
            if block["type"] == "text":
                print(block["text"], end="", flush=True)
            elif block["type"] == "reasoning":
                reasoning = block.get("reasoning", "")
                if reasoning:
                    print(f"[THINK] {reasoning}", end="")
    print()

await stream_with_blocks()
```



## `astream_events` — Semantic Event Stream

`astream_events` gives you named events instead of raw chunks.  
Useful when you want to react to specific lifecycle moments (start, token, end).

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai")

async def stream_events():
    async for event in model.astream_events("Tell me a fun fact.", version="v2"):
        kind = event["event"]

        if kind == "on_chat_model_start":
            print("[START] Model started generating")
            print("        Input:", event["data"]["input"])

        elif kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            token = chunk.content
            if token:
                print(token, end="", flush=True)

        elif kind == "on_chat_model_end":
            print("\n[END] Model finished")
            output = event["data"]["output"]
            print("      usage_metadata :", output.usage_metadata)
            print("      finish_reason  :", output.response_metadata.get("finish_reason"))

await stream_events()
```

Available event types:

```
on_chat_model_start   — model received the input, about to generate
on_chat_model_stream  — a chunk of output was produced
on_chat_model_end     — model finished, full output available
on_tool_start         — a tool is being invoked
on_tool_end           — a tool returned a result
on_chain_start        — a chain started execution
on_chain_end          — a chain finished execution
```



## Full Debug Print — Inspect Every Chunk

Use this in your notebook to understand exactly what each chunk contains.

```python
from langchain.chat_models import init_chat_model
import json

model = init_chat_model("gpt-4o-mini", model_provider="openai")

chunk_count = 0

for chunk in model.stream("What is deep learning?"):
    chunk_count += 1
    print(f"\n{'='*50}")
    print(f"CHUNK #{chunk_count}")
    print(f"{'='*50}")

    print(f"type                : {type(chunk).__name__}")
    print(f".type               : {chunk.type}")
    print(f".id                 : {chunk.id}")
    print(f".content            : {repr(chunk.content)}")

    # Tool call chunks
    print(f".tool_call_chunks   : {chunk.tool_call_chunks}")
    print(f".tool_calls         : {chunk.tool_calls}")

    # Content blocks
    print(f".content_blocks     : {chunk.content_blocks}")

    # Metadata (usually empty until last chunk)
    print(f".usage_metadata     : {chunk.usage_metadata}")
    print(f".response_metadata  : {chunk.response_metadata}")
    print(f".additional_kwargs  : {chunk.additional_kwargs}")

print(f"\nTotal chunks received: {chunk_count}")
```



## Watching `usage_metadata` Appear on the Last Chunk

Token counts are `None` on all chunks except the last one.

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai")

last_chunk = None

for i, chunk in enumerate(model.stream("What is the capital of Japan?")):
    last_chunk = chunk
    has_usage = chunk.usage_metadata is not None
    has_finish = bool(chunk.response_metadata.get("finish_reason"))
    print(f"Chunk {i:03d} | content={repr(chunk.content):<20} | has_usage={has_usage} | has_finish={has_finish}")

print("\n--- LAST CHUNK METADATA ---")
print("usage_metadata    :", last_chunk.usage_metadata)
print("response_metadata :", last_chunk.response_metadata)
```

Expected output:
```
Chunk 000 | content=''           | has_usage=False | has_finish=False
Chunk 001 | content='Tokyo'      | has_usage=False | has_finish=False
Chunk 002 | content=' is'        | has_usage=False | has_finish=False
...
Chunk 012 | content=''           | has_usage=True  | has_finish=True

--- LAST CHUNK METADATA ---
usage_metadata    : {'input_tokens': 16, 'output_tokens': 9, 'total_tokens': 25, ...}
response_metadata : {'finish_reason': 'stop', 'model_name': 'gpt-4o-mini-...'}
```


## Accumulate + Inspect Full Assembled Message

```python
import json
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai")

full = None

for chunk in model.stream("What is Python used for?"):
    full = chunk if full is None else full + chunk
    print(chunk.content, end="", flush=True)

print("\n")
print("=" * 60)
print("ASSEMBLED MESSAGE DEBUG")
print("=" * 60)

print("type              :", type(full).__name__)
print("content           :", full.content[:100])
print("usage_metadata    :")
print(json.dumps(full.usage_metadata, indent=2, default=str))
print("response_metadata :")
print(json.dumps(full.response_metadata, indent=2, default=str))
print("tool_calls        :", full.tool_calls)
print("content_blocks    :", full.content_blocks)
```

---

## Content Block Types — Reference

```
Block Type          When it appears                       Key fields
----------          ---------------                       ----------
"text"              Normal text response                   block["text"]
"reasoning"         Extended thinking (Anthropic o-series) block["reasoning"]
"tool_call_chunk"   Tool call being built up               block["name"], block["args"], block["id"], block["index"]
"tool_use"          Completed tool call (Anthropic native) block["name"], block["input"]
"image"             Image output from model                block["base64"], block["mime_type"]
```

---

## Streaming with System Prompt

```python
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

model = init_chat_model("gpt-4o-mini", model_provider="openai")

messages = [
    SystemMessage("You are an expert Python teacher. Be concise and use code examples."),
    HumanMessage("Explain list comprehensions.")
]

for chunk in model.stream(messages):
    print(chunk.content, end="", flush=True)

print()
```

---

## Streaming with Multiple Providers

`stream()` works identically across all providers:

```python
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

providers = [
    ("OpenAI",    init_chat_model("gpt-4o-mini", model_provider="openai")),
    ("Anthropic", ChatAnthropic(model="claude-sonnet-4-6")),
    ("Groq",      ChatGroq(model="llama-3.3-70b-versatile")),
    ("Gemini",    ChatGoogleGenerativeAI(model="gemini-2.0-flash")),
]

prompt = "Explain recursion in exactly two sentences."

for name, model in providers:
    print(f"\n[{name}] ", end="")
    for chunk in model.stream(prompt):
        print(chunk.content, end="", flush=True)
    print()
```

---

## Quick Reference Table

| Method | Sync/Async | Returns | Use for |
|---|---|---|---|
| `model.stream(input)` | Sync | Iterator of `AIMessageChunk` | Basic streaming in scripts/notebooks |
| `model.astream(input)` | Async | AsyncIterator of `AIMessageChunk` | Streaming in async apps / FastAPI |
| `model.astream_events(input, version="v2")` | Async | AsyncIterator of event dicts | Fine-grained lifecycle events |

| Field on `AIMessageChunk` | What it contains | Available on |
|---|---|---|
| `.content` | Partial text (often `""`) | All chunks |
| `.type` | Always `"AIMessageChunk"` | All chunks |
| `.id` | Message ID | First non-empty chunk |
| `.tool_call_chunks` | Partial tool call fragments | Chunks when tool is being called |
| `.tool_calls` | Partially assembled tool calls | Chunks when tool is being called |
| `.content_blocks` | Typed blocks: text, reasoning, tool_call_chunk | All chunks |
| `.usage_metadata` | Token counts | Last chunk only |
| `.response_metadata` | finish_reason, model_name | Last chunk only |

---

## What Comes Next

- `batch()` — send multiple prompts in parallel, get a list of `AIMessage` objects
- `with_structured_output()` — force the model to return JSON or Pydantic objects
- `bind_tools()` — let the model call external functions/APIs