# LangChain Messages — Developer Notes

> **Package:** `langchain-core` | **Docs:** https://docs.langchain.com/oss/python/langchain/messages
> **Version:** LangChain v1+ (content_blocks introduced in v1)

---

## Overview

Messages are the **fundamental unit of context** passed to and returned from LangChain chat models. Every model invocation takes a list of messages as input and returns an `AIMessage` as output. They carry three things: a **role** (who sent it), **content** (the payload — text, images, audio, etc.), and **metadata** (token usage, IDs, response info).

LangChain provides a standard message interface that works identically across all providers (OpenAI, Anthropic, Nebius, NVIDIA, etc.).

---

## Table of Contents

1. [Message Types](#1-message-types)
2. [Three Ways to Pass Messages](#2-three-ways-to-pass-messages)
3. [HumanMessage](#3-humanmessage)
4. [SystemMessage](#4-systemmessage)
5. [AIMessage](#5-aimessage)
6. [ToolMessage](#6-toolmessage)
7. [Message Content](#7-message-content)
8. [Standard Content Blocks](#8-standard-content-blocks)
9. [Multimodal Content](#9-multimodal-content)
10. [Streaming and Chunks](#10-streaming-and-chunks)
11. [Common Patterns](#11-common-patterns)
12. [Quick Reference](#12-quick-reference)

---

## 1. Message Types

| Type | Class | Role | When Created |
|---|---|---|---|
| System | `SystemMessage` | `system` | You — sets model behavior |
| Human | `HumanMessage` | `user` | You — user input |
| AI | `AIMessage` | `assistant` | Model — response output |
| Tool | `ToolMessage` | `tool` | You — tool execution result |

---

## 2. Three Ways to Pass Messages

All three formats are equivalent — pick whichever fits your context.

```python
# ── imports ────────────────────────────────────────────────────────────────
from langchain_nebius import ChatNebius
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

model = ChatNebius(model="Qwen/Qwen3-30B-A3B-fast", temperature=0.6)

# ── format 1: string shortcut (single HumanMessage) ───────────────────────
response = model.invoke("What is machine learning?")

# ── format 2: message objects ─────────────────────────────────────────────
messages = [
    SystemMessage("You are a concise ML expert."),
    HumanMessage("What is machine learning?"),
]
response = model.invoke(messages)

# ── format 3: dict / OpenAI chat format ───────────────────────────────────
messages = [
    {"role": "system",    "content": "You are a concise ML expert."},
    {"role": "user",      "content": "What is machine learning?"},
    {"role": "assistant", "content": "ML is..."},   # inject prior AI turn
    {"role": "user",      "content": "Give an example."},
]
response = model.invoke(messages)
```

> ⚠️ **Watch out:** String shortcut `model.invoke("text")` is always treated as a single `HumanMessage`. It does not accept a system prompt — use message objects for that.

---

## 3. HumanMessage

Represents user input. Can contain text, images, audio, files, or any multimodal data.

```python
# ── imports ────────────────────────────────────────────────────────────────
from langchain_core.messages import HumanMessage

# ── basic text ────────────────────────────────────────────────────────────
msg = HumanMessage("What is the KV cache?")

# ── with optional metadata ────────────────────────────────────────────────
msg = HumanMessage(
    content="Hello!",
    name="akshay",      # identify the user (provider-dependent behavior)
    id="msg_001",       # for tracing / logging
)

# ── multimodal: text + image ──────────────────────────────────────────────
msg = HumanMessage(content=[
    {"type": "text",  "text": "What is in this image?"},
    {"type": "image", "url": "https://example.com/chart.png"},
])
```

**Constructor parameters:**

| Param | Type | Description |
|---|---|---|
| `content` | `str \| list` | Text string or list of content blocks |
| `name` | `str` | Optional user identifier |
| `id` | `str` | Optional unique message ID for tracing |

---

## 4. SystemMessage

Sets the model's persona, tone, and constraints. Placed first in the message list.

```python
# ── imports ────────────────────────────────────────────────────────────────
from langchain_core.messages import SystemMessage, HumanMessage

# ── simple ────────────────────────────────────────────────────────────────
system = SystemMessage("You are a helpful coding assistant.")

# ── detailed persona ──────────────────────────────────────────────────────
system = SystemMessage("""
You are a senior Python developer with expertise in LLM inference systems.
Always provide runnable code examples.
Be concise. Never explain what you are about to do — just do it.
""")

response = model.invoke([system, HumanMessage("How does vLLM handle KV cache?")])
print(response.content)
```

> 💡 **Tip:** Static system prompts repeated across many requests are candidates for provider-level prompt caching (OpenAI, Anthropic). Keep the system prompt at the top and don't vary it per-request.

---

## 5. AIMessage

The object returned by every `model.invoke()` call. Contains the response text, tool calls, token usage, and response metadata.

```python
# ── imports ────────────────────────────────────────────────────────────────
from langchain_nebius import ChatNebius

model = ChatNebius(model="Qwen/Qwen3-30B-A3B-fast", temperature=0.6)
response = model.invoke("Explain flash attention.")

# ── type ──────────────────────────────────────────────────────────────────
print(type(response))           # <class 'langchain_core.messages.ai.AIMessage'>

# ── text content ──────────────────────────────────────────────────────────
print(response.content)         # the actual answer text

# ── shortcut for text ─────────────────────────────────────────────────────
print(response.text)            # same as .content for text-only responses

# ── token usage ───────────────────────────────────────────────────────────
print(response.usage_metadata)
# {'input_tokens': 12, 'output_tokens': 187, 'total_tokens': 199}

# ── response metadata (model name, finish reason, etc.) ───────────────────
print(response.response_metadata)

# ── unique message ID ─────────────────────────────────────────────────────
print(response.id)

# ── tool calls (empty list if none) ──────────────────────────────────────
print(response.tool_calls)      # [] or [{"name": ..., "args": ..., "id": ...}]
```

### AIMessage key attributes

| Attribute | Type | Description |
|---|---|---|
| `.content` | `str \| list` | Raw content of the response |
| `.text` | `str` | Text content shortcut |
| `.content_blocks` | `list[dict]` | Standardized parsed content (v1+) |
| `.tool_calls` | `list[dict]` | Tool calls made by the model |
| `.usage_metadata` | `dict \| None` | Token counts |
| `.response_metadata` | `dict \| None` | Provider metadata (finish reason, model name) |
| `.id` | `str` | Unique message identifier |

### Token usage breakdown

```python
# ── full usage_metadata structure ─────────────────────────────────────────
usage = response.usage_metadata
# {
#   'input_tokens': 12,
#   'output_tokens': 187,
#   'total_tokens': 199,
#   'input_token_details':  {'cache_read': 0, 'audio': 0},
#   'output_token_details': {'reasoning': 120, 'audio': 0},  # if reasoning model
# }

input_toks    = usage.get("input_tokens",  0)
output_toks   = usage.get("output_tokens", 0)
reasoning_tok = usage.get("output_token_details", {}).get("reasoning", 0)
```

### Manually inserting AIMessage into history

```python
# ── inject a prior AI turn to simulate conversation history ────────────────
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

messages = [
    SystemMessage("You are a helpful assistant."),
    HumanMessage("Can you help me with Python?"),
    AIMessage("Of course! What do you need help with?"),   # injected manually
    HumanMessage("How do I sort a list?"),
]
response = model.invoke(messages)
```

---

## 6. ToolMessage

Carries the result of a tool execution back to the model. Must reference the tool call ID from the preceding `AIMessage`.

```python
# ── imports ────────────────────────────────────────────────────────────────
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

model = ChatNVIDIA(model="nvidia/nemotron-super-120b-v1")

# ── step 1: user asks, model decides to call a tool ───────────────────────
ai_msg = AIMessage(
    content=[],
    tool_calls=[{
        "name":  "get_stock_price",
        "args":  {"symbol": "NVDA"},
        "id":    "call_abc123",      # ← must reference this ID
    }]
)

# ── step 2: execute tool, wrap result ─────────────────────────────────────
tool_result = "NVDA: $875.20"
tool_msg = ToolMessage(
    content=tool_result,
    tool_call_id="call_abc123",      # ← must match ai_msg tool call id
    name="get_stock_price",
)

# ── step 3: model processes tool result ───────────────────────────────────
messages = [
    HumanMessage("What is NVIDIA's stock price?"),
    ai_msg,
    tool_msg,
]
response = model.invoke(messages)
print(response.content)
```

### ToolMessage with artifact (metadata not sent to model)

```python
# ── artifact: metadata stored locally, not sent to model ──────────────────
tool_msg = ToolMessage(
    content="Flash attention reduces memory from O(n²) to O(n).",   # sent to model
    tool_call_id="call_xyz",
    name="search_papers",
    artifact={                       # NOT sent to model — for your app only
        "paper_id": "arxiv:2205.14135",
        "page": 4,
        "confidence": 0.94,
    }
)
# Access artifact downstream
print(tool_msg.artifact["paper_id"])   # "arxiv:2205.14135"
```

**ToolMessage required fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `content` | `str` | ✓ | Stringified tool output sent to model |
| `tool_call_id` | `str` | ✓ | Must match the `id` in the AIMessage tool call |
| `name` | `str` | ✓ | Name of the tool that was called |
| `artifact` | `dict` | ✕ | Extra data NOT forwarded to model |

---

## 7. Message Content

The `content` attribute of any message accepts three formats:

```python
# ── format 1: plain string (most common) ──────────────────────────────────
msg = HumanMessage("Hello!")

# ── format 2: provider-native list (OpenAI-style) ─────────────────────────
msg = HumanMessage(content=[
    {"type": "text",      "text": "What is in this image?"},
    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
])

# ── format 3: LangChain standard content blocks (cross-provider) ──────────
msg = HumanMessage(content_blocks=[
    {"type": "text",  "text": "What is in this image?"},
    {"type": "image", "url": "https://example.com/img.jpg"},
])
# content_blocks populates .content automatically — type-safe interface
```

> ⚠️ **Watch out:** Provider-native content format (format 2) is provider-specific. Use `content_blocks` (format 3) for code that runs against multiple providers.

---

## 8. Standard Content Blocks

`AIMessage.content_blocks` is a standardized, lazily-parsed view of `content`. Provider-specific formats (Anthropic `thinking`, OpenAI `reasoning`) are normalized into a consistent schema.

```python
# ── reading content blocks from a response ────────────────────────────────
response = model.invoke("Explain speculative decoding step by step.")

for block in response.content_blocks:
    if block["type"] == "text":
        print("Text:", block["text"])
    elif block["type"] == "reasoning":
        print("Reasoning:", block["reasoning"][:100], "...")
    elif block["type"] == "tool_call":
        print("Tool call:", block["name"], block["args"])
```

### Content block types

**Core blocks:**

| Block Type | `type` value | Key Fields | Use |
|---|---|---|---|
| Text | `"text"` | `text`, `annotations` | Standard text output |
| Reasoning | `"reasoning"` | `reasoning`, `extras` | Model chain-of-thought |

**Multimodal blocks:**

| Block Type | `type` value | Key Fields | Use |
|---|---|---|---|
| Image | `"image"` | `url` or `base64`, `mime_type`, `id` | Image input/output |
| Audio | `"audio"` | `url` or `base64`, `mime_type`, `id` | Audio input |
| Video | `"video"` | `url` or `base64`, `mime_type`, `id` | Video input |
| File (PDF etc.) | `"file"` | `url` or `base64`, `mime_type`, `id` | Document input |
| Plain text doc | `"text-plain"` | `text`, `mime_type` | `.txt`, `.md` files |

**Tool blocks:**

| Block Type | `type` value | Key Fields | Use |
|---|---|---|---|
| Tool call | `"tool_call"` | `name`, `args`, `id` | Function call made by model |
| Tool call chunk | `"tool_call_chunk"` | `name`, `args`, `index` | Streaming fragment |
| Invalid tool call | `"invalid_tool_call"` | `name`, `error` | JSON parse failure |

**Server-side tool blocks:**

| Block Type | `type` value | Key Fields | Use |
|---|---|---|---|
| Server tool call | `"server_tool_call"` | `id`, `name`, `args` | Executed server-side |
| Server tool result | `"server_tool_result"` | `tool_call_id`, `status`, `output` | Server tool output |

### Reasoning blocks example (Anthropic vs OpenAI normalization)

```python
# ── Anthropic returns "thinking" blocks ───────────────────────────────────
from langchain_core.messages import AIMessage

anthropic_msg = AIMessage(
    content=[
        {"type": "thinking", "thinking": "The user wants...", "signature": "abc"},
        {"type": "text",     "text": "Here is my answer."},
    ],
    response_metadata={"model_provider": "anthropic"},
)
print(anthropic_msg.content_blocks)
# [{'type': 'reasoning', 'reasoning': 'The user wants...', 'extras': {'signature': 'abc'}},
#  {'type': 'text', 'text': 'Here is my answer.'}]

# ── OpenAI returns "reasoning" summary blocks ─────────────────────────────
openai_msg = AIMessage(
    content=[
        {"type": "reasoning", "id": "rs_001", "summary": [
            {"type": "summary_text", "text": "Step 1: analyze..."},
        ]},
        {"type": "text", "text": "Final answer.", "id": "msg_001"},
    ],
    response_metadata={"model_provider": "openai"},
)
print(openai_msg.content_blocks)
# [{'type': 'reasoning', 'id': 'rs_001', 'reasoning': 'Step 1: analyze...'},
#  {'type': 'text', 'text': 'Final answer.', 'id': 'msg_001'}]
```

Both normalize to `type: "reasoning"` — your parsing code works unchanged across providers.

### Serialize content blocks (opt-in via `output_version="v1"`)

```python
# ── store standardized content blocks in message.content ──────────────────
from langchain_nebius import ChatNebius

model = ChatNebius(
    model="Qwen/Qwen3-30B-A3B-fast",
    output_version="v1",              # content will use standard block format
)
# or via env var: LC_OUTPUT_VERSION=v1
```

---

## 9. Multimodal Content

Pass images, audio, video, and documents directly in `HumanMessage.content`.

```python
# ── image from URL ────────────────────────────────────────────────────────
msg = {
    "role": "user",
    "content": [
        {"type": "text",  "text": "Describe this architecture diagram."},
        {"type": "image", "url": "https://example.com/arch.png"},
    ]
}

# ── image from base64 ─────────────────────────────────────────────────────
import base64
with open("chart.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

msg = {
    "role": "user",
    "content": [
        {"type": "text",  "text": "What trend does this chart show?"},
        {"type": "image", "base64": b64, "mime_type": "image/png"},
    ]
}

# ── PDF document ──────────────────────────────────────────────────────────
msg = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Summarize this paper."},
        {"type": "file", "url": "https://arxiv.org/pdf/2205.14135"},
    ]
}

# ── audio ─────────────────────────────────────────────────────────────────
msg = {
    "role": "user",
    "content": [
        {"type": "text",  "text": "Transcribe this audio."},
        {"type": "audio", "base64": audio_b64, "mime_type": "audio/wav"},
    ]
}
```

> ⚠️ **Watch out:** Not all models support all modalities. Nebius (Qwen3) supports text only. NVIDIA Nemotron Nano Omni supports text + audio. Check the model card before passing multimodal content.

---

## 10. Streaming and Chunks

Streaming returns `AIMessageChunk` objects. Accumulate them to build the full message.

```python
# ── sync streaming ────────────────────────────────────────────────────────
from langchain_nebius import ChatNebius

model = ChatNebius(model="Qwen/Qwen3-30B-A3B-fast", temperature=0.6)

chunks = []
full_message = None

for chunk in model.stream("Explain PagedAttention in vLLM."):
    print(chunk.text, end="", flush=True)         # print token-by-token
    chunks.append(chunk)
    full_message = chunk if full_message is None else full_message + chunk

print()
print(f"\nTotal chunks: {len(chunks)}")
print(f"Full response type: {type(full_message)}")   # AIMessageChunk (addable)
print(f"Content: {full_message.content}")
```

```python
# ── async streaming ───────────────────────────────────────────────────────
async def stream_response():
    async for chunk in model.astream("What is speculative decoding?"):
        print(chunk.text, end="", flush=True)

await stream_response()   # works directly in Jupyter
```

```python
# ── streaming with tool calls ─────────────────────────────────────────────
for chunk in model_with_tools.stream("What's the stock price of NVDA?"):
    for tool_chunk in chunk.tool_call_chunks:
        print(f"Tool: {tool_chunk['name']} | Args so far: {tool_chunk['args']}")
```

---

## 11. Common Patterns

### Multi-turn conversation loop

```python
# ── stateless multi-turn: manually grow the message list ──────────────────
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_nebius import ChatNebius

model   = ChatNebius(model="Qwen/Qwen3-30B-A3B-fast", temperature=0.6)
history = [SystemMessage("You are a concise assistant. Keep replies under 3 sentences.")]

questions = [
    "What is the KV cache in LLM inference?",
    "How does vLLM improve on the naive KV cache?",
    "What is PagedAttention?",
]

for q in questions:
    history.append(HumanMessage(q))
    response = model.invoke(history)
    history.append(response)             # AIMessage goes back into history
    print(f"Q: {q}\nA: {response.content}\n")
```

### Tool calling full loop

```python
# ── imports ────────────────────────────────────────────────────────────────
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA

@tool
def get_gpu_specs(model_name: str) -> str:
    """Get technical specifications for an NVIDIA GPU model."""
    specs = {
        "H100": "80GB HBM3, 3.35TB/s bandwidth, 989 TFLOPS BF16",
        "A100": "80GB HBM2e, 2TB/s bandwidth, 312 TFLOPS BF16",
    }
    return specs.get(model_name, f"No data for {model_name}")


model        = ChatNVIDIA(model="nvidia/nemotron-super-120b-v1")
model_w_tool = model.bind_tools([get_gpu_specs])

messages = [HumanMessage("What are the H100 GPU specs?")]

# ── turn 1: model decides to call a tool ──────────────────────────────────
ai_response = model_w_tool.invoke(messages)
messages.append(ai_response)

# ── execute tool calls ─────────────────────────────────────────────────────
for tc in ai_response.tool_calls:
    result = get_gpu_specs.invoke(tc["args"])
    messages.append(ToolMessage(
        content=str(result),
        tool_call_id=tc["id"],
        name=tc["name"],
    ))

# ── turn 2: model synthesizes tool result ─────────────────────────────────
final = model_w_tool.invoke(messages)
print(final.content)
```

### Inspecting content blocks after invoke

```python
# ── parse all block types from a response ─────────────────────────────────
response = model.invoke("Plan a 3-step RAG pipeline.")

for i, block in enumerate(response.content_blocks):
    btype = block["type"]
    if btype == "text":
        print(f"[{i}] TEXT:      {block['text'][:80]}...")
    elif btype == "reasoning":
        print(f"[{i}] REASONING: {block['reasoning'][:80]}...")
    elif btype == "tool_call":
        print(f"[{i}] TOOL CALL: {block['name']}({block['args']})")
    else:
        print(f"[{i}] OTHER:     {btype}")
```

---

## 12. Quick Reference

### Message classes

| Class | Import | Role | When to use |
|---|---|---|---|
| `SystemMessage` | `langchain_core.messages` | `system` | Set model behavior / persona |
| `HumanMessage` | `langchain_core.messages` | `user` | User input, multimodal content |
| `AIMessage` | `langchain_core.messages` | `assistant` | Model response — also inject into history |
| `ToolMessage` | `langchain_core.messages` | `tool` | Return tool execution result to model |
| `AIMessageChunk` | `langchain_core.messages` | `assistant` | Streaming fragment — accumulate with `+` |

### AIMessage attributes cheatsheet

| Attribute | Type | Description |
|---|---|---|
| `.content` | `str \| list` | Raw content |
| `.text` | `str` | Text shortcut |
| `.content_blocks` | `list[dict]` | Standardized blocks (v1+) |
| `.tool_calls` | `list[dict]` | Tool calls: `[{name, args, id}]` |
| `.usage_metadata` | `dict` | `{input_tokens, output_tokens, total_tokens}` |
| `.response_metadata` | `dict` | Finish reason, model name, etc. |
| `.id` | `str` | Unique message ID |

### Content block types cheatsheet

| `type` value | Purpose | Key fields |
|---|---|---|
| `"text"` | Text output | `text` |
| `"reasoning"` | Thinking / chain-of-thought | `reasoning`, `extras` |
| `"image"` | Image data | `url` or `base64`, `mime_type` |
| `"audio"` | Audio data | `url` or `base64`, `mime_type` |
| `"video"` | Video data | `url` or `base64`, `mime_type` |
| `"file"` | PDF / document | `url` or `base64`, `mime_type` |
| `"tool_call"` | Function invocation | `name`, `args`, `id` |
| `"tool_call_chunk"` | Streaming tool fragment | `name`, `args`, `index` |
| `"server_tool_call"` | Server-executed tool | `id`, `name`, `args` |
| `"server_tool_result"` | Server tool output | `tool_call_id`, `status`, `output` |
| `"non_standard"` | Provider escape hatch | `value` |

### Input format comparison

| Format | Code | Best for |
|---|---|---|
| String | `model.invoke("text")` | Single standalone request |
| Message objects | `model.invoke([SystemMessage(...), HumanMessage(...)])` | Multi-turn, system prompts |
| Dict / OpenAI format | `model.invoke([{"role": "user", "content": "..."}])` | Portability, JSON serialization |
| content_blocks | `HumanMessage(content_blocks=[...])` | Multimodal, cross-provider |

### Gotchas

| Gotcha | Fix |
|---|---|
| `ToolMessage` causes `400` error | Ensure `tool_call_id` matches exactly the ID in `AIMessage.tool_calls` |
| `.content_blocks` returns empty | Upgrade to LangChain v1+ — property not available in older versions |
| Reasoning blocks not appearing | Enable reasoning: `.with_thinking_mode(enabled=True)` for NVIDIA; `thinking={"type": "enabled", "budget_tokens": 1024}` for Anthropic |
| Provider-native content breaks on switch | Use `content_blocks=` instead of `content=` with raw dicts |
| Token counts show 0 | Some providers don't return usage on all endpoints — read `response.usage_metadata` directly |
| `model.invoke("text")` ignores system prompt | Use `[SystemMessage(...), HumanMessage("text")]` instead |

---

> ✓ **Official docs:** https://docs.langchain.com/oss/python/langchain/messages
> ✓ **API reference:** https://reference.langchain.com/python/langchain/messages