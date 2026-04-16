# **LangChain — Using Different LLMs**
## Part 1: The `invoke` Method


## What is `invoke`?

`invoke()` is the most basic way to call a chat model in LangChain.

- You send a message (or a list of messages) to the model
- The model generates the **full response first**, then returns it
- The return type is an `AIMessage` object (not a plain string)
- Access the text using `.content` on the returned object

**When to use invoke:**
- Short factual queries
- API backends that need a complete response before processing
- When you do not need partial/streaming output



## How LangChain Connects to Providers

LangChain uses a unified function `init_chat_model()` as the standard entry point.  
Every provider has its own installable package but they all share the same interface.  
You can swap providers without rewriting your invoke logic.

```
Provider          Package                    Env Variable
---------         -------                    ------------
OpenAI            langchain-openai           OPENAI_API_KEY
Anthropic         langchain-anthropic        ANTHROPIC_API_KEY
Google Gemini     langchain-google-genai     GOOGLE_API_KEY
Groq              langchain-groq             GROQ_API_KEY
Cerebras          langchain-cerebras         CEREBRAS_API_KEY
NVIDIA            langchain-nvidia-ai-endpoints   NVIDIA_API_KEY
Nebius            langchain-nebius           NEBIUS_API_KEY
OpenRouter        langchain-openrouter       OPENROUTER_API_KEY
HuggingFace       langchain-huggingface      HUGGINGFACEHUB_API_TOKEN
```

---

## Installation

```bash
# Install core + all providers used in these notes
pip install langchain
pip install langchain-openai
pip install langchain-anthropic
pip install langchain-google-genai
pip install langchain-groq
pip install langchain-cerebras
pip install langchain-nvidia-ai-endpoints
pip install langchain-nebius
pip install langchain-openrouter
pip install langchain-huggingface
pip install python-dotenv
```

---

## Setup — Load API Keys

```python
import os
from dotenv import load_dotenv

load_dotenv()   # reads from your .env file

# All keys are now available as environment variables
# LangChain providers pick them up automatically
```

---

## `init_chat_model` — The Unified Initializer

```python
from langchain.chat_models import init_chat_model

# Signature
# init_chat_model(model, model_provider=None, **kwargs)

# kwargs you can pass:
#   temperature   — controls randomness (0.0 = deterministic, 1.0 = creative)
#   max_tokens    — max length of the response
#   timeout       — seconds before the request is cancelled
#   max_retries   — how many times to retry on failure (default 6)
```

---

## invoke — Input Formats

`invoke()` accepts three input formats:

```python
# 1. Plain string (simplest)
response = model.invoke("What is the capital of France?")

# 2. List of dicts (conversation history style)
response = model.invoke([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "What is the capital of France?"}
])

# 3. List of LangChain message objects
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

response = model.invoke([
    SystemMessage("You are a helpful assistant."),
    HumanMessage("What is the capital of France?")
])

# Access the response text
print(response.content)
```

---

## Provider Examples

### 1. OpenAI

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "gpt-4o-mini",
    model_provider="openai",
    temperature=0.7,
    max_tokens=500
)

response = model.invoke("Explain transformers in one paragraph.")
print(response.content)
```

Using the provider class directly:

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
response = model.invoke("Explain transformers in one paragraph.")
print(response.content)
```

---

### 2. Anthropic (Claude)

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "claude-sonnet-4-6",
    model_provider="anthropic",
    temperature=0.5,
    max_tokens=500
)

response = model.invoke("What is backpropagation?")
print(response.content)
```

Using the provider class directly:

```python
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.5)
response = model.invoke("What is backpropagation?")
print(response.content)
```

---

### 3. Google Gemini

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "gemini-2.0-flash",
    model_provider="google_genai",
    temperature=0.3
)

response = model.invoke("Summarize the water cycle in 3 sentences.")
print(response.content)
```

Using the provider class directly:

```python
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
response = model.invoke("Summarize the water cycle in 3 sentences.")
print(response.content)
```

---

### 4. Groq

Groq runs open-source models (Llama, Mixtral, etc.) on its own fast inference hardware.

```python
from langchain_groq import ChatGroq

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.6,
    max_tokens=500
)

response = model.invoke("What is attention mechanism in neural networks?")
print(response.content)
```

---

### 5. Cerebras

Cerebras is a hardware provider offering very fast inference on wafer-scale chips.

```python
from langchain_cerebras import ChatCerebras

model = ChatCerebras(
    model="llama-3.3-70b",
    temperature=0.5
)

response = model.invoke("What is gradient descent?")
print(response.content)
```

---

### 6. NVIDIA NIM

NVIDIA NIM serves optimized models through its API catalog.

```python
from langchain_nvidia_ai_endpoints import ChatNVIDIA

model = ChatNVIDIA(
    model="meta/llama-3.3-70b-instruct",
    temperature=0.5,
    max_tokens=500
)

response = model.invoke("Explain GPU parallelism briefly.")
print(response.content)
```

---

### 7. Nebius

Nebius AI Studio gives API access to open-source models like Qwen, Llama, etc.

```python
from langchain_nebius import ChatNebius

model = ChatNebius(
    model="Qwen/Qwen3-30B-A3B-fast",
    temperature=0.6
)

response = model.invoke("What is a language model?")
print(response.content)
```

---

### 8. OpenRouter

OpenRouter is a unified gateway to hundreds of models from many providers.

```python
from langchain_openrouter import ChatOpenRouter

model = ChatOpenRouter(
    model="meta-llama/llama-3.3-70b-instruct",
    temperature=0.5
)

response = model.invoke("What is reinforcement learning?")
print(response.content)
```

---

### 9. HuggingFace

HuggingFace hosts thousands of open-source models accessible via Inference API.

```python
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    temperature=0.7,
    max_new_tokens=512,
)

model = ChatHuggingFace(llm=llm)
response = model.invoke("What is the difference between CNN and RNN?")
print(response.content)
```

---

## invoke — With Conversation History

You can pass a list of messages to simulate a multi-turn conversation.

```python
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

model = init_chat_model("gpt-4o-mini", model_provider="openai")

conversation = [
    SystemMessage("You are a Python tutor. Keep answers short and clear."),
    HumanMessage("What is a list in Python?"),
    AIMessage("A list is an ordered, mutable collection. Example: my_list = [1, 2, 3]"),
    HumanMessage("How do I add an item to it?")
]

response = model.invoke(conversation)
print(response.content)
```

---

## invoke — With Parameters (temperature, max_tokens)

```python
from langchain.chat_models import init_chat_model

# temperature = 0   -> very deterministic, consistent answers
# temperature = 1   -> more creative, varied answers

model_precise = init_chat_model(
    "gpt-4o-mini",
    model_provider="openai",
    temperature=0,
    max_tokens=200
)

model_creative = init_chat_model(
    "gpt-4o-mini",
    model_provider="openai",
    temperature=1.0,
    max_tokens=200
)

prompt = "Write a one-line tagline for a coffee shop."

print(model_precise.invoke(prompt).content)
print(model_creative.invoke(prompt).content)
```

---

## invoke — Accessing the Full AIMessage Object

The response is not just a string. It is an `AIMessage` object with extra metadata.

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai")
response = model.invoke("What is 2 + 2?")

print(type(response))             # <class 'langchain_core.messages.ai.AIMessage'>
print(response.content)           # "2 + 2 equals 4."
print(response.response_metadata) # token counts, model name, finish reason, etc.
print(response.usage_metadata)    # input_tokens, output_tokens, total_tokens
```

---

## invoke — Token Usage Tracking

```python
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler

model = init_chat_model("gpt-4o-mini", model_provider="openai")

callback = UsageMetadataCallbackHandler()

response = model.invoke(
    "Explain overfitting in machine learning.",
    config={"callbacks": [callback]}
)

print(response.content)
print(callback.usage_metadata)
# {'gpt-4o-mini-...': {'input_tokens': 12, 'output_tokens': 80, 'total_tokens': 92}}
```

---

## invoke — Comparing Multiple Providers

```python
from langchain.chat_models import init_chat_model

prompt = "What is the meaning of life? Answer in one sentence."

providers = [
    ("gpt-4o-mini",          "openai"),
    ("claude-sonnet-4-6",    "anthropic"),
    ("gemini-2.0-flash",     "google_genai"),
]

for model_name, provider in providers:
    model = init_chat_model(model_name, model_provider=provider, temperature=0)
    response = model.invoke(prompt)
    print(f"[{provider}] {response.content}\n")
```

---

## Quick Reference Table

| Category | Detail |
|---|---|
| Method | `model.invoke(input)` |
| Input types | string, list of dicts, list of message objects |
| Return type | `AIMessage` |
| Access text | `response.content` |
| Access metadata | `response.response_metadata` |
| Access token counts | `response.usage_metadata` |
| Key parameter: temperature | 0 = deterministic, 1 = creative |
| Key parameter: max_tokens | limits response length |
| Key parameter: max_retries | retries on failure, default = 6 |
| Key parameter: timeout | seconds before request is cancelled |

---

## Notes on Model Name Formats

```python
# Format 1: provider prefix in the model string
model = init_chat_model("openai:gpt-4o-mini")
model = init_chat_model("anthropic:claude-sonnet-4-6")
model = init_chat_model("google_genai:gemini-2.0-flash")

# Format 2: separate model_provider argument
model = init_chat_model("gpt-4o-mini", model_provider="openai")

# Both formats produce identical behaviour
```

### **What Comes Next**

- `stream()` — get tokens as they are generated, one by one
- `batch()` — send multiple prompts in parallel
- `with_structured_output()` — force the model to return JSON or Pydantic objects
- `bind_tools()` — let the model call external functions/APIs

</br>

---

# **Deep Dive into `invoke` Response (`AIMessage`)**

All Attributes, Fields, Methods, and Debugging Prints

## **What Does `invoke` Actually Return?**

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai")
response = model.invoke("What is Python?")

print(type(response))
# <class 'langchain_core.messages.ai.AIMessage'>
```

`invoke` returns an `AIMessage` object, which is a Pydantic model (built on top of `BaseMessage`).  
It contains many fields beyond just `.content`. This note covers all of them.


## **Complete List of All Fields on `AIMessage`**

```
Field                  Type                    Description
-----                  ----                    -----------
.content               str | list              The actual text response from the model
.type                  Literal["ai"]           Always "ai" — identifies message type
.id                    str | None              Unique message ID assigned by the provider
.name                  str | None              Optional name label for the message
.additional_kwargs     dict                    Extra raw data from the provider (legacy)
.response_metadata     dict                    Provider metadata: model name, tokens, finish reason, logprobs
.usage_metadata        UsageMetadata | None    Standardized token counts across all providers
.tool_calls            list[ToolCall]          Tool calls the model wants to make
.invalid_tool_calls    list[InvalidToolCall]   Tool calls that failed to parse
.content_blocks        list[ContentBlock]      Typed structured content (text, image, reasoning, tool calls)
.lc_attributes         dict                    Attributes included in serialization
```

## **Printing Every Field One Block at a Time**

### Block 1: The Basics

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai")
response = model.invoke("Explain list comprehensions in Python briefly.")

# The actual text output
print("content        :", response.content)

# Message type identifier (always "ai")
print("type           :", response.type)

# Unique message ID from the provider
print("id             :", response.id)

# Optional name (usually None unless set manually)
print("name           :", response.name)
```

---

### Block 2 — Token Usage (`usage_metadata`)

This is LangChain's standardized token count — consistent across all providers.

```python
print("usage_metadata :", response.usage_metadata)

# Unpack each field inside usage_metadata
if response.usage_metadata:
    um = response.usage_metadata
    print("  input_tokens       :", um.get("input_tokens"))
    print("  output_tokens      :", um.get("output_tokens"))
    print("  total_tokens       :", um.get("total_tokens"))

    # These may be present for some providers (OpenAI, Anthropic)
    print("  input_token_details  :", um.get("input_token_details"))   # cache_read, audio
    print("  output_token_details :", um.get("output_token_details"))  # reasoning, audio
```

Sample output (OpenAI):
```
usage_metadata : {'input_tokens': 18, 'output_tokens': 95, 'total_tokens': 113,
                  'input_token_details': {'audio': 0, 'cache_read': 0},
                  'output_token_details': {'audio': 0, 'reasoning': 0}}
```

Sample output (Anthropic):
```
usage_metadata : {'input_tokens': 18, 'output_tokens': 95, 'total_tokens': 113,
                  'input_token_details': {'cache_read': 0, 'cache_creation': 0}}
```

---

### Block 3 — Provider Metadata (`response_metadata`)

This is raw metadata returned by the provider's API. Contents vary per provider.

```python
import json

print("response_metadata:")
print(json.dumps(response.response_metadata, indent=2, default=str))
```

Sample output (OpenAI):
```json
{
  "token_usage": {
    "completion_tokens": 95,
    "prompt_tokens": 18,
    "total_tokens": 113,
    "completion_tokens_details": {"reasoning_tokens": 0, "audio_tokens": 0},
    "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0}
  },
  "model_name": "gpt-4o-mini-2024-07-18",
  "system_fingerprint": "fp_abc123",
  "finish_reason": "stop",
  "logprobs": null
}
```

Sample output (Anthropic):
```json
{
  "id": "msg_01Xyz...",
  "model": "claude-sonnet-4-6-20250929",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 18,
    "output_tokens": 95,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0
  }
}
```

Sample output (Groq):
```json
{
  "token_usage": {
    "completion_tokens": 95,
    "prompt_tokens": 18,
    "total_tokens": 113,
    "queue_time": 0.001
  },
  "model_name": "llama-3.3-70b-versatile",
  "finish_reason": "stop"
}
```

---

### Block 4 — Specific Keys Inside `response_metadata`

```python
meta = response.response_metadata

# Model name used (useful when using configurable models)
print("model_name     :", meta.get("model_name"))

# Why generation stopped: "stop", "length", "tool_calls", "content_filter"
print("finish_reason  :", meta.get("finish_reason"))

# OpenAI system fingerprint (identifies model version/config)
print("system_fingerprint :", meta.get("system_fingerprint"))

# Log probabilities (only present if logprobs=True was set on the model)
print("logprobs       :", meta.get("logprobs"))

# Raw token usage dict from the provider (different from usage_metadata)
print("token_usage    :", meta.get("token_usage"))
```

---

### Block 5 — Additional Kwargs

```python
# Legacy field. Contains raw extra data from the provider.
# Usually empty unless tool calls are present or provider sends extra fields.
print("additional_kwargs:")
print(json.dumps(response.additional_kwargs, indent=2, default=str))
```

---

### Block 6 — Tool Calls (when the model called a tool)

```python
# List of tool calls the model is requesting to make
print("tool_calls:", response.tool_calls)

# Loop through each tool call
for tc in response.tool_calls:
    print("  name :", tc["name"])
    print("  args :", tc["args"])
    print("  id   :", tc["id"])
    print("  type :", tc["type"])  # always "tool_call"

# Tool calls that had parsing errors
print("invalid_tool_calls:", response.invalid_tool_calls)
```

---

### Block 7 — Content Blocks

`content_blocks` gives structured typed access to the response content.  
Especially useful when the model returns reasoning, images, or multiple parts.

```python
print("content_blocks:", response.content_blocks)

for block in response.content_blocks:
    print("  block type:", block["type"])

    if block["type"] == "text":
        print("  text:", block["text"])

    elif block["type"] == "reasoning":
        print("  reasoning:", block.get("reasoning"))

    elif block["type"] == "tool_use":
        print("  tool_name :", block.get("name"))
        print("  tool_input:", block.get("input"))

    elif block["type"] == "image":
        print("  image mime_type:", block.get("mime_type"))
        print("  image base64 (first 40 chars):", str(block.get("base64", ""))[:40])
```

---

### Block 8 — Serialization Fields (Internal / Debug)

These come from the Pydantic/LangChain serialization system.

```python
# Attributes included in LangChain serialization
print("lc_attributes  :", response.lc_attributes)

# Class identifier path used for serialization/deserialization
print("lc_id          :", response.lc_id())

# Whether this class supports LangChain serialization
print("is_lc_serializable:", response.is_lc_serializable())

# The module namespace of this class
print("lc_namespace   :", response.get_lc_namespace())
```

---

## Full Debug Print — Print Everything at Once

Use this in your notebook when you want to see the entire `AIMessage` structure.

```python
import json
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai")
response = model.invoke("What is Python?")

print("=" * 60)
print("FULL AIMessage DEBUG")
print("=" * 60)

print("\n--- BASIC FIELDS ---")
print("type           :", response.type)
print("id             :", response.id)
print("name           :", response.name)
print("content        :", response.content)

print("\n--- USAGE METADATA (standardized) ---")
print(json.dumps(response.usage_metadata, indent=2, default=str))

print("\n--- RESPONSE METADATA (raw provider data) ---")
print(json.dumps(response.response_metadata, indent=2, default=str))

print("\n--- ADDITIONAL KWARGS ---")
print(json.dumps(response.additional_kwargs, indent=2, default=str))

print("\n--- TOOL CALLS ---")
print(response.tool_calls)

print("\n--- INVALID TOOL CALLS ---")
print(response.invalid_tool_calls)

print("\n--- CONTENT BLOCKS ---")
for i, block in enumerate(response.content_blocks):
    print(f"  block[{i}]:", block)

print("\n--- SERIALIZATION INFO ---")
print("lc_attributes      :", response.lc_attributes)
print("is_lc_serializable :", response.is_lc_serializable())
print("lc_namespace       :", response.get_lc_namespace())
print("lc_id              :", response.lc_id())

print("=" * 60)
```

---

## `pretty_repr` and `pretty_print` — Human Readable

```python
# Returns a formatted string for display
print(response.pretty_repr())

# Directly prints to stdout — same result but no return value
response.pretty_print()
```

Sample output:
```
================================== Ai Message ==================================

Python is a high-level, general-purpose programming language known for its
clear syntax and readability...
```

With tool calls:
```
================================== Ai Message ==================================
Let me check the weather.
Tool Calls:
  get_weather (call_abc123)
  Call ID: call_abc123
    Args:
      city: Paris
```

---

## `model_dump` — Convert to Plain Python Dict

```python
# Converts the entire AIMessage into a plain Python dict
data = response.model_dump()
print(type(data))   # <class 'dict'>
print(data.keys())
# dict_keys(['content', 'additional_kwargs', 'response_metadata', 'type',
#            'name', 'id', 'tool_calls', 'invalid_tool_calls', 'usage_metadata'])

# Access fields from the dict
print(data["content"])
print(data["usage_metadata"])
print(data["response_metadata"])
```

---

## `to_json` — Serialize to LangChain JSON Format

```python
import json

# Serialize to a LangChain-compatible JSON structure
serialized = response.to_json()
print(type(serialized))   # <class 'dict'>

# Pretty print
print(json.dumps(serialized, indent=2, default=str))
```

Sample output structure:
```json
{
  "lc": 1,
  "type": "constructor",
  "id": ["langchain", "schema", "messages", "AIMessage"],
  "kwargs": {
    "content": "Python is...",
    "type": "ai",
    "id": "chatcmpl-abc123",
    "tool_calls": [],
    "usage_metadata": {...},
    "response_metadata": {...}
  }
}
```

---

## `dumps` / `dumpd` — LangChain Serialization Utilities

```python
from langchain_core.load import dumps, dumpd

# dumps: serialize to JSON string
json_string = dumps(response, pretty=True)
print(json_string[:300])

# dumpd: serialize to a plain Python dict
dict_form = dumpd(response)
print(dict_form["kwargs"]["content"])
```

---

## Printing `finish_reason` — Why Did the Model Stop?

```python
finish_reason = response.response_metadata.get("finish_reason")
print("finish_reason:", finish_reason)

# Possible values:
# "stop"           — model finished naturally
# "length"         — hit max_tokens limit
# "tool_calls"     — model is calling a tool (not done yet)
# "content_filter" — output was filtered
```

---

## Using `dir()` to Discover All Attributes at Runtime

```python
# Print all non-dunder attributes and methods on an AIMessage
attrs = [a for a in dir(response) if not a.startswith("__")]
for a in attrs:
    print(a)
```

Key items you will see:
```
additional_kwargs
content
content_blocks
get_lc_namespace
id
invalid_tool_calls
is_lc_serializable
lc_attributes
lc_id
lc_secrets
model_dump
model_json_schema
name
pretty_print
pretty_repr
response_metadata
to_json
to_json_not_implemented
tool_calls
type
usage_metadata
```

---

## Accessing `text` Property (Shortcut)

```python
# .text is a shortcut that returns the string content
# It is a TextAccessor object that behaves like a string
print(response.text)

# Equivalent to:
print(response.content)
```

---

## Checking Token Counts — Quick Helpers

```python
# Quick function to print a token summary after any invoke call
def token_summary(response):
    um = response.usage_metadata
    if um:
        print(f"Input tokens  : {um.get('input_tokens', 'N/A')}")
        print(f"Output tokens : {um.get('output_tokens', 'N/A')}")
        print(f"Total tokens  : {um.get('total_tokens', 'N/A')}")
        
        # Cache info (Anthropic)
        details = um.get("input_token_details", {})
        if details.get("cache_read"):
            print(f"Cache read    : {details['cache_read']}")
        if details.get("cache_creation"):
            print(f"Cache created : {details['cache_creation']}")
        
        # Reasoning tokens (OpenAI o-series)
        out_details = um.get("output_token_details", {})
        if out_details.get("reasoning"):
            print(f"Reasoning tokens : {out_details['reasoning']}")
    else:
        print("No usage metadata returned by this provider.")

token_summary(response)
```

---

## Extracting Model Name from the Response

Some providers include the model name with version in `response_metadata`.  
This is useful when using configurable models to know what was actually used.

```python
meta = response.response_metadata

# OpenAI style
model_used = meta.get("model_name")

# Anthropic style
if not model_used:
    model_used = meta.get("model")

# Groq style
if not model_used:
    model_used = meta.get("model_name")

print("Model actually used:", model_used)
```

---

## Complete Field Reference Table

| Field | Access | Type | Notes |
|---|---|---|---|
| Response text | `response.content` | str | Always present |
| Short text access | `response.text` | str | Same as `.content` for text responses |
| Message type | `response.type` | str | Always `"ai"` |
| Message ID | `response.id` | str or None | Assigned by provider |
| Message name | `response.name` | str or None | Usually None |
| Standardized tokens | `response.usage_metadata` | dict or None | Consistent across all providers |
| Raw provider data | `response.response_metadata` | dict | Contents vary by provider |
| Extra provider fields | `response.additional_kwargs` | dict | Legacy, usually empty |
| Tool requests | `response.tool_calls` | list | Empty if no tools |
| Bad tool calls | `response.invalid_tool_calls` | list | Calls that failed to parse |
| Typed blocks | `response.content_blocks` | list | Text, image, reasoning, tool_use |
| Finish reason | `response.response_metadata["finish_reason"]` | str | stop / length / tool_calls |
| Model name | `response.response_metadata["model_name"]` | str | Provider specific key |
| Pretty string | `response.pretty_repr()` | str | Human readable format |
| Dict form | `response.model_dump()` | dict | Standard Python dict |
| JSON string | `response.to_json()` | dict | LangChain serialization format |
| Serialization check | `response.is_lc_serializable()` | bool | Always True for AIMessage |

---

## What Comes Next

- `stream()` — receiving these same fields but in chunks (`AIMessageChunk`)
- `batch()` — getting a list of `AIMessage` objects, one per prompt
- `bind_tools()` — populating `tool_calls` in the response
- `with_structured_output()` — bypassing `.content` and getting parsed objects directly