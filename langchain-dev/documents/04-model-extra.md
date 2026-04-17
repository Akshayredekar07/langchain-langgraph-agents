# **LangChain Models Advanced Topics**

## **Table of Contents**
1. [Model Profiles](#1-model-profiles)
2. [Multimodal](#2-multimodal)
3. [Reasoning](#3-reasoning)
4. [Local Models (Ollama)](#4-local-models-ollama)
5. [Prompt Caching](#5-prompt-caching)
6. [Server-Side Tool Use](#6-server-side-tool-use)
7. [Rate Limiting](#7-rate-limiting)
8. [Base URL & Proxy Settings](#8-base-url--proxy-settings)
9. [Log Probabilities](#9-log-probabilities)
10. [Token Usage](#10-token-usage)
11. [Invocation Config](#11-invocation-config)
12. [Configurable Models](#12-configurable-models)


## **1. Model Profiles**


Model profiles are dictionaries exposing a model's capabilities, context window size, supported modalities, tool calling, structured output, reasoning, etc. They allow you to **write adaptive code** that behaves differently depending on what the underlying model supports, without hardcoding model names.

Data is sourced from [models.dev](https://models.dev) and merged with LangChain-specific augmentations. Profiles are a **beta** feature — format may change.

**Key fields:**
- `max_input_tokens` — context window size
- `tool_calling` — supports function/tool calling
- `image_inputs` — accepts image input
- `reasoning_output` — exposes chain-of-thought
- `structured_output` — native JSON/schema output



### Example 1 — Inspect and Print a Model's Profile
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 1: Inspect Model Profile
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import json
from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenRouter(model="z-ai/glm-5.1", temperature=0.7)

# Retrieve and display the full profile
profile = model.profile
print("━━━ Model Profile ━━━")
print(json.dumps(profile, indent=2))

# Check specific capabilities before using them
if profile.get("tool_calling"):
    print("\nThis model supports tool calling")
else:
    print("\nTool calling not supported — use prompt-based extraction instead")

if profile.get("image_inputs"):
    print("This model accepts image inputs")
else:
    print("Image inputs not supported")

print(f"\nMax input tokens: {profile.get('max_input_tokens', 'unknown')}")
```



### Example 2 — Override Profile for a Custom/Proxy Model
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 2: Custom Profile for a Proxy/Router Model
# Real-world use: You route through OpenRouter but the profile data
# is missing or incorrect for a newer model.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv()

custom_profile = {
    "max_input_tokens": 128_000,
    "tool_calling": True,
    "structured_output": True,
    "image_inputs": False,
    "reasoning_output": False,
}

# Pass profile at instantiation
model = ChatOpenRouter(
    model="z-ai/glm-5.1",
    temperature=0.7,
    profile=custom_profile,
)

print("Custom profile applied:")
print(model.profile)
```



### Example 3 — Adaptive Summarizer Based on Context Window
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 3: Adaptive Summarization — uses profile to decide chunk size
# Real-world use: You have a long document pipeline and want to
# automatically adapt to whichever model is currently loaded.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenRouter(model="z-ai/glm-5.1", temperature=0.7)

def adaptive_summarize(text: str, model) -> str:
    max_tokens = model.profile.get("max_input_tokens", 4096)

    # Simple heuristic: 1 token ≈ 4 chars
    max_chars = max_tokens * 4
    if len(text) > max_chars:
        print(f"  Text too long ({len(text)} chars). Truncating to {max_chars} chars.")
        text = text[:max_chars]

    response = model.invoke([HumanMessage(content=f"Summarize this:\n\n{text}")])
    return response.content

# Demo
long_text = "LangChain is a framework for building LLM applications. " * 50
summary = adaptive_summarize(long_text, model)
print(f"Summary: {summary}")
```



## 2. Multimodal


Multimodal models can process **text + images** (and sometimes audio/video). LangChain supports three formats:
1. **Cross-provider standard format** (recommended for portability)
2. **OpenAI chat completions format**
3. **Native provider format** (e.g., Anthropic's own spec)

For image input, you typically provide a `base64`-encoded image or a URL. The response's `content_blocks` may include image data if the model generates images.

>  `z-ai/glm-5.1` via OpenRouter may or may not support image inputs. For multimodal, switch to a vision-capable model like `openai/gpt-4o` or `anthropic/claude-opus-4-6`.



### Example 1 — Image URL Input (Cross-Provider Format)
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 1: Send an image URL to a vision model
# Real-world use: Analyze product images in an e-commerce chatbot
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# Switch to a vision model for multimodal tasks
vision_model = ChatOpenRouter(model="openai/gpt-4o", temperature=0.7)

message = HumanMessage(
    content=[
        {
            "type": "image_url",
            "image_url": {
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
            },
        },
        {
            "type": "text",
            "text": "Describe this image in one sentence. What breed might this cat be?",
        },
    ]
)

response = vision_model.invoke([message])
print(f"Vision Response: {response.content}")
```



### Example 2 — Base64 Image Input (Local Image File)
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 2: Send a local image as base64
# Real-world use: Medical imaging chatbot, receipt OCR, document analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import base64
from pathlib import Path
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

vision_model = ChatOpenRouter(model="openai/gpt-4o", temperature=0.7)

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Replace with your actual image path
# image_path = "receipt.jpg"
# image_data = encode_image(image_path)

# For demo, we'll use a placeholder base64 string
image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

message = HumanMessage(
    content=[
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_data}"
            },
        },
        {
            "type": "text",
            "text": "What do you see in this image? Extract any text if present.",
        },
    ]
)

response = vision_model.invoke([message])
print(f"OCR/Vision Result: {response.content}")
```



### Example 3 — Check Content Blocks in Response
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 3: Parse multimodal response content blocks
# Real-world use: Handle responses that mix text + generated images
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

vision_model = ChatOpenRouter(model="openai/gpt-4o", temperature=0.7)

response = vision_model.invoke(
    [HumanMessage(content="Describe the Eiffel Tower briefly.")]
)

# Iterate content blocks — future-proof for image output models
print("━━━ Content Blocks ━━━")
for block in response.content_blocks:
    if block.get("type") == "text":
        print(f"[TEXT] {block['text']}")
    elif block.get("type") == "image":
        print(f"[IMAGE] mime={block.get('mime_type')} base64_len={len(block.get('base64',''))}")
    else:
        print(f"[OTHER] {block}")
```



## 3. Reasoning


Reasoning models (like `o3`, `claude-opus-4-6`, DeepSeek-R1) internally generate a **chain-of-thought** before producing the final answer. LangChain surfaces this via `content_blocks` with `type: "reasoning"`.

You can often control reasoning effort:
- **Categorical**: `"low"` / `"medium"` / `"high"`
- **Token budget**: e.g., `thinking={"type": "enabled", "budget_tokens": 5000}`

Reasoning blocks are streamed before the final answer block.



### Example 1 — Stream Reasoning + Final Answer
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 1: Stream reasoning steps separately from the answer
# Real-world use: Show "thinking..." UI while the model reasons
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv()

# Use a reasoning-capable model via OpenRouter
reasoning_model = ChatOpenRouter(model="deepseek/deepseek-r1", temperature=0.7)

question = "A train travels 60 km/h for 2.5 hours, then 80 km/h for 1.5 hours. What is the total distance?"

print("━━━ Streaming Response ━━━")
reasoning_steps = []
answer_parts = []

for chunk in reasoning_model.stream(question):
    for block in chunk.content_blocks:
        if block.get("type") == "reasoning":
            reasoning_steps.append(block.get("text", ""))
            print(f" [THINKING] {block.get('text', '')}", end="", flush=True)
        elif block.get("type") == "text":
            answer_parts.append(block.get("text", ""))
            print(block.get("text", ""), end="", flush=True)

print("\n\n━━━ Final Answer ━━━")
print("".join(answer_parts))
print(f"\nReasoning steps captured: {len(reasoning_steps)}")
```



### Example 2 — Collect Full Reasoning Without Streaming
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 2: Invoke and collect full reasoning for logging/debugging
# Real-world use: Audit trail for financial/medical decision support
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv()

reasoning_model = ChatOpenRouter(model="deepseek/deepseek-r1", temperature=0.7)

question = "Should a 35-year-old with moderate risk tolerance invest in bonds or growth ETFs? Explain the tradeoffs."

response = reasoning_model.invoke(question)

reasoning_text = ""
answer_text = ""

for block in response.content_blocks:
    if block.get("type") == "reasoning":
        reasoning_text += block.get("text", "")
    elif block.get("type") == "text":
        answer_text += block.get("text", "")

print("━━━ Chain of Thought ━━━")
print(reasoning_text[:500] + "..." if len(reasoning_text) > 500 else reasoning_text)

print("\n━━━ Final Answer ━━━")
print(answer_text)

# Save reasoning to an audit log
import json, datetime
audit_log = {
    "timestamp": datetime.datetime.now().isoformat(),
    "question": question,
    "reasoning_summary": reasoning_text[:200],
    "answer": answer_text,
}
print("\n━━━ Audit Log Entry ━━━")
print(json.dumps(audit_log, indent=2))
```



### Example 3 — Compare Reasoning vs Non-Reasoning on Math Problem
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 3: Side-by-side comparison — reasoning vs standard model
# Real-world use: Benchmark which model is better for complex queries
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv()

standard_model  = ChatOpenRouter(model="z-ai/glm-5.1", temperature=0.7)
reasoning_model = ChatOpenRouter(model="deepseek/deepseek-r1", temperature=0.7)

problem = "What is 17 × 23 + (144 / 12) - 58? Show the answer only."

print("━━━ Standard Model ━━━")
std_resp = standard_model.invoke(problem)
print(std_resp.content)

print("\n━━━ Reasoning Model ━━━")
r_resp = reasoning_model.invoke(problem)
for block in r_resp.content_blocks:
    if block.get("type") == "text":
        print(block["text"])
```



## 4. Local Models (Ollama)


Running models locally via **Ollama** is ideal when:
- Data privacy is critical (no data leaves your machine)
- You want to use fine-tuned or custom models
- You want to avoid API costs

Ollama runs as a local HTTP server (default: `http://localhost:11434`). Install from [ollama.com](https://ollama.com) and pull models with `ollama pull <model>`.

```bash
# Install a model locally
ollama pull llama3
ollama pull mistral
ollama pull phi3
```



### Example 1 — Basic Local Chat with Ollama
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 1: Basic Ollama chat
# Real-world use: Offline assistant for air-gapped environments
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# pip install langchain-ollama

from langchain_ollama import ChatOllama

# Make sure Ollama is running: `ollama serve`
model = ChatOllama(model="llama3", temperature=0.7)

response = model.invoke("Explain what a transformer neural network is in two sentences.")
print(response.content)
```



### Example 2 — Streaming from a Local Model
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 2: Stream output from local Ollama model
# Real-world use: Real-time private document Q&A
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain_ollama import ChatOllama

model = ChatOllama(model="mistral", temperature=0.7)

print("━━━ Streaming Local Response ━━━")
for chunk in model.stream("Write a Python function to reverse a string. Only code, no explanation."):
    print(chunk.content, end="", flush=True)
print()
```



### Example 3 — Multi-Turn Conversation with Local Model
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 3: Multi-turn private HR assistant (no data leaves machine)
# Real-world use: Confidential employee queries processed locally
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

model = ChatOllama(model="llama3", temperature=0.7)

history = [
    SystemMessage(content="You are a private HR assistant. Answer concisely.")
]

queries = [
    "How many vacation days do employees get per year?",
    "What's the process for requesting medical leave?",
]

for query in queries:
    history.append(HumanMessage(content=query))
    response = model.invoke(history)
    history.append(AIMessage(content=response.content))
    print(f"User: {query}")
    print(f"Assistant: {response.content}\n")
```



## 5. Prompt Caching


Prompt caching reduces latency and cost when you repeatedly send the **same prefix** (e.g., a long system prompt or document context). Two strategies:

- **Implicit caching** (OpenAI, Gemini): automatic, no code changes needed
- **Explicit caching** (Anthropic, Bedrock): you mark cache breakpoints manually

Cache hits are reflected in `response.usage_metadata`. Most providers only activate caching above a **minimum token threshold** (e.g., Anthropic requires 1024+ tokens).



### Example 1 — Implicit Caching with OpenAI
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 1: OpenAI implicit caching — same prefix, reduced cost on repeat
# Real-world use: A legal Q&A bot that always prepends a 10k-token policy doc
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Long static system prompt (would be cached automatically)
legal_context = """
You are a legal assistant. The following is the company's IP policy (v3.2):
1. All inventions created during employment are company property.
2. Employees must disclose inventions within 30 days.
3. Non-compete clauses apply for 12 months post-employment.
""" * 50  # Make it longer to trigger caching threshold

questions = [
    "Who owns an invention I create on weekends using my own computer?",
    "How long do I have to disclose a new invention?",
]

for q in questions:
    messages = [
        SystemMessage(content=legal_context),
        HumanMessage(content=q),
    ]
    resp = model.invoke(messages)
    usage = resp.usage_metadata
    print(f"Q: {q}")
    print(f"A: {resp.content}")
    print(f"Cache read tokens: {usage.get('input_token_details', {}).get('cache_read', 0)}\n")
```



### Example 2 — Explicit Anthropic Prompt Caching Middleware
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 2: Explicit cache breakpoints with Anthropic
# Real-world use: A RAG system that loads a large knowledge base once
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain_anthropic import ChatAnthropic, AnthropicPromptCachingMiddleware
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

base_model = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.7)

# Wrap with caching middleware
model = AnthropicPromptCachingMiddleware(model=base_model)

large_knowledge_base = "Python is a high-level programming language. " * 300

questions = ["What is Python?", "Is Python high-level?"]

for q in questions:
    messages = [
        SystemMessage(content=[
            {"type": "text", "text": large_knowledge_base, "cache_control": {"type": "ephemeral"}}
        ]),
        HumanMessage(content=q),
    ]
    resp = model.invoke(messages)
    print(f"Q: {q}")
    print(f"A: {resp.content}")
    print(f"Usage: {resp.usage_metadata}\n")
```



### Example 3 — Track Cache Savings Across Multiple Queries
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 3: Measure cache efficiency across a batch of queries
# Real-world use: Customer support bot — measure ROI of caching
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

faq_context = ("Our refund policy: items can be returned within 30 days. "
               "Shipping is free over $50. Support hours: 9am-5pm EST. ") * 60

customer_queries = [
    "What is your return policy?",
    "Do you offer free shipping?",
    "When is support available?",
    "Can I return after 30 days?",
    "What's the minimum for free shipping?",
]

total_input = 0
total_cached = 0

for q in customer_queries:
    resp = model.invoke([
        SystemMessage(content=faq_context),
        HumanMessage(content=q)
    ])
    meta = resp.usage_metadata
    input_t = meta.get("input_tokens", 0)
    cached_t = meta.get("input_token_details", {}).get("cache_read", 0)
    total_input += input_t
    total_cached += cached_t
    print(f"Q: {q[:40]:<40} | input: {input_t:>5} | cached: {cached_t:>5}")

cache_rate = (total_cached / total_input * 100) if total_input else 0
print(f"\nCache efficiency: {cache_rate:.1f}% ({total_cached}/{total_input} tokens served from cache)")
```



## 6. Server-Side Tool Use


Some providers run tool-calling loops **on their servers** — the model can call web search, a code interpreter, etc., and return the results in a single API response. This contrasts with **client-side tool calling**, where you receive a `tool_call`, run it yourself, and send back a `ToolMessage`.

With server-side tools, the response `content_blocks` contain:
- `server_tool_call` — the tool that was invoked
- `server_tool_result` — the result returned to the model
- `text` — the final answer (possibly with `annotations` / citations)



### Example 1 — Web Search Tool (OpenAI)
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 1: Server-side web search via OpenAI
# Real-world use: News summarization chatbot with live data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("gpt-4.1-mini", model_provider="openai")

tool = {"type": "web_search_preview"}
model_with_search = model.bind_tools([tool])

response = model_with_search.invoke("What are the top AI news stories this week?")

for block in response.content_blocks:
    if block["type"] == "server_tool_call":
        print(f" Searched: {block['args'].get('query', '')}")
    elif block["type"] == "server_tool_result":
        print(f"Search status: {block.get('status', '')}")
    elif block["type"] == "text":
        print(f"\n Answer:\n{block['text'][:500]}...")
```



### Example 2 — Extract Citations from Server Tool Response
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 2: Extract source citations from server-side search response
# Real-world use: Research assistant that provides sourced answers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("gpt-4.1-mini", model_provider="openai")
model_with_search = model.bind_tools([{"type": "web_search_preview"}])

response = model_with_search.invoke("What is the current population of India?")

answer = ""
citations = []

for block in response.content_blocks:
    if block["type"] == "text":
        answer = block.get("text", "")
        annotations = block.get("annotations", [])
        for ann in annotations:
            if ann.get("type") == "citation":
                citations.append({
                    "title": ann.get("title", ""),
                    "url": ann.get("url", ""),
                })

print(f"Answer: {answer}\n")
print("Sources:")
for i, c in enumerate(citations, 1):
    print(f"  {i}. {c['title']} — {c['url']}")
```



### Example 3 — Combine Web Search + Structured Extraction
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 3: Server-side search → extract structured data
# Real-world use: Competitive intelligence tool that fetches + structures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class CompanyInfo(BaseModel):
    name: str = Field(description="Company name")
    founded: str = Field(description="Year founded")
    ceo: str = Field(description="Current CEO")
    valuation: str = Field(description="Current valuation or market cap")

# Step 1: Fetch live info
search_model = init_chat_model("gpt-4.1-mini", model_provider="openai")
search_model = search_model.bind_tools([{"type": "web_search_preview"}])

raw_response = search_model.invoke("Find current info about OpenAI: CEO, founding year, valuation")

raw_text = " ".join(
    b.get("text", "") for b in raw_response.content_blocks if b["type"] == "text"
)

# Step 2: Structure the fetched text
extractor = init_chat_model("gpt-4.1-mini", model_provider="openai")
extractor = extractor.with_structured_output(CompanyInfo)

structured = extractor.invoke(f"Extract company info from this text:\n\n{raw_text}")
print(structured)
```



## 7. Rate Limiting


Rate limiting prevents you from hitting provider API limits. LangChain provides `InMemoryRateLimiter` which:
- Is **thread-safe** and can be shared across multiple threads
- Uses a **token bucket** algorithm: `max_bucket_size` controls burst capacity
- Operates on **request count only** (not token size)

Key parameters:
- `requests_per_second` — steady-state rate
- `check_every_n_seconds` — polling interval
- `max_bucket_size` — max burst



### Example 1 — Basic Rate Limiter (1 req/10s)
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 1: Basic rate limiter setup
# Real-world use: Free-tier API with strict RPM limits
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import time
from langchain_openrouter import ChatOpenRouter
from langchain_core.rate_limiters import InMemoryRateLimiter
from dotenv import load_dotenv

load_dotenv()

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.2,   # 1 request every 5 seconds
    check_every_n_seconds=0.1,
    max_bucket_size=3,         # Allow burst of up to 3
)

model = ChatOpenRouter(
    model="z-ai/glm-5.1",
    temperature=0.7,
    rate_limiter=rate_limiter,
)

queries = ["What is Python?", "What is JavaScript?", "What is Rust?"]

for q in queries:
    start = time.time()
    resp = model.invoke(q)
    elapsed = time.time() - start
    print(f"Q: {q:<25} | Time: {elapsed:.2f}s | A: {resp.content[:40]}...")
```



### Example 2 — Shared Rate Limiter Across Multiple Models
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 2: Share one limiter between two models
# Real-world use: A pipeline using different models but same API quota
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain_openrouter import ChatOpenRouter
from langchain_core.rate_limiters import InMemoryRateLimiter
from dotenv import load_dotenv

load_dotenv()

# Shared limiter — enforces global rate across both models
shared_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,
    check_every_n_seconds=0.05,
    max_bucket_size=5,
)

classifier = ChatOpenRouter(
    model="z-ai/glm-5.1",
    temperature=0.3,
    rate_limiter=shared_limiter,
)

summarizer = ChatOpenRouter(
    model="z-ai/glm-5.1",
    temperature=0.7,
    rate_limiter=shared_limiter,  # Same limiter object
)

texts = [
    "Apple released new chips today.",
    "The stock market dropped 2% on Monday.",
]

for text in texts:
    category = classifier.invoke(f"Classify as TECH or FINANCE: '{text}'. One word only.")
    summary = summarizer.invoke(f"Summarize in 5 words: '{text}'")
    print(f"Text: {text}")
    print(f"  Category: {category.content.strip()}")
    print(f"  Summary: {summary.content.strip()}\n")
```



### Example 3 — Rate Limiter with Async Batch Processing
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 3: Async batch with rate limiting
# Real-world use: Processing 100 support tickets without hitting limits
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import asyncio
from langchain_openrouter import ChatOpenRouter
from langchain_core.rate_limiters import InMemoryRateLimiter
from dotenv import load_dotenv

load_dotenv()

rate_limiter = InMemoryRateLimiter(
    requests_per_second=2,
    check_every_n_seconds=0.05,
    max_bucket_size=5,
)

model = ChatOpenRouter(
    model="z-ai/glm-5.1",
    temperature=0.7,
    rate_limiter=rate_limiter,
)

tickets = [
    "My order hasn't arrived after 2 weeks.",
    "I was charged twice for the same item.",
    "The product arrived damaged.",
    "I need to change my delivery address.",
    "How do I apply a discount code?",
]

async def process_ticket(ticket: str, idx: int) -> dict:
    resp = await model.ainvoke(f"Categorize and respond briefly: '{ticket}'")
    return {"id": idx, "ticket": ticket, "response": resp.content}

async def main():
    tasks = [process_ticket(t, i) for i, t in enumerate(tickets)]
    results = await asyncio.gather(*tasks)
    for r in results:
        print(f"[#{r['id']}] {r['ticket'][:40]}")
        print(f"       → {r['response'][:80]}\n")

asyncio.run(main())
```



## 8. Base URL & Proxy Settings


Many providers offer **OpenAI-compatible APIs** — you can point LangChain's OpenAI integration at any compatible endpoint. Common uses:
- **Local inference servers** (vLLM, LM Studio, llama.cpp server)
- **Corporate proxies** that forward to cloud providers
- **Model routers** (OpenRouter, LiteLLM)

> 💡 For OpenRouter specifically, prefer `ChatOpenRouter` from `langchain-openrouter`. For LiteLLM, use `ChatLiteLLM`.



### Example 1 — Point to a Local vLLM Server
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 1: Use vLLM or LM Studio running locally
# Real-world use: On-premise deployment for sensitive data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

# vLLM starts a local server compatible with OpenAI API
# Start with: vllm serve meta-llama/Llama-3.2-3B-Instruct --port 8000
model = init_chat_model(
    model="meta-llama/Llama-3.2-3B-Instruct",
    model_provider="openai",
    base_url="http://localhost:8000/v1",
    api_key="not-needed",    # vLLM doesn't require a real key
)

response = model.invoke("What is the speed of light?")
print(response.content)
```



### Example 2 — Together AI via OpenAI-Compatible Base URL
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 2: Together AI (OpenAI-compatible endpoint)
# Real-world use: Access open-source models cheaply via Together AI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model="meta-llama/Llama-3-70b-chat-hf",
    model_provider="openai",
    base_url="https://api.together.xyz/v1",
    api_key=os.environ["TOGETHER_API_KEY"],
    temperature=0.7,
)

response = model.invoke("Write a haiku about machine learning.")
print(response.content)
```



### Example 3 — HTTP Proxy for Corporate Firewall
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 3: Corporate HTTP proxy configuration
# Real-world use: Enterprise deployment behind a corporate proxy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    openai_proxy="http://corporate-proxy.company.com:8080",
)

response = model.invoke("Summarize the benefits of HTTPS in one sentence.")
print(response.content)
```



## 9. Log Probabilities


**Log probabilities** (logprobs) tell you how confident the model was about each token it generated. A logprob of 0 means 100% confident; more negative values mean less confident.

Use cases:
- **Confidence scoring**: Is the model sure about its answer?
- **Calibration research**: Compare model confidence vs. actual accuracy
- **Candidate selection**: Pick the token with the highest probability

Not all providers support logprobs. OpenAI and some compatible endpoints do.



### Example 1 — Basic Logprobs Extraction
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 1: Get token-level log probabilities
# Real-world use: Hallucination detection — low-confidence tokens flagged
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import math
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model="gpt-4o-mini",
    model_provider="openai",
).bind(logprobs=True)

response = model.invoke("What is the capital of Germany?")

logprobs_data = response.response_metadata.get("logprobs", {})
content_logprobs = logprobs_data.get("content", [])

print(f"Answer: {response.content}\n")
print("Token Confidences:")
for item in content_logprobs[:10]:     # Show first 10 tokens
    token = item.get("token", "")
    lp    = item.get("logprob", 0)
    prob  = math.exp(lp) * 100
    bar   = "█" * int(prob / 5)
    print(f"  '{token:>12}' | logprob: {lp:>7.3f} | prob: {prob:>5.1f}% {bar}")
```



### Example 2 — Confidence Score for a Yes/No Question
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 2: Confidence score for classification
# Real-world use: Medical triage — flag low-confidence diagnoses for review
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import math
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model="gpt-4o-mini",
    model_provider="openai",
).bind(logprobs=True)

symptoms = [
    "Patient has fever, cough, and shortness of breath.",
    "Patient reports mild headache after a long day of screen time.",
]

for symptom in symptoms:
    messages = [
        SystemMessage(content="Answer only YES or NO."),
        HumanMessage(content=f"Is this potentially serious? {symptom}"),
    ]
    resp = model.invoke(messages)

    first_token_lp = resp.response_metadata.get("logprobs", {}).get("content", [{}])[0].get("logprob", 0)
    confidence = math.exp(first_token_lp) * 100

    flag = " REVIEW" if confidence < 80 else "OK"
    print(f"Symptom: {symptom[:50]}...")
    print(f"  Answer: {resp.content} | Confidence: {confidence:.1f}% | {flag}\n")
```



### Example 3 — Compare Token Distributions for Ambiguous Queries
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 3: Inspect top-k token alternatives via top_logprobs
# Real-world use: See what alternative answers the model considered
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import math
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model="gpt-4o-mini",
    model_provider="openai",
).bind(logprobs=True, top_logprobs=5)  # Get top 5 alternatives per token

resp = model.invoke("Complete this: The sky is ___. One word.")
logprobs_data = resp.response_metadata.get("logprobs", {}).get("content", [])

print(f"Answer: {resp.content}\n")
print("Token alternatives at position 1:")
if logprobs_data:
    for alt in logprobs_data[0].get("top_logprobs", []):
        token = alt.get("token", "")
        prob  = math.exp(alt.get("logprob", -99)) * 100
        print(f"  '{token}': {prob:.1f}%")
```



## 10. Token Usage


Token usage tracking helps you:
- **Monitor costs** across model calls
- **Enforce budgets** in production
- **Audit** which models are consuming the most tokens

LangChain provides two approaches:
- `UsageMetadataCallbackHandler` — aggregates usage across multiple invocations
- Context manager `get_usage_metadata_callback` — scoped to a block

Usage appears in `AIMessage.usage_metadata`:
```
{"input_tokens": N, "output_tokens": N, "total_tokens": N}
```



### Example 1 — Track Usage Across Multiple Models with Callback
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 1: Aggregate token usage across two models
# Real-world use: Cost dashboard for a multi-model pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler
from dotenv import load_dotenv

load_dotenv()

model_a = init_chat_model(model="gpt-4o-mini",           model_provider="openai")
model_b = init_chat_model(model="claude-haiku-4-5-20251001", model_provider="anthropic")

callback = UsageMetadataCallbackHandler()

_ = model_a.invoke("Explain recursion in one sentence.", config={"callbacks": [callback]})
_ = model_b.invoke("Explain recursion in one sentence.", config={"callbacks": [callback]})

print("━━━ Token Usage Report ━━━")
for model_name, usage in callback.usage_metadata.items():
    print(f"\nModel: {model_name}")
    print(f"  Input tokens:  {usage.get('input_tokens', 0):>6}")
    print(f"  Output tokens: {usage.get('output_tokens', 0):>6}")
    print(f"  Total tokens:  {usage.get('total_tokens', 0):>6}")
```



### Example 2 — Per-Request Token Usage
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 2: Log token usage per request with cost estimation
# Real-world use: Per-user billing in a SaaS AI product
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenRouter(model="z-ai/glm-5.1", temperature=0.7)

# Rough pricing (adjust to actual rates)
INPUT_PRICE_PER_1K  = 0.0002
OUTPUT_PRICE_PER_1K = 0.0006

queries = [
    ("user_001", "Summarize the French Revolution in 3 bullets."),
    ("user_002", "Write a Python hello world program."),
    ("user_003", "What is machine learning?"),
]

total_cost = 0.0
for user_id, q in queries:
    resp = model.invoke(q)
    usage = resp.usage_metadata
    inp   = usage.get("input_tokens", 0)
    out   = usage.get("output_tokens", 0)
    cost  = (inp / 1000 * INPUT_PRICE_PER_1K) + (out / 1000 * OUTPUT_PRICE_PER_1K)
    total_cost += cost
    print(f"[{user_id}] in={inp} out={out} cost=${cost:.6f}")

print(f"\nTotal estimated cost: ${total_cost:.6f}")
```



### Example 3 — Token Budget Enforcement (Abort if Overage)
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 3: Enforce a per-session token budget
# Real-world use: Free-tier users capped at N tokens per day
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenRouter(model="z-ai/glm-5.1", temperature=0.7)

TOKEN_BUDGET = 500
used_tokens  = 0

questions = [
    "What is a neural network?",
    "Explain gradient descent.",
    "What is backpropagation?",
    "Define loss function.",
    "What is overfitting?",
]

for q in questions:
    if used_tokens >= TOKEN_BUDGET:
        print(f"\nBudget exhausted ({used_tokens}/{TOKEN_BUDGET} tokens). Stopping.")
        break

    resp = model.invoke(q)
    used = resp.usage_metadata.get("total_tokens", 0)
    used_tokens += used
    print(f"Q: {q}")
    print(f"A: {resp.content[:80]}...")
    print(f"   Tokens this call: {used} | Total: {used_tokens}/{TOKEN_BUDGET}\n")
```



## 11. Invocation Config


Every `invoke` / `stream` / `batch` call accepts a `config` dict (`RunnableConfig`) that controls runtime behavior. Key fields:

| Field | Purpose |
|-------|---------|
| `run_name` | Label shown in LangSmith traces |
| `tags` | Categorization tags for filtering |
| `metadata` | Arbitrary key-value data attached to the run |
| `callbacks` | List of callback handlers |
| `configurable` | Runtime-configurable model fields |
| `max_concurrency` | Concurrency limit for `.batch()` |

This is particularly powerful with **LangSmith** for observability.



### Example 1 — Tag and Label Runs for LangSmith
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 1: Run naming and tagging for observability
# Real-world use: Distinguish prod vs dev traces in LangSmith
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenRouter(model="z-ai/glm-5.1", temperature=0.7)

response = model.invoke(
    "What are the SOLID principles in software engineering?",
    config={
        "run_name": "solid_principles_explainer",
        "tags": ["education", "engineering", "production"],
        "metadata": {
            "user_id": "u_9821",
            "session_id": "sess_abc123",
            "environment": "production",
        },
    },
)

print(response.content)
```



### Example 2 — Custom Callback for Live Logging
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 2: Custom callback that logs token usage to a file
# Real-world use: Audit logging for compliance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import json, datetime
from langchain_openrouter import ChatOpenRouter
from langchain_core.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv

load_dotenv()

class AuditLogger(BaseCallbackHandler):
    def __init__(self, log_file="audit.jsonl"):
        self.log_file = log_file

    def on_llm_end(self, response, **kwargs):
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "output": response.generations[0][0].text[:100] if response.generations else "",
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"Logged to {self.log_file}")

model = ChatOpenRouter(model="z-ai/glm-5.1", temperature=0.7)

resp = model.invoke(
    "Summarize the causes of World War I in two sentences.",
    config={
        "run_name": "history_qa",
        "callbacks": [AuditLogger("history_audit.jsonl")],
    },
)
print(resp.content)
```



### Example 3 — Batch with Max Concurrency Control
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 3: Batch with concurrency limit via config
# Real-world use: Process 100 documents without overwhelming the API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenRouter(model="z-ai/glm-5.1", temperature=0.7)

inputs = [
    "What is Docker?",
    "What is Kubernetes?",
    "What is Terraform?",
    "What is Ansible?",
    "What is Jenkins?",
]

# Run max 2 at a time to stay within rate limits
results = model.batch(
    inputs,
    config={"max_concurrency": 2},
)

for q, r in zip(inputs, results):
    print(f"Q: {q}")
    print(f"A: {r.content[:80]}...\n")
```



## 12. Configurable Models


`init_chat_model` without a fixed model creates a **configurable model** — the model, temperature, provider, and max_tokens can be swapped at call time via `config={"configurable": {...}}`.

This is powerful for:
- **A/B testing** different models with the same chain
- **User-selected models** in a UI
- **Multi-tenant apps** where different customers use different models

You can also use `config_prefix` to avoid naming collisions when chaining multiple configurable models.



### Example 1 — Runtime Model Switching
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 1: Switch models at runtime with configurable model
# Real-world use: User can pick GPT-4 or Claude from a dropdown
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

# No model specified — fully configurable
configurable_model = init_chat_model(temperature=0)

question = "Write one sentence about the moon."

for model_name, provider in [
    ("gpt-4o-mini",             "openai"),
    ("claude-haiku-4-5-20251001", "anthropic"),
]:
    resp = configurable_model.invoke(
        question,
        config={"configurable": {"model": model_name, "model_provider": provider}},
    )
    print(f"[{model_name}] {resp.content}")
```



### Example 2 — Configurable Fields with Prefix (Multi-Model Chain)
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 2: Two configurable models in one chain with prefixes
# Real-world use: Drafter + Reviewer pipeline — swap each independently
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

drafter = init_chat_model(
    model="gpt-4o-mini",
    temperature=0.8,
    configurable_fields=("model", "model_provider", "temperature"),
    config_prefix="drafter",
)

reviewer = init_chat_model(
    model="gpt-4o-mini",
    temperature=0.2,
    configurable_fields=("model", "model_provider", "temperature"),
    config_prefix="reviewer",
)

topic = "the importance of sleep for productivity"

config = {
    "configurable": {
        "drafter_temperature": 0.9,     # Creative draft
        "reviewer_temperature": 0.1,    # Strict review
    }
}

draft_resp = drafter.invoke(
    f"Write a short paragraph about {topic}.",
    config=config,
)
print(f"━━━ Draft ━━━\n{draft_resp.content}\n")

review_resp = reviewer.invoke(
    f"Review and improve this paragraph for clarity:\n\n{draft_resp.content}",
    config=config,
)
print(f"━━━ Reviewed ━━━\n{review_resp.content}")
```



### Example 3 — Configurable Model with Bound Tools
```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Example 3: bind_tools on a configurable model
# Real-world use: A/B test which model is better at tool calling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class GetWeather(BaseModel):
    """Get the current weather in a given location."""
    location: str = Field(description="City and country, e.g. Mumbai, India")

class GetFlightPrice(BaseModel):
    """Get the price of a flight between two cities."""
    origin:      str = Field(description="Origin city")
    destination: str = Field(description="Destination city")

configurable_model = init_chat_model(temperature=0)
model_with_tools   = configurable_model.bind_tools([GetWeather, GetFlightPrice])

query = "What's the weather in Paris and how much is a flight from London to Paris?"

for model_id, provider in [
    ("gpt-4o-mini",   "openai"),
    ("claude-haiku-4-5-20251001", "anthropic"),
]:
    resp = model_with_tools.invoke(
        query,
        config={"configurable": {"model": model_id, "model_provider": provider}},
    )
    print(f"\n[{model_id}] Tool calls: {len(resp.tool_calls)}")
    for tc in resp.tool_calls:
        print(f"  → {tc['name']}({tc['args']})")
```



## Quick Reference

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Standard imports used throughout
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain_openrouter import ChatOpenRouter        # Default model
from langchain_openai    import ChatOpenAI             # OpenAI-native
from langchain_anthropic import ChatAnthropic          # Anthropic-native
from langchain_ollama    import ChatOllama             # Local models
from langchain.chat_models  import init_chat_model     # Provider-agnostic

from langchain_core.messages      import HumanMessage, SystemMessage, AIMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.callbacks     import UsageMetadataCallbackHandler
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenRouter(model="z-ai/glm-5.1", temperature=0.7)
```



*Generated for LangChain Advanced Topics — all examples use initialized model syntax.*