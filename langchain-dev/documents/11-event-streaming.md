# **Event Streaming**

## **1. What Event Streaming Is**

- LangChain agents are built on LangGraph, so they inherit the same low-level Pregel streaming engine.
- On top of that, LangChain adds **agent-focused projections** — typed views for messages, tool calls, state, and custom updates.
- Recommended API for most app/frontend use cases: `stream_events(..., version="v3")`.
- Returns a **run object** with multiple independent typed projections instead of raw stream-mode tuples.
- Marked **experimental** — API may still change before becoming the default streaming interface.

### **Setup used across every example below**

```python
# **setup.py**
import os
from dotenv import load_dotenv

load_dotenv()  # loads NEBIUS_API_KEY / NVIDIA_API_KEY from .env

from langchain_nebius import ChatNebius
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# **Primary model used in most examples**
nebius_model = ChatNebius(
    model="meta-llama/Llama-3.3-70B-Instruct",
    api_key=os.getenv("NEBIUS_API_KEY"),
    temperature=0.3,
)

# **Used specifically for reasoning-content examples (Nemotron exposes reasoning tokens)**
nvidia_reasoning_model = ChatNVIDIA(
    model="nvidia/llama-3.1-nemotron-70b-instruct",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.3,
)
```

`.env` file:

```
NEBIUS_API_KEY=your_nebius_key_here
NVIDIA_API_KEY=your_nvidia_key_here
```

## **2. Agent Messages (`stream.messages`)**

- Use when you want model output **as it's generated**, call by call.
- Each item is a `ChatModelStream` tied to one LLM call, exposing `.text`, `.reasoning`, `.tool_calls`, `.output`.
- `message.node` tells you which graph node produced the message.
- Usage metadata: `message.output.usage_metadata`.

**Example 1 — basic token-by-token streaming**

```python
from langchain.agents import create_agent
from setup import nebius_model

def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It's always sunny in {city}!"

agent = create_agent(model=nebius_model, tools=[get_weather])

stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "What's the weather in Pune?"}]},
    version="v3",
)

for message in stream.messages:
    for delta in message.text:
        print(delta, end="", flush=True)

print()
final_state = stream.output
print(final_state["messages"][-1].content)
```

**Example 2 — printing node name + usage metadata**

```python
from langchain.agents import create_agent
from setup import nebius_model

agent = create_agent(model=nebius_model, tools=[])

stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "Explain vector databases in 2 lines."}]},
    version="v3",
)

for message in stream.messages:
    print(f"[node={message.node}] ", end="")
    for delta in message.text:
        print(delta, end="", flush=True)
    print()

    full_message = message.output
    usage = full_message.usage_metadata
    if usage:
        print("usage:", usage)
```

## **3. Reasoning Content (`message.reasoning`)**

- Same mechanics as `.text`, but for reasoning tokens.
- Only populated for models that emit reasoning content blocks (e.g. Nemotron-style reasoning models).

**Example 1 — separate reasoning vs answer streams**

```python
from langchain.agents import create_agent
from setup import nvidia_reasoning_model

agent = create_agent(model=nvidia_reasoning_model, tools=[])

stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "Is 1099 a prime number? Think it through."}]},
    version="v3",
)

for message in stream.messages:
    for delta in message.reasoning:
        print(f"[thinking] {delta}", end="", flush=True)
    for delta in message.text:
        print(delta, end="", flush=True)
```

**Example 2 — capturing reasoning into a variable instead of printing live**

```python
from langchain.agents import create_agent
from setup import nvidia_reasoning_model

agent = create_agent(model=nvidia_reasoning_model, tools=[])

stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "Should I use FAISS or Qdrant for 5M vectors?"}]},
    version="v3",
)

reasoning_log = []
answer_log = []

for message in stream.messages:
    for delta in message.reasoning:
        reasoning_log.append(delta)
    for delta in message.text:
        answer_log.append(delta)

print("REASONING:\n", "".join(reasoning_log))
print("\nANSWER:\n", "".join(answer_log))
```

## **4. Tool Calls — Two Separate Projections**

**A. `message.tool_calls`** — what the model *decided to call* (LLM-output side, streamed as it's generated).
**B. `stream.tool_calls`** — what *actually happened* when the tool ran (execution side).

**Example 1 — `message.tool_calls`: watching argument chunks stream in**

```python
from langchain.agents import create_agent
from setup import nebius_model

def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a ticker symbol."""
    return f"{ticker} is trading at $412.50"

agent = create_agent(model=nebius_model, tools=[get_stock_price])

stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "What's the price of NVDA?"}]},
    version="v3",
)

for message in stream.messages:
    for chunk in message.tool_calls:
        print("tool call chunk:", chunk)

    finalized = message.tool_calls.get()
    if finalized:
        print("finalized tool calls:", finalized)
```

**Example 2 — `stream.tool_calls`: watching execution lifecycle + output + errors**

```python
from langchain.agents import create_agent
from setup import nebius_model

def divide(a: float, b: float) -> float:
    """Divide a by b."""
    return a / b

agent = create_agent(model=nebius_model, tools=[divide])

stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "What is 100 divided by 0?"}]},
    version="v3",
)

for call in stream.tool_calls:
    print(f"{call.tool_name}({call.input})")
    for delta in call.output_deltas:
        print(delta, end="", flush=True)
    print("\noutput:", call.output, "| error:", call.error)
```

## **5. Streaming Sub-Agents (`stream.subagents`)**

- Triggers when a `create_agent` call invokes another **named** `create_agent`, usually wrapped as a tool.
- `.name` = the name assigned at creation. `.cause` = the tool call that dispatched it.
- Only named agents show up here — unnamed ones only appear under `.subgraphs`.

**Example — supervisor delegating to a named weather sub-agent**

```python
from langchain.agents import create_agent
from setup import nebius_model

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

weather_agent = create_agent(
    model=nebius_model,
    tools=[get_weather],
    name="weather_agent",
)

def call_weather(query: str) -> str:
    """Query the weather agent."""
    result = weather_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content

supervisor = create_agent(
    model=nebius_model,
    tools=[call_weather],
    name="supervisor",
)

stream = supervisor.stream_events(
    {"messages": [{"role": "user", "content": "What's the weather in Mumbai?"}]},
    version="v3",
)

for subagent in stream.subagents:
    print(f"{subagent.name} (cause={subagent.cause}): ", end="")
    for message in subagent.messages:
        for token in message.text:
            print(token, end="", flush=True)
    print()
```

## **6. State and Final Output**

**Example 1 — step-by-step state snapshots**

```python
from langchain.agents import create_agent
from setup import nebius_model

agent = create_agent(model=nebius_model, tools=[])

stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "List 3 use cases for RAG."}]},
    version="v3",
)

for snapshot in stream.values:
    print("STATE SNAPSHOT:", snapshot)
```

**Example 2 — only the final output**

```python
from langchain.agents import create_agent
from setup import nebius_model

agent = create_agent(model=nebius_model, tools=[])

stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "Summarize LangGraph in one sentence."}]},
    version="v3",
)

# **drain the stream without inspecting intermediate steps**
for _ in stream.messages:
    pass

final_state = stream.output
print(final_state["messages"][-1].content)
```

## **7. Consuming Multiple Projections at Once**

**Example 1 — async, concurrent projections with `asyncio.gather`**

```python
import asyncio
from langchain.agents import create_agent
from setup import nebius_model

def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"It's always sunny in {city}!"

agent = create_agent(model=nebius_model, tools=[get_weather])

async def main():
    stream = await agent.astream_events(
        {"messages": [{"role": "user", "content": "Weather in Delhi, then explain monsoons briefly."}]},
        version="v3",
    )

    async def consume_messages():
        async for message in stream.messages:
            async for delta in message.text:
                print(delta, end="", flush=True)

    async def consume_tool_calls():
        async for call in stream.tool_calls:
            print(f"\n[tool] {call.tool_name}({call.input})")

    await asyncio.gather(consume_messages(), consume_tool_calls())

asyncio.run(main())
```

**Example 2 — sync, single loop with `stream.interleave`**

```python
from langchain.agents import create_agent
from setup import nebius_model

def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"It's always sunny in {city}!"

agent = create_agent(model=nebius_model, tools=[get_weather])

stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "Weather in Pune?"}]},
    version="v3",
)

for name, item in stream.interleave("messages", "tool_calls", "values"):
    if name == "messages":
        for delta in item.text:
            print(delta, end="", flush=True)
    elif name == "tool_calls":
        print(f"\n[tool] {item.tool_name}({item.input})")
    elif name == "values":
        print("\n[state]", item)
```

**Example 3 — raw low-level iteration (escape hatch)**

```python
from langchain.agents import create_agent
from setup import nebius_model

agent = create_agent(model=nebius_model, tools=[])

stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "Hi"}]},
    version="v3",
)

for event in stream:
    print(event["method"], event["params"]["namespace"], event["params"]["data"])
```

## **8. Custom Updates / Custom Transformers**

**Example 1 — direct transformer passed at call time**

```python
from langchain.agents import create_agent
from langgraph.pregel.stream import StreamTransformer
from setup import nebius_model

class ToolActivityTransformer(StreamTransformer):
    """Emits a custom 'tool_activity' channel whenever a tool starts."""
    key = "tool_activity"

    def transform(self, event):
        if event.get("event") == "on_tool_start":
            return f"tool started: {event['name']}"
        return None

def search_docs(query: str) -> str:
    """Search internal docs."""
    return f"3 results found for '{query}'"

agent = create_agent(model=nebius_model, tools=[search_docs])

stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "Search docs for 'retrieval'."}]},
    version="v3",
    transformers=[ToolActivityTransformer],
)

for activity in stream.extensions["tool_activity"]:
    print("ACTIVITY:", activity)
```

**Example 2 — registering the transformer via middleware instead**

```python
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from setup import nebius_model

class ToolActivityMiddleware(AgentMiddleware):
    transformers = (ToolActivityTransformer,)  # from example 1

def search_docs(query: str) -> str:
    """Search internal docs."""
    return f"3 results found for '{query}'"

agent = create_agent(
    model=nebius_model,
    tools=[search_docs],
    middleware=[ToolActivityMiddleware()],
)

stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "Search docs for 'embeddings'."}]},
    version="v3",
)

for activity in stream.extensions["tool_activity"]:
    print("ACTIVITY:", activity)
```

## **9. PII Middleware — Redacting Streamed Wire Output**

- Without `apply_to_output=True`, PII redaction only happens on final state (`after_model` hook) — live streamed tokens could leak raw PII before redaction applies.
- With it, a registered transformer scrubs PII from text deltas, tool-call args, tool outputs, and state snapshots **before** they leave the run.

**Example — redacting email addresses live while streaming**

```python
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from setup import nebius_model

agent = create_agent(
    model=nebius_model,
    tools=[],
    middleware=[
        PIIMiddleware("email", strategy="redact", apply_to_output=True),
    ],
)

stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "My email is akshay@example.com, confirm it back to me."}]},
    version="v3",
)

for message in stream.messages:
    for delta in message.text:
        print(delta, end="", flush=True)  # email should appear redacted, not raw
```

## **10. Practical Decision Guide**

- Live token-by-token answer text → `stream.messages` → `.text`
- Model's reasoning/thinking → `stream.messages` → `.reasoning`
- What tool the model wants to call + arguments → `message.tool_calls`
- What actually happened when a tool ran (output/errors) → `stream.tool_calls`
- Step-by-step state snapshots → `stream.values`
- Final result only → `stream.output`
- Multi-agent setup, per-agent labeled output → `stream.subagents`
- Full graph-level visibility including unnamed subgraphs → `stream.subgraphs`
- Anything not covered by a built-in projection → custom transformer → `stream.extensions`

## **11. Things to Verify Before Relying on This**

- `version="v3"` is explicitly experimental in LangChain's own docs — pin your `langchain` version.
- Middleware-based transformer registration requires `langchain>=1.3.2`.
- `stream.subagents` only surfaces **named** `create_agent` runs — unnamed inner agents only show up under `.subgraphs`.
- `apply_to_output=True` on `PIIMiddleware` must be explicitly set — it is not the default behavior.