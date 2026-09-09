# **LangChain — Streaming Notes**

**Continuation of:** LangChain Event Streaming
**Topic:** Streaming (`stream_mode`, `stream()` / `astream()`)

## 1. **What This Page Covers**

- LangChain's original/underlying streaming system — surfaces real-time updates from agent runs via `stream_mode`.
- Official docs now recommend **event streaming** (`stream_events(version="v3")`) for new applications, since it gives separate typed iterators per projection instead of branching on `stream_mode` chunk type.
- This page (and these notes) covers the older but still-supported `stream()` / `astream()` interface — needed because a lot of real-world code and examples still use it, and event streaming is built on top of these same underlying stream modes.

## 2. **Supported Stream Modes**

| Mode | Description |
|---|---|
| `updates` | State updates after each agent step; if multiple nodes run in the same step, updates are streamed separately |
| `messages` | Tuples of `(token, metadata)` from any node where an LLM is invoked |
| `custom` | Arbitrary user-defined data streamed from inside graph nodes via a stream writer |

- Pass one or more modes as a list to `stream()` / `astream()`.

## 3. **Agent Progress (`stream_mode="updates"`)**

- Emits an event after every agent step.
- For an agent that calls one tool, expect three updates in order:
  1. LLM node → `AIMessage` with tool call request
  2. Tool node → `ToolMessage` with execution result
  3. LLM node → final `AIMessage` response
- Pass `thread_id` via `config` to checkpoint the conversation so follow-up turns resume the same history — requires the agent to have a `checkpointer` (e.g. `InMemorySaver()`); without one, `thread_id` has nothing to persist to.
- `thread_id` is independent of `stream_mode` — they're unrelated settings that happen to be passed together.

```python
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from langchain_nebius import ChatNebius

load_dotenv()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

model = ChatNebius(model="zai-org/GLM-5.2", temperature=0.3)
agent = create_agent(model=model, tools=[get_weather], checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": str(uuid7())}}

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in Pune?"}]},
    config=config,
    stream_mode="updates",
):
    for source, update in chunk.items():
        print(f"step: {source}")
        print(f"content: {update['messages'][-1].content_blocks}")
```

## 4. **LLM Tokens (`stream_mode="messages"`)**

- Streams individual LLM tokens as they're produced.
- Each yielded chunk is a `{"type": "messages", "data": (token, metadata)}` dict under `version="v2"` (older `version` defaults yield raw `(token, metadata)` tuples instead — check which version your code targets).
- `metadata["langgraph_node"]` tells you which node emitted the token (`model` vs `tools`).
- `token.content_blocks` gives normalized content — includes both `tool_call_chunk` blocks (partial tool-call JSON while it streams in) and `text` blocks (the final answer streaming in).
- Important gotcha: if an agent (built with `create_agent`) is wrapped as a node inside a parent `StateGraph`, `stream_mode="messages"` on the parent will **not** emit token chunks from the inner agent's LLM calls unless you pass `subgraphs=True`.

```python
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_nebius import ChatNebius

load_dotenv()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

model = ChatNebius(model="meta-llama/Meta-Llama-3.1-70B-Instruct")
agent = create_agent(model=model, tools=[get_weather])

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="messages",
    version="v2",
):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        print(f"node: {metadata['langgraph_node']}")
        print(f"content: {token.content_blocks}\n")
```

## 5. **Custom Updates (`stream_mode="custom"`)**

- Lets a tool emit arbitrary user-defined progress signals mid-execution (e.g. `"Fetched 10/100 records"`).
- Uses `get_stream_writer()` from `langgraph.config` inside the tool function.
- Important restriction: once you add `get_stream_writer()` inside a tool, that tool can no longer be invoked outside a LangGraph execution context (it will fail standalone).

```python
from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.config import get_stream_writer
from langchain_nebius import ChatNebius

load_dotenv()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    writer = get_stream_writer()
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"

model = ChatNebius(model="zai-org/GLM-5.2")
agent = create_agent(model=model, tools=[get_weather])

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="custom",
    version="v2",
):
    if chunk["type"] == "custom":
        print(chunk["data"])
```

## 6. **Streaming Multiple Modes at Once**

- Pass `stream_mode=["updates", "custom"]` (or any combination) as a list.
- Every chunk is a `StreamPart` dict with keys `type`, `ns`, `data` — same uniform shape regardless of which mode produced it.
- `chunk["type"]` tells you which mode fired; `chunk["data"]` is the payload for that mode.

```python
from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.config import get_stream_writer
from langchain_nebius import ChatNebius

load_dotenv()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    writer = get_stream_writer()
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"

model = ChatNebius(model="zai-org/GLM-5.2")
agent = create_agent(model=model, tools=[get_weather])

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode=["updates", "custom"],
    version="v2",
):
    print(f"stream_mode: {chunk['type']}")
    print(f"content: {chunk['data']}\n")
```

## 7. **Streaming Reasoning / Thinking Tokens**

- Applies to models that perform internal reasoning before the final answer (must have reasoning explicitly enabled on the model, e.g. Anthropic's `thinking={"type": "enabled", "budget_tokens": ...}`).
- LangChain normalizes provider-specific reasoning formats (Anthropic `thinking` blocks, OpenAI `reasoning` summaries, etc.) into one standard `"reasoning"` content block type.
- With `stream_events(version="v3")`, filter `message.reasoning` separately from `message.text` — cleaner than manually filtering content blocks by type.

```python
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

model = ChatNVIDIA(model="nvidia/llama-3.1-nemotron-70b-instruct")
agent = create_agent(model=model, tools=[get_weather])

stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    version="v3",
)
for message in stream.messages:
    for token in message.reasoning:
        print(f"[thinking] {token}", end="")
    for token in message.text:
        print(token, end="", flush=True)
```

## 8. **Streaming Tool Calls (Partial + Completed)**

- Two things you may want simultaneously:
  1. Partial JSON as tool-call arguments are generated (`tool_call_chunks`)
  2. The completed, parsed tool call once the model finishes generating it
- `stream_mode="messages"` alone only gives you the incremental chunks — completed messages live in state.
- Use `stream_mode=["messages", "updates"]` to get both: `"messages"` chunks for live tokens, `"updates"` chunks for the completed message once a node finishes.

```python
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import AIMessage, AIMessageChunk, AnyMessage, ToolMessage
from langchain_nebius import ChatNebius

load_dotenv()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

model = ChatNebius(model="zai-org/GLM-5.2")
agent = create_agent(model=model, tools=[get_weather])

def render_chunk(token: AIMessageChunk) -> None:
    if token.text:
        print(token.text, end="|")
    if token.tool_call_chunks:
        print(token.tool_call_chunks)

def render_completed(message: AnyMessage) -> None:
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"Tool calls: {message.tool_calls}")
    if isinstance(message, ToolMessage):
        print(f"Tool response: {message.content_blocks}")

input_message = {"role": "user", "content": "What is the weather in Boston?"}
for chunk in agent.stream(
    {"messages": [input_message]},
    stream_mode=["messages", "updates"],
    version="v2",
):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        if isinstance(token, AIMessageChunk):
            render_chunk(token)
    elif chunk["type"] == "updates":
        for source, update in chunk["data"].items():
            if source in ("model", "tools"):
                render_completed(update["messages"][-1])
```

## 9. **Accessing Completed Messages Not Tracked in State**

- If completed messages *are* tracked in state, `stream_mode=["messages", "updates"]` (above) already covers it.
- If they're **not** reflected in state updates (e.g. an intermediate evaluation made inside middleware), two options:
  1. Use `get_stream_writer()` inside that middleware/tool to push the completed message onto the `custom` stream manually.
  2. Aggregate message chunks yourself in the streaming loop using the `+` operator on `AIMessageChunk`, and detect completion via `token.chunk_position == "last"`.

**Example — custom stream writer inside an `after_agent` guardrail middleware**

```python
from typing import Any, Literal
from dotenv import load_dotenv
from langchain.agents.middleware import after_agent, AgentState
from langgraph.runtime import Runtime
from langchain.messages import AIMessage
from langgraph.config import get_stream_writer
from pydantic import BaseModel
from langchain_nebius import ChatNebius

load_dotenv()

class ResponseSafety(BaseModel):
    """Evaluate a response as safe or unsafe."""
    evaluation: Literal["safe", "unsafe"]

safety_model = ChatNebius(model="zai-org/GLM-5.2")

@after_agent(can_jump_to=["end"])
def safety_guardrail(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    stream_writer = get_stream_writer()
    if not state["messages"]:
        return None

    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return None

    model_with_tools = safety_model.bind_tools([ResponseSafety], tool_choice="any")
    result = model_with_tools.invoke([
        {"role": "system", "content": "Evaluate this AI response as generally safe or unsafe."},
        {"role": "user", "content": f"AI response: {last_message.text}"},
    ])
    stream_writer(result)

    tool_call = result.tool_calls[0]
    if tool_call["args"]["evaluation"] == "unsafe":
        last_message.content = "I cannot provide that response. Please rephrase your request."

    return None
```

**Example — aggregating chunks manually instead**

```python
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import AIMessageChunk
from langchain_nebius import ChatNebius

load_dotenv()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

model = ChatNebius(model="zai-org/GLM-5.2")
agent = create_agent(model=model, tools=[get_weather])

input_message = {"role": "user", "content": "What is the weather in Boston?"}
full_message = None
for chunk in agent.stream(
    {"messages": [input_message]},
    stream_mode=["messages", "updates"],
    version="v2",
):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        if isinstance(token, AIMessageChunk):
            full_message = token if full_message is None else full_message + token
            if token.chunk_position == "last":
                if full_message.tool_calls:
                    print(f"Tool calls: {full_message.tool_calls}")
                full_message = None
```

## 10. **Streaming With Human-in-the-Loop**

- Builds on the `["messages", "updates"]` pattern above, plus:
  1. Configure the agent with `HumanInTheLoopMiddleware(interrupt_on={...})` and a `checkpointer` (interrupts need conversation state to resume from).
  2. Collect `Interrupt` objects that show up under `source == "__interrupt__"` in the `"updates"` stream.
  3. Build a decision per interrupt (`"approve"`, `"edit"`, etc.) — order of decisions must match order of collected interrupts.
  4. Resume execution by streaming a `Command(resume=decisions)` through the **same** streaming loop instead of a new input message.

```python
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import AIMessage, AIMessageChunk, AnyMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Interrupt
from langchain_nebius import ChatNebius

load_dotenv()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

checkpointer = InMemorySaver()
model = ChatNebius(model="zai-org/GLM-5.2")
agent = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"get_weather": True})],
    checkpointer=checkpointer,
)

def render_chunk(token: AIMessageChunk) -> None:
    if token.text:
        print(token.text, end="|")
    if token.tool_call_chunks:
        print(token.tool_call_chunks)

def render_completed(message: AnyMessage) -> None:
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"Tool calls: {message.tool_calls}")
    if isinstance(message, ToolMessage):
        print(f"Tool response: {message.content_blocks}")

def render_interrupt(interrupt: Interrupt) -> None:
    for request in interrupt.value["action_requests"]:
        print(request["description"])

config = {"configurable": {"thread_id": "some_id"}}
input_message = {"role": "user", "content": "Look up weather in Boston and San Francisco."}
interrupts = []

for chunk in agent.stream(
    {"messages": [input_message]},
    config=config,
    stream_mode=["messages", "updates"],
    version="v2",
):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        if isinstance(token, AIMessageChunk):
            render_chunk(token)
    elif chunk["type"] == "updates":
        for source, update in chunk["data"].items():
            if source in ("model", "tools"):
                render_completed(update["messages"][-1])
            if source == "__interrupt__":
                interrupts.extend(update)
                render_interrupt(update[0])

decisions = {}
for interrupt in interrupts:
    decisions[interrupt.id] = {
        "decisions": [{"type": "approve"} for _ in interrupt.value["action_requests"]]
    }

for chunk in agent.stream(
    Command(resume=decisions),
    config=config,
    stream_mode=["messages", "updates"],
    version="v2",
):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        if isinstance(token, AIMessageChunk):
            render_chunk(token)
    elif chunk["type"] == "updates":
        for source, update in chunk["data"].items():
            if source in ("model", "tools"):
                render_completed(update["messages"][-1])
```

## 11. **Streaming From Sub-Agents**

- Needed when multiple LLMs exist in one graph and you need to know which agent produced which tokens.
- Assign `name=` to each agent at `create_agent(...)` time — LangChain attaches that name to metadata as `lc_agent_name` during `"messages"` streaming, and to every `AIMessage` that agent generates.
- Requires `subgraphs=True` on `stream()` / `astream()` to receive token chunks from the inner (wrapped) agent at all.

```python
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import AIMessage, AIMessageChunk, AnyMessage, ToolMessage
from langchain_nebius import ChatNebius

load_dotenv()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

weather_model = ChatNebius(model="zai-org/GLM-5.2")
weather_agent = create_agent(model=weather_model, tools=[get_weather], name="weather_agent")

def call_weather_agent(query: str) -> str:
    """Query the weather agent."""
    result = weather_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].text

supervisor_model = ChatNebius(model="zai-org/GLM-5.2")
agent = create_agent(model=supervisor_model, tools=[call_weather_agent], name="supervisor")

def render_chunk(token: AIMessageChunk) -> None:
    if token.text:
        print(token.text, end="|")
    if token.tool_call_chunks:
        print(token.tool_call_chunks)

def render_completed(message: AnyMessage) -> None:
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"Tool calls: {message.tool_calls}")
    if isinstance(message, ToolMessage):
        print(f"Tool response: {message.content_blocks}")

input_message = {"role": "user", "content": "What is the weather in Boston?"}
current_agent = None
for chunk in agent.stream(
    {"messages": [input_message]},
    stream_mode=["messages", "updates"],
    subgraphs=True,
    version="v2",
):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]
        if agent_name := metadata.get("lc_agent_name"):
            if agent_name != current_agent:
                print(f"[{agent_name}]:")
                current_agent = agent_name
        if isinstance(token, AIMessageChunk):
            render_chunk(token)
    elif chunk["type"] == "updates":
        for source, update in chunk["data"].items():
            if source in ("model", "tools"):
                render_completed(update["messages"][-1])
```

## 12. **Disabling Streaming**

- Set `streaming=False` at model init to disable token-level streaming for that specific model.
- Use cases: multi-agent setups where only some agents should stream, mixing streaming-capable and non-capable models, or preventing certain outputs from reaching a LangSmith-deployed client.
- If a chat model integration doesn't support the `streaming` parameter directly, use `disable_streaming=True` instead — this is available on every chat model via the base class.

```python
from dotenv import load_dotenv
from langchain_nebius import ChatNebius

load_dotenv()

model = ChatNebius(model="zai-org/GLM-5.2", streaming=False)
```

## 13. **v2 Streaming Format**

- Requires LangGraph >= 1.1.
- Pass `version="v2"` to `stream()` / `astream()` to get a unified chunk shape: every chunk is a `StreamPart` dict with `type`, `ns`, `data` — regardless of stream mode or how many modes were requested.
- Old (v1, current default) format instead yields raw `(mode, data)` tuples that must be manually unpacked — more error-prone when combining multiple modes.
- v2 also changes `invoke()`'s return type: instead of a plain state dict, you get a `GraphOutput` object with `.value` (the state) and `.interrupts` (tuple of `Interrupt` objects, empty if none) — cleanly separates state from interrupt metadata instead of mixing them.

```python
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_nebius import ChatNebius

load_dotenv()

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

model = ChatNebius(model="zai-org/GLM-5.2")
agent = create_agent(model=model, tools=[get_weather])

# v2 unified format
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode=["updates", "custom"],
    version="v2",
):
    print(chunk["type"])
    print(chunk["data"])

# v2 invoke() return shape
result = agent.invoke({"messages": [{"role": "user", "content": "Hello"}]}, version="v2")
print(result.value)
print(result.interrupts)
```

## 14. **Decision Guide — Which Mode to Use**

- Need step-by-step agent progress (tool call requested, tool result, final answer) → `stream_mode="updates"`
- Need live LLM token-by-token output → `stream_mode="messages"`
- Need to emit your own custom progress signals from inside a tool → `stream_mode="custom"`
- Need both live tokens and completed/parsed messages → `stream_mode=["messages", "updates"]`
- Need reasoning/thinking tokens separately from the answer → filter `"reasoning"` content blocks (or use `message.reasoning` under `stream_events`)
- Need to know which of several agents produced a given token → set `name=` per agent + `subgraphs=True`
- Need human approval mid-run → `HumanInTheLoopMiddleware` + collect `__interrupt__` updates + resume with `Command(resume=...)`
- Want one consistent chunk shape regardless of mode → `version="v2"`
- Building a new app from scratch with no legacy constraints → prefer `stream_events(version="v3")` (event streaming) over this whole `stream_mode` system

## 15. **Things to Verify Before Relying on This**

- `version="v2"` requires LangGraph >= 1.1 — check your installed version.
- `subgraphs=True` is mandatory to get token chunks out of an agent wrapped as a node inside another `StateGraph` — easy to forget and get silent no-token-output.
- `get_stream_writer()` makes a tool unusable outside a LangGraph execution context — don't use it in tools you also want to unit-test standalone.
- `thread_id` alone does nothing without a `checkpointer` configured on the agent.
- Docs' own recommendation is to prefer event streaming (`stream_events`) for new applications — this `stream_mode` system is the older, lower-level path, kept mainly for existing code and edge cases event streaming doesn't yet cover.