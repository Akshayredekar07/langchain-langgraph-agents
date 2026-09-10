# **LangChain Agent Runtime**

## **1. What the runtime object is**

1. `create_agent` compiles to a LangGraph graph, and LangGraph exposes a `Runtime` object carrying everything about the current execution that isn't part of the conversation itself.
2. Five things live on `Runtime`:
   - **Context** — static, per-invocation data: user id, db connections, config, feature flags.
   - **Store** — a `BaseStore` instance for long-term memory that survives across threads.
   - **Stream writer** — writes custom events into the `"custom"` stream mode.
   - **Execution info** — thread id, run id, attempt number for the current execution.
   - **Server info** — assistant id, graph id, authenticated user; populated only when running on LangGraph Server.
3. The point is dependency injection: instead of hardcoding a user id or DB connection or reading from global/module-level state, you declare a typed `context_schema` and pass the actual value in at `invoke()` time. This makes tools and middleware testable in isolation and reusable across different callers.
4. Context is static for the duration of one invocation — it's not something a tool or middleware hook mutates mid-run. State (the message list, custom state fields) is the mutable side; context is the read-only side.

---

## **2. Declaring and passing context**

1. Define a `context_schema` — typically a `@dataclass`, though any structure LangGraph accepts for context works — and pass it to `create_agent(context_schema=...)`.
2. Supply the actual instance via the `context=` argument on `.invoke()` (or `.stream()`), not inside the `{"messages": [...]}` payload — context is a separate channel from conversation state.
3. Every tool and middleware hook in that invocation can read from the same context instance; you set it once per call, not per tool.

```python
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_nebius import ChatNebius

load_dotenv()


@dataclass
class Context:
    user_name: str


agent = create_agent(
    model=ChatNebius(model="Qwen/Qwen3-235B-A22B-Instruct-2507"),
    tools=[],
    context_schema=Context,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    context=Context(user_name="Akshay"),
)
print(result["messages"][-1].content)
```

---

## **3. Runtime inside tools**

1. Add a `runtime: ToolRuntime[Context]` parameter to a `@tool`-decorated function; LangChain injects the current `Runtime` automatically — it is not something the caller passes as a normal tool argument, and the model never sees or fills it in.
2. `runtime.context` gives typed access to whatever `context_schema` instance was passed to `.invoke()`.
3. `runtime.store` gives access to long-term memory (a `BaseStore`) if one was configured on `create_agent(store=...)`; it's `None` otherwise, so guard with `if runtime.store:` before using it.
4. Store operations: `store.put(namespace, key, value_dict)` to write, `store.get(namespace, key)` to read one item, `store.search(namespace, filter=..., query=...)` for similarity/filter search. `namespace` is a tuple, conventionally `(scope, ...)` such as `("users",)` or `(user_id, "preferences")`.
5. Long-term memory persists **across threads** (different conversations), unlike agent state, which is scoped to one thread via the checkpointer.

```python
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore
from langchain_nebius import ChatNebius

load_dotenv()


@dataclass
class Context:
    user_id: str


@tool
def get_preference(key: str, runtime: ToolRuntime[Context]) -> str:
    """Fetch a stored user preference by key."""
    if runtime.store is None:
        return "no store configured"
    item = runtime.store.get(("preferences",), runtime.context.user_id)
    if item is None:
        return "no preferences saved"
    return str(item.value.get(key, "not set"))


@tool
def save_preference(key: str, value: str, runtime: ToolRuntime[Context]) -> str:
    """Save a user preference by key."""
    assert runtime.store is not None
    existing = runtime.store.get(("preferences",), runtime.context.user_id)
    data = dict(existing.value) if existing else {}
    data[key] = value
    runtime.store.put(("preferences",), runtime.context.user_id, data)
    return "saved"


store = InMemoryStore()
agent = create_agent(
    model=ChatNebius(model="Qwen/Qwen3-235B-A22B-Instruct-2507"),
    tools=[get_preference, save_preference],
    context_schema=Context,
    store=store,
)

agent.invoke(
    {"messages": [{"role": "user", "content": "Save that I prefer short answers, key 'style'."}]},
    context=Context(user_id="user-1"),
)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my style preference?"}]},
    context=Context(user_id="user-1"),
)
print(result["messages"][-1].content)
```

---

## **4. Execution info and server info inside tools**

1. `runtime.execution_info` gives thread id, run id, and attempt number for the current execution — identity/retry metadata, always available regardless of deployment target.
2. `runtime.server_info` is populated only when actually running on LangGraph Server (assistant id, graph id, authenticated user); it is `None` during local development or any non-server deployment — always null-check before use.
3. Requires `deepagents>=0.5.0` or `langgraph>=1.1.5` — a version-gated feature, not available in older LangGraph installs.
4. Useful pattern: gate a tool or hook on `server_info.user` to require an authenticated caller only when running behind LangGraph Server, while still working locally without a server.

```python
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_nebius import ChatNebius

load_dotenv()


@tool
def whoami(runtime: ToolRuntime) -> str:
    """Report execution and server identity for the current run."""
    info = runtime.execution_info
    parts = [f"thread={info.thread_id}", f"run={info.run_id}", f"attempt={info.attempt}"]
    if runtime.server_info is not None:
        parts.append(f"assistant={runtime.server_info.assistant_id}")
        if runtime.server_info.user is not None:
            parts.append(f"user={runtime.server_info.user.identity}")
    else:
        parts.append("not running on LangGraph Server")
    return ", ".join(parts)


agent = create_agent(model=ChatNebius(model="Qwen/Qwen3-235B-A22B-Instruct-2507"), tools=[whoami])
result = agent.invoke({"messages": [{"role": "user", "content": "Who am I running as?"}]})
print(result["messages"][-1].content)
```

---

## **5. Runtime inside middleware**

1. **Node-style hooks** (`before_agent`, `before_model`, `after_model`, `after_agent`) receive `Runtime` directly as their second parameter — same object type as inside tools.
2. **Wrap-style hooks** (`wrap_model_call`, `wrap_tool_call`) don't get `Runtime` as a direct parameter; instead it hangs off `request.runtime` on the `ModelRequest` / `ToolCallRequest`.
3. `@dynamic_prompt` is a convenience decorator specifically for building the system prompt from context — the function takes a `ModelRequest` and returns the prompt string; `request.runtime.context` is how it reaches the injected context.
4. Typed generics matter here: annotate as `Runtime[Context]` (node-style) so `runtime.context` is typed, not just `Runtime`.
5. Same execution-info/server-info version gate applies (`deepagents>=0.5.0` or `langgraph>=1.1.5`).

```python
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import dynamic_prompt, ModelRequest, before_model, after_model
from langgraph.runtime import Runtime
from langchain_nebius import ChatNebius

load_dotenv()


@dataclass
class Context:
    user_name: str


@dynamic_prompt
def personalized_prompt(request: ModelRequest) -> str:
    return f"You are a helpful assistant. Address the user as {request.runtime.context.user_name}."


@before_model
def log_before(state: AgentState, runtime: Runtime[Context]) -> dict | None:
    print(f"starting call for {runtime.context.user_name}")
    return None


@after_model
def log_after(state: AgentState, runtime: Runtime[Context]) -> dict | None:
    print(f"finished call for {runtime.context.user_name}")
    return None


agent = create_agent(
    model=ChatNebius(model="Qwen/Qwen3-235B-A22B-Instruct-2507"),
    tools=[],
    middleware=[personalized_prompt, log_before, log_after],
    context_schema=Context,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    context=Context(user_name="Akshay"),
)
print(result["messages"][-1].content)
```

---

## **6. Auth gating with server info in middleware**

1. A common `before_model` pattern: reject unauthenticated calls before any model spend, but only enforce it when actually deployed behind LangGraph Server — `server_info is None` locally means the check should pass through rather than block development.
2. Raising an exception from a hook halts the run; this is distinct from the `jump_to="end"` pattern, which ends gracefully with a message instead of raising.

```python
from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from langchain_nebius import ChatNebius

load_dotenv()


@before_model
def auth_gate(state: AgentState, runtime: Runtime) -> dict | None:
    server = runtime.server_info
    if server is not None and server.user is None:
        raise ValueError("Authentication required")
    return None


agent = create_agent(model=ChatNebius(model="Qwen/Qwen3-235B-A22B-Instruct-2507"), tools=[], middleware=[auth_gate])
result = agent.invoke({"messages": [{"role": "user", "content": "hello"}]})
print(result["messages"][-1].content)
```

---

## **Decision guide**

1. Need to pass per-call, read-only config (user id, DB handle, feature flags) into tools/middleware without globals → `context_schema` + `context=` at invoke time.
2. Need memory that outlives a single thread/conversation → `store` on `create_agent` + `runtime.store` inside tools, not agent state.
3. Need memory scoped to just this conversation → use agent state (messages, custom state fields), not the store.
4. Need to build a system prompt from per-user data → `@dynamic_prompt` reading `request.runtime.context`, rather than a `wrap_model_call` hand-rolling the same thing.
5. Need to know thread/run identity for logging or idempotency → `runtime.execution_info`, available everywhere.
6. Need to gate behavior on being behind a real deployed server (vs. local dev) → `runtime.server_info`, always null-checked, since it's `None` off-server.
7. Need to short-circuit before doing paid work → node-style hook returning `jump_to="end"`; need to hard-fail on a broken precondition → raise inside the hook instead.

---

## **Things to verify before relying on this**

1. `runtime.execution_info` and `runtime.server_info` require `deepagents>=0.5.0` or `langgraph>=1.1.5` — this is an explicit version gate in the docs; confirm your installed `langgraph`/`deepagents` versions before using either field, since older installs won't have them.
2. The exact shape of `server.user` (an `identity` attribute was shown) and `execution_info` (`thread_id`, `run_id`, `attempt`) comes from a single doc example — treat field names as indicative and check the live `Runtime` type/reference docs if your code depends on them precisely.
3. `context_schema` accepting a `@dataclass` is what every example shows; whether `TypedDict` or Pydantic models are also accepted for `context_schema` specifically (as opposed to `state_schema`, which does support `TypedDict`) wasn't confirmed in this source — check the API reference before assuming.
4. `store.search(namespace, filter=..., query=...)` with vector similarity requires the store to be configured with an embedding index (`IndexConfig`) — a plain `InMemoryStore()` without an index will not do semantic search even though `.get`/`.put` still work; confirm your store setup matches what `search` needs.
5. `ChatNebius` model names in the code are illustrative — verify current model availability on Nebius Token Factory before running.
6. Runtime context/store injection into tools built on `langchain-nebius`/`langchain-nvidia-ai-endpoints` chat models was not specifically confirmed in these docs — the mechanism is framework-level (LangGraph), so it should be provider-agnostic, but hasn't been directly verified against these two packages.