# **LangChain Context Engineering — Notes**

Source: LangChain v1 docs (`langchain/context-engineering`), cross-checked against `langchain/middleware/custom` and current GitHub issue tracker as of Sep 2026. This topic is a framework for organizing everything covered in the middleware and runtime notes — read those first for hook mechanics; this is about *what to put where*.

---

## **1. Why agents fail, and what context engineering means**

1. When an agent misbehaves, it's rarely that the model is incapable — it's that the model wasn't given the right information/tools in the right format for the task at hand.
2. Context engineering is the practice of controlling exactly what the LLM sees (and what tools/hooks can read and write) at each point in the agent loop, rather than throwing everything at the model and hoping.
3. The agent loop is two repeating steps: model call (prompt + tools in, response or tool-request out) → tool execution (run requested tools, return results) — repeats until the model stops requesting tools.
4. Three things to control: **model context** (what goes into each model call), **tool context** (what tools read/write), **life-cycle context** (what happens between steps — summarization, guardrails, logging).
5. Middleware (covered separately) is the mechanical means to all three ends — this doc is really "how to think about middleware usage," not a new API.

---

## **2. Transient vs persistent, and the three data sources**

1. **Transient context** — what the LLM sees for one specific call. Changing it (via `request.override(...)` inside `wrap_model_call`) does not change what's saved in state; the next call starts fresh unless you change it again.
2. **Persistent context** — what gets written into state (or the store) and therefore survives into every future turn. Changing this means returning a dict from a node-style hook, or a `Command` from a wrap-style hook.
3. Three data sources, each with different scope:
   - **Runtime context** — static per-invocation configuration (user id, API keys, db connections, permissions). Conversation-scoped, read-only during the run.
   - **State** — short-term memory: messages, uploaded files, auth flags, tool results. Conversation-scoped, mutable during the run.
   - **Store** — long-term memory: preferences, extracted insights, historical data. Persists **across** conversations/threads.
4. Model context, tool context, and life-cycle context can each draw on any of these three sources — the table in this doc is really a 3x3 matrix (context type × data source), and most of the examples below are one cell of that matrix.

---

## **3. Model context — system prompt**

1. The system prompt is the highest-leverage lever for reliability: different users/stages/roles genuinely need different instructions, not one static blob.
2. `@dynamic_prompt` is the dedicated decorator for this — it takes a `ModelRequest` and returns the prompt string, run fresh before every model call (transient, not persisted).
3. Drive it from whichever data source has the relevant signal: `request.messages` / `request.state` for conversation-derived facts (e.g. message count), `request.runtime.store` for durable per-user preferences, `request.runtime.context` for static per-call config like role or environment.

```python
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langgraph.store.memory import InMemoryStore
from langchain_nebius import ChatNebius

load_dotenv()


@dataclass
class Context:
    user_id: str
    user_role: str


@dynamic_prompt
def adaptive_prompt(request: ModelRequest) -> str:
    base = "You are a helpful assistant."
    if len(request.messages) > 10:
        base += " This is a long conversation - be concise."
    store = request.runtime.store
    if store is not None:
        prefs = store.get(("preferences",), request.runtime.context.user_id)
        if prefs:
            base += f" User prefers {prefs.value.get('communication_style', 'balanced')} responses."
    if request.runtime.context.user_role == "admin":
        base += " User has admin access."
    return base


agent = create_agent(
    model=ChatNebius(model="Qwen/Qwen3-235B-A22B-Instruct-2507"),
    tools=[],
    middleware=[adaptive_prompt],
    context_schema=Context,
    store=InMemoryStore(),
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Summarize our conversation so far."}]},
    context=Context(user_id="u1", user_role="admin"),
)
print(result["messages"][-1].content)
```

---

## **4. Model context — messages**

1. Injecting extra content into the message list for a single call (via `wrap_model_call` + `request.override(messages=...)`) is transient by default — it changes what the model sees this turn without touching saved state.
2. Typical injections: file metadata the user uploaded this session (from state), a writing-style guide pulled from long-term memory (from store), jurisdiction-specific compliance rules (from runtime context).
3. Append injected context near the **end** of the message list, not the start — models weight recent context more heavily; several of the built-in examples explicitly append rather than prepend for this reason.
4. To make a message-list change *persistent* instead of transient, don't just override — return an `ExtendedModelResponse` with a `Command(update={"messages": [...]})` from `wrap_model_call`, or use `before_model`/`after_model`/`wrap_tool_call` to modify what gets saved to state.

```python
from typing import Callable
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain_nebius import ChatNebius

load_dotenv()


@wrap_model_call
def inject_uploaded_files(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    uploaded = request.state.get("uploaded_files", [])
    if not uploaded:
        return handler(request)
    lines = [f"- {f['name']} ({f['type']}): {f['summary']}" for f in uploaded]
    note = "Files available this conversation:\n" + "\n".join(lines)
    messages = [*request.messages, {"role": "user", "content": note}]
    return handler(request.override(messages=messages))


agent = create_agent(model=ChatNebius(model="Qwen/Qwen3-235B-A22B-Instruct-2507"), tools=[], middleware=[inject_uploaded_files])
result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "What files do I have?"}],
        "uploaded_files": [{"name": "report.pdf", "type": "pdf", "summary": "Q3 sales figures"}],
    }
)
print(result["messages"][-1].content)
```

---

## **5. Model context — tools**

1. Too many tools overloads context and increases wrong-tool-choice errors; too few limits what the agent can actually do — tool selection is a real reliability lever, not just housekeeping.
2. Every tool needs a clear name, docstring, and per-argument descriptions — `@tool(parse_docstring=True)` pulls argument docs straight from the function docstring's `Args:` section, so the docstring quality directly affects how well the model uses the tool.
3. Dynamic tool filtering follows the same `wrap_model_call` + `request.override(tools=...)` pattern as messages/model/format — filter `request.tools` down based on state (auth status, conversation stage), store (per-user feature flags), or runtime context (role-based permissions).
4. All candidate tools must still be registered on `create_agent(tools=[...])` up front — filtering narrows what's offered per call, it doesn't register new tools at runtime (that's a separate MCP-style dynamic registration pattern).

```python
from typing import Callable
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain_nebius import ChatNebius
from langchain.tools import tool

load_dotenv()


@tool(parse_docstring=True)
def public_search(query: str) -> str:
    """Search public documentation.

    Use this for general questions that don't require authentication.

    Args:
        query: The search query
    """
    return f"public results for {query}"


@tool(parse_docstring=True)
def delete_account(user_id: str) -> str:
    """Delete a user account permanently.

    Args:
        user_id: The account to delete
    """
    return f"deleted {user_id}"


@wrap_model_call
def gate_by_auth(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    if not request.state.get("authenticated", False):
        tools = [t for t in request.tools if t.name == "public_search"]
        request = request.override(tools=tools)
    return handler(request)


agent = create_agent(
    model=ChatNebius(model="Qwen/Qwen3-235B-A22B-Instruct-2507"),
    tools=[public_search, delete_account],
    middleware=[gate_by_auth],
)
result = agent.invoke({"messages": [{"role": "user", "content": "Search the docs for pricing."}], "authenticated": False})
print(result["messages"][-1].content)
```

---

## **6. Model context — model selection**

1. Same `request.override(model=...)` mechanism as tools/messages — pick which model instance actually handles this call, per call, based on state (conversation length), store (user's saved model preference), or runtime context (cost tier, environment).
2. Initialize candidate models once, outside the hook — re-constructing a `ChatModel` on every call is wasted work and can break connection pooling/caching.
3. This is the standard way to do cost-based routing (cheap model for short/simple turns, stronger model once context/complexity grows) without writing a separate router agent.

```python
from typing import Callable
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain_nebius import ChatNebius

load_dotenv()

efficient = ChatNebius(model="Qwen/Qwen3-8B")
standard = ChatNebius(model="Qwen/Qwen3-235B-A22B-Instruct-2507")


@wrap_model_call
def route_by_length(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    model = standard if len(request.messages) > 10 else efficient
    return handler(request.override(model=model))


agent = create_agent(model=efficient, tools=[], middleware=[route_by_length])
result = agent.invoke({"messages": [{"role": "user", "content": "What's 2 + 2?"}]})
print(result["messages"][-1].content)
```

---

## **7. Model context — response format**

1. Passing a schema as `response_format` guarantees the agent's final message conforms to it — the agent still runs its normal model/tool loop, then coerces the last response into the schema once it's done calling tools.
2. Field names, types, and `Field(description=...)` text all steer the model's extraction, the same way tool docstrings steer tool use — treat schema authoring as a prompting task, not just a typing exercise.
3. Response format can be selected dynamically the same way as model/tools: `request.override(response_format=SchemaClass)` inside `wrap_model_call`, driven by conversation stage, stored user preference, or role/environment from runtime context.

```python
from typing import Callable
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain_nebius import ChatNebius

load_dotenv()


class BriefAnswer(BaseModel):
    answer: str = Field(description="A brief answer")


class DetailedAnswer(BaseModel):
    answer: str = Field(description="A detailed answer")
    reasoning: str = Field(description="Explanation of the reasoning")


@wrap_model_call
def pick_format(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    fmt = DetailedAnswer if len(request.messages) >= 3 else BriefAnswer
    return handler(request.override(response_format=fmt))


agent = create_agent(model=ChatNebius(model="Qwen/Qwen3-235B-A22B-Instruct-2507"), tools=[], middleware=[pick_format])
result = agent.invoke({"messages": [{"role": "user", "content": "Why is the sky blue?"}]})
print(result["structured_response"])
```

---

## **8. Tool context — reads**

1. Real tools almost always need more than the model's literal arguments: user id for a DB query, an API key, current session flags — this comes from `ToolRuntime`, not from asking the model to pass it as an argument.
2. `runtime.state` — read current session/conversation state (e.g. an `authenticated` flag set earlier in the run).
3. `runtime.context` — read static per-invocation config (user id, API keys, db connection string) set once at `.invoke(context=...)` time.
4. `runtime.store` — read long-term memory keyed by namespace + id, independent of which thread/conversation is currently running.
5. Never have the model itself hold or pass secrets like API keys as tool arguments — pull them from `runtime.context` inside the tool instead, where the model never sees them.

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
    api_key: str


@tool
def check_authentication(runtime: ToolRuntime) -> str:
    """Check if the current session is authenticated."""
    return "authenticated" if runtime.state.get("authenticated", False) else "not authenticated"


@tool
def fetch_account_summary(runtime: ToolRuntime[Context]) -> str:
    """Fetch a summary of the user's account using the configured API key."""
    return f"account summary for {runtime.context.user_id} (key ends ...{runtime.context.api_key[-4:]})"


agent = create_agent(
    model=ChatNebius(model="Qwen/Qwen3-235B-A22B-Instruct-2507"),
    tools=[check_authentication, fetch_account_summary],
    context_schema=Context,
    store=InMemoryStore(),
)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Am I logged in, and what's my account summary?"}], "authenticated": True},
    context=Context(user_id="user_123", api_key="sk-abcd1234"),
)
print(result["messages"][-1].content)
```

---

## **9. Tool context — writes**

1. Tools write to **state** by returning a `Command(update={...})` instead of a plain string — this is how a tool (e.g. an authentication check) can flip a flag that later hooks or tools read.
2. Tools write to the **store** with `store.put(namespace, key, value)` directly — no `Command` needed, since the store isn't part of the graph's state/reducer system; it's an external persistence layer.
3. Read-modify-write is the standard pattern for store updates: `get` the existing value, merge the new field in, `put` the merged dict back — don't blindly overwrite, or you'll clobber other saved preferences under the same key.

```python
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langgraph.store.memory import InMemoryStore
from langchain_nebius import ChatNebius

load_dotenv()


@dataclass
class Context:
    user_id: str


@tool
def authenticate(password: str, runtime: ToolRuntime) -> Command:
    """Authenticate the current session."""
    ok = password == "correct-password"
    return Command(update={"authenticated": ok})


@tool
def save_preference(key: str, value: str, runtime: ToolRuntime[Context]) -> str:
    """Persist a user preference to long-term memory."""
    existing = runtime.store.get(("preferences",), runtime.context.user_id)
    prefs = dict(existing.value) if existing else {}
    prefs[key] = value
    runtime.store.put(("preferences",), runtime.context.user_id, prefs)
    return f"saved {key}={value}"


agent = create_agent(
    model=ChatNebius(model="Qwen/Qwen3-235B-A22B-Instruct-2507"),
    tools=[authenticate, save_preference],
    context_schema=Context,
    store=InMemoryStore(),
)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Log in with password correct-password, then save that I prefer email replies."}]},
    context=Context(user_id="user_123"),
)
print(result["messages"][-1].content)
```

---

## **10. Life-cycle context — summarization as the persistent-vs-transient example**

1. Life-cycle context is what happens *between* model/tool steps: summarization, guardrails, logging — implemented with the same middleware hooks, just conceptually separate from "what goes into this one call."
2. `SummarizationMiddleware` is the canonical example of a **persistent** update: unlike transiently trimming messages for one call, it permanently rewrites state — replaces older messages with a summary that every future turn will see, not just the current one.
3. Contrast directly with Section 4 (message injection): that pattern changes what's sent to the model without touching state; summarization changes state itself, so the change is permanent.

```python
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_nebius import ChatNebius

load_dotenv()

model = ChatNebius(model="Qwen/Qwen3-235B-A22B-Instruct-2507")

agent = create_agent(
    model=model,
    tools=[],
    middleware=[
        SummarizationMiddleware(model=model, trigger={"tokens": 3000}, keep=("messages", 15)),
    ],
)
result = agent.invoke({"messages": [{"role": "user", "content": "Let's talk through my project plan in detail."}]})
print(result["messages"][-1].content)
```

---

## **Decision guide**

1. Need the model to behave differently per user/role/stage but the change shouldn't outlive one call → transient, `wrap_model_call` + `request.override(...)`.
2. Need a change to survive into future turns of the same conversation → persistent, return a dict (node-style) or `ExtendedModelResponse(command=Command(...))` (wrap-style) so it lands in state.
3. Need a change to survive into a completely different conversation → the store, not state — `runtime.store.put(...)`.
4. Deciding what a tool needs beyond the model's literal arguments → `ToolRuntime`: `runtime.state` for this-session facts, `runtime.context` for static config/secrets, `runtime.store` for durable per-user data.
5. Building a tool's docstring → treat `Args:` descriptions as prompt engineering for tool selection, not just documentation for humans.
6. Choosing where to gate dangerous or expensive tools → filter `request.tools` in `wrap_model_call` based on auth/role, rather than relying on the system prompt alone to tell the model "don't use this."
7. Long conversation getting expensive → `SummarizationMiddleware` for persistent compression; if it's specifically tool-output bloat, `ContextEditingMiddleware` (covered in the middleware notes) instead.

---

## **Things to verify before relying on this**

1. There's an open GitHub issue (langchain-ai/langchain #36568, filed against `langchain-core` 1.2.25 / `langchain` 1.2.15) reporting that narrowing `response_format` via `request.override(response_format=ToolStrategy(...))` inside `wrap_model_call` doesn't actually narrow what the model sees — confirm this is fixed in your installed version before relying on dynamic response-format narrowing for anything safety-critical.
2. `request.override(...)` is the current method name; an older/alternate name `request.replace(...)` appears in at least one in-flight docs PR — if code samples elsewhere use `.replace()`, treat that as stale rather than an alternative API.
3. `ToolRuntime` exposing `.state` directly (as used in Section 8) vs. accessing state through `request.state` in `wrap_model_call` — both are shown in the source docs but weren't cross-verified against the same LangChain version; confirm both attribute names exist together in your installed release.
4. `store.put`/`store.get` shown here use `InMemoryStore()`, which has no persistence across process restarts — production long-term memory needs a durable backend (e.g. `PostgresStore`), not shown in these examples.
5. `ChatNebius` model names in the code are illustrative — verify current model availability on Nebius Token Factory before running.
6. Whether `ChatNebius`/`ChatNVIDIA` support structured `response_format` (Section 7) with the same guarantees as OpenAI/Anthropic-backed models wasn't confirmed here — test schema conformance directly against these providers before depending on it.