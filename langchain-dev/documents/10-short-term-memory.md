# LangGraph — Short-Term Memory (Thread-Scoped Persistence)

> LangChain 1.x / LangGraph 0.3+ — June 2026
> These notes cover **thread-scoped** (within-session) memory only.
> For cross-session / cross-thread memory → see `long-term-memory` (Stores).

---

## Table of Contents

1. [Mental Model](#1-mental-model)
2. [Checkpointer — Core Primitive](#2-checkpointer--core-primitive)
3. [Checkpointer Backends](#3-checkpointer-backends)
4. [Custom AgentState](#4-custom-agentstate)
5. [Context Window Management Patterns](#5-context-window-management-patterns)
   - 5a. Trim Messages (`@before_model`)
   - 5b. Delete Messages (`@after_model`)
   - 5c. Summarize Messages (`SummarizationMiddleware`)
6. [Accessing & Writing State](#6-accessing--writing-state)
   - 6a. Read state in a tool
   - 6b. Write state from a tool (`Command`)
   - 6c. Dynamic prompt from state (`@dynamic_prompt`)
7. [Before Model / After Model Hooks](#7-before-model--after-model-hooks)
8. [Production Patterns & Pitfalls](#8-production-patterns--pitfalls)
9. [Quick Reference Table](#9-quick-reference-table)

---

## 1. Mental Model

Short-term memory = **per-thread conversation history** held in the agent's LangGraph state and
persisted via a **checkpointer** after every node execution.

```
User message
     │
     ▼
┌────────────────────────────────────────┐
│  Graph Invocation (thread_id = "t1")  │
│                                        │
│  before_model ──► model ──► tools     │
│        │              │               │
│        └── checkpoint written ────────┼──► Checkpointer backend
└────────────────────────────────────────┘         (RAM / SQLite / Postgres)
```

Key facts:

- **Thread** = one conversation session. Same `thread_id` → same state is restored on every call.
- **Checkpoint** = snapshot of the full graph state saved after every node execution (not just at the end).
- **State key `messages`** uses the `add_messages` reducer — it appends, not overwrites.
- **`InMemorySaver`** is RAM-only. Process restart = state lost. Dev/test only.
- **`PostgresSaver`** / **`SqliteSaver`** = durable. Required for production.

> ⚠️ **Short-term ≠ Long-term.** A new `thread_id` starts with a blank state even for the same
> user. Use `InMemoryStore` / `MongoDBStore` for facts that must survive across sessions.

---

## 2. Checkpointer — Core Primitive

The checkpointer is the single required parameter that enables short-term memory. Without it,
every `agent.invoke()` call is stateless.

### Minimal implementation

```python
# ══════════════════════════════════════════════════════════════════════════════
# SHORT-TERM MEMORY — MINIMAL CHECKPOINTER SETUP
# ══════════════════════════════════════════════════════════════════════════════
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# ── Tool definition ────────────────────────────────────────────────────────────
def lookup_orders() -> str:
    """Fetch recent orders for the current user."""
    return "Order #4821: delivered. Order #4822: in transit."

# ── Agent with checkpointer ────────────────────────────────────────────────────
agent = create_agent(
    model="anthropic:claude-sonnet-4-6",   # provider:model-name string
    tools=[lookup_orders],
    checkpointer=InMemorySaver(),           # enables thread-scoped memory
)

# ── Thread config — same thread_id restores the same state ─────────────────────
thread_config = {"configurable": {"thread_id": "user-priya-session-1"}}

# Turn 1
r1 = agent.invoke(
    {"messages": [{"role": "user", "content": "Hi, I'm Priya. Check my orders."}]},
    thread_config,
)["messages"][-1].content
print(r1)  # mentions order #4821 and #4822

# Turn 2 — agent still knows the name "Priya" because state was checkpointed
r2 = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    thread_config,
)["messages"][-1].content
print(r2)  # "Your name is Priya."
```

**Parameter breakdown — `create_agent`:**

| Parameter | Type | Purpose |
|---|---|---|
| `model` | `str` | `"provider:model-name"` string, e.g. `"anthropic:claude-sonnet-4-6"` |
| `tools` | `list` | Python callables or `@tool`-decorated functions |
| `checkpointer` | `BaseCheckpointSaver` | Storage backend for state snapshots |
| `middleware` | `list` | Ordered list of middleware hooks (trim, summarize, etc.) |
| `state_schema` | `type[AgentState]` | Custom state class (optional, extends `AgentState`) |
| `context_schema` | `type` | Per-invocation read-only context (Pydantic or TypedDict) |
| `system_prompt` | `str` | Static system prompt for the model |

**Parameter breakdown — `thread_config`:**

```python
thread_config = {
    "configurable": {
        "thread_id": "user-priya-session-1",
        # Optional in multi-tenant setups:
        # "checkpoint_ns": "tenant-acme",
    }
}
```

| Field | Required | Notes |
|---|---|---|
| `thread_id` | Yes | Unique identifier per conversation. Reuse = resume. |
| `checkpoint_ns` | No | Namespace for multi-tenant isolation (e.g. per-org) |

---

## 3. Checkpointer Backends

Choose the backend based on deployment stage:

| Backend | Package | Use case | Survives restart |
|---|---|---|---|
| `InMemorySaver` | `langgraph` (built-in) | Dev / unit tests | No |
| `SqliteSaver` | `langgraph-checkpoint-sqlite` | Single-server with persistence | Yes |
| `PostgresSaver` | `langgraph-checkpoint-postgres` | Production, horizontal scale | Yes |
| `AsyncPostgresSaver` | same | Production + async FastAPI | Yes |
| `CosmosDBSaver` | `langchain-azure-cosmosdb` | Azure deployments | Yes |

### Production: PostgresSaver

```bash
pip install langgraph-checkpoint-postgres psycopg[binary]
```

```python
# ══════════════════════════════════════════════════════════════════════════════
# PRODUCTION CHECKPOINTER — PostgresSaver
# ══════════════════════════════════════════════════════════════════════════════
from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:postgres@localhost:5432/myapp?sslmode=disable"

# ── context manager handles connection lifecycle ───────────────────────────────
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()         # creates checkpoint tables on first run (idempotent)

    agent = create_agent(
        model="anthropic:claude-sonnet-4-6",
        tools=[],
        checkpointer=checkpointer,
    )

    # Recommended pattern: encode tenant + user + session into thread_id
    # This enables per-tenant isolation and per-user conversation history
    tenant_id  = "acme"
    user_id    = "priya-sharma"
    session_id = "2026-06-04-001"
    config = {
        "configurable": {
            "thread_id": f"{tenant_id}:{user_id}:{session_id}",
        }
    }

    result = agent.invoke({"messages": "What did we discuss last time?"}, config)
```

**`PostgresSaver.from_conn_string` parameters:**

| Parameter | Default | Notes |
|---|---|---|
| `conn_string` | required | Standard DSN string |
| `schema` | `"public"` | Postgres schema for checkpoint tables |

> ⚠️ **Call `.setup()` exactly once** on first use. Safe to call again (it's idempotent), but
> typically done at app startup. Without it, tables don't exist and all invocations fail.

### Case study — production multi-tenant deployment

A production pattern for multi-tenant systems uses structured `thread_id` strings of the form
`"tenant-{id}:user-{id}:session-{id}"` together with `checkpoint_ns` for per-tenant isolation.
A connection pool is placed in front of Postgres to avoid exhausting connections as workers scale.

```python
# ── Connection pool pattern for high-concurrency deployments ──────────────────
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://user:pass@host:5432/langgraph?sslmode=require"
pool = ConnectionPool(
    conninfo=DB_URI,
    max_size=10,          # tune: workers × max_size < postgres.max_connections × 0.7
)

with pool.connection() as conn:
    saver = PostgresSaver(conn)
    saver.setup()
    agent = create_agent(model="anthropic:claude-sonnet-4-6", tools=[], checkpointer=saver)
```

---

## 4. Custom AgentState

By default, `AgentState` only tracks `messages`. Extend it to carry additional structured fields
(user context, preferences, intermediate results) that persist across turns within a session.

### Minimal implementation

```python
# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM AGENT STATE — extending AgentState
# ══════════════════════════════════════════════════════════════════════════════
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver
from typing import Optional

# ── Custom state schema ────────────────────────────────────────────────────────
class SupportAgentState(AgentState):
    user_id: str                     # populated once at session start
    plan_tier: str                   # e.g. "enterprise" / "starter"
    resolved_issues: list[str]       # accumulates across turns
    escalation_flag: bool            # set by after_model middleware if needed
    last_intent: Optional[str]       # inferred intent from latest message

# ── Agent with custom state ────────────────────────────────────────────────────
agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[],
    state_schema=SupportAgentState,  # register custom schema
    checkpointer=InMemorySaver(),
)

# All custom fields can be passed at invocation time
result = agent.invoke(
    {
        "messages":         [{"role": "user", "content": "My SSO integration broke."}],
        "user_id":          "usr_8821",
        "plan_tier":        "enterprise",
        "resolved_issues":  [],
        "escalation_flag":  False,
        "last_intent":      None,
    },
    {"configurable": {"thread_id": "support-thread-001"}},
)
```

**`state_schema` rules:**

- Must subclass `AgentState` (which provides the `messages` field with `add_messages` reducer)
- Fields are type-checked at runtime via Pydantic
- New fields added after deployment: mark as `Optional[T] = None` or `NotRequired[T]` to avoid
  deserialization crashes on existing checkpoints

> ⚠️ **Never rename a state field in place.** Old checkpoint rows will fail to deserialize on the
> next invocation. Add new fields with defaults; deprecate old ones by keeping them as `Optional`.

---

## 5. Context Window Management Patterns

Long conversations hit context limits. Three strategies in order of information loss:

| Strategy | Information loss | Cost | When to use |
|---|---|---|---|
| Trim messages | High (old turns dropped) | Zero | Short-lived sessions where recency > history |
| Delete messages | High (permanent removal) | Zero | Audit/compliance: specific messages must be purged |
| Summarize | Low (semantic content kept) | Small (second LLM call) | Production agents with long sessions |

---

### 5a. Trim Messages — `@before_model` middleware

Runs **before** the model call. Modifies the message list in state so the model only sees a
window of recent messages. State in the checkpointer still contains the full history.

```python
# ══════════════════════════════════════════════════════════════════════════════
# TRIM MESSAGES — @before_model middleware
# ══════════════════════════════════════════════════════════════════════════════
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langchain_core.runnables import RunnableConfig
from typing import Any


@before_model
def keep_recent_window(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
    Trims the message list to first message + last N messages before each
    model call. The full history is still in the checkpointer — this only
    reduces what the model sees per invocation.
    """
    messages = state.get("messages", [])  # safe access

    # Nothing to trim if conversation is short
    if len(messages) <= 4:
        return None

    first_msg = messages[0]

    # Keep last 4 messages; preserve even/odd parity for human/ai turn alignment
    tail = messages[-4:] if len(messages) % 2 == 0 else messages[-5:]
    trimmed = [first_msg] + tail

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),  # wipe current view
            *trimmed,                               # replace with trimmed window
        ]
    }


agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[],
    middleware=[keep_recent_window],   # registered in middleware list
    checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "trim-demo-1"}}
agent.invoke({"messages": "Hi, I'm Arjun."}, config)
agent.invoke({"messages": "What is gradient descent?"}, config)
agent.invoke({"messages": "Explain backprop."}, config)
agent.invoke({"messages": "What's my name?"}, config)  # still returns "Arjun"
```

**Key parameters / objects:**

| Object | Role |
|---|---|
| `@before_model` | Decorator that registers the function as a pre-model hook |
| `state: AgentState` | Current graph state with full message list |
| `runtime: Runtime` | Access to context, config, tools — not used here but required in signature |
| `RemoveMessage(id=REMOVE_ALL_MESSAGES)` | Special sentinel: clears all messages from the state view |
| `return None` | Signal to LangGraph: no changes needed, proceed as-is |

**Case study — SRE debugging agent:**
In a Site Reliability Engineering scenario, agents analyzing stack traces accumulate
large histories. Middleware that trims old messages prevents the "main model" from becoming
confused by stale log output, while keeping recent context intact for the current debugging
session. The trimming happens automatically under the hood without changes to invoke calls.

---

### 5b. Delete Messages — `@after_model` middleware

Runs **after** the model call. Permanently deletes messages from the LangGraph state (and
therefore from the checkpointer). Useful for compliance (PII scrubbing, sensitive content removal).

```python
# ══════════════════════════════════════════════════════════════════════════════
# DELETE MESSAGES — @after_model middleware (permanent removal from state)
# ══════════════════════════════════════════════════════════════════════════════
from langchain.messages import RemoveMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_model


# ── Helper: safe content access (avoids AttributeError on non-text messages) ───
def _get_content(msg: BaseMessage) -> str:
    """Safely extract text content from any message type."""
    if hasattr(msg, "content"):
        content = msg.content
        # content can be str or list[dict] (multimodal)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # extract text blocks only
            return " ".join(
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
    return ""


PII_PATTERNS = ["password", "api_key", "secret", "token", "ssn"]


@after_model
def scrub_sensitive_responses(state: AgentState, runtime: Runtime) -> dict | None:
    """
    After model responds, check the latest AI message for sensitive content.
    If found, remove it from state entirely (permanent — checkpointer will
    not persist it on next write).
    """
    messages = state.get("messages", [])
    if not messages:
        return None

    last_msg = messages[-1]
    text = _get_content(last_msg).lower()

    if any(pattern in text for pattern in PII_PATTERNS):
        return {
            "messages": [RemoveMessage(id=last_msg.id)]  # remove by message ID
        }
    return None


agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[],
    middleware=[scrub_sensitive_responses],
    checkpointer=InMemorySaver(),
    system_prompt="You are a helpful assistant. Never reveal credentials.",
)
```

**`RemoveMessage` usage:**

```python
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

# Remove one specific message by ID
RemoveMessage(id=some_message.id)

# Remove ALL messages from state
RemoveMessage(id=REMOVE_ALL_MESSAGES)
```

> ⚠️ **Deletion validity constraint.** After deleting, the remaining message sequence must still
> be valid for your LLM provider. Most providers require: (1) history starts with a `user` message,
> (2) every `assistant` message with a tool call is followed by the corresponding `tool` result.
> Violating this causes provider API errors, not LangGraph errors.

---

### 5c. Summarize Messages — `SummarizationMiddleware`

Most information-preserving strategy. When a token threshold is hit, a cheap summarizer model
compresses older messages into a summary paragraph. Recent messages are kept verbatim.

```python
# ══════════════════════════════════════════════════════════════════════════════
# SUMMARIZATION MIDDLEWARE — SummarizationMiddleware
# ══════════════════════════════════════════════════════════════════════════════
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

agent = create_agent(
    model="anthropic:claude-sonnet-4-6",       # main model
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="anthropic:claude-haiku-4-5-20251001",  # cheap fast summarizer
            trigger=("tokens", 4000),           # summarize when history > 4000 tokens
            keep=("messages", 10),              # always keep last 10 messages verbatim
        )
    ],
    checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "support-session-vikram-001"}}

agent.invoke({"messages": "Hi, I'm Vikram. I'm having issues with the billing API."}, config)
agent.invoke({"messages": "The error is a 402 on POST /v1/invoices."}, config)
agent.invoke({"messages": "My account is on the Pro plan."}, config)
agent.invoke({"messages": "Can you summarize the issue so far?"}, config)  # still has full context
```

**`SummarizationMiddleware` parameters:**

| Parameter | Type | Description |
|---|---|---|
| `model` | `str` | Model used for summarization. Use a cheap, fast model (Haiku, phi4-mini, gpt-4o-mini). |
| `trigger` | `tuple[str, int]` | `("tokens", N)` — trigger after N tokens, or `("messages", N)` — after N messages |
| `keep` | `tuple[str, int]` | `("messages", N)` — always retain last N messages verbatim after summarization |
| `token_counter` | `callable` | Custom function to count tokens. Defaults to model's tokenizer. |
| `prompt_template` | `str` | Custom prompt for the summarizer model. |
| `max_tokens_for_summary` | `int \| None` | Max tokens fed to summarizer call. `None` = no limit. |

**Case study — long debugging session:**
In a real SRE use case, agents analyzing large stack traces use `SummarizationMiddleware`
with a fast model (like `phi4-mini`) as the summarizer and `trigger=("messages", 10)` to compress
older exchanges while retaining the last 3 raw messages. The developer invoking the agent sees no
difference — the compression is entirely transparent to the call site.

> 💡 **Tip:** Use `trigger=("tokens", 4000)` (token-based) rather than `("messages", 10)`
> (count-based) in production. Token count more accurately tracks actual context pressure since
> message lengths vary significantly.

---

## 6. Accessing & Writing State

State is available to tools and middleware via the `runtime` object. The LLM never sees
`runtime` — it is hidden from the tool schema automatically.

---

### 6a. Read state in a tool

```python
# ══════════════════════════════════════════════════════════════════════════════
# READ STATE IN A TOOL — via ToolRuntime
# ══════════════════════════════════════════════════════════════════════════════
from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime
from typing import Optional


class CustomerState(AgentState):
    customer_id: str
    plan_tier: Optional[str]    # Optional — may not be set yet


@tool
def get_billing_info(runtime: ToolRuntime) -> str:
    """
    Retrieve billing details for the current customer.
    The 'runtime' parameter is hidden from the model — do not include it
    in the docstring description visible to the LLM.
    """
    # ── safe state access with getattr fallback ─────────────────────────────────
    state = runtime.state if hasattr(runtime, "state") else {}

    customer_id = (
        state.get("customer_id")
        if isinstance(state, dict)
        else getattr(state, "customer_id", None)
    )
    plan_tier = (
        state.get("plan_tier")
        if isinstance(state, dict)
        else getattr(state, "plan_tier", "unknown")
    )

    if not customer_id:
        return "No customer ID found in session state."

    return f"Customer {customer_id} is on the {plan_tier} plan. Last invoice: INR 4,200."


agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[get_billing_info],
    state_schema=CustomerState,
)

result = agent.invoke({
    "messages":     [{"role": "user", "content": "Show me my billing info."}],
    "customer_id":  "cust_98231",
    "plan_tier":    "pro",
})
print(result["messages"][-1].content)
```

---

### 6b. Write state from a tool — `Command`

Tools can return a `Command` object to **mutate state** and update the message history
simultaneously. This is how tools propagate results back into the agent's memory.

```python
# ══════════════════════════════════════════════════════════════════════════════
# WRITE STATE FROM A TOOL — Command update pattern
# ══════════════════════════════════════════════════════════════════════════════
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langchain.agents import create_agent, AgentState
from langgraph.types import Command
from pydantic import BaseModel
from typing import Optional


class LeadState(AgentState):
    lead_name: Optional[str]         # written by lookup_lead tool
    crm_id: Optional[str]            # written by lookup_lead tool


class RequestContext(BaseModel):     # read-only, per-invocation
    auth_token: str


@tool
def lookup_lead(runtime: ToolRuntime) -> Command:
    """
    Looks up lead information from CRM and stores it in agent state so
    subsequent tools (e.g. draft_email) can access it without re-fetching.
    """
    # ── read from per-invocation context (not state) ──────────────────────────
    auth_token = (
        runtime.context.auth_token
        if hasattr(runtime, "context") and hasattr(runtime.context, "auth_token")
        else ""
    )

    # Simulate CRM lookup
    lead_data = {"name": "Rahul Mehta", "crm_id": "CRM-44021"}

    return Command(
        update={
            "lead_name": lead_data["name"],     # write to custom state field
            "crm_id":    lead_data["crm_id"],   # write to custom state field
            "messages": [
                ToolMessage(
                    content=f"Found lead: {lead_data['name']} (ID: {lead_data['crm_id']})",
                    tool_call_id=runtime.tool_call_id,  # required for ToolMessage
                )
            ],
        }
    )


@tool
def draft_email(runtime: ToolRuntime) -> str:
    """Draft a follow-up email to the lead found in state."""
    state = runtime.state if hasattr(runtime, "state") else {}
    lead_name = (
        state.get("lead_name")
        if isinstance(state, dict)
        else getattr(state, "lead_name", None)
    )

    if not lead_name:
        return "Lead not found in state. Please call lookup_lead first."

    return f"Subject: Following up, {lead_name}\nBody: Hi {lead_name}, ..."


agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[lookup_lead, draft_email],
    state_schema=LeadState,
    context_schema=RequestContext,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Look up the lead and draft an email."}]},
    context=RequestContext(auth_token="Bearer xyz"),
)
```

**`Command` update pattern:**

| Field in `update` dict | What it does |
|---|---|
| Custom state field (e.g. `lead_name`) | Directly sets that field in `AgentState` |
| `"messages"` key | Appends `ToolMessage` to conversation history (required for tool result tracking) |

> ⚠️ **Always set `tool_call_id`** on the `ToolMessage` inside `Command.update["messages"]`.
> It must match `runtime.tool_call_id`. Without it, most providers raise a validation error
> because they require every tool call to have a corresponding result with matching ID.

---

### 6c. Dynamic prompt from state — `@dynamic_prompt`

Injects context-aware instructions into the system prompt on every model call, based on runtime
context (not persisted state). Useful for personalisation and role-based system prompts.

```python
# ══════════════════════════════════════════════════════════════════════════════
# DYNAMIC SYSTEM PROMPT — @dynamic_prompt middleware
# ══════════════════════════════════════════════════════════════════════════════
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from typing import TypedDict


class UserContext(TypedDict):
    user_name: str
    preferred_language: str   # "en" | "hi" | "mr"
    role: str                 # "admin" | "viewer"


def get_inventory(item: str) -> str:
    """Get current inventory count for an item."""
    inventory = {"rice": 120, "wheat": 45, "sugar": 8}
    count = inventory.get(item.lower(), -1)
    return f"{item}: {count} units" if count >= 0 else f"{item}: not found"


@dynamic_prompt
def role_aware_system_prompt(request: ModelRequest) -> str:
    """
    Builds a system prompt that includes user name, role, and language
    preference from the per-invocation context.
    """
    # ── safe context access ────────────────────────────────────────────────────
    ctx = getattr(request.runtime, "context", {}) or {}

    user_name = ctx.get("user_name", "User") if isinstance(ctx, dict) else getattr(ctx, "user_name", "User")
    role = ctx.get("role", "viewer") if isinstance(ctx, dict) else getattr(ctx, "role", "viewer")
    lang = ctx.get("preferred_language", "en") if isinstance(ctx, dict) else getattr(ctx, "preferred_language", "en")

    base = f"You are an inventory assistant. Address the user as {user_name}."
    if role == "admin":
        base += " The user has admin privileges and can update inventory."
    base += f" Respond in {'Hindi' if lang == 'hi' else 'English'}."
    return base


agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[get_inventory],
    middleware=[role_aware_system_prompt],
    context_schema=UserContext,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "How much rice do we have?"}]},
    context=UserContext(user_name="Rohit", preferred_language="en", role="admin"),
)
for msg in result["messages"]:
    msg.pretty_print()
```

---

## 7. Before Model / After Model Hooks

Execution flow with both hooks active:

```
__start__
    │
    ▼
before_model   ◄──────────────── (re-enters here after each tool call)
    │
    ▼
  model  ──────────────────────► (if no tool call → __end__)
    │
    ▼
after_model
    │
    ├──► tools  (if tool calls present)
    │       │
    │       └──► before_model (loops back)
    │
    └──► __end__ (if no tool calls)
```

| Hook | Decorator | Runs when | Primary use cases |
|---|---|---|---|
| `@before_model` | `before_model` | Before every model call, including after tool returns | Trim messages, inject context, token budget check |
| `@after_model` | `after_model` | After every model response | PII scrubbing, content filtering, response logging, cost tracking |
| `@dynamic_prompt` | `dynamic_prompt` | Before model call (prompt injection point) | Personalised system prompt from context/state |

```python
# ── Combining multiple middleware (order matters) ──────────────────────────────
agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[],
    middleware=[
        role_aware_system_prompt,     # 1. build system prompt
        keep_recent_window,           # 2. trim message window
        scrub_sensitive_responses,    # 3. scrub PII after model responds
    ],
    checkpointer=InMemorySaver(),
)
```

> 💡 **Middleware ordering:** `@before_model` hooks run in list order before the model.
> `@after_model` hooks run in list order after the model. `@dynamic_prompt` is a special
> `@before_model` that specifically targets the system prompt injection point.

---

## 8. Production Patterns & Pitfalls

### Pattern 1 — Structured `thread_id` for multi-tenant apps

```python
# Encode tenant + user + session to avoid cross-contamination
thread_id = f"tenant:{tenant_id}:user:{user_id}:session:{session_id}"
# e.g. "tenant:acme:user:priya-sharma:session:2026-06-04-001"
```

### Pattern 2 — Async PostgresSaver for FastAPI

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import asyncio

async def build_agent():
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.asetup()   # async version of setup()
        agent = create_agent(
            model="anthropic:claude-sonnet-4-6",
            tools=[],
            checkpointer=checkpointer,
        )
        return agent
```

### Pattern 3 — State schema evolution (additive only)

```python
# SAFE — new field has a default
class AgentStateV2(AgentState):
    user_id: str
    language: str = "en"    # new field: default prevents crash on old checkpoints

# UNSAFE — removes or renames field
class AgentStateV2Bad(AgentState):
    uid: str     # renamed from user_id — old checkpoints will fail to deserialize
```

### Pattern 4 — Checkpoint TTL (stale state cleanup)

Every node write creates a new checkpoint row. After a million turns you have a very large,
slow table. Run a nightly job to delete checkpoints for threads older than N days. Do not
discover this at 90% disk usage.

```python
# Nightly cleanup job example (run as a Celery Beat task or cron)
import psycopg, datetime

def purge_old_checkpoints(days_old: int = 30):
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=days_old)
    with psycopg.connect(DB_URI) as conn:
        conn.execute(
            "DELETE FROM checkpoints WHERE created_at < %s",
            (cutoff,),
        )
        conn.commit()
```

### Common pitfalls

| Pitfall | Cause | Fix |
|---|---|---|
| Agent doesn't remember previous turn | Missing `checkpointer` or wrong `thread_id` | Verify `checkpointer` is set and `thread_id` is identical across calls |
| `KeyError` on state field | Custom field not in `AgentState` subclass | Add field to `state_schema` class |
| Provider API error after delete | Message sequence is invalid | Ensure history starts with user message; tool calls have matching tool results |
| `ToolMessage` not linked to call | Missing `tool_call_id` | Set `tool_call_id=runtime.tool_call_id` in every `ToolMessage` |
| State crash after deploy | Renamed field in `AgentState` | Only add fields with defaults; never rename |
| Memory leak in production | No checkpoint TTL | Add nightly cleanup job |
| AttributeError on `runtime.state` | Accessing `runtime` outside tool | Always guard with `hasattr(runtime, "state")` |
| VSCode type warning on `state.get()` | `AgentState` is a TypedDict-like, not plain dict | Use `getattr(state, "field", default)` for Pydantic states; `state.get()` for dict states |

---

## 9. Quick Reference Table

| Class / Decorator | Import | One-line purpose |
|---|---|---|
| `InMemorySaver` | `langgraph.checkpoint.memory` | RAM checkpointer. Dev/test only. |
| `SqliteSaver` | `langgraph.checkpoint.sqlite` | File-backed checkpointer. Single-server prod. |
| `PostgresSaver` | `langgraph.checkpoint.postgres` | DB-backed checkpointer. Multi-server prod. |
| `AsyncPostgresSaver` | `langgraph.checkpoint.postgres.aio` | Async variant for FastAPI/async apps. |
| `AgentState` | `langchain.agents` | Base state schema with `messages` field. |
| `create_agent` | `langchain.agents` | Factory function to build a checkpointed agent. |
| `@before_model` | `langchain.agents.middleware` | Hook that runs before every model call. |
| `@after_model` | `langchain.agents.middleware` | Hook that runs after every model response. |
| `@dynamic_prompt` | `langchain.agents.middleware` | Before-model hook for system prompt injection. |
| `SummarizationMiddleware` | `langchain.agents.middleware` | Auto-summarizes history at token threshold. |
| `RemoveMessage` | `langchain.messages` | Marks a message for deletion from state. |
| `REMOVE_ALL_MESSAGES` | `langgraph.graph.message` | Sentinel ID to clear all messages from state view. |
| `Command` | `langgraph.types` | Return type from tools that updates state + messages. |
| `ToolRuntime` | `langchain.tools` | Runtime injected into tools; exposes `state`, `context`, `tool_call_id`. |
| `Runtime` | `langgraph.runtime` | Runtime injected into middleware; exposes `context`, `config`. |

---

> 🔗 **Official docs:**
> - Memory overview: https://docs.langchain.com/oss/python/langgraph/memory
> - Persistence / checkpointers: https://docs.langchain.com/oss/python/langgraph/persistence
> - Prebuilt middleware: https://docs.langchain.com/oss/python/langchain/middleware/built-in
> - Middleware reference: https://reference.langchain.com/python/langchain/middleware