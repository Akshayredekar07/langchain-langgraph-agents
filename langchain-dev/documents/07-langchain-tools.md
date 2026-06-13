# **LangChain Tools**
---
## Model Setup (used in all examples)

```python
import os
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from pydantic import SecretStr

load_dotenv()

openrouter_key = os.getenv("OPENROUTER_API_KEY")
assert openrouter_key, "OPENROUTER_API_KEY is not set in .env"

model = ChatOpenRouter(
    model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    api_key=SecretStr(openrouter_key),
    temperature=0.6,
    max_tokens=1024,
)
```

---

## What Is a Tool

A **tool** is a typed, callable Python function the LLM can invoke. The model reads the name + docstring to decide when to call it, and uses the type hints to build the JSON schema for arguments.

**Minimal anatomy:**

```
name        ← function name (or @tool("custom_name"))
description ← docstring (what the model reads)
args_schema ← auto-inferred from type hints, or Pydantic BaseModel
return      ← str | dict | Command
```

---

## Architecture Diagram

```
User Message
     │
     ▼
┌──────────────┐
│  LLM / Model │  ← reads tool name + docstring + schema
└──────┬───────┘
       │ decides to call tool(args)
       ▼
┌──────────────────────────────┐
│         ToolNode             │
│  ┌────────────────────────┐  │
│  │  @tool function        │  │
│  │  runtime: ToolRuntime  │  │  ← injected, hidden from model
│  │    .state              │  │  ← short-term (conversation)
│  │    .context            │  │  ← immutable (user_id, role)
│  │    .store              │  │  ← long-term (cross-session)
│  │    .stream_writer      │  │  ← real-time progress
│  │    .tool_call_id       │  │  ← correlation ID
│  └────────────────────────┘  │
└──────────────────────────────┘
       │
       ▼
  ToolMessage  →  back into LLM context
```

---

## 01 — @tool Decorator (Simplest)

Use when: single callable, no complex state, straightforward I/O.

```python
from langchain_core.tools import tool

@tool
def get_stock_price(ticker: str) -> str:
    """Fetch the current stock price for a given ticker symbol (e.g. AAPL, TSLA)."""
    # Production: replace with real API call (yfinance, Alpha Vantage, etc.)
    prices = {"AAPL": 189.45, "TSLA": 245.10, "NVDA": 875.20}
    price = prices.get(ticker.upper())
    if price is None:
        return f"Ticker {ticker} not found."
    return f"{ticker.upper()}: ${price}"

# Inspect schema
print(get_stock_price.name)          # get_stock_price
print(get_stock_price.description)   # "Fetch the current stock price..."
print(get_stock_price.args_schema.schema())
```

**Override name + description:**

```python
@tool("stock_price_lookup", description="Look up live equity prices by ticker. Use for any financial query.")
def fetch_price(ticker: str) -> str:
    """Internal implementation."""
    return f"{ticker}: $100"

print(fetch_price.name)  # stock_price_lookup
```

---

## 02 — Pydantic Schema (Complex Inputs)

Use when: multiple fields, enums, validation, optional params.

**Production example — e-commerce order search:**

```python
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.tools import tool

class OrderSearchInput(BaseModel):
    customer_id: str = Field(description="Customer UUID from the auth token")
    status: Literal["pending", "shipped", "delivered", "cancelled"] = Field(
        default="pending",
        description="Filter orders by status"
    )
    limit: int = Field(default=10, ge=1, le=100, description="Max results (1–100)")

@tool(args_schema=OrderSearchInput)
def search_orders(customer_id: str, status: str = "pending", limit: int = 10) -> dict:
    """Search orders for a customer. Supports filtering by status and pagination."""
    # Production: query Postgres/MongoDB
    fake_orders = [
        {"id": "ORD-001", "status": "pending", "total": 1299.00},
        {"id": "ORD-002", "status": "shipped", "total": 450.00},
    ]
    filtered = [o for o in fake_orders if o["status"] == status][:limit]
    return {"customer_id": customer_id, "orders": filtered, "count": len(filtered)}
```

---

## 03 — BaseTool Subclass (Full Control)

Use when: you need `__init__` (inject DB clients, HTTP sessions), async, custom error handling.

**Production example — RAG retrieval tool with injected vector store:**

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Any

class RAGQueryInput(BaseModel):
    query: str = Field(description="Natural language question to retrieve context for")
    top_k: int = Field(default=5, ge=1, le=20)

class VectorStoreRetrieverTool(BaseTool):
    name: str = "rag_retriever"
    description: str = (
        "Retrieve relevant context from the company knowledge base. "
        "Use this before answering any product, policy, or technical question."
    )
    args_schema: Type[BaseModel] = RAGQueryInput

    # Injected at init — not exposed to LLM
    vector_store: Any = None  # e.g. QdrantClient or FAISS

    def _run(self, query: str, top_k: int = 5) -> str:
        if self.vector_store is None:
            return "Vector store not initialised."
        # Production: self.vector_store.similarity_search(query, k=top_k)
        docs = [f"[Doc {i}] Relevant chunk for: {query}" for i in range(top_k)]
        return "\n".join(docs)

    async def _arun(self, query: str, top_k: int = 5) -> str:
        # Async version for production FastAPI / LangGraph async graphs
        return self._run(query, top_k)

# Instantiate with real client
retriever_tool = VectorStoreRetrieverTool(vector_store=None)  # replace None
```

---

## 04 — ToolRuntime (Context Injection)

`runtime: ToolRuntime` is **auto-injected** at execution time and **hidden from the model** — it never appears in the tool schema.

### 04-A State (Short-Term Memory)

Conversation messages + custom state fields. Lives for one session.

**Production example — customer support bot tracking escalation count:**

```python
from langchain_core.tools import tool
from langchain.tools import ToolRuntime

@tool
def check_escalation_threshold(runtime: ToolRuntime) -> str:
    """Check if current conversation should be escalated to a human agent."""
    messages = runtime.state.get("messages", [])
    escalation_count = runtime.state.get("escalation_triggers", 0)

    negative_keywords = ["frustrated", "refund", "lawyer", "useless", "cancel"]
    last_msg = messages[-1].content.lower() if messages else ""
    triggered = any(kw in last_msg for kw in negative_keywords)

    if triggered or escalation_count >= 2:
        return "ESCALATE: Transfer to human agent immediately."
    return f"Continue. Escalation triggers so far: {escalation_count}"
```

**Updating state with Command:**

```python
from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langchain.agents import AgentState

class SupportState(AgentState):
    escalation_triggers: int
    resolved: bool

@tool
def mark_resolved(runtime: ToolRuntime[None, SupportState]) -> Command:
    """Mark the current support ticket as resolved."""
    return Command(
        update={
            "resolved": True,
            "messages": [
                ToolMessage(
                    content="Ticket marked resolved. Closing conversation.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
```

### 04-B Context (Immutable Per-Run)

Passed at `.invoke()` time. Use for auth data — user_id, role, tenant_id.

**Production example — multi-tenant SaaS, role-based data access:**

```python
from dataclasses import dataclass
from langchain_core.tools import tool
from langchain.tools import ToolRuntime

@dataclass
class TenantContext:
    user_id: str
    tenant_id: str
    role: str  # "admin" | "viewer" | "editor"

TENANT_DATA = {
    "tenant_acme": {"revenue": 4_200_000, "users": 1500},
    "tenant_xyz":  {"revenue": 800_000,  "users": 320},
}

@tool
def get_tenant_metrics(runtime: ToolRuntime[TenantContext]) -> str:
    """Fetch business metrics for the current user's tenant."""
    ctx = runtime.context
    if ctx.role == "viewer":
        return "Access denied: viewers cannot see revenue data."
    data = TENANT_DATA.get(ctx.tenant_id)
    if not data:
        return "Tenant not found."
    return (
        f"Tenant: {ctx.tenant_id}\n"
        f"Revenue: ${data['revenue']:,}\n"
        f"Users: {data['users']}"
    )

# Invocation — pass context at runtime
from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7

agent = create_agent(
    model,
    tools=[get_tenant_metrics],
    context_schema=TenantContext,
    system_prompt="You are a business analytics assistant.",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What are our metrics?"}]},
    config={"configurable": {"thread_id": str(uuid7())}},
    context=TenantContext(user_id="u-001", tenant_id="tenant_acme", role="admin"),
)
```

### 04-C Store (Long-Term Memory, Cross-Session)

Persists across conversations. Namespace/key pattern.

**Production example — personalised recommendation memory:**

```python
from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from langgraph.store.memory import InMemoryStore  # swap PostgresStore in prod

@tool
def save_user_preference(
    category: str,
    preference: str,
    runtime: ToolRuntime
) -> str:
    """Save a user preference to long-term memory (e.g. language, tone, topic)."""
    user_id = runtime.context.user_id  # requires context_schema with user_id
    runtime.store.put(("preferences", user_id), category, {"value": preference})
    return f"Saved: {category} = {preference}"

@tool
def get_user_preference(category: str, runtime: ToolRuntime) -> str:
    """Retrieve a previously saved user preference."""
    user_id = runtime.context.user_id
    record = runtime.store.get(("preferences", user_id), category)
    if record:
        return f"{category}: {record.value['value']}"
    return f"No preference set for {category}."
```

**Wire up store:**

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

agent = create_agent(
    model,
    tools=[save_user_preference, get_user_preference],
    store=store,
    context_schema=...,
)
```

### 04-D Stream Writer (Real-Time Progress)

**Production example — long-running data pipeline with live updates:**

```python
from langchain_core.tools import tool
from langchain.tools import ToolRuntime
import time

@tool
def run_data_pipeline(dataset_name: str, runtime: ToolRuntime) -> str:
    """Run a multi-stage ETL pipeline. Streams progress back to the user."""
    writer = runtime.stream_writer
    stages = ["Extracting data", "Validating schema", "Transforming records", "Loading to warehouse"]

    for i, stage in enumerate(stages, 1):
        writer(f"[{i}/{len(stages)}] {stage}...")
        time.sleep(0.5)  # Production: real async work here

    return f"Pipeline complete for dataset '{dataset_name}'. 4 stages passed."
```

### 04-E Execution Info

**Production example — idempotency guard on retry:**

```python
from langchain_core.tools import tool
from langchain.tools import ToolRuntime

@tool
def charge_customer(amount: float, runtime: ToolRuntime) -> str:
    """Charge the customer's card. Safe to call; handles retry deduplication."""
    info = runtime.execution_info
    attempt = info.node_attempt if info else 0
    run_id = info.run_id if info else "unknown"

    if attempt > 0:
        # Production: check idempotency key in payments DB before re-charging
        return f"Retry attempt {attempt} detected (run_id={run_id}). Skipping duplicate charge."

    # Production: call Stripe / Razorpay API
    return f"Charged ${amount:.2f} successfully. run_id={run_id}"
```

---

## 05 — Return Types

### Return String

Default. Plain text back to the model.

```python
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Mumbai: 32°C, Humid. Chance of rain: 60%."
```

### Return Dict (Structured)

Model can reason over specific fields.

```python
@tool
def get_invoice(invoice_id: str) -> dict:
    """Fetch invoice details by ID."""
    return {
        "invoice_id": invoice_id,
        "amount": 12500.00,
        "currency": "INR",
        "status": "unpaid",
        "due_date": "2026-06-15",
    }
```

### Return Command (State Mutation)

Use when the tool must write to agent state.

```python
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain.tools import ToolRuntime

@tool
def set_preferred_language(language: str, runtime: ToolRuntime) -> Command:
    """Set the user's preferred response language for this session."""
    return Command(
        update={
            "preferred_language": language,
            "messages": [
                ToolMessage(
                    content=f"Language preference set to: {language}.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
```

---

## 06 — Error Handling (Middleware)

**Production pattern — never let a raw exception crash the agent:**

```python
from collections.abc import Callable
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
import logging

logger = logging.getLogger(__name__)

@wrap_tool_call
def resilient_tool_execution(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage],
) -> ToolMessage:
    """Catch any tool exception and return a structured error message."""
    try:
        return handler(request)
    except ValueError as e:
        logger.warning("Validation error in tool %s: %s", request.tool_call["name"], e)
        return ToolMessage(
            content=f"Invalid input: {e}. Please correct and retry.",
            tool_call_id=request.tool_call["id"],
        )
    except TimeoutError:
        return ToolMessage(
            content="Tool timed out. The external service may be unavailable.",
            tool_call_id=request.tool_call["id"],
        )
    except Exception as e:
        logger.error("Unhandled tool error: %s", e, exc_info=True)
        return ToolMessage(
            content=f"Tool error ({type(e).__name__}). Engineering has been notified.",
            tool_call_id=request.tool_call["id"],
        )

agent = create_agent(
    model,
    tools=[get_stock_price, search_orders],
    middleware=[resilient_tool_execution],
)
```

---

## 07 — Dynamic Tool Selection (Middleware)

### Filter pre-registered tools by role

**Production example — RBAC for a legal document agent:**

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

@dataclass
class UserContext:
    role: str  # "paralegal" | "associate" | "partner"

ROLE_ALLOWED_TOOLS = {
    "paralegal": {"search_case_law", "read_document"},
    "associate":  {"search_case_law", "read_document", "draft_brief", "compare_clauses"},
    "partner":    None,  # None = all tools
}

@wrap_model_call
def rbac_tool_filter(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    ctx = request.runtime.context if request.runtime else None
    role = ctx.role if ctx else "paralegal"
    allowed = ROLE_ALLOWED_TOOLS.get(role)
    if allowed is not None:
        filtered = [t for t in request.tools if t.name in allowed]
        request = request.override(tools=filtered)
    return handler(request)

agent = create_agent(
    model,
    tools=[...],  # all tools
    middleware=[rbac_tool_filter],
    context_schema=UserContext,
)

# Paralegal gets only 2 tools; partner gets all
agent.invoke(
    {"messages": [{"role": "user", "content": "Draft a motion."}]},
    context=UserContext(role="paralegal"),
)
```

### Runtime tool registration (dynamic, from external registry)

**Production example — MCP-loaded tools:**

```python
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ToolCallRequest
from langchain_core.tools import tool

@tool
def crm_get_contact(email: str) -> str:
    """Look up a CRM contact by email address."""
    return f"Contact: Arjun Sharma, Company: TechCorp, Stage: Negotiation"

class MCPDynamicToolMiddleware(AgentMiddleware):
    """Loads tools at request-time from an MCP server or registry."""

    def wrap_model_call(self, request: ModelRequest, handler):
        # In production: fetch tool schemas from MCP server
        dynamic_tools = [crm_get_contact]
        updated = request.override(tools=[*request.tools, *dynamic_tools])
        return handler(updated)

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        # Route execution to the correct dynamic tool
        if request.tool_call["name"] == "crm_get_contact":
            return handler(request.override(tool=crm_get_contact))
        return handler(request)

agent = create_agent(
    model,
    tools=[],
    middleware=[MCPDynamicToolMiddleware()],
)
```

---

## 08 — Full Working Agent (All Pieces Together)

**Production example — Internal IT helpdesk agent:**

```python
import os
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain.agents.middleware import wrap_tool_call
from langchain.tools.tool_node import ToolCallRequest
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from pydantic import SecretStr, BaseModel, Field
from typing import Literal, Callable
from collections.abc import Callable

load_dotenv()

# ── Model ────────────────────────────────────────────────────────────
model = ChatOpenRouter(
    model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    api_key=SecretStr(os.getenv("OPENROUTER_API_KEY")),
    temperature=0.6,
    max_tokens=1024,
)

# ── Context ──────────────────────────────────────────────────────────
@dataclass
class EmployeeContext:
    employee_id: str
    department: str
    role: str  # "engineer" | "manager" | "hr"

# ── Tools ────────────────────────────────────────────────────────────
class TicketInput(BaseModel):
    title: str = Field(description="Short description of the IT issue")
    priority: Literal["low", "medium", "high", "critical"] = "medium"

@tool(args_schema=TicketInput)
def create_support_ticket(
    title: str,
    priority: str,
    runtime: ToolRuntime[EmployeeContext]
) -> str:
    """Create an IT support ticket on behalf of the employee."""
    emp = runtime.context
    ticket_id = f"TKT-{hash(title) % 9999:04d}"
    # Production: POST to Jira / ServiceNow API
    return (
        f"Ticket {ticket_id} created.\n"
        f"Employee: {emp.employee_id} ({emp.department})\n"
        f"Priority: {priority}\n"
        f"Title: {title}"
    )

@tool
def get_my_tickets(runtime: ToolRuntime[EmployeeContext]) -> str:
    """List all open tickets for the current employee."""
    emp = runtime.context
    # Production: query ServiceNow filtered by emp.employee_id
    return f"Open tickets for {emp.employee_id}: TKT-0042 (VPN issue, high)"

@tool
def get_kb_article(issue_type: str, runtime: ToolRuntime) -> str:
    """Search the IT knowledge base for self-service resolution steps."""
    kb = {
        "vpn": "1. Restart Cisco AnyConnect\n2. Flush DNS\n3. Reboot",
        "password": "Go to https://sso.company.com/reset",
        "laptop":  "Run diagnostics: Win+R > msinfo32",
    }
    for k, v in kb.items():
        if k in issue_type.lower():
            return v
    return "No article found. Creating a ticket is recommended."

@tool
def save_employee_device(device_name: str, runtime: ToolRuntime[EmployeeContext]) -> str:
    """Save the employee's primary device to long-term memory."""
    emp = runtime.context
    runtime.store.put(("devices",), emp.employee_id, {"device": device_name})
    return f"Device '{device_name}' saved for employee {emp.employee_id}."

# ── Error middleware ──────────────────────────────────────────────────
@wrap_tool_call
def catch_tool_errors(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage],
) -> ToolMessage:
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool failed: {e}. Please try again.",
            tool_call_id=request.tool_call["id"],
        )

# ── Agent ─────────────────────────────────────────────────────────────
store = InMemoryStore()

agent = create_agent(
    model,
    tools=[create_support_ticket, get_my_tickets, get_kb_article, save_employee_device],
    context_schema=EmployeeContext,
    store=store,
    middleware=[catch_tool_errors],
    system_prompt=(
        "You are an IT helpdesk assistant. "
        "Always try the knowledge base before creating a ticket. "
        "Be concise and action-oriented."
    ),
)

# ── Run ───────────────────────────────────────────────────────────────
result = agent.invoke(
    {"messages": [{"role": "user", "content": "My VPN keeps disconnecting every 10 minutes."}]},
    context=EmployeeContext(employee_id="EMP-1042", department="Engineering", role="engineer"),
)
print(result["messages"][-1].content)
```

---

## 09 — Reserved Parameter Names

| Name      | Reserved For                                     | Use Instead             |
|-----------|--------------------------------------------------|-------------------------|
| `config`  | Internal `RunnableConfig` injection              | `runtime.config`        |
| `runtime` | `ToolRuntime` injection (DO name it this)        | — (this is correct)     |

Do **not** name your own tool args `config` or `runtime` — they will conflict with injected parameters and cause runtime errors.

---

## 10 — Tool Return Type Decision Tree

```
Does the tool mutate agent state?
    YES → return Command(update={...}) with ToolMessage
    NO  →
        Is the output structured (multiple fields)?
            YES → return dict
            NO  → return str
```

---

## 11 — Quick Reference Cheatsheet

| Pattern               | When to Use                                              | Key API                         |
|-----------------------|----------------------------------------------------------|---------------------------------|
| `@tool` decorator     | Simple function, one job                                 | `@tool`                         |
| Pydantic schema       | Multiple args, validation, enums                         | `args_schema=MyModel`           |
| `BaseTool` subclass   | Injected clients, async, custom `__init__`               | `class T(BaseTool): _run()`     |
| `ToolRuntime.state`   | Read/write conversation state                            | `runtime.state["messages"]`     |
| `ToolRuntime.context` | Read immutable per-run data (auth, role, tenant)         | `runtime.context.user_id`       |
| `ToolRuntime.store`   | Persist data across sessions                             | `runtime.store.put/get`         |
| `stream_writer`       | Emit progress during long ops                            | `runtime.stream_writer("msg")`  |
| `execution_info`      | Retry deduplication, idempotency                         | `runtime.execution_info.attempt`|
| `Command` return      | Write to graph state from inside tool                    | `return Command(update={...})`  |
| Error middleware      | Catch exceptions, return clean ToolMessage               | `@wrap_tool_call`               |
| RBAC filter           | Limit tool access by role/permission                     | `@wrap_model_call`              |
| Dynamic registration  | Load tools at runtime (MCP, DB)                          | `AgentMiddleware`               |