## **LangChain Human-in-the-Loop (HITL)**

### **1. What it is and how it relates to prior notes**

1. `HumanInTheLoopMiddleware` adds human oversight to **tool calls specifically** — it pauses execution when a model proposes a risky action (writing a file, running SQL) and waits for a decision before the tool actually runs.
2. This is narrower than the general `before_agent`/`after_agent` custom guardrails covered in `[[langchain_guardrails_notes.md]]` — those hooks wrap the whole agent turn; HITL middleware hooks specifically into the point between "model proposed tool calls" and "tools execute."
3. Mechanism: each tool call is checked against a configurable policy (`interrupt_on`). If it matches, the middleware raises a LangGraph `interrupt`, which halts execution and persists graph state via the checkpointer, so the run can resume later — potentially in a different process.
4. This is the deep-dive on the HITL middleware itself; layering it with PII detection or other guardrails is the same "stack multiple middleware" idea already covered in the guardrails notes.

### **2. Decision types**

| Decision | Effect | Typical use |
|---|---|---|
| `approve` | Executes the tool with original arguments, unchanged | Send an email draft as written |
| `edit` | Executes the tool with modified arguments | Change a recipient before sending |
| `reject` | Skips execution, returns rejection feedback to the model | Deny a file deletion and explain why |
| `respond` | Skips execution, returns the human's message as a synthetic successful tool result | Answer an `ask_user`-style tool directly |

1. `reject` and `respond` are not interchangeable: `reject` tells the model the action failed; `respond` tells the model the action succeeded with the human's message as the result. Using `respond` to deny a side-effecting tool is wrong — the model will believe it worked.
2. Which decisions are legal per tool is set in `interrupt_on` via `allowed_decisions` — not every tool needs to allow all four.
3. When editing arguments, keep changes conservative — a large edit can make the model re-evaluate its whole plan and re-trigger tool calls unexpectedly.
4. Multiple simultaneous interrupts require one decision per action, in the same order the actions were presented.

### **3. Configuring interrupts**

1. Add `HumanInTheLoopMiddleware` to the agent's `middleware` list; map each tool name to `True` (all four decisions allowed, default description), `False` (auto-approve, no interrupt), or an `InterruptOnConfig` dict restricting `allowed_decisions`.
2. A checkpointer is mandatory — interrupts rely on LangGraph's persistence layer to survive the pause. `InMemorySaver` is fine for prototyping; production needs a durable one (`AsyncPostgresSaver`, `MongoDBSaver`).
3. `description_prefix` sets the default text shown to the human reviewer; a per-tool `description` (string or callable) overrides it.

```python
import os
from dotenv import load_dotenv
from langchain_nebius import ChatNebius
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

model = ChatNebius(
    model="meta-llama/Llama-3.3-70B-Instruct-fast",
    api_key=os.environ["NEBIUS_API_KEY"],
)

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file path."""
    return f"wrote {len(content)} chars to {path}"

@tool
def execute_sql(query: str) -> str:
    """Execute a raw SQL query."""
    return f"executed: {query}"

@tool
def read_data(table: str) -> str:
    """Read rows from a table, read-only."""
    return f"rows from {table}"

agent = create_agent(
    model=model,
    tools=[write_file, execute_sql, read_data],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_file": True,
                "execute_sql": {"allowed_decisions": ["approve", "reject"]},
                "read_data": False,
            },
            description_prefix="Tool execution pending approval",
        ),
    ],
    checkpointer=InMemorySaver(),
)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "hitl-demo-1"}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Delete old records from the database"}]},
        config=config,
        version="v2",
    )
    print(result.interrupts)
```

### **4. Conditional interrupts (`when` predicate)**

1. By default every tool call listed in `interrupt_on` pauses. Add a `when` callable to a tool's `InterruptOnConfig` to interrupt only when the call's arguments match a condition — everything else auto-approves.
2. `when` receives a `ToolCallRequest` and returns `True` to interrupt, `False` to skip the interrupt. Calls that evaluate `False` never enter the reviewer's batch.
3. Requires `langchain>=1.3.3` — this is a recent addition, not part of the original HITL middleware release.

```python
import os
from dotenv import load_dotenv
from langchain_nebius import ChatNebius
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, ToolCallRequest
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

model = ChatNebius(model="meta-llama/Llama-3.3-70B-Instruct-fast", api_key=os.environ["NEBIUS_API_KEY"])

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file path."""
    return f"wrote {len(content)} chars to {path}"

@tool
def execute_sql(query: str) -> str:
    """Execute a raw SQL query."""
    return f"executed: {query}"

def writes_outside_workspace(request: ToolCallRequest) -> bool:
    path = request.tool_call["args"].get("path", "")
    return not path.startswith("/workspace/")

def is_write_query(request: ToolCallRequest) -> bool:
    query = request.tool_call["args"].get("query", "")
    return not query.lstrip().upper().startswith("SELECT")

agent = create_agent(
    model=model,
    tools=[write_file, execute_sql],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_file": {"allowed_decisions": ["approve", "edit", "reject"], "when": writes_outside_workspace},
                "execute_sql": {"allowed_decisions": ["approve", "reject"], "when": is_write_query},
            },
        ),
    ],
    checkpointer=InMemorySaver(),
)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "hitl-demo-2"}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Write a note to /tmp/scratch.txt saying hello"}]},
        config=config,
        version="v2",
    )
    print(result.interrupts)
```

### **5. Responding to interrupts and resuming**

1. `invoke(..., version="v2")` returns a `GraphOutput` with `.value` and `.interrupts` — this is the newer, typed invoke surface (LangGraph >= 1.1, opt-in, backward compatible). The older default (no `version` kwarg) returns a plain dict with the interrupt payload under `"__interrupt__"` instead.
2. Resume with `agent.invoke(Command(resume={"decisions": [...]}), config=config, version="v2")` using the **same thread ID** — one decision object per paused action, same order as `.interrupts` listed them.
3. Decision payload shapes: `{"type": "approve"}`; `{"type": "edit", "edited_action": {"name": ..., "args": {...}}}`; `{"type": "reject", "message": "..."}`; `{"type": "respond", "message": "..."}`.
4. Omitting `message` on `reject` falls back to a default that tells the model not to retry the same call unless the user asks again — for side-effecting tools, write a specific message telling the model whether to abandon, ask a follow-up, or try something safer.

```python
import os
from dotenv import load_dotenv
from langchain_nebius import ChatNebius
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()

model = ChatNebius(model="meta-llama/Llama-3.3-70B-Instruct-fast", api_key=os.environ["NEBIUS_API_KEY"])

@tool
def execute_sql(query: str) -> str:
    """Execute a raw SQL query."""
    return f"executed: {query}"

agent = create_agent(
    model=model,
    tools=[execute_sql],
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"execute_sql": {"allowed_decisions": ["approve", "edit", "reject"]}})],
    checkpointer=InMemorySaver(),
)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "hitl-demo-3"}}

    paused = agent.invoke(
        {"messages": [{"role": "user", "content": "Delete rows older than 30 days from the logs table"}]},
        config=config,
        version="v2",
    )
    print(paused.interrupts)

    edited = agent.invoke(
        Command(resume={"decisions": [{
            "type": "edit",
            "edited_action": {"name": "execute_sql", "args": {"query": "DELETE FROM logs WHERE created_at < NOW() - INTERVAL '90 days'"}},
        }]}),
        config=config,
        version="v2",
    )
    print(edited.value["messages"][-1].content)
```

### **6. Streaming with interrupts**

1. `stream_events(..., version="v3")` streams LLM tokens via `stream.messages` while the run is in progress, and exposes `stream.interrupted` / `stream.interrupts` once the run pauses.
2. Resume the same way as non-streaming: `agent.stream_events(Command(resume={"decisions": [...]}), config=config, version="v3")`.

```python
import os
from dotenv import load_dotenv
from langchain_nebius import ChatNebius
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()

model = ChatNebius(model="meta-llama/Llama-3.3-70B-Instruct-fast", api_key=os.environ["NEBIUS_API_KEY"])

@tool
def execute_sql(query: str) -> str:
    """Execute a raw SQL query."""
    return f"executed: {query}"

agent = create_agent(
    model=model,
    tools=[execute_sql],
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"execute_sql": {"allowed_decisions": ["approve", "reject"]}})],
    checkpointer=InMemorySaver(),
)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "hitl-demo-4"}}

    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": "Delete old records from the database"}]},
        config=config,
        version="v3",
    )
    for message in stream.messages:
        for token in message.text:
            print(token, end="", flush=True)

    if stream.interrupted:
        print("\n\ninterrupt:", stream.interrupts)

    resumed = agent.stream_events(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config,
        version="v3",
    )
    for message in resumed.messages:
        for token in message.text:
            print(token, end="", flush=True)
```

### **7. Execution lifecycle**

1. Implemented as an `after_model` hook — it runs after the model generates a response but before any tool calls execute.
2. Steps: model responds → middleware inspects tool calls for matches against `interrupt_on` → if any match, builds a `HITLRequest` (`action_requests` + `review_configs`) and calls `interrupt()` → execution pauses → human decisions come back as `HITLResponse` → middleware executes approved/edited calls, synthesizes `ToolMessage`s for rejected calls, returns human text directly as `ToolMessage`s for `respond` calls → execution resumes.
3. This confirms HITL is a middleware built on the same `after_model` hook style already covered in `[[langchain_middleware_notes.md]]` — it's not a separate mechanism from the general middleware system, just a specific built-in that uses it.

### **8. Custom HITL logic**

1. For anything the built-in middleware doesn't cover, build directly on the low-level `interrupt()` primitive plus the middleware abstraction — call `interrupt(value)` inside a custom `wrap_tool_call` or node, halt there, and read the resume value back as the return of `interrupt()`.
2. This is the same underlying primitive that powers `HumanInTheLoopMiddleware` — reach for it directly when you need decision types or routing logic outside the four built-in ones (approve/edit/reject/respond).

```python
import os
from typing import Callable
from dotenv import load_dotenv
from langchain_nebius import ChatNebius
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call, ToolCallRequest
from langchain.messages import ToolMessage
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt

load_dotenv()

model = ChatNebius(model="meta-llama/Llama-3.3-70B-Instruct-fast", api_key=os.environ["NEBIUS_API_KEY"])

@tool
def send_payment(amount: float, recipient: str) -> str:
    """Send a payment to a recipient."""
    return f"sent {amount} to {recipient}"

@wrap_tool_call
def require_payment_confirmation(request: ToolCallRequest, handler: Callable) -> ToolMessage:
    if request.tool_call["name"] != "send_payment":
        return handler(request)
    decision = interrupt({
        "action": "confirm_payment",
        "args": request.tool_call["args"],
    })
    if decision.get("confirmed"):
        return handler(request)
    return ToolMessage(content="payment cancelled by reviewer", tool_call_id=request.tool_call["id"])

agent = create_agent(
    model=model,
    tools=[send_payment],
    middleware=[require_payment_confirmation],
    checkpointer=InMemorySaver(),
)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "hitl-demo-5"}}
    paused = agent.invoke(
        {"messages": [{"role": "user", "content": "Send 500 to Rohan"}]},
        config=config,
        version="v2",
    )
    print(paused.interrupts)
```

### **Decision guide**

1. Simple approve/edit/reject/respond gate on specific tool names → **`HumanInTheLoopMiddleware`** with `interrupt_on`, that's the entire job.
2. Only some calls of an allowed tool are risky (e.g. writes outside a workspace, non-`SELECT` SQL) → add a `when` predicate instead of gating the whole tool.
3. Need a durable, multi-process pause (review happens hours later, in a different service) → make sure the checkpointer is a real persistence backend, not `InMemorySaver`.
4. Need streaming UX while the agent works, with the ability to still interrupt → `stream_events(version="v3")` and check `stream.interrupted`.
5. Need a decision type the four built-ins don't cover, or need to gate something other than a tool call → drop to the raw `interrupt()` primitive inside a custom `wrap_tool_call`/node, as in section 8.

### **Things to verify before relying on this**

1. `version="v2"` on `invoke`/`stream_events` (`GraphOutput` with `.value`/`.interrupts`) is an **opt-in surface introduced in LangGraph v1.1.0** — confirm the installed `langgraph` version supports it before using it as the default pattern; the legacy path (`result["__interrupt__"]`, no `version` kwarg) still works and is what older code will show.
2. The `when` predicate on `InterruptOnConfig` requires `langchain>=1.3.3` specifically — check the installed version before relying on conditional interrupts; on older `langchain` this kwarg won't exist.
3. `stream_events(version="v3")` is a separate versioned surface from `invoke(version="v2")` — don't assume the two version strings track the same underlying release; verify both against the changelog for the installed `langgraph`/`langchain` pair.
4. Section 8's custom `wrap_tool_call` + raw `interrupt()` pattern is constructed from the documented `after_model`/`wrap_tool_call` mechanics, not copied verbatim from a docs example — test the resume payload shape (`decision.get("confirmed")`) against your actual reviewer UI before trusting it in production.
5. `langchain-nebius` remains the same small externally-maintained partner package flagged in prior notes (PyPI v0.1.x) — re-verify tool-calling behavior under `HumanInTheLoopMiddleware` specifically, since interrupt/resume flows depend on the model producing well-formed tool calls consistently across the pause.