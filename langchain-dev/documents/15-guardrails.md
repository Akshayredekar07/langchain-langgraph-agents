# **Guardrails**

## **1. What a guardrail is in LangChain**

1. A guardrail is a safety check that validates or filters content at a defined point in an agent's execution — before it starts, after it finishes, or around a model/tool call.
2. There is no separate "guardrail API" — guardrails are implemented using the same middleware hooks (`before_agent`, `after_agent`, `wrap_model_call`, `wrap_tool_call`, `before_model`, `after_model`, `before_tool_call`/`wrap_tool_call`) covered in the middleware docs. Guardrails are a *use case* for middleware, not a distinct system.
3. Common targets: PII leakage, prompt injection, harmful/inappropriate content, business-rule enforcement, output quality/accuracy checks.
4. Two complementary approaches:
   - **Deterministic** — regex, keyword lists, explicit rule checks. Fast, predictable, cheap; misses nuance.
   - **Model-based** — an LLM or classifier judges content semantically. Catches subtle violations; slower, costs tokens, and is itself a call that can fail or be gamed.
5. LangChain ships two ready-made guardrail middlewares (`PIIMiddleware`, `HumanInTheLoopMiddleware`) and leaves everything else to custom middleware built on the same hooks.

### **Hook reference**

| Hook | When it runs | Can `jump_to`? | Return type |
|---|---|---|---|
| `before_agent` | Once, before the agent starts | yes (`"end"`) | `dict \| None` |
| `before_model` | Before each model call | yes (`"end"`, `"model"`) | `dict \| None` |
| `after_model` | After each model call | yes (`"end"`, `"tools"`, `"model"`) | `dict \| None` |
| `after_agent` | Once, after the agent completes | **no** (it's the last node) | `dict \| None` |
| `wrap_model_call` | Wraps every model call | no | `ModelResponse` / `ExtendedModelResponse` |
| `wrap_tool_call` | Wraps every tool call | no | `ToolMessage \| Command` |

**Execution order** with `[M1, M2, M3]`:
- `before_*` hooks: `M1 → M2 → M3`
- `after_*` hooks: `M3 → M2 → M1` (reverse)
- `wrap_*` hooks: nested — first middleware wraps the rest (`M1` outermost)

### **Jump targets**

- `'end'` — go to the last node (or first `after_agent` if any).
- `'tools'` — go to the tool execution node.
- `'model'` — go to the first `before_model` hook.

> **Important constraint:** you must declare the jump targets on the hook via `@hook_config(can_jump_to=[...])` (or `can_jump_to=[...]` on the decorator). `create_agent` inspects this metadata to wire the graph edges; if you don't declare a destination, returning `{"jump_to": "end"}` will fail at graph build time.

---

## **2. Built-in guardrail — PII detection**

1. `PIIMiddleware(pii_type, strategy=..., apply_to_input/output/tool_results=...)` is a deterministic guardrail: pattern-match, then act.
2. Built-in `pii_type`s: `email`, `credit_card` (Luhn-validated), `ip`, `mac_address`, `url`. Anything else needs a custom `detector` (regex string, compiled regex, or a function returning `PIIMatch` dicts).
3. Strategies:
   - `redact` (default) — replace with `[REDACTED_{PII_TYPE}]`.
   - `mask` — partially obscure, e.g. `****-****-****-1234`.
   - `hash` — deterministic hash, useful when you need to correlate the same value across messages without storing it in the clear. Format: `<type_hash:digest>`.
   - `block` — raises `PIIDetectionError` the moment it's detected.
4. Coverage flags are independent and default narrow: `apply_to_input=True` by default (checks user messages before the model call); `apply_to_output` and `apply_to_tool_results` both default `False` — you must opt in explicitly if you want the model's own output or tool results scanned too.
5. **Stream transformer (langchain≥1.3.2):** with `apply_to_output=True`, a transformer is also installed that scrubs PII from:
   - Streamed AI text deltas (`content-block-delta` of type `text-delta`)
   - Streamed tool-call argument chunks (including the finalized `tool_call` block on `content-block-finish`)
   - Tool execution events on the `tools` channel (`tool-started.input`, `tool-output-delta`, `tool-finished.output`, `tool-error.message`)
   - State snapshots on the `values` channel — message lists are walked and each message's `.content` is redacted on a **fresh copy** (state itself stays intact for `before_model` / `after_model` to act on independently)

   This matters because `after_model` state-level redaction happens *after* a tool has already streamed; the transformer closes the window where live readers of `astream_events(version="v3")` or `run.messages` would otherwise see raw PII.

6. Register one `PIIMiddleware` instance per `pii_type` (and per direction if you want different strategies for input vs. output).

### **2a. Canonical example**

```python
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware

agent = create_agent(
    model="gpt-5",                                # any chat model works
    tools=[customer_service_tool, email_tool],
    middleware=[
        PIIMiddleware("email",       strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask",   apply_to_input=True),
        PIIMiddleware("api_key",     detector=r"sk-[a-zA-Z0-9]{32}",
                      strategy="block", apply_to_input=True),
    ],
)
```

### **2b. Extra example — DLP on tool results**

Tool outputs frequently carry PII *from your own systems* (a customer lookup tool returns an email, a SQL tool returns a phone number). Default is off — turn it on:

```python
agent = create_agent(
    model="gpt-5",
    tools=[customer_lookup, sql_query],
    middleware=[
        # Input: just redact emails so the model can still reason about the request.
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        # Output & tool results: mask credit cards everywhere they show up
        # (model's reply, tool result that came back from a DB, etc.).
        PIIMiddleware("credit_card", strategy="mask",
                      apply_to_input=True,
                      apply_to_output=True,
                      apply_to_tool_results=True),
        # Hard stop on API keys, anywhere.
        PIIMiddleware("api_key", detector=r"sk-[a-zA-Z0-9]{32}",
                      strategy="block",
                      apply_to_input=True,
                      apply_to_output=True,
                      apply_to_tool_results=True),
    ],
)
```

### 2c. Extra example — output redaction with stream coverage

If you also stream the agent to a UI, this is the right shape — input redacted for the model, output redacted both in state *and* on the wire:

```python
agent = create_agent(
    model="gpt-5",
    tools=[...],
    middleware=[
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("email", strategy="redact", apply_to_output=True),  # + stream transformer
    ],
)

# Anyone reading the stream will see redacted emails, not raw ones.
async for event in agent.astream_events(
    {"messages": [{"role": "user", "content": "Email me at jane@example.com"}]},
    version="v3",
):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content)   # already redacted
```

> **Stream caveat (real gotcha):** if you implement custom output scrubbing in `wrap_model_call` instead of using `PIIMiddleware`, tokens may have *already reached* the browser by the time you scrub. The model has produced the response by the time `wrap_model_call` sees the completed response. PIIMiddleware handles this by installing the stream transformer. If you need custom rules, either express them as a `PIIMiddleware` custom detector, or don't stream the screened agent.

### 2d. Extra example — custom detector as a callable

For detectors that need a validation step beyond regex (e.g. a corporate employee-ID format with a checksum):

```python
import re
from langchain.agents.middleware import PIIMiddleware

EMPLOYEE_ID_RE = re.compile(r"\bEID-\d{6,10}\b")

def employee_id_detector(text: str) -> list[dict]:
    return [
        {"text": m.group(), "start": m.start(), "end": m.end()}
        for m in EMPLOYEE_ID_RE.finditer(text)
    ]

agent = create_agent(
    model="gpt-5",
    tools=[...],
    middleware=[
        PIIMiddleware("employee_id", detector=employee_id_detector, strategy="hash"),
    ],
)
```

The returned dicts must have `text`, `start`, `end` keys; PIIMiddleware handles the rest.

---

## 3. Built-in guardrail — human-in-the-loop (verified + extra examples)

1. `HumanInTheLoopMiddleware(interrupt_on={...}, description_prefix=...)` pauses execution before a matching tool call and waits for a human decision instead of running it automatically.
2. `interrupt_on` maps tool name → one of:
   - `True` — all decision types allowed (`approve`, `edit`, `reject`, `respond`).
   - `False` — never interrupt for this tool (auto-approved).
   - a dict (`InterruptOnConfig`):
     - `allowed_decisions: list[Literal["approve","edit","reject","respond"]]`
     - `description: str | Callable[[ToolCall, AgentState, Runtime], str]`
     - `when: Callable[[ToolCallRequest], bool]` (langchain≥1.3.3)
3. `description_prefix` sets the default text prefix used to build the interrupt message when a tool doesn't specify its own `description`. Default: `"Tool execution requires approval"`.
4. Requires a checkpointer (`InMemorySaver` for dev, a persistent one for production) — the paused state must survive between the interrupt and the human's eventual response, which can be seconds or days later.
5. Requires a `thread_id` in `config["configurable"]` so the same paused run can be resumed later.
6. Flow:
   - `agent.invoke({"messages": [...]}, config=config)` runs until it hits an interrupt-worthy tool call and returns with `result["__interrupt__"]` populated (contains `action_requests` and `review_configs`).
   - Resume with `agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=config)` using the *same* `thread_id`.
   - **If one AI message triggered multiple tool calls that all require approval, they're grouped into a single interrupt** — `decisions` must be supplied positionally, one per `action_request`, in the same order. **Validate the count** before you call `invoke` again — a mismatch throws.
   - The four decision types:
     - `approve` — run the tool as proposed.
     - `edit` — run the tool with edited args.
     - `reject` — skip execution, send a `ToolMessage` carrying your rejection reasoning back to the agent so it can decide its next step.
     - `respond` — return the human's `message` as a successful `ToolMessage` and skip execution. **Only valid for "ask user" style tools** where the human's reply *is* the tool result. Using it to "deny" a side-effecting tool makes the model think the action succeeded.

### 3a. Canonical example (from official docs)

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

@tool
def delete_records(table: str) -> str:
    """Delete all records from a table."""
    return f"deleted rows from {table}"

@tool
def check_status(table: str) -> str:
    """Read-only status check."""
    return f"{table} is healthy"

agent = create_agent(
    model="gpt-5",
    tools=[delete_records, check_status],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "delete_records": {"allowed_decisions": ["approve", "reject"]},
                "check_status": False,
            },
            description_prefix="Tool execution pending approval",
        ),
    ],
)

config = {"configurable": {"thread_id": "session-1"}}
paused = agent.invoke(
    {"messages": [{"role": "user", "content": "Delete old rows from the logs table"}]},
    config=config,
)
print(paused.get("__interrupt__"))

resumed = agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config,
)
print(resumed["messages"][-1].content)
```

### 3b. Extra example — `when` predicate for conditional interrupts (langchain≥1.3.3)

Don't interrupt on every `execute_sql` — only on write queries. Predicate returns `True` to interrupt, `False` to auto-approve:

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware

def is_write_query(req) -> bool:
    # req is a ToolCallRequest; .tool_call has the model-proposed args.
    sql = (req.tool_call.get("args", {}).get("query") or "").lstrip().lower()
    return sql.startswith(("insert", "update", "delete", "drop", "alter"))

def writes_outside_workspace(req) -> bool:
    path = req.tool_call.get("args", {}).get("path", "")
    return not path.startswith("/workspace/")

agent = create_agent(
    model="gpt-5",
    tools=[execute_sql, write_file, read_file],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "execute_sql": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "when": is_write_query,            # reads are auto-approved
                },
                "write_file": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "when": writes_outside_workspace,  # only risky paths interrupt
                },
            },
        ),
    ],
)
```

> **Note:** when the predicate returns `False`, that call runs without interrupting. Calls that evaluate to `False` are never added to the interrupt batch, so a reviewer only sees the actions that need a decision.

### 3c. Extra example — `description` as a callable for a richer interrupt UI

The default description is fine for a CLI, but a UI usually wants the args pre-formatted, a link to the entity being changed, a diff vs. a known-safe baseline, etc. Pass a callable:

```python
import json
from langchain.agents.middleware import HumanInTheLoopMiddleware

def format_sql_request(tool_call, state, runtime) -> str:
    args = tool_call["args"]
    return (
        f"⚠️  The agent wants to run this SQL:\n\n"
        f"```sql\n{args.get('query', '').strip()}\n```\n\n"
        f"Target DB: `{args.get('database', '?')}`\n"
        f"Conversation: `{runtime.context.get('session_id', 'unknown')}`\n"
    )

agent = create_agent(
    model="gpt-5",
    tools=[execute_sql],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "execute_sql": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "description": format_sql_request,   # callable, not string
                    "when": is_write_query,
                },
            },
            description_prefix="Tool execution requires approval",
        ),
    ],
)
```

### 3d. Extra example — the multi-tool-call grouping gotcha (defensive resume)

When one AI message produces, say, three tool calls and two of them are interrupt-worthy, they ship as **one** `__interrupt__` payload with two `action_requests`. Your resume code must supply exactly N decisions in order — otherwise the middleware throws.

```python
paused = agent.invoke(
    {"messages": [{"role": "user", "content": "Email alice, then bob, then delete the draft"}]},
    config=config,
)

interrupt = paused["__interrupt__"][0].value
action_requests = interrupt["action_requests"]   # list, ordered

assert all(a["name"] in ("send_email", "delete_draft") for a in action_requests)

# Build decisions positionally, one per action_request, same order.
decisions = []
for action in action_requests:
    if action["name"] == "send_email":
        decisions.append({"type": "approve"})
    else:                                           # delete_draft
        decisions.append({
            "type": "reject",
            "message": "Drafts need two-person review — escalate manually.",
        })

assert len(decisions) == len(action_requests)      # never skip this check

resumed = agent.invoke(
    Command(resume={"decisions": decisions}),
    config=config,
)
```

> **Real-world failure mode:** an approval UI shows a card per action but the resume code assumes a 1:1 single-call scenario. When the model batches tool calls, decisions get out of order and the wrong edit lands on the wrong tool. Always drive resume from `action_requests`, never from UI state alone.

### 3e. Extra example — `respond` for an "ask user" placeholder tool

The cleanest pattern when a tool is intentionally a stand-in for human input (clarification, preference, "what color?"):

```python
@tool
def ask_user(question: str) -> str:
    """Ask the user a clarifying question. The human's reply is the result."""
    raise NotImplementedError("intercepted by HumanInTheLoopMiddleware")

agent = create_agent(
    model="gpt-5",
    tools=[ask_user, ...],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "ask_user": {"allowed_decisions": ["respond"]},  # only respond
            },
        ),
    ],
)

# 1. Agent calls ask_user("What color?")
paused = agent.invoke(
    {"messages": [{"role": "user", "content": "I want a new notebook"}]},
    config=config,
)

# 2. You show the question in the UI, capture the human's reply.
# 3. Resume with respond — the human's message becomes the ToolMessage.
resumed = agent.invoke(
    Command(resume={"decisions": [{"type": "respond", "message": "Blue."}]}),
    config=config,
)
# The agent now sees: ToolMessage(content="Blue.") and continues.
```

> **Don't confuse `respond` with `reject`.** Both skip tool execution, but `respond` returns the human's message as a **successful** tool result (so the model treats it as "I got my answer"), while `reject` returns your message as feedback that the model uses to decide its next move.

---

## 4. Custom guardrail — `before_agent` / `before_model` (deterministic, session- or request-level)

1. `before_agent` runs once per invocation, before any model or tool call — the cheapest place to reject a request outright. Good fit: auth checks, rate limits, banned-keyword filters, anything that only needs the initial user message.
2. `before_model` runs before **every** model call, so it sees the latest message in a multi-turn conversation. Use it for mid-conversation injection detection, per-turn policy enforcement, and any rule that needs to look at the actual user input for that turn.
3. Must declare `@hook_config(can_jump_to=[...])` (or `can_jump_to=[...]` on the decorator) to be allowed to short-circuit via `jump_to`.
4. Return `{"messages": [...], "jump_to": "end"}` to stop the run and hand back a canned response instead of ever calling the model.
5. Cheaper than a model-based check because it runs before any LLM call is made at all — no tokens spent on a request you're going to reject anyway.

### 4a. The original `before_agent` example is fine — class form

```python
from typing import Any
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langgraph.runtime import Runtime

class KeywordFilter(AgentMiddleware):
    def __init__(self, banned: list[str]):
        super().__init__()
        self.banned = [b.lower() for b in banned]

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None
        first = state["messages"][0]
        if first.type != "human":
            return None
        content = str(first.content).lower()
        if any(term in content for term in self.banned):
            return {
                "messages": [{"role": "assistant",
                              "content": "I can't help with that request."}],
                "jump_to": "end",
            }
        return None
```

### 4b. Extra example — same filter, decorator form (more concise)

The decorator form is preferred in new code. It produces an `AgentMiddleware` subclass for you.

```python
from langchain.agents.middleware import before_agent, AgentState
from langgraph.runtime import Runtime
from typing import Any

BANNED = ["hack", "exploit", "malware"]

@before_agent(can_jump_to=["end"])
def keyword_filter(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    if not state["messages"]:
        return None
    first = state["messages"][0]
    if first.type != "human":
        return None
    content = str(first.content).lower()
    if any(b in content for b in BANNED):
        return {
            "messages": [{"role": "assistant",
                          "content": "I can't help with that request."}],
            "jump_to": "end",
        }
    return None

agent = create_agent(model="gpt-5", tools=[], middleware=[keyword_filter])
```

### 4c. Extra example — `before_model` for mid-conversation prompt-injection detection

`before_agent` only sees the first message. A multi-turn attack might bury injection in turn 3, or sneak it into a tool result the model reflects back as context. `before_model` runs on **every** call, so it can catch this.

```python
import re
from langchain.agents.middleware import before_model, AgentState
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.runtime import Runtime
from typing import Any

# Heuristic — for production, swap in a classifier.
INJECTION_PATTERNS = [
    r"ignore (all|previous|above) instructions",
    r"you are now .*without restrictions",
    r"system\s*:\s*",                                  # fake system turns
    r"<\|im_start\|>system",                           # role tags leaking in
    r"reveal (the )?(system prompt|instructions)",
]

def looks_like_injection(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in INJECTION_PATTERNS)

@before_model(can_jump_to=["end"])
def injection_firewall(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    # Only inspect the *newest* human turn; older context is "blessed" already.
    latest_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    if latest_human and looks_like_injection(str(latest_human.content)):
        return {
            "messages": [{
                "role": "assistant",
                "content": "I can't process that request — it looks like a "
                           "prompt-injection attempt. Please rephrase.",
            }],
            "jump_to": "end",
        }
    return None
```

> **Why not `before_agent`?** Because `before_agent` only runs *once* — at the start. In a multi-turn agent, a turn-1 clean request that includes a turn-3 injection would slip through. If you need every-turn enforcement, `before_model` is the hook.

---

## 5. Custom guardrail — `after_agent` (output-level, model-based)

1. `after_agent` runs once, after the agent has fully finished, on the complete final response — the natural place for output quality or safety checks that need to see the whole answer.
2. Good fit: LLM-as-judge safety scoring, compliance scans, tone/quality checks — anything nuanced enough that regex won't catch it.
3. A separate, usually smaller/cheaper model is typically used for the judgment call, kept distinct from the main agent model.
4. **Two valid patterns** for replacing an unsafe final answer:
   - **In-place mutation** (original notes, still documented but relies on `AIMessage.content` mutability — not guaranteed for content-block-style messages):
     ```python
     if "UNSAFE" in verdict.content:
         last_message.content = "I can't provide that response. Please rephrase your request."
     return None
     ```
   - **Dict-return with `jump_to="end"`** (the modern, recommended form for new code — see 5b below).
5. Trade-off versus `before_agent`: runs after the full (possibly expensive, multi-tool-call) agent loop has already executed, so it can't prevent token/tool spend — it only prevents the unsafe answer from reaching the user.

### 5a. The original in-place mutation example (still in the official docs, but use with care)

```python
from typing import Any
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.messages import AIMessage
from langgraph.runtime import Runtime

class SafetyCheck(AgentMiddleware):
    def __init__(self, judge_model):
        super().__init__()
        self.judge_model = judge_model

    @hook_config(can_jump_to=["end"])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None
        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            return None
        prompt = (f"Evaluate if this response is safe and appropriate. "
                  f"Reply only SAFE or UNSAFE.\n\nResponse: {last.content}")
        verdict = self.judge_model.invoke([{"role": "user", "content": prompt}])
        if "UNSAFE" in verdict.content:
            last.content = "I can't provide that response. Please rephrase your request."
        return None
```

### 5b. Extra example — same logic, decorator form, dict-return pattern (recommended)

Cleaner, doesn't depend on mutating message objects, and works for content-block-style messages too:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import after_agent, AgentState
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from typing import Any

def make_safety_guardrail(judge_model):
    @after_agent(can_jump_to=["end"])
    def safety_guardrail(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None
        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            return None

        verdict = judge_model.invoke([{
            "role": "user",
            "content": (f"Reply only SAFE or UNSAFE.\n\nResponse: {last.content}"),
        }])

        if "UNSAFE" in (verdict.content or "").upper():
            return {
                "messages": [AIMessage(content=(
                    "I can't provide that response. Please rephrase your request."
                ))],
                "jump_to": "end",
            }
        return None
    return safety_guardrail

main_model    = init_chat_model("gpt-5")         # the main agent
judge_model   = init_chat_model("gpt-5-mini")    # cheap scorer

agent = create_agent(
    model=main_model,
    tools=[],
    middleware=[make_safety_guardrail(judge_model)],
)
```

> The decorator form is also easier to test in isolation (you can call the inner function with a state and runtime directly), and you can parameterize it via closure for cheap re-use across agents.

### 5c. Extra example — async judge

If your agent runs async (and most production agents do), the judge call needs to be awaited too. Otherwise you block the event loop on a judge that itself takes 200–800ms.

```python
from langchain.agents.middleware import aafter_agent, AgentState
from langgraph.runtime import Runtime
from typing import Any

def make_async_safety_guardrail(judge_model):
    @aafter_agent(can_jump_to=["end"])
    async def safety_guardrail(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None
        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            return None

        verdict = await judge_model.ainvoke([{
            "role": "user",
            "content": f"Reply only SAFE or UNSAFE.\n\nResponse: {last.content}",
        }])

        if "UNSAFE" in (verdict.content or "").upper():
            return {
                "messages": [AIMessage(content=(
                    "I can't provide that response."
                ))],
                "jump_to": "end",
            }
        return None
    return safety_guardrail
```

> **Implementation note:** the sync version (`after_agent`) is **not** called on the async path. If your agent is invoked with `ainvoke`, you must register the async variant. This is a footgun in mixed sync/async codebases.

---

## 6. `wrap_model_call` and `wrap_tool_call` — the "around the call" hooks

These are different from the node-style hooks above. The pattern is:

```python
@wrap_model_call
def my_middleware(request, handler):
    # mutate request (optional)
    response = handler(request)        # actually runs the inner model call
    # mutate response (optional)
    return response
```

`request` carries the current `state` and `runtime`. `handler(request)` is the call to the *next* middleware in the chain (or the model itself, if you're the outermost). You can short-circuit by **not** calling `handler` and returning a fabricated `ModelResponse` instead — but be careful, because skipping the model call means you've effectively decided the answer yourself.

### 6a. Extra example — output DLP that survives streaming (no `PIIMiddleware`)

Use case: you have a custom redaction rule that PIIMiddleware doesn't ship (e.g. project-internal codenames, customer-account-format strings). Implementing it in `wrap_model_call` is correct for **state-level** scrubbing, but be aware it doesn't catch streamed tokens that already left the wire.

```python
from dataclasses import replace
from langchain.agents.middleware import (
    AgentMiddleware, ModelRequest, ModelResponse,
)
from langchain_core.messages import AIMessage

class OutputFirewall(AgentMiddleware):
    """Scrub a custom sensitive pattern from model output."""

    def wrap_model_call(self, request, handler):
        response = handler(request)
        scrubbed = []
        for message in response.result:
            if isinstance(message, AIMessage) and isinstance(message.content, str):
                clean = message.content.replace("PROJECT_CODENAME", "[REDACTED]")
                if clean != message.content:
                    message = message.model_copy(update={"content": clean})
                scrubbed.append(message)
            else:
                scrubbed.append(message)
        return replace(response, result=scrubbed)

    async def awrap_model_call(self, request, handler):
        response = await handler(request)
        # same scrubbing as the sync version
        ...
        return response
```

### 6b. Extra example — `wrap_tool_call` for business-rule enforcement

When a rule is about *what a specific tool is being asked to do*, you can intercept the call and either edit, reject, or audit it.

```python
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.runtime import Runtime

class TransferLimitGuard(AgentMiddleware):
    """Block transfers over $10,000 unless tagged 'compliance_approved'."""

    async def awrap_tool_call(self, request, handler):
        call = request.tool_call
        if call["name"] == "initiate_transfer":
            amount = call["args"].get("amount", 0)
            if amount > 10_000 and not call["args"].get("compliance_approved"):
                # Option A: reject and return a synthetic ToolMessage
                return ToolMessage(
                    content="Transfer rejected: amount exceeds $10k without "
                            "compliance approval. Ask a human to approve.",
                    tool_call_id=call["id"],
                )
                # Option B: forward the call but log the attempt
                # return await handler(request)
        return await handler(request)
```

> **Why `wrap_tool_call` and not `HumanInTheLoopMiddleware`?** Because the rule is binary and you want it enforced automatically — no human in the loop. If you also wanted a human to optionally review borderline cases, layer `HumanInTheLoopMiddleware` on top: it intercepts *before* `wrap_tool_call` runs.

---

## 7. Layering multiple guardrails (verified + ordering)

1. Guardrails are just middleware, so they compose by list order and follow the same execution rules: `before_agent` hooks run first-to-last, `after_agent` hooks run last-to-first.
2. A layered stack typically goes (cheapest/most-certain first, expensive/judgement-based last):
   1. **Rate limit / auth** (`before_agent`) — reject unauthenticated or abusive traffic before any work.
   2. **Deterministic input filter** (`before_agent`) — banned keywords, simple injection patterns.
   3. **PII redaction on input** (`PIIMiddleware` with `apply_to_input=True`) — strip sensitive user input.
   4. **PII redaction on output + tool results** (`PIIMiddleware` with `apply_to_output=True`, `apply_to_tool_results=True`) — strip what the model and your tools produced. This is two separate `PIIMiddleware` instances if you want different strategies per direction.
   5. **Tool-call business rules** (`wrap_tool_call`) — enforce per-tool invariants.
   6. **Human approval for sensitive tools** (`HumanInTheLoopMiddleware` with `interrupt_on`) — gate destructive actions.
   7. **Output safety** (`after_agent` with a judge model) — last-mile quality/safety check.
3. Ordering has real effect, not just style: put the cheap deterministic filter first so an obviously bad request never reaches the model at all; put the human-approval and model-based checks later since they're only relevant once the agent has actually produced something to check.
4. Two `PIIMiddleware` instances for the same `pii_type` but different `apply_to_*` flags (one for input, one for output) is the documented pattern — not a single instance with both flags set, if you want independent strategies per direction.

### 7a. The original layered example is correct. Extra layered example — DLP + HITL + judge

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    PIIMiddleware, HumanInTheLoopMiddleware,
    AgentMiddleware, AgentState, hook_config, after_agent,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langchain_core.messages import AIMessage, HumanMessage
from typing import Any


# 1. Keyword filter — deterministic, before model
class KeywordFilter(AgentMiddleware):
    def __init__(self, banned: list[str]):
        super().__init__()
        self.banned = [b.lower() for b in banned]
    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        first = state["messages"][0] if state["messages"] else None
        if first and first.type == "human" and any(
            b in str(first.content).lower() for b in self.banned
        ):
            return {"messages": [{"role": "assistant",
                                  "content": "Blocked by policy."}],
                    "jump_to": "end"}
        return None


# 2. Output safety — model-based, after agent
def make_output_safety(judge_model):
    @after_agent(can_jump_to=["end"])
    def _guard(state, runtime):
        if not state["messages"]:
            return None
        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            return None
        verdict = judge_model.invoke([{
            "role": "user",
            "content": f"Reply only SAFE or UNSAFE.\n\nResponse: {last.content}",
        }])
        if "UNSAFE" in (verdict.content or "").upper():
            return {"messages": [AIMessage("I can't provide that response.")],
                    "jump_to": "end"}
        return None
    return _guard


@tool
def send_email(to: str, body: str) -> str:
    """Send an email."""
    return f"sent to {to}"


main_model  = init_chat_model("gpt-5")
judge_model = init_chat_model("gpt-5-mini")

agent = create_agent(
    model=main_model,
    tools=[send_email],
    checkpointer=InMemorySaver(),
    middleware=[
        KeywordFilter(banned=["hack", "exploit"]),                         # 1. input filter
        PIIMiddleware("email", strategy="redact", apply_to_input=True),    # 2. PII in
        PIIMiddleware("email", strategy="redact", apply_to_output=True),   # 3. PII out (+ stream)
        PIIMiddleware("ssn",  strategy="block",  apply_to_tool_results=True),  # 4. PII in tool results
        HumanInTheLoopMiddleware(interrupt_on={"send_email": True}),       # 5. human approval
        make_output_safety(judge_model),                                   # 6. output judge
    ],
)
```

---

## 8. Real-world example — Restaurant support agent (production shape)

This is the canonical layered pattern from a production walkthrough, simplified. It shows how 4-5 guardrails compose around a real customer-facing agent.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    PIIMiddleware, HumanInTheLoopMiddleware, after_agent,
    AgentMiddleware, AgentState, hook_config,
)
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langchain.tools import tool
from typing import Any


# ---- tools --------------------------------------------------------------
@tool
def place_order(item: str, quantity: int) -> str:
    """Place a restaurant food order. No human approval needed."""
    return f"Order confirmed: {quantity} x {item}."

@tool
def get_order(order_id: str) -> dict:
    """Fetch order details by ID."""
    return {"id": order_id, "status": "in_kitchen", "total": 42}

@tool
def process_refund(order_id: str, amount: int, reason: str) -> str:
    """Process a refund for a cancelled order. Requires human approval."""
    return f"REFUNDED: order {order_id}, ${amount}, reason: {reason}"

@tool
def decline_refund(order_id: str, reason: str) -> str:
    """Decline a refund request. Requires human approval."""
    return f"REFUND DECLINED: order {order_id}, reason: {reason}"


# ---- guardrail 1: deterministic content filter --------------------------
class ContentFilter(AgentMiddleware):
    def __init__(self, banned: list[str]):
        super().__init__()
        self.banned = [b.lower() for b in banned]

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None
        first = state["messages"][0]
        if first.type == "human" and any(
            b in str(first.content).lower() for b in self.banned
        ):
            return {"messages": [{"role": "assistant",
                                  "content": "I can't help with that."}],
                    "jump_to": "end"}
        return None


# ---- guardrail 2: output safety (LLM judge) -----------------------------
def make_output_safety(judge):
    @after_agent(can_jump_to=["end"])
    def _guard(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None
        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            return None
        v = judge.invoke([{"role": "user", "content":
            f"Reply only SAFE or UNSAFE.\n\nResponse: {last.content}"}])
        if "UNSAFE" in (v.content or "").upper():
            return {"messages": [AIMessage("I can't provide that response.")],
                    "jump_to": "end"}
        return None
    return _guard


# ---- assembly -----------------------------------------------------------
main_model  = init_chat_model("gpt-5-nano", temperature=0)
judge_model = init_chat_model("gpt-5-nano", temperature=0)

agent = create_agent(
    model=main_model,
    tools=[place_order, get_order, process_refund, decline_refund],
    system_prompt=(
        "You are a restaurant support agent.\n"
        "1. For new orders, call place_order(item, quantity).\n"
        "2. For status checks, call get_order(order_id).\n"
        "3. For refunds: ALWAYS call get_order first. If refundable, call "
        "   process_refund(order_id, amount, reason). Otherwise call "
        "   decline_refund(order_id, reason)."
    ),
    checkpointer=InMemorySaver(),
    middleware=[
        # Layer 1: deterministic content filter
        ContentFilter(banned=["abuse", "fraud", "phishing"]),
        # Layer 2: PII protection (input, output, tool results)
        PIIMiddleware("email",       strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask",
                      apply_to_input=True, apply_to_output=True,
                      apply_to_tool_results=True),
        PIIMiddleware("phone",       strategy="redact", apply_to_tool_results=True),
        # Layer 3: human approval for the two refund tools
        HumanInTheLoopMiddleware(
            interrupt_on={
                "process_refund": {"allowed_decisions": ["approve", "edit", "reject"]},
                "decline_refund": {"allowed_decisions": ["approve", "edit", "reject"]},
                # place_order, get_order auto-approved (no entry)
            },
            description_prefix="Refund action pending approval",
        ),
        # Layer 4: LLM-judge output safety
        make_output_safety(judge_model),
    ],
)
```

The order is the design:
- **Cheap, certain checks first** (keyword filter) — bad traffic never hits the model.
- **PII protection everywhere it could leak** — input, output, and tool results.
- **Human approval only on the destructive tools** — refund actions.
- **Expensive but nuanced check last** — output safety judge. The model has already done its work; we just decide whether to ship the answer.

---

## 9. Decision guide (verified, with notes)

| Situation | Hook | Why |
|---|---|---|
| Catch known bad substrings/patterns cheaply, before any model call | `before_agent`, deterministic, `jump_to="end"` | One-time cost, never invokes the model. |
| Detect emails/cards/IPs/custom-regex PII in user input or model output | `PIIMiddleware`, one instance per type per direction | Built-in covers the common cases; custom detector covers the rest. Use the **stream transformer** (langchain≥1.3.2) for output PII when you also stream the agent. |
| A tool call is destructive, financial, or externally visible (send, delete, deploy) | `HumanInTheLoopMiddleware`; budget for the checkpointer and thread-id plumbing this requires. | The gate sits before the tool runs; side effects can't happen until a human decides. |
| Need nuanced judgment on the final answer that no regex can express (tone, subtle harm, compliance language) | `after_agent` with a separate judge model; accept the extra latency/cost and that it can't stop tool-call spend that already happened. | Use a cheaper, separate model for the judge. Prefer the decorator + dict-return form for new code. |
| Need injection/attack detection mid-conversation, not just at start/end | `wrap_model_call` / `wrap_tool_call` (or `before_model` for inspection-only) rather than only at the edges — `before_agent`/`after_agent` only see the first and last message, not what happens in between. | Wrap the call if you need to mutate; use node-style `before_model` if you just need to inspect and possibly short-circuit. |
| Multiple guardrail concerns at once | Stack them in the middleware list, cheapest/most-certain first, and remember ordering changes both cost and behavior, not just readability. | Composition is just list order; the engine wires execution rules automatically. |
| Rule is about a specific tool's *arguments* or *behavior*, not about content at the edges | `wrap_tool_call` | Most natural place to encode per-tool invariants; can also synthesize a `ToolMessage` to reject without calling the tool. |
| Need a tool that's really a placeholder for human input (e.g. `ask_user`) | `HumanInTheLoopMiddleware` with `allowed_decisions=["respond"]` | The human's `message` becomes the tool result, the tool is never executed. |

---

## 10. Things to verify before relying on this (verified, updated)

| # | Original concern | Verified status |
|---|---|---|
| 1 | "respond" decision type | ✅ Confirmed first-class in current docs (Python & JS). Returns the human's `message` as a successful `ToolMessage` and skips tool execution. **Use it only for "ask user" style tools**, not to deny side-effecting tools — denial is `reject`. |
| 2 | Positional, grouped-decision resume contract | ✅ Confirmed. "Decisions must be supplied positionally, one per `action_request`, in the same order." Real production gotcha when one AI message triggers multiple tool calls. Write resume code defensively. |
| 3 | `description_prefix` and per-tool `description` overrides | ✅ Confirmed. `description_prefix` default: `"Tool execution requires approval"`. Per-tool `description` can be a **string or a callable** `(tool_call, state, runtime) -> str`. Conditional `when` predicate (langchain≥1.3.3) is also supported. |
| 4 | `after_agent` in-place mutation pattern | ✅ Still in the official docs. ⚠️ **But** the decorator + dict-return form is now the recommended pattern for new code. The in-place form relies on `AIMessage.content` mutability — not guaranteed for content-block-style messages (e.g. Anthropic). |
| 5 | `ChatNebius` model names illustrative | ✅ Correct. Always verify against the live Nebius Token Factory model list — model IDs rotate. |
| 6 | `langchain-nebius` × HITL | ⚠️ Not directly documented. The middleware is provider-agnostic; it should work, but test in your pipeline. Stream transformer is a host-side concern, not model-side, so it should be unaffected by the model provider. |

### Additional version notes worth flagging

- **langchain≥1.3.2** — PII stream transformer (`apply_to_output=True` scrubs streamed wire output, not just state).
- **langchain≥1.3.3** — Conditional HITL via `when` predicate.
- **langchain≥1.3.14** — `ToolErrorMiddleware` (catches tool exceptions, converts to `ToolMessage` for the model to recover from).
- Older versions work for the core patterns but won't have these specific features.

---

## 11. Quick reference — copy/paste snippets

### A. PII DLP for a customer-facing agent

```python
middleware=[
    PIIMiddleware("email",       strategy="redact", apply_to_input=True),
    PIIMiddleware("credit_card", strategy="mask",
                  apply_to_input=True, apply_to_output=True,
                  apply_to_tool_results=True),
    PIIMiddleware("phone",       strategy="redact", apply_to_tool_results=True),
    PIIMiddleware("api_key",     detector=r"sk-[a-zA-Z0-9]{32}",
                  strategy="block",
                  apply_to_input=True, apply_to_output=True,
                  apply_to_tool_results=True),
]
```

### B. Human approval for destructive tools only

```python
middleware=[
    HumanInTheLoopMiddleware(
        interrupt_on={
            "send_email":       True,                                       # default — all decisions
            "execute_sql":      {"allowed_decisions": ["approve", "reject"]},
            "write_file":       {"allowed_decisions": ["approve", "edit", "reject"],
                                 "when": writes_outside_workspace},
            "ask_user":         {"allowed_decisions": ["respond"]},
            "read_only_tool":   False,                                     # auto-approve
        },
        description_prefix="Action pending approval",
    ),
]
```

### C. Deterministic + LLM-judge combo

```python
@before_agent(can_jump_to=["end"])
def keyword_filter(state, runtime):
    # ... same as 4b
    return None

def make_safety(judge):
    @after_agent(can_jump_to=["end"])
    def _g(state, runtime):
        # ... same as 5b
        return None
    return _g

middleware=[keyword_filter, make_safety(judge_model)]
```

---

## Sources

- LangChain Guardrails (Python): https://docs.langchain.com/oss/python/langchain/guardrails
- LangChain Guardrails (JS): https://docs.langchain.com/oss/javascript/langchain/guardrails
- Human-in-the-Loop (Python): https://docs.langchain.com/oss/python/langchain/human-in-the-loop
- Human-in-the-Loop (JS): https://docs.langchain.com/oss/javascript/langchain/human-in-the-loop
- Custom middleware: https://docs.langchain.com/oss/python/langchain/middleware/custom
- Built-in middleware: https://docs.langchain.com/oss/python/langchain/middleware/built-in
- `PIIMiddleware` reference: https://reference.langchain.com/python/langchain/agents/middleware/pii/PIIMiddleware
- `HumanInTheLoopMiddleware` reference: https://reference.langchain.com/python/langchain/agents/middleware/human_in_the_loop/HumanInTheLoopMiddleware
- `@after_agent` decorator: https://reference.langchain.com/python/langchain/agents/middleware/types/after_agent
- `@wrap_model_call`: https://reference.langchain.com/python/langchain/agents/middleware/types/wrap_model_call
- Event streaming & transformers: https://docs.langchain.com/oss/python/langchain/event-streaming
- Real-world production post (Soba Labs): https://sobalabs.ai/blog/langgraph-interrupts-production/
- Real-world pattern (CopilotKit guardrails): https://docs.copilotkit.ai/langgraph-fastapi/guardrails
- Real-world pattern (Restaurant support, LinkedIn): https://www.linkedin.com/pulse/from-ai-demo-production-designing-guardrails-agents-langchain-danial-daogc
- LangChain blog — "How Middleware Lets You Customize Your Agent Harness": https://www.langchain.com/blog/how-middleware-lets-you-customize-your-agent-harness
- Crash course: https://krishcnaik.substack.com/p/guardrails-with-langchain-a-complete
- DeepWiki middleware reference: https://deepwiki.com/langchain-ai/langchain/4.1-agent-system-with-middleware
