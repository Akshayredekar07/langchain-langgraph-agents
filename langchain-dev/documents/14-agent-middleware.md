# **LangChain Agent Middleware — Notes**

Source: LangChain v1 docs (`middleware/overview`, `middleware/built-in`, `middleware/custom`), cross-checked against LangChain 1.0 GA (Oct 22, 2025) release notes and current PyPI/package status as of Sep 2026.

---

## **1. What middleware is**

1. Middleware is a set of hooks that run at defined points inside the `create_agent` loop (model call → tool call → repeat until no more tool calls).
2. It is the mechanism LangChain v1 introduced to replace ad-hoc subclassing of the old `AgentExecutor` — customization now happens by composing hook functions/classes, not by overriding agent internals.
3. Typical uses: logging/analytics, prompt/tool/output transformation, retries and fallbacks, early termination, rate limiting, guardrails, PII redaction.
4. Middleware is not a separate runtime — `create_agent` compiles to a LangGraph graph, and middleware hooks are nodes/wrappers inside that graph. This means a middleware-equipped agent can be dropped into a larger `StateGraph` as a node or subgraph and every hook still fires.
5. Passed as a list to `create_agent(..., middleware=[...])`. Order in the list matters (see execution order, Section 8).

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
        SummarizationMiddleware(model=model, trigger=("tokens", 3000), keep=("messages", 15)),
    ],
)

result = agent.invoke({"messages": [{"role": "user", "content": "Explain what middleware does in one line."}]})
print(result["messages"][-1].content)
```

---

## **2. The two hook styles**

1. **Node-style hooks** run sequentially at a fixed point and return a dict that gets merged into state via the graph's reducers.
   - `before_agent` — once, before the agent starts.
   - `before_model` — before every model call.
   - `after_model` — after every model response.
   - `after_agent` — once, after the agent finishes.
2. **Wrap-style hooks** run *around* a call and receive a `handler` you must invoke yourself, so you control whether the underlying call happens 0, 1, or N times.
   - `wrap_model_call` — around each model call.
   - `wrap_tool_call` — around each tool call.
3. Node-style hooks are for observation/validation/state bookkeeping. Wrap-style hooks are for control flow — retries, fallback models, caching, short-circuiting, request/response rewriting.
4. Both decorator (`@before_model`, `@wrap_model_call`, ...) and class-based (`AgentMiddleware` subclass) forms exist. Decorators are for one hook; classes are for multiple hooks, shared config, or sync+async pairs.

```python
from typing import Any
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, after_model, AgentState
from langgraph.runtime import Runtime
from langchain_nebius import ChatNebius

load_dotenv()


@before_model
def log_before(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"messages so far: {len(state['messages'])}")
    return None


@after_model
def log_after(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"model said: {state['messages'][-1].content[:80]}")
    return None


agent = create_agent(
    model=ChatNebius(model="Qwen/Qwen3-235B-A22B-Instruct-2507"),
    tools=[],
    middleware=[log_before, log_after],
)

agent.invoke({"messages": [{"role": "user", "content": "Say hi in five words."}]})
```

---

## **3. Wrap-style hooks in depth**

1. A wrap hook receives `(request, handler)`. Call `handler(request)` to proceed normally; skip it to short-circuit; loop over it for retries.
2. For `wrap_model_call`, `request` is a `ModelRequest` (has `.messages`, `.system_message`, `.model`, `.tools`, and `.override(...)` to produce a modified copy) and `handler` returns a `ModelResponse`.
3. For `wrap_tool_call`, `request` is a `ToolCallRequest` (has `.tool_call` with `name`/`args`/`id`) and `handler` returns a `ToolMessage` or a `Command`.
4. `request.override(...)` is the standard way to change the model, system prompt, or tool list for a single call without mutating agent-wide config.
5. Multiple wrap hooks nest like function calls: the first middleware in the list is outermost and wraps every other middleware plus the model/tool call itself.

```python
from typing import Callable
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain_nebius import ChatNebius
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

primary = ChatNebius(model="Qwen/Qwen3-235B-A22B-Instruct-2507")
backup = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")


@wrap_model_call
def fallback_on_error(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    try:
        return handler(request)
    except Exception as exc:
        print(f"primary model failed ({exc}); retrying with backup")
        return handler(request.override(model=backup))


agent = create_agent(model=primary, tools=[], middleware=[fallback_on_error])
result = agent.invoke({"messages": [{"role": "user", "content": "What is the capital of Japan?"}]})
print(result["messages"][-1].content)
```

---

## **4. State updates from hooks**

1. Node-style hooks update state simply by returning a dict — the graph's reducers merge it.
2. Wrap-style hooks cannot just return a dict, because their return value *is* the model/tool response. To attach a state update alongside it, `wrap_model_call` returns an `ExtendedModelResponse(model_response=..., command=Command(update={...}))`. `wrap_tool_call` returns a `Command` directly.
3. Custom state fields must be declared via a `state_schema` (a `TypedDict` extending `AgentState`) — pass it to `@before_model(state_schema=...)` or set `state_schema` as a class attribute.
4. When several middleware layers return `ExtendedModelResponse` commands: message updates are additive (both get appended); for non-reducer scalar fields, the outermost middleware's value wins on conflict; if an outer middleware retries (calls `handler()` more than once), only the last inner command survives.

```python
from typing import Any, Callable
from typing_extensions import NotRequired
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentState,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    wrap_model_call,
)
from langgraph.types import Command
from langchain_nebius import ChatNebius

load_dotenv()


class CallState(AgentState):
    model_calls: NotRequired[int]


@wrap_model_call(state_schema=CallState)
def count_calls(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ExtendedModelResponse:
    response = handler(request)
    current = request.state.get("model_calls", 0)
    return ExtendedModelResponse(model_response=response, command=Command(update={"model_calls": current + 1}))


agent = create_agent(model=ChatNebius(model="Qwen/Qwen3-235B-A22B-Instruct-2507"), tools=[], middleware=[count_calls])
result = agent.invoke({"messages": [{"role": "user", "content": "Hello"}], "model_calls": 0})
print(result["model_calls"])
```

---

## **5. Agent jumps (early exit / rerouting)**

1. Any node-style hook can return `{"jump_to": target}` to skip the normal next step.
2. Valid targets: `'end'` (finish immediately), `'tools'` (go straight to the tools node), `'model'` (go back to the model node).
3. Must declare intent with `@hook_config(can_jump_to=[...])` (decorator) or the same on the class method — LangGraph validates jumps against this declaration at compile time.
4. Common use: hard caps on message count, blocked-content filters, cutting a run short after a policy violation is detected in the model's own output.

```python
from typing import Any
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import after_model, hook_config, AgentState
from langchain.messages import AIMessage
from langgraph.runtime import Runtime
from langchain_nebius import ChatNebius

load_dotenv()


@after_model
@hook_config(can_jump_to=["end"])
def stop_on_flag(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    last = state["messages"][-1]
    if "CANNOT_HELP" in str(last.content):
        return {"messages": [AIMessage("Stopping this run per policy.")], "jump_to": "end"}
    return None


agent = create_agent(model=ChatNebius(model="Qwen/Qwen3-235B-A22B-Instruct-2507"), tools=[], middleware=[stop_on_flag])
agent.invoke({"messages": [{"role": "user", "content": "Just answer normally."}]})
```

---

## **6. Execution order with multiple middleware**

1. Given `middleware=[m1, m2, m3]`:
   - `before_agent`: `m1 → m2 → m3` (declaration order).
   - `before_model`: `m1 → m2 → m3`.
   - `wrap_model_call`: nests `m1(m2(m3(model)))` — `m1` is outermost.
   - `after_model`: `m3 → m2 → m1` (reverse order).
   - `after_agent`: `m3 → m2 → m1` (reverse order).
2. Practical implication: put critical/gatekeeping middleware (limits, PII blocking) first in the list so it wraps everything else.
3. This ordering is fixed by the framework, not configurable per hook — the only lever is list order.

```python
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, after_model, AgentState
from langgraph.runtime import Runtime
from langchain_nebius import ChatNebius

load_dotenv()


def make_logger(tag: str):
    @before_model
    def before(state: AgentState, runtime: Runtime):
        print(f"before:{tag}")
        return None

    @after_model
    def after(state: AgentState, runtime: Runtime):
        print(f"after:{tag}")
        return None

    return [before, after]


agent = create_agent(
    model=ChatNebius(model="Qwen/Qwen3-235B-A22B-Instruct-2507"),
    tools=[],
    middleware=[*make_logger("first"), *make_logger("second")],
)
agent.invoke({"messages": [{"role": "user", "content": "hi"}]})
```

---

## **7. Custom middleware — class-based**

1. Preferred when you need multiple hooks together, init-time configuration, or separate sync/async implementations (`before_model` + `abefore_model`, etc.).
2. Subclass `AgentMiddleware`; override whichever hook methods you need.
3. Class attributes `state_schema`, `tools`, and `transformers` let a middleware extend agent state, ship its own tools (like `TodoListMiddleware` shipping `write_todos`), or register stream transformers.

```python
from typing import Any
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime
from langchain_nebius import ChatNebius

load_dotenv()


class TokenBudget(AgentMiddleware):
    def __init__(self, max_calls: int = 5):
        super().__init__()
        self.max_calls = max_calls
        self.calls = 0

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        self.calls += 1
        if self.calls > self.max_calls:
            return {"jump_to": "end"}
        return None


agent = create_agent(model=ChatNebius(model="Qwen/Qwen3-235B-A22B-Instruct-2507"), tools=[], middleware=[TokenBudget(max_calls=3)])
agent.invoke({"messages": [{"role": "user", "content": "Count to three."}]})
```

---

## **8. Common custom-middleware patterns**

1. **Dynamic system prompt** — read `request.system_message.content_blocks`, append a block, rebuild with `SystemMessage`, pass through `request.override(system_message=...)`.
2. **Dynamic model selection** — inspect `request.messages` / `request.state` and swap `request.override(model=...)`, e.g. cheap model for short conversations, stronger model once context grows.
3. **Dynamic tool selection** — filter `request.tools` at call time based on state/permissions to shorten prompts and improve tool-choice accuracy; all tools must still be registered on `create_agent` up front.
4. **Tool call monitoring** — wrap `wrap_tool_call`, log args/timing/exceptions around `handler(request)`.
5. **Prompt caching (Anthropic-style providers)** — append a content block with `"cache_control": {"type": "ephemeral"}` to `system_message.content_blocks`; only meaningful for providers that support prompt caching.

```python
from typing import Callable
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.messages import SystemMessage
from langchain_nebius import ChatNebius

load_dotenv()

cheap = ChatNebius(model="Qwen/Qwen3-8B")
strong = ChatNebius(model="Qwen/Qwen3-235B-A22B-Instruct-2507")


@wrap_model_call
def route_by_length(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    model = strong if len(request.messages) > 6 else cheap
    return handler(request.override(model=model))


@wrap_model_call
def add_style_note(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    blocks = list(request.system_message.content_blocks) + [{"type": "text", "text": "Answer concisely."}]
    return handler(request.override(system_message=SystemMessage(content=blocks)))


agent = create_agent(model=cheap, tools=[], middleware=[route_by_length, add_style_note])
result = agent.invoke({"messages": [{"role": "user", "content": "What's 12 * 7?"}]})
print(result["messages"][-1].content)
```

---

## **9. Reliability middleware — tool error, tool retry, model retry, model fallback**

1. `ToolErrorMiddleware` converts a raised exception into an error `ToolMessage` the model can see and recover from, instead of crashing the run. Its `on_error` handler decides which exceptions to convert (return string/content) vs. let propagate (return `None`). Requires `langchain>=1.3.14`.
2. `ToolRetryMiddleware` retries a failing tool call with exponential backoff (`max_retries`, `backoff_factor`, `initial_delay`, `max_delay`, `jitter`); `on_failure` controls what happens once retries are exhausted (`'continue'`, `'error'`, or a custom formatter).
3. `ModelRetryMiddleware` is the same idea for model calls instead of tool calls.
4. `ModelFallbackMiddleware(*models)` tries alternative models in order if the primary model call fails — useful for provider outages or cost-based degradation.
5. To combine `ToolRetryMiddleware` with `ToolErrorMiddleware`, place the retry middleware *before* (earlier in the list, i.e. inner) the error middleware, and set `on_failure="error"` on the retry so the final exception actually reaches `ToolErrorMiddleware`.

```python
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware, ToolErrorMiddleware, ModelFallbackMiddleware
from langchain_nebius import ChatNebius
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.tools import tool

load_dotenv()


@tool
def flaky_lookup(query: str) -> str:
    """Look up a value that sometimes fails."""
    import random
    if random.random() < 0.5:
        raise ConnectionError("upstream timed out")
    return f"result for {query}"


def on_error(exc: Exception, request) -> str | None:
    if isinstance(exc, ConnectionError):
        return f"`{request.tool_call['name']}` failed after retries: {type(exc).__name__}"
    return None


agent = create_agent(
    model=ChatNebius(model="Qwen/Qwen3-235B-A22B-Instruct-2507"),
    tools=[flaky_lookup],
    middleware=[
        ToolRetryMiddleware(max_retries=2, retry_on=(ConnectionError,), on_failure="error"),
        ToolErrorMiddleware(on_error=on_error),
        ModelFallbackMiddleware(ChatNVIDIA(model="meta/llama-3.3-70b-instruct")),
    ],
)
result = agent.invoke({"messages": [{"role": "user", "content": "Look up 'pune weather station'."}]})
print(result["messages"][-1].content)
```

---

## **10. Context-window middleware — summarization, context editing**

1. `SummarizationMiddleware` compresses older conversation history into a summary when a trigger condition is met, keeping recent messages intact.
   - `trigger`: `("tokens", n)`, `("messages", n)`, `("fraction", f)` of context size, a dict combining several with AND logic, or a list combining several with OR logic.
   - `keep`: how much recent context to retain, in the same tuple form (usually messages).
   - `fraction` triggers rely on the model's profile data (`langchain>=1.1`); without that, pass an explicit `tokens`/`messages` trigger or supply a manual `profile` on `init_chat_model`.
   - Only compresses text; image/audio/video blocks in older messages are lost once summarized (only recent `keep` messages retain them).
2. `ContextEditingMiddleware` with `ClearToolUsesEdit` clears older tool-call outputs (replacing them with a placeholder) once total tokens exceed `trigger`, always preserving the most recent `keep` tool results. Cheaper than summarization since it doesn't need an extra model call.
3. Use summarization for genuinely long dialogues; use context editing specifically to control bloat from big tool outputs (search results, RAG chunks) without touching the rest of the conversation.

```python
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, ContextEditingMiddleware, ClearToolUsesEdit
from langchain_nebius import ChatNebius

load_dotenv()

model = ChatNebius(model="Qwen/Qwen3-235B-A22B-Instruct-2507")

agent = create_agent(
    model=model,
    tools=[],
    middleware=[
        SummarizationMiddleware(model=model, trigger=("tokens", 3000), keep=("messages", 10)),
        ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=20000, keep=3)]),
    ],
)
result = agent.invoke({"messages": [{"role": "user", "content": "Let's have a long chat about databases."}]})
print(result["messages"][-1].content)
```

---

## **11. Safety and control middleware — human-in-the-loop, call limits, PII**

1. `HumanInTheLoopMiddleware(interrupt_on={...})` pauses execution before specific tool calls for approval/edit/reject. Requires a checkpointer (e.g. `InMemorySaver`) because the graph must persist state across the interrupt. Match tool names exactly as they appear on `.name` (a `@tool`-decorated function's name comes from the function name).
2. `ModelCallLimitMiddleware(thread_limit=..., run_limit=...)` caps model calls per thread (needs a checkpointer) and/or per single invocation. `exit_behavior`: `'end'` (graceful) or `'error'` (raise).
3. `ToolCallLimitMiddleware(thread_limit=..., run_limit=..., tool_name=...)` caps tool calls globally or per named tool. `exit_behavior`: `'continue'` (block excess calls with error messages, model decides how to proceed), `'error'`, or `'end'` (single-tool only).
4. `PIIMiddleware(pii_type, strategy=..., apply_to_input/output/tool_results=...)` detects and handles PII. Built-in types: `email`, `credit_card`, `ip`, `mac_address`, `url`. Strategies: `'block'` (raise), `'redact'`, `'mask'`, `'hash'`. Custom types via a regex string, compiled regex, or a detector function returning `PIIMatch` dicts.
5. `apply_to_output=True` on `PIIMiddleware` also redacts streamed wire output (text deltas, tool args, tool outputs) from `langchain>=1.3.2`, not just the final message.

```python
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, ToolCallLimitMiddleware, PIIMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_nebius import ChatNebius
from langchain.tools import tool

load_dotenv()


@tool
def send_notification(to: str, message: str) -> str:
    """Send a notification to a user."""
    return f"sent to {to}"


agent = create_agent(
    model=ChatNebius(model="Qwen/Qwen3-235B-A22B-Instruct-2507"),
    tools=[send_notification],
    checkpointer=InMemorySaver(),
    middleware=[
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        ToolCallLimitMiddleware(tool_name="send_notification", run_limit=2, exit_behavior="continue"),
        HumanInTheLoopMiddleware(interrupt_on={"send_notification": {"allowed_decisions": ["approve", "edit", "reject"]}}),
    ],
)

config = {"configurable": {"thread_id": "demo-1"}}
result = agent.invoke({"messages": [{"role": "user", "content": "Notify akshay@example.com that the job finished."}]}, config=config)
print(result)
```

---

## **12. Capability middleware — todo list, tool selection, shell, filesystem, subagents**

1. `TodoListMiddleware()` adds a `write_todos` tool plus prompting so the agent plans and tracks multi-step tasks explicitly.
2. `LLMToolSelectorMiddleware(model=..., max_tools=..., always_include=[...])` uses a (usually cheaper) LLM to pick a relevant subset of tools before the main model call — useful once you have 10+ tools.
3. `ProviderToolSearchMiddleware(searchable_tools=[...])` defers tool schemas behind the provider's own server-side tool search, so the model discovers tools on demand instead of receiving every schema up front. Only works with providers that support server-side tool search (Anthropic Claude Sonnet 4+/Opus 4+/Haiku 4.5+, OpenAI gpt-5.5+ per docs) — raises `ValueError` on unsupported providers.
4. `ShellToolMiddleware(workspace_root=..., execution_policy=...)` gives the agent a persistent shell; `execution_policy` (`HostExecutionPolicy`, `DockerExecutionPolicy`, `CodexSandboxExecutionPolicy`) controls isolation. Does not currently support human-in-the-loop interrupts.
5. `FilesystemMiddleware` (from the `deepagents` package) exposes `ls` / `read_file` / `write_file` / `edit_file` tools over graph state by default; wrap with `CompositeBackend` + `StoreBackend` to persist files (e.g. under `/memories/`) across threads instead of per-run only.
6. `SubAgentMiddleware` (also `deepagents`) exposes a `task` tool so the main agent can delegate to named subagents (each with its own model/tools/prompt) or a built-in `general-purpose` subagent, isolating context from the main loop.
7. `FilesystemFileSearchMiddleware(root_path=..., use_ripgrep=True)` adds `glob_search` and `grep_search` tools for code/file exploration.

```python
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware, LLMToolSelectorMiddleware
from langchain_nebius import ChatNebius
from langchain.tools import tool

load_dotenv()

model = ChatNebius(model="Qwen/Qwen3-235B-A22B-Instruct-2507")
selector_model = ChatNebius(model="Qwen/Qwen3-8B")


@tool
def read_file(path: str) -> str:
    """Read a file."""
    return f"contents of {path}"


@tool
def write_file(path: str, content: str) -> str:
    """Write a file."""
    return "written"


agent = create_agent(
    model=model,
    tools=[read_file, write_file],
    middleware=[
        TodoListMiddleware(),
        LLMToolSelectorMiddleware(model=selector_model, max_tools=2),
    ],
)
result = agent.invoke({"messages": [{"role": "user", "content": "Plan and then read config.yaml."}]})
print(result["messages"][-1].content)
```

---

## **13. Testing/utility middleware — LLM tool emulator, rubric grading**

1. `LLMToolEmulator(tools=..., model=...)` replaces real tool execution with LLM-generated plausible outputs — for prototyping/testing agent flow before real tools exist. `tools=None` emulates everything, `[]` emulates nothing, a list emulates only those tools.
2. `RubricMiddleware` (from `deepagents`, beta, requires `deepagents>=0.6.5`) lets the agent self-evaluate output against a rubric and iterate up to `max_iterations` until it's satisfied. API may still change — treat as beta.

```python
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolEmulator
from langchain_nebius import ChatNebius
from langchain.tools import tool

load_dotenv()


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    raise NotImplementedError("not wired up yet")


agent = create_agent(
    model=ChatNebius(model="Qwen/Qwen3-235B-A22B-Instruct-2507"),
    tools=[get_weather],
    middleware=[LLMToolEmulator(tools=["get_weather"])],
)
result = agent.invoke({"messages": [{"role": "user", "content": "What's the weather in Pune?"}]})
print(result["messages"][-1].content)
```

---

## **14. Tracing configuration**

1. Each middleware hook is traced by default (inputs and outputs recorded, e.g. to LangSmith).
2. `trace_policy = TracePolicy(process_inputs=omit_payload)` on a middleware class drops the traced input payload — useful when a middleware's input (e.g. full message history) is uninformative for debugging that hook specifically.
3. `configure_trace_policy(TracePolicy(...))` sets a global default across all middleware; a middleware's own `trace_policy` overrides the global one. Requires `langchain>=1.3.15`.

```python
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, TracePolicy, omit_payload
from langchain_nebius import ChatNebius

load_dotenv()


class QuietLogger(AgentMiddleware):
    trace_policy = TracePolicy(process_inputs=omit_payload)

    def before_model(self, state, runtime):
        print("model about to be called")
        return None


agent = create_agent(model=ChatNebius(model="Qwen/Qwen3-235B-A22B-Instruct-2507"), tools=[], middleware=[QuietLogger()])
agent.invoke({"messages": [{"role": "user", "content": "hi"}]})
```

---

## **Decision guide**

1. Need to just observe/log without changing flow → node-style hook (`before_model`/`after_model`).
2. Need to change what gets sent to the model or which model handles the call → `wrap_model_call` with `request.override(...)`.
3. Need retries/fallback around a flaky tool or model → `ToolRetryMiddleware` / `ModelRetryMiddleware` / `ModelFallbackMiddleware`, not hand-rolled `try/except` in your own tool code.
4. Conversation will run long → `SummarizationMiddleware`; tool outputs specifically are the bloat source → `ContextEditingMiddleware` instead (or in addition).
5. A tool call is sensitive (writes, sends, spends money) → `HumanInTheLoopMiddleware`, and remember the checkpointer requirement.
6. Need hard cost/runaway-loop protection → `ModelCallLimitMiddleware` / `ToolCallLimitMiddleware` before anything else in the list, so the cap wraps other middleware.
7. Handling user data with compliance requirements → `PIIMiddleware`, decide input vs. output vs. tool-result coverage explicitly; don't assume defaults cover all three.
8. 10+ tools and the model picks wrong ones → `LLMToolSelectorMiddleware` (cross-provider) or `ProviderToolSearchMiddleware` (only on providers with native tool search).
9. Multi-step planning tasks → `TodoListMiddleware`; context isolation for a sub-task → `SubAgentMiddleware`; need a real filesystem the agent can read/write/persist → `FilesystemMiddleware` with a `CompositeBackend`.
10. Prototyping before real tool implementations exist → `LLMToolEmulator`.
11. Building your own custom middleware: single hook and no shared state → decorator; multiple hooks, config, or sync+async → `AgentMiddleware` subclass.

---

## **Things to verify before relying on this**

1. `RubricMiddleware` is explicitly marked beta by LangChain and requires `deepagents>=0.6.5` — confirm current `deepagents` version and API stability before depending on it.
2. `ProviderToolSearchMiddleware` provider/model support (Claude Sonnet 4+/Opus 4+/Haiku 4.5+, OpenAI gpt-5.5+) is provider-and-version gated — re-check against current model docs before use, since server-side tool search support changes as providers ship new models.
3. `ToolErrorMiddleware` requires `langchain>=1.3.14`; `PIIMiddleware` streamed-output redaction requires `langchain>=1.3.2`; `trace_policy` requires `langchain>=1.3.15` — confirm your installed `langchain` version satisfies these before assuming the feature is present.
4. `SummarizationMiddleware`'s `fraction`-based triggers depend on model profile data being available for your chat model class; `ChatNebius`/`ChatNVIDIA` profile support was not confirmed in these docs — use explicit `tokens`/`messages` triggers unless verified.
5. `FilesystemMiddleware`, `SubAgentMiddleware`, and `RubricMiddleware` all ship from the separate `deepagents` package, not `langchain` itself — install and version-check `deepagents` independently.
6. `ShellToolMiddleware` explicitly does not support human-in-the-loop interrupts yet per the docs — don't combine those two assuming it works.
7. `langchain-nebius` is a young, externally maintained partner package (PyPI v0.1.x as of writing, not yet merged into the core `langchain` monorepo) — confirm `ChatNebius` model name strings and tool-calling support against current PyPI docs before production use.
8. This notes file assumes LangChain 1.0 GA (`create_agent`, not `langgraph.prebuilt.create_react_agent`, which is deprecated) — verify your environment isn't pinned to a pre-1.0 `langchain` where these imports and hook signatures differ.