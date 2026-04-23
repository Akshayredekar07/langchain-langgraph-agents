# Deep Agents — Customization Notes

> **Package:** `deepagents` | **Function:** `create_deep_agent` | **Docs:** https://docs.langchain.com/oss/python/deepagents/customization
> **Last updated:** 2026-04-18
> **Companion to:** `deepagents-notes.md` (overview)

---

## Overview

`create_deep_agent` is the single factory for all Deep Agent configuration. Every behavioral aspect — model, tools, middleware, subagents, backends, permissions, memory, and structured output — is controlled through its parameters. This document covers all customization options in depth.

**Full signature:**

```python
from deepagents import create_deep_agent

create_deep_agent(
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    subagents: Sequence[SubAgent | CompiledSubAgent | AsyncSubAgent] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    response_format: ResponseFormat | type | dict | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    checkpointer=None,   # required for HITL
    store=None,          # required for StoreBackend / cross-thread memory
    ...
) -> CompiledStateGraph
```

> 🔗 **See also:** [Full API reference](https://reference.langchain.com/python/deepagents/graph/create_deep_agent)

---

## 1. Model

Three ways to pass a model — all equivalent at runtime:

```python
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model

# Option A: provider:model string (simplest)
agent = create_deep_agent(model="anthropic:claude-sonnet-4-6")

# Option B: init_chat_model (use when you need custom params)
model = init_chat_model(model="anthropic:claude-sonnet-4-6", max_retries=10, timeout=120)
agent = create_deep_agent(model=model)

# Option C: direct model class (most explicit)
from langchain_anthropic import ChatAnthropic
model = ChatAnthropic(model="claude-sonnet-4-6")
agent = create_deep_agent(model=model)
```

**Provider string reference:**

| Provider | String format | Example | Env var |
|---|---|---|---|
| Anthropic | `anthropic:{model}` | `anthropic:claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai:{model}` | `openai:gpt-5.4` | `OPENAI_API_KEY` |
| Azure OpenAI | `azure_openai:{model}` | `azure_openai:gpt-5.4` | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` + `OPENAI_API_VERSION` |
| Google Gemini | `google_genai:{model}` | `google_genai:gemini-3.1-pro-preview` | `GOOGLE_API_KEY` |
| AWS Bedrock | `{model}` + `model_provider="bedrock_converse"` | `anthropic.claude-sonnet-4-6` | AWS credentials |
| HuggingFace | `{repo_id}` + `model_provider="huggingface"` | `microsoft/Phi-3-mini-4k-instruct` | `HUGGINGFACEHUB_API_TOKEN` |
| Ollama | `ollama:{model}` | `ollama:devstral-2` | — (local) |

**Azure setup:**

```python
import os
from deepagents import create_deep_agent

os.environ["AZURE_OPENAI_API_KEY"] = "..."
os.environ["AZURE_OPENAI_ENDPOINT"] = "..."
os.environ["OPENAI_API_VERSION"] = "2025-03-01-preview"

agent = create_deep_agent(model="azure_openai:gpt-5.4")
```

**HuggingFace setup:**

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="microsoft/Phi-3-mini-4k-instruct",
    model_provider="huggingface",
    temperature=0.7,
    max_tokens=1024,
)
```

### Connection Resilience

LangChain models auto-retry with exponential backoff. Defaults: **6 retries**, covers 429 + 5xx. Does NOT retry 401/404.

```python
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

agent = create_deep_agent(
    model=init_chat_model(
        model="google_genai:gemini-3.1-pro-preview",
        max_retries=10,   # bump for unreliable networks
        timeout=120,      # seconds; increase for slow/long tasks
    ),
)
```

> 💡 **Tip:** For long-running tasks on unreliable networks, set `max_retries=10–15` and pair with a checkpointer so progress survives failures.

---

## 2. Tools

Custom tools are plain Python functions — type hints and docstrings become the tool schema the LLM sees. The built-in tools (planning, FS, subagents) are always present; custom tools are additive.

```python
import os
from typing import Literal
from tavily import TavilyClient
from deepagents import create_deep_agent

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
) -> dict:
    """Run a web search."""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[internet_search],
)
```

**Tool format options:**

| Format | When to use |
|---|---|
| Plain Python function | Most common — type hints + docstring = schema |
| `@tool` decorated function | When you need explicit name/description overrides |
| `BaseTool` subclass | Complex tools needing lifecycle methods |
| `dict` | Pass raw JSON schema directly |

---

## 3. System Prompt

Deep Agents include a built-in default system prompt (teaches planning, FS tools, subagents). Your custom prompt is **appended** to, not replacing, the default. Override entirely only if you know what you're doing.

```python
from deepagents import create_deep_agent

research_instructions = """\
You are an expert researcher. Your job is to conduct \
thorough research, and then write a polished report.
"""

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    system_prompt=research_instructions,
)
```

---

## 4. Middleware

Middleware intercepts the agent loop — before/after tool calls, before/after agent steps. Deep Agents wire several middleware by default.

### Default Middleware (always active)

| Middleware | Purpose |
|---|---|
| `TodoListMiddleware` | Tracks agent's todo list for task planning |
| `FilesystemMiddleware` | Powers `ls`, `read_file`, `write_file`, `edit_file` tools |
| `SubAgentMiddleware` | Enables the `task` tool for spawning subagents |
| `SummarizationMiddleware` | Auto-compacts long message history to prevent context overflow |
| `AnthropicPromptCachingMiddleware` | Reduces redundant token processing (Anthropic models only) |
| `PatchToolCallsMiddleware` | Fixes message history when tool calls are interrupted |

### Conditional Middleware (activated by config)

| Middleware | Activated when |
|---|---|
| `MemoryMiddleware` | `memory=[...]` is passed |
| `SkillsMiddleware` | `skills=[...]` is passed |
| `HumanInTheLoopMiddleware` | `interrupt_on={...}` is passed |

### Custom Middleware

```python
from langchain.tools import tool
from langchain.agents.middleware import wrap_tool_call
from deepagents import create_deep_agent

@tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is sunny."

call_count = [0]  # use list for mutability in closure

@wrap_tool_call
def log_tool_calls(request, handler):
    """Intercept and log every tool call."""
    call_count[0] += 1
    print(f"[Tool call #{call_count[0]}]: {request.name}")
    result = handler(request)   # execute the tool
    print(f"[Completed #{call_count[0]}]")
    return result

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[get_weather],
    middleware=[log_tool_calls],
)
```

> ⚠️ **Watch out — no shared mutable state in middleware.** Concurrent subagents and parallel tool calls will race on `self.x += 1`. Use graph state instead:

```python
# WRONG — race condition
class MyMiddleware(AgentMiddleware):
    def __init__(self):
        self.counter = 0
    def before_agent(self, state, runtime):
        self.counter += 1   # ← will corrupt under concurrency

# CORRECT — update graph state
class MyMiddleware(AgentMiddleware):
    def before_agent(self, state, runtime):
        return {"counter": state.get("counter", 0) + 1}  # ← thread-safe
```

---

## 5. Subagents

Subagents isolate detailed subtasks in their own context window — the parent agent's history doesn't bleed in. Configured as dicts and passed to `subagents=`.

```python
import os
from tavily import TavilyClient
from deepagents import create_deep_agent

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def internet_search(query: str, max_results: int = 5) -> dict:
    """Run a web search."""
    return tavily_client.search(query, max_results=max_results)

research_subagent = {
    "name": "research-agent",
    "description": "Used to research more in depth questions",  # LLM sees this to decide when to spawn
    "system_prompt": "You are a great researcher.",
    "tools": [internet_search],
    "model": "openai:gpt-5.2",  # Optional — defaults to parent model if omitted
}

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    subagents=[research_subagent],
)
```

**Subagent dict keys:**

| Key | Required | Description |
|---|---|---|
| `name` | Yes | Unique identifier, used in `task` tool calls |
| `description` | Yes | Shown to LLM — determines when to delegate |
| `system_prompt` | No | Specialization instructions for this subagent |
| `tools` | No | Tools available only to this subagent |
| `model` | No | Override model; defaults to parent model |

---

## 6. Backends (Virtual Filesystems)

Backends power the agent's virtual filesystem (`ls`, `read_file`, `write_file`, `edit_file`). Swap backends without changing agent logic.

### Backend Comparison

| Backend | Persistence | Shell (`execute`) | Use Case |
|---|---|---|---|
| `StateBackend` *(default)* | Single thread only | No | Dev/testing |
| `FilesystemBackend` | Local disk | No | Single-machine production |
| `LocalShellBackend` | Local disk | Yes (host shell) | Local coding agent — use with caution |
| `StoreBackend` | Cross-thread (LangGraph store) | No | Multi-session/user persistence |
| `CompositeBackend` | Configurable per route | Depends | Mixed storage strategies |
| Sandbox backends | Cloud-isolated | Yes (sandboxed) | Safe code execution |

### StateBackend (default)

```python
from deepagents import create_deep_agent
from deepagents.backends import StateBackend

# Implicit (default)
agent = create_deep_agent(model="anthropic:claude-sonnet-4-6")

# Explicit
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    backend=StateBackend(),
)
```

### FilesystemBackend

```python
from deepagents.backends import FilesystemBackend

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    backend=FilesystemBackend(root_dir=".", virtual_mode=True),
)
```

### LocalShellBackend

```python
from deepagents.backends import LocalShellBackend

# ⚠️ Grants unrestricted shell access on your host
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    backend=LocalShellBackend(root_dir=".", env={"PATH": "/usr/bin:/bin"}),
)
```

### StoreBackend (cross-thread persistence)

```python
from langgraph.store.memory import InMemoryStore
from deepagents.backends import StoreBackend
from deepagents import create_deep_agent

store = InMemoryStore()  # swap with Redis/Postgres-backed store in production

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    backend=StoreBackend(
        namespace=lambda ctx: (ctx.runtime.context.user_id,),  # isolate per user
    ),
    store=store,
)
```

### CompositeBackend (mixed routing)

```python
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    backend=CompositeBackend(
        default=StateBackend(),          # ephemeral for most files
        routes={
            "/memories/": StoreBackend(),  # persistent for /memories/ path
        }
    ),
    store=InMemoryStore(),
)
```

### Sandbox Backends (isolated code execution)

Use sandboxes when you need the agent to run shell commands safely without touching your host. Each sandbox backend adds the `execute` tool.

**Modal:**

```python
import modal
from langchain_modal import ModalSandbox
from deepagents import create_deep_agent

app = modal.App.lookup("your-app")
modal_sandbox = modal.Sandbox.create(app=app)
backend = ModalSandbox(sandbox=modal_sandbox)

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    backend=backend,
)
try:
    result = agent.invoke({"messages": [{"role": "user", "content": "Write and test a Python package"}]})
finally:
    modal_sandbox.terminate()
```

**Daytona:**

```python
from daytona import Daytona
from langchain_daytona import DaytonaSandbox
from deepagents import create_deep_agent

sandbox = Daytona().create()
backend = DaytonaSandbox(sandbox=sandbox)

agent = create_deep_agent(model="anthropic:claude-sonnet-4-6", backend=backend)
try:
    result = agent.invoke({"messages": [{"role": "user", "content": "Create a small Python package and run pytest"}]})
finally:
    sandbox.stop()
```

**Runloop:**

```python
import os
from runloop_api_client import RunloopSDK
from langchain_runloop import RunloopSandbox
from deepagents import create_deep_agent

client = RunloopSDK(bearer_token=os.environ["RUNLOOP_API_KEY"])
devbox = client.devbox.create()
backend = RunloopSandbox(devbox=devbox)

agent = create_deep_agent(model="anthropic:claude-sonnet-4-6", backend=backend)
try:
    result = agent.invoke({"messages": [{"role": "user", "content": "Run the test suite"}]})
finally:
    devbox.shutdown()
```

---

## 7. Human-in-the-Loop (HITL)

Pause agent execution before specific tool calls and wait for human approval. Requires a **checkpointer** — HITL will not work without one.

```python
from langchain.tools import tool
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver

@tool
def delete_file(path: str) -> str:
    """Delete a file from the filesystem."""
    return f"Deleted {path}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Sent email to {to}"

@tool
def read_file(path: str) -> str:
    """Read a file."""
    return f"Contents of {path}"

checkpointer = MemorySaver()  # REQUIRED

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[delete_file, send_email, read_file],
    checkpointer=checkpointer,
    interrupt_on={
        "delete_file": True,                                      # approve / edit / reject
        "send_email": {"allowed_decisions": ["approve", "reject"]}, # no editing
        "read_file": False,                                       # no interrupt
    },
)
```

**`interrupt_on` values:**

| Value | Behavior |
|---|---|
| `True` | Pause; human can approve, edit args, or reject |
| `False` | Never interrupt |
| `{"allowed_decisions": ["approve", "reject"]}` | Pause; no arg editing allowed |

---

## 8. Skills

Skills are markdown files with domain-specific instructions, templates, and reference info. The agent loads them **lazily** — only when it determines the skill is relevant. This avoids bloating context at startup.

Skills are stored in the backend filesystem under a `/skills/` directory.

```python
from urllib.request import urlopen
from deepagents import create_deep_agent
from deepagents.backends.utils import create_file_data
from langgraph.checkpoint.memory import MemorySaver

# Fetch skill content (or read from local file)
skill_url = "https://raw.githubusercontent.com/langchain-ai/deepagents/refs/heads/main/libs/cli/examples/skills/langgraph-docs/SKILL.md"
with urlopen(skill_url) as response:
    skill_content = response.read().decode("utf-8")

checkpointer = MemorySaver()

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    skills=["/skills/"],           # path prefix — agent discovers all skills under this dir
    checkpointer=checkpointer,
)

result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "What is LangGraph?"}],
        "files": {
            "/skills/langgraph-docs/SKILL.md": create_file_data(skill_content)
        },  # seed StateBackend with the skill file
    },
    config={"configurable": {"thread_id": "my-thread-1"}},
)
```

**Skills vs Tools:**

| | Tools | Skills |
|---|---|---|
| What it is | Python function | Markdown file |
| Loaded | Always | Lazily (on demand) |
| Best for | Actions / API calls | Instructions, templates, reference info |
| Token cost | Low (just schema) | Deferred until needed |

---

## 9. Memory (AGENTS.md)

Pass AGENTS.md files as always-loaded context for the agent. Unlike skills (lazy), memory files are loaded at every invocation.

```python
from urllib.request import urlopen
from deepagents import create_deep_agent
from deepagents.backends.utils import create_file_data
from langgraph.checkpoint.memory import MemorySaver

with urlopen("https://raw.githubusercontent.com/.../AGENTS.md") as response:
    agents_md = response.read().decode("utf-8")

checkpointer = MemorySaver()

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    memory=["/AGENTS.md"],        # list of paths to always-load memory files
    checkpointer=checkpointer,
)

result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "What's in your memory?"}],
        "files": {"/AGENTS.md": create_file_data(agents_md)},  # seed the file
    },
    config={"configurable": {"thread_id": "my-thread"}},
)
```

**Memory vs Skills summary:**

| | `memory=[...]` | `skills=[...]` |
|---|---|---|
| Load timing | Every invocation | Lazily, when agent decides it's relevant |
| Format | AGENTS.md (any markdown) | SKILL.md files with structured format |
| Token cost | Always paid | Deferred |
| Best for | Always-needed context (DB schema, user prefs) | Optional domain expertise |

---

## 10. Structured Output

Force the agent to return a validated Pydantic model. The result lands in `result["structured_response"]`.

```python
from pydantic import BaseModel, Field
from deepagents import create_deep_agent

class WeatherReport(BaseModel):
    """Structured weather report."""
    location: str = Field(description="The location for this report")
    temperature: float = Field(description="Current temperature in Celsius")
    condition: str = Field(description="e.g. sunny, cloudy, rainy")
    humidity: int = Field(description="Humidity percentage")
    wind_speed: float = Field(description="Wind speed in km/h")
    forecast: str = Field(description="Brief 24-hour forecast")

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    response_format=WeatherReport,
    tools=[internet_search],
)

result = agent.invoke({"messages": [{"role": "user", "content": "Weather in Pune?"}]})

report: WeatherReport = result["structured_response"]
print(report.temperature, report.condition)
```

---

## Quick Reference

| Parameter | Type | Purpose |
|---|---|---|
| `model` | `str \| BaseChatModel` | Provider + model to use |
| `tools` | `list` | Custom tool functions |
| `system_prompt` | `str \| SystemMessage` | Append task-specific instructions |
| `middleware` | `list[AgentMiddleware]` | Add cross-cutting hooks (logging, retries, etc.) |
| `subagents` | `list[dict]` | Spawn specialized child agents for isolation |
| `backend` | `BackendProtocol` | Filesystem backend (state / disk / store / sandbox) |
| `skills` | `list[str]` | Paths to lazy-loaded SKILL.md files |
| `memory` | `list[str]` | Paths to always-loaded AGENTS.md files |
| `response_format` | `type[BaseModel]` | Enforce Pydantic structured output |
| `interrupt_on` | `dict[str, bool\|config]` | HITL approval gates per tool |
| `checkpointer` | `BaseCheckpointSaver` | Required for HITL and stateful resumption |
| `store` | `BaseStore` | Required for StoreBackend / cross-thread memory |

**Key classes & imports:**

| Class / Function | Import |
|---|---|
| `create_deep_agent` | `from deepagents import create_deep_agent` |
| `StateBackend` | `from deepagents.backends import StateBackend` |
| `FilesystemBackend` | `from deepagents.backends import FilesystemBackend` |
| `LocalShellBackend` | `from deepagents.backends import LocalShellBackend` |
| `StoreBackend` | `from deepagents.backends import StoreBackend` |
| `CompositeBackend` | `from deepagents.backends import CompositeBackend` |
| `MemorySaver` | `from langgraph.checkpoint.memory import MemorySaver` |
| `InMemoryStore` | `from langgraph.store.memory import InMemoryStore` |
| `init_chat_model` | `from langchain.chat_models import init_chat_model` |
| `create_file_data` | `from deepagents.backends.utils import create_file_data` |
| `wrap_tool_call` | `from langchain.agents.middleware import wrap_tool_call` |

🔗 **Customization docs:** https://docs.langchain.com/oss/python/deepagents/customization
🔗 **API reference:** https://reference.langchain.com/python/deepagents/graph/create_deep_agent
🔗 **Backends:** https://docs.langchain.com/oss/python/deepagents/backends
🔗 **HITL:** https://docs.langchain.com/oss/python/deepagents/human-in-the-loop
🔗 **Sandboxes:** https://docs.langchain.com/oss/python/deepagents/sandboxes