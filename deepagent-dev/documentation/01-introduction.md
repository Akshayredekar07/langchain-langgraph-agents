# Deep Agents SDK — Developer Notes

> **Package:** `deepagents` | **Built on:** LangChain + LangGraph | **Docs:** https://docs.langchain.com
> **Last updated:** 2026-04-18

---

## Overview

`deepagents` is an **agent harness** built on top of LangChain's core abstractions and the LangGraph runtime. It solves the problem of building production-grade agents that handle complex, multi-step tasks — providing planning, context management, subagent delegation, filesystem backends, and long-term memory out of the box. Think of it as "LangChain agents with batteries included."

**Choose Deep Agents when you need:**
- Agents that plan and decompose tasks before acting
- Context window management across long sessions
- Delegating subtasks to specialized subagents
- Persistent memory across conversation threads
- Human approval gates for sensitive operations
- Provider-agnostic model swapping (Anthropic, OpenAI, Gemini, Ollama, etc.)

**Don't use it when:** You only need a simple single-step agent — use `langchain.create_agent` or a raw LangGraph graph instead.

**Core abstractions:**
- `create_deep_agent` — Factory function that returns a runnable agent with all built-in tools wired up
- `write_todos` — Built-in tool: agents use this to plan and track task decomposition
- Virtual Filesystem — In-memory or on-disk file store for offloading large context (`ls`, `read_file`, `write_file`, `edit_file`)
- `task` tool — Spawns isolated subagents for specialized subtasks
- Backends — Pluggable filesystem backends (in-memory, local disk, LangGraph store, sandboxes)
- Skills — Reusable packaged workflows with domain-specific instructions

---

## Installation & Setup

```bash
# Core package
pip install -qU deepagents

# Install your model provider's integration
pip install langchain-anthropic       # Anthropic Claude
pip install langchain-openai          # OpenAI
pip install langchain-google-genai    # Google Gemini
pip install langchain-ollama          # Ollama (local)
pip install langchain-openrouter      # OpenRouter
pip install langchain-fireworks       # Fireworks AI
pip install langchain-baseten         # Baseten
```

**Environment variables:**

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | If using Anthropic | Claude API key |
| `OPENAI_API_KEY` | If using OpenAI | OpenAI API key |
| `GOOGLE_API_KEY` | If using Google | Gemini API key |
| `LANGSMITH_API_KEY` | No | Enable LangSmith observability |
| `LANGSMITH_TRACING` | No | Set `true` to activate tracing |

---

## Core Concepts

### `create_deep_agent`

The single entry point for building any deep agent. Returns a LangGraph-backed runnable with all built-in tools already configured.

```python
from deepagents import create_deep_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",   # provider:model-name format
    tools=[get_weather],                    # your custom tools
    system_prompt="You are a helpful assistant",
)

# Invoke (blocking)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in Pune?"}]}
)
```

**Key parameters:**

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | required | `provider:model-name` string |
| `tools` | `list` | `[]` | Custom tool functions to expose to the agent |
| `system_prompt` | `str` | built-in | Override the default system prompt |

**Supported model string formats:**

| Provider | Format | Example |
|---|---|---|
| Anthropic | `anthropic:{model}` | `anthropic:claude-sonnet-4-6` |
| OpenAI | `openai:{model}` | `openai:gpt-5.4` |
| Google | `google_genai:{model}` | `google_genai:gemini-3.1-pro-preview` |
| Ollama | `ollama:{model}` | `ollama:devstral-2` |
| OpenRouter | `openrouter:{org}/{model}` | `openrouter:anthropic/claude-sonnet-4-6` |
| Fireworks | `fireworks:{path}` | `fireworks:accounts/fireworks/models/qwen3p5-397b` |
| Baseten | `baseten:{org}/{model}` | `baseten:zai-org/GLM-5` |

---

### Built-in Tools

Deep Agents ships with a set of tools pre-wired into every agent — no manual registration needed.

| Tool | Purpose |
|---|---|
| `write_todos` | Agent breaks task into discrete steps; tracks progress |
| `ls` | List virtual filesystem contents |
| `read_file` | Read a file from the virtual FS into context |
| `write_file` | Write/save large output to virtual FS |
| `edit_file` | Patch a specific section of a virtual FS file |
| `task` | Spawn a subagent with isolated context |
| `execute` | Run shell commands (only with sandbox backend) |

> 💡 **Tip:** The `write_todos` tool teaches the agent to plan before acting. If your agent is jumping straight to execution without thinking, ensure the system prompt isn't suppressing this behavior.

---

### Virtual Filesystem

The virtual FS prevents context window overflow by letting agents offload large outputs (e.g. research summaries, intermediate results) to storage instead of keeping them in the message list.

```python
# Agents use these tools internally — no direct Python API needed
# The agent will call them automatically as needed:
# write_file("summary.md", content)       → saves to backend
# read_file("summary.md")                 → loads back into context
# ls()                                    → lists all stored files
# edit_file("summary.md", patch)          → in-place edit
```

---

### Pluggable Backends

Swap the filesystem backend without changing agent code.

| Backend | Use Case | Notes |
|---|---|---|
| In-memory (default) | Development, testing | Lost on process exit |
| Local disk | Single-machine workflows | Persists to disk |
| LangGraph Store | Cross-thread/session persistence | Uses LangGraph's built-in store |
| Modal sandbox | Isolated code execution | Cloud execution environment |
| Daytona sandbox | Dev environment isolation | |
| Deno sandbox | Lightweight JS-native isolation | |
| Custom backend | Any use case | Implement the backend interface |

```python
from deepagents import create_deep_agent
from deepagents.backends import LocalDiskBackend

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    backend=LocalDiskBackend(base_path="./agent-workspace"),
)
```

---

### Subagent Spawning (`task` tool)

The `task` tool lets the main agent delegate isolated subtasks to specialized child agents. Each subagent gets its own clean context window — the parent's history doesn't bleed in.

```python
# The main agent calls `task` autonomously. You configure it declaratively:
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[...],
    # Subagents can inherit parent config or override it
)

# Internally, when the agent uses the `task` tool:
# task(description="Summarize research paper X", ...)
#   → Spawns child agent
#   → Child runs independently
#   → Returns result to parent
```

> ⚠️ **Watch out:** Subagent spawning increases token usage significantly. Monitor via LangSmith to catch runaway recursion.

---

### Long-term Memory

Persist information across separate conversation threads using LangGraph's Memory Store.

```python
from langgraph.store.memory import InMemoryStore
from deepagents import create_deep_agent

store = InMemoryStore()  # swap with a DB-backed store in production

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    store=store,           # agents can now save/retrieve across threads
)
```

---

### Filesystem Permissions

Declaratively restrict which files and directories agents can read or write.

```python
from deepagents import create_deep_agent
from deepagents.permissions import PermissionRules

rules = PermissionRules(
    allow_read=["./data/**", "./docs/**"],
    deny_write=["./secrets/**", ".env"],
)

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    permissions=rules,
)
```

---

### Human-in-the-Loop

Require human approval before sensitive tool calls using LangGraph's interrupt mechanism.

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[delete_records, send_email],
    human_in_the_loop=["delete_records", "send_email"],  # tools needing approval
)

# On interrupt, the graph pauses and waits for external resume signal
# Resume with: agent.invoke(..., config={"thread_id": "...", "checkpoint_id": "..."})
```

---

### Skills

Skills package reusable domain knowledge and instructions into loadable modules.

```python
from deepagents import create_deep_agent
from deepagents.skills import load_skill

coding_skill = load_skill("python-expert")

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    skills=[coding_skill],
)
```

---

## Streaming

```python
# Stream tokens/events as the agent runs
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Analyze this dataset"}]}
):
    print(chunk, end="", flush=True)
```

---

## Observability — LangSmith

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=your_key_here
```

Once set, every `agent.invoke()` or `agent.stream()` call is automatically traced in LangSmith — full tool call visibility, token counts, latency, and error traces.

---

## Common Patterns

### Pattern: Multi-Step Research Agent

```python
from deepagents import create_deep_agent
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search],
    system_prompt=(
        "You are a research assistant. Always write a todo list before starting. "
        "Save intermediate findings to files so you don't lose context."
    ),
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Research the latest advances in RAG systems and write a report"}]
})
```

### Pattern: Code Execution Agent (Sandbox)

```python
from deepagents import create_deep_agent
from deepagents.backends import ModalSandboxBackend

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    backend=ModalSandboxBackend(),   # enables `execute` shell tool
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Write and run a Python script that computes primes up to 100"}]
})
```

### Pattern: Persistent Memory Agent

```python
from langgraph.store.memory import InMemoryStore
from deepagents import create_deep_agent

store = InMemoryStore()

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    store=store,
)

# Thread 1
agent.invoke(
    {"messages": [{"role": "user", "content": "My name is Omkar"}]},
    config={"thread_id": "user-123"},
)

# Thread 2 — agent can recall "Omkar" if instructed to persist it
agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    config={"thread_id": "user-123"},
)
```

---

## When to Use Deep Agents vs Alternatives

| Scenario | Use |
|---|---|
| Complex, multi-step tasks needing planning | **Deep Agents SDK** |
| Durable execution, streaming, HITL workflows | **Deep Agents SDK** (via LangGraph runtime) |
| Simple single-tool agent | `langchain.create_agent` |
| Custom graph-based control flow | Raw **LangGraph** |
| Terminal/CLI coding agent | **Deep Agents CLI** (`deepagents` CLI) |
| Code editor integration | **ACP integration** (Zed, etc.) |

---

## Quick Reference

| Class / Function | Purpose | Import |
|---|---|---|
| `create_deep_agent` | Create a full deep agent | `from deepagents import create_deep_agent` |
| `LocalDiskBackend` | Persist FS to local disk | `from deepagents.backends import LocalDiskBackend` |
| `ModalSandboxBackend` | Sandboxed code execution via Modal | `from deepagents.backends import ModalSandboxBackend` |
| `PermissionRules` | Declarative filesystem access control | `from deepagents.permissions import PermissionRules` |
| `load_skill` | Load a packaged skill module | `from deepagents.skills import load_skill` |
| `InMemoryStore` | Cross-thread memory store (LangGraph) | `from langgraph.store.memory import InMemoryStore` |

**CLI:**
```bash
# Install and use the Deep Agents CLI (terminal coding agent)
pip install deepagents
deepagents --help
```

**ACP (code editor integration):**
- Connector for Zed and compatible editors
- See: `/oss/python/deepagents/acp`

- 🔗 **Official docs:** https://docs.langchain.com/oss/python/deepagents/
- 🔗 **PyPI:** https://pypi.org/project/deepagents/
- 🔗 **GitHub:** https://github.com/langchain-ai/deepagents
- 🔗 **API Reference:** https://reference.langchain.com/python/deepagents/
- 🔗 **LangSmith:** https://smith.langchain.com