## **AI Agent Context Files**

### Complete Engineering Reference  `deepagents` SDK · `SKILL.md` · `AGENTS.md` · `CLAUDE.md`

## Table of Contents
1. [What is DeepAgents?](#1-what-is-deepagents)
2. [Architecture How It Differs from a Basic Agent](#2-architecture)
3. [Installation & Setup](#3-installation--setup)
4. [`create_deep_agent` Core API](#4-create_deep_agent--core-api)
5. [Built-in Capabilities & Middleware Stack](#5-built-in-capabilities--middleware-stack)
6. [Async & MCP Tools](#6-async--mcp-tools)
7. [Subagents (Inline & Async)](#7-subagents-inline--async)
8. [Context Engineering & Memory](#8-context-engineering--memory)
9. [**`SKILL.md` Agent Skills System**](#9-skillmd--agent-skills-system)
10. [**`AGENTS.md` Cross-Agent Memory File**](#10-agentsmd--cross-agent-memory-file)
11. [**`CLAUDE.md` Claude-Native Project Instructions**](#11-claudemd--claude-native-project-instructions)
12. [File Hierarchy & Comparison Table](#12-file-hierarchy--comparison-table)
13. [Quick Reference](#13-quick-reference)


## 1. What is DeepAgents?

`deepagents` is an **agent harness** — a standalone library built on LangGraph that gives any LLM the
infrastructure to handle real-world, complex tasks. Think of it as the open-source version of what powers
Claude Code, Deep Research, and Manus.

**Core insight:** The difference between a shallow agent and a deep agent is NOT the LLM. It's the harness:
- Planning → breaks multi-step tasks into tracked subtasks
- Filesystem → offloads large context, prevents window overflow
- Subagents → parallelism and specialization
- Detailed system prompt → (Claude Code-inspired) guides tool use with examples

```python
# pip install deepagents langchain-anthropic

from deepagents import create_deep_agent

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[my_tool],
    system_prompt="You are a research assistant."
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Research RAG and write a summary."}]
})

print(result["messages"][-1].content)
print(result["files"])  # files the agent created during the task
```

---

## 2. Architecture

### Basic ReAct vs. Deep Agent

| Capability | Basic `create_agent` | `create_deep_agent` |
|---|---|---|
| Tool calling loop | Yes | Yes |
| Planning (`write_todos`) | No | Yes (built-in) |
| Filesystem (read/write/edit/ls) | No | Yes (built-in) |
| Context summarization | No | Yes (automatic) |
| Subagent delegation | No | Yes |
| AGENTS.md memory loading | No | Yes |
| Skills (SKILL.md) | No | Yes |
| Human-in-the-loop | No | Yes |
| Shell execution (sandboxed) | No | Yes |
| LangGraph runtime (streaming, checkpointing) | Depends on implementation | Yes |

### Middleware Stack (what `create_deep_agent` wires up)

```
User Prompt
     │
     ▼
┌─────────────────────────────────────┐
│  MemoryMiddleware   (AGENTS.md)      │  ← loads memory files into system prompt
├─────────────────────────────────────┤
│  SkillsMiddleware   (SKILL.md)       │  ← exposes skill descriptions (lazy load)
├─────────────────────────────────────┤
│  TodoListMiddleware (write_todos)    │  ← planning tool
├─────────────────────────────────────┤
│  FilesystemMiddleware               │  ← ls, read_file, write_file, edit_file
├─────────────────────────────────────┤
│  SubagentMiddleware                 │  ← task delegation to subagents
├─────────────────────────────────────┤
│  SummarizationMiddleware            │  ← auto-compacts context window
├─────────────────────────────────────┤
│  LLM + User Tools                   │  ← your custom tools injected here
└─────────────────────────────────────┘
     │
     ▼
  LangGraph CompiledGraph (invoke / stream / ainvoke / astream)
```

`create_deep_agent` **returns a standard LangGraph `CompiledGraph`** — fully compatible with Studio,
checkpointers, streaming, and human-in-the-loop.

---

## 3. Installation & Setup

```bash
# Core (defaults to Claude Sonnet 4)
pip install deepagents langchain-anthropic

# Other providers
pip install deepagents langchain-openai        # OpenAI
pip install deepagents langchain-google-genai  # Gemini
pip install deepagents langchain-ollama        # Ollama (local)
pip install deepagents langchain-openrouter    # OpenRouter

# MCP support (required for async MCP tools)
pip install langchain-mcp-adapters
```

```bash
# Environment variables
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
LANGSMITH_TRACING=true   # optional but recommended
LANGSMITH_API_KEY=...
```

---

## 4. `create_deep_agent` — Core API

### Signature

```python
from deepagents import create_deep_agent, async_create_deep_agent

agent = create_deep_agent(
    model=None,              # str | LanguageModelLike | None → defaults to Claude Sonnet 4
    tools=[],                # list[BaseTool | Callable | dict]
    system_prompt="",        # str — inserted into the larger built-in system prompt
    subagents=[],            # list[SubAgent | CompiledSubAgent | AsyncSubAgent]
    memory=[],               # list[str]  — file paths to AGENTS.md-style memory files
    skills=[],               # list[str]  — directories containing SKILL.md files
    backend=None,            # storage backend (StateBackend, StoreBackend, FilesystemBackend)
    checkpointer=None,       # LangGraph checkpointer for persistence
    state_schema=None,       # custom state class extending base agent state
    debug=False,             # verbose tool call logging
)
```

### Provider strings

```python
# Anthropic
model = "anthropic:claude-sonnet-4-6"

# OpenAI
model = "openai:gpt-5.4"

# Google
model = "google_genai:gemini-3.1-pro-preview"

# Ollama (local)
model = "ollama:devstral-2"

# OpenRouter
model = "openrouter:anthropic/claude-sonnet-4-6"

# Or pass a LangChain model object directly
from langchain.chat_models import init_chat_model
model = init_chat_model("anthropic:claude-sonnet-4-6", temperature=0.7)
```

### Invocation patterns

```python
# ── Sync invoke ──────────────────────────────────────────────────────
result = agent.invoke({
    "messages": [{"role": "user", "content": "Your task here"}]
})
final_answer = result["messages"][-1].content
created_files = result["files"]  # dict of filename → content

# ── Async invoke ─────────────────────────────────────────────────────
result = await agent.ainvoke({
    "messages": [{"role": "user", "content": "Your task here"}]
})

# ── Streaming ────────────────────────────────────────────────────────
async for chunk in agent.astream({"messages": [...]}):
    for node_name, node_output in chunk.items():
        if isinstance(node_output, dict):
            raw = node_output.get("messages")
            messages = raw.value if hasattr(raw, "value") else raw
            for msg in (messages if isinstance(messages, list) else [messages]):
                if content := getattr(msg, "content", None):
                    print(f"[{node_name}] {content}", flush=True)

# ── Multi-turn / conversation history ───────────────────────────────
history = []
for user_msg in ["Turn 1", "Turn 2", "Turn 3"]:
    history.append({"role": "user", "content": user_msg})
    result = await agent.ainvoke({"messages": history})
    assistant_reply = result["messages"][-1].content
    history.append({"role": "assistant", "content": assistant_reply})

# ── With LangGraph persistence (thread-level memory) ─────────────────
from langgraph.checkpoint.memory import MemorySaver
agent = create_deep_agent(model=model, checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "session-42"}}
agent.invoke({"messages": [...]}, config=config)
```

---

## 5. Built-in Capabilities & Middleware Stack

### 5.1 Planning — `write_todos`

The agent uses a simple no-op `write_todos` tool (inspired by Claude Code's `TodoWrite`) to create and
track a plan *before* execution. This keeps long-running tasks on track.

```
[Agent internal call]
write_todos([
  "1. Search for RAG papers",
  "2. Extract trends from top 5",
  "3. Write 3-bullet summary"
])
```

The todos live in the agent's context (not persisted externally by default). The agent checks them off
as it progresses.

### 5.2 Virtual Filesystem

The agent gets four file tools for offloading context:

| Tool | Description |
|---|---|
| `ls [path]` | list directory contents |
| `read_file <path>` | read a file into context |
| `write_file <path> <content>` | create/overwrite a file |
| `edit_file <path> <old> <new>` | targeted string replacement |

**Why this matters for RAG pipelines:** The agent can write intermediate results to files, preventing
context window overflow. For example: write 20 papers to `papers.md`, then read and synthesize on demand.

### 5.3 Auto-Summarization

When context grows too long, `SummarizationMiddleware` automatically compacts older messages.
- Preserves architectural decisions and key results
- Discards redundant tool outputs
- Keeps the agent effective across extended sessions

You can also trigger compaction manually with the `compact` tool.

### 5.4 Built-in System Prompt

`deepagents` ships with a **detailed, Claude Code-inspired system prompt** that contains:
- How to use the planning tool with examples
- When/how to use filesystem tools
- How to delegate to subagents
- Few-shot examples of proper tool usage

Your `system_prompt=` argument is **inserted into** this larger prompt, not replacing it.

> **Watch out:** The built-in system prompt is long (~50+ instructions). This is intentional — it's
> what makes the agent "deep". Don't fight it; extend it via `system_prompt=`.

---

## 6. Async & MCP Tools

**Critical:** MCP tools are inherently async (network I/O). Use `async_create_deep_agent` for all MCP setups.

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from deepagents import async_create_deep_agent

async def build_agent():
    # ── MCP server connections ─────────────────────────────────────
    connections = {
        "huggingface": {
            "transport": "http",
            "url": "https://huggingface.co/mcp",
            "headers": {"Authorization": f"Bearer {HF_TOKEN}"},
        },
        # Add more servers:
        # "github": {"transport": "http", "url": "...", "headers": {...}}
    }

    client = MultiServerMCPClient(connections)
    mcp_tools = await client.get_tools()   # → list[BaseTool]

    print(f"Loaded {len(mcp_tools)} MCP tools:")
    for t in mcp_tools:
        print(f"  • {t.name}: {t.description[:60]}...")

    agent = await async_create_deep_agent(
        model="anthropic:claude-sonnet-4-6",
        tools=mcp_tools,                   # MCP tools injected here
        system_prompt="You are a HF research assistant.",
        debug=True,
    )
    return agent

agent = asyncio.run(build_agent())
```

### Common HuggingFace MCP tools exposed

| Tool | What it does |
|---|---|
| `model_search` | Search Hub models by query |
| `dataset_search` | Search Hub datasets |
| `spaces_semantic_search` | Find Gradio Spaces |
| `papers_semantic_search` | Search ML papers |
| `documentation_search` | Search HF docs |
| `hub_repository_details` | Get model/dataset/space info |

---

## 7. Subagents (Inline & Async)

Subagents let the main agent **delegate** work to specialized agents. Each subagent is itself a
`create_deep_agent`-style agent with its own tools and prompt.

### 7.1 Inline Subagents (blocking)

```python
from deepagents import create_deep_agent, SubAgent

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    subagents=[
        SubAgent(
            name="researcher",
            description="Performs deep research on a topic.",
            prompt="You are an expert researcher. Thoroughly search and summarize findings.",
            tools=["web_search"],           # tool names available to this subagent
            model={"model": "anthropic:claude-sonnet-4-6"},
        ),
        SubAgent(
            name="writer",
            description="Writes polished long-form content from research notes.",
            prompt="You are a technical writer. Transform raw notes into clear prose.",
        ),
    ],
)
```

The main agent gains a `task` tool to call subagents. Inline subagents **block** the main agent
until they complete — good for short focused tasks.

### 7.2 Async Subagents (non-blocking, v0.5+)

For long-running tasks (deep research, large-scale code analysis), use `AsyncSubAgent`. The main agent
fires-and-forgets, can continue working, and checks back when needed.

```python
from deepagents import AsyncSubAgent, create_deep_agent

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    subagents=[
        AsyncSubAgent(
            name="researcher",
            description="Performs deep research in the background.",
            url="https://my-agent-server.dev",   # remote Agent Protocol server
            graph_id="research_agent",
        ),
    ],
)
# Main agent gains 5 tools: start_async_task, check_async_task,
# update_async_task, cancel_async_task, list_async_tasks
```

| Type | Blocks main? | Use case |
|---|---|---|
| `SubAgent` | Yes | Short, focused subtasks (seconds) |
| `CompiledSubAgent` | Yes | Pre-built LangGraph graphs |
| `AsyncSubAgent` | No | Long tasks (minutes): research, analysis |

---

## 8. Context Engineering & Memory

### 8.1 Short-term (in-session)

Managed automatically:
- Conversation `messages` history
- Scratch files via `write_file` / `read_file`
- Auto-summarization when context grows too long

### 8.2 Long-term (cross-session) — Backends

Control where files are stored:

```python
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    memory=["/memories/AGENTS.md"],       # memory file path
    skills=["/skills/"],                  # skills directory
    backend=CompositeBackend(
        default=StateBackend(),           # in-memory (per conversation)
        routes={
            "/memories/": StoreBackend(   # persistent across threads
                namespace=lambda rt: (rt.server_info.assistant_id,)
            ),
            "/skills/": StoreBackend(
                namespace=lambda rt: (rt.server_info.assistant_id,)
            ),
        },
    ),
)
```

| Backend | Persistence | When to use |
|---|---|---|
| `StateBackend` | In-memory, per thread | Default, local dev |
| `StoreBackend` | Persistent, cross-thread | Production memory |
| `FilesystemBackend` | Local disk | Local persistence |
| Custom sandbox | Isolated execution | Code execution agents |

---

## 9. `SKILL.md` — Agent Skills System

### What is a Skill?

A **Skill** is a directory containing a `SKILL.md` file (plus optional scripts/docs/templates) that
gives the agent specialized, reusable domain knowledge. Skills are the **procedural memory** of an agent —
they tell it *how* to perform a task, not *what* it already knows.

Skills follow the open [Agent Skills Specification](https://agentskills.io/specification) and work
across Claude Code, Claude.ai, and `deepagents`.

### Filesystem Layout

```
skills/
├── rag-pipeline/
│   ├── SKILL.md          ← required: instructions + frontmatter
│   ├── embed_docs.py     ← optional: bundled script
│   └── schema.json       ← optional: reference data
├── arxiv-search/
│   ├── SKILL.md
│   └── arxiv_search.py
└── langgraph-docs/
    └── SKILL.md
```

### `SKILL.md` File Format

```markdown
---
name: rag-pipeline                  # unique skill identifier
description: >                      # shown to agent at startup (max 1024 chars)
  Use this skill when the user asks to build, debug, or optimize a RAG
  pipeline. Covers embedding, chunking, retrieval, and reranking patterns.
license: MIT                        # optional
compatibility: Requires langchain-community
metadata:                           # optional key-value metadata
  author: akshay
  version: "1.0"
allowed-tools: read_file, write_file, execute   # optional: restrict which tools skill can use
---

# RAG Pipeline Skill

## Overview
This skill provides step-by-step guidance for building production-grade RAG systems.

## Instructions

### 1. Understand the data source
Ask the user about document format (PDF, web, DB), volume, and update frequency.

### 2. Choose chunking strategy
- Fixed-size: simple, fast. Use for uniform documents.
- Semantic: expensive but better recall. Use for mixed content.
- Use `embed_docs.py` for batch embedding (run: `python embed_docs.py --input ./docs`)

### 3. Select vector store
Refer to `schema.json` for supported stores and their connection strings.

### 4. Build retrieval chain
Use the LangChain LCEL pattern:
```python
chain = retriever | prompt | llm | StrOutputParser()
```

### 5. Evaluate
Run RAGAs evaluation after building. Target: faithfulness > 0.85, answer_relevancy > 0.80.
```

### How Skills Load (Progressive Disclosure)

```
Agent startup:
  → Reads SKILL.md frontmatter only for each skill
  → Adds skill name + description to system prompt (lightweight)

When user prompt matches a skill:
  → Agent reads full SKILL.md body into context
  → If SKILL.md references other files (scripts, docs), agent reads those too
  → Scripts are EXECUTED via bash; only output enters context (not the code itself)
  → Files NOT needed for this task are NEVER read → context stays lean
```

> 💡 **Tip:** This **progressive disclosure** pattern is key — there's no token penalty for bundling
> large reference docs into a skill if they aren't accessed on every call.

### Registering Skills with DeepAgents

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    skills=["/path/to/skills/"],           # directory containing skill subdirs
)

# Seed skills into a StoreBackend for persistence
from deepagents.backends.utils import create_file_data
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
store.put(
    ("my-agent",),
    "/skills/langgraph-docs/SKILL.md",
    create_file_data("""---
name: langgraph-docs
description: Fetch LangGraph docs when user asks LangGraph questions.
---
# Instructions
Use fetch_url to read https://docs.langchain.com/llms.txt, then fetch relevant pages.
"""),
)
```

### Skills vs. Tools vs. System Prompt

| | Tools | System Prompt | Skills (SKILL.md) |
|---|---|---|---|
| **What** | Executable functions | Always-on instructions | On-demand domain knowledge |
| **When loaded** | Every call | Every call | Only when relevant |
| **Context cost** | Result only | Always | Zero until triggered |
| **Bundled scripts** | No | No | Yes |
| **Reusable across agents** | Manually | No | Yes (shared repo) |
| **Best for** | Data fetching, APIs | Global rules, persona | Domain workflows, SOPs |

---

## 10. `AGENTS.md` — Cross-Agent Memory File

### What is `AGENTS.md`?

`AGENTS.md` is the **agent-agnostic** project instruction file — the open standard equivalent of
`CLAUDE.md`. It's designed to work across Claude, GPT-based Copilot, Gemini, and any other agent
that reads it.

In `deepagents`, it also serves as the **long-term memory file** — the agent reads it at session
start and can write new knowledge back to it between conversations.

### Two Roles of `AGENTS.md`

**Role 1: Project/Codebase Instructions** (OpenAI Codex, GitHub Copilot, multi-tool teams)
```markdown
# AGENTS.md

## Project Overview
Production RAG system for enterprise search. FastAPI backend, LangChain core, PGVector.

## Architecture
- Ingestion: `src/ingestion/` → chunker → embedder → PGVector
- Retrieval: `src/retrieval/` → hybrid BM25 + dense → reranker
- API: `src/api/` → FastAPI endpoints

## Commands
- Run tests: `pytest tests/ -x --tb=short`
- Lint: `ruff check src/ --fix`
- Embed docs: `python scripts/embed.py --input data/`

## Rules
- Never commit secrets. Use `.env` + `python-dotenv`.
- All embeddings use `text-embedding-3-small` at dim=1536.
- Chunking: 512 tokens, 50-token overlap.

## Programmatic Checks (before finishing any task)
- Run `ruff check src/` → must pass
- Run `pytest tests/unit/` → must pass
```

**Role 2: Agent Long-term Memory** (deepagents `memory=` parameter)
```markdown
# AGENTS.md — Agent Memory

## Learned Preferences
- User prefers detailed code comments
- User works with Python 3.11, no walrus operator for compatibility

## Session Learnings
- 2025-04-10: Discovered PGVector index needs HNSW for >100k docs
- 2025-04-15: User's embedding model changed to voyage-3

## Codebase Notes
- `chunker.py:split_docs()` is the single source of truth for chunking
- Do not modify `schema.sql` directly — use Alembic migrations
```

### Loading `AGENTS.md` into DeepAgents

```python
from deepagents import create_deep_agent
from deepagents.backends import StoreBackend, CompositeBackend, StateBackend
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# Seed initial memory
from deepagents.backends.utils import create_file_data
store.put(
    ("rag-agent",),
    "/memories/AGENTS.md",
    create_file_data("""## Project context
RAG system for enterprise search. Python 3.11, FastAPI, PGVector.
## Preferences
- Concise responses
- Code first, explanation second
"""),
)

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    memory=["/memories/AGENTS.md"],         # agent reads this at startup
    backend=lambda rt: CompositeBackend(
        default=StateBackend(rt),
        routes={
            "/memories/": StoreBackend(
                rt, namespace=lambda rt: ("rag-agent",)
            ),
        },
    ),
    store=store,
)
# The agent can now update AGENTS.md using edit_file when it learns new things
```

### Best Practices for `AGENTS.md`

```
✅ DO:
  - Keep it under 200 lines (beyond that → modular subdirectory files)
  - Place critical rules EARLY (agents suffer "lost in the middle" on long files)
  - Document non-obvious things only (agents already know TypeScript ≠ Python)
  - Include exact commands ("pytest tests/ -x" not "run tests")
  - Version control it — treat updates as code changes
  - Write it yourself; don't auto-generate it

❌ DON'T:
  - Auto-generate with an LLM (ETH Zurich study: LLM-generated files ↓ success rate by 0.5–2%)
  - Document things the agent can infer from the codebase itself
  - Include every command that could possibly be needed (bloat = worse performance)
  - Add a "Project Structure" section if you follow a standard framework layout
```

> ⚠️ **Watch out:** Per ETH Zurich research (AGENTbench, 2026): LLM-generated `AGENTS.md` files
> consistently *increased* step count and *decreased* task success. Human-written files with
> non-inferable, task-specific details gave a ~4% improvement. Write it manually.

---

## 11. `CLAUDE.md` — Claude-Native Project Instructions

### What is `CLAUDE.md`?

`CLAUDE.md` is **Claude's orientation manual** for a specific project. Claude Code reads it
automatically at session start, injecting it into context before any user message. It acts as a
persistent, session-scoped system prompt living in version control.

```
~/.claude/CLAUDE.md          ← global (applies to ALL projects)
./CLAUDE.md                  ← project root (this project only)
./src/CLAUDE.md              ← subdirectory (this folder only, merged with parent)
```

Claude merges all found CLAUDE.md files, respecting scope hierarchy.

### `CLAUDE.md` File Structure

```markdown
# CLAUDE.md — RAG System

## Project Context
Production RAG API for enterprise search. FastAPI + LangChain + PGVector. Python 3.11.

## Tech Stack
- Embeddings: `text-embedding-3-small` (dim=1536, always)
- Vector store: PGVector with HNSW index
- LLM: Claude Sonnet via Anthropic SDK
- Framework: FastAPI + pydantic-settings

## Commands
```bash
pytest tests/ -x --tb=short          # run tests, stop on first fail
ruff check src/ --fix                 # lint + autofix
python scripts/embed.py --input data/ # embed new documents
alembic upgrade head                  # apply DB migrations
```

## Hard Rules
- NEVER modify `schema.sql` directly. Use Alembic migrations.
- NEVER store secrets in code. Use `.env` + `python-dotenv`.
- NEVER use `print()` in production code. Use `logging`.
- ALL embeddings must go through `src/core/embedder.py`, not ad-hoc calls.

## Code Style
- Type hints everywhere. No bare `Any`.
- Async-first: use `async def` for all I/O operations.
- Prefer `ruff` over `flake8`. Follow project's `pyproject.toml` rules.

## Response Style
- Code first, then brief explanation.
- Don't explain basics. I know Python and ML.
- Be direct. Skip preamble.
```

### Writing Effective `CLAUDE.md` Files

**The 5-question framework before adding any line:**
1. Would Claude get this wrong without being told? → include
2. Can Claude infer this from the codebase? → skip
3. Is this universally applicable to every task? → include
4. Is this only relevant to one specific task? → skip (put in a slash command instead)
5. Does this compete with actual work context? → shorten it

**Sizing guidance:**

| File size | Effect |
|---|---|
| < 100 lines | ✅ Optimal. High signal density. |
| 100-300 lines | ⚠️ Acceptable. Review for redundancy. |
| > 300 lines | ❌ Danger zone. Token budget stolen from actual work. |

> 💡 **Tip:** "Progressive disclosure" applies here too. Don't stuff everything in `CLAUDE.md`. Instead,
> tell Claude *how to find* information: "For DB schema details, read `schema.sql`." Claude will fetch
> it on-demand only when relevant.

### Auto-Memory Feature

Claude Code's auto-memory can write back to `CLAUDE.md`:

```
Session → Claude learns new project insight → writes it to CLAUDE.md
Next session → CLAUDE.md includes the learned insight → Claude arrives pre-oriented
```

This creates a **compounding knowledge loop** across sessions. Enable it in Claude Code settings.

> ⚠️ **Watch out:** The "silent rule dropout" problem. Claude Code has documented issues ignoring
> `CLAUDE.md` instructions mid-session ("lost in the middle"). Keep files short, place critical
> rules early, and start fresh sessions for new tasks.

---

## 12. File Hierarchy & Comparison Table

### The Full Ecosystem

```
Project
├── CLAUDE.md           ← Claude Code specific. Always loaded.
├── AGENTS.md           ← Agent-agnostic. Always loaded by most tools.
├── .cursorrules        ← Cursor specific
├── .github/
│   └── copilot-instructions.md  ← GitHub Copilot
├── .claude/
│   ├── commands/       ← Slash commands (manually triggered)
│   ├── agents/         ← Custom Claude Code subagent definitions
│   └── skills/         ← Claude Code skill directories (SKILL.md)
└── skills/             ← DeepAgents skills directory (SKILL.md)
```

### Master Comparison Table

| | `CLAUDE.md` | `AGENTS.md` | `SKILL.md` |
|---|---|---|---|
| **Purpose** | Claude-native project instructions | Cross-agent instructions + agent memory | Reusable domain workflow |
| **Scope** | Project/global | Project/cross-agent/memory | Task-specific capability |
| **Load timing** | Always, at session start | Always, at session start | On-demand (progressive disclosure) |
| **Written by** | Developer | Developer (+ agent updates it) | Developer |
| **Agent can write?** | Via auto-memory | Yes (memory mode) | Read-only (typically) |
| **Cross-tool?** | Claude only | All agents | All deepagents-compatible agents |
| **Ideal size** | < 300 lines | < 200 lines | Any (lazy-loaded) |
| **Location** | `./CLAUDE.md` | `./AGENTS.md` | `skills/<name>/SKILL.md` |
| **Contains** | Rules, commands, style, context | Architecture, commands, memory | Instructions, scripts, templates |

### Multi-Tool Team Pattern (symlink)

```bash
# Keep a single source of truth for teams using both Claude and other tools
ln -s AGENTS.md CLAUDE.md   # CLAUDE.md is a symlink to AGENTS.md
# Add a note at top of AGENTS.md:
# "Note: CLAUDE.md is a symlink to AGENTS.md. They are the same file."
```

---

## 13. Quick Reference

### DeepAgents API Cheatsheet

| Function / Class | Purpose |
|---|---|
| `create_deep_agent(...)` | Create sync deep agent (returns `CompiledGraph`) |
| `async_create_deep_agent(...)` | Create async deep agent (required for MCP tools) |
| `SubAgent` | Inline blocking subagent spec |
| `AsyncSubAgent` | Non-blocking remote subagent spec (v0.5+) |
| `CompiledSubAgent` | Pre-built LangGraph graph as subagent |
| `StateBackend` | In-memory per-thread file storage |
| `StoreBackend` | Persistent cross-thread storage (LangGraph Store) |
| `FilesystemBackend` | Local disk storage |
| `CompositeBackend` | Route different paths to different backends |
| `MemorySaver` | LangGraph checkpointer for thread-level persistence |
| `MultiServerMCPClient` | Load MCP tools from multiple servers |

### Built-in Agent Tools

| Tool | Category | Description |
|---|---|---|
| `write_todos` | Planning | Create/update task list |
| `ls` | Filesystem | List directory |
| `read_file` | Filesystem | Read file into context |
| `write_file` | Filesystem | Create/overwrite file |
| `edit_file` | Filesystem | Targeted string replacement |
| `execute` | Shell | Run bash commands (sandbox only) |
| `task` | Subagents | Delegate to a subagent |
| `compact` | Context | Manually compress conversation history |

### File Format Templates

**SKILL.md minimum viable:**
```markdown
---
name: my-skill
description: Use this when the user asks about [X]. Covers [Y, Z].
---
# My Skill
## Instructions
1. Step one
2. Step two
```

**AGENTS.md minimum viable:**
```markdown
# AGENTS.md
## Project: [Name] — [one-line description]
## Commands
- Test: `pytest tests/ -x`
- Lint: `ruff check src/ --fix`
## Rules
- [Non-obvious constraint 1]
- [Non-obvious constraint 2]
```

**CLAUDE.md minimum viable:**
```markdown
# CLAUDE.md
## Context
[One-line project description. Stack. Language.]
## Commands
- [Exact command strings]
## Rules
- [What Claude would get wrong without being told]
## Style
- [Code style and response preferences]
```

---

> 🔗 **Official Docs:**
> - DeepAgents: https://docs.langchain.com/oss/python/deepagents/overview
> - Agent Skills spec: https://agentskills.io/specification
> - DeepAgents GitHub: https://github.com/langchain-ai/deepagents
> - Claude Code docs: https://code.claude.com/docs
> - DeepAgents Reference: https://reference.langchain.com/python/deepagents