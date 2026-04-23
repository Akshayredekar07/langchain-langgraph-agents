# agent.py — HuggingFace Deep Agent
# ──────────────────────────────────────────────────────────────────────────────
# Upgraded from the basic MCP agent to include:
#   • AGENTS.md  → persistent memory loaded into system prompt
#   • skills/    → 3 custom skills (hf-hub-search, hf-rag-advisor, web-research)
#   • FilesystemBackend → agent can read/write files for context offloading
#   • MemorySaver checkpointer → thread-level conversation persistence
#
# GitHub resources for official pre-built skills:
#   • deepagents examples  : github.com/langchain-ai/deepagents/tree/main/libs/cli/examples/skills
#   • langchain-skills repo: github.com/langchain-ai/langchain-skills
#     └── skills: framework-selection, langchain-dependencies, deep-agents-core
#
# Install:
#   pip install deepagents langchain-anthropic langchain-mcp-adapters python-dotenv
# ──────────────────────────────────────────────────────────────────────────────

import asyncio
import os
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver

import deepagents
import inspect

# ── Compat shim: works with both sync and async factory ───────────────────────
create_deep_agent = getattr(
    deepagents,
    "async_create_deep_agent",  # prefer async (required for MCP tools)
    deepagents.create_deep_agent,
)

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent            # project root
AGENTS_FILE = BASE_DIR / "AGENTS.md"        # local disk path for verification/debugging
SKILLS_DIR_PATH = BASE_DIR / "skills"       # local disk path for verification/debugging

# Deep Agents skill/memory sources must be virtual POSIX paths relative to backend root.
AGENTS_MD = "/AGENTS.md"
SKILLS_DIR = "/skills/"

HF_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

async def build_agent():
    """
    Build a deep agent with:
      - HuggingFace MCP server (model/dataset/paper search)
      - AGENTS.md memory (loaded into system prompt at startup)
      - 3 SKILL.md skills (lazy-loaded on demand via progressive disclosure)
      - FilesystemBackend (agent can write intermediate results to files)
      - MemorySaver checkpointer (thread-level conversation history)

    Why async_create_deep_agent?
    MCP tools are inherently async (network I/O). The sync version
    does not support them. Always use async for MCP setups.
    """

    # ── 1. LLM ────────────────────────────────────────────────────────────────
    model = init_chat_model(
        model="claude-sonnet-4-6",
        model_provider="anthropic",
        temperature=0.7,
        max_tokens=2048,  # bumped up for richer skill-guided responses
    )

    # ── 2. HuggingFace MCP Server ─────────────────────────────────────────────
    # Official HF MCP endpoint: https://huggingface.co/mcp
    # Transport: streamable HTTP (MCP spec 2025-03-26)
    # Auth: Bearer token
    #
    # Available tools:
    #   model_search, dataset_search, spaces_semantic_search,
    #   papers_semantic_search, documentation_search, hub_repository_details
    connections: dict[str, Any] = {
        "huggingface": {
            "transport": "http",
            "url": "https://huggingface.co/mcp",
            "headers": {
                "Authorization": f"Bearer {HF_TOKEN}",
            },
        }
    }
    client = MultiServerMCPClient(connections)

    # ── 3. Load MCP tools ────────────────────────────────────────────────────
    mcp_tools = await client.get_tools()

    print(f"\n[OK] Loaded {len(mcp_tools)} HuggingFace MCP tools:")
    for t in mcp_tools:
        print(f"   • {t.name}: {t.description[:70]}...")

    # ── 4. Import backends ────────────────────────────────────────────────────
    # FilesystemBackend: agent reads/writes real files on disk under BASE_DIR
    # This lets the agent offload large intermediate results to files
    # (e.g., write 20 paper abstracts to /research/rag-papers.md)
    from deepagents.backends import FilesystemBackend

    # ── 5. Build the deep agent ───────────────────────────────────────────────
    agent_or_coro = create_deep_agent(
        model=model,

        # ── MCP tools from HuggingFace ──────────────────────────────────────
        tools=mcp_tools,

        # ── AGENTS.md: persistent memory ────────────────────────────────────
        # Loaded by MemoryMiddleware at startup → injected into system prompt.
        # Agent can update it using edit_file when it learns new things.
        memory=[AGENTS_MD],

        # ── Skills: lazy-loaded domain knowledge ─────────────────────────────
        # At startup: only the `name` + `description` from each SKILL.md
        # frontmatter is read → tiny context cost.
        # When user prompt matches: full SKILL.md body is loaded on-demand.
        # Skills loaded:
        #   hf-hub-search  → how to use HF MCP tools effectively
        #   hf-rag-advisor → RAG pipeline design patterns
        #   web-research   → structured research workflow
        skills=[SKILLS_DIR],

        # ── Filesystem backend ───────────────────────────────────────────────
        # Agent can use ls, read_file, write_file, edit_file tools
        # rooted at BASE_DIR. Useful for offloading large research outputs.
        backend=FilesystemBackend(root_dir=str(BASE_DIR), virtual_mode=True),

        # ── Thread-level memory (conversation history) ───────────────────────
        # Lets you resume conversations across calls using thread_id config.
        checkpointer=MemorySaver(),

        # ── Custom system prompt ─────────────────────────────────────────────
        # This is APPENDED to deepagents' built-in system prompt (which
        # already covers planning, filesystem usage, and subagent delegation).
        system_prompt=(
            "You are an expert AI/ML research assistant with live access to the "
            "HuggingFace Hub via MCP tools. Always plan your steps using write_todos "
            "before executing. Use HuggingFace MCP tools to search models, datasets, "
            "papers, and documentation. When results are long, write them to files "
            "with write_file to avoid context overflow. Apply relevant skills for "
            "structured task execution."
        ),

        # debug=True,
    )

    agent = await agent_or_coro if asyncio.iscoroutine(agent_or_coro) else agent_or_coro

    print("\n[OK] Agent built with:")
    print(f"   Memory  : {AGENTS_MD} -> {AGENTS_FILE}")
    print(f"   Skills  : {SKILLS_DIR} -> {SKILLS_DIR_PATH}")
    print(f"     • hf-hub-search   (HF model/dataset/paper search)")
    print(f"     • hf-rag-advisor  (RAG pipeline design)")
    print(f"     • web-research    (structured research workflow)")
    print(f"   Backend : FilesystemBackend(root={BASE_DIR}, virtual_mode=True)")
    print(f"   Memory  : MemorySaver (thread-level persistence)")

    return agent


# ══════════════════════════════════════════════════════════════════════════════
#  RUN MODES
# ══════════════════════════════════════════════════════════════════════════════

# ── Config helper for thread persistence ─────────────────────────────────────
def make_config(thread_id: str = "default") -> RunnableConfig:
    """Pass this to agent.invoke() / astream() to persist conversation history."""
    return cast(RunnableConfig, {"configurable": {"thread_id": thread_id}})


async def run_invoke():
    """
    Single-shot invocation.
    Demonstrates: memory + skill auto-activation for a RAG-related query.
    """
    agent = await build_agent()
    config = make_config("demo-invoke")

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Find the top 3 embedding models on HuggingFace for a "
                        "production RAG pipeline. Compare them in a table with "
                        "dimensions, max tokens, and license."
                    ),
                }
            ]
        },
        config=config,
    )

    print("\n── FINAL ANSWER ──────────────────────────────────────────────────")
    print(result["messages"][-1].content)

    # Show files the agent wrote (if any)
    if result.get("files"):
        print("\n── FILES WRITTEN BY AGENT ────────────────────────────────────")
        for fname in result["files"]:
            print(f"   [FILE] {fname}")


async def run_streaming():
    """
    Streaming invocation — better UX for long research tasks.
    Demonstrates: paper search + write_todos planning visible in stream.
    """
    agent = await build_agent()
    config = make_config("demo-stream")

    print("\n── STREAMING ─────────────────────────────────────────────────────")
    async for chunk in agent.astream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Search for RAG-related papers on HuggingFace "
                        "and write a 3-bullet summary of recent trends. "
                        "Then save the full paper list to a file."
                    ),
                }
            ]
        },
        config=config,
    ):
        for node_name, node_output in chunk.items():
            if not isinstance(node_output, dict):
                continue

            raw = node_output.get("messages")
            if raw is None:
                continue

            # Unwrap LangGraph Overwrite wrapper if present
            messages = raw.value if hasattr(raw, "value") else raw
            if not isinstance(messages, list):
                messages = [messages]

            for msg in messages:
                content = getattr(msg, "content", None)
                if content:
                    print(f"[{node_name}] {content}", flush=True)


async def run_multi_turn():
    """
    Multi-turn conversation — demonstrates:
      1. Skill activation on Turn 1 (HF model search)
      2. Context continuity on Turn 2 (follow-up on previous results)
      3. Memory update on Turn 3 (agent stores learned preference in AGENTS.md)
    """
    agent = await build_agent()
    # Use the SAME thread_id across all turns → conversation history persists
    config = make_config("demo-multi-turn")

    turns = [
        "Search for sentence-transformer models on HuggingFace for semantic search.",
        "Which of those would work best for a multilingual RAG pipeline?",
        (
            "Show me the full details of your top recommendation, and please "
            "remember that I prefer models with Apache 2.0 license for all future searches."
        ),
    ]

    for turn_num, user_msg in enumerate(turns, 1):
        print(f"\n[TURN {turn_num} — USER] {user_msg}")
        print("─" * 60)

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_msg}]},
            config=config,
        )

        # In multi-turn with MemorySaver, full history is managed internally.
        # We just pass the latest user message each time.
        assistant_msg = result["messages"][-1].content
        print(f"[AGENT] {assistant_msg[:500]}...")
        if len(assistant_msg) > 500:
            print(f"         ... ({len(assistant_msg)} chars total)")


async def run_rag_advisor():
    """
    Dedicated run mode to trigger the hf-rag-advisor skill.
    Shows progressive disclosure in action — skill body is loaded only for this query.
    """
    agent = await build_agent()
    config = make_config("demo-rag-advisor")

    print("\n── RAG ADVISOR MODE ──────────────────────────────────────────────")
    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Design a complete RAG pipeline for a legal document search "
                        "system. Recommend HuggingFace models for each stage: "
                        "embedding, reranking, and generation. Include integration "
                        "code using LangChain."
                    ),
                }
            ]
        },
        config=config,
    )

    print("\n── RAG PIPELINE DESIGN ───────────────────────────────────────────")
    print(result["messages"][-1].content)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "invoke"

    print(f"\nStarting HF Deep Agent — mode: [{mode}]")
    print("=" * 60)

    modes = {
        "invoke":     run_invoke,       # python agent.py invoke
        "stream":     run_streaming,    # python agent.py stream
        "multi":      run_multi_turn,   # python agent.py multi
        "rag":        run_rag_advisor,  # python agent.py rag
    }

    if mode not in modes:
        print(f"Unknown mode '{mode}'. Choose: {list(modes.keys())}")
        sys.exit(1)

    asyncio.run(modes[mode]())
