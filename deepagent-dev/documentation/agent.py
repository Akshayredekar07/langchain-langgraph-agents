# agent.py
import asyncio
import os
from typing import Any
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
import deepagents
import inspect

create_deep_agent = getattr(
    deepagents,
    "async_create_deep_agent",
    deepagents.create_deep_agent,
)

load_dotenv()

HF_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]


async def build_agent():
    """
    Build an async deep agent wired to the HuggingFace MCP server.

    Why async_create_deep_agent?
    MCP tools are inherently async (network I/O). The sync version
    doesn't support them — use async_create_deep_agent for all MCP setups.
    """

    # ── 1. LLM  ────────────────────────────────────────────────────────────────
    # HuggingFace Inference API — requires tool-calling support.
    # Phi-3-mini supports tool calling via the HF Inference API.
    model = init_chat_model(
        # model="microsoft/Phi-3-mini-4k-instruct",
        model="claude-sonnet-4-6",
        model_provider="anthropic",
        temperature=0.7,
        max_tokens=1024,
    )

    # ── 2. HuggingFace MCP Server ──────────────────────────────────────────────
    # Official HF MCP endpoint: https://huggingface.co/mcp
    # Transport: streamable HTTP (MCP spec 2025-03-26)
    # Auth: Bearer token passed as header
    #
    # Built-in tools exposed:
    #   - model_search          → search HF Hub models
    #   - dataset_search        → search HF Hub datasets
    #   - spaces_semantic_search→ find Gradio apps/Spaces
    #   - papers_semantic_search→ search ML papers
    #   - documentation_search  → search HF docs
    #   - hub_repository_details→ get model/dataset/space info
    connections: dict[str, Any] = {
        "huggingface": {
            "transport": "http",                  # streamable HTTP
            "url": "https://huggingface.co/mcp",
            "headers": {
                "Authorization": f"Bearer {HF_TOKEN}",
            },
        }
    }
    client = MultiServerMCPClient(connections)

    # ── 3. Load MCP tools as LangChain BaseTool objects ────────────────────────
    mcp_tools = await client.get_tools()

    members = inspect.getmembers(mcp_tools)
    for name, value in members:
        print(f"{name:30} → {type(value).__name__}")

    print(f" Loaded {len(mcp_tools)} HuggingFace MCP tools:")
    for t in mcp_tools:
        print(f"   • {t.name}: {t.description[:80]}...")

    # ── 4. Build the deep agent (supports async/sync factory across versions) ─
    agent_or_coro = create_deep_agent(
        model=model,
        tools=mcp_tools,                           # MCP tools injected here
        system_prompt=(
            "You are an AI research assistant with access to the Hugging Face Hub. "
            "Always plan your steps before acting. Use the HuggingFace tools to "
            "search models, datasets, papers, and documentation as needed."
        ),
        debug=True,
    )

    agent = await agent_or_coro if asyncio.iscoroutine(agent_or_coro) else agent_or_coro
    return agent


async def run_invoke():
    """Single-shot invocation."""
    agent = await build_agent()

    result = await agent.ainvoke({
        "messages": [
            {
                "role": "user",
                "content": (
                    "Find the top 3 text-generation models on Hugging Face "
                    # "and summarize what each one is good for."
                )
            }
        ]
    })

    print("\n── FINAL ANSWER ──────────────────────────────────────────")
    print(result["messages"][-1].content)


async def run_streaming():
    """Streaming invocation — better for long agentic tasks."""
    agent = await build_agent()

    print("\n── STREAMING ─────────────────────────────────────────────")
    async for chunk in agent.astream({
        "messages": [
            {
                "role": "user",
                "content": (
                    "Search for RAG-related papers on Hugging Face "
                    "and write a 3-bullet summary of recent trends."
                )
            }
        ]
    }):
        for node_name, node_output in chunk.items():
            if not isinstance(node_output, dict):
                continue

            raw = node_output.get("messages")
            if raw is None:
                continue

            # ── Fix: Overwrite is a LangGraph wrapper around a list ──────────
            # Unwrap it if needed before iterating
            messages = raw.value if hasattr(raw, "value") else raw
            if not isinstance(messages, list):
                messages = [messages]

            for msg in messages:
                content = getattr(msg, "content", None)
                if content:
                    print(f"[{node_name}] {content}", flush=True)

async def run_multi_turn():
    """Multi-turn conversation with the agent."""
    agent = await build_agent()

    history = []

    turns = [
        "Search for sentence-transformers models on Hugging Face.",
        "Which of those models would work best for semantic search in a RAG pipeline?",
        "Show me details about the top one.",
    ]

    for user_msg in turns:
        print(f"\n[USER] {user_msg}")
        history.append({"role": "user", "content": user_msg})

        result = await agent.ainvoke({"messages": history})

        assistant_msg = result["messages"][-1].content
        print(f"[AGENT] {assistant_msg[:300]}...")

        # Append assistant reply to history for next turn
        history.append({"role": "assistant", "content": assistant_msg})


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "invoke"

    if mode == "stream":
        asyncio.run(run_streaming())
    elif mode == "multi":
        asyncio.run(run_multi_turn())
    else:
        asyncio.run(run_invoke())