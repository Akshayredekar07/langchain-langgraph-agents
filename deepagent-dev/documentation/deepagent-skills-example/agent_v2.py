# agent.py — HuggingFace Deep Agent
# Run modes:
#   python agent.py invoke   — single-shot query
#   python agent.py debug    — single-shot with full verification output
#   python agent.py stream   — streaming output
#   python agent.py multi    — multi-turn conversation
#   python agent.py rag      — RAG advisor skill demo

import asyncio
import os
from pathlib import Path
from typing import Any, Literal, Protocol, overload
from uuid import UUID

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend


import deepagents

# ── Compat shim: works with both sync and async factory 
create_deep_agent = getattr(
    deepagents,
    "async_create_deep_agent",
    deepagents.create_deep_agent,
)

load_dotenv()

BASE_DIR = Path(__file__).parent
AGENTS_FILE = BASE_DIR / "AGENTS.md"
SKILLS_DIR_PATH = BASE_DIR / "skills"
MEMORIES_DIR_PATH = BASE_DIR / "memories"

AGENTS_MD = "/AGENTS.md"
SKILLS_DIR = "/skills/"

HF_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

SEP = "-" * 60


class DeepAgentRunnable(Protocol):
    async def ainvoke(self, input: Any, config: RunnableConfig | None = None) -> Any: ...
    def astream(self, input: Any, config: RunnableConfig | None = None) -> Any: ...


class VerificationCallbackHandler(BaseCallbackHandler):
    """
    Intercepts LLM calls and tool calls to prove:

    (a) MEMORY loaded    — AGENTS.md first line appears in the system message
                           that is sent to the LLM on the very first call.

    (b) SKILL injected   — skill names appear in the system message at startup
                           (proof that SkillsMiddleware injected the descriptions).

    (c) SKILL activated  — a read_file call whose input path ends in SKILL.md
                           fires when the agent decides to use a skill body.
                           This is progressive disclosure confirmed.

    (d) All tool calls   — every tool call is printed so you can follow the
                           full execution path.
    """

    def __init__(self, verbose_system_prompt: bool = False):
        self.verbose_system_prompt = verbose_system_prompt
        self._first_llm_call = True
        self._skill_reads: list[str] = []
        self._tool_calls: list[str] = []

    def on_chat_model_start(
        self,
        serialized: dict,
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if not self._first_llm_call:
            return
        self._first_llm_call = False

        # Pull system message from the first batch
        system_content = ""
        for msg in messages[0]:
            cls_name = msg.__class__.__name__
            msg_type = getattr(msg, "type", cls_name)
            if "system" in str(msg_type).lower() or "System" in cls_name:
                system_content = (
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                )
                break

        print(f"\n[VERIFY] First LLM call — inspecting system prompt")
        print(SEP)

        # (a) Memory check
        agents_md_path = AGENTS_FILE
        if agents_md_path.exists():
            first_line = (
                agents_md_path.read_text(encoding="utf-8").strip().splitlines()[0]
            )
            if first_line in system_content:
                print("[VERIFY] MEMORY  OK   — AGENTS.md content found in system prompt")
            else:
                print("[VERIFY] MEMORY  WARN — AGENTS.md content NOT found in system prompt")
                print(f"         Searched for first line: '{first_line[:70]}'")
        else:
            print(f"[VERIFY] MEMORY  WARN — AGENTS.md not found at: {AGENTS_FILE}")

        # (b) Skill description injection check
        skills_path = SKILLS_DIR_PATH
        if skills_path.exists():
            for skill_dir in skills_path.iterdir():
                if skill_dir.is_dir():
                    name = skill_dir.name
                    found = name in system_content
                    status = "OK  " if found else "WARN"
                    detail = "(description injected)" if found else "(NOT found in prompt)"
                    print(f"[VERIFY] SKILL   {status}  — '{name}' {detail}")
        else:
            print(f"[VERIFY] SKILLS  WARN — skills dir not found: {SKILLS_DIR_PATH}")

        # (c) Optional: full system prompt dump
        if self.verbose_system_prompt and system_content:
            print(f"\n[VERIFY] FULL SYSTEM PROMPT ({len(system_content)} chars):")
            print(SEP)
            print(system_content[:2000])
            if len(system_content) > 2000:
                print(f"... [{len(system_content) - 2000} more chars hidden]")

        print(SEP)

    def on_tool_start(
        self,
        serialized: dict,
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name", "unknown_tool")
        self._tool_calls.append(tool_name)

        # Skill body activation: read_file call on a SKILL.md path
        if tool_name == "read_file" and "SKILL.md" in str(input_str):
            skill_path = input_str.strip().strip('"').strip("'")
            parts = Path(skill_path).parts
            skill_name = parts[-2] if len(parts) >= 2 else skill_path
            self._skill_reads.append(skill_name)
            print(f"\n[VERIFY] SKILL ACTIVATED — '{skill_name}' SKILL.md body being read")
            print(f"         Path  : {skill_path}")
            print(f"         Reason: task matched skill description (progressive disclosure)")
        else:
            print(f"[VERIFY] tool_call — {tool_name}({input_str[:80].strip()})")

    def on_tool_end(self, output: str, *, run_id: UUID, **kwargs: Any) -> None:
        pass  # tool output noise suppressed; visible in LangSmith

    def print_summary(self) -> None:
        print(f"\n[VERIFY] SESSION SUMMARY")
        print(SEP)
        print(f"  Total tool calls  : {len(self._tool_calls)}")
        print(f"  Skills activated  : {len(self._skill_reads)}")
        for s in self._skill_reads:
            print(f"    -> {s}")
        print(f"  All tools used    : {', '.join(self._tool_calls) or 'none'}")
        print(SEP)


def dump_system_prompt(result: dict, max_chars: int = 3000) -> None:
    """
    Inspect the system message that was sent to the LLM in this invocation.
    Confirms AGENTS.md is embedded and skill descriptions are injected.

    Usage:
        result = await agent.ainvoke(...)
        dump_system_prompt(result)
    """
    messages = result.get("messages", [])
    if not messages:
        print("[dump_system_prompt] No messages in result.")
        return

    system_msg = None
    for msg in messages:
        cls_name = msg.__class__.__name__
        msg_type = getattr(msg, "type", cls_name)
        if "system" in str(msg_type).lower() or "System" in cls_name:
            system_msg = msg
            break

    if system_msg is None:
        print("[dump_system_prompt] No system message found.")
        print(f"  Types: {[m.__class__.__name__ for m in messages[:5]]}")
        return

    content = (
        system_msg.content
        if isinstance(system_msg.content, str)
        else str(system_msg.content)
    )

    print(f"\n[METHOD 2] System prompt sent to LLM ({len(content)} chars)")
    print(SEP)
    print(content[:max_chars])
    if len(content) > max_chars:
        print(f"\n... [truncated — {len(content) - max_chars} more chars]")
    print(SEP)

    # Memory check
    agents_md_path = AGENTS_FILE
    if agents_md_path.exists():
        first_line = agents_md_path.read_text(encoding="utf-8").strip().splitlines()[0]
        found = first_line in content
        print(f"[METHOD 2] AGENTS.md present in prompt : {'YES' if found else 'NO'}")

    # Skill checks
    for skill_dir in SKILLS_DIR_PATH.iterdir():
        if skill_dir.is_dir():
            found = skill_dir.name in content
            print(
                f"[METHOD 2] Skill '{skill_dir.name}' description in prompt : "
                f"{'YES' if found else 'NO'}"
            )


@overload
async def build_agent(debug_callbacks: Literal[False] = False) -> DeepAgentRunnable: ...


@overload
async def build_agent(
    debug_callbacks: Literal[True],
) -> tuple[DeepAgentRunnable, VerificationCallbackHandler]: ...


async def build_agent(
    debug_callbacks: bool = False,
) -> DeepAgentRunnable | tuple[DeepAgentRunnable, VerificationCallbackHandler]:
    """
    Args:
        debug_callbacks: When True, returns (agent, VerificationCallbackHandler)
                         so the handler can be passed into config["callbacks"].
    Returns:
        agent if debug_callbacks is False, else (agent, VerificationCallbackHandler)
    """

    model = init_chat_model(
        model="claude-sonnet-4-6",
        model_provider="anthropic",
        temperature=0.2,
        max_tokens=2048,
    )

    connections: dict[str, Any] = {
        "huggingface": {
            "transport": "http",
            "url": "https://huggingface.co/mcp",
            "headers": {"Authorization": f"Bearer {HF_TOKEN}"},
        }
    }
    client = MultiServerMCPClient(connections)
    mcp_tools = await client.get_tools()

    research_agent = create_deep_agent(
        model=model,
        tools=mcp_tools,

        # AGENTS.md — loaded by MemoryMiddleware into the system prompt.
        # The agent can update this file with edit_file to persist new learnings.
        memory=[AGENTS_MD],

        # Skills directory — SkillsMiddleware reads frontmatter of each SKILL.md
        # at startup and injects name + description into the system prompt.
        # Full SKILL.md body is only read_file'd when the agent decides to use it.
        skills=[SKILLS_DIR],

        # virtual_mode=True — agent tools use POSIX virtual paths like /AGENTS.md.
        # Persist all file writes under /memories/ on disk, while keeping the
        # default StateBackend for everything else.
        backend=CompositeBackend(
            default=StateBackend(),
            routes={
                "/memories/": FilesystemBackend(
                    root_dir=str(MEMORIES_DIR_PATH),
                    virtual_mode=True,
                ),
            },
        ),

        checkpointer=MemorySaver(),

        system_prompt=(
            "You are an expert AI/ML research assistant with live access to the "
            "HuggingFace Hub via MCP tools. Always plan your steps using write_todos "
            "before executing. Use HuggingFace MCP tools to search models, datasets, "
            "papers, and documentation. When results are long, write them to files "
            "with write_file to avoid context overflow. When calling write_file, you "
            "must always provide both required arguments: file_path and content. Use "
            "virtual POSIX paths under /memories/, such as /memories/report.md, "
            "never Windows paths. Do not call write_file with empty or partial "
            "arguments. Apply relevant skills for structured task execution. When calling write_file, you must always provide both required arguments: file_path and content."
            
        ),

        debug=True,
    )

    agent = await research_agent if asyncio.iscoroutine(research_agent) else research_agent

    print(f"\n[OK] Agent built with:")
    print(f"   Memory  : {AGENTS_MD} -> {AGENTS_FILE}")
    print(f"   Skills  : {SKILLS_DIR} -> {SKILLS_DIR_PATH}")
    print(f"   Files   : /memories/ -> {MEMORIES_DIR_PATH}")
    print(f"   Backend : CompositeBackend(default=StateBackend, /memories/=FilesystemBackend)")
    print(f"   Memory  : MemorySaver (thread-level persistence)")

    if debug_callbacks:
        handler = VerificationCallbackHandler(verbose_system_prompt=False)
        return agent, handler

    return agent


def make_config(thread_id: str = "default", callbacks: list | None = None) -> RunnableConfig:
    config: dict = {"configurable": {"thread_id": thread_id}}
    if callbacks:
        config["callbacks"] = callbacks
    return config  # type: ignore


async def run_invoke():
    """Standard single-shot — no verification output."""
    agent = await build_agent()
    config = make_config("demo-invoke")

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": (
            "Find the top 3 embedding models on HuggingFace for a "
            "production RAG pipeline. Compare them in a table with "
            "dimensions, max tokens, and license."
        )}]},
        config=config,
    )

    print("\n── FINAL ANSWER ──")
    print(result["messages"][-1].content)

    if result.get("files"):
        print("\n── FILES WRITTEN BY AGENT ──")
        for fname in result["files"]:
            print(f"   {fname}")


async def run_debug():
    """
    Full verification mode. Shows in the terminal:

      [VERIFY] MEMORY  OK   — AGENTS.md first line found in system prompt
      [VERIFY] SKILL   OK   — skill name found in system prompt (startup injection)
      [VERIFY] tool_call    — every tool call as it fires
      [VERIFY] SKILL ACTIVATED — read_file on SKILL.md (progressive disclosure)
      [VERIFY] SESSION SUMMARY — totals at the end
      [METHOD 2] system prompt dump — post-invoke raw inspection

    Run with: python agent.py debug
    """
    result = await build_agent(debug_callbacks=True)
    agent, handler = result

    # Method 1: pass handler via config callbacks
    config = make_config("demo-debug", callbacks=[handler])

    print("\n── RUNNING WITH VERIFICATION (Method 1 + Method 2) ──")
    print(SEP)

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": (
            "I need a multilingual sentence transformer model for a RAG pipeline. "
            "Find options on HuggingFace and recommend the best one."
        )}]},
        config=config,
    )

    handler.print_summary()          # Method 1 summary
    dump_system_prompt(result, 2000) # Method 2 post-invoke inspection

    print("\n── FINAL ANSWER ──")
    print(result["messages"][-1].content)


async def run_streaming():
    """Streaming — paper search with file offloading."""
    agent = await build_agent()
    config = make_config("demo-stream")

    print("\n── STREAMING ──")
    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": (
            "Search for RAG-related papers on HuggingFace and write a 3-bullet "
            "summary of recent trends. Save the full paper list to a file."
        )}]},
        config=config,
    ):
        for node_name, node_output in chunk.items():
            if not isinstance(node_output, dict):
                continue
            raw = node_output.get("messages")
            if raw is None:
                continue
            messages = raw.value if hasattr(raw, "value") else raw
            if not isinstance(messages, list):
                messages = [messages]
            for msg in messages:
                content = getattr(msg, "content", None)
                if content:
                    print(f"[{node_name}] {content}", flush=True)


async def run_multi_turn():
    """Multi-turn — uses same thread_id across 3 turns."""
    agent = await build_agent()
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
        print(f"\n[TURN {turn_num} - USER] {user_msg}")
        print(SEP)

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_msg}]},
            config=config,
        )

        assistant_msg = result["messages"][-1].content
        print(f"[AGENT] {assistant_msg}")


async def run_rag_advisor():
    """Triggers the hf-rag-advisor skill via a RAG-specific query."""
    agent = await build_agent()
    config = make_config("demo-rag-advisor")

    print("\n── RAG ADVISOR MODE ──")
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": (
            "Design a complete RAG pipeline for a legal document search system. "
            "Recommend HuggingFace models for each stage: embedding, reranking, "
            "and generation. Include integration code using LangChain."
        )}]},
        config=config,
    )

    print("\n── RAG PIPELINE DESIGN ──")
    print(result["messages"][-1].content)


MODES = {
    "invoke": run_invoke,
    "debug":  run_debug,
    "stream": run_streaming,
    "multi":  run_multi_turn,
    "rag":    run_rag_advisor,
}

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "invoke"

    print(f"\nStarting HF Deep Agent — mode: [{mode}]")
    print("──" * 60)

    if mode not in MODES:
        print(f"Unknown mode '{mode}'. Choose: {list(MODES.keys())}")
        sys.exit(1)

    asyncio.run(MODES[mode]())
