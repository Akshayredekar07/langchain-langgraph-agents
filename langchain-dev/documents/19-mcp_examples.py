"""
LangChain v1.x + MCP: real-world style examples.

This file intentionally does NOT use Slack, GitHub, or PostgreSQL.

The examples model the kinds of MCP integrations commonly used in
developer/enterprise agents:

    filesystem -> project files
    git        -> repository state
    fetch      -> public web documentation
    time       -> timezone/date operations
    memory     -> persistent knowledge
    calculator -> deterministic business calculation

The MCP ecosystem's official reference servers include Filesystem, Git,
Fetch, Memory, Sequential Thinking, and Time. The examples below use
in-process FastMCP servers so the file remains directly runnable and easy
to study. In production, these server implementations can be replaced by
remote MCP servers without changing the LangChain agent pattern.

Install:

    uv add "langchain[mcp]" fastmcp langchain-nebius python-dotenv

.env:

    NEBIUS_API_KEY=your-key

Run:

    python langchain_mcp_real_world_examples.py
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import FastMCP

from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_nebius import ChatNebius

load_dotenv()


# ============================================================================
# Shared model
# ============================================================================

def get_model():
    return ChatNebius(model="Qwen/Qwen3-30B-A3B")


# ============================================================================
# 1. FILESYSTEM MCP
#
# Real use:
#
#   "Find the configuration file for the payment service and explain it."
#
# The agent should NOT receive arbitrary Python filesystem access directly.
# The filesystem MCP server should expose a restricted workspace.
# ============================================================================

WORKSPACE = Path("./mcp_workspace").resolve()
WORKSPACE.mkdir(parents=True, exist_ok=True)

(WORKSPACE / "config.yaml").write_text(
    """app:
  name: payment-service
  environment: development

database:
  pool_size: 10
  timeout_seconds: 30

features:
  retry_enabled: true
""",
    encoding="utf-8",
)

(WORKSPACE / "README.md").write_text(
    """# Payment Service

The service processes customer payments.

Important files:
- config.yaml
- deployment.md
- retry-policy.md
""",
    encoding="utf-8",
)

(WORKSPACE / "deployment.md").write_text(
    """# Deployment

The payment service is deployed as a container.

Development uses a single instance.
Production uses multiple replicas.
""",
    encoding="utf-8",
)

filesystem_server = FastMCP("filesystem")


def safe_path(relative_path: str) -> Path:
    root = WORKSPACE.resolve()
    path = (root / relative_path).resolve()

    if path != root and root not in path.parents:
        raise ValueError("Path is outside the allowed workspace")

    return path


@filesystem_server.tool
def list_files(directory: str = ".") -> list[str]:
    """List files in the allowed workspace directory."""
    path = safe_path(directory)

    if not path.is_dir():
        raise ValueError("Not a directory")

    return [
        str(item.relative_to(WORKSPACE))
        for item in path.iterdir()
    ]


@filesystem_server.tool
def read_file(path: str) -> str:
    """Read a text file from the allowed workspace."""
    file_path = safe_path(path)

    if not file_path.is_file():
        raise ValueError("File does not exist")

    return file_path.read_text(encoding="utf-8")


@filesystem_server.tool
def search_files(query: str) -> list[str]:
    """Search filenames and text contents in the allowed workspace."""
    matches = []

    for file_path in WORKSPACE.rglob("*"):
        if not file_path.is_file():
            continue

        relative = str(file_path.relative_to(WORKSPACE))

        if query.lower() in relative.lower():
            matches.append(relative)
            continue

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        if query.lower() in content.lower():
            matches.append(relative)

    return matches


async def example_01_filesystem_agent():
    """
    User request:

        "Find the retry configuration and explain it."

    MCP is acting as the controlled interface to the filesystem.
    """

    async with MCPAdapter(filesystem_server) as adapter:
        tools = await adapter.list_tools()

        agent = create_agent(get_model(), tools)

        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Find the retry configuration in the project "
                            "workspace and explain what it does."
                        ),
                    }
                ]
            }
        )

        print(result["messages"][-1].content)


# ============================================================================
# 2. GIT MCP
#
# Real use:
#
#   "What changed recently?"
#   "Why is this file modified?"
#   "Summarize the current branch."
#
# A production Git MCP server would wrap git commands/API safely.
# ============================================================================

git_server = FastMCP("git")


@git_server.tool
def git_status() -> dict:
    """Return a simplified repository status."""
    return {
        "branch": "feature/payment-retry",
        "clean": False,
        "modified": [
            "config.yaml",
            "src/retry.py",
        ],
        "untracked": [
            "tests/test_retry.py",
        ],
    }


@git_server.tool
def git_log(limit: int = 5) -> list[dict]:
    """Return recent commits."""
    commits = [
        {
            "hash": "a81f2c1",
            "message": "Add retry policy",
            "author": "Akshay",
        },
        {
            "hash": "92bc104",
            "message": "Improve payment timeout handling",
            "author": "Priya",
        },
        {
            "hash": "7d821aa",
            "message": "Add payment service metrics",
            "author": "Rahul",
        },
    ]

    return commits[:limit]


@git_server.tool
def git_diff(file: str | None = None) -> str:
    """Return a simplified diff."""
    if file == "config.yaml":
        return """- timeout_seconds: 20
+ timeout_seconds: 30
+ retry_enabled: true
"""

    return """diff --git a/config.yaml b/config.yaml
@@
-timeout_seconds: 20
+timeout_seconds: 30
+retry_enabled: true

diff --git a/src/retry.py b/src/retry.py
@@
+MAX_RETRIES = 3
"""


async def example_02_git_agent():
    """
    Real developer workflow:

        filesystem MCP + git MCP

    The agent can correlate the file contents with repository history.
    """

    async with (
        MCPAdapter(filesystem_server) as filesystem,
        MCPAdapter(git_server) as git,
    ):
        filesystem_tools = await filesystem.list_tools()
        git_tools = await git.list_tools()

        agent = create_agent(
            get_model(),
            filesystem_tools + git_tools,
        )

        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Investigate the current payment-service changes. "
                            "Check git status, recent commits, and the configuration "
                            "file. Explain what changed and why it may matter."
                        ),
                    }
                ]
            }
        )

        print(result["messages"][-1].content)


# ============================================================================
# 3. FETCH MCP
#
# Real use:
#
#   Agent reads public documentation, release notes, RFCs, or web pages.
#
# Important:
#   Fetch is an open-world capability. In production, use URL/host policies
#   and SSRF protection. Never treat arbitrary outbound HTTP as harmless.
# ============================================================================

fetch_server = FastMCP("fetch")


@fetch_server.tool
async def fetch_url(url: str) -> str:
    """
    Fetch a public URL.

    This demo intentionally does not perform arbitrary network access.
    It represents the interface exposed by a real Fetch MCP server.
    """
    allowed = {
        "https://docs.example.com/retry-policy": (
            "# Retry Policy\n\n"
            "Payments are retried up to three times for transient failures."
        ),
        "https://docs.example.com/timeouts": (
            "# Timeouts\n\n"
            "The payment API timeout is 30 seconds."
        ),
    }

    if url not in allowed:
        raise ValueError(
            "URL is not in the demo allowlist"
        )

    return allowed[url]


async def example_03_fetch_agent():
    """
    The important production concept is not "make an HTTP request".

    It is:

        agent
          |
          +-- filesystem MCP
          |
          +-- git MCP
          |
          +-- fetch MCP
          |
          v
        answer

    Each capability has its own boundary and policy.
    """

    async with MCPAdapter(fetch_server) as adapter:
        tools = await adapter.list_tools()

        agent = create_agent(get_model(), tools)

        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Read the retry policy documentation at "
                            "https://docs.example.com/retry-policy "
                            "and summarize it."
                        ),
                    }
                ]
            }
        )

        print(result["messages"][-1].content)


# ============================================================================
# 4. TIME MCP
#
# Real use:
#
#   - scheduling
#   - incident timestamps
#   - timezone conversion
#   - deadline calculations
#
# This should be a deterministic tool, not something the LLM calculates.
# ============================================================================

time_server = FastMCP("time")


@time_server.tool
def current_time() -> str:
    """Return current UTC time."""
    return datetime.now(timezone.utc).isoformat()


@time_server.tool
def convert_utc_to_offset(utc_iso: str, offset_hours: int) -> str:
    """Convert a UTC timestamp to a fixed hour offset."""
    value = datetime.fromisoformat(
        utc_iso.replace("Z", "+00:00")
    )

    from datetime import timedelta

    converted = value + timedelta(hours=offset_hours)

    return converted.isoformat()


async def example_04_time_agent():
    async with MCPAdapter(time_server) as adapter:
        tools = await adapter.list_tools()

        agent = create_agent(get_model(), tools)

        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "What is the current UTC time, and what time is "
                            "that in India?"
                        ),
                    }
                ]
            }
        )

        print(result["messages"][-1].content)


# ============================================================================
# 5. MEMORY MCP
#
# Real use:
#
#   Store/retrieve durable application knowledge:
#
#       customer preferences
#       project decisions
#       architecture decisions
#       recurring facts
#
# This is different from LangGraph checkpointer state.
#
# Checkpointer:
#     conversation/workflow state
#
# MCP memory:
#     external durable knowledge capability
# ============================================================================

memory_server = FastMCP("memory")

MEMORY: dict[str, list[str]] = {}


@memory_server.tool
def remember(key: str, value: str) -> str:
    """Store a durable piece of application knowledge."""
    MEMORY.setdefault(key, []).append(value)
    return "Memory stored"


@memory_server.tool
def recall(key: str) -> list[str]:
    """Retrieve stored application knowledge."""
    return MEMORY.get(key, [])


async def example_05_memory_agent():
    async with MCPAdapter(memory_server) as adapter:
        tools = await adapter.list_tools()

        agent = create_agent(get_model(), tools)

        await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Remember that the payment service uses a "
                            "30-second timeout and three retries."
                        ),
                    }
                ]
            }
        )

        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "What do we know about the payment service?"
                        ),
                    }
                ]
            }
        )

        print(result["messages"][-1].content)


# ============================================================================
# 6. MULTIPLE MCP SERVERS
#
# This is where MCP becomes useful.
#
# One agent:
#
#     Filesystem MCP
#            |
#     Git MCP ----+
#                 |
#     Fetch MCP --+--> LangChain Agent --> model
#                 |
#     Time MCP ---+
#                 |
#     Memory MCP -+
#
# The agent receives all capabilities as normal LangChain tools.
# ============================================================================

async def example_06_multi_mcp_agent():
    async with (
        MCPAdapter(filesystem_server) as filesystem,
        MCPAdapter(git_server) as git,
        MCPAdapter(fetch_server) as fetch,
        MCPAdapter(time_server) as time,
        MCPAdapter(memory_server) as memory,
    ):
        tools = []

        tools.extend(await filesystem.list_tools())
        tools.extend(await git.list_tools())
        tools.extend(await fetch.list_tools())
        tools.extend(await time.list_tools())
        tools.extend(await memory.list_tools())

        agent = create_agent(get_model(), tools)

        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Investigate the payment-service retry configuration. "
                            "Read the project configuration, check recent git "
                            "changes, read the retry policy documentation, and "
                            "include the current UTC time in the report. "
                            "Do not modify any files."
                        ),
                    }
                ]
            }
        )

        print(result["messages"][-1].content)


# ============================================================================
# 7. REAL PRODUCTION PATTERN: READ VS WRITE
#
# Never give one general agent unlimited write access.
#
# Read MCP:
#     list_files
#     read_file
#     git_status
#     git_log
#
# Write MCP:
#     write_file
#     edit_file
#     git_commit
#     deploy
#
# The write agent should have stronger authorization.
# ============================================================================

write_server = FastMCP("filesystem-write")


@write_server.tool(
    annotations={
        "destructiveHint": True,
    }
)
def write_file(path: str, content: str) -> str:
    """Write a file inside the allowed workspace."""
    file_path = safe_path(path)

    file_path.write_text(content, encoding="utf-8")

    return f"Wrote {path}"


async def example_07_write_boundary():
    """
    Demonstrates the architectural boundary.

    Read capabilities are provided by filesystem_server.
    Write capabilities are isolated in write_server.

    In production, this boundary would normally also include:
        - authentication
        - authorization
        - audit logging
        - human approval
        - rate limiting
    """

    async with MCPAdapter(write_server) as adapter:
        tools = await adapter.list_tools()

        for tool in tools:
            print(
                tool.name,
                tool.metadata,
            )


# ============================================================================
# 8. PRODUCTION-STYLE CODE AGENT
#
# This combines the most realistic developer workflow from the servers above.
#
# User:
#
#   "Review the retry implementation."
#
# Agent:
#
#   filesystem -> read code
#   git        -> inspect changes
#   fetch      -> read external policy
#   memory     -> retrieve architecture decision
#   time       -> timestamp the report
#
# No GitHub, Slack, or PostgreSQL is necessary to demonstrate MCP's value.
# ============================================================================

async def example_08_code_review_agent():
    async with (
        MCPAdapter(filesystem_server) as filesystem,
        MCPAdapter(git_server) as git,
        MCPAdapter(fetch_server) as fetch,
        MCPAdapter(memory_server) as memory,
        MCPAdapter(time_server) as time,
    ):
        tools = []

        tools.extend(await filesystem.list_tools())
        tools.extend(await git.list_tools())
        tools.extend(await fetch.list_tools())
        tools.extend(await memory.list_tools())
        tools.extend(await time.list_tools())

        agent = create_agent(get_model(), tools)

        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Perform a read-only review of the payment retry "
                            "implementation. Inspect the workspace and git "
                            "changes. Check the retry policy documentation. "
                            "Recall any stored architecture decisions. "
                            "Report potential configuration or implementation "
                            "problems. Do not change anything."
                        ),
                    }
                ]
            }
        )

        print(result["messages"][-1].content)


# ============================================================================
# 9. MCP TOOL DISCOVERY
#
# Production applications should not blindly assume a server's tool catalog.
# Discover tools at runtime.
# ============================================================================

async def example_09_tool_discovery():
    async with MCPAdapter(filesystem_server) as adapter:
        tools = await adapter.list_tools()

        print("Filesystem MCP tools:")
        for tool in tools:
            print(
                json.dumps(
                    {
                        "name": tool.name,
                        "description": tool.description,
                    },
                    indent=2,
                )
            )


# ============================================================================
# 10. MAIN
# ============================================================================

EXAMPLES = {
    "1": example_01_filesystem_agent,
    "2": example_02_git_agent,
    "3": example_03_fetch_agent,
    "4": example_04_time_agent,
    "5": example_05_memory_agent,
    "6": example_06_multi_mcp_agent,
    "7": example_07_write_boundary,
    "8": example_08_code_review_agent,
    "9": example_09_tool_discovery,
}


async def main():
    print(
        """
LangChain + MCP examples

1. Filesystem MCP
2. Git MCP
3. Fetch MCP
4. Time MCP
5. Memory MCP
6. Multiple MCP servers
7. Read/write capability separation
8. Production-style code review agent
9. MCP tool discovery
"""
    )

    # Change this number while learning.
    choice = "6"

    await EXAMPLES[choice]()


if __name__ == "__main__":
    asyncio.run(main())
