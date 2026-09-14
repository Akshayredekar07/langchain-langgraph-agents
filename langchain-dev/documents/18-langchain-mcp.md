# **LangChain MCP Notes**

## **1. Overview**

1. `langchain.mcp` is the built-in namespace for connecting LangChain agents to MCP (Model Context Protocol) servers.
2. It replaces the standalone `langchain-mcp-adapters` package. `MultiServerMCPClient` (the old entry point) collapses into a single class: `MCPAdapter`.
3. It is built on top of FastMCP, which supplies transport inference, protocol negotiation, connection management, caching, and authentication. `langchain.mcp` is the thin LangChain-specific layer on top.
4. `MCPAdapter.list_tools()` discovers a server's tools and returns them as standard LangChain tools — pass them directly to `create_agent`.
5. Requires `langchain[mcp]>=1.4.0` and `fastmcp>=4.0.0`. As of this writing the `1.4.x` line is still pre-release (alpha), so `pip install --pre` is required — see section 14.

```python
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_nebius import ChatNebius

load_dotenv()


async def quickstart():
    model = ChatNebius(model="Qwen/Qwen3-30B-A3B")
    async with MCPAdapter("https://example.com/mcp") as adapter:
        tools = await adapter.list_tools()
        agent = create_agent(model, tools)
        return await agent.ainvoke(
            {"messages": [{"role": "user", "content": "What tools do you have?"}]}
        )
```

## **2. Install and Beta Status**

1. Install path: `pip install "langchain[mcp]"` — this pulls in FastMCP as a dependency.
2. Importing from `langchain.mcp` raises a `LangChainBetaWarning` once per process. This is intentional and signals the API may still change.
3. The namespace requires `fastmcp>=4.0.0` — the general-availability FastMCP line whose multi-server routing (`MCPConfig`, `ClientGroup`) the adapter depends on.
4. If migrating from the standalone `langchain-mcp-adapters` package: drop that dependency, add the `mcp` extra, and swap `MultiServerMCPClient` for `MCPAdapter`. The connection config keeps the same `MCPConfig` shape (`{"mcpServers": {...}}`).

```python
from dotenv import load_dotenv
from langchain.mcp import MCPAdapter

load_dotenv()

# Fired once per process the first time langchain.mcp is imported:
# LangChainBetaWarning: the langchain.mcp API may change without notice.


async def list_remote_tools(url: str):
    async with MCPAdapter(url) as adapter:
        tools = await adapter.list_tools()
        return [tool.name for tool in tools]
```

## **3. MCPAdapter and Transports**

1. `MCPAdapter` infers the transport from whatever target it is given — the target's type decides how the connection is made.
2. Supported targets:

| Target type | Behavior |
|---|---|
| `str` (http/https URL) | Streamable HTTP. Rejected if it looks like a filesystem path instead of a URL. |
| `Path` | Launched as a subprocess over stdio, one subprocess per adapter. |
| Pre-configured transport object (e.g. `StreamableTransport`) | Full manual control over the FastMCP transport. |
| In-process `FastMCP` server instance | Connected in-memory — no subprocess, no socket. Ideal for tests. |
| `MCPConfig` dict (`{"mcpServers": {...}}`) | Several servers behind one aggregate adapter (see section 5). |
| Pre-built `fastmcp.Client` | Full control over transport, caching, and protocol negotiation. |

3. A `str` target must look like an `http`/`https` URL. FastMCP resolves a string by testing it as a filesystem path first, so a string naming an existing `.py`/`.js` file would otherwise launch it as a subprocess — `MCPAdapter` rejects strings that don't match a URL shape to avoid that ambiguity.
4. The in-process `FastMCP` target is the most useful one for local development and notes/testing: no network, no subprocess, fully deterministic.

```python
from dotenv import load_dotenv
from fastmcp import FastMCP
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_nebius import ChatNebius

load_dotenv()

server = FastMCP("notes-demo")


@server.tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


async def run_in_memory():
    model = ChatNebius(model="Qwen/Qwen3-30B-A3B")
    async with MCPAdapter(server) as adapter:
        tools = await adapter.list_tools()
        agent = create_agent(model, tools)
        return await agent.ainvoke(
            {"messages": [{"role": "user", "content": "What is 14 plus 28?"}]}
        )
```

## **4. Connection Lifecycle**

1. `MCPAdapter` is an async context manager. Entering it connects the underlying client; exiting it releases the connection.
2. Discovery (`list_tools()`) must happen inside the `async with` block, but the returned tools hold a reference to the client, so they remain callable after the context exits.
3. The tools are reentrant: each invocation opens the client, runs the call, and releases it — whether or not a connection is already held elsewhere. This means a single agent run opens one session per tool call rather than pinning one connection open for the whole run.
4. To hold one session open across several tool calls (avoid reconnect overhead per call), keep the adapter's `async with` block open around the entire agent invocation instead of just the discovery step.

```python
from dotenv import load_dotenv
from fastmcp import FastMCP
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

server = FastMCP("lifecycle-demo")


@server.tool
def echo(text: str) -> str:
    """Echo text back."""
    return text


async def discover_then_exit():
    # Default pattern: discover + build the agent inside the context,
    # then exit. The agent stays usable afterward because the tools
    # hold their own client reference.
    model = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")
    async with MCPAdapter(server) as adapter:
        tools = await adapter.list_tools()
        agent = create_agent(model, tools)
    return await agent.ainvoke({"messages": [{"role": "user", "content": "Echo hi"}]})


async def hold_session_open():
    # Keep the context open around the whole call to reuse one connection
    # across every tool call the run makes, instead of one per call.
    model = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")
    async with MCPAdapter(server) as adapter:
        tools = await adapter.list_tools()
        agent = create_agent(model, tools)
        return await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Echo hi twice"}]}
        )
```

## **5. Multiple Servers**

1. Two ways to give one agent tools from several servers: `MCPConfig` (one aggregate connection) or `ClientGroup` (one connection per server).
2. Choose `MCPConfig` when a single aggregate connection is enough. Choose `ClientGroup` when servers need different protocol eras, different per-server authentication, or independently configured connection pools.
3. `MCPConfig` prefixes every tool name with its config key (e.g. `weather_get_forecast`, `calc_add`), so identical tool names across servers stay distinguishable.
4. An `MCPConfig` fleet negotiates **one** protocol era for the whole fleet. Adding a single legacy-only server drops every server in that config down to the legacy era.
5. `ClientGroup` keeps each server on its own connection, so each member keeps the best protocol era, auth, and handlers its own server supports. It namespaces tools the same way (`{server}_{tool}`).

```python
from dotenv import load_dotenv
from fastmcp.client import Client
from fastmcp.client.group import ClientGroup
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_nebius import ChatNebius

load_dotenv()

CONFIG = {
    "mcpServers": {
        "weather": {"url": "https://weather.example.com/mcp"},
        "calc": {"url": "https://calc.example.com/mcp"},
    }
}


async def one_aggregate_connection():
    model = ChatNebius(model="Qwen/Qwen3-30B-A3B")
    async with MCPAdapter(CONFIG) as adapter:
        tools = await adapter.list_tools()
        agent = create_agent(model, tools)
        return await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Forecast for Oslo?"}]}
        )


async def independent_connections():
    model = ChatNebius(model="Qwen/Qwen3-30B-A3B")
    group = ClientGroup(
        {
            "weather": Client("https://weather.example.com/mcp", mode="legacy"),
            "calc": Client("https://calc.example.com/mcp", mode="auto"),
        }
    )
    async with MCPAdapter(group) as adapter:
        tools = await adapter.list_tools()
        agent = create_agent(model, tools)
        return await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Add 4 and 9."}]}
        )
```

## **6. Authentication**

1. Auth is delegated entirely to FastMCP: whatever a `fastmcp.Client` accepts for `auth`, `MCPAdapter` accepts.
2. `auth` takes one of: a bearer-token string, the literal string `"oauth"` (full OAuth 2.1 with dynamic client registration), or any `httpx.Auth` instance.
3. Bearer token: no discovery, no browser, no refresh — just a static `Authorization: Bearer <token>` header.
4. OAuth: `"oauth"` runs discovery, dynamic client registration (the client registers itself at runtime instead of a pre-provisioned client ID), the browser redirect, and the token exchange. Tokens are in-memory by default, so each run repeats the browser step unless a prebuilt `OAuth` provider with `token_storage` is supplied.
5. Per-server auth: when different servers need different credentials, give each its own `Client` inside a `ClientGroup` rather than trying to share one credential across an `MCPConfig` fleet.
6. Per-user auth (deployments): resolve the caller's identity at the server boundary, then build the MCP client with a token minted/exchanged for that specific user inside the graph factory — never one shared credential for every run.

```python
from dotenv import load_dotenv
from fastmcp.client import Client
from fastmcp.client.auth import BearerAuth
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_nebius import ChatNebius

load_dotenv()


async def bearer_token_connection(url: str, token: str):
    model = ChatNebius(model="Qwen/Qwen3-30B-A3B")
    async with MCPAdapter(Client(url, auth=BearerAuth(token))) as adapter:
        tools = await adapter.list_tools()
        return create_agent(model, tools)


async def per_user_connection(runtime, servers: dict, token_for):
    user = runtime.user.identity if runtime.user is not None else "anonymous"
    auth = BearerAuth(token_for(user))
    config = {"mcpServers": {name: {"url": url} for name, url in servers.items()}}
    model = ChatNebius(model="Qwen/Qwen3-30B-A3B")
    async with MCPAdapter(Client(config, auth=auth)) as adapter:
        tools = await adapter.list_tools()
        return create_agent(model, tools)
```

## **7. Tool Discovery and Metadata**

1. `adapter.list_tools()` returns standard LangChain `BaseTool` instances — pass them to `create_agent` exactly like any hand-written tool.
2. Each adapted tool may carry MCP provenance under `tool.metadata["mcp"]`, with three optional nested groups:
   - `tool.annotations` — MCP hints such as `read_only_hint`, `destructive_hint`.
   - `tool._meta` — opaque metadata the server chose to attach.
   - `server` — identity of the MCP server that advertised the tool (`name`, `version`).
3. Every nested field is optional. A server may supply annotations, `_meta`, server identity, some combination, or none — read defensively with `.get(...)` chains and defaults rather than direct indexing.

```python
from dotenv import load_dotenv
from fastmcp import FastMCP
from langchain.mcp import MCPAdapter
from langchain.tools import BaseTool

load_dotenv()

server = FastMCP("metadata-demo")


@server.tool(annotations={"destructiveHint": True})
def delete_record(record_id: str) -> str:
    """Delete a record by id."""
    return f"deleted {record_id}"


def is_destructive(tool: BaseTool) -> bool:
    annotations = (
        (tool.metadata or {}).get("mcp", {}).get("tool", {}).get("annotations", {})
    )
    return annotations.get("destructive_hint", False)


async def inspect_metadata():
    async with MCPAdapter(server) as adapter:
        tools = await adapter.list_tools()
        return {tool.name: is_destructive(tool) for tool in tools}
```

## **8. Tool Outputs**

1. MCP tool results arrive as LangChain content blocks on the resulting `ToolMessage`, not a raw string. Image and file content convert into standardized `image`/`file` blocks alongside `text` blocks.
2. When a tool returns MCP `structuredContent`, the adapter attaches it to `message.artifact["structured_content"]` rather than folding it into the model-visible text. The artifact type is `MCPToolArtifact`.
3. Errors: an MCP result carries an `isError` flag. A server-reported error (`isError=True`) becomes a `ToolMessage` with `status="error"` carrying the server's own message — the agent reads it and can self-correct. A transport or session failure (dropped connection) raises instead, because a model cannot act on that.

```python
from dotenv import load_dotenv
from fastmcp import FastMCP
from langchain.mcp import MCPAdapter

load_dotenv()

server = FastMCP("output-demo")


@server.tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("division by zero")
    return a / b


async def read_content_blocks():
    async with MCPAdapter(server) as adapter:
        [divide_tool] = await adapter.list_tools()

    message = await divide_tool.ainvoke(
        {"name": "divide", "args": {"a": 10, "b": 2}, "id": "1", "type": "tool_call"}
    )
    for block in message.content_blocks:
        if block["type"] == "text":
            print(f"text: {block['text']}")
        elif block["type"] == "image":
            print(f"image mime type: {block.get('mime_type')}")
    return message


async def read_server_error():
    async with MCPAdapter(server) as adapter:
        [divide_tool] = await adapter.list_tools()

    message = await divide_tool.ainvoke(
        {"name": "divide", "args": {"a": 10, "b": 0}, "id": "1", "type": "tool_call"}
    )
    return message.status
```

## **9. Human-in-the-Loop for MCP Tools**

1. The `destructive_hint` annotation (section 7) lets a gate be built from what a server *declares* about a tool, instead of hardcoding tool names.
2. Pattern: read the annotation once at load time to build a set of destructive tool names, then give `InterruptOnConfig` a `when` predicate that checks incoming `ToolCallRequest.tool_call["name"]` against that set.
3. This is the same `HumanInTheLoopMiddleware` covered in `langchain_guardrails_notes.md` — the MCP-specific part is only how the destructive set is built (from server annotations instead of a manually maintained list).
4. The predicate also sees `request.tool_call["args"]`, so a tool can be allowed to run freely for safe inputs and only pause for risky ones (e.g. `delete_file` targeting a protected path).

```python
from dotenv import load_dotenv
from fastmcp import FastMCP
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig
from langchain.mcp import MCPAdapter
from langchain.tools import BaseTool
from langchain.tools.tool_node import ToolCallRequest
from langchain_nebius import ChatNebius
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

server = FastMCP("hitl-demo")


@server.tool(annotations={"destructiveHint": True})
def delete_file(path: str) -> str:
    """Delete a file."""
    return f"deleted {path}"


def is_destructive(tool: BaseTool) -> bool:
    annotations = (
        (tool.metadata or {}).get("mcp", {}).get("tool", {}).get("annotations", {})
    )
    return annotations.get("destructive_hint", False)


async def gate_destructive_tools():
    model = ChatNebius(model="Qwen/Qwen3-30B-A3B")
    async with MCPAdapter(server) as adapter:
        tools = await adapter.list_tools()
        destructive = {tool.name for tool in tools if is_destructive(tool)}

        def needs_approval(request: ToolCallRequest) -> bool:
            return request.tool_call["name"] in destructive

        gate = InterruptOnConfig(allowed_decisions=["approve", "reject"], when=needs_approval)
        interrupt_on = {tool.name: gate for tool in tools}
        return create_agent(
            model,
            tools,
            middleware=[HumanInTheLoopMiddleware(interrupt_on=interrupt_on)],
            checkpointer=InMemorySaver(),
        )
```

## **10. Elicitation**

1. Elicitation is the MCP mechanism for a server to ask the client for input in the middle of a tool call (e.g. "which date would you like to book?").
2. `MCPAdapter` answers elicitation automatically through a LangGraph `interrupt()` — the question surfaces to whatever's already reviewing the agent's run, gets answered, and the call resumes.
3. Elicitation is **on by default**. Every client the adapter builds is armed to advertise the capability. A prebuilt client that already carries its own elicitation handler is honored instead of overridden.
4. Resuming an interrupted run requires persistence — attach a checkpointer, or there is nowhere for the paused run to wait.
5. Answers are keyed by the server's own request key: `Command(resume={"responses": {key: answer}})`. Each answer's `action` is one of `accept` (with `content` matching the request schema), `decline` (refused, call continues), or `cancel` (whole call abandoned).
6. Only elicitation is handled this way. A server asking for **sampling** (running an LLM completion on the client's behalf) or **roots** (reachable local paths) raises `NotImplementedError` — the modern, sessionless protocol has no live back-channel for those two request types.

```python
from typing import Any

from dotenv import load_dotenv
from fastmcp import FastMCP
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_nebius import ChatNebius
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()

server = FastMCP("elicitation-demo")


@server.tool
async def book_table(party_size: int, ctx) -> str:
    """Book a table, asking the client for a date if one is not given."""
    result = await ctx.elicit(
        message="What date would you like?",
        response_type=str,
    )
    return f"booked for {party_size} on {result.data}"


async def book_with_elicitation():
    model = ChatNebius(model="Qwen/Qwen3-30B-A3B")
    async with MCPAdapter(server) as adapter:
        tools = await adapter.list_tools()
        agent = create_agent(model, tools, checkpointer=InMemorySaver())
        config: Any = {"configurable": {"thread_id": "booking-1"}}

        paused = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Book a table for 4."}]}, config
        )
        [interrupt] = paused["__interrupt__"]
        [question] = interrupt.value["requests"]

        answer = {"action": "accept", "content": {"date": "2026-09-14"}}
        return await agent.ainvoke(
            Command(resume={"responses": {question["key"]: answer}}), config
        )
```

## **11. Protocol Eras and Caching**

1. MCP has two protocol eras: **legacy** (begins with an `initialize` handshake) and **modern** (protocol version `2026-07-28` and later; discovers support by probing `server/discover`).
2. FastMCP negotiates the era per connection — nothing on the LangChain side needs to know which era a given server speaks.
3. To hold tools from servers on different eras in one agent, give each server its own connection (a `ClientGroup`, or one adapter per server) rather than putting them in one `MCPConfig` fleet, which negotiates a single shared era for everyone in it.
4. A prebuilt `fastmcp.Client` pins the era with `mode`: `"legacy"` forces the handshake era, `"auto"` (default) negotiates the newest era the server understands.
5. Discovery caching is opt-in and only takes effect against modern-era servers that advertise cache hints. `list_tools()` accepts `cache_mode`:

| `cache_mode` | Behavior |
|---|---|
| `use` (default) | Serve a cached list if present and within the server's TTL hint, otherwise fetch and store. |
| `refresh` | Fetch fresh and repopulate the cache. |
| `bypass` | Skip the cache entirely. |

6. The cache itself, and per-principal isolation of it, is configured on the client via `Client(cache=...)` — `cache_mode` only selects how a given `list_tools()` call reads that configured cache.

```python
from dotenv import load_dotenv
from fastmcp.client import Client
from langchain.mcp import MCPAdapter

load_dotenv()


async def eras_side_by_side(legacy_url: str, modern_url: str):
    legacy = Client(legacy_url, mode="legacy")
    modern = Client(modern_url, mode="auto")
    async with (
        MCPAdapter(legacy) as legacy_adapter,
        MCPAdapter(modern) as modern_adapter,
    ):
        return await legacy_adapter.list_tools() + await modern_adapter.list_tools()


async def cached_discovery(url: str):
    async with MCPAdapter(url) as adapter:
        served_from_cache = await adapter.list_tools(cache_mode="use")
        forced_fresh = await adapter.list_tools(cache_mode="refresh")
        return served_from_cache, forced_fresh
```

## **12. Scaling a Deployment**

1. A deployment serving many concurrent runs should discover tools per run (so each run sees the current catalog) but reuse connections underneath, rather than reconnecting from scratch on every request.
2. Build the agent inside a `langgraph dev` graph factory so it is called once per run; pair `cache_mode="use"` with a shared HTTP connection pool to absorb the repeated-discovery cost.
3. By default each FastMCP client manages its own HTTP connections. Across a fleet of servers or many concurrent runs, that means many independent pools. Sharing one pool means passing an `httpx_client_factory` that draws from a single transport and never lets any one client close it.
4. In a `langgraph dev` graph factory, annotated parameter and return types must be importable at runtime, not only under `TYPE_CHECKING` — `langgraph-api` inspects the factory with `typing.get_type_hints()`, and an unresolvable annotation causes it to inject a config dict instead of the actual runtime object.

```python
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_nebius import ChatNebius

load_dotenv()

SERVERS = {
    "weather": "http://localhost:8001/mcp",
    "calc": "http://localhost:8002/mcp",
}


async def make_graph():
    """Called once per run by `langgraph dev`."""
    config = {"mcpServers": {name: {"url": url} for name, url in SERVERS.items()}}
    model = ChatNebius(model="Qwen/Qwen3-30B-A3B")
    async with MCPAdapter(config) as adapter:
        tools = await adapter.list_tools(cache_mode="use")
        return create_agent(model, tools)
```

```python
import httpx
from dotenv import load_dotenv
from fastmcp.client import Client
from fastmcp.client.group import ClientGroup
from fastmcp.client.transports import StreamableHttpTransport
from langchain.mcp import MCPAdapter

load_dotenv()

_POOL = httpx.AsyncHTTPTransport()


class SharedPool(httpx.AsyncBaseTransport):
    handle_async_request = _POOL.handle_async_request

    async def aclose(self) -> None:
        pass


def client_factory(**kwargs) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=SharedPool(), **kwargs)


async def load_over_shared_pool(servers: dict):
    group = ClientGroup(
        {
            name: Client(StreamableHttpTransport(url, httpx_client_factory=client_factory))
            for name, url in servers.items()
        }
    )
    async with MCPAdapter(group) as adapter:
        return await adapter.list_tools()
```

## **13. Decision Guide**

1. Connecting to exactly one server → pass the target directly (URL, `Path`, or in-process server). Don't reach for `MCPConfig`/`ClientGroup` until there is a second server.
2. Several servers that can share one protocol era and one auth scheme → `MCPConfig`. Simpler, one connection, automatic tool-name prefixing.
3. Several servers on different protocol eras, with different auth, or needing independent connection pools → `ClientGroup`. More setup, but each server keeps its own negotiated capabilities.
4. Local development / testing a server you're also writing → in-process `FastMCP` target. No subprocess, no socket, deterministic.
5. A script-based local tool server → `Path` target (stdio subprocess).
6. Remote production server → `http`/`https` URL, or a prebuilt `Client` if auth or transport needs configuring.
7. Deployment serving many concurrent users → per-run discovery inside a graph factory, `cache_mode="use"`, shared HTTP pool, per-user auth via `BearerAuth(token_for(user))`.
8. A destructive-sounding MCP tool (delete, drop, send, pay) → gate it with `HumanInTheLoopMiddleware`, driven off the server's own `destructive_hint` annotation rather than a hardcoded tool-name list.
9. A tool that needs mid-call input from the user → rely on default elicitation-as-interrupt; just make sure a checkpointer is attached so the run has somewhere to pause.
10. A server that wants sampling or roots from the client → not supported by this adapter; it will raise `NotImplementedError`. Needs a different integration path.

## **14. Things to Verify Before Relying on This**

1. `langchain.mcp` is beta and, as of the most recent check, shipping in the `1.4.0` alpha line (`langchain==1.4.0a3` requires `pip install --pre`). Confirm the installed version is stable (`1.4.0` final or later) before depending on the API shape in production code — beta means the API may still change.
2. The namespace requires `fastmcp>=4.0.0`. Verify the installed `fastmcp` version meets this before debugging otherwise-unexplained `MCPConfig`/`ClientGroup` failures.
3. This entirely supersedes the standalone `langchain-mcp-adapters` package (`MultiServerMCPClient` → `MCPAdapter`). If any existing code still imports the old package, it is on the deprecated path — check the official migration guide before assuming any 1:1 method mapping.
4. `langchain-nebius` remains a small, externally maintained partner package (PyPI, still `v0.1.x` at last check) — it is not yet colocated in the LangChain monorepo. `ChatNebius` extends `BaseChatOpenAI` and is documented as supporting tool calling and structured output, but this has not been verified end-to-end against `MCPAdapter`-supplied tools specifically — test tool-calling through an MCP-backed agent before relying on it, rather than assuming parity with the OpenAI/Anthropic integrations.
5. `langchain-nvidia-ai-endpoints` (`ChatNVIDIA`) tool-calling support varies by underlying hosted model — verify the specific model string supports function calling before wiring it into an MCP agent; not every model NVIDIA hosts through the endpoint does.
6. The elicitation code example in section 10 uses `ctx.elicit(...)` inside a `@server.tool` — this is FastMCP server-side API, not `langchain.mcp`. Confirm the exact signature against the installed FastMCP version; server-side elicitation helpers have moved before.
7. Deprecated transports: HTTP+SSE (protocol version `2024-11-05`) is deprecated in favor of Streamable HTTP. FastMCP still ships `SSETransport` for back-compatibility (`MCPAdapter(Client(SSETransport(url)))`), but don't build new servers against it. WebSocket has no FastMCP transport at all.
8. Carried forward from `langchain_middleware_notes.md`: GitHub issue `langchain-ai/langchain#36568` — `wrap_model_call` response-format narrowing via `ToolStrategy` had no effect in at least `langchain 1.2.15`/`langchain-core 1.2.25`. Relevant here if combining MCP tools with structured-output middleware in the same agent — check whether that issue is still open before assuming narrowing works.
9. Protocol era negotiation and `server/discover` probing (section 11) is new-spec behavior (`2026-07-28` protocol version). Servers still running older SDK versions may not support it — verify the target server's SDK version if era-related connection errors appear.
10. The shared-connection-pool pattern in section 12 subclasses `httpx.AsyncBaseTransport` directly. This is an internal `httpx` extension point, not a documented `langchain.mcp` or FastMCP API — re-verify it against the installed `httpx` version, since transport internals are more likely to shift across `httpx` releases than the public client API.