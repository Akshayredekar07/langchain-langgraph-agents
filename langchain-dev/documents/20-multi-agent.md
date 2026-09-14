## **LangChain Multi-Agent Patterns**

### **1. Why multi-agent at all**

1. A single agent with the right tools and a dynamic prompt often beats a multi-agent setup — multi-agent adds coordination overhead, not free capability.
2. Multi-agent earns its cost when you need one or more of:
   - **Context management** — specialized knowledge without overwhelming one context window.
   - **Distributed development** — separate teams own separate agents with clear boundaries.
   - **Parallelization** — spawn workers for subtasks and run them concurrently.
3. Concrete triggers: a single agent has too many tools and picks the wrong one, tasks need long domain-specific prompts + tools, or you need to gate capabilities behind sequential preconditions.
4. Everything here sits under **context engineering** — the real design question is always "what does each agent see."
5. There is also a higher-level harness, **Deep Agents**, built on top of `create_agent` that ships subagents, skills, planning, a virtual filesystem, and context management out of the box — cross-check against `[[langchain_middleware_notes.md]]` if using `FilesystemMiddleware` / `SubAgentMiddleware`, since those version independently of core `langchain`.

### **2. The four patterns, at a glance**

| Pattern | Core mechanism | Direct user interaction | Multi-hop |
|---|---|---|---|
| Subagents | Main agent calls subagents as tools | No (subagent talks to main agent only) | Yes |
| Handoffs | Tool call updates state → behavior/agent changes | Yes | Yes |
| Skills | Agent loads specialized prompts on demand via a tool | Yes | Yes |
| Router | A classification step dispatches to agent(s) | Limited (router is usually stateless) | No |

1. **Supervisor (subagents) vs Router** — easy to confuse. A supervisor is a full agent that maintains conversation state and decides across multiple turns which subagent to call. A router is typically one classification step (LLM call or rules) that dispatches without maintaining ongoing conversation state.
2. Patterns compose: a subagents architecture can invoke tools that run a router or a custom workflow; subagents can load skills internally.

### **3. Subagents pattern**

1. Central main agent (**supervisor**) calls subagents wrapped as tools.
2. Subagents are **stateless by default** — no memory of past interactions, all conversation memory lives in the main agent. This gives context isolation: each subagent call gets a clean context window.
3. Key characteristics: centralized control (all routing passes through the main agent), no direct user interaction (subagents return to the main agent, not the user — though interrupts inside a subagent can pause for user input), parallel execution possible in a single turn.
4. Use it when: multiple distinct domains (calendar, email, CRM, DB), subagents don't need to talk to the user directly, or you want centralized workflow control. For a handful of tools, just use one agent.

#### **3.1 Tool-per-agent implementation**

```python
import os
from dotenv import load_dotenv
from langchain_nebius import ChatNebius
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

model = ChatNebius(
    model="meta-llama/Llama-3.3-70B-Instruct-fast",
    api_key=os.environ["NEBIUS_API_KEY"],
)

@tool
def get_stock_price(ticker: str) -> str:
    """Look up the latest price for a stock ticker."""
    prices = {"AAPL": "227.10", "MSFT": "412.30", "NVDA": "138.55"}
    return prices.get(ticker.upper(), "unknown ticker")

research_agent = create_agent(
    model=model,
    tools=[get_stock_price],
    system_prompt="You research stock prices and summarize findings in two sentences.",
)

@tool("research", description="Research a stock ticker and return findings")
def call_research_agent(query: str) -> str:
    result = research_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content

main_agent = create_agent(
    model=model,
    tools=[call_research_agent],
    system_prompt="You are a portfolio assistant. Delegate research to the research tool.",
)

if __name__ == "__main__":
    response = main_agent.invoke(
        {"messages": [{"role": "user", "content": "How is NVDA doing?"}]}
    )
    print(response["messages"][-1].content)
```

#### **3.2 Single dispatch tool with an agent registry**

1. Alternative to a tool per agent: one parameterized `task` tool that invokes any registered subagent by name.
2. Trade-off: simpler composition and strong context isolation, less per-agent customization than tool-per-agent.
3. Good for distributed teams, large or growing agent counts, or when subagents share the exact capabilities of the main agent (isolation becomes the point, not specialization).
4. Subagents can be discovered via system-prompt enumeration (< 10 agents, static), an enum constraint on the tool schema (type-safe, still static), or tool-based discovery (`list_agents`, for large/dynamic registries).

```python
import os
from dotenv import load_dotenv
from langchain_nebius import ChatNebius
from langchain.agents import create_agent
from langchain.tools import tool
from enum import Enum

load_dotenv()

model = ChatNebius(model="meta-llama/Llama-3.3-70B-Instruct-fast", api_key=os.environ["NEBIUS_API_KEY"])

research_agent = create_agent(model=model, system_prompt="You are a research specialist. Be concise.")
writer_agent = create_agent(model=model, system_prompt="You are a writing specialist. Be concise.")

SUBAGENTS = {"research": research_agent, "writer": writer_agent}

class AgentName(str, Enum):
    RESEARCH = "research"
    WRITER = "writer"

@tool
def task(agent_name: AgentName, description: str) -> str:
    """Launch an ephemeral subagent for a task.

    Available agents:
    - research: research and fact-finding
    - writer: content creation and editing
    """
    agent = SUBAGENTS[agent_name.value]
    result = agent.invoke({"messages": [{"role": "user", "content": description}]})
    return result["messages"][-1].content

main_agent = create_agent(
    model=model,
    tools=[task],
    system_prompt=(
        "You coordinate specialized sub-agents. Available: research, writer. "
        "Use the task tool to delegate work."
    ),
)

if __name__ == "__main__":
    response = main_agent.invoke(
        {"messages": [{"role": "user", "content": "Research the benefits of vector databases, then draft a two-line summary."}]}
    )
    print(response["messages"][-1].content)
```

#### **3.3 Sync vs async execution**

1. **Sync (default)** — main agent waits for the subagent to finish before continuing. Use when the main agent's next step depends on the result, tasks have order dependencies, or a subagent failure should block the response. Simple, but freezes the conversation on long-running subagents.
2. **Async** — main agent kicks off a background job (not Python `async`/`await` — a separately tracked job) and stays responsive. Needs a three-tool pattern: start job (returns a job ID), check status, get result. Notify the user on completion (e.g. surface a click that sends a `HumanMessage` like "check job_123").
3. Use async when subagent work is independent of the ongoing conversation and the user shouldn't have to wait.

#### **3.4 Context engineering for subagents**

1. **Subagent specs** (name + description) are the main agent's only signal for *when* to call a subagent — treat them as prompting levers, not documentation.
2. **Subagent inputs** — control what the subagent receives via custom state and `ToolRuntime`. Two named modes (from Deep Agents): **isolated** (default, only the task description — focused, but re-derives context) and **forked** (full parent message history — seeded, but larger prompt, less isolation).
3. **Subagent outputs** — either prompt the subagent to return exactly what's needed (remind it the supervisor only sees the final message), or format the result in code using a `Command` to pass extra state keys back alongside the tool message.
4. **Checkpointing** — subagents default to inherited-checkpointer mode: fresh state per call, safe in parallel. Compile with `checkpointer=True` for a subagent that must persist its own history across invocations. Because subagents run inside tool functions, `get_state(subgraphs=True)` on the main graph will not surface subagent state.

```python
import os
from typing import Annotated
from dotenv import load_dotenv
from langchain_nebius import ChatNebius
from langchain.agents import AgentState, create_agent
from langchain.tools import tool, ToolRuntime, InjectedToolCallId
from langchain.messages import ToolMessage
from langgraph.types import Command

load_dotenv()

model = ChatNebius(model="meta-llama/Llama-3.3-70B-Instruct-fast", api_key=os.environ["NEBIUS_API_KEY"])

class TicketState(AgentState):
    priority: str

triage_agent = create_agent(model=model, system_prompt="Classify support tickets by urgency in one line.")

@tool("triage", description="Send a ticket description to the triage subagent")
def call_triage(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    runtime: ToolRuntime[None, TicketState],
) -> Command:
    result = triage_agent.invoke({"messages": [{"role": "user", "content": query}]})
    verdict = result["messages"][-1].content
    return Command(
        update={
            "priority": verdict,
            "messages": [ToolMessage(content=verdict, tool_call_id=tool_call_id)],
        }
    )

main_agent = create_agent(
    model=model,
    tools=[call_triage],
    state_schema=TicketState,
    system_prompt="You route support tickets. Use the triage tool first, then respond to the user.",
)

if __name__ == "__main__":
    response = main_agent.invoke(
        {"messages": [{"role": "user", "content": "Production database is down for all customers."}]}
    )
    print(response["messages"][-1].content)
```

### **4. Handoffs pattern**

1. Behavior changes dynamically based on **state**. A tool updates a state variable (e.g. `current_step`, `active_agent`) that persists across turns; the system reads it to change configuration (prompt, tools) or route to a different agent.
2. Term coined by OpenAI for tool calls like `transfer_to_sales_agent` that transfer control.
3. Key characteristics: state-driven behavior, tool-based transitions, **direct user interaction at every state** (unlike subagents), persistent state across turns.
4. Use it when you need sequential constraints (unlock capabilities only after preconditions), the agent must converse directly with the user across states, or you're building a multi-stage conversational flow (classic case: customer support collecting warranty info before offering a resolution).
5. Two implementations: **single agent with middleware** (simpler, recommended default) or **multiple agent subgraphs** (only when you need genuinely different agent implementations, e.g. one node that is itself a complex graph).
6. Always pair the `AIMessage` that triggered a transition with a matching `ToolMessage` — LLMs expect every tool call to have a response; skip this and the conversation history is malformed.

#### **4.1 Single agent with middleware**

```python
import os
from typing import Callable
from dotenv import load_dotenv
from langchain_nebius import ChatNebius
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()

model = ChatNebius(model="meta-llama/Llama-3.3-70B-Instruct-fast", api_key=os.environ["NEBIUS_API_KEY"])

class SupportState(AgentState):
    current_step: str = "triage"
    warranty_status: str | None = None

@tool
def record_warranty_status(status: str, runtime: ToolRuntime[None, SupportState]) -> Command:
    """Record warranty status and move to the specialist step."""
    return Command(
        update={
            "messages": [ToolMessage(content=f"warranty status: {status}", tool_call_id=runtime.tool_call_id)],
            "warranty_status": status,
            "current_step": "specialist",
        }
    )

@tool
def provide_solution(runtime: ToolRuntime[None, SupportState]) -> str:
    """Provide a resolution based on the recorded warranty status."""
    return f"Resolution based on warranty status: {runtime.state.get('warranty_status')}"

@wrap_model_call
def apply_step_config(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    step = request.state.get("current_step", "triage")
    configs = {
        "triage": {"prompt": "Ask whether the device is under warranty, then call record_warranty_status.", "tools": [record_warranty_status]},
        "specialist": {"prompt": "Call provide_solution and relay the result to the user.", "tools": [provide_solution]},
    }
    config = configs[step]
    request = request.override(system_prompt=config["prompt"], tools=config["tools"])
    return handler(request)

agent = create_agent(
    model=model,
    tools=[record_warranty_status, provide_solution],
    state_schema=SupportState,
    middleware=[apply_step_config],
    checkpointer=InMemorySaver(),
)

if __name__ == "__main__":
    thread = {"configurable": {"thread_id": "ticket-1"}}
    first = agent.invoke({"messages": [{"role": "user", "content": "My phone screen is cracked"}]}, thread)
    print(first["messages"][-1].content)
    second = agent.invoke({"messages": [{"role": "user", "content": "Yes, still under warranty"}]}, thread)
    print(second["messages"][-1].content)
```

#### **4.2 Multiple agent subgraphs**

1. Distinct agents live as separate graph nodes; handoff tools navigate between them with `Command(goto=..., graph=Command.PARENT)`.
2. Requires deliberate context engineering — decide exactly which messages cross the boundary. Passing only the handoff pair (triggering `AIMessage` + acknowledgement `ToolMessage`) avoids confusing the receiving agent with irrelevant internal reasoning and keeps token cost down; if more context is needed, summarize it into the `ToolMessage` content instead of forwarding raw history.
3. When ending a turn back to the user, make sure the final message is an `AIMessage` — that's what tells the UI the agent is done.

```python
import os
from typing import Literal
from dotenv import load_dotenv
from typing_extensions import NotRequired
from langchain_nebius import ChatNebius
from langchain.agents import AgentState, create_agent
from langchain.messages import AIMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

load_dotenv()

model = ChatNebius(model="meta-llama/Llama-3.3-70B-Instruct-fast", api_key=os.environ["NEBIUS_API_KEY"])

class RoutedState(AgentState):
    active_agent: NotRequired[str]

def handoff_message(runtime: ToolRuntime, target: str) -> tuple[AIMessage, ToolMessage]:
    last_ai = next(m for m in reversed(runtime.state["messages"]) if isinstance(m, AIMessage))
    ack = ToolMessage(content=f"transferred to {target}", tool_call_id=runtime.tool_call_id)
    return last_ai, ack

@tool
def transfer_to_support(runtime: ToolRuntime) -> Command:
    """Transfer the conversation to the support agent."""
    last_ai, ack = handoff_message(runtime, "support_agent")
    return Command(goto="support_agent", update={"active_agent": "support_agent", "messages": [last_ai, ack]}, graph=Command.PARENT)

@tool
def transfer_to_sales(runtime: ToolRuntime) -> Command:
    """Transfer the conversation to the sales agent."""
    last_ai, ack = handoff_message(runtime, "sales_agent")
    return Command(goto="sales_agent", update={"active_agent": "sales_agent", "messages": [last_ai, ack]}, graph=Command.PARENT)

sales_agent = create_agent(model=model, tools=[transfer_to_support], system_prompt="You handle sales. Transfer technical issues to support.")
support_agent = create_agent(model=model, tools=[transfer_to_sales], system_prompt="You handle support. Transfer pricing questions to sales.")

def call_sales(state: RoutedState) -> Command:
    return sales_agent.invoke(state)

def call_support(state: RoutedState) -> Command:
    return support_agent.invoke(state)

def route_after(state: RoutedState) -> Literal["sales_agent", "support_agent", "__end__"]:
    messages = state.get("messages", [])
    if messages and isinstance(messages[-1], AIMessage) and not messages[-1].tool_calls:
        return "__end__"
    return state.get("active_agent") or "sales_agent"

def route_initial(state: RoutedState) -> Literal["sales_agent", "support_agent"]:
    return state.get("active_agent") or "sales_agent"

builder = StateGraph(RoutedState)
builder.add_node("sales_agent", call_sales)
builder.add_node("support_agent", call_support)
builder.add_conditional_edges(START, route_initial, ["sales_agent", "support_agent"])
builder.add_conditional_edges("sales_agent", route_after, ["sales_agent", "support_agent", END])
builder.add_conditional_edges("support_agent", route_after, ["sales_agent", "support_agent", END])
graph = builder.compile()

if __name__ == "__main__":
    result = graph.invoke({"messages": [{"role": "user", "content": "My login is broken"}]})
    for message in result["messages"]:
        message.pretty_print()
```

### **5. Skills pattern**

1. Specialized capabilities packaged as invocable "skills" — primarily **prompt-driven** specializations a single agent loads on demand while staying in control.
2. Conceptually identical to Anthropic's Agent Skills / `llms.txt` (Jeremy Howard) — progressive disclosure of documentation, applied here to prompts and domain knowledge instead of docs.
3. Key characteristics: prompt-driven specialization, progressive disclosure (skills surface only when relevant), team distribution, lighter weight than a full subagent, and skills can reference scripts/templates/other resources without loading them upfront.
4. Use it when one agent needs many possible specializations, you don't need hard constraints between them (unlike handoffs), or teams need to build capabilities independently. Typical fits: coding assistants (skills per language), knowledge bases (skills per domain), creative tools (skills per format).
5. Extensions: **dynamic tool registration** (loading a skill also registers new tools, e.g. a `database_admin` skill adds backup/restore/migrate tools), **hierarchical skills** (a skill can expose sub-skills, loaded independently), **reference awareness** (a skill's prompt points to other files the agent reads only when needed).

```python
import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

model = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", api_key=os.environ["NVIDIA_API_KEY"])

SKILLS = {
    "write_sql": "You are a SQL expert. Write PostgreSQL. Always parameterize queries.",
    "review_legal_doc": "You are a contract reviewer. Flag ambiguous liability and indemnity clauses.",
}

@tool
def load_skill(skill_name: str) -> str:
    """Load a specialized skill prompt.

    Available skills:
    - write_sql: SQL query writing expert
    - review_legal_doc: legal document reviewer
    """
    return SKILLS.get(skill_name, f"unknown skill: {skill_name}")

agent = create_agent(
    model=model,
    tools=[load_skill],
    system_prompt=(
        "You are a helpful assistant with access to two skills: "
        "write_sql and review_legal_doc. Call load_skill before doing "
        "specialized work, then follow the returned instructions."
    ),
)

if __name__ == "__main__":
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "Write a query to fetch the 5 most recent orders for a customer_id."}]}
    )
    print(response["messages"][-1].content)
```

### **6. Router pattern**

1. A **routing step** classifies input and directs it to specialized agent(s); results are synthesized into one response.
2. Key characteristics: router decomposes the query, zero or more specialized agents run (in parallel where possible), results get synthesized back into a coherent answer.
3. Use it when you have distinct verticals (separate knowledge domains, each needing its own agent), want to query multiple sources in parallel, or want a synthesized combined answer.
4. **Router vs subagents** — a router is a dedicated, usually stateless classification step (single LLM call or rules) that dispatches; it typically does not hold conversation history or do multi-turn orchestration, it's a preprocessing step. A supervisor (subagents) is an ongoing agent that decides across turns. Pick a router for clear categories and lightweight/deterministic classification; pick a supervisor for conversation-aware orchestration.
5. Implementation primitives: `Command(goto=agent_name)` to route to a single agent, `Send(agent_name, payload)` to fan out to several agents in parallel.
6. Routers are stateless by default — for multi-turn conversations, wrap the stateless router as a tool that a conversational agent calls, so state lives in the calling agent instead.

```python
import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_nebius import ChatNebius
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

load_dotenv()

model = ChatNebius(model="meta-llama/Llama-3.3-70B-Instruct-fast", api_key=os.environ["NEBIUS_API_KEY"])

class RouterState(TypedDict):
    query: str
    results: list[str]

class Classification(TypedDict):
    query: str
    agent: str

github_agent = create_agent(model=model, system_prompt="You answer questions about code, issues, and pull requests. Be concise.")
docs_agent = create_agent(model=model, system_prompt="You answer questions about internal documentation. Be concise.")

def classify(query: str) -> list[Classification]:
    lowered = query.lower()
    hits = []
    if "pr" in lowered or "code" in lowered or "issue" in lowered:
        hits.append({"query": query, "agent": "github"})
    if "doc" in lowered or "wiki" in lowered or "guide" in lowered:
        hits.append({"query": query, "agent": "docs"})
    return hits or [{"query": query, "agent": "docs"}]

def route(state: RouterState):
    return [Send(c["agent"], {"query": c["query"], "results": []}) for c in classify(state["query"])]

def run_github(state: RouterState) -> RouterState:
    result = github_agent.invoke({"messages": [{"role": "user", "content": state["query"]}]})
    return {"results": [result["messages"][-1].content]}

def run_docs(state: RouterState) -> RouterState:
    result = docs_agent.invoke({"messages": [{"role": "user", "content": state["query"]}]})
    return {"results": [result["messages"][-1].content]}

def synthesize(state: RouterState) -> RouterState:
    combined = "\n".join(state["results"])
    return {"results": [combined]}

builder = StateGraph(RouterState)
builder.add_node("github", run_github)
builder.add_node("docs", run_docs)
builder.add_node("synthesize", synthesize)
builder.add_conditional_edges(START, route, ["github", "docs"])
builder.add_edge("github", "synthesize")
builder.add_edge("docs", "synthesize")
builder.add_edge("synthesize", END)
graph = builder.compile()

if __name__ == "__main__":
    result = graph.invoke({"query": "What does the onboarding guide say about PR reviews?", "results": []})
    print(result["results"][-1])
```

### **7. Custom workflow pattern**

1. Build bespoke execution flow directly in **LangGraph**, mixing deterministic logic with agentic steps.
2. Any of the other four patterns can be embedded as nodes in a custom workflow — this is the escape hatch when none of the named patterns fits cleanly.
3. No dedicated code example here since it's just LangGraph `StateGraph` composition — see sections 4.2 and 6 above, both of which are already custom workflows built from `StateGraph`.

### **8. Performance trade-offs across patterns**

1. **One-shot request** ("buy coffee") — Handoffs, Skills, and Router are cheapest at 3 model calls each; Subagents costs 4 because results flow back through the main agent (the cost of centralized control).
2. **Repeat request in the same conversation** — stateful patterns (Handoffs, Skills) save 40-50% of calls on the second turn because the active configuration/loaded skill is already in context. Subagents repeats the full 4-call flow every time (stateless by design — strong isolation, no discount). Router repeats its classification call every time unless wrapped as a tool in a stateful agent.
3. **Multi-domain request** ("compare Python, JS, Rust") — Subagents and Router win on token efficiency (~9K tokens) because each specialist works in an isolated context. Skills is cheapest on call count (3) but accumulates all loaded skill content into one growing context (~15K tokens). Handoffs is worst here — it's inherently sequential and can't parallelize across domains (7+ calls, 14K+ tokens).
4. Net read: pick Subagents or Router for large-context, parallelizable, multi-domain work; pick Handoffs or Skills for simple, single-focus, or repeat-heavy interactions.

### **Decision guide**

1. Need centralized orchestration across multiple distinct tool domains, subagents don't talk to the user directly → **Subagents**.
2. Need to enforce a strict sequence (collect X before allowing Y) or the agent must keep talking to the user across stages → **Handoffs** (middleware version unless you truly need separate agent implementations).
3. One agent, many optional specializations, no hard ordering between them, want to keep the prompt small until a specialization is actually needed → **Skills**.
4. Clear, classifiable input categories, want parallel fan-out to several domain agents and a synthesized answer, don't need ongoing conversation state in the router itself → **Router**.
5. None of the above fits as-is → drop to a **custom LangGraph workflow** and embed whichever patterns you need as nodes.
6. Repeat-heavy, low-latency conversations favor Handoffs/Skills; large parallel multi-domain workloads favor Subagents/Router — cross-check section 8 against your actual call/token budget before committing.

### **Things to verify before relying on this**

1. `langchain-nebius` is still a small externally-maintained partner package (PyPI v0.1.3 as of this check) — there is an open GitHub issue (`langchain-ai/langchain#37169`) proposing to port it into the core monorepo under `libs/partners/`, not yet merged. Re-check model names, tool-calling, and structured-output parity before depending on it in a new pattern.
2. `langgraph-swarm` (`create_handoff_tool`, `create_swarm`) is explicitly marked by LangChain as superseded — the docs now recommend the manual tool-calling handoff pattern shown in section 4 for most cases. Don't reach for `langgraph-swarm` unless the manual pattern genuinely can't cover the need.
3. The subagents "single dispatch tool" and handoffs "subgraph" examples both use `Command` and `Send` from `langgraph.types` directly — confirm these primitives haven't moved as `langgraph` versions independently of `langchain`.
4. `wrap_model_call` + `request.override(...)` (section 4.1) is the same middleware mechanism flagged in `[[langchain_context_engineering_notes.md]]` and affected by GitHub issue `langchain-ai/langchain#36568` (response_format narrowing via `ToolStrategy` has no effect in at least 1.2.15/1.2.25) — that issue is about `response_format`, not `system_prompt`/`tools` overrides used here, but re-check if you extend this pattern to structured output.
5. Router section's `classify()` function is a placeholder rule-based stub for the example — real routers should use structured output (`with_structured_output` / `ToolStrategy`) for classification; verify current `ChatNebius`/`ChatNVIDIA` structured-output support before swapping it in.
6. Deep Agents' "isolated" vs "forked" subagent input naming (section 3.4) is a Deep Agents concept layered on top of the base subagents pattern — the base `create_agent` API doesn't have a `mode` parameter itself; forking there is something you implement manually via `ToolRuntime` state, as shown.