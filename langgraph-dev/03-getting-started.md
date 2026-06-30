# **Getting Started With LangGraph**

## What Is LangGraph, In Plain Words

Think of a normal LLM app like a single road with no turns. The user asks something, it goes to the model, the model replies, done. That works fine until your agent needs to think in loops: call a tool, check the result, maybe call another tool, maybe ask a human, then finally answer. A straight road cannot do that. You need an actual map with branches and loops.

That is what LangGraph is. It is a low-level orchestration framework and runtime built by the LangChain team for building, managing, and deploying long-running, stateful agents. It is used in production by companies like Klarna, Uber, and J.P. Morgan.

LangGraph models your agent as a directed graph:

- **Nodes** are steps that do work, usually plain Python functions (call the model, call a tool, parse a result, ask a human).
- **Edges** are the connections that decide what happens next, either fixed or conditional.
- **State** is a shared object that flows through every node, getting read and updated as it goes.

This shape naturally supports loops, branches, retries, and memory, things a simple chain cannot express cleanly.

LangGraph is intentionally low-level. It does not hand you a prebuilt agent architecture or hide prompt design from you. If you want a higher-level, batteries-included agent loop, LangChain's own agent abstractions sit on top of LangGraph and are a better starting point for very simple use cases. LangGraph is for when you need real control.

### How It Relates To LangChain And LangSmith

People often mix these up, so here is the short version:

- **LangChain** is the toolkit layer: model wrappers, prompt templates, tool definitions, output parsers.
- **LangGraph** is the orchestration layer: it decides how those building blocks connect, loop, branch, and persist state over time.
- **LangSmith** is the observability and deployment platform: tracing, evaluation, and hosting for whatever you build with LangGraph or LangChain.

You can use LangGraph without LangChain, but in most real projects you will use LangChain components (model wrappers, tools) inside LangGraph nodes.

---

## Installation

You need Python and pip (or uv) installed first.

**Using pip:**

```bash
pip install -U langgraph
```

**Using uv:**

```bash
uv add langgraph
```

If you plan to follow the Anthropic-based examples in this guide, you also need an Anthropic account and API key. Set it as an environment variable in your terminal:

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### Optional But Recommended: LangSmith Tracing

Once your graphs get more than a couple of nodes, debugging blind is painful. LangSmith lets you trace every step, see exactly what each node received and returned, and catch where things went wrong.

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="your-key-here"
```

---

## Your First Graph: Hello World

This is the smallest possible LangGraph program. It has one node that fakes a model response, wired between a start point and an end point.

```python
from langgraph.graph import StateGraph, MessagesState, START, END

def mock_llm(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "hello world"}]}

graph = StateGraph(MessagesState)
graph.add_node(mock_llm)
graph.add_edge(START, "mock_llm")
graph.add_edge("mock_llm", END)
graph = graph.compile()

graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})
```

Walk through what is happening here, line by line in human terms:

1. `StateGraph(MessagesState)` creates a blank graph and tells it that the shared state will follow the `MessagesState` shape, basically a list of chat messages.
2. `add_node(mock_llm)` registers a function as a step in the graph. By default the node's name is taken from the function name.
3. `add_edge(START, "mock_llm")` says "when the graph starts, go straight to this node."
4. `add_edge("mock_llm", END)` says "after this node finishes, stop."
5. `.compile()` turns your blueprint into something you can actually run. Before compiling, the graph is just a definition, it cannot do anything.
6. `.invoke(...)` runs it once with an initial state and gives you back the final state.

Every graph needs at least one edge from `START` and at least one path that eventually reaches `END`. Skip either one and LangGraph will throw an error when you try to compile.

---

## Core Building Blocks, Explained Simply

### State

State is just a shared dictionary-like object that every node can read from and write to. You define its shape using `TypedDict`.

```python
from typing_extensions import TypedDict

class MyState(TypedDict):
    question: str
    answer: str
```

Each node receives the current state and returns a dictionary of the fields it wants to update. Here is the part beginners trip over: by default, when a node returns a value for a key, it **overwrites** that key. If you want a field to accumulate instead (like a running list of chat messages), you need to tell LangGraph that explicitly using a reducer.

```python
from typing_extensions import TypedDict, Annotated
import operator

class MyState(TypedDict):
    messages: Annotated[list, operator.add]  # new messages get appended, not overwritten
    step_count: int  # this one just gets replaced each time
```

`operator.add` is the most common reducer: instead of replacing the list, it concatenates the new value onto the old one.

### Nodes

A node is just a Python function. It takes the current state as input and returns a dictionary with the fields it wants to change.

```python
def greet(state: MyState):
    return {"answer": f"Hello, {state['question']}"}
```

You register it with:

```python
graph.add_node("greet", greet)
```

If you don't pass a name, LangGraph uses the function's name automatically (as seen in the hello world example above).

### Edges

Edges connect nodes and define the flow.

**Direct edge** (always go from A to B):

```python
graph.add_edge("node_a", "node_b")
```

**Conditional edge** (decide where to go based on the current state):

```python
def route(state: MyState):
    if state["step_count"] > 5:
        return "stop_node"
    return "continue_node"

graph.add_conditional_edges("node_a", route, ["stop_node", "continue_node"])
```

This is the actual decision-making mechanism in an agent. A conditional edge function looks at the state and returns the name of whichever node should run next. This is exactly how a tool-calling agent decides "should I call another tool, or am I done?"

**Entry and finish shortcuts:**

```python
graph.set_entry_point("first_node")   # same as add_edge(START, "first_node")
graph.set_finish_point("last_node")   # same as add_edge("last_node", END)
```

### Compiling

`StateGraph` is only a builder. It cannot run anything on its own. Calling `.compile()` turns it into a `CompiledStateGraph`, which behaves like a standard LangChain Runnable, meaning it supports:

- `.invoke()` — run once, get the final result
- `.stream()` — get intermediate steps as they happen
- `.ainvoke()` / `.astream()` — async versions
- `.batch()` — run multiple inputs at once

---

## A Complete Worked Example: Calculator Agent

This builds a small agent that can add, multiply, and divide numbers using tool calls, looping back to the model until it decides it is done. This mirrors the official LangGraph quickstart.

### Step 1: Define The Model And Tools

```python
from langchain.tools import tool
from langchain.chat_models import init_chat_model

model = init_chat_model("claude-sonnet-4-6", temperature=0)

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add a and b."""
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """Divide a and b."""
    return a / b

tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)
```

### Step 2: Define State

```python
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
```

### Step 3: The Model Node

This node calls the model and tracks how many times it has been called.

```python
from langchain.messages import SystemMessage

def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""
    return {
        "messages": [
            model_with_tools.invoke(
                [SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")]
                + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }
```

### Step 4: The Tool Node

This node actually executes whatever tool the model asked for.

```python
from langchain.messages import ToolMessage

def tool_node(state: dict):
    """Performs the tool call"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}
```

### Step 5: The Routing Logic

This decides whether to loop back for another tool call or stop.

```python
from typing import Literal
from langgraph.graph import StateGraph, START, END

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return END
```

### Step 6: Wire It All Together

```python
agent_builder = StateGraph(MessagesState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")  # this is the loop

agent = agent_builder.compile()
```

Notice the loop: `tool_node` points back to `llm_call`. That edge is what lets the agent call a tool, see the result, and decide to call another tool, repeating until `should_continue` returns `END`. A plain linear chain cannot express this at all.

### Step 7: Run It

```python
from langchain.messages import HumanMessage

messages = [HumanMessage(content="Add 3 and 4.")]
result = agent.invoke({"messages": messages})

for m in result["messages"]:
    m.pretty_print()
```

### Optional: Visualize The Graph

```python
from IPython.display import Image, display
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))
```

---

## Two Ways To Build Agents In LangGraph

### The Graph API

What you just saw above. You explicitly define nodes and edges as a graph. This is the better mental model when your flow has real branching logic, multiple conditional paths, or you want to visualize the structure.

### The Functional API

Instead of nodes and edges, you write a normal Python function with a loop, and mark sub-steps with `@task`. Same underlying engine, different style.

```python
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages
from langchain.messages import SystemMessage, HumanMessage, ToolCall
from langchain_core.messages import BaseMessage

@task
def call_llm(messages: list[BaseMessage]):
    return model_with_tools.invoke(
        [SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")]
        + messages
    )

@task
def call_tool(tool_call: ToolCall):
    tool = tools_by_name[tool_call["name"]]
    return tool.invoke(tool_call)

@entrypoint()
def agent(messages: list[BaseMessage]):
    model_response = call_llm(messages).result()

    while True:
        if not model_response.tool_calls:
            break
        tool_result_futures = [call_tool(tc) for tc in model_response.tool_calls]
        tool_results = [fut.result() for fut in tool_result_futures]
        messages = add_messages(messages, [model_response, *tool_results])
        model_response = call_llm(messages).result()

    messages = add_messages(messages, model_response)
    return messages
```

Use the Graph API if you think visually in terms of flowcharts. Use the Functional API if you would rather just write a loop with regular `if` and `while` statements.

---

## Common Commands Reference

A quick cheat sheet of the commands and methods you will use constantly.

**Setup commands:**

```bash
pip install -U langgraph              # install LangGraph
uv add langgraph                      # install LangGraph with uv
export ANTHROPIC_API_KEY="..."        # set model API key
export LANGSMITH_TRACING=true         # turn on tracing
export LANGSMITH_API_KEY="..."        # LangSmith key
```

**Building a graph:**

```python
StateGraph(SchemaClass)                              # create a new graph builder
graph.add_node("name", function)                     # register a node
graph.add_node(function)                              # register, name inferred from function
graph.add_edge("a", "b")                              # direct edge
graph.add_conditional_edges("a", router_fn, [...])    # branching edge
graph.set_entry_point("node")                         # shortcut for START edge
graph.set_finish_point("node")                        # shortcut for END edge
graph.compile()                                       # turn builder into runnable graph
```

**Running a compiled graph:**

```python
compiled_graph.invoke(initial_state)            # run once, get final state
compiled_graph.stream(initial_state)            # get state updates as they happen
compiled_graph.batch([state1, state2])          # run multiple inputs
await compiled_graph.ainvoke(initial_state)     # async run
async for chunk in compiled_graph.astream(initial_state):  # async streaming
    ...
```

**Inspecting a graph:**

```python
compiled_graph.get_graph()                      # get the graph structure
compiled_graph.get_graph(xray=True).draw_mermaid_png()   # visual diagram
```

---

## Things That Trip Beginners Up

- **Forgetting the END edge.** If no path in your graph leads to `END`, compiling will fail. LangGraph treats this as a dead end it refuses to allow.
- **State overwriting instead of merging.** If two fields both get touched by different nodes but you didn't add a reducer like `operator.add`, the second node's return value silently replaces the first one's, not adds to it.
- **Calling methods on the builder instead of the compiled graph.** `StateGraph` itself cannot be invoked. Only the object returned by `.compile()` can run.
- **Infinite loops.** If your conditional edge never returns `END` under some condition, the graph will loop forever. Track a counter in state and force an exit once it passes a threshold, as a safety net.
- **Mixing up LangChain and LangGraph responsibilities.** LangChain wraps the model and tools. LangGraph decides the order and looping logic around them. Confusing the two often leads to writing orchestration logic inside a single giant LangChain chain instead of a graph, which then can't loop properly.

---

## Where To Go Next

Once the basics above feel natural, these are the next concepts worth learning, roughly in order of how often you'll need them:

1. **Persistence and checkpointing** — saving graph state so an agent can pause and resume later, even after a crash or restart.
2. **Human-in-the-loop** — pausing a graph mid-run so a person can review or edit the state before it continues.
3. **Memory** — giving an agent both short-term working memory for a single conversation and long-term memory across sessions.
4. **LangGraph Studio** — a visual debugger for stepping through a running graph.
5. **LangSmith tracing and deployment** — moving from local prototyping to a properly observable, hosted agent.

The official documentation index lives at the LangChain docs site and is worth bookmarking as your graphs get more complex than the examples here.