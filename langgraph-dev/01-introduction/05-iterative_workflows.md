# LangGraph Iterative Workflows — Notes

## **What Is an Iterative Workflow**

Till now we saw conditional workflows, where the graph picks one path out of many and moves forward. An iterative workflow is different — here the graph goes back to a node it already ran, and runs it again. This is basically a loop, but built with graph edges instead of a Python `while` loop.

You need this any time the task is "do this, check if it's good enough, if not do it again." A few real examples:

- An LLM writes a tweet, another step checks if it's good, if not it goes back and rewrites it
- An agent calls a tool, checks the result, and if the task is not done yet, it calls another tool
- A support ticket gets processed one by one from a list until the list is empty
- A RAG pipeline retrieves documents, checks if they're relevant, and re-searches with a better query if not

In plain graph terms: a **normal edge** goes A → B. A **loop** is when an edge goes back, like B → A. That back edge is what creates the "iteration."

---

## **The Core Building Block: A Back Edge**

The whole trick of iterative workflows is just this — one node's output leads to a conditional edge, and one of the options in that conditional edge points back to a node you already visited.

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    text: str

def generate(state: State) -> State:
    return {"text": "some generated text"}

def check(state: State) -> str:
    if "good" in state["text"]:
        return "done"
    return "retry"

graph = StateGraph(State)
graph.add_node("generate", generate)
graph.add_edge(START, "generate")
graph.add_conditional_edges("generate", check, {"done": END, "retry": "generate"})

workflow = graph.compile()
```

Here `"retry"` points back to `"generate"` itself. That one line is the loop. Nothing fancy, no special "loop" keyword in LangGraph — it's just a normal edge, but the destination happens to be a node you've already been to.

---

## **Simple Example: Generate → Evaluate → Refine**

This is the most common iterative pattern you'll see in real projects — an LLM generates something, another step grades it, and if it's not good, it goes back and improves it. Think of things like: AI writes a joke, checks if it's funny, and rewrites it if not.

```python
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    topic: str
    joke: str
    feedback: str
    grade: str

class Feedback(BaseModel):
    grade: Literal["funny", "not_funny"] = Field(description="Is the joke funny")
    feedback: str = Field(description="How to improve it if not funny")

llm = ...  # your chat model
evaluator = llm.with_structured_output(Feedback)

def generate_joke(state: State) -> State:
    if state.get("feedback"):
        prompt = f"Write a joke about {state['topic']}. Take this feedback into account: {state['feedback']}"
    else:
        prompt = f"Write a joke about {state['topic']}"
    return {"joke": llm.invoke(prompt).content}

def evaluate_joke(state: State) -> State:
    result = evaluator.invoke(f"Grade this joke: {state['joke']}")
    return {"grade": result.grade, "feedback": result.feedback}

def route_after_evaluation(state: State) -> Literal["done", "retry"]:
    return "done" if state["grade"] == "funny" else "retry"

graph = StateGraph(State)
graph.add_node("generate_joke", generate_joke)
graph.add_node("evaluate_joke", evaluate_joke)

graph.add_edge(START, "generate_joke")
graph.add_edge("generate_joke", "evaluate_joke")
graph.add_conditional_edges(
    "evaluate_joke",
    route_after_evaluation,
    {"done": END, "retry": "generate_joke"},
)

workflow = graph.compile()
```

Notice the flow: `generate_joke` → `evaluate_joke` → (loop back to `generate_joke` OR go to `END`). This same shape works for basically any "produce and improve" task — content writing, code generation with test checking, essay grading, resume tailoring, and so on.

---

## **Why You Must Add a Stop Condition (Very Important)**

A loop that never stops will keep calling your LLM forever, and LangGraph will eventually throw an error on its own once it crosses the default step limit (usually 25 steps). This is called the **recursion limit**, and hitting it is almost always a sign your loop logic has a bug, not that your task genuinely needed more steps.

So every iterative workflow needs its own explicit stop condition, on top of whatever LangGraph gives you by default. The two common ways:

**1. A max iteration counter in state**

```python
class State(TypedDict):
    joke: str
    feedback: str
    grade: str
    iteration: int
    max_iteration: int

def route_after_evaluation(state: State) -> Literal["done", "retry"]:
    if state["grade"] == "funny" or state["iteration"] >= state["max_iteration"]:
        return "done"
    return "retry"
```

You bump `iteration` by 1 inside your generate node each time it runs. This is the safest pattern — even if the LLM keeps grading things poorly forever, your loop stops after N tries.

**2. Increasing the recursion limit when you call the graph**

```python
result = workflow.invoke(initial_state, config={"recursion_limit": 50})
```

This just raises LangGraph's own hard ceiling. Use this only for graphs that legitimately need many steps — don't use this as a fix for a loop that should be terminating on its own but isn't. If you're not expecting your graph to go through many iterations and you're hitting the limit, that almost always means there's a bug in your exit logic, not that the limit needs to be bigger.

**Rule of thumb:** always have your own counter-based exit condition. Treat `recursion_limit` as a safety net, not your actual stop logic.

---

## **Iterative Example: Processing a List of Items One by One**

Another very common use of loops — you have a list of things to process, and you handle them one at a time using the same node, until the list is empty. This shows up in batch processing pipelines, ticket queues, and grading systems.

```python
from typing import TypedDict, List

class State(TypedDict):
    tasks: List[str]
    completed: List[str]
    current_task: str

def pick_task(state: State) -> State:
    if state["tasks"]:
        current = state["tasks"].pop(0)
        return {"current_task": current, "tasks": state["tasks"]}
    return {"current_task": None}

def process_task(state: State) -> State:
    completed = state["completed"] + [state["current_task"]]
    return {"completed": completed}

def router(state: State) -> str:
    return "loop" if state["tasks"] else "finish"

graph = StateGraph(State)
graph.add_node("pick_task", pick_task)
graph.add_node("process_task", process_task)

graph.add_edge(START, "pick_task")
graph.add_edge("pick_task", "process_task")
graph.add_conditional_edges("process_task", router, {"loop": "pick_task", "finish": END})

workflow = graph.compile()
```

Here the "state that shrinks" (`tasks` list getting smaller) is itself the exit condition — no separate counter needed, since the list running out naturally ends the loop.

---

## **Advanced: The ReAct Agent Loop (Tool-Calling Agents)**

This is the loop pattern behind almost every "AI agent" you've heard of — the model decides whether it needs a tool, calls it, looks at the result, and decides again. This keeps looping until the model decides it has enough information to answer.

```python
from langgraph.graph import MessagesState
from langchain.messages import SystemMessage, ToolMessage

def llm_call(state: MessagesState):
    response = llm_with_tools.invoke(
        [SystemMessage(content="You are a helpful assistant.")] + state["messages"]
    )
    return {"messages": [response]}

def tool_node(state: MessagesState):
    results = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        output = tool.invoke(tool_call["args"])
        results.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
    return {"messages": results}

def should_continue(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    return "tool_node" if last_message.tool_calls else END

graph = StateGraph(MessagesState)
graph.add_node("llm_call", llm_call)
graph.add_node("tool_node", tool_node)

graph.add_edge(START, "llm_call")
graph.add_conditional_edges("llm_call", should_continue, {"tool_node": "tool_node", END: END})
graph.add_edge("tool_node", "llm_call")

workflow = graph.compile()
```

The loop here is `llm_call → tool_node → llm_call → tool_node → ...` and it only stops once the model responds without asking for another tool call. This is why agent frameworks are naturally built on graphs with loops — a single "chain" cannot represent "call a tool an unknown number of times."

**A real production bug to know about:** even careful stop-condition logic can occasionally fail if the underlying model keeps requesting tools no matter what (some model versions/providers have had bugs like this). This is exactly why the `recursion_limit` safety net exists — it's your last line of defense even when your `should_continue` logic looks correct on paper.

---

## **Advanced: Human-in-the-Loop Iteration**

Sometimes the "check" step in your loop isn't done by another LLM call — it's done by an actual person reviewing the output and asking for changes. LangGraph supports pausing a graph mid-run, waiting for a human, and then resuming — which is a form of iteration where a human is the "evaluator" in the loop.

The core idea (conceptually, without going deep into the API since it changes across versions): a node calls something like `interrupt(...)`, which freezes the graph at that exact point and hands the paused state back to your app. A human looks at it (through a dashboard, Slack message, etc.), decides "approve" or "send back for changes," and your app resumes the graph with that decision. If it's "send back for changes," the graph loops back to the generation node, same as the automated version — just with a person instead of an LLM deciding whether to loop again.

This pattern is common in:
- Draft approval workflows (blog posts, legal documents, marketing copy)
- High-stakes actions where you don't want full automation (sending money, deleting data, contacting a customer)
- Any pipeline where quality matters more than speed

---

## **Advanced: Parallel Iteration (Send API)**

Normal loops process one thing at a time. But sometimes you want to run the same node many times in parallel — for example, writing 5 sections of a report at once, instead of one after another. LangGraph has a `Send` mechanism for this.

```python
from langgraph.types import Send

def assign_workers(state):
    return [Send("write_section", {"section": s}) for s in state["sections"]]

graph.add_conditional_edges("plan_sections", assign_workers, ["write_section"])
```

This isn't a "loop" in the back-edge sense — it's more like a fan-out. But it solves the same underlying problem as an iterative workflow: "run this node once per item in a list." The difference is speed — `Send` runs all the section-writers at the same time instead of one after another, which matters a lot once you have many items to process.

---

## **Common Mistakes (Learned From Real Bug Reports)**

**1. Forgetting the stop condition entirely.** The loop runs forever until LangGraph's recursion limit kills it with an ugly error. Always design the exit condition first, before writing the loop body.

**2. Relying only on `recursion_limit` as your stop logic.** This is a safety net, not a design pattern. If your graph is regularly hitting the limit, the fix is almost never "raise the limit" — it's "fix why the loop isn't exiting."

**3. Returning the entire state from a node instead of just the changed keys.** In LangGraph, a node should return only the fields it actually updated, not the whole state object. Returning everything can silently overwrite other fields other nodes were tracking, especially in graphs with parallel branches.

**4. No counter when the exit condition depends on an LLM's judgment.** If your loop exits when an LLM says "this is good now," you need a hard iteration cap too — an LLM can be wrong or inconsistent about grading its own (or another LLM's) output, and without a cap that becomes an infinite loop in practice.

**5. Side effects (API calls, sending emails, writing to a database) placed before a human-in-the-loop pause.** If the graph re-runs that part after resuming, the side effect can fire twice. Put side effects after the pause point, not before it.

---

## **When to Actually Use an Iterative Workflow**

Not everything needs a loop. Use one when:

- The output needs to be "checked and improved," not just produced once (content generation, code generation, structured extraction with validation)
- You're processing a list of items one at a time, or a queue that grows/shrinks over time
- You're building an agent that needs to decide, act, observe, and decide again — an unknown number of times, not a fixed number of steps
- Quality matters more than a single-shot answer, and you're willing to pay for multiple LLM calls to get there

Don't use one when a single call already gets you where you need to be — loops add latency, cost, and a bit of complexity (you now have to think about termination), so they should be a deliberate choice, not a default.