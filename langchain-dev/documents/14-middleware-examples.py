import argparse
import os
from typing import Any, Callable, cast

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ContextEditingMiddleware,
    HumanInTheLoopMiddleware,
    LLMToolEmulator,
    LLMToolSelectorMiddleware,
    ModelFallbackMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware, #type: ignore
    ToolRetryMiddleware,
    before_model,
    after_model,
    wrap_model_call,
)
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_nebius import ChatNebius
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langgraph.runtime import Runtime
from typing_extensions import NotRequired

load_dotenv()


def make_model(provider: str, role: str = "main"):
    if provider == "nebius":
        key = os.getenv("NEBIUS_API_KEY")
        if not key:
            raise RuntimeError("NEBIUS_API_KEY is missing from .env")
        names = {
            "main": os.getenv(
                "NEBIUS_MODEL",
                "Qwen/Qwen3-235B-A22B-Instruct-2507",
            ),
            "cheap": os.getenv("NEBIUS_CHEAP_MODEL", "Qwen/Qwen3-8B"),
        }
        return ChatNebius(model=names.get(role, names["main"]), api_key=key)

    if provider == "nvidia":
        key = os.getenv("NVIDIA_API_KEY")
        if not key:
            raise RuntimeError("NVIDIA_API_KEY is missing from .env")
        names = {
            "main": os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct"),
            "cheap": os.getenv(
                "NVIDIA_CHEAP_MODEL", "meta/llama-3.1-8b-instruct"
            ),
        }
        return ChatNVIDIA(model=names.get(role, names["main"]), api_key=key)

    raise ValueError("provider must be 'nebius' or 'nvidia'")


def basic_hooks(provider: str):
    model = make_model(provider)

    @before_model
    def before(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"messages={len(state['messages'])}")
        return None

    @after_model
    def after(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"answer={str(state['messages'][-1].content)[:100]}")
        return None

    agent = create_agent(model=model, tools=[], middleware=[before, after])
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Say hello in five words."}]}
    )
    print(result["messages"][-1].content)


def model_fallback(provider: str):
    primary = make_model(provider, "main")
    backup_provider = "nvidia" if provider == "nebius" else "nebius"
    backup = make_model(backup_provider, "main")

    agent = create_agent(
        model=primary,
        tools=[],
        middleware=[ModelFallbackMiddleware(backup)],
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is the capital of Japan?"}]}
    )
    print(result["messages"][-1].content)


def dynamic_model(provider: str):
    cheap = make_model(provider, "cheap")
    strong = make_model(provider, "main")

    @wrap_model_call
    def route(
        request,
        handler: Callable,
    ):
        model = strong if len(request.messages) > 6 else cheap
        return handler(request.override(model=model))

    agent = create_agent(model=cheap, tools=[], middleware=[route])
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Explain middleware in three short points.",
                }
            ]
        }
    )
    print(result["messages"][-1].content)


def state_update(provider: str):
    model = make_model(provider)

    class CallState(AgentState):
        model_calls: NotRequired[int]

    @wrap_model_call(state_schema=CallState)
    def count_calls(request, handler):
        response = handler(request)
        count = request.state.get("model_calls", 0) + 1
        return type(response)(
            model_response=response,
            command=Command(update={"model_calls": count}),
        )

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[count_calls],
        state_schema=CallState,
    )
    result = agent.invoke(
        cast(
            CallState,
            {
            "messages": [{"role": "user", "content": "Hello."}],
            "model_calls": 0,
            },
        )
    )
    print("model_calls:", result["model_calls"])


def early_exit(provider: str):
    model = make_model(provider)

    @after_model
    def stop(state: AgentState, runtime: Runtime):
        content = str(state["messages"][-1].content)
        if "CANNOT_HELP" in content:
            return {
                "messages": [],
                "jump_to": "end",
            }
        return None

    agent = create_agent(model=model, tools=[], middleware=[stop])
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Answer normally."}]}
    )
    print(result["messages"][-1].content)


def ordering(provider: str):
    model = make_model(provider)

    @before_model
    def first_before(state: AgentState, runtime: Runtime):
        print("before:first")
        return None

    @before_model
    def second_before(state: AgentState, runtime: Runtime):
        print("before:second")
        return None

    @after_model
    def first_after(state: AgentState, runtime: Runtime):
        print("after:first")
        return None

    @after_model
    def second_after(state: AgentState, runtime: Runtime):
        print("after:second")
        return None

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[first_before, second_before, first_after, second_after],
    )
    agent.invoke({"messages": [{"role": "user", "content": "Hi."}]})


def class_middleware(provider: str):
    model = make_model(provider)

    class CallLimit(AgentMiddleware):
        def __init__(self, limit: int):
            super().__init__()
            self.limit = limit
            self.calls = 0

        def before_model(self, state: AgentState, runtime: Runtime):
            self.calls += 1
            if self.calls > self.limit:
                return {"jump_to": "end"}
            return None

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[CallLimit(3)],
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Count from one to three."}]}
    )
    print(result["messages"][-1].content)


def reliability(provider: str):
    model = make_model(provider)

    @tool
    def flaky_lookup(query: str) -> str:
        """Look up a value and occasionally fail."""
        raise ConnectionError(f"upstream failed for {query}")

    def on_error(exc: Exception, request) -> str | None:
        if isinstance(exc, ConnectionError):
            return f"{request.tool_call['name']} failed: {type(exc).__name__}"
        return None

    agent = create_agent(
        model=model,
        tools=[flaky_lookup],
        middleware=[
            ToolRetryMiddleware(
                max_retries=2,
                retry_on=(ConnectionError,),
                on_failure="error",
            ),
            ToolErrorMiddleware(on_error=on_error),
        ],
    )
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Look up the Pune weather station.",
                }
            ]
        }
    )
    print(result["messages"][-1].content)


def context_control(provider: str):
    model = make_model(provider)
    agent = create_agent(
        model=model,
        tools=[],
        middleware=[
            SummarizationMiddleware(
                model=model,
                trigger=("messages", 20),
                keep=("messages", 8),
            )
        ],
    )
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Start a discussion about database indexing.",
                }
            ]
        }
    )
    print(result["messages"][-1].content)


def context_editing(provider: str):
    from langchain.agents.middleware import ClearToolUsesEdit

    model = make_model(provider)
    agent = create_agent(
        model=model,
        tools=[],
        middleware=[
            ContextEditingMiddleware(
                edits=[ClearToolUsesEdit(trigger=20000, keep=3)]
            )
        ],
    )
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "Explain how tool output can bloat context."}
            ]
        }
    )
    print(result["messages"][-1].content)


def safety(provider: str):
    model = make_model(provider)

    @tool
    def send_notification(to: str, message: str) -> str:
        """Send a notification."""
        return f"sent to {to}"

    agent = create_agent(
        model=model,
        tools=[send_notification],
        checkpointer=InMemorySaver(),
        middleware=[
            PIIMiddleware("email", strategy="redact", apply_to_input=True),
            ToolCallLimitMiddleware(
                tool_name="send_notification",
                run_limit=2,
                exit_behavior="continue",
            ),
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "send_notification": {
                        "allowed_decisions": ["approve", "edit", "reject"]
                    }
                }
            ),
        ],
    )
    config: RunnableConfig = {"configurable": {"thread_id": "middleware-demo"}}
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Notify person@example.com that the job finished.",
                }
            ]
        },
        config=config,
    )
    print(result)


def tool_selection(provider: str):
    model = make_model(provider, "main")
    selector = make_model(provider, "cheap")

    @tool
    def read_file(path: str) -> str:
        """Read a file."""
        return f"contents of {path}"

    @tool
    def write_file(path: str, content: str) -> str:
        """Write a file."""
        return f"wrote {path}"

    @tool
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"Sunny in {city}"

    agent = create_agent(
        model=model,
        tools=[read_file, write_file, get_weather],
        middleware=[
            LLMToolSelectorMiddleware(model=selector, max_tools=2)
        ],
    )
    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "Read config.yaml and summarize it."}
            ]
        }
    )
    print(result["messages"][-1].content)


def emulator(provider: str):
    model = make_model(provider)

    @tool
    def get_weather(city: str) -> str:
        """Get current weather for a city."""
        raise NotImplementedError("real weather API is not connected")

    agent = create_agent(
        model=model,
        tools=[get_weather],
        middleware=[LLMToolEmulator(tools=["get_weather"], model=model)],
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather in Pune?"}]}
    )
    print(result["messages"][-1].content)


def pii(provider: str):
    model = make_model(provider)
    agent = create_agent(
        model=model,
        tools=[],
        middleware=[
            PIIMiddleware(
                "email",
                strategy="redact",
                apply_to_input=True,
                apply_to_output=True,
            )
        ],
    )
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "My email is person@example.com. Repeat it back.",
                }
            ]
        }
    )
    print(result["messages"][-1].content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "example",
        choices=[
            "hooks",
            "fallback",
            "dynamic-model",
            "state",
            "early-exit",
            "ordering",
            "class",
            "reliability",
            "context",
            "context-editing",
            "safety",
            "tool-selection",
            "emulator",
            "pii",
        ],
    )
    parser.add_argument(
        "--provider",
        choices=["nebius", "nvidia"],
        default="nebius",
    )
    args = parser.parse_args()

    examples = {
        "hooks": basic_hooks,
        "fallback": model_fallback,
        "dynamic-model": dynamic_model,
        "state": state_update,
        "early-exit": early_exit,
        "ordering": ordering,
        "class": class_middleware,
        "reliability": reliability,
        "context": context_control,
        "context-editing": context_editing,
        "safety": safety,
        "tool-selection": tool_selection,
        "emulator": emulator,
        "pii": pii,
    }
    examples[args.example](args.provider)


if __name__ == "__main__":
    main()
