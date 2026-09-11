import os
from dataclasses import dataclass
from typing import Callable

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    SummarizationMiddleware,
    dynamic_prompt,
    wrap_model_call,
)
from langchain.tools import ToolRuntime, tool
from langchain_nebius import ChatNebius
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

load_dotenv()


model = ChatNebius(
    model="Qwen/Qwen3-235B-A22B-Instruct-2507",
    api_key=os.getenv("NEBIUS_API_KEY"),
)


@dataclass
class Context:
    user_id: str
    user_role: str
    api_key: str = ""


def dynamic_prompt_example():
    @dynamic_prompt
    def adaptive_prompt(request: ModelRequest) -> str:
        prompt = "You are a helpful assistant."

        if len(request.messages) > 10:
            prompt += " This is a long conversation, so answer concisely."

        store = request.runtime.store
        if store is not None:
            item = store.get(
                ("preferences",),
                request.runtime.context.user_id,
            )
            if item:
                style = item.value.get("communication_style", "balanced")
                prompt += f" Use a {style} communication style."

        if request.runtime.context.user_role == "admin":
            prompt += " The user has admin access."

        return prompt

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[adaptive_prompt],
        context_schema=Context,
        store=InMemoryStore(),
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Summarize our conversation so far.",
                }
            ]
        },
        context=Context(user_id="user_123", user_role="admin"),
    )

    print(result["messages"][-1].content)


def message_injection_example():
    @wrap_model_call
    def inject_files(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        uploaded = request.state.get("uploaded_files", [])
        if not uploaded:
            return handler(request)

        lines = [
            f"- {item['name']} ({item['type']}): {item['summary']}"
            for item in uploaded
        ]
        note = "Files available in this conversation:\n" + "\n".join(lines)
        messages = [*request.messages, {"role": "user", "content": note}]
        return handler(request.override(messages=messages))

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[inject_files],
    )

    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "What files do I have?"}
            ],
            "uploaded_files": [
                {
                    "name": "report.pdf",
                    "type": "pdf",
                    "summary": "Q3 sales figures",
                }
            ],
        }
    )

    print(result["messages"][-1].content)


@tool(parse_docstring=True)
def public_search(query: str) -> str:
    """Search public documentation.

    Args:
        query: The search query.
    """
    return f"Public results for: {query}"


@tool(parse_docstring=True)
def delete_account(user_id: str) -> str:
    """Delete a user account permanently.

    Args:
        user_id: The account to delete.
    """
    return f"Deleted account {user_id}"


def dynamic_tools_example():
    @wrap_model_call
    def gate_tools(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        if not request.state.get("authenticated", False):
            tools = [item for item in request.tools if item.name == "public_search"]
            return handler(request.override(tools=tools))
        return handler(request)

    agent = create_agent(
        model=model,
        tools=[public_search, delete_account],
        middleware=[gate_tools],
    )

    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "Search the docs for pricing."}
            ],
            "authenticated": False,
        }
    )

    print(result["messages"][-1].content)


def model_routing_example():
    efficient = ChatNebius(
        model="Qwen/Qwen3-8B",
        api_key=os.getenv("NEBIUS_API_KEY"),
    )
    strong = ChatNebius(
        model="Qwen/Qwen3-235B-A22B-Instruct-2507",
        api_key=os.getenv("NEBIUS_API_KEY"),
    )

    @wrap_model_call
    def route_by_length(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        chosen = strong if len(request.messages) > 10 else efficient
        return handler(request.override(model=chosen))

    agent = create_agent(
        model=efficient,
        tools=[],
        middleware=[route_by_length],
    )

    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "What is 2 + 2?"}
            ]
        }
    )

    print(result["messages"][-1].content)


class BriefAnswer(BaseModel):
    answer: str = Field(description="A brief answer")


class DetailedAnswer(BaseModel):
    answer: str = Field(description="A detailed answer")
    reasoning: str = Field(description="Explanation of the reasoning")


def response_format_example():
    @wrap_model_call
    def choose_format(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        schema = DetailedAnswer if len(request.messages) >= 3 else BriefAnswer
        return handler(request.override(response_format=schema))

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[choose_format],
    )

    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "Why is the sky blue?"}
            ]
        }
    )

    print(result["structured_response"])


@tool
def check_authentication(runtime: ToolRuntime) -> str:
    """Check whether the current session is authenticated."""
    return (
        "authenticated"
        if runtime.state.get("authenticated", False)
        else "not authenticated"
    )


@tool
def fetch_account_summary(runtime: ToolRuntime[Context]) -> str:
    """Fetch the current user's account summary."""
    return (
        f"Account summary for {runtime.context.user_id}. "
        f"Key suffix: {runtime.context.api_key[-4:]}"
    )


def tool_runtime_read_example():
    agent = create_agent(
        model=model,
        tools=[check_authentication, fetch_account_summary],
        context_schema=Context,
        store=InMemoryStore(),
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Am I logged in, and what is my account summary?",
                }
            ],
            "authenticated": True,
        },
        context=Context(
            user_id="user_123",
            user_role="user",
            api_key="secret-key-1234",
        ),
    )

    print(result["messages"][-1].content)


@tool
def authenticate(password: str, runtime: ToolRuntime) -> Command:
    """Authenticate the current session."""
    return Command(update={"authenticated": password == "correct-password"})


@tool
def save_preference(
    key: str,
    value: str,
    runtime: ToolRuntime[Context],
) -> str:
    """Save a user preference to long-term memory."""
    existing = runtime.store.get(
        ("preferences",),
        runtime.context.user_id,
    )
    preferences = dict(existing.value) if existing else {}
    preferences[key] = value
    runtime.store.put(
        ("preferences",),
        runtime.context.user_id,
        preferences,
    )
    return f"Saved {key}={value}"


def tool_write_example():
    agent = create_agent(
        model=model,
        tools=[authenticate, save_preference],
        context_schema=Context,
        store=InMemoryStore(),
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Log in with password correct-password, "
                        "then save that I prefer email replies."
                    ),
                }
            ]
        },
        context=Context(user_id="user_123", user_role="user"),
    )

    print(result["messages"][-1].content)


def summarization_example():
    agent = create_agent(
        model=model,
        tools=[],
        middleware=[
            SummarizationMiddleware(
                model=model,
                trigger={"tokens": 3000},
                keep=("messages", 15),
            )
        ],
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Let's discuss my project plan in detail. "
                        "Start by asking me about the architecture."
                    ),
                }
            ]
        }
    )

    print(result["messages"][-1].content)


def main():
    examples = {
        "dynamic-prompt": dynamic_prompt_example,
        "message-injection": message_injection_example,
        "dynamic-tools": dynamic_tools_example,
        "model-routing": model_routing_example,
        "response-format": response_format_example,
        "tool-runtime-read": tool_runtime_read_example,
        "tool-write": tool_write_example,
        "summarization": summarization_example,
    }

    for name in examples:
        print(name)

    dynamic_prompt_example()


if __name__ == "__main__":
    main()
