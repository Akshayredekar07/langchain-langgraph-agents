import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.tools import tool
from langchain_nebius import ChatNebius

load_dotenv()

model = ChatNebius(
    model="Qwen/Qwen3-235B-A22B-Instruct-2507",
    api_key=os.getenv("NEBIUS_API_KEY"),
)


@tool
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return a * b


def basic_agent():
    agent = create_agent(model=model, tools=[add, multiply])

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Calculate (25 + 15) * 3.",
                }
            ]
        }
    )

    print(result["messages"][-1].content)


def system_prompt_agent():
    agent = create_agent(
        model=model,
        tools=[add, multiply],
        system_prompt=(
            "You are a concise math assistant. "
            "Use the available tools for calculations."
        ),
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is 125 multiplied by 8?",
                }
            ]
        }
    )

    print(result["messages"][-1].content)


def multi_turn_agent():
    agent = create_agent(model=model, tools=[add, multiply])

    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "My project has 12 tasks."},
                {"role": "assistant", "content": "Understood. You have 12 tasks."},
                {
                    "role": "user",
                    "content": "If each task takes 3 hours, how many hours is that?",
                },
            ]
        }
    )

    print(result["messages"][-1].content)


class Answer(BaseModel):
    answer: str = Field(description="The final answer")
    confidence: int = Field(
        description="Confidence from 1 to 100",
        ge=1,
        le=100,
    )


def structured_agent():
    agent = create_agent(
        model=model,
        tools=[add, multiply],
        response_format=Answer,
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Explain what an AI agent is in two sentences.",
                }
            ]
        }
    )

    print(result["structured_response"])


def streaming_agent():
    agent = create_agent(model=model, tools=[add, multiply])

    for chunk in agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Explain how an agent uses tools.",
                }
            ]
        },
        stream_mode="updates",
    ):
        print(chunk)


def message_streaming_agent():
    agent = create_agent(model=model, tools=[add, multiply])

    for chunk in agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Explain tool calling in LangChain.",
                }
            ]
        },
        stream_mode="messages",
    ):
        message, metadata = chunk
        if message.content:
            print(message.content, end="", flush=True)

    print()


def summarization_agent():
    agent = create_agent(
        model=model,
        tools=[add, multiply],
        middleware=[
            SummarizationMiddleware(
                model=model,
                trigger={"messages": 20},
                keep={"messages": 8},
            )
        ],
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "We are designing a customer-support agent. "
                        "It should answer clearly and use tools when "
                        "calculations are required."
                    ),
                }
            ]
        }
    )

    print(result["messages"][-1].content)


def main():
    examples = {
        "basic": basic_agent,
        "system-prompt": system_prompt_agent,
        "multi-turn": multi_turn_agent,
        "structured": structured_agent,
        "stream": streaming_agent,
        "message-stream": message_streaming_agent,
        "summarization": summarization_agent,
    }

    print("Available examples:")
    for name in examples:
        print(f"  {name}")

    print("\nRunning: basic\n")
    basic_agent()


if __name__ == "__main__":
    main()
