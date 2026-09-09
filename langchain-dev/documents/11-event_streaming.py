import argparse
import asyncio
import os
from typing import Any, Iterator

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, PIIMiddleware
from langchain_nebius import ChatNebius
from langchain_nvidia_ai_endpoints import ChatNVIDIA
# PowerShell setup:
# python -m pip install --upgrade langgraph langchain langchain-core langchain-nebius langchain-nvidia-ai-endpoints
# StreamTransformer is not a public import in current LangGraph releases. Use
# the compatible protocol below instead: a key and a transform method.



load_dotenv()


def nebius_model():
    key = os.getenv("NEBIUS_API_KEY")
    if not key:
        raise RuntimeError("NEBIUS_API_KEY is missing from .env")
    return ChatNebius(
        model="meta-llama/Llama-3.3-70B-Instruct",
        api_key=key, #type: ignore
        temperature=0.3,
    )


def reasoning_model():
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        raise RuntimeError("NVIDIA_API_KEY is missing from .env")
    return ChatNVIDIA(
        model="nvidia/llama-3.1-nemotron-70b-instruct",
        api_key=key,
        temperature=0.3,
    )


def example_messages():
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"It's always sunny in {city}!"

    agent = create_agent(model=nebius_model(), tools=[get_weather])
    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": "What's the weather in Pune?"}]},
        version="v3",
    )

    for message in stream.messages:
        for delta in message.text:
            print(delta, end="", flush=True)

    print()
    print(stream.output["messages"][-1].content)


def example_usage():
    agent = create_agent(model=nebius_model(), tools=[])
    stream = agent.stream_events(
        {
            "messages": [
                {"role": "user", "content": "Explain vector databases in 2 lines."}
            ]
        },
        version="v3",
    )

    for message in stream.messages:
        print(f"[node={message.node}] ", end="")
        for delta in message.text:
            print(delta, end="", flush=True)
        print()
        print("usage:", message.output.usage_metadata)


def example_reasoning():
    agent = create_agent(model=reasoning_model(), tools=[])
    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": "Is 1099 a prime number? Think it through."}]},
        version="v3",
    )

    for message in stream.messages:
        for delta in message.reasoning:
            print(f"[thinking] {delta}", end="", flush=True)
        for delta in message.text:
            print(delta, end="", flush=True)

    print()


def example_tool_calls():
    def get_stock_price(ticker: str) -> str:
        """Get the current stock price for a ticker symbol."""
        return f"{ticker} is trading at $412.50"

    agent = create_agent(model=nebius_model(), tools=[get_stock_price])
    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": "What's the price of NVDA?"}]},
        version="v3",
    )

    for message in stream.messages:
        for chunk in message.tool_calls:
            print("tool call chunk:", chunk)
        finalized = message.tool_calls.get()
        if finalized:
            print("finalized tool calls:", finalized)


def example_tool_execution():
    def divide(a: float, b: float) -> float:
        """Divide a by b."""
        return a / b

    agent = create_agent(model=nebius_model(), tools=[divide])
    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": "What is 100 divided by 0?"}]},
        version="v3",
    )

    for call in stream.tool_calls:
        print(f"{call.tool_name}({call.input})")
        for delta in call.output_deltas:
            print(delta, end="", flush=True)
        print("\noutput:", call.output, "| error:", call.error)


def example_subagents():
    def get_weather(city: str) -> str:
        """Get weather for a given city."""
        return f"It's always sunny in {city}!"

    weather_agent = create_agent(
        model=nebius_model(),
        tools=[get_weather],
        name="weather_agent",
    )

    def call_weather(query: str) -> str:
        """Query the weather agent."""
        result = weather_agent.invoke(
            {"messages": [{"role": "user", "content": query}]}
        )
        return result["messages"][-1].content

    supervisor = create_agent(
        model=nebius_model(),
        tools=[call_weather],
        name="supervisor",
    )

    stream = supervisor.stream_events(
        {"messages": [{"role": "user", "content": "What's the weather in Mumbai?"}]},
        version="v3",
    )

    for subagent in stream.subagents:
        print(f"{subagent.name} (cause={subagent.cause}): ", end="")
        for message in subagent.messages:
            for token in message.text:
                print(token, end="", flush=True)
        print()


def example_state():
    agent = create_agent(model=nebius_model(), tools=[])
    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": "List 3 use cases for RAG."}]},
        version="v3",
    )

    for snapshot in stream.values:
        print("STATE SNAPSHOT:", snapshot)


def example_interleave():
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"It's always sunny in {city}!"

    agent = create_agent(model=nebius_model(), tools=[get_weather])
    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": "Weather in Pune?"}]},
        version="v3",
    )

    for name, item in stream.interleave("messages", "tool_calls", "values"):
        if name == "messages":
            for delta in item.text:
                print(delta, end="", flush=True)
        elif name == "tool_calls":
            print(f"\n[tool] {item.tool_name}({item.input})")
        else:
            print("\n[state]", item)


def example_raw():
    agent = create_agent(model=nebius_model(), tools=[])
    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": "Hi"}]},
        version="v3",
    )

    for event in stream:
        print(event["method"], event["params"]["namespace"], event["params"]["data"])


class ToolActivityTransformer:
    """Custom stream transformer using LangGraph's key/transform protocol."""

    def __init__(self, key: str = "tool_activity") -> None:
        self.key = key

    def transform(self, chunk: Any) -> Iterator[Any]:
        if isinstance(chunk, dict) and chunk.get("event") == "on_tool_start":
            yield {self.key: f"tool started: {chunk['name']}"}


def example_transformer():
    def search_docs(query: str) -> str:
        """Search internal docs."""
        return f"3 results found for '{query}'"

    agent = create_agent(model=nebius_model(), tools=[search_docs])
    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": "Search docs for retrieval."}]},
        version="v3",
    )

    transformer = ToolActivityTransformer()
    for chunk in stream:
        for activity in transformer.transform(chunk):
            print("ACTIVITY:", activity[transformer.key])


class ToolActivityMiddleware(AgentMiddleware):
    transformers = (ToolActivityTransformer(),)


def example_middleware():
    def search_docs(query: str) -> str:
        """Search internal docs."""
        return f"3 results found for '{query}'"

    agent = create_agent(
        model=nebius_model(),
        tools=[search_docs],
        middleware=[ToolActivityMiddleware()],
    )

    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": "Search docs for embeddings."}]},
        version="v3",
    )

    for activity in stream.extensions["tool_activity"]:
        print("ACTIVITY:", activity)


def example_pii():
    agent = create_agent(
        model=nebius_model(),
        tools=[],
        middleware=[
            PIIMiddleware("email", strategy="redact", apply_to_output=True)
        ],
    )

    stream = agent.stream_events(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "My email is akshay@example.com, confirm it back to me.",
                }
            ]
        },
        version="v3",
    )

    for message in stream.messages:
        for delta in message.text:
            print(delta, end="", flush=True)

    print()


def example_async():
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"It's always sunny in {city}!"

    agent = create_agent(model=nebius_model(), tools=[get_weather])

    async def main():
        stream = agent.astream_events(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Weather in Delhi, then explain monsoons briefly.",
                    }
                ]
            },
            version="v3",
        )

        async def consume_messages():
            async for message in stream.messages: #type: ignore
                for delta in message.text:
                    print(delta, end="", flush=True)

        async def consume_tools():
            async for call in stream.tool_calls: #type: ignore
                print(f"\n[tool] {call.tool_name}({call.input})")

        await asyncio.gather(consume_messages(), consume_tools())

    asyncio.run(main())


def main():
    examples = {
        "messages": example_messages,
        "usage": example_usage,
        "reasoning": example_reasoning,
        "tool-calls": example_tool_calls,
        "tool-execution": example_tool_execution,
        "subagents": example_subagents,
        "state": example_state,
        "interleave": example_interleave,
        "raw": example_raw,
        "transformer": example_transformer,
        "middleware": example_middleware,
        "pii": example_pii,
        "async": example_async,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("example", choices=examples)
    args = parser.parse_args()
    examples[args.example]()


if __name__ == "__main__":
    main()
