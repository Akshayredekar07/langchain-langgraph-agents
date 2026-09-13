import asyncio
from typing import TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_nebius import ChatNebius

load_dotenv()

model = ChatNebius(model="Qwen/Qwen3-Coder-480B-A35B-Instruct")


@tool
async def add(a: int, b: int) -> int:
    """Add two integers."""
    await asyncio.sleep(0.1)
    return a + b


@tool
async def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    await asyncio.sleep(0.1)
    return a * b


async def basic_async_agent():
    agent = create_agent(
        model=model,
        tools=[add, multiply],
        system_prompt="You are a helpful math assistant.",
    )

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Calculate (12 + 8) * 3."}]}
    )
    print(result["messages"][-1].content)


async def async_stream_updates():
    agent = create_agent(
        model=model,
        tools=[add, multiply],
    )

    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": "Calculate 25 * 4."}]},
        stream_mode="updates",
    ):
        print(chunk)


async def async_stream_messages():
    agent = create_agent(
        model=model,
        tools=[add, multiply],
    )

    async for token, metadata in agent.astream(
        {"messages": [{"role": "user", "content": "Explain why 7 * 8 = 56."}]},
        stream_mode="messages",
    ):
        if token.content:
            print(token.content, end="", flush=True)
    print()


async def async_batch():
    agent = create_agent(
        model=model,
        tools=[add, multiply],
    )

    requests = [
        {"messages": [{"role": "user", "content": "What is 10 + 5?"}]},
        {"messages": [{"role": "user", "content": "What is 6 * 7?"}]},
        {"messages": [{"role": "user", "content": "What is 100 + 250?"}]},
    ]

    results = await agent.abatch(requests)

    for result in results:
        print(result["messages"][-1].content)


async def concurrent_agents():
    agent = create_agent(
        model=model,
        tools=[add, multiply],
    )

    questions = [
        "Calculate 15 + 27.",
        "Calculate 9 * 11.",
        "Calculate 50 + 75.",
    ]

    results = await asyncio.gather(
        *(
            agent.ainvoke({"messages": [{"role": "user", "content": question}]})
            for question in questions
        )
    )

    for result in results:
        print(result["messages"][-1].content)


async def main():
    await basic_async_agent()
    # await async_stream_updates()
    # await async_stream_messages()
    # await async_batch()
    # await concurrent_agents()


if __name__ == "__main__":
    asyncio.run(main())
