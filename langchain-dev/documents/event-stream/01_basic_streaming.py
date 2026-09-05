import asyncio
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_nebius import ChatNebius
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import warnings
warnings.filterwarnings("ignore")

load_dotenv()


def sync_stream_nebius():
    model = ChatNebius(model="zai-org/GLM-5.2", temperature=0.3)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant who answers in 2 short sentences."),
        ("human", "Tell me about the planet Mars."),
    ])
    chain = prompt | model

    for chunk in chain.stream({}):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()


async def async_stream_nvidia():
    model = ChatNebius(model="zai-org/GLM-5.2", temperature=0.3)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a poet who writes in 4 lines."),
        ("human", "Write a haiku about the ocean."),
    ])
    chain = prompt | model

    async for chunk in chain.astream({}):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()


def stream_lcel_chain():
    model = ChatNebius(model="zai-org/GLM-5.2", temperature=0.3)
    parser = StrOutputParser()
    chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a concise assistant. Answer in 1 sentence."),
            ("human", "What is the capital of {country}?"),
        ])
        | model
        | parser
    )

    for chunk in chain.stream({"country": "Japan"}):
        print(chunk, end="", flush=True)
    print()


def accumulate_chunks():
    model = ChatNebius(model="zai-org/GLM-5.2", temperature=0.3)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "List 3 colors of the rainbow."),
    ])
    chain = prompt | model

    full_message = None
    for chunk in chain.stream({}):
        full_message = chunk if full_message is None else full_message + chunk
        if chunk.content:
            print(chunk.content, end="", flush=True)

    print(f"\nContent: {full_message.content!r}") #type: ignore
    if full_message.usage_metadata: #type: ignore
        print(f"Usage: {full_message.usage_metadata}") #type: ignore


if __name__ == "__main__":
    sync_stream_nebius()
    asyncio.run(async_stream_nvidia())
    stream_lcel_chain()
    accumulate_chunks()