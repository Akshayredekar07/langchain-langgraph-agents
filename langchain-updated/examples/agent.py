from langchain.agents import create_agent
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()
from langchain.chat_models import init_chat_model

model = init_chat_model(model="moonshotai/kimi-k2-instruct-0905", model_provider="groq")

def get_weather(city: str) -> str:
    """Get the weather of the city"""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)


def run_demo() -> None:
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
    )
    pprint(response)


if __name__ == "__main__":
    run_demo()
