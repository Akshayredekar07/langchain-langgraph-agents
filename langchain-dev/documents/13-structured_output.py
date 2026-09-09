import argparse
import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain.agents import create_agent
from langchain.agents.structured_output import (
    MultipleStructuredOutputsError,
    StructuredOutputValidationError,
    ToolStrategy,
)
from langchain_nebius import ChatNebius
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()


def make_model(provider: str):
    if provider == "nvidia":
        if not os.getenv("NVIDIA_API_KEY"):
            raise RuntimeError("NVIDIA_API_KEY is missing from .env")
        return ChatNVIDIA(model="nvidia/nemotron-3-super-120b-a12b")

    if provider == "nebius":
        if not os.getenv("NEBIUS_API_KEY"):
            raise RuntimeError("NEBIUS_API_KEY is missing from .env")
        return ChatNebius(model="Qwen/Qwen3-30B-A3B-fast")

    raise ValueError("provider must be 'nvidia' or 'nebius'")


class ContactInfo(BaseModel):
    name: str = Field(description="Person's name")
    email: str = Field(description="Person's email")
    phone: str = Field(description="Person's phone number")


class ProductReview(BaseModel):
    rating: int | None = Field(description="Rating from 1 to 5", ge=1, le=5)
    sentiment: Literal["positive", "negative"] = Field(
        description="Review sentiment"
    )
    key_points: list[str] = Field(
        description="Important review points, each 1 to 3 words"
    )


@dataclass
class MeetingAction:
    task: str
    assignee: str
    priority: Literal["low", "medium", "high"]


class ReviewDict(TypedDict):
    rating: int | None
    sentiment: Literal["positive", "negative"]
    key_points: list[str]


class CustomerComplaint(BaseModel):
    issue_type: Literal["product", "service", "shipping", "billing"]
    severity: Literal["low", "medium", "high"]
    description: str


def example_auto(provider: str):
    model = make_model(provider)
    agent = create_agent(model=model, response_format=ProductReview)

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Analyze this review: "
                        "'Great product, 5 out of 5 stars. "
                        "Fast shipping, but expensive.'"
                    ),
                }
            ]
        }
    )

    print(result["structured_response"])


def example_tool_pydantic(provider: str):
    model = make_model(provider)
    agent = create_agent(
        model=model,
        response_format=ToolStrategy(ProductReview),
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Analyze this review: "
                        "'Great product, 5 out of 5 stars. "
                        "Fast shipping, but expensive.'"
                    ),
                }
            ]
        }
    )

    print(result["structured_response"])


def example_dataclass(provider: str):
    model = make_model(provider)
    agent = create_agent(
        model=model,
        response_format=ToolStrategy(MeetingAction),
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "From the meeting: Sarah needs to update "
                        "the project timeline as soon as possible."
                    ),
                }
            ]
        }
    )

    print(result["structured_response"])


def example_typeddict(provider: str):
    model = make_model(provider)
    agent = create_agent(
        model=model,
        response_format=ToolStrategy(ReviewDict),
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Analyze this review: "
                        "'The camera is excellent and easy to use, "
                        "but the battery is weak.'"
                    ),
                }
            ]
        }
    )

    print(result["structured_response"])


def example_union(provider: str):
    model = make_model(provider)
    agent = create_agent(
        model=model,
        response_format=ToolStrategy(
            ProductReview | CustomerComplaint
        ),
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Analyze this text: "
                        "'The product works well, but shipping was delayed.'"
                    ),
                }
            ]
        }
    )

    print(result["structured_response"])


def example_custom_message(provider: str):
    model = make_model(provider)
    agent = create_agent(
        model=model,
        response_format=ToolStrategy(
            schema=MeetingAction,
            tool_message_content="Action item captured.",
        ),
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Sarah must update the project timeline "
                        "with high priority."
                    ),
                }
            ]
        }
    )

    print(result["structured_response"])


def example_validation_retry(provider: str):
    model = make_model(provider)
    agent = create_agent(
        model=model,
        response_format=ToolStrategy(ProductReview),
        system_prompt=(
            "Extract the review faithfully. "
            "The rating must be between 1 and 5."
        ),
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Parse this review: Amazing product, 10/10!",
                }
            ]
        }
    )

    print(result["structured_response"])


def example_custom_errors(provider: str):
    model = make_model(provider)

    def handle_error(error: Exception) -> str:
        if isinstance(error, StructuredOutputValidationError):
            return "The structured data was invalid. Correct it and try again."
        if isinstance(error, MultipleStructuredOutputsError):
            return "Return only one structured response."
        return f"Structured output error: {error}"

    agent = create_agent(
        model=model,
        response_format=ToolStrategy(
            schema=ProductReview,
            handle_errors=handle_error,
        ),
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Analyze this review: "
                        "'Excellent product, 5/5, but expensive.'"
                    ),
                }
            ]
        }
    )

    print(result["structured_response"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "example",
        choices=[
            "auto",
            "tool-pydantic",
            "dataclass",
            "typeddict",
            "union",
            "custom-message",
            "validation-retry",
            "custom-errors",
        ],
    )
    parser.add_argument(
        "--provider",
        choices=["nvidia", "nebius"],
        default="nvidia",
    )
    args = parser.parse_args()

    examples = {
        "auto": example_auto,
        "tool-pydantic": example_tool_pydantic,
        "dataclass": example_dataclass,
        "typeddict": example_typeddict,
        "union": example_union,
        "custom-message": example_custom_message,
        "validation-retry": example_validation_retry,
        "custom-errors": example_custom_errors,
    }

    examples[args.example](args.provider)


if __name__ == "__main__":
    main()
