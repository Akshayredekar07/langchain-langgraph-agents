# Multi-Platform Content Pipeline


import os
import operator

from dotenv import load_dotenv
from langchain_nebius import ChatNebius
from typing_extensions import TypedDict
from typing import Literal, List, Annotated
from pydantic import BaseModel, Field, SecretStr
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, AIMessage
from langgraph.types import Send

load_dotenv()


# ── structured output schemas ─────────────────────────────────────────
class TopicClassification(BaseModel):
    content_goal: Literal["promotional", "educational", "announcement"] = Field(
        description="The overall goal of the content"
    )
    needs_research: bool = Field(
        description="True if the topic needs a few factual talking points gathered before writing"
    )


class DraftEvaluation(BaseModel):
    grade: Literal["approved", "needs_revision"] = Field(description="Is the draft good enough to publish")
    feedback: str = Field(description="What to improve if not approved")


# ── graph state ────────────────────────────────────────────────────────
# total=False makes every key optional, since fields fill in gradually as
# nodes run, and nodes only ever return the fields they changed.


class ContentState(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], add_messages]
    content_goal: str
    needs_research: bool
    research_notes: str
    platforms: List[str]
    platform_drafts: Annotated[List[dict], operator.add]
    final_posts: dict


class PlatformState(TypedDict, total=False):
    platform: str
    post_topic: str
    post_goal: str
    post_research_notes: str
    draft: str
    feedback: str
    grade: str
    iteration: int
    max_iteration: int
    platform_drafts: Annotated[List[dict], operator.add]



# ── model setup ────────────────────────────────────────────────────────
model = ChatNebius(
    model="MiniMaxAI/MiniMax-M2.5",
    api_key=SecretStr(os.environ.get("NEBIUS_API_KEY", "")), 
)

classifier_model = model.with_structured_output(TopicClassification)
evaluator_model = model.with_structured_output(DraftEvaluation)

STYLE_GUIDE = {
    "twitter": "Keep it under 280 characters, punchy, at most 2 hashtags.",
    "linkedin": "Professional tone, 3 to 5 sentences, minimal hashtags.",
    "instagram": "Casual and visual tone, can use emojis and a few hashtags.",
}




# Top-Level Graph Nodes


# ── classify the topic (goal + whether research is needed) ────────────
def classify_topic(state: ContentState) -> ContentState:
    topic = state.get("messages", [])[-1].content

    prompt = f"""Classify this social media topic: "{topic}"

    Decide the content goal (promotional, educational, or announcement) and whether
    it needs a few factual talking points researched before writing posts about it."""

    result = classifier_model.invoke(prompt)
    return {
        "content_goal": result["content_goal"],  
        "needs_research": result["needs_research"],  
    }


def route_after_classify(state: ContentState) -> Literal["research", "skip_research"]:
    return "research" if state.get("needs_research") else "skip_research"




# ── gather quick factual talking points ────────────────────────────────
def research_topic(state: ContentState) -> ContentState:
    topic = state.get("messages", [])[-1].content

    prompt = f"Give 3 to 4 short factual talking points about: {topic}"
    notes = model.invoke(prompt).content

    return {"research_notes": str(notes)} 



# ── decide which platforms to generate posts for ──────────────────────
def plan_platforms(state: ContentState) -> ContentState:
    _ = state
    return {"platforms": ["twitter", "linkedin", "instagram"]}


def assign_platform_workers(state: ContentState) -> List[Send]:
    topic = state.get("messages", [])[-1].content

    return [
        Send(
            "platform_pipeline",
            {
                "platform": platform,
                "post_topic": topic,
                "post_goal": state.get("content_goal", ""),
                "post_research_notes": state.get("research_notes", ""),
                "draft": "",
                "feedback": "",
                "grade": "",
                "iteration": 0,
                "max_iteration": 2,
            },
        )
        for platform in state.get("platforms", [])
    ]




# ── merge all platform drafts into the final reply ─────────────────────
def combine_results(state: ContentState) -> ContentState:
    platform_drafts = state.get("platform_drafts", [])
    final_posts = {item["platform"]: item["content"] for item in platform_drafts}

    lines = [f"**{platform.capitalize()}**\n{content}" for platform, content in final_posts.items()]
    reply = "\n\n".join(lines) if lines else "No posts were generated."

    return {"final_posts": final_posts, "messages": [AIMessage(content=reply)]}





# Per-Platform Subgraph Nodes


# ── write (or rewrite, if feedback exists) a platform-specific draft ──
def generate_draft(state: PlatformState) -> PlatformState:
    platform = state.get("platform", "")
    topic = state.get("post_topic", "")
    feedback = state.get("feedback", "")
    guide = STYLE_GUIDE.get(platform, "")
    research_notes = state.get("post_research_notes", "none")
    post_goal = state.get("post_goal", "")

    if feedback:
        prompt = f"""Rewrite this {platform} post about {topic}.
Feedback to address: {feedback}
Style guide: {guide}
Research notes: {research_notes}"""
    else:
        prompt = f"""Write a {platform} post about {topic}.
Content goal: {post_goal}
Style guide: {guide}
Research notes: {research_notes}"""

    draft = str(model.invoke(prompt).content)
    return {"draft": draft, "iteration": state.get("iteration", 0) + 1}  




# ── grade the draft and produce feedback if it needs work ─────────────
def evaluate_draft(state: PlatformState) -> PlatformState:
    platform = state.get("platform", "")
    draft = state.get("draft", "")
    prompt = f"Evaluate this {platform} post for quality and fit for the platform:\n\n{draft}"
    result = evaluator_model.invoke(prompt)

    return {"grade": result.get("grade", ""), "feedback": result.get("feedback", "")}



def route_after_evaluate(state: PlatformState) -> Literal["approved", "retry"]:
    if state.get("grade") == "approved":
        return "approved"
    if state.get("iteration", 0) >= state.get("max_iteration", 0):
        return "approved"
    return "retry"



# ── record the approved draft for this platform ────────────────────────
def record_result(state: PlatformState) -> PlatformState:
    return {
        "platform_drafts": [
            {
                "platform": state.get("platform", ""),
                "content": state.get("draft", ""),
                "iterations": state.get("iteration", 0),
            }
        ]
    }





# Build the Per-Platform Subgraph


def build_platform_pipeline():
    pipeline = StateGraph(PlatformState)

    pipeline.add_node("generate_draft", generate_draft)
    pipeline.add_node("evaluate_draft", evaluate_draft)
    pipeline.add_node("record_result", record_result)

    pipeline.add_edge(START, "generate_draft")
    pipeline.add_edge("generate_draft", "evaluate_draft")
    pipeline.add_conditional_edges(
        "evaluate_draft",
        route_after_evaluate,
        {"approved": "record_result", "retry": "generate_draft"},
    )
    pipeline.add_edge("record_result", END)

    return pipeline.compile()


platform_pipeline = build_platform_pipeline()





# Build the Top-Level Graph


graph = StateGraph(ContentState)

graph.add_node("classify_topic", classify_topic)
graph.add_node("research_topic", research_topic)
graph.add_node("plan_platforms", plan_platforms)
graph.add_node("platform_pipeline", platform_pipeline)
graph.add_node("combine_results", combine_results)

graph.add_edge(START, "classify_topic")
graph.add_conditional_edges(
    "classify_topic",
    route_after_classify,
    {"research": "research_topic", "skip_research": "plan_platforms"},
)
graph.add_edge("research_topic", "plan_platforms")
graph.add_conditional_edges("plan_platforms", assign_platform_workers, ["platform_pipeline"])
graph.add_edge("platform_pipeline", "combine_results")
graph.add_edge("combine_results", END)

workflow = graph.compile()