# **LangChain Structured Output Notes**

## **1. What structured output solves**

1. Without it, an agent's final answer is free-form text — you'd have to regex or re-prompt to pull fields out of it.
2. With `response_format` set on `create_agent`, the model's answer is captured, validated against a schema, and placed in `result["structured_response"]` as a Pydantic instance, dict, or dataclass — not natural language you have to parse.
3. This only applies to `create_agent` (the v1 agent factory). Using structured output on a bare chat model (no agent, no tools) is a separate, narrower mechanism (`model.with_structured_output` / model-level `response_format`) not covered here.
4. Four possible values for `response_format`:
   - `ToolStrategy[Schema]` — force structured output via a tool call
   - `ProviderStrategy[Schema]` — use the provider's native structured-output API
   - `Schema` (bare type) — LangChain auto-picks one of the above
   - `None` — no structured output requested (default)

## **2. Schema types accepted**

1. Both strategies accept the same four schema shapes:
   - Pydantic `BaseModel` subclass → returns a validated Pydantic instance
   - Python `@dataclass` → returns a dict
   - `TypedDict` → returns a dict
   - Raw JSON Schema dict (must include top-level `title` and `description`) → returns a dict
2. JSON Schema dicts are never auto-detected — they must always be wrapped explicitly in `ToolStrategy(...)` or `ProviderStrategy(...)`, even when passed to `response_format`.
3. `ToolStrategy` additionally accepts a `Union` of schemas — the model picks whichever fits the conversation.

```python
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_nebius import ChatNebius

load_dotenv()

class MeetingNote(BaseModel):
    """Structured summary of a meeting snippet."""
    topic: str = Field(description="What the meeting segment was about")
    owner: str = Field(description="Person responsible for the follow-up")
    urgency: str = Field(description="One of: low, medium, high")

model = ChatNebius(model="Qwen/Qwen3-32B", temperature=0.2)

agent = create_agent(
    model=model,
    tools=[],
    response_format=ToolStrategy(MeetingNote),
)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Priya needs to redo the onboarding deck before Friday, this is urgent."
    }]
})

print(result["structured_response"])
print(type(result["structured_response"]))
```

## **3. Auto-selection behavior (bare schema type)**

1. Passing the schema type directly — `response_format=MeetingNote` — lets LangChain choose the strategy for you.
2. Selection rule:
   - `ProviderStrategy` if the model/provider pair supports native structured output (confirmed for OpenAI, Anthropic, xAI at doc time; provider support keeps expanding — see section 8)
   - `ToolStrategy` for everything else
3. Whether a given model supports native structured output is read from the model's *profile data* (`langchain>=1.1`). If profile data isn't available for a custom or less common integration, either:
   - pass an explicit `custom_profile` dict with `"structured_output": True` into `init_chat_model`, or
   - skip auto-detection entirely and wrap the schema in `ToolStrategy` yourself.
4. If `tools` are also passed to the agent, the model must support using tools and structured output at the same time — not every provider does, and this has been an active bug area (see section 8).

```python
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

class Ticket(BaseModel):
    """Support ticket extracted from a user message."""
    category: str = Field(description="bug, question, or feature_request")
    summary: str = Field(description="One-line summary of the issue")

model = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", temperature=0.1)

agent = create_agent(
    model=model,
    tools=[],
    response_format=Ticket,
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "The dark mode toggle resets every time I refresh the page."}]
})

print(result["structured_response"])
```

## **4. Provider strategy**

1. `ProviderStrategy` routes the schema through the provider's native structured-output API (e.g. OpenAI's `response_format`, Anthropic's tool-forced JSON). This is the most reliable path when the provider supports it — the provider itself enforces the schema, not LangChain's retry logic.
2. Signature:
   ```python
   class ProviderStrategy(Generic[SchemaT]):
       schema: type[SchemaT]
       strict: bool | None = None
   ```
3. `strict=True` requests stricter schema adherence. Only honored by providers that expose a strict mode (OpenAI, xAI at doc time). Requires `langchain>=1.2`.
4. If the provider does not actually support native structured output, the agent silently falls back to `ToolStrategy` rather than erroring.

```python
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_nebius import ChatNebius

load_dotenv()

class Invoice(BaseModel):
    """Invoice fields extracted from freeform text."""
    vendor: str = Field(description="Name of the vendor")
    amount: float = Field(description="Total amount due")
    due_date: str = Field(description="Due date in YYYY-MM-DD")

model = ChatNebius(model="Qwen/Qwen3-32B", temperature=0)

agent = create_agent(
    model=model,
    tools=[],
    response_format=ProviderStrategy(Invoice, strict=True),
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Invoice from Acme Supplies, $482.50, due 2026-09-30."}]
})

print(result["structured_response"])
```

## **5. Tool calling strategy**

1. `ToolStrategy` is the fallback path for models without native structured-output support. It works by exposing the schema as a callable tool the model is expected to invoke exactly once with the extracted fields.
2. Signature:
   ```python
   class ToolStrategy(Generic[SchemaT]):
       schema: type[SchemaT]
       tool_message_content: str | None
       handle_errors: Union[bool, str, type[Exception], tuple[type[Exception], ...], Callable[[Exception], str]]
   ```
3. This works with essentially any tool-calling-capable model — it is the safe default for providers like Nebius and NVIDIA NIM where native structured output support is not guaranteed.
4. Because it uses the normal tool-calling machinery, it composes with real tools in the same agent without special-casing.

```python
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

class ReviewAnalysis(BaseModel):
    """Analysis of a product review."""
    rating: int = Field(description="Rating out of 5", ge=1, le=5)
    sentiment: Literal["positive", "negative", "neutral"]
    key_points: list[str] = Field(description="1-3 word phrases summarizing the review")

model = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", temperature=0.3)

agent = create_agent(
    model=model,
    tools=[],
    response_format=ToolStrategy(ReviewAnalysis),
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Solid build quality, but the battery life is disappointing. 3/5."}]
})

print(result["structured_response"])
```

## **6. Custom tool message content**

1. By default, when `ToolStrategy` captures a result, it inserts a `ToolMessage` into the conversation reading `"Returning structured response: {...}"`.
2. `tool_message_content` overrides that text with a fixed string of your choosing — useful when the raw dict dump shouldn't leak into a user-facing transcript.
3. This only changes what's shown in message history; it does not change what ends up in `result["structured_response"]`.

```python
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_nebius import ChatNebius

load_dotenv()

class ActionItem(BaseModel):
    """Action item extracted from a meeting transcript."""
    task: str = Field(description="The task to complete")
    assignee: str = Field(description="Person responsible")
    priority: Literal["low", "medium", "high"]

model = ChatNebius(model="Qwen/Qwen3-32B", temperature=0.2)

agent = create_agent(
    model=model,
    tools=[],
    response_format=ToolStrategy(
        schema=ActionItem,
        tool_message_content="Action item logged.",
    ),
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Arjun should fix the CI pipeline before end of day, it's blocking everyone."}]
})

for msg in result["messages"]:
    if type(msg).__name__ == "ToolMessage":
        print(msg.content)

print(result["structured_response"])
```

## **7. Error handling on structured output**

1. Two failure modes are handled automatically when `handle_errors` allows a retry:
   - **Multiple structured outputs**: the model calls more than one structured-output tool in a single turn — each gets an error `ToolMessage`, and the model is asked to pick one.
   - **Schema validation failure**: the model's tool call doesn't satisfy the schema (e.g. a rating of 10 on a 1–5 scale) — the validation error is fed back and the model retries.
2. `handle_errors` controls this behavior:
   - `True` (default) — catch all errors, default message, retry
   - `str` — catch all errors, always use this custom message
   - `type[Exception]` — only retry on this exception type, others propagate
   - `tuple[type[Exception], ...]` — only retry on these types
   - `Callable[[Exception], str]` — custom function returning the retry message, can branch on `StructuredOutputValidationError` vs `MultipleStructuredOutputsError`
   - `False` — no retry, exception raised immediately

```python
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import (
    ToolStrategy,
    StructuredOutputValidationError,
    MultipleStructuredOutputsError,
)
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

class ProductRating(BaseModel):
    """A validated product rating."""
    rating: int = Field(description="Rating from 1-5", ge=1, le=5)
    comment: str = Field(description="Review comment")

def handle_error(error: Exception) -> str:
    if isinstance(error, StructuredOutputValidationError):
        return "That value is out of range. Use a rating between 1 and 5."
    if isinstance(error, MultipleStructuredOutputsError):
        return "Only one structured response is allowed. Pick the best match."
    return f"Error: {error}"

model = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", temperature=0)

agent = create_agent(
    model=model,
    tools=[],
    response_format=ToolStrategy(schema=ProductRating, handle_errors=handle_error),
    system_prompt="Parse product reviews. Never invent a rating that wasn't stated.",
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Amazing product, 10/10 would buy again!"}]
})

print(result["structured_response"])
```

## **8. Decision guide**

1. Model is OpenAI, Anthropic, or xAI and doesn't need tools alongside structured output → pass the bare schema type, let auto-selection use `ProviderStrategy`.
2. Model is a smaller or self-hosted provider (Nebius, NVIDIA NIM, most open-weight endpoints) → use `ToolStrategy` explicitly; don't rely on auto-detected native support.
3. Need tools and structured output in the same agent call → check the specific model/provider combination first — this has been a known break point (multiple GitHub issues, section 9) — and default to `ToolStrategy` unless you've confirmed the provider handles both simultaneously.
4. Extraction target could plausibly be one of several shapes (e.g. either a contact or an event) → `ToolStrategy(Union[SchemaA, SchemaB])`.
5. Need the strictest possible schema enforcement and the provider supports it → `ProviderStrategy(schema, strict=True)`, on `langchain>=1.2`.
6. Want retries to fail loudly during development instead of silently correcting → `handle_errors=False`, switch back to `True` or a custom handler once schema and prompt are stable.

## **9. Things to verify before relying on this**

1. Native provider structured-output support for `langchain_nebius.ChatNebius` and `langchain_nvidia_ai_endpoints.ChatNVIDIA` is not confirmed — check each package's model profile / changelog before assuming `ProviderStrategy` will auto-select for them; `ToolStrategy` is the safer default until confirmed.
2. `strict=True` on `ProviderStrategy` requires `langchain>=1.2` — confirm installed version with `pip show langchain` before using it.
3. Combining real tools with `response_format` in the same `create_agent` call has open bugs against specific providers (e.g. Gemini 3 forced into `ToolStrategy` unnecessarily, OpenAI `strict` flag not propagating through `ProviderStrategy`) — test this combination against your actual model before trusting it in production.
4. Model "profile data" auto-detection depends on `langchain>=1.1` and on the specific integration package shipping profile metadata — some community/self-hosted integrations may not have this populated, silently causing incorrect strategy selection.
5. `create_react_agent` (the older prebuilt) still appears in some docs and community posts for structured output — it is not the v1 API and should not be used; `create_agent` from `langchain.agents` is correct.
6. Exact class paths (`langchain.agents.structured_output.ToolStrategy`, `.ProviderStrategy`, `.StructuredOutputValidationError`, `.MultipleStructuredOutputsError`) should be re-checked against your installed version — these are reference-doc paths, not confirmed against a live import in this session.