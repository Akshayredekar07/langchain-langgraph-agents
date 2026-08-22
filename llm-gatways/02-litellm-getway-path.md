# LLM / AI Gateways: Research

## Part 1: What an LLM Gateway Actually Does

An LLM gateway sits between your application and the model providers (OpenAI, Anthropic, Bedrock, Vertex, Groq, and so on). It gives you:

- One endpoint and one API format instead of many SDKs
- Routing, load balancing, and automatic fallback when a provider fails or rate limits you
- Cost tracking and budget limits per team, project, or user
- Caching (exact and semantic) so repeated prompts do not cost twice
- Guardrails: PII filtering, content moderation, prompt injection checks
- Observability: logs, traces, spend dashboards
- Virtual keys and access control instead of sharing raw provider keys

This is infrastructure, not a framework like LangChain. LangChain helps you build the agent logic. A gateway helps you run that logic reliably in production across many models and teams.

---

## Part 2: Top 20 Popular LLM/AI Gateways, by Category

### Category A: Open source, self-hosted (you run the control plane)

1. **LiteLLM** (MIT license) — the most widely adopted open source gateway, over 40,000 GitHub stars, supports 100+ providers, Python SDK plus a proxy server. Core moved to a Rust core with Python SDK in 2026 for speed.
2. **Bifrost** (Apache 2.0, by Maxim AI) — written in Go, positioned as the fastest option with sub microsecond overhead, native MCP support.
3. **Portkey Gateway** — the gateway component became Apache 2.0 in March 2026, the managed platform around it stays proprietary.
4. **LLM Gateway (llmgateway.io)** — fully open source, self-hostable, zero markup pricing.
5. **Envoy AI Gateway** — Apache 2.0, CNCF-aligned, built on Envoy Proxy, Kubernetes native, reached its 1.0 stable release in 2026.
6. **Apache APISIX (ai-proxy plugin)** — an established API gateway with AI plugins added for LLM traffic.

### Category B: Managed aggregators (hosted, you do not run infrastructure)

7. **OpenRouter** — consolidated billing across a large model catalog, strong for prototyping.
8. **Requesty** — managed multi-provider router with cost optimization focus.
9. **Eden AI** — managed aggregator covering LLMs plus other AI APIs (OCR, speech, vision).

### Category C: Smart routers (pick the best or cheapest model per request automatically)

10. **Martian** — routes each request to the model that fits cost and quality targets.
11. **Not Diamond** — similar automatic model selection approach.
12. **Unify AI** — routing based on live benchmarks across providers.

### Category D: Cloud provider native gateways

13. **AWS Bedrock** — less a traditional gateway, more a managed model access layer inside AWS.
14. **Azure API Management (APIM) AI Gateway** — the most feature complete cloud native option: token based rate limits, semantic caching, circuit breakers, a unified model API in preview.
15. **Google Vertex AI Model Garden** — Google's equivalent managed access layer.
16. **Cloudflare AI Gateway** — edge cached proxy, near zero operational overhead, fits naturally if you already run on Cloudflare.

### Category E: API gateway platforms extended for AI traffic

17. **Kong AI Gateway** — extends Kong's existing API management platform with LLM routing and plugins, strongest if you already run Kong.
18. **Zuplo** — API gateway vendor with AI specific policies added.
19. **Vercel AI Gateway** — zero token markup, default provider for the Vercel AI SDK, GA since 2025.

### Category F: Observability platforms that added gateway features

20. **Braintrust Gateway** and **Helicone** — both started as LLM observability/eval platforms and added routing on top. Note: Helicone was acquired by Mintlify in 2026 and is now in maintenance mode rather than active development, worth knowing if you are picking a tool to learn long term.

**Where LiteLLM sits:** it is consistently ranked as either the most popular or the default open source pick across nearly every 2026 comparison. It is the best starting point to learn because it is free, self-hostable, has the largest community, and its concepts (virtual keys, routing, fallback, guardrails) transfer directly to every other gateway on this list.

---

## Part 3: LiteLLM Deep Dive

LiteLLM ships as two things, and this distinction matters for your learning plan:

### 3.1 The SDK (Python library, in-process)
- `litellm.completion()` and `litellm.acompletion()` call 100+ providers through one function signature, same shape as the OpenAI SDK
- `litellm.Router` gives you load balancing, retries, and fallback across deployments without running a separate server
- Good for: scripts, notebooks, small services where you do not want a standalone proxy process

### 3.2 The Proxy (self-hosted server, the real gateway)
- A FastAPI (now Rust-core-backed) server you deploy with Docker
- OpenAI-compatible REST API, so anything built for the OpenAI API works against it unchanged, including LangChain, LlamaIndex, Claude Code, Cursor
- Admin dashboard UI, virtual key management, per-team and per-project budgets
- Guardrails: PII filters, content moderation, prompt injection detection, pluggable custom guardrail classes, and load balancing across multiple guardrail providers
- Caching: exact match and semantic caching, backed by Redis, S3, or GCS
- Cost tracking stored in PostgreSQL, live sync of a model price and context window map so new models are supported day zero
- Enterprise edition (paid) adds SSO/SAML, RBAC, audit logs, per-project budget isolation

### 3.3 Feature comparison snapshot against the rest of the market

| Feature | LiteLLM | Bifrost | Portkey | Cloudflare AI Gateway | Kong AI Gateway |
|---|---|---|---|---|---|
| Self-hosted, open source core | Yes | Yes | Gateway only | No | Depends on Kong deployment |
| License | MIT | Apache 2.0 | Apache 2.0 (gateway) | Proprietary managed | Mixed |
| Provider count | 100+ | Broad | Broad | Multiple | Broad via plugins |
| Virtual keys, budgets | Yes | Yes | Yes | Limited | Yes |
| Guardrails | Yes, pluggable | Yes | Yes | Basic | Yes, plugin based |
| Semantic caching | Yes | Yes | Yes | Basic caching | Depends on plugin |
| MCP support | Yes | Native, strong | Growing | Limited | Growing |
| Best fit | General purpose, largest community | Lowest latency, enterprise governance | Managed platform with enterprise polish | Teams already on Cloudflare | Teams already running Kong |

A fact worth knowing rather than hiding from: LiteLLM has had real security incidents, including a March 2026 supply chain issue where two published versions carried credential harvesting malware. The fix pattern (pin to `-stable` releases, keep the proxy network isolated, rotate keys, verify package integrity before upgrading) is itself a good thing to practice, since production gateway security is a real skill.

---

## Part 4: langchain-litellm — Can You Get LiteLLM's Full Potential Through It?

Short answer: **partially, and it is important to understand exactly where the line sits.**

`langchain-litellm` is an official LangChain integration package (`pip install langchain-litellm`) maintained under the `langchain-ai` GitHub org. It gives you:

- `ChatLiteLLM` — a LangChain chat model wrapper around `litellm.completion()`, so you get LangChain's chain, tool calling, and structured output interface while LiteLLM handles the provider translation underneath
- `ChatLiteLLMRouter` — wraps `litellm.Router`, so you get load balancing and fallback across deployments inside a LangChain chat model
- `LiteLLMEmbeddings` and `LiteLLMEmbeddingsRouter` — same idea for embedding models
- `LiteLLMOCRLoader` — a document loader that calls a LiteLLM **proxy's** OCR endpoint

What this package does **not** give you directly:
- Virtual key management, per-team budgets, SSO, RBAC, audit logs
- Guardrail configuration (PII filters, moderation, prompt injection checks)
- The admin dashboard
- Semantic caching configuration

That is expected and not a gap in the package. Those are **proxy-server** features. They are configured once in the LiteLLM proxy's `config.yaml` and enforced server-side, not per client library. The way you reach the full feature set from LangChain is not through `langchain-litellm` at all. It is through LangChain's plain `ChatOpenAI` class, pointed at your running LiteLLM proxy:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4",
    base_url="http://localhost:4000",   # your LiteLLM proxy
    api_key="sk-litellm-virtual-key",   # a virtual key issued by the proxy
)
```

Because the LiteLLM proxy exposes an OpenAI-compatible API, this single line gets you every governance feature (budgets, guardrails, caching, virtual keys) for free, with zero LiteLLM-specific LangChain code at all.

So the practical answer to your question:

- Use **`langchain-litellm`** (`ChatLiteLLM`, `ChatLiteLLMRouter`) when you want LiteLLM's multi-provider SDK and in-process routing directly inside a LangChain app, without running a standalone proxy server. This is the lighter, embedded path.
- Use **`ChatOpenAI` pointed at the LiteLLM proxy** when you want the full gateway: virtual keys, team budgets, guardrails, semantic caching, dashboard, audit logs. This is the production, governed path.
- The two are not mutually exclusive. Many real setups run the proxy for governance and still use `langchain-litellm`'s router class in application code that talks directly to providers for latency sensitive paths.

This is also a genuinely good thing to demonstrate in your GitHub projects, since it shows you understand gateway architecture at a level past "I called an API."