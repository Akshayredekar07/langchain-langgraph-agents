# LLM / AI Gateways: Learning Plan

Companion to `llm-gateways-research.md`. Read the research doc first for context on what a gateway is, how LiteLLM works, and how `langchain-litellm` fits in.

---

## How This Plan Works

Each stage below adds one gateway capability as its own small project, building on the previous stage. By the end you have one cumulative, production shaped system rather than ten disconnected scripts. This mirrors the "0 to 1, build from scratch" approach you already use in your other repos.

### Stage 0: Foundations (no project, just setup)
- Read how `litellm.completion()` normalizes requests across providers
- Get free or trial keys for two providers (for example OpenAI and Groq, since Groq has a generous free tier)
- Understand the difference between the SDK and the proxy before writing any code

### Stage 1 — Project: Single Interface, Multiple Providers
- Use `litellm.completion()` to call OpenAI, then swap the model string to call Groq or Gemini, same function, no other code changes
- Feature learned: unified API format

### Stage 2 — Project: Fallback Chat CLI
- Build a small command line chatbot that tries a primary model and automatically falls back to a secondary model on error or rate limit, using `litellm.completion()`'s built in fallback list
- Feature learned: automatic failover

### Stage 3 — Project: Load Balanced Router
- Use `litellm.Router` with multiple deployments of the same model (for example two API keys or two regions) and watch requests distribute across them
- Feature learned: load balancing strategies (least busy, round robin, lowest cost)

### Stage 4 — Project: Cost Tracker
- Log the cost of every call using LiteLLM's built in cost calculation, write it to a local SQLite table, build a tiny script that reports spend by model
- Feature learned: cost tracking at the SDK level, before touching the proxy

### Stage 5 — Project: Deploy the LiteLLM Proxy
- Run the LiteLLM proxy with Docker, write a `config.yaml` with two or three models
- Create your first virtual key from the admin UI, call the proxy from a plain `curl` request and from Python using the OpenAI SDK pointed at `http://localhost:4000`
- Feature learned: the proxy is a real, separate service, and anything OpenAI-compatible works against it unmodified

### Stage 6 — Project: Team Budgets and Access Control
- Add two virtual keys with different per-key budgets and rate limits on the proxy
- Write a small script that intentionally exceeds one budget and shows the proxy blocking the request
- Feature learned: multi-tenant governance

### Stage 7 — Project: Guardrails
- Add a PII filter guardrail and a content moderation guardrail to the proxy config
- Send a request containing a fake PII string and confirm it is caught before reaching the model
- Feature learned: pre-call and post-call guardrail hooks, including writing one custom guardrail class yourself

### Stage 8 — Project: Caching Layer
- Stand up Redis, wire it into the proxy config for exact-match caching, then enable semantic caching
- Send the same question phrased two different ways and confirm the second call is served from cache
- Feature learned: cost reduction through caching, and the tradeoffs of semantic cache thresholds

### Stage 9 — Project: Observability
- Connect the proxy to Langfuse (a tool you already use) so every request, cost, and latency number shows up in a trace dashboard
- Feature learned: production observability, which also strengthens your existing LangFuse experience for your resume

### Stage 10 — Project: LangChain Integration, Both Paths
- Part A: use `ChatLiteLLMRouter` from `langchain-litellm` directly in a LangChain agent, no proxy involved
- Part B: point LangChain's `ChatOpenAI` at your Stage 5 to 9 proxy and rebuild the same agent
- Compare the two in a short write-up: what each path gives you and what it does not
- Feature learned: exactly the distinction covered in the research doc, now proven with working code instead of just reading about it

---

## Capstone Project — Gateway-Backed Agentic RAG System

Combine everything into one repository that extends your existing Agentic RAG project:

- LangChain (or LangGraph) agent with multi-turn memory, same as your current project
- All model calls routed through your LiteLLM proxy, not called directly
- Per-team virtual keys (for example one key for "retrieval" calls, one for "generation" calls) with separate budgets
- Guardrails on both directions: input PII filtering, output content checks
- Semantic caching enabled for repeated queries
- Full request tracing in Langfuse
- Fallback chain: primary model to a cheaper backup model on failure
- Deployed with Docker Compose (proxy plus your app plus Redis plus Postgres), documented with a professional README, architecture diagram, and a short section explaining the langchain-litellm versus proxy-via-ChatOpenAI design decision you made and why

This capstone is the kind of project that reads as "engineer who builds production infrastructure," not "engineer who called an API," which fits directly into the GitHub positioning gap you have been working on closing.

---

## Suggested Order of Study

1. Stages 0 to 4: one to two days, all local, no Docker needed yet
2. Stages 5 to 9: three to five days, this is where the real gateway skill is built
3. Stage 10: half a day, mostly a comparison exercise
4. Capstone: treat it as its own repository with its own README, not a folder inside an existing repo

If you want, I can start with Stage 1 code right now, or set up the Stage 5 Docker Compose file and `config.yaml` so you have a working proxy to experiment with today.