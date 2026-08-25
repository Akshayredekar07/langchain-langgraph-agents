# LiteLLM — Overview

> Source-grounded notes (web search: docs.litellm.ai, litellm.ai, dev.to, medium, deepwiki, mintlify). No LLM-proxy guessing. Last updated 2026-08-25.

## 0.1 What is LiteLLM?

LiteLLM is an **open-source library + self-hosted proxy server**, maintained by BerriAI, that exposes **100+ LLM providers behind a single OpenAI-compatible API**. It exists to solve the N-SDKs problem: every provider ships a different SDK, auth flow, error format, and streaming shape. LiteLLM normalizes them all.

The project ships as **two independent products** that share the same Python package:

| Product | What it is | When to use |
|---|---|---|
| **Python SDK** (`litellm.completion(...)`) | An in-process library. You import it, call it, it talks to providers. | One developer / one app. No infra. |
| **Proxy Server** (this is what these notes cover) | A FastAPI HTTP server on port 4000 that speaks the OpenAI API. | Multi-user, multi-team, multi-provider, cost tracking, key management, rate limits, observability. |

The proxy is the **production deployment path**. The SDK is the dev path. They are not mutually exclusive.

## 0.2 What is the LiteLLM Proxy specifically?

The LiteLLM Proxy is an **OpenAI-compatible gateway**:

- Accepts requests on the same shape as the OpenAI API (`/v1/chat/completions`, `/v1/embeddings`, `/v1/responses`, `/v1/images`, `/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/rerank`, `/v1/batches`).
- Routes to any of 100+ backends based on a YAML config.
- Returns responses in the OpenAI shape regardless of which provider answered.
- Layers on enterprise features: auth, virtual keys, spend tracking, retries, fallbacks, load balancing, caching, guardrails, OTel tracing.

Any client that already speaks OpenAI (LangChain, LlamaIndex, Cursor, Open WebUI, raw `openai-python`, `openai-node`, `curl`) points at the proxy base URL with **zero code changes**.

## 0.3 Supported endpoints (out of the box)

```
/v1/chat/completions        # most LLM calls
/v1/completions             # legacy text completions
/v1/embeddings
/v1/rerank
/v1/images
/v1/images/edits
/v1/audio/speech
/v1/audio/transcriptions
/v1/responses               # OpenAI Responses API bridge
/v1/batches
/chat/completions           # unversioned alias (also accepted)
/completions
/embeddings
/models                     # GET — list of available model aliases
/health/liveliness
/health/readiness
/key/*                      # key management API (admin)
/user/*                     # user management (admin)
/team/*                     # team management (admin)
/mcp/*                      # MCP gateway endpoints
/a2a/*                      # A2A gateway endpoints
/ui                         # admin dashboard
```

## 0.4 Why people reach for it (the actual pain points it solves)

1. **One client, many providers.** Write code against OpenAI's API; swap model names to switch between OpenAI, Anthropic, Bedrock, Vertex, Azure, Gemini, Mistral, Groq, Cohere, Together, Ollama, vLLM, HuggingFace, NVIDIA NIM, etc.
2. **No provider lock-in.** A `config.yaml` change deploys a new provider to all clients.
3. **Cost & usage visibility.** Every call is metered per key / user / team / model with token counts and dollar cost.
4. **Failover.** Primary provider goes down → automatic retry on a fallback model with no client change.
5. **Load balancing.** Multiple API keys / deployments of the same model share traffic.
6. **Rate limiting & budgets.** Virtual keys with TPM/RPM caps and dollar budgets that auto-revoke.
7. **Single login for the org.** SSO, RBAC, audit log.
8. **Caching.** Identical prompts skipped on second hit → real cost savings.
9. **Observability.** One callback config fans out to Langfuse / Datadog / OTel / S3 / etc.
10. **Day-zero model support.** New models go live the same day the provider ships them.

## 0.5 Performance numbers (from the maintainer + community load tests)

- **1,500+ requests/second** sustained on a single proxy.
- **P95 ~8 ms** at 1,000 RPS (router overhead only).
- **Gateway adds ~4 ms** of overhead even for sub-50ms latency-sensitive paths.

So the proxy is fast enough to sit in front of interactive workloads.

## 0.6 Mental model

```
your app
   │
   │  speaks OpenAI API
   ▼
┌────────────────────────────┐
│      LiteLLM Proxy         │  ← port 4000
│  - auth (virtual keys)     │
│  - rate limits / budgets   │
│  - cache (Redis)           │
│  - router / load balancer  │
│  - guardrails              │
│  - callbacks (logging)     │
└────────────────────────────┘
   │            │            │
   ▼            ▼            ▼
 OpenAI     Anthropic    Bedrock / Vertex / ...
```

A useful framing (per Gary Stafford): **the SDK is a client, the proxy is a platform.** The SDK makes one developer's life easier. The proxy makes an entire organization's LLM usage governable.

## 0.7 When *not* to use the proxy

- You're a single developer with one provider key and no cost concerns. Just use the SDK or the provider's SDK directly.
- You need a fully managed solution with zero ops. Look at Portkey (managed), Cloudflare AI Gateway, or the hosted LiteLLM enterprise tier.
- Your traffic is so extreme that you can't tolerate even 4ms overhead — give that one service a direct connection.

## 0.8 Editions

| Edition | License | Notes |
|---|---|---|
| **Open source** | MIT (core) | Proxy server, SDK, virtual keys, caching, callbacks, RBAC, JWT auth. Most of what you need. |
| **Enterprise** | Paid | SSO / SCIM, audit, multi-region, managed hosting, team-scoped RBAC, guaranteed SLAs. |

The proxy itself runs fine for a team of 50+ on the open-source edition. You only need enterprise when your compliance team says so.

## 0.9 TL;DR

> A single FastAPI server that takes OpenAI-shaped requests and fans them out to 100+ LLM providers, with auth, budgets, caching, routing, retries, fallbacks, and observability — all configured in one `config.yaml` and managed through a web UI or REST API. Production-grade, MIT-licensed, ~1.5k RPS, ~4ms overhead.

---

Next: [01-quickstart.md](./01-quickstart.md) — install and run it.
