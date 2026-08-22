<h1><strong>LLM and AI Gateways: Introduction and Research</strong></h1>

<h1><strong>1. What an LLM gateway actually is</strong></h1>

<strong>Short definition</strong>

An LLM gateway is a piece of infrastructure that sits between your application and the model providers (OpenAI, Anthropic, AWS Bedrock, Google Vertex, Groq, Mistral, and so on). It speaks the OpenAI API format on one side and translates to every provider on the other.

<strong>Why it exists</strong>

Every provider has a different SDK, a different auth scheme, a different streaming protocol, different rate limits, and a different failure mode. If you call all of them directly from your application, you write the same glue five times and you still have no observability, no budgets, no failover, and no guardrails.

A gateway gives you one endpoint, one key format, one place to enforce policy, and one place to look when something goes wrong.

<strong>How it works in one sentence</strong>

Your app sends a normal OpenAI format request to the gateway. The gateway checks auth and budget, looks up cache, picks a target provider, translates the request, forwards it, translates the response back, logs the spend, and returns the result.

<h1><strong>2. The five core jobs</strong></h1>

Every gateway does some subset of these five things. If a tool does none of them, it is not a gateway.

  1. <strong>Normalize</strong> request and response formats across providers
  2. <strong>Route</strong> each request, by cost, latency, quality, region, or team
  3. <strong>Enforce policy</strong>, including rate limits, budgets, virtual keys, and role based access control
  4. <strong>Cache</strong> responses, both exact match and semantic similarity
  5. <strong>Emit telemetry</strong>, meaning logs, traces, cost attribution, and dashboards

When you evaluate a gateway, ask which of these five it does well, which it does poorly, and which it does not do at all.

<h1><strong>3. Gateway versus framework</strong></h1>

A common confusion is mixing up an LLM gateway with LangChain, LlamaIndex, Haystack, or the Vercel AI SDK. They live in different layers.

<table>
<tr><th>Layer</th><th>What it does</th><th>Examples</th></tr>
<tr><td>Agent framework</td><td>Helps you write the agent logic: chains, tools, memory, retrieval, prompts</td><td>LangChain, LlamaIndex, Haystack, CrewAI, AutoGen</td></tr>
<tr><td>LLM gateway</td><td>Helps you run that logic reliably across many models and teams</td><td>LiteLLM, Bifrost, Portkey, OpenRouter, Kong AI</td></tr>
<tr><td>Model provider</td><td>The actual model API</td><td>OpenAI, Anthropic, Google, Mistral, AWS Bedrock</td></tr>
<tr><td>Observability and eval</td><td>Records, replays, scores, and improves runs</td><td>Langfuse, Helicone, Braintrust, Arize</td></tr>
</table>

The short version: <strong>frameworks call the model. Gateways route the call.</strong>

A real production stack uses all four layers. A framework composes the agent. The agent calls a model through a gateway. An observability tool records what happened.

<h1><strong>4. Mental model: the airport</strong></h1>

Think of an LLM gateway as an airport.

  * Your application is a passenger with a ticket
  * The airport checks your ID (auth), your boarding pass (virtual key), and your luggage weight (budget)
  * It routes you to the right gate (provider), sometimes the closest runway, sometimes the cheapest airline, sometimes a backup if the first is closed
  * It logs your flight for billing and analytics
  * It enforces security screening (guardrails)

Airlines specialize in the actual flying. The airport is the boring but critical infrastructure that makes the system work safely at scale.

<h1><strong>5. The 20 gateway landscape</strong></h1>

The 2026 market has split into 6 categories. Pick by category first, then by name.

<strong>Category A: open source, self hosted</strong>

You run the control plane. No vendor reads your prompts.

<table>
<tr><th>Name</th><th>License</th><th>Standout</th><th>Pick when</th></tr>
<tr><td>LiteLLM</td><td>MIT</td><td>Largest community, 100 plus providers, OpenAI compatible</td><td>Default. Prototyping, internal tools, under 500 RPS</td></tr>
<tr><td>Bifrost</td><td>Apache 2.0</td><td>About 11 microseconds overhead at 5000 RPS, written in Go</td><td>High throughput agent platforms</td></tr>
<tr><td>Portkey Gateway</td><td>Apache 2.0 core</td><td>1,600 plus models, deep guardrails</td><td>Regulated workloads, PII, HIPAA, audit</td></tr>
<tr><td>LLM Gateway</td><td>Open source</td><td>210 to 300 plus models, zero markup</td><td>Cost sensitive, no frills</td></tr>
<tr><td>Envoy AI Gateway</td><td>Apache 2.0</td><td>Reuses existing Envoy Proxy, Kubernetes native</td><td>Platform team already runs Envoy</td></tr>
<tr><td>Apache APISIX</td><td>Apache 2.0</td><td>AI plugin on existing API gateway</td><td>Platform team already runs APISIX</td></tr>
</table>

<strong>Category B: managed aggregators</strong>

A SaaS company runs the gateway. You sign up, get a key, send requests.

<table>
<tr><th>Name</th><th>Standout</th><th>Pick when</th></tr>
<tr><td>OpenRouter</td><td>300 plus models, 5 minute setup</td><td>Prototyping, zero infra, broad model catalog</td></tr>
<tr><td>Requesty</td><td>8 ms P50, 5 percent markup, agentic mode</td><td>Production routing with caching</td></tr>
<tr><td>Eden AI</td><td>Covers OCR, speech, vision, translation</td><td>Multi modal AI pipelines</td></tr>
</table>

<strong>Category C: smart routers</strong>

These tools pick the right model for each request automatically, based on cost and quality tradeoffs.

<table>
<tr><th>Name</th><th>Standout</th><th>Pick when</th></tr>
<tr><td>Martian</td><td>Learned cost and quality router</td><td>Trust an auto router to balance cost and quality</td></tr>
<tr><td>Not Diamond</td><td>Auto model selection</td><td>Alternative smart router to compare</td></tr>
<tr><td>Unify AI</td><td>Live benchmark driven routing</td><td>Quality is the top priority</td></tr>
</table>

<strong>Category D: cloud provider native</strong>

Built into the cloud platform. You get them free if you already use that cloud.

<table>
<tr><th>Name</th><th>Standout</th><th>Pick when</th></tr>
<tr><td>AWS Bedrock</td><td>IAM integrated, multi model</td><td>AWS only workloads</td></tr>
<tr><td>Azure APIM AI Gateway</td><td>Token rate limits, semantic cache</td><td>Azure only workloads</td></tr>
<tr><td>Google Vertex Model Garden</td><td>GCP native</td><td>GCP only workloads</td></tr>
<tr><td>Cloudflare AI Gateway</td><td>Edge cached, free tier, near zero ops</td><td>Already on Cloudflare Workers</td></tr>
</table>

<strong>Category E: API gateway platforms with AI plugins</strong>

Traditional API management tools that added AI plugins.

<table>
<tr><th>Name</th><th>Standout</th><th>Pick when</th></tr>
<tr><td>Kong AI Gateway</td><td>Reuses existing Kong, plugin based</td><td>Already on Kong</td></tr>
<tr><td>Zuplo</td><td>Lightweight API management with AI</td><td>Already on Zuplo</td></tr>
<tr><td>Vercel AI Gateway</td><td>Zero markup, Vercel native</td><td>Next.js and Vercel stack</td></tr>
</table>

<strong>Category F: observability platforms that added gateway features</strong>

Started as logging and metrics platforms, then bolted on routing.

<table>
<tr><th>Name</th><th>Standout</th><th>Pick when</th></tr>
<tr><td>Braintrust Gateway</td><td>Eval first, gateway added</td><td>Already using Braintrust evals</td></tr>
<tr><td>Helicone</td><td>Rust gateway, about 5 ms overhead</td><td>Observability first teams (note: acquired by Mintlify in 2026, now in maintenance mode)</td></tr>
</table>

<h1><strong>6. Decision tree: how to pick</strong></h1>

Ask these four questions in order.

  1. Do you need to self host, or is managed fine?
  2. Will your traffic exceed 500 RPS sustained?
  3. Do you need compliance guardrails (PII, jailbreak, audit)?
  4. Do you need automatic model selection per request?

<strong>Quick collapse</strong>

  * <strong>Self host plus free plus huge community</strong>: LiteLLM
  * <strong>Managed plus zero ops plus every model</strong>: OpenRouter for prototyping, Portkey Cloud for production
  * <strong>High throughput at 5,000 plus RPS</strong>: Bifrost (Go) or LiteLLM Rust core when GA
  * <strong>Regulated workloads</strong>: Portkey (managed) or TrueFoundry (VPC)
  * <strong>Automatic model selection</strong>: Martian, Not Diamond, or Unify

If you do not know which to pick, default to LiteLLM self hosted (or OpenRouter for prototyping). You can migrate later without rewriting your application code, because they all expose the same OpenAI compatible API.

<h1><strong>7. Why LiteLLM is the default starting point</strong></h1>

LiteLLM is not the fastest gateway. It is not the most feature complete. It won for three reasons, in order.

  1. <strong>Provider breadth</strong>. 100 plus providers, day zero support for new models, live price and context window map
  2. <strong>OpenAI compatibility</strong>. Every tool that already speaks OpenAI (LangChain, LlamaIndex, Claude Code, Cursor, Vercel AI SDK) works against LiteLLM unchanged by changing one URL
  3. <strong>Community</strong>. About 55,000 GitHub stars, the most discussions, the most Stack Overflow answers, the most blog posts

The 2026 rewrite of the core from Python to Rust (Python SDK kept) is a bet that LiteLLM can keep its community position while closing the latency gap with Go based competitors like Bifrost.

<h1><strong>8. LiteLLM in two pieces</strong></h1>

LiteLLM ships as two things, and the distinction matters.

<strong>SDK: Python library, in process</strong>

  * `litellm.completion()` and `litellm.acompletion()` call 100 plus providers through one function signature
  * `litellm.Router` gives you load balancing, retries, and fallback across deployments without running a separate server
  * Good for scripts, notebooks, and small services where you do not want a standalone proxy process

<strong>Proxy: self hosted server, the real gateway</strong>

  * A FastAPI server (now Rust core backed) you deploy with Docker
  * OpenAI compatible REST API, so anything built for the OpenAI API works against it unchanged
  * Admin dashboard UI, virtual key management, per team and per project budgets
  * Guardrails: PII filters, content moderation, prompt injection detection, pluggable custom guardrail classes
  * Caching: exact match and semantic caching, backed by Redis, S3, or GCS
  * Cost tracking stored in PostgreSQL, live sync of model price and context window map so new models are supported day zero
  * Enterprise edition (paid) adds single sign on, SAML, role based access control, audit logs, per project budget isolation

A common student mistake is to only learn one piece and assume the other does not exist. The SDK is convenience. The proxy is infrastructure.

<h1><strong>9. langchain-litellm: what you actually get</strong></h1>

<strong>Short answer</strong>: partially, and it is important to understand exactly where the line sits.

`langchain-litellm` is the official LangChain integration package (`pip install langchain-litellm`) maintained under the `langchain-ai` GitHub org. It gives you:

  * `ChatLiteLLM`: a LangChain chat model wrapper around `litellm.completion()`, so you get LangChain's chain, tool calling, and structured output interface while LiteLLM handles provider translation
  * `ChatLiteLLMRouter`: wraps `litellm.Router`, so you get load balancing and fallback across deployments inside a LangChain chat model
  * `LiteLLMEmbeddings` and `LiteLLMEmbeddingsRouter`: same idea for embedding models
  * `LiteLLMOCRLoader`: a document loader that calls a LiteLLM proxy's OCR endpoint

What the package does <strong>not</strong> give you directly:

  * Virtual key management, per team budgets, single sign on, role based access control, audit logs
  * Guardrail configuration
  * The admin dashboard
  * Semantic caching configuration

That is expected. Those are proxy server features, configured once in the LiteLLM proxy's `config.yaml` and enforced server side, not per client library.

<strong>The way to reach the full feature set from LangChain</strong> is not through `langchain-litellm` at all. It is through LangChain's plain `ChatOpenAI` class, pointed at your running LiteLLM proxy.

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4",
    base_url="http://localhost:4000",
    api_key="sk-litellm-virtual-key",
)
```

Because the LiteLLM proxy exposes an OpenAI compatible API, this single line gets you every governance feature (budgets, guardrails, caching, virtual keys) for free, with zero LiteLLM specific LangChain code at all.

<strong>Decision rule</strong>

  * Use `langchain-litellm` (`ChatLiteLLM`, `ChatLiteLLMRouter`) when you want LiteLLM's multi provider SDK and in process routing directly inside a LangChain app, without running a standalone proxy server. This is the lighter, embedded path.
  * Use `ChatOpenAI` pointed at the LiteLLM proxy when you want the full gateway: virtual keys, team budgets, guardrails, semantic caching, dashboard, audit logs. This is the production, governed path.
  * The two are not mutually exclusive. Many real setups run the proxy for governance and still use `langchain-litellm`'s router class in application code for latency sensitive paths.

<h1><strong>10. The 2026 security reality</strong></h1>

A fact worth knowing, not hiding: LiteLLM had a turbulent 2026 for security, and understanding it is itself a useful skill.

<strong>March 2026: supply chain attack</strong>

A threat actor tracked as TeamPCP obtained a maintainer's PyPI publishing credentials and published two poisoned versions: `v1.82.7` and `v1.82.8`. The malware harvested cloud credentials and installed persistent backdoors. If you `pip install litellm` on those versions, you ship credential harvesting code into your environment.

LiteLLM shipped a clean `v1.83.0` from a new CI/CD v2 pipeline with isolated environments, Trusted Publishing on PyPI, and stricter security gates.

<strong>April 2026: Veria Labs audit</strong>

After the March incident, Veria Labs was brought in. They found several CVEs (CVE-2026-35029, CVE-2026-35030) that required a valid API key to exploit. The default LiteLLM configuration was not affected. All fixed in `v1.83.0`.

<strong>April to June 2026: the serious CVE chain</strong>

  * <strong>CVE-2026-42271</strong>: command injection in LiteLLM's MCP preview endpoints, affecting versions 1.74.2 through 1.83.6
  * <strong>CVE-2026-48710 (BadHost)</strong>: a host header validation bypass in the Starlette web framework
  * Chained together, these give unauthenticated remote code execution, with practical severity equivalent to CVSS 10.0
  * CISA added CVE-2026-42271 to the KEV (Known Exploited Vulnerabilities) catalog on June 8, 2026, with a federal remediation deadline of June 22, 2026

The full fix: LiteLLM version 1.83.14 stable or later, AND Starlette 1.0.1 or later. Either alone leaves you exposed.

<strong>The fix pattern to memorize</strong>

When a security incident hits a gateway you depend on, do a checklist, not a vibe.

  1. <strong>Pin to a known safe version</strong>. For LiteLLM, never run `latest`, never run a nightly, run `-stable`, and verify package integrity before upgrading
  2. <strong>Keep the proxy network isolated</strong>. Do not expose the admin UI to the public internet
  3. <strong>Rotate keys on suspicion</strong>. Every API key, database password, cloud credential, and SSH key on a host that ever ran the affected version
  4. <strong>Audit version history</strong>. Local environments, CI/CD pipelines, Docker builds, deployment logs
  5. <strong>Subscribe to vendor security feeds</strong>. LiteLLM publishes at `docs.litellm.ai/blog/tags/security`

The point of teaching this is not "LiteLLM is unsafe". The point is that every production gateway has a security posture that you, the engineer, are responsible for.

<h1><strong>11. One page cheat sheet</strong></h1>

<table>
<tr><th>Question</th><th>Answer</th></tr>
<tr><td>What is an LLM gateway?</td><td>Infrastructure between your app and model providers that normalizes, routes, enforces policy, caches, and logs</td></tr>
<tr><td>Why use one?</td><td>One API, fallback when providers fail, cost and budget control, caching, guardrails, observability</td></tr>
<tr><td>How is it different from LangChain?</td><td>LangChain builds the agent. A gateway runs it reliably across many models and teams</td></tr>
<tr><td>Which gateway should I learn first?</td><td>LiteLLM. Open source, largest community, about 55,000 stars, 100 plus providers, OpenAI compatible</td></tr>
<tr><td>LiteLLM SDK versus Proxy?</td><td>SDK is in process Python. Proxy is a self hosted server with virtual keys, budgets, guardrails, dashboard</td></tr>
<tr><td>How do I use LiteLLM from LangChain?</td><td>`langchain-litellm` for the SDK path, or `ChatOpenAI` with `base_url` pointed at the proxy for the full feature set</td></tr>
<tr><td>Is LiteLLM safe to self host?</td><td>Pin to version 1.83.14 stable or later and Starlette 1.0.1 or later. Isolate the proxy, rotate keys, subscribe to advisories</td></tr>
<tr><td>Fastest gateway?</td><td>Bifrost (about 11 microseconds, Go). LiteLLM Rust core closes the gap (about 0.7 ms p99 in their July 2026 benchmark)</td></tr>
<tr><td>Most feature rich gateway?</td><td>Portkey (guardrails, observability, audit). Now backed by Palo Alto Networks after the May 2026 acquisition</td></tr>
<tr><td>Easiest zero ops gateway?</td><td>OpenRouter for prototyping, Cloudflare AI Gateway if you already use Cloudflare</td></tr>
<tr><td>Best for compliance?</td><td>Portkey (SOC 2, HIPAA, GDPR), TrueFoundry (VPC, on prem), or Kong if you already run it</td></tr>
</table>

<h1><strong>12. Pointers</strong></h1>

  * LiteLLM docs: `https://docs.litellm.ai/`
  * LiteLLM security feed: `https://docs.litellm.ai/blog/tags/security`
  * LiteLLM release notes: `https://docs.litellm.ai/release_notes/`
  * langchain-litellm: `https://docs.langchain.com/oss/python/integrations/chat/litellm`
  * Portkey docs: `https://docs.portkey.ai/`
  * Bifrost: `https://github.com/maximhq/bifrost`
  * OpenRouter: `https://openrouter.ai/`
  * CISA KEV: `https://www.cisa.gov/known-exploited-vulnerabilities-catalog`
