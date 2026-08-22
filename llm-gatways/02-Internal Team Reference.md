# LLM / AI Gateways — Internal Team Reference

> **Owner:** Platform / AI Infra
> **Status:** Living document — update when versions, security advisories, or vendor posture change.
> **Last verified:** 2026-08-22 (LiteLLM v1.97.0 / safe line v1.83.14-stable).

---

## 1. TL;DR

- **Default gateway:** LiteLLM (self-hosted, MIT). Largest community, 100+ providers, OpenAI-compatible, ~55k GitHub stars.
- **Default SDK pattern (LangChain):** `ChatLiteLLM` for in-process multi-provider; `ChatOpenAI(base_url=...)` for governed traffic.
- **Default proxy URL (internal):** `http://litellm.internal:4000`
- **Safe self-host baseline:** LiteLLM `v1.83.14-stable` or later **AND** Starlette `v1.0.1` or later. Anything older is exposed to CVE-2026-42271 + CVE-2026-48710 (unauth RCE chain, on CISA KEV).
- **Use managed when:** team has no infra capacity, or we need a feature LiteLLM lacks (e.g., Bifrost latency at 5,000+ RPS, Portkey enterprise guardrails, OpenRouter model breadth for prototyping).

---

## 2. Why a gateway (one paragraph)

Every model provider ships a different SDK, auth scheme, streaming protocol, rate-limit shape, and pricing model. A gateway gives us one OpenAI-format endpoint, one virtual key per team, per-team budgets, PII / prompt-injection guardrails, semantic caching, automatic failover, and cost attribution. The application code stays vendor-neutral; the governance lives in one place.

---

## 3. Quick comparison — all 20 gateways

| # | Gateway | License | Hosting | Providers | Standout | Pick when |
|---|---|---|---|---|---|---|
| 1 | **LiteLLM** | MIT | Self-host | 100+ | Largest community, broadest coverage | Default. Prototyping, internal tools, < 500 RPS. |
| 2 | **Bifrost** | Apache 2.0 | Self-host (Go) | 12+ direct, 1,000+ advertised | ~11 μs overhead, 5,000+ RPS | High-throughput agent platforms. |
| 3 | **Portkey Gateway (core)** | Apache 2.0 (core) | Self-host core / managed control plane | 1,600+ | Deepest guardrails + observability | Regulated workloads, PII, HIPAA, audit. |
| 4 | **LLM Gateway (llmgateway.io)** | Open source | Self-host | 210–300+ | Zero markup, simple | Cost-sensitive, no frills. |
| 5 | **Envoy AI Gateway** | Apache 2.0 (CNCF) | Self-host | Provider via filters | Reuses our existing Envoy | Platform team already runs Envoy. |
| 6 | **Apache APISIX (ai-proxy)** | Apache 2.0 | Self-host | Plugin-based | Reuses our existing APISIX | Platform team already runs APISIX. |
| 7 | **OpenRouter** | Proprietary SaaS | Hosted | 300–400+ | Largest model marketplace, 5-min setup | Prototyping, model breadth, zero ops. |
| 8 | **Requesty** | Proprietary SaaS | Hosted | 400+ | 8 ms P50, 5% markup, agentic mode | Production routing with caching. |
| 9 | **Eden AI** | Proprietary SaaS | Hosted | Many AI modalities | OCR, speech, vision unified | Multi-modal pipelines. |
| 10 | **Martian** | Proprietary SaaS | Hosted | 100+ | Learned cost/quality router | Trust an auto-router for cost/quality. |
| 11 | **Not Diamond** | Proprietary SaaS | Hosted | Multi | Auto model selection | Alternative smart router. |
| 12 | **Unify AI** | Proprietary SaaS | Hosted | Multi | Live-benchmark-driven routing | Quality-led routing. |
| 13 | **AWS Bedrock** | Proprietary | AWS-managed | Multi | IAM-integrated | AWS-only workloads. |
| 14 | **Azure APIM AI Gateway** | Proprietary | Azure-managed | Multi | Token rate limits, semantic cache | Azure-only workloads. |
| 15 | **Google Vertex Model Garden** | Proprietary | GCP-managed | Multi | GCP-native | GCP-only workloads. |
| 16 | **Cloudflare AI Gateway** | Proprietary SaaS | Hosted / edge | Major | Edge-cached, free tier | Already on Cloudflare Workers/Pages. |
| 17 | **Kong AI Gateway** | Mixed | Self-host / enterprise | Provider via plugins | Reuses our existing Kong | Already on Kong. |
| 18 | **Zuplo** | Proprietary | Hosted | Multi | Lightweight API mgmt + AI | Already on Zuplo. |
| 19 | **Vercel AI Gateway** | Proprietary SaaS | Hosted | Major | Zero markup, Vercel-native | Next.js / Vercel stack. |
| 20 | **Helicone** | Apache 2.0 | Self-host (Rust) | 20+ providers, 100+ models | ~5 ms overhead, 3,000 RPS, observability-first | **Maintenance mode** (acquired by Mintlify 2026) — do not adopt for new work. |

---

## 4. Decision matrix — which to use when

| Situation | Use | Why |
|---|---|---|
| Internal tool / dev script, one developer | OpenAI SDK direct | No infra needed. |
| LangChain app, internal, latency-sensitive | `ChatLiteLLM` (SDK path) | In-process, no extra hop. |
| LangChain app, production, multi-team | `ChatOpenAI` → LiteLLM proxy | Virtual keys, budgets, guardrails, audit. |
| Need a model not on our LiteLLM list | OpenRouter (sandbox) → vendor key into LiteLLM | Add to LiteLLM config if we use it more than 3x/month. |
| > 500 RPS sustained | Bifrost (or LiteLLM Rust core) | Python path hits ceiling around 1,000 RPS. |
| HIPAA / SOC 2 / regulated workload | Portkey (managed) or TrueFoundry VPC | LiteLLM enterprise tier for self-host; Portkey for managed. |
| Customer data must not leave our infra | LiteLLM self-host (Bifrost self-host) | On-prem control plane. |
| Edge / Cloudflare stack | Cloudflare AI Gateway | Free, zero ops, already in the network path. |
| Compliance + we already run Kong | Kong AI Gateway | One less new tool. |
| Quality leaderboard matters more than cost | Unify AI | Live benchmark-driven routing. |

---

## 5. Deployment runbook — LiteLLM (our default)

### 5.1 Component map

```
┌────────────┐    ┌──────────────┐    ┌────────────┐
│  Apps /    │───▶│  LiteLLM     │───▶│  Postgres  │  (spend logs, virtual keys)
│  LangChain │    │  Proxy       │    └────────────┘
│  / Agents  │    │  (port 4000) │    ┌────────────┐
└────────────┘    │              │───▶│   Redis    │  (cache + rate limit counters)
                  │  Admin UI    │    └────────────┘
                  │  (port 4001) │
                  └──────────────┘
```

### 5.2 docker-compose.yml (template)

```yaml
services:
  litellm:
    image: ghcr.io/berriai/litellm:v1.97.0-stable
    # Pin a specific version. Never use :main-latest in prod.
    # Safe self-host line as of 2026-08: v1.83.14-stable or later,
    # plus Starlette v1.0.1+ (transitive dep, pin in requirements.txt).
    ports:
      - "4000:4000"
      - "4001:4001"   # admin UI; bind to internal network only
    volumes:
      - ./config.yaml:/app/config.yaml:ro
    environment:
      - DATABASE_URL=postgresql://litellm:REDACTED@postgres:5432/litellm
      - REDIS_URL=redis://redis:6379
      - LITELLM_MASTER_KEY=${LITELLM_MASTER_KEY}   # vault-managed
      - LITELLM_SALT=${LITELLM_SALT}               # for hash, vault-managed
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    networks: [internal]
    # DO NOT expose 4001 to the public internet.

  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_USER=litellm
      - POSTGRES_PASSWORD=REDACTED
      - POSTGRES_DB=litellm
    volumes:
      - litellm_pg:/var/lib/postgresql/data
    networks: [internal]
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: ["redis-server", "--maxmemory", "512mb", "--maxmemory-policy", "allkeys-lru"]
    networks: [internal]
    restart: unless-stopped

volumes:
  litellm_pg:

networks:
  internal:
    driver: bridge
```

### 5.3 config.yaml (template)

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: gpt-4o
      api_key: os.environ/OPENAI_API_KEY
  - model_name: gpt-4o-mini
    litellm_params:
      model: gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY
  - model_name: claude-sonnet
    litellm_params:
      model: claude-sonnet-4-5
      api_key: os.environ/ANTHROPIC_API_KEY
  - model_name: claude-haiku
    litellm_params:
      model: claude-haiku-4-5
      api_key: os.environ/ANTHROPIC_API_KEY
  - model_name: gemini-flash
    litellm_params:
      model: vertex_ai/gemini-2.5-flash
      vertex_project: os.environ/GCP_PROJECT
      vertex_location: us-central1

router_settings:
  num_retries: 2
  timeout: 30
  fallbacks:
    - gpt-4o: [claude-sonnet]
    - claude-sonnet: [gpt-4o]
  enable_caching: true
  caching_groups:
    - ["gpt-4o-mini", "claude-haiku"]   # cross-model cache for cheap tier

litellm_settings:
  drop_params: true
  success_callback: ["langfuse"]          # or ["prometheus"] for metrics
  # PII / prompt-injection guardrails
  guardrails:
    - type: pii_detection
      config:
        categories: ["ssn", "credit_card", "email", "phone"]
        action: "block"
    - type: prompt_injection
      config:
        threshold: 0.8
        action: "block"

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL
  telemetry: false
  # Disable JWT auth unless you specifically need it.
  # enable_jwt_auth: false  (default; leave it that way)

# Virtual keys are issued via the admin API or UI, not in this file.
# See section 5.4.
```

### 5.4 Virtual keys (via admin API)

```bash
# Issue a key for a team
curl -X POST "http://litellm.internal:4000/key/generate" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "team_id": "team-data-eng",
    "key_alias": "data-eng-prod-2026Q3",
    "models": ["gpt-4o", "claude-sonnet", "claude-haiku", "gpt-4o-mini"],
    "max_budget": 2000,
    "budget_duration": "30d",
    "duration": "90d",
    "metadata": {"owner": "[email protected]", "cost_center": "eng-ai"}
  }'
```

Response:

```json
{
  "key": "sk-litellm-virtual-REDACTED",
  "key_alias": "data-eng-prod-2026Q3",
  "expires": "2026-11-19T..."
}
```

Distribute the virtual key to the team via 1Password / vault. **Never** share the master key or any real provider key with application code.

---

## 6. LangChain integration (canonical patterns)

### 6.1 Internal / dev — SDK path (in-process)

```python
from langchain_litellm import ChatLiteLLM

llm = ChatLiteLLM(
    model="gpt-4o",                    # or "claude-sonnet", "vertex_ai/gemini-..."
    # api_key taken from OPENAI_API_KEY / ANTHROPIC_API_KEY / etc. env vars
)
resp = llm.invoke("Summarize the attached doc.")
```

### 6.2 Production / governed — proxy path

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    base_url="http://litellm.internal:4000",
    api_key="sk-litellm-virtual-REDACTED",   # per-team virtual key from vault
    temperature=0.2,
    timeout=30,
)
resp = llm.invoke("Summarize the attached doc.")
```

### 6.3 Embeddings

```python
from langchain_litellm import LiteLLMEmbeddings        # SDK
# from langchain_openai import OpenAIEmbeddings         # via proxy
embeddings = LiteLLMEmbeddings(model="text-embedding-3-small")
vec = embeddings.embed_query("hello world")
```

### 6.4 Pattern: tag every call for spend attribution

```python
resp = llm.invoke(
    prompt,
    config={"metadata": {"tags": ["feature:inbox-summarizer", "env:prod"]}},
)
```

Tags show up in the dashboard and the spend log. Use them consistently.

---

## 7. Security baseline (mandatory for every LiteLLM deployment)

### 7.1 Version pins

| Component | Required version | Reason |
|---|---|---|
| LiteLLM | `v1.83.14-stable` or later | Closes CVE-2026-42271 (MCP command injection), CVE-2026-48710 chain, CVE-2026-47101 chain. |
| Starlette | `v1.0.1` or later | Closes BadHost host-header bypass. |
| **Never** run | `v1.82.7`, `v1.82.8` | TeamPCP supply chain attack (March 2026). |
| **Avoid** running | `:main-latest`, `:nightly` in prod | Mutable tags. Pin exact version + `*-stable` suffix. |

Track the safe line: `docs.litellm.ai/release_notes/`. Security tags: `docs.litellm.ai/blog/tags/security`.

### 7.2 Network isolation checklist

- [ ] Admin UI (port 4001) bound to internal network only. **No public exposure.**
- [ ] Proxy API (port 4000) fronted by an authenticating reverse proxy (Nginx / Envoy / Cloudflare Access).
- [ ] Postgres and Redis on an isolated Docker network. Not internet-reachable.
- [ ] `LITELLM_MASTER_KEY` rotated via vault, not env file.
- [ ] `enable_jwt_auth: false` unless an explicit need exists.

### 7.3 Key rotation runbook

When a CVE is published that affects a version we run, or on a suspected compromise:

1. **Identify scope.** List every host, container, CI runner, and dev laptop that ran the affected version.
   ```bash
   # In a Python env that might have run it:
   pip show litellm | grep -E "^(Name|Version):"
   # Look for v1.82.7 / v1.82.8 specifically.
   ```
2. **Rotate immediately.** Treat as compromised:
   - All provider API keys (OpenAI, Anthropic, GCP, AWS).
   - All DB credentials used by LiteLLM.
   - All cloud IAM credentials on the host.
   - All SSH keys, Kubeconfig tokens, anything in `.env` on the host.
3. **Inspect.** Look for `litellm_init.pth` in `site-packages`, outbound traffic to `models.litellm[.]cloud` (not affiliated with LiteLLM), unusual child processes, new crontabs, new IAM users.
4. **Upgrade.** Move to safe version (see 7.1), restart, re-verify.
5. **Audit.** Pull deployment logs + Docker build history + CI cache. Determine time window of exposure.
6. **Report.** File an internal incident. Notify Security team. Document timeline.

### 7.4 CVE monitoring (subscribe)

- LiteLLM security blog tag: `https://docs.litellm.ai/blog/tags/security`
- LiteLLM release notes: `https://docs.litellm.ai/release_notes/`
- CISA KEV: `https://www.cisa.gov/known-exploited-vulnerabilities-catalog`
- Internal: `#ai-infra-security` Slack channel for alerts.

---

## 8. Spend & usage monitoring

### 8.1 What the dashboard gives us out of the box

- Per-virtual-key spend (current month, rolling window).
- Per-model breakdown.
- Per-team rollup.
- Top-N consumers (key + tag).

URL: `http://litellm.internal:4001/ui` (admin key required).

### 8.2 Programmatic spend check

```bash
# Per-key spend
curl "http://litellm.internal:4000/spend/keys?key=sk-litellm-virtual-REDACTED" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY"

# Per-team spend
curl "http://litellm.internal:4000/team/info?team_id=team-data-eng" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY"
```

### 8.3 Recommended tagging convention

| Tag | Meaning | Example |
|---|---|---|
| `feature:<slug>` | Which product feature owns the call | `feature:inbox-summarizer` |
| `env:<env>` | Deployment environment | `env:prod`, `env:staging` |
| `tenant:<id>` | Customer / tenant (multi-tenant products) | `tenant:acme-corp` |
| `agent:<name>` | Agent / chain name | `agent:research-assistant` |

Tags are passed via `metadata={"tags": [...]}` on the LangChain call, or via the OpenAI SDK `extra_body={"metadata": {"tags": [...]}}`. The proxy stores them and shows them in the spend dashboard.

### 8.4 Alerting (suggested thresholds)

- Key spend at **80% of monthly budget** → Slack DM to key owner.
- Key spend at **100% of monthly budget** → auto-block + page on-call.
- Per-key request volume > 5× its 7-day median → Slack alert (possible runaway loop).

---

## 9. Incident response — the 2026 LiteLLM CVE chain, summarized

The 2026 LiteLLM security timeline (one-page mental model for on-call):

| Date | Event | Action |
|---|---|---|
| Feb 2026 | Obsidian Security reports three chained CVEs to BerriAI privately. | Tracked; fixes in flight. |
| Mar 2026 | TeamPCP poisons `v1.82.7` and `v1.82.8` on PyPI (supply chain). Credential-harvesting payload. | **DO NOT RUN those versions.** If you did, rotate everything on that host. |
| Mar 30, 2026 | Clean `v1.83.0` released from new CI/CD v2 pipeline. | Upgrade. |
| Apr 25, 2026 | `v1.83.14-stable` released. Closes Obsidian CVE-2026-47101 chain (CVSS 9.9) and the JWT/cache CVEs. | **Recommended minimum self-host version.** |
| May 8, 2026 | `v1.83.7` released with MCP endpoint authorization. Closes CVE-2026-42271. | Upgrade. |
| Jun 8, 2026 | CISA adds CVE-2026-42271 to KEV after in-the-wild exploitation. | Treat as emergency. |
| Aug 15, 2026 | `v1.97.0` released (tool-result guardrails, auto-router deployment affinity, admin viewer parity). | Track; upgrade on next maintenance window. |

**If we are running anything older than `v1.83.14-stable`:**
1. Page on-call.
2. Open a security ticket.
3. Upgrade during next business window, not "later."
4. Add Starlette `v1.0.1+` to the upgrade in the same PR.

---

## 10. Vendor risk notes (decisions already made)

| Vendor | Posture | Why we are / are not using it |
|---|---|---|
| LiteLLM | **In use.** Default. | Community, breadth, self-host, no markup. |
| Bifrost | Available on request. | Reach for it if sustained traffic > 1,000 RPS. |
| Portkey | **In use for regulated workloads.** | Strongest guardrails + observability. Core is Apache 2.0; managed control plane is proprietary. Now backed by Palo Alto Networks (acquired May 2026) — re-evaluate annually. |
| OpenRouter | Used in sandbox / prototyping only. | 5.5% markup on credits — re-evaluate at $5K/mo spend. |
| Helicone | **Not for new work.** | Acquired by Mintlify 2026, in maintenance mode. |
| Requesty | Evaluating for agentic workloads. | 8 ms P50 + caching — promising for high-volume agents. |
| Cloudflare AI Gateway | In use where we already run on Cloudflare edge. | Free, zero ops. |
| TrueFoundry | Under evaluation for VPC-deployed enterprise workloads. | HIPAA / on-prem option. |
| Kong / Envoy / APISIX | **Do not deploy new** unless we already run them. | Reuse existing API gateway if applicable; do not introduce a new one for AI traffic alone. |

---

## 11. Common operational tasks

### 11.1 Add a new model to the proxy

1. Get the provider API key into vault. Reference it as `os.environ/...` in `config.yaml`.
2. Add an entry under `model_list`.
3. Reload config:
   ```bash
   curl -X POST "http://litellm.internal:4000/config/update" \
     -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
     -H "Content-Type: application/json" \
     -d @new-config.yaml
   ```
4. Smoke test from the admin UI or a temporary virtual key.
5. Add the model to the allowed list on existing virtual keys if needed.

### 11.2 Add a new team / virtual key

```bash
# Create team
curl -X POST "http://litellm.internal:4000/team/new" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"team_alias": "team-ml-research", "max_budget": 1500, "budget_duration": "30d"}'

# Generate key for that team
curl -X POST "http://litellm.internal:4000/key/generate" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"team_id": "team-ml-research", "models": ["gpt-4o", "claude-sonnet"], "max_budget": 500, "budget_duration": "30d"}'
```

### 11.3 Block a model temporarily

```bash
curl -X POST "http://litellm.internal:4000/model/update" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "gpt-4o", "model_info": {"access_groups": ["disabled"]}}'
```

Or simply remove it from `config.yaml` and reload.

### 11.4 Investigate a request

```bash
# All requests for a key in the last hour
curl "http://litellm.internal:4000/spend/logs?key=sk-litellm-virtual-REDACTED&start_time=2026-08-22T13:00:00Z" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY"
```

The response includes request ID, model, token counts, latency, status, and tags. Cross-reference request ID with Langfuse / OTel for trace.

### 11.5 Failover test (verify a fallback works)

```bash
# Force a request through, then break the primary upstream by revoking the key in vault and rolling.
# Observe that the proxy routes to the fallback model.
curl "http://litellm.internal:4000/chat/completions" \
  -H "Authorization: Bearer sk-litellm-virtual-REDACTED" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "ping"}]}'
```

### 11.6 Rotate the LiteLLM master key

1. Generate new master key in vault.
2. Update `LITELLM_MASTER_KEY` env var on the proxy.
3. Restart the proxy.
4. Update any scripts that use the master key (admin API calls) to read the new value from vault.
5. Revoke the old master key (there is no in-app revocation; treat it as compromised and ensure no other system had a copy).

---

## 12. Glossary (for new joiners)

| Term | Meaning |
|---|---|
| Provider | Company hosting the model (OpenAI, Anthropic, Google, AWS Bedrock, etc.). |
| Virtual key | A proxy-issued key bound to a budget, team, and allowed models. |
| Master key | The single admin key. Treat as root. |
| Guardrail | A request/response check (PII, prompt injection, moderation). |
| Semantic cache | Cache hit on prompt similarity, not just exact match. |
| MCP | Model Context Protocol. Standard for tool calls. Has had CVEs in LiteLLM. |
| CVE | Public vulnerability ID. |
| KEV | CISA Known Exploited Vulnerabilities catalog. Real-world exploitation confirmed. |
| RCE | Remote Code Execution. Worst class. |
| OpenAI-compatible API | REST shape matching OpenAI's `/v1/chat/completions`. The lingua franca. |
| Fallback | Secondary model the proxy tries if the primary fails or rate-limits. |
| Virtual key rotation | Issuing a new key to a team; the old key is invalidated. |
| Tag | Label on a request for spend attribution. Convention: `feature:`, `env:`, `tenant:`, `agent:`. |

---

## 13. Pointers

- LiteLLM docs: `https://docs.litellm.ai/`
- LiteLLM security feed: `https://docs.litellm.ai/blog/tags/security`
- LiteLLM release notes: `https://docs.litellm.ai/release_notes/`
- LangChain × LiteLLM: `https://docs.langchain.com/oss/python/integrations/chat/litellm`
- LangChain × OpenAI (proxy path): `https://python.langchain.com/docs/integrations/chat/openai/`
- CISA KEV: `https://www.cisa.gov/known-exploited-vulnerabilities-catalog`
- Internal Slack: `#ai-infra`, `#ai-infra-security`
- On-call rotation: see PagerDuty schedule `ai-infra-primary`
- Vault path: `secret/ai-infra/litellm/*`
