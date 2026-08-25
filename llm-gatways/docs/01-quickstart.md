# Quickstart — Install & First Run

## 1.1 Install

### Option A — `uv` tool (recommended for dev)

```bash
uv tool install 'litellm[proxy]'
```

This gives you the `litellm` CLI globally without polluting your project venv.

### Option B — pip

```bash
pip install 'litellm[proxy]'
```

The `[proxy]` extra pulls in FastAPI, uvicorn, Prisma, asyncpg, and the provider SDKs needed for the server.

### Option C — Docker (recommended for production-ish)

```bash
docker pull ghcr.io/berriai/litellm:latest
```

`ghcr.io/berriai/litellm-database:latest` is a variant with pre-built Prisma binaries — use it if you connect to Postgres for fast startup.

## 1.2 Quickstart path 1 — One-shot CLI (no config)

Just to feel it. Serves a single model on port 4000:

```bash
export OPENAI_API_KEY=sk-...
litellm --model gpt-3.5-turbo
# INFO: Proxy running on http://0.0.0.0:4000
```

Now point any OpenAI client at it:

```python
from openai import OpenAI

client = OpenAI(
    api_key="anything",               # not checked at this layer
    base_url="http://0.0.0.0:4000",
)

resp = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Write a haiku about proxies"}],
)
print(resp.choices[0].message.content)
```

The OpenAI SDK thinks it's talking to OpenAI. It's not. It's talking to LiteLLM, which is talking to OpenAI on its behalf.

You can chain multiple `--model` flags to expose more than one model at a time:

```bash
litellm \
  --model gpt-3.5-turbo \
  --model anthropic/claude-3-5-sonnet \
  --model ollama/llama3
```

## 1.3 Quickstart path 2 — `config.yaml` (the real path)

### Step 1 — `config.yaml`

```yaml
model_list:
  - model_name: gpt-4o              # user-facing alias
    litellm_params:
      model: openai/gpt-4o          # provider/model
      api_key: os.environ/OPENAI_API_KEY

  - model_name: gpt-4o              # same alias, different deployment (load-balanced)
    litellm_params:
      model: azure/gpt-4o
      api_key: os.environ/AZURE_API_KEY
      api_base: os.environ/AZURE_API_BASE
      api_version: "2024-02-15-preview"

  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-latest
      api_key: os.environ/ANTHROPIC_API_KEY

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY   # required for prod
  # database_url: postgresql://...            # needed for keys/budgets (see below)
```

The `os.environ/VAR_NAME` syntax tells LiteLLM to read the value at startup from the environment — never hardcode keys in YAML.

### Step 2 — start

```bash
litellm --config config.yaml
# or with verbose debug:
litellm --config config.yaml --detailed_debug
```

Useful flags:

| Flag | Default | Purpose |
|---|---|---|
| `--config / -c PATH` | none | Path to `config.yaml`. |
| `--host` | `0.0.0.0` | Bind host. Use `127.0.0.1` for local-only. |
| `--port` | `4000` | Bind port. |
| `--num_workers` | logical CPUs or 4 | uvicorn/gunicorn workers. |
| `--detailed_debug` | off | Verbose logs. |
| `--log_config PATH` | none | uvicorn log config file. |
| `--keepalive_timeout` | uvicorn default | Seconds. |
| `--ssl_keyfile_path` / `--ssl_certfile_path` | none | Direct TLS termination. |
| `--test` | off | Validate config without starting. |

All of the above also accept env-var equivalents: `HOST`, `PORT`, `NUM_WORKERS`, `KEEPALIVE_TIMEOUT`, etc.

### Step 3 — call it

```bash
# any OpenAI client
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role":"user","content":"hi"}]
  }'
```

```python
from openai import OpenAI

c = OpenAI(api_key="sk-your-master-key", base_url="http://localhost:4000")
print(c.chat.completions.create(
    model="claude-sonnet",
    messages=[{"role":"user","content":"hi"}],
).choices[0].message.content)
```

## 1.4 Quickstart path 3 — Docker

```bash
docker run \
  -v $(pwd)/litellm_config.yaml:/app/config.yaml \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -p 4000:4000 \
  ghcr.io/berriai/litellm:latest \
  --config /app/config.yaml --detailed_debug
```

Or with Docker Compose (gateway + Postgres for state):

```yaml
# docker-compose.yml
services:
  litellm:
    image: ghcr.io/berriai/litellm-database:latest
    ports:
      - "4000:4000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/litellm
    depends_on:
      - db
  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=litellm
      - POSTGRES_PASSWORD=password
```

```bash
docker compose up
```

The official `docker-compose.yml` is also at https://docs.litellm.ai/docker-compose.yml (one-shot pull + run):

```bash
curl -sSL https://docs.litellm.ai/docker-compose.yml | docker compose -f - up -d
```

## 1.5 Verify it works

```bash
# liveness — process up?
curl http://localhost:4000/health/liveliness
# → "I'm alive!"

# readiness — DB + dependencies wired up?
curl http://localhost:4000/health/readiness

# list models the proxy knows about
curl http://localhost:4000/models \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY"

# admin UI
open http://localhost:4000/ui
```

## 1.6 Production minimum (don't skip this)

Before you let anyone hit it:

1. **Set `master_key`.** Otherwise anyone can call the proxy with `api_key="anything"`.
2. **Set up Postgres + `DATABASE_URL`.** Without a DB you cannot create virtual keys, track spend, or use the UI.
3. **Set up Redis** (separate from the DB) for caching and distributed rate limiting.
4. **Tighten `--host` and put a TLS-terminating ingress in front.** Don't expose the proxy on `0.0.0.0:4000` to the internet.
5. **Run ≥2 replicas** behind a load balancer. LiteLLM is stateless.

## 1.7 Sanity check — what gets installed where

| Component | What | Where |
|---|---|---|
| `litellm` binary / module | The proxy + SDK | `pip` site-packages / `uv` tool dir |
| `config.yaml` | Your model & policy | wherever you point `--config` |
| `.env` | Secret values | anywhere, exported into env |
| `prisma/` | DB schema migrations | generated on first run, lives in the proxy's working dir |
| `litellm.db` (if SQLite) | local dev DB | generated next to `config.yaml` if no `DATABASE_URL` set |

If you don't set `DATABASE_URL`, LiteLLM falls back to a local SQLite file — fine for dev, never for prod.

---

Next: [02-config-yaml.md](./02-config-yaml.md) — the full anatomy of `config.yaml`.
