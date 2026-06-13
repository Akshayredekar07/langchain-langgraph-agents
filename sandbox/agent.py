import os
import sys
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
from langgraph.types import Overwrite

stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
if callable(stdout_reconfigure):
    stdout_reconfigure(encoding="utf-8", errors="replace")

env_file = os.getenv("ENV_FILE")
load_dotenv(env_file or find_dotenv(".env", usecwd=True))
load_dotenv(override=False)

from sandbox_env import generate_pdf, generate_docx, generate_pptx

assert os.getenv("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY not set"

raw_anthropic_model = os.getenv("CLAUDE_MODEL") or os.getenv("ANTHROPIC_MODEL")
ANTHROPIC_MODEL = (
    raw_anthropic_model
    if raw_anthropic_model and raw_anthropic_model.startswith("claude-")
    else "claude-sonnet-4-6"
)

# ✅ FIX 1: Raise default timeout to 600s (matches Anthropic SDK default)
MODEL_TIMEOUT_SECONDS = int(os.getenv("MODEL_TIMEOUT_SECONDS", "600"))

model = init_chat_model(
    f"anthropic:{ANTHROPIC_MODEL}",
    timeout=MODEL_TIMEOUT_SECONDS,
    max_retries=2,  # ✅ FIX 2: Allow retries on transient failures
)

agent = create_deep_agent(
    model=model,
    tools=[generate_pdf, generate_docx, generate_pptx],
    system_prompt=(
        "You are a document generation assistant. "
        "When the user asks for a PDF, DOCX, or PPTX, call the appropriate tool. "
        "Always pass a profile_id of 'default' unless the user specifies one."
    ),
)


def get_messages(node_data):
    if isinstance(node_data, Overwrite):
        node_data = node_data.value
    if isinstance(node_data, dict):
        msgs = node_data.get("messages", [])
    elif hasattr(node_data, "messages"):
        msgs = node_data.messages
    else:
        msgs = node_data
    if isinstance(msgs, Overwrite):
        msgs = msgs.value
    if isinstance(msgs, list):
        return msgs
    return []


def normalize_stream_chunk(chunk):
    if isinstance(chunk, dict):
        return chunk.get("type"), chunk.get("data")
    if isinstance(chunk, tuple):
        if len(chunk) == 3:
            _, chunk_type, data = chunk
            return chunk_type, data
        if len(chunk) == 2:
            chunk_type, data = chunk
            return chunk_type, data
    return None, None


def stream_agent(user_input: str):
    if raw_anthropic_model and raw_anthropic_model != ANTHROPIC_MODEL:
        print(f"[AGENT] ignoring non-Claude ANTHROPIC_MODEL={raw_anthropic_model!r}", flush=True)
    print(f"[AGENT] model=anthropic:{ANTHROPIC_MODEL}", flush=True)
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode=["updates", "messages"],
        subgraphs=True,
        version="v2",
    ):
        chunk_type, data = normalize_stream_chunk(chunk)

        if chunk_type == "updates":
            for node_name, node_data in data.items():
                msgs = get_messages(node_data)
                if not msgs:
                    continue
                for msg in msgs:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"\n[TOOL CALL] {tc['name']}")
                            print(f"  args: {tc['args']}")
                    elif getattr(msg, "type", None) == "tool":
                        print(f"\n[TOOL RESULT] {msg.name}")
                        print(f"  {str(msg.content)[:300]}")

        elif chunk_type == "messages":
            token, _ = data
            if getattr(token, "content", None) and getattr(token, "type", None) == "ai":
                print(token.content, end="", flush=True)


COMPLEX_QUERY = """
Generate a pdf report titled 'OAuth 2.0 & OpenID Connect — A Production Guide' with the following sections:

## 1. Introduction

Modern applications delegate authentication and authorization to identity providers using **OAuth 2.0**
and **OpenID Connect (OIDC)**. This report covers core flows, token mechanics, security pitfalls,
and production implementation patterns.

---

## 2. Core Concepts

### 2.1 OAuth 2.0 Roles

- **Resource Owner** — the end user who owns the data
- **Client** — the application requesting access (your app)
- **Authorization Server** — issues tokens (e.g. Auth0, Keycloak, Google)
- **Resource Server** — the API that accepts access tokens

### 2.2 Token Types

| Token Type    | Format  | Lifespan  | Purpose                        |
|---------------|---------|-----------|--------------------------------|
| Access Token  | JWT     | 5–60 min  | Authorize API calls            |
| Refresh Token | Opaque  | Days–weeks| Obtain new access tokens       |
| ID Token      | JWT     | Short     | Authenticate the user (OIDC)   |

### 2.3 Grant Types

- **Authorization Code + PKCE** — web and mobile apps (recommended)
- **Client Credentials** — machine-to-machine (M2M) flows
- **Device Code** — CLI tools and smart devices
- **Implicit** — deprecated, avoid in new systems

---

## 3. Mathematical Foundations

### 3.1 PKCE — Proof Key for Code Exchange

The code verifier is a random string. The challenge is derived as:

    code_verifier  = random_string(43–128 chars, URL-safe)
    code_challenge = BASE64URL( SHA-256( code_verifier ) )

This prevents authorization code interception attacks because:

    If attacker intercepts code → they cannot derive code_verifier
    Without code_verifier → token endpoint rejects the exchange

### 3.2 JWT Structure

A JWT consists of three Base64URL-encoded parts:

    token = BASE64URL(header) + "." + BASE64URL(payload) + "." + BASE64URL(signature)

Signature verification (RS256):

    valid = RSA_VERIFY( SHA256(header + "." + payload), signature, public_key )

---

## 4. Code Implementation

### 4.1 Authorization Code Flow with PKCE (Python)

```python
import secrets
import hashlib
import base64
import httpx

def generate_pkce_pair():
    code_verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(code_verifier.encode()).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return code_verifier, code_challenge

def build_auth_url(client_id, redirect_uri, scope, state, code_challenge):
    base = "https://auth.example.com/authorize"
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scope,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    return base + "?" + "&".join(f"{k}={v}" for k, v in params.items())
```

### 4.2 Token Exchange

```python
async def exchange_code_for_tokens(code, code_verifier, client_id, redirect_uri):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://auth.example.com/oauth/token",
            json={
                "grant_type": "authorization_code",
                "code": code,
                "code_verifier": code_verifier,
                "client_id": client_id,
                "redirect_uri": redirect_uri,
            },
        )
        response.raise_for_status()
        return response.json()  # access_token, refresh_token, id_token
```

### 4.3 Validating a JWT Access Token (FastAPI)

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer
from jose import jwt, JWTError
import httpx

JWKS_URL = "https://auth.example.com/.well-known/jwks.json"
AUDIENCE  = "https://api.example.com"
ISSUER    = "https://auth.example.com/"

async def get_jwks():
    async with httpx.AsyncClient() as client:
        r = await client.get(JWKS_URL)
        return r.json()

async def verify_token(token: str = Depends(HTTPBearer())):
    try:
        jwks = await get_jwks()
        payload = jwt.decode(
            token.credentials,
            jwks,
            algorithms=["RS256"],
            audience=AUDIENCE,
            issuer=ISSUER,
        )
        return payload
    except JWTError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
```

### 4.4 Machine-to-Machine (Client Credentials)

```python
async def get_m2m_token(client_id: str, client_secret: str, audience: str):
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://auth.example.com/oauth/token",
            json={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
                "audience": audience,
            },
        )
        r.raise_for_status()
        return r.json()["access_token"]
```

---

## 5. System Architecture

### 5.1 Authorization Code Flow (Mermaid)

```mermaid
sequenceDiagram
    participant U as User
    participant A as App (Client)
    participant AS as Auth Server
    participant RS as Resource Server

    U->>A: Click Login
    A->>AS: GET /authorize?code_challenge=...
    AS->>U: Show Login Page
    U->>AS: Submit Credentials
    AS->>A: Redirect with ?code=...
    A->>AS: POST /token (code + code_verifier)
    AS->>A: access_token + refresh_token + id_token
    A->>RS: GET /api/data (Bearer access_token)
    RS->>A: Protected Resource
```

### 5.2 Token Refresh Flow

    access_token expires
        → POST /oauth/token { grant_type: refresh_token, refresh_token: <rt> }
        → Auth Server validates refresh token (rotation check)
        → Returns new access_token (+ rotated refresh_token)
        → If refresh_token expired → force re-login

---

## 6. Security Checklist

1. Always use **Authorization Code + PKCE** for user-facing apps — never Implicit flow
2. Validate `iss`, `aud`, `exp`, `nbf` claims on every JWT — reject if any fail
3. Store tokens in **httpOnly cookies**, never in localStorage (XSS risk)
4. Enable **refresh token rotation** and detect reuse (sign of token theft)
5. Set short access token TTL (15 min recommended for sensitive APIs)
6. Use **state parameter** to prevent CSRF on the redirect callback
7. Restrict scopes to minimum required — follow principle of least privilege
8. Cache JWKS with a TTL, but re-fetch on unknown `kid` (key rotation)

---

## 7. Key Observations

> "PKCE is not optional — it is the minimum bar for any public client, regardless of whether the app is a SPA or mobile."

> "Most OAuth breaches are not protocol failures — they are implementation failures: missing state validation, overly broad scopes, or tokens stored in localStorage."

---

## 8. Conclusion

OAuth 2.0 and OIDC provide a **battle-tested framework** for delegated authorization and federated identity.
The protocol is sound — but correct implementation requires validating every claim, rotating secrets,
and keeping token lifetimes short. Build on a certified library and audit your flows against the
OAuth 2.0 Security Best Current Practice (BCP).

Output filename: OAuth2_Production_Guide.pdf
"""

query = "Generte pptx on topic of software engineering lifecycle SDLC cover all points of genral pptx have from intorduction to conclusion"

if __name__ == "__main__":
    stream_agent(query)
    print()
