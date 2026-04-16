# **LangGraph Agent Chat UI**

> **Covers:** Local Dev Server · Agent Chat UI · LangSmith Studio · Hosted Deployment  
> **Stack:** Python 3.11+ · LangGraph CLI · Node.js · LangSmith

---

## **Table of Contents**

1. [Project Structure](#1-project-structure)
2. [Prerequisites & Installation](#2-prerequisites--installation)
3. [Core Configuration Files](#3-core-configuration-files)
4. [Approach 1 — LangSmith Studio (Simplest)](#4-approach-1--langsmith-studio-simplest)
5. [Approach 2 — Agent Chat UI (localhost:5173)](#5-approach-2--agent-chat-ui-localhost5173)
6. [Approach 3 — Hosted Deployment](#6-approach-3--hosted-deployment)
7. [Common Mistakes — What NOT To Do](#7-common-mistakes--what-not-to-do)
8. [Troubleshooting Reference](#8-troubleshooting-reference)
9. [Quick Comparison Table](#9-quick-comparison-table)

---

### **1. Project Structure**

```
D:\Langchain-dev\langchain-deepdive\
│
│   langgraph.json          ← root config (used by all approaches)
│   requirements.txt        ← Python dependencies
│   .env                    ← ALL your API keys go here
│
├───examples\
│       agent.py            ← your compiled Python agent/graph
│
└───my-chat-ui\             ← only needed for Approach 2
        langgraph.json      ← points back to your agent
        .env                ← copy of keys (needed if pnpm run dev starts its own server)
```

---

### **2. Prerequisites & Installation**

### **Python Environment**

```powershell
# Verify Python version (3.11+ required)
python --version

# Install LangGraph CLI with inmem support — ALWAYS use [inmem]
pip install -U "langgraph-cli[inmem]"

# Verify the dev command is now available
langgraph --help
# Must show: build, dev, dockerfile, new, up
```

> **Critical:** `pip install langgraph-cli` (without `[inmem]`) will NOT include the `dev` command. You will only see `build`, `dockerfile`, `new`, and `up`. Always use `pip install "langgraph-cli[inmem]"`.

### **Install Agent Dependencies**

```powershell
pip install langchain langchain-groq python-dotenv
```

### **Node.js (Only for Approach 2)**

```powershell
# Check if Node.js v18+ is installed
node --version
npm --version

# Install pnpm (required by create-agent-chat-app)
npm install -g pnpm
pnpm --version
```

---

## **3. Core Configuration Files**

### **`langgraph.json` (Root — Used by All Approaches)**

Location: `D:\Langchain-dev\langchain-deepdive\langgraph.json`

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./examples/agent.py:agent"
  },
  "env": "./.env"
}
```

> **Important:** The part after `:` must match the exact variable name of your compiled graph in `agent.py`:
>
> ```python
> graph  = workflow.compile()   # → use  "./examples/agent.py:graph"
> agent  = workflow.compile()   # → use  "./examples/agent.py:agent"
> app    = workflow.compile()   # → use  "./examples/agent.py:app"
> ```

### **`.env` File (Root)**

Location: `D:\Langchain-dev\langchain-deepdive\.env`

```env
# LLM Provider Key (use whichever your agent uses)
GROQ_API_KEY=gsk_...
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# LangSmith (required for Approach 1 and Approach 3)
LANGSMITH_API_KEY=lsv2_...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=my-agent

# Set to false to disable tracing during local development
# LANGCHAIN_TRACING_V2=false
```

### **`requirements.txt`**

```txt
langchain
langchain-groq
python-dotenv
langgraph
```

---

## **4. Approach 1 — LangSmith Studio (Simplest)**

**Best for:** Visual graph debugging, step-through execution, inspecting node state, no Node.js required.

**How it works:** You run a local dev server, and the LangSmith Studio web app (hosted by LangChain) connects to your local machine via `baseUrl`.

---

### **Step 1 — Get a Free LangSmith API Key**

1. Go to [smith.langchain.com](https://smith.langchain.com) and sign up (free tier is sufficient).
2. Navigate to **Settings → API Keys → Create API Key**.
3. Copy the key — it starts with `lsv2_...`.

### **Step 2 — Add the Key to Your `.env`**

```env
LANGSMITH_API_KEY=lsv2_pt_xxxxxxxxxxxxxxxx
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=my-agent
```

### **Step 3 — Start the LangGraph Dev Server**

Run from your **root project folder** (not inside `my-chat-ui`):

```powershell
cd D:\Langchain-dev\langchain-deepdive
langgraph dev
```

Expected terminal output:

```
Welcome to
╦  ┌─┐┌┐┌┌─┐╔═╗┬─┐┌─┐┌─┐┬ ┬
║  ├─┤││││ ┬║ ╦├┬┘├─┤├─┘├─┤
╩═╝┴ ┴┘└┘└─┘╚═╝┴└─┴ ┴┴  ┴ ┴

Ready!
- API:              http://localhost:2024
- Docs:             http://localhost:2024/docs
- LangGraph Studio: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

### **Step 4 — Open Studio in Chrome**

Copy the Studio URL from your terminal and open it in **Chrome**:

```
https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

> **Use Chrome only.** Safari blocks localhost connections. If you must use Safari or Firefox, run with a tunnel instead:
>
> ```powershell
> langgraph dev --tunnel
> ```
> Then use the tunnel URL shown in the terminal.

### **What You Can Do in Studio**

| Feature | How |
|---|---|
| Chat with your agent | Type in the right panel input box |
| See graph visualization | Left panel shows nodes and edges |
| Step through execution | Click Interrupt → select nodes to pause on |
| Inspect node state | Click any node after a run to see inputs/outputs |
| Fork a thread | Hover a step → click pencil → edit → Fork |
| View traces | All runs auto-logged to your LangSmith project |

---

## 5. **Approach 2 — Agent Chat UI (localhost:5173)**

**Best for:** Clean minimal chat interface, no LangSmith account required, easy sharing with others.

**How it works:** `pnpm run dev` starts both the React frontend (port 5173) and the LangGraph server (port 2024) together from inside `my-chat-ui`.

---

### **Step 1 — Ensure Node.js and pnpm Are Installed**

```powershell
node --version    # must be v18+
npm --version
npm install -g pnpm
pnpm --version
```

### **Step 2 — Create the Chat App**

Run from your **root project folder**:

```powershell
cd D:\Langchain-dev\langchain-deepdive
npx create-agent-chat-app@latest --project-name my-chat-ui
```

Answer the prompts like this:

```
Which package manager?              → pnpm
Auto-install dependencies?          → Yes
Which framework?                    → Vite
Which pre-built agents to include?  → Press Space to DESELECT ALL, then Enter
```

> **Deselect ALL pre-built agents.** The ReAct, Research, Memory, and Retrieval agents all require `OPENAI_API_KEY` and `TAVILY_API_KEY`. Leaving them selected will crash the server on startup with API key errors.

### **Step 3 — Install Dependencies (If Auto-Install Failed)**

```powershell
cd my-chat-ui
pnpm install
```

### **Step 4 — Configure `my-chat-ui\langgraph.json`**

Open `D:\Langchain-dev\langchain-deepdive\my-chat-ui\langgraph.json` and replace ALL content with:

```json
{
  "dependencies": ["../../"],
  "graphs": {
    "agent": "../../examples/agent.py:agent"
  },
  "env": "../../.env"
}
```

> `../../` navigates two levels up from `my-chat-ui\` to your root project folder.

### **Step 5 — Start Everything**

Make sure you are inside `my-chat-ui`:

```powershell
cd D:\Langchain-dev\langchain-deepdive\my-chat-ui
pnpm run dev
```

Wait for both of these lines to appear:

```
[0] web:dev:   ➜  Local:   http://localhost:5173/
[1] agents:dev: info: ▪ Registering graph with id 'agent'
```

> The first time you run this, it will download `@langchain/langgraph-cli`. Wait 30–60 seconds for it to finish before opening the browser.

### **Step 6 — Connect to Your Agent**

Open `http://localhost:5173` in your browser. When prompted for connection details:

| Field | Value |
|---|---|
| **Deployment URL** | `http://localhost:2024` |
| **Graph ID** | `agent` |
| **LangSmith API Key** | leave blank |

Click **Connect** — the chat interface will appear.

### Finding Your Graph ID

Your Graph ID is the key name under `"graphs"` in `langgraph.json`:

```json
{
  "graphs": {
    "agent": "..."   ← "agent" is your Graph ID
  }
}
```

---

## **6. Approach 3 — Hosted Deployment**

**Best for:** Stable shared environment, teammates can test without your local machine running, production-like setup.

**How it works:** You push your project to GitHub, connect it to LangSmith Deployments, and Studio connects to the live deployed graph.

---

### **Step 1 — Prepare Your Repo**

Make sure your repository contains at minimum:

```
your-repo/
    examples/agent.py
    langgraph.json
    requirements.txt        ← or pyproject.toml
    .env                    ← do NOT commit this — add to .gitignore
```

Minimum `requirements.txt`:

```txt
langchain
langchain-groq
python-dotenv
langgraph
```

Ensure `langgraph.json` is at the project root and correctly points to your agent.

### **Step 2 — Push to GitHub**

```powershell
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### **Step 3 — Create a Deployment in LangSmith**

1. Go to [smith.langchain.com](https://smith.langchain.com) and sign in.
2. Navigate to **Deployments** in the left sidebar.
3. Click **New Deployment**.
4. Connect your GitHub repository.
5. Set the app path to the folder containing `langgraph.json` (usually the repo root).
6. Add your environment variables (API keys) in the deployment settings.
7. Click **Deploy** and wait for the build to complete.

### **Step 4 — Open Studio for Your Deployment**

Once the deployment shows a **Ready** status, click **Open Studio** from the deployment details page. Studio will connect directly to your live graph — no local server needed.

---

## **7. Common Mistakes — What NOT To Do**

### **Installing CLI Without `[inmem]`**

```powershell
# WRONG — this gives you an old version without the dev command
pip install langgraph-cli

# CORRECT
pip install -U "langgraph-cli[inmem]"
```

If you see only `build`, `dockerfile`, `new`, `up` in `langgraph --help` and no `dev`, this is the problem.

---

### **Opening `127.0.0.1:2024` Directly in the Browser**

`http://127.0.0.1:2024` is the **raw API endpoint**, not a UI. It only shows `{"ok":true}`.

The correct URLs are:
- Studio: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`
- Chat UI: `http://localhost:5173`

---

###** Mismatching the Graph Variable Name**

Your `langgraph.json` graph path must exactly match the variable name at the bottom of `agent.py`:

```python
# agent.py
my_graph = workflow.compile()  # variable is "my_graph"
```

```json
// langgraph.json — WRONG
{ "graphs": { "agent": "./examples/agent.py:agent" } }

// langgraph.json — CORRECT
{ "graphs": { "agent": "./examples/agent.py:my_graph" } }
```

---

### **Selecting Pre-Built Agents Without Their API Keys**

The ReAct, Research, Memory, and Retrieval agents that come with `create-agent-chat-app` all require:
- `OPENAI_API_KEY` (all agents)
- `TAVILY_API_KEY` (Research agent)

Selecting them without these keys crashes the server on startup. If you only have a Groq or Anthropic key, **deselect all pre-built agents** during setup.

---

### **Running Both Servers at the Same Time**

`pnpm run dev` (inside `my-chat-ui`) already starts its own LangGraph server on port 2024.  
Running `langgraph dev` separately at the same time causes a **port conflict**.

```powershell
# WRONG — do not do both at the same time
langgraph dev          # Terminal 1
pnpm run dev           # Terminal 2  ← port 2024 conflict!

# CORRECT — just run one
pnpm run dev           # handles both web UI and LangGraph server
```

---

### **Missing or Wrong `.env` File Path**

Without a `.env` file, the server throws `ENOENT: no such file or directory`.

The `.env` path in `langgraph.json` must be correct relative to where `langgraph.json` lives:

```json
// Root langgraph.json — .env is in the same folder
{ "env": "./.env" }

// my-chat-ui/langgraph.json — .env is two levels up
{ "env": "../../.env" }
```

---

### **Using Safari or Firefox with LangSmith Studio**

Safari and Firefox block localhost connections from remote pages. Studio will appear blank or fail to connect.

**Fix:** Use Chrome. Or run with a tunnel:

```powershell
langgraph dev --tunnel
# Then use the tunnel URL shown in the terminal, not the localhost URL
```

---

### **Using `pnpm` Before Installing It**

```powershell
# WRONG — pnpm not installed yet
pnpm install

# CORRECT — install pnpm first
npm install -g pnpm
pnpm install
```

---

### **Manually Adding `LANGSMITH_API_KEY` in the Desktop Studio App**

If you are using the old LangGraph Desktop Studio app, do **not** manually add `LANGSMITH_API_KEY` to the `.env`. The desktop app sets it automatically on login — adding it manually can cause auth conflicts.

This does **not** apply to the web-based Studio at `smith.langchain.com`.

---

### **Leaving the `env` Path Pointing to a Parent Folder That Doesn't Exist**

The original guide suggests `"env": "../.env"` (going one level above `langchain-deepdive`). Make sure that path actually exists on your machine. Use `"./.env"` if your `.env` is in the same folder as `langgraph.json`.

---

## **8. Troubleshooting Reference**

| Error / Symptom | Cause | Fix |
|---|---|---|
| `No such command 'dev'` | Old `langgraph-cli` without `[inmem]` | `pip install -U "langgraph-cli[inmem]"` |
| `pnpm is not recognized` | pnpm not installed | `npm install -g pnpm` |
| `{"ok":true}` in browser | Opened raw API URL | Go to `localhost:5173` or use Studio URL |
| `ENOENT: no such file .env` | `.env` missing or wrong path | Create `.env` at the path specified in `langgraph.json` |
| `No Tavily API key found` | Pre-built research agent selected without key | Remove pre-built agents from `langgraph.json` |
| `Cannot find graph 'agent'` | Variable name mismatch | Match graph variable in `agent.py` to `langgraph.json` |
| Studio shows blank page | Safari/Firefox blocks localhost | Use Chrome, or run `langgraph dev --tunnel` |
| Port 2024 already in use | Running both `langgraph dev` and `pnpm run dev` | Stop `langgraph dev` — `pnpm run dev` starts its own on port 2024 |
| LangSmith Studio won't connect | Missing `LANGSMITH_API_KEY` or server not running | Add key to `.env` and ensure `langgraph dev` is running |
| `ModuleNotFoundError: langchain` | Dependencies not installed in active environment | `pip install langchain langchain-groq python-dotenv` |
| Chat UI crashes on startup | Pre-built agents missing API keys | Deselect all pre-built agents in `my-chat-ui/langgraph.json` |
| `langgraph dev` exits instantly | Python syntax error in `agent.py` | Check `agent.py` for errors; run `python examples/agent.py` to test |

---

## **9. Quick Comparison Table**

| | Approach 1: LangSmith Studio | Approach 2: Agent Chat UI | Approach 3: Hosted |
|---|---|---|---|
| **URL** | `smith.langchain.com/studio/...` | `http://localhost:5173` | LangSmith Deployments |
| **Node.js required** | No | Yes (v18+) | No |
| **LangSmith account** | Yes (free) | No | Yes |
| **Docker required** | No | No | No |
| **Start command** | `langgraph dev` | `pnpm run dev` (inside `my-chat-ui`) | Push to GitHub → Deploy |
| **Local server running** | Yes (your machine) | Yes (your machine) | No (cloud) |
| **Best for** | Graph debugging, step-through, node inspection | Clean chat UI, no account needed | Shared/team environments |
| **Browser** | Chrome only | Any | Any |

---

## **Quick Test Checklist**

After setup, verify everything is working by sending:

```
what is the weather in sf
```

Expected behavior:
1. Agent makes a tool call to `get_weather`
2. Final assistant response returns the weather text

If both steps happen, your setup is complete.

---

*Guide covers: `langgraph-cli[inmem]` · `create-agent-chat-app v0.1.6+` · Node.js v18+ · LangSmith Studio Web UI · Python 3.11+*