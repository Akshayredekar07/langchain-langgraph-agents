---
name: hf-hub-search
description: >
  Use this skill for ANY request involving HuggingFace Hub: searching models,
  datasets, papers, Spaces, or getting repository details. Covers model
  comparison, filtering by task/license/language, and summarizing model cards.
  Trigger on keywords: "find model", "best model for", "HuggingFace", "HF Hub",
  "compare models", "embedding model", "text-generation", "search papers".
license: MIT
metadata:
  author: akshay
  version: "1.0"
  mcp_server: "https://huggingface.co/mcp"
allowed-tools: model_search, dataset_search, spaces_semantic_search, papers_semantic_search, documentation_search, hub_repository_details
---

# HuggingFace Hub Search Skill

## Overview
This skill guides you on using the HuggingFace MCP tools effectively to find
and evaluate models, datasets, papers, and Spaces from the Hub.

---

## Available MCP Tools (via HF MCP Server)

| Tool | Input | Best For |
|---|---|---|
| `model_search` | query string | Find models by task, keyword, or architecture |
| `dataset_search` | query string | Find datasets by domain or task |
| `spaces_semantic_search` | query string | Find Gradio demo apps / Spaces |
| `papers_semantic_search` | query string | Find ML papers (arXiv-linked) |
| `documentation_search` | query string | Search HF docs (transformers, diffusers, etc.) |
| `hub_repository_details` | repo_id | Get full model card, metadata, tags, downloads |

---

## Step-by-Step Instructions

### 1. Model Search Workflow

When the user asks for models for a specific task:

```
Step 1: Call model_search with a precise query
        → e.g., "sentence transformers for semantic similarity"
        → e.g., "text generation instruct models under 7B"

Step 2: From results, extract top 3-5 candidates

Step 3: For each candidate, call hub_repository_details to get:
        - Parameter count
        - License (apache-2.0, mit, llama, etc.)
        - Downloads / likes (popularity signal)
        - Inference API availability
        - Max context length / embedding dimension

Step 4: Present comparison as a markdown table
```

### 2. RAG Model Selection Pattern

When user asks for models for a RAG pipeline, always cover:
- **Embedding model** → `model_search("sentence transformers embedding")`
- **Reranker** → `model_search("cross-encoder reranker ms-marco")`
- **Generator/LLM** → `model_search("instruction following text generation")`

Then present as a 3-tier RAG stack recommendation.

### 3. Paper Research Workflow

```
Step 1: papers_semantic_search with the topic
Step 2: Extract paper titles, arXiv IDs, and abstracts
Step 3: Summarize key contributions as bullet points
Step 4: Map papers to available HF model implementations (use model_search)
```

### 4. Output Format

Always structure model recommendations like this:

```markdown
## Model: `org/model-name`
- **Task**: text-classification
- **Parameters**: 110M
- **Embedding dim**: 768
- **Max tokens**: 512
- **License**: Apache 2.0
- **Inference API**: ✅ Available
- **Downloads**: 5.2M/month
- **Best for**: [one-line use case]

\```python
# Quick integration snippet
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("org/model-name")
embeddings = model.encode(["your text here"])
\```
```

---

## Common Query Patterns

| User asks | Query to use |
|---|---|
| Best embedding model for RAG | `"sentence transformers semantic search retrieval"` |
| Fast small LLM | `"small fast instruct language model 1B 3B"` |
| Multilingual model | `"multilingual text embedding"` |
| Latest RAG papers | `"retrieval augmented generation 2024 2025"` |
| Vision-language models | `"vision language model multimodal"` |

---

## Notes
- Always check `hub_repository_details` before recommending — download counts and
  Inference API availability change frequently.
- For proprietary models (llama, gemma, etc.), note if gating/access request is needed.
- Prefer models with `transformers` tag for easy HuggingFace ecosystem integration.