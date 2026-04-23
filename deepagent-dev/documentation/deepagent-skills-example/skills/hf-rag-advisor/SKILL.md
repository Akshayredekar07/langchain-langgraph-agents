---
name: hf-rag-advisor
description: >
  Use this skill when the user asks about building, optimizing, or debugging a
  RAG (Retrieval-Augmented Generation) pipeline using HuggingFace models. Covers
  chunking strategy, embedding selection, vector store choice, reranking, and
  LLM integration. Trigger on: "RAG", "retrieval", "vector store", "embeddings",
  "chunking", "semantic search pipeline", "reranker", "context window".
license: MIT
metadata:
  author: akshay
  version: "1.0"
allowed-tools: model_search, hub_repository_details, papers_semantic_search
---

# HuggingFace RAG Advisor Skill

## Overview
A structured workflow to recommend and implement a complete RAG pipeline
using HuggingFace models. Each stage has a recommended model search query
and integration code pattern.

---

## RAG Pipeline Stages

```
Documents → Chunker → Embedder → Vector Store
                                      ↓
User Query → Query Embedder → Retriever → Reranker → LLM → Answer
```

---

## Stage 1: Chunking Strategy

Before searching for models, determine chunking approach:

| Strategy | When to use | Chunk size |
|---|---|---|
| Fixed-size | Uniform docs (PDFs, articles) | 512 tokens, 50 overlap |
| Semantic | Mixed content, QA pairs | 256–512 tokens |
| Hierarchical | Long docs with structure | Parent: 1024, Child: 256 |

```python
# Standard fixed-size chunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "],
)
chunks = splitter.split_documents(docs)
```

---

## Stage 2: Embedding Model Selection

Call `model_search("sentence transformers embedding retrieval")` and filter by:
- **Dimension**: 384 (fast/small) → 768 (balanced) → 1536 (best quality)
- **Max tokens**: must be ≥ chunk_size. Prefer 512+.
- **License**: apache-2.0 or mit for production

**Recommended stack by use case:**

| Use case | Model | Dim | Notes |
|---|---|---|---|
| Fast / low-resource | `all-MiniLM-L6-v2` | 384 | 80M params, great default |
| Balanced production | `all-mpnet-base-v2` | 768 | Higher quality |
| Best quality | `bge-large-en-v1.5` | 1024 | BAAI, top MTEB |
| Multilingual | `paraphrase-multilingual-mpnet-base-v2` | 768 | 50+ languages |

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},  # cosine similarity
)
```

---

## Stage 3: Vector Store Recommendation

| Store | Best for | Notes |
|---|---|---|
| Chroma | Local dev, prototyping | Zero config |
| FAISS | Local, large-scale batch | No persistence out-of-box |
| PGVector | Production, SQL integration | Needs Postgres |
| Qdrant | Production, cloud-native | Best filtering support |
| Pinecone | Managed cloud | Highest ops simplicity |

```python
# PGVector (production)
from langchain_community.vectorstores import PGVector

vectorstore = PGVector(
    embeddings=embeddings,
    collection_name="rag_docs",
    connection=CONNECTION_STRING,
    use_jsonb=True,
)
```

---

## Stage 4: Retriever + Reranker

After retrieval, always apply a reranker for precision:

```python
# Step 1: Retrieve top-k candidates (k=20, broader net)
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Step 2: Rerank with cross-encoder (returns top-n=5)
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

reranker_model = HuggingFaceCrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)
compressor = CrossEncoderReranker(model=reranker_model, top_n=5)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever,
)
```

Call `model_search("cross encoder reranker ms-marco")` to find latest reranker versions.

---

## Stage 5: LLM Integration

Search for generator: `model_search("instruction following text generation function calling")`

```python
# Using HF Inference API (no GPU needed)
from langchain_huggingface import HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1,
)

# LCEL RAG chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

rag_chain = (
    {"context": compression_retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("Your question here")
```

---

## Evaluation

After building, always evaluate with RAGAs:

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

# Target benchmarks for production
# faithfulness       > 0.85
# answer_relevancy   > 0.80
# context_recall     > 0.75
```

Search for latest RAGAs-compatible models: `papers_semantic_search("RAG evaluation faithfulness 2025")`