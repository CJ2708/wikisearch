---
title: WikiSearch
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8666
pinned: false
---

<!-- The block above is read by Hugging Face Spaces when deploying there; it is
     ignored when running locally or on other hosts. -->

# WikiSearch — a hybrid search engine + RAG over Wikipedia

A personal search engine that answers your questions using Wikipedia. You type a
query; it finds relevant Wikipedia articles, ranks their passages with a **hybrid
of BM25 (lexical) and semantic embeddings (meaning)**, and then an **LLM (via
Ollama or Groq)** writes a grounded, cited answer — Retrieval-Augmented
Generation (RAG).

## How it works

```
  your query
      │
      ▼
┌─────────────────────┐   1. Expand the query into several searches, fetch the best
│  wikipedia_client   │      candidate articles via the Wikipedia API, fuse results.
└─────────────────────┘
      │  articles
      ▼
┌─────────────────────┐   2. Split into passages and rank them with a HYBRID of BM25
│   search (hybrid)   │      (lexical) + embedding similarity (semantic), fused by RRF.
└─────────────────────┘
      │  top passages
      ▼
┌─────────────────────┐   3. Feed the top passages to the LLM (Ollama or Groq) and ask
│   rag (LLM)         │      it to answer using ONLY those sources, with [n] cites.
└─────────────────────┘
      │  grounded answer + sources
      ▼
   web UI  (static/)
```

| Layer       | Tech                                                       |
|-------------|------------------------------------------------------------|
| Backend     | Python + FastAPI                                           |
| Retrieval   | Multi-query expansion + **hybrid BM25 + embeddings (RRF)** |
| Embeddings  | `fastembed` (ONNX, `BAAI/bge-small-en-v1.5`) — no PyTorch  |
| Generation  | Pluggable: **Ollama** (local) or **Groq** (cloud)         |
| Frontend    | Vanilla HTML/CSS/JS (no build step)                        |

### Pluggable LLM backend

The answer-generation model is swappable via the `LLM_PROVIDER` env var:

* `ollama` — a model on your own machine. Great for development; no API key.
* `groq` — hosted open models (Llama/Qwen) via [Groq](https://console.groq.com)'s
  free API. Great for deployment, where running a local model isn't practical.
* `auto` (default) — use Groq if `GROQ_API_KEY` is set, else fall back to Ollama.

The retrieval half (Wikipedia + BM25) is identical no matter which backend runs.

## Prerequisites

- Python 3.9+
- An LLM backend — **either**:
  - **Local:** [Ollama](https://ollama.com) installed with a model pulled
    (`ollama pull llama3.2`), **or**
  - **Cloud:** a free [Groq API key](https://console.groq.com/keys) (recommended
    for deployment, and gives much stronger answers via a 70B model).

## Setup

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

## Run

```bash
./run.sh
```

Then open <http://localhost:8666>. Toggle "Generate an AI answer (RAG)" off if you
only want the raw BM25-ranked passages (faster, no LLM).

## API

| Method | Path           | Description                                   |
|--------|----------------|-----------------------------------------------|
| POST   | `/api/search`  | BM25-ranked passages only (no LLM).           |
| POST   | `/api/ask`     | Full RAG: retrieval + grounded cited answer.  |
| GET    | `/api/health`  | Reports whether Ollama + the model are ready. |

Example:

```bash
curl -s -X POST http://localhost:8666/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"query": "How does the BM25 ranking algorithm work?"}' | python3 -m json.tool
```

## Configuration

All knobs live in [app/config.py](app/config.py) and can be overridden with
environment variables, e.g.:

```bash
OLLAMA_MODEL=llama3.1:8b CANDIDATE_ARTICLES=8 TOP_PASSAGES=6 ./run.sh
```

| Variable                   | Default                   | Meaning                                          |
|----------------------------|---------------------------|--------------------------------------------------|
| `LLM_PROVIDER`             | `auto`                    | `ollama`, `groq`, or `auto`.                     |
| `GROQ_API_KEY`             | _(empty)_                 | Your Groq key (enables the cloud backend).       |
| `GROQ_MODEL`               | `llama-3.3-70b-versatile` | Which Groq model generates answers.              |
| `OLLAMA_MODEL`             | `qwen2.5:7b`              | Which Ollama model generates answers.            |
| `USE_EMBEDDINGS`           | `true`                    | Hybrid semantic ranking (needs ~400 MB RAM).     |
| `EMBED_MODEL`              | `BAAI/bge-small-en-v1.5`  | The embedding model used for semantic ranking.   |
| `CANDIDATE_ARTICLES`       | `8`                       | Articles pulled from Wikipedia per query.        |
| `MAX_PASSAGES_PER_ARTICLE` | `10`                      | Caps chunks from huge articles (keeps it fast).  |
| `TOP_PASSAGES`             | `6`                       | Passages kept as context / shown as sources.     |
| `WIKI_LANG`                | `en`                      | Wikipedia language edition.                      |

For local development you can put these in a `.env` file (see `.env.example`).

## Deploy it live (free)

GitHub stores the code; a host runs it. A local Ollama model can't be hosted
cheaply, so the live version uses the **Groq** backend. Get a free key first at
<https://console.groq.com/keys>, then push this repo to GitHub.

### Recommended: Hugging Face Spaces (16 GB RAM free)

Because the semantic ranking loads an embedding model (~400 MB RAM), the best
free host is **[Hugging Face Spaces](https://huggingface.co/spaces)** — generous
RAM and built for ML demos. The repo already includes the Space config (the YAML
header in this README + the [`Dockerfile`](Dockerfile)).

1. Create a **New Space** → **Docker** (blank).
2. Push this repo to the Space (or link the GitHub repo).
3. In **Settings → Variables and secrets**, add `GROQ_API_KEY` (and
   `LLM_PROVIDER=groq`). Open the Space URL — full hybrid quality, live.

### Alternative: Render (free tier is only 512 MB RAM)

Render's free instance is too small for the embedding model, so semantic ranking
is disabled there (it falls back to lexical-only). The included
[`render.yaml`](render.yaml) sets `USE_EMBEDDINGS=false` to keep it from
crashing. New → **Blueprint** → pick the repo, set `GROQ_API_KEY` in the
**Environment** tab. For full quality on Render, use a paid instance and set
`USE_EMBEDDINGS=true`.

## Ideas to extend it

- Cache fetched articles + their embeddings so repeat queries are instant.
- Add a cross-encoder re-ranker for an even sharper top result.
- Stream the LLM answer token-by-token to the UI.
- Swap Wikipedia for your own document corpus to make it a private knowledge search.
