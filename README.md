# WikiSearch — a BM25 + RAG search engine over Wikipedia

A personal search engine that answers your questions using Wikipedia. You type a
query; it finds relevant Wikipedia articles, ranks their passages with **BM25**
(the classic search-engine ranking algorithm), and then a **local LLM (via
Ollama)** writes a grounded, cited answer — Retrieval-Augmented Generation (RAG).

No API keys, no cloud — everything runs on your machine except the read-only
calls to Wikipedia's public API.

## How it works

```
  your query
      │
      ▼
┌─────────────────────┐   1. Find candidate articles via the Wikipedia search API
│  wikipedia_client   │      and download their plain-text content.
└─────────────────────┘
      │  articles
      ▼
┌─────────────────────┐   2. Split articles into ~120-word passages and rank them
│   search (BM25)     │      with BM25 (term-frequency saturation + length norm).
└─────────────────────┘
      │  top passages
      ▼
┌─────────────────────┐   3. Feed the top passages to a local Ollama model and ask
│   rag (Ollama)      │      it to answer using ONLY those sources, with [n] cites.
└─────────────────────┘
      │  grounded answer + sources
      ▼
   web UI  (static/)
```

| Layer       | Tech                                                   |
|-------------|--------------------------------------------------------|
| Backend     | Python + FastAPI                                       |
| Retrieval   | Multi-query expansion + `rank-bm25` over passages      |
| Generation  | Pluggable: **Ollama** (local) or **Groq** (cloud)      |
| Frontend    | Vanilla HTML/CSS/JS (no build step)                    |

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

| Variable             | Default                   | Meaning                                          |
|----------------------|---------------------------|--------------------------------------------------|
| `LLM_PROVIDER`       | `auto`                    | `ollama`, `groq`, or `auto`.                     |
| `GROQ_API_KEY`       | _(empty)_                 | Your Groq key (enables the cloud backend).       |
| `GROQ_MODEL`         | `llama-3.3-70b-versatile` | Which Groq model generates answers.              |
| `OLLAMA_MODEL`       | `llama3.2`                | Which Ollama model generates answers.            |
| `CANDIDATE_ARTICLES` | `8`                       | Articles pulled from Wikipedia per query.        |
| `PASSAGE_WORDS`      | `120`                     | Approx. words per rankable passage.              |
| `TOP_PASSAGES`       | `5`                       | Passages kept as context / shown as sources.     |
| `WIKI_LANG`          | `en`                      | Wikipedia language edition.                      |

For local development you can put these in a `.env` file (see `.env.example`).

## Deploy it live (free)

GitHub stores the code; a host runs it. The retrieval (Wikipedia + BM25) runs
anywhere, but a local Ollama model can't be hosted cheaply — so the live version
uses the **Groq** backend instead.

**1. Get a free Groq API key** at <https://console.groq.com/keys>.

**2. Push this repo to GitHub** (see below).

**3. Deploy on [Render](https://render.com) (easiest):**
   - New → **Blueprint** → pick your GitHub repo. Render reads
     [`render.yaml`](render.yaml) and provisions the service.
   - In the service's **Environment** tab, set `GROQ_API_KEY` to your key.
   - Open the generated `*.onrender.com` URL — it's live.

   _(The included [`Dockerfile`](Dockerfile) also works on Hugging Face Spaces,
   Fly.io, or Railway if you prefer those.)_

## Ideas to extend it

- Add **semantic re-ranking** (embeddings) on top of BM25 for a hybrid retriever.
- Cache fetched articles so repeat queries are instant.
- Stream the LLM answer token-by-token to the UI.
- Swap Wikipedia for your own document corpus to make it a private knowledge search.
