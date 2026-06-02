"""FastAPI backend wiring the search pipeline together.

Pipeline for a query:
    1. fetch candidate documents  -> web (Tavily) OR Wikipedia, per `source`
    2. search.PassageIndex        -> hybrid (BM25 + embeddings) + cross-encoder rerank
    3. rag.generate_answer        -> grounded, cited answer via Ollama or Groq

Endpoints:
    GET  /                -> the web UI (static/index.html)
    POST /api/search      -> ranked passages only (fast, no LLM)
    POST /api/ask         -> retrieval + RAG answer (the full experience)
    GET  /api/health      -> dependency status (LLM, embeddings, sources)
"""
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, embeddings, rag, reranker, search, web_search, wikipedia_client

app = FastAPI(title="Search Engine (Web + Wikipedia, Hybrid Retrieval + RAG)")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@app.on_event("startup")
def _warm_up_models() -> None:
    """Load the ML models at startup so the first query isn't slow."""
    embeddings.warm_up()
    reranker.warm_up()


class Query(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    # "web" (Tavily) or "wikipedia"; falls back to the configured default.
    source: Optional[str] = None


def _resolve_source(requested: Optional[str]) -> str:
    """Pick the source: honor the request, else default; fall back if unusable."""
    source = (requested or config.SEARCH_SOURCE).lower()
    if source == "web" and not web_search.available():
        return "wikipedia"   # no Tavily key configured -> use Wikipedia
    return source


def _fetch_documents(query: str, source: str):
    if source == "wikipedia":
        return wikipedia_client.get_candidate_articles(query)
    return web_search.get_candidate_documents(query)


def _retrieve(query: str, source: str):
    """Shared retrieval: fetch candidate docs, then hybrid-rank + rerank passages."""
    documents = _fetch_documents(query, source)
    index = search.PassageIndex(documents)
    passages = index.search(query, config.TOP_PASSAGES)
    return documents, passages


@app.post("/api/search")
def api_search(q: Query):
    """Return ranked passages without invoking the LLM (fast path)."""
    started = time.perf_counter()
    source = _resolve_source(q.source)
    documents, passages = _retrieve(q.query, source)
    return {
        "query": q.query,
        "source": source,
        "articles_considered": len(documents),
        "results": passages,
        "took_ms": round((time.perf_counter() - started) * 1000),
    }


@app.post("/api/ask")
def api_ask(q: Query):
    """Full RAG: retrieve passages, then generate a grounded, cited answer."""
    started = time.perf_counter()
    source = _resolve_source(q.source)
    documents, passages = _retrieve(q.query, source)
    generation = rag.generate_answer(q.query, passages)
    return {
        "query": q.query,
        "source": source,
        "answer": generation["answer"],
        "error": generation["error"],
        "articles_considered": len(documents),
        "sources": passages,
        "model": rag.active_model_name(),
        "took_ms": round((time.perf_counter() - started) * 1000),
    }


@app.get("/api/health")
def api_health():
    return {
        "status": "ok",
        "provider": config.active_provider(),
        "model": rag.active_model_name(),
        "llm_ready": rag.model_available(),
        "semantic_ranking": embeddings.available(),
        "reranker": reranker.available(),
        "web_search": web_search.available(),
        "default_source": _resolve_source(None),
    }


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


# Serve CSS/JS (mounted last so it doesn't shadow the API routes).
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
