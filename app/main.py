"""FastAPI backend wiring the search pipeline together.

Pipeline for a query:
    1. wikipedia_client.get_candidate_articles  -> fetch relevant articles
    2. search.PassageIndex                       -> BM25-rank their passages
    3. rag.generate_answer                       -> grounded answer via Ollama

Endpoints:
    GET  /                -> the web UI (static/index.html)
    POST /api/search      -> BM25 results only (fast, no LLM)
    POST /api/ask         -> BM25 retrieval + RAG answer (the full experience)
    GET  /api/health      -> dependency status (Ollama reachable, model pulled)
"""
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, rag, search, wikipedia_client

app = FastAPI(title="Wikipedia Search Engine (BM25 + RAG)")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class Query(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)


def _retrieve(query: str):
    """Shared retrieval stage: fetch candidate articles, BM25-rank passages."""
    articles = wikipedia_client.get_candidate_articles(query)
    index = search.PassageIndex(articles)
    passages = index.search(query, config.TOP_PASSAGES)
    return articles, passages


@app.post("/api/search")
def api_search(q: Query):
    """Return BM25-ranked passages without invoking the LLM (fast path)."""
    started = time.perf_counter()
    articles, passages = _retrieve(q.query)
    return {
        "query": q.query,
        "articles_considered": len(articles),
        "results": passages,
        "took_ms": round((time.perf_counter() - started) * 1000),
    }


@app.post("/api/ask")
def api_ask(q: Query):
    """Full RAG: retrieve passages, then generate a grounded, cited answer."""
    started = time.perf_counter()
    articles, passages = _retrieve(q.query)
    generation = rag.generate_answer(q.query, passages)
    return {
        "query": q.query,
        "answer": generation["answer"],
        "error": generation["error"],
        "articles_considered": len(articles),
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
    }


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


# Serve CSS/JS (mounted last so it doesn't shadow the API routes).
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
