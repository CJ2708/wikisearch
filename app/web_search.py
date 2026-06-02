"""Open-web search source, powered by Tavily.

This is the web counterpart to wikipedia_client. Instead of querying Wikipedia,
it asks Tavily (a search API built for RAG) to find relevant pages AND return
their cleaned text content in a single call — so we don't have to crawl or scrape
HTML ourselves. Tavily's servers do the fetching, which also sidesteps the
rate-limiting that hits direct requests from shared cloud IPs.

It returns the SAME shape as wikipedia_client.get_candidate_articles
({"title", "url", "text"} dicts), so the rest of the pipeline (passage splitting,
hybrid ranking, reranking, RAG) works unchanged regardless of the source.
"""
from typing import List, Dict

import httpx

from . import config


def available() -> bool:
    """Web search is usable only if a Tavily API key is configured."""
    return bool(config.TAVILY_API_KEY)


def get_candidate_documents(query: str) -> List[Dict[str, str]]:
    """Search the web and return candidate documents with their text content.

    Returns a list of {"title", "url", "text"} dicts, best-first. On any failure
    (missing key, network, API error) returns [] so the caller degrades
    gracefully instead of crashing.
    """
    if not config.TAVILY_API_KEY:
        return []

    payload = {
        "api_key": config.TAVILY_API_KEY,
        "query": query,
        "search_depth": config.TAVILY_SEARCH_DEPTH,
        "max_results": config.TAVILY_MAX_RESULTS,
        "include_raw_content": True,   # full page text, not just a snippet
        "include_answer": False,
    }

    try:
        resp = httpx.post(config.TAVILY_URL, json=payload, timeout=30.0)
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except (httpx.HTTPError, ValueError):
        return []

    documents: List[Dict[str, str]] = []
    for r in results:
        # Prefer the full extracted page text; fall back to the snippet.
        text = (r.get("raw_content") or r.get("content") or "").strip()
        if not text:
            continue
        documents.append(
            {
                "title": r.get("title") or r.get("url", "Untitled"),
                "url": r.get("url", ""),
                "text": text,
            }
        )
    return documents
