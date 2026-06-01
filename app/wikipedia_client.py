"""Fetches candidate Wikipedia articles for a query.

This is the "crawl" stage of the engine, but instead of crawling the open web we
lean on Wikipedia's own search API to find a handful of relevant articles, then
download their plain-text content. BM25 (in search.py) does the real ranking
afterwards over the passages of these articles.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional

import httpx

from . import config
from .search import tokenize


def _client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": config.USER_AGENT},
        timeout=20.0,
    )


def _api_get(client: httpx.Client, params: dict) -> Optional[dict]:
    """GET the Wikipedia API with retry/backoff; return JSON or None.

    Wikipedia rate-limits anonymous traffic from shared/datacenter IPs (common on
    cloud hosts), returning 429. We retry with exponential backoff, respecting any
    Retry-After header, and on persistent failure return None so the caller can
    degrade gracefully instead of crashing the request.
    """
    delay = 0.5
    for attempt in range(config.WIKI_MAX_RETRIES):
        try:
            resp = client.get(config.WIKI_API_URL, params=params)
            if resp.status_code in (429, 503):
                wait = float(resp.headers.get("Retry-After", delay))
                time.sleep(min(wait, 5.0))
                delay *= 2
                continue
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, ValueError):
            time.sleep(delay)
            delay *= 2
    return None


def search_titles(query: str, limit: int) -> List[str]:
    """Ask Wikipedia which article titles best match the query.

    Returns a list of page titles (most relevant first), or [] on failure.
    """
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "srprop": "",          # we only need titles here
        "format": "json",
    }
    with _client() as client:
        data = _api_get(client, params)
    if not data:
        return []
    hits = data.get("query", {}).get("search", [])
    return [hit["title"] for hit in hits]


def _fetch_one_extract(client: httpx.Client, title: str) -> Optional[Dict[str, str]]:
    """Download the plain-text extract for a single title.

    We fetch one title per request on purpose: Wikipedia's TextExtracts API
    truncates batched multi-title requests, returning full text for only the
    first page and empty extracts for the rest. One-per-request is reliable.
    """
    params = {
        "action": "query",
        "prop": "extracts|info",
        "explaintext": "1",     # plain text, no HTML
        "exsectionformat": "plain",
        "inprop": "url",
        "titles": title,
        "redirects": "1",       # follow redirects to the canonical article
        "format": "json",
    }
    data = _api_get(client, params)
    if not data:
        return None
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        text = (page.get("extract") or "").strip()
        if not text:
            return None
        return {
            "title": page.get("title", ""),
            "url": page.get("fullurl", ""),
            "text": text,
        }
    return None


def fetch_extracts(titles: List[str]) -> List[Dict[str, str]]:
    """Download the plain-text extract for each title, concurrently.

    Returns a list of {"title", "url", "text"} dicts in the same relevance order
    as ``titles``. Articles that come back empty (e.g. disambiguation pages) are
    skipped.
    """
    if not titles:
        return []

    # Cap concurrency to stay polite: too many simultaneous requests from one IP
    # trips Wikipedia's rate limiter (429), especially on shared cloud hosts.
    workers = min(config.WIKI_MAX_CONCURRENCY, len(titles))
    with _client() as client:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fetched = list(pool.map(lambda t: _fetch_one_extract(client, t), titles))

    # Keep input (relevance) order and drop the empties.
    return [article for article in fetched if article]


def _expand_queries(query: str) -> List[str]:
    """Turn one user question into several search queries (query expansion).

    A verbose natural-language question ("How does the BM25 algorithm work?")
    often matches few/poor articles when sent verbatim, because Wikipedia weighs
    every filler word. Distinctive single terms ("bm25"), on the other hand,
    surface the exact article. So we search for several variants and fuse them.
    """
    queries: List[str] = [query]                       # the original question
    keywords = [t for t in tokenize(query) if len(t) >= 3]
    if keywords:
        queries.append(" ".join(keywords))             # keyword-only version
    # Identifier-like tokens (containing a digit, e.g. "bm25", "covid19", "gpt4")
    # are highly distinctive topic names — search each on its own. We avoid
    # plain single words here because they pull in generic, off-topic articles.
    distinctive = [t for t in dict.fromkeys(keywords) if any(c.isdigit() for c in t)]
    queries.extend(distinctive[:2])
    # De-duplicate while preserving order, cap the number of API round-trips.
    seen, out = set(), []
    for q in queries:
        q = q.strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out[:5]


def _gather_titles(query: str) -> List[str]:
    """Search several query variants in parallel and fuse the rankings.

    Fusion rule: a title's score rewards (a) appearing in more variant result
    sets and (b) ranking highly within each. This Reciprocal-Rank-Fusion-style
    merge reliably promotes the genuinely relevant articles to the top.
    """
    limit = config.CANDIDATE_ARTICLES
    variants = _expand_queries(query)

    workers = min(config.WIKI_MAX_CONCURRENCY, len(variants))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        result_sets = list(pool.map(lambda q: search_titles(q, limit), variants))

    scores: Dict[str, float] = {}
    first_seen: Dict[str, int] = {}
    order = 0
    for titles in result_sets:
        for rank, title in enumerate(titles):
            scores[title] = scores.get(title, 0.0) + 1.0 / (rank + 1)  # RRF-style
            if title not in first_seen:
                first_seen[title] = order
                order += 1

    ranked = sorted(scores, key=lambda t: (-scores[t], first_seen[t]))
    return ranked[:limit]


def get_candidate_articles(query: str) -> List[Dict[str, str]]:
    """End-to-end: query -> relevant Wikipedia article texts."""
    titles = _gather_titles(query)
    return fetch_extracts(titles)
