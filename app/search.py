"""BM25 retrieval over Wikipedia article passages.

BM25 (Best Matching 25) is the classic bag-of-words ranking function used by real
search engines and libraries like Elasticsearch/Lucene. Given a query it scores
each document by how well its term frequencies match the query terms, with two
key refinements over plain TF-IDF:

  * term-frequency saturation (k1): the 10th occurrence of a word adds less than
    the 2nd, so keyword stuffing doesn't dominate.
  * length normalization (b): long documents don't win just for being long.

Here a "document" is a ~120-word passage carved out of an article, so we can
pinpoint the most relevant *part* of an article, not just the article itself.
"""
import re
from typing import List, Dict

import numpy as np
from rank_bm25 import BM25Okapi

from . import config, embeddings


_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Very small English stopword list — common words that carry little signal and
# would otherwise inflate scores for any document containing them.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "is", "are", "was", "were", "be", "been", "being", "as", "by", "with",
    "that", "this", "these", "those", "it", "its", "from", "into", "about",
    "what", "which", "who", "whom", "how", "when", "where", "why", "do", "does",
}


def tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _split_into_passages(article: Dict[str, str]) -> List[Dict[str, str]]:
    """Break one article into word-bounded passages.

    We split on blank lines first (Wikipedia extracts use them between
    paragraphs), then pack paragraphs into chunks of ~PASSAGE_WORDS words so each
    passage is a coherent, citable unit.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", article["text"]) if p.strip()]
    passages: List[Dict[str, str]] = []

    buffer: List[str] = []
    word_count = 0
    for para in paragraphs:
        words = para.split()
        if not words:
            continue
        buffer.append(para)
        word_count += len(words)
        if word_count >= config.PASSAGE_WORDS:
            passages.append(_make_passage(article, buffer, len(passages)))
            buffer, word_count = [], 0

    if buffer:
        passages.append(_make_passage(article, buffer, len(passages)))
    # Keep only the lead + earliest sections, where the answer almost always is.
    return passages[: config.MAX_PASSAGES_PER_ARTICLE]


def _make_passage(article: Dict[str, str], paragraphs: List[str], pos: int) -> Dict:
    return {
        "title": article["title"],
        "url": article["url"],
        "text": " ".join(paragraphs),
        "pos": pos,  # 0 == the article's lead/summary section
    }


def _rrf(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion weight: high for rank 0, decaying smoothly."""
    return 1.0 / (k + rank)


class PassageIndex:
    """A hybrid (lexical + semantic) index built fresh for each query's articles.

    Ranking combines two complementary signals via Reciprocal Rank Fusion:
      * BM25         — exact term matching (great for names, numbers, rare words)
      * embeddings   — semantic similarity (great for paraphrased questions)
    plus a small boost for each article's lead section, where Wikipedia states
    the most direct answer. If embeddings are unavailable, it degrades to
    lexical-only ranking automatically.

    Because we only rank a handful of articles per query, building the index on
    the fly is cheap and keeps the engine stateless — no database required.
    """

    def __init__(self, articles: List[Dict[str, str]]):
        self.passages: List[Dict] = []
        for article in articles:
            self.passages.extend(_split_into_passages(article))

        # Guard against an empty corpus (BM25Okapi can't handle zero docs).
        tokenized = [tokenize(p["text"]) or ["__empty__"] for p in self.passages]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def _ranks_from_scores(self, scores) -> Dict[int, int]:
        """Map passage index -> its rank (0 = best) for a score array."""
        order = sorted(range(len(self.passages)), key=lambda i: scores[i], reverse=True)
        return {idx: rank for rank, idx in enumerate(order)}

    def search(self, query: str, top_k: int) -> List[Dict]:
        """Return the top_k passages, best first, using hybrid ranking."""
        if not self._bm25 or not self.passages:
            return []

        query_tokens = tokenize(query) or tokenize(query.lower())
        if not query_tokens:
            return []

        bm25_scores = self._bm25.get_scores(query_tokens)
        bm25_ranks = self._ranks_from_scores(bm25_scores)

        # Optional semantic signal.
        sem_ranks: Dict[int, int] = {}
        sem_scores = None
        if embeddings.available():
            qv = embeddings.embed_query(query)
            # Guard against blank passages, whose embeddings can be degenerate.
            texts = [p["text"] if p["text"].strip() else p["title"] for p in self.passages]
            pv = embeddings.embed_passages(texts)
            if qv is not None and pv is not None:
                # Scrub any non-finite values from degenerate vectors BEFORE the
                # multiply, so the matmul itself stays clean. bge embeddings are
                # unit-normalized, so the dot product is cosine similarity.
                qv = np.nan_to_num(qv, nan=0.0, posinf=0.0, neginf=0.0)
                pv = np.nan_to_num(pv, nan=0.0, posinf=0.0, neginf=0.0)
                # errstate silences a spurious numpy/Apple-silicon BLAS warning;
                # the inputs are already scrubbed, so the result is valid.
                with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                    sem_scores = pv @ qv
                sem_ranks = self._ranks_from_scores(sem_scores)

        # Fuse the rankings. With no semantic signal this is pure BM25 + lead boost.
        fused: List[tuple] = []
        for i, passage in enumerate(self.passages):
            score = _rrf(bm25_ranks[i])
            if sem_ranks:
                score += _rrf(sem_ranks[i])
            if passage["pos"] == 0:
                score += config.LEAD_BOOST
            fused.append((score, i))

        fused.sort(key=lambda x: x[0], reverse=True)

        results: List[Dict] = []
        for rank, (score, idx) in enumerate(fused[:top_k], start=1):
            # Drop passages that match on neither signal at all.
            if bm25_scores[idx] <= 0 and (sem_scores is None or sem_scores[idx] <= 0.2):
                continue
            passage = self.passages[idx]
            results.append(
                {
                    "rank": rank,
                    "score": round(float(score), 4),
                    "relevance": round(float(sem_scores[idx]), 3) if sem_scores is not None else None,
                    "title": passage["title"],
                    "url": passage["url"],
                    "text": passage["text"],
                }
            )
        return results
