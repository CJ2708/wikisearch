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

from rank_bm25 import BM25Okapi

from . import config


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
            passages.append(_make_passage(article, buffer))
            buffer, word_count = [], 0

    if buffer:
        passages.append(_make_passage(article, buffer))
    return passages


def _make_passage(article: Dict[str, str], paragraphs: List[str]) -> Dict[str, str]:
    return {
        "title": article["title"],
        "url": article["url"],
        "text": " ".join(paragraphs),
    }


class PassageIndex:
    """An in-memory BM25 index built fresh for each query's candidate articles.

    Because we only ever rank a handful of articles per query, building the index
    on the fly is cheap and keeps the engine stateless — no database required.
    """

    def __init__(self, articles: List[Dict[str, str]]):
        self.passages: List[Dict[str, str]] = []
        for article in articles:
            self.passages.extend(_split_into_passages(article))

        # Guard against an empty corpus (BM25Okapi can't handle zero docs).
        tokenized = [tokenize(p["text"]) or ["__empty__"] for p in self.passages]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, top_k: int) -> List[Dict]:
        """Return the top_k passages with their BM25 scores, best first."""
        if not self._bm25 or not self.passages:
            return []

        query_tokens = tokenize(query) or tokenize(query.lower())
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            range(len(self.passages)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results: List[Dict] = []
        for rank, idx in enumerate(ranked[:top_k], start=1):
            score = float(scores[idx])
            if score <= 0:
                continue  # no query term matched this passage
            passage = self.passages[idx]
            results.append(
                {
                    "rank": rank,
                    "score": round(score, 3),
                    "title": passage["title"],
                    "url": passage["url"],
                    "text": passage["text"],
                }
            )
        return results
