"""Semantic embeddings for hybrid retrieval, powered by fastembed (ONNX).

Why this exists: plain BM25 matches *words*, not *meaning*. It can't tell that
"What is the capital of Australia?" is answered by "Canberra is the capital city
of Australia" while "Capital punishment in Australia has been abolished" — which
shares the same words — is irrelevant. Embeddings map text to vectors where
semantic similarity is a dot product, so the right passage wins.

We use fastembed (ONNX runtime, no PyTorch) to keep the install light and
deployable on free hosts. The model is small (BAAI/bge-small-en-v1.5, 384-dim)
and lazy-loaded on first use. If anything goes wrong (model can't download, host
too constrained), `available()` returns False and the engine falls back to
lexical-only ranking — degraded but still working.
"""
from typing import List, Optional

import numpy as np

from . import config

_model = None            # cached TextEmbedding instance
_load_failed = False     # once loading fails we stop retrying


def _get_model():
    """Lazily construct the embedding model; cache it; never raise."""
    global _model, _load_failed
    if _model is not None or _load_failed:
        return _model
    try:
        from fastembed import TextEmbedding

        _model = TextEmbedding(model_name=config.EMBED_MODEL)
    except Exception:
        _load_failed = True
        _model = None
    return _model


def available() -> bool:
    """True if semantic reranking can be used right now."""
    if not config.USE_EMBEDDINGS:
        return False
    return _get_model() is not None


def embed_query(query: str) -> Optional[np.ndarray]:
    """Embed a search query (uses the model's retrieval-query prefix)."""
    model = _get_model()
    if model is None:
        return None
    try:
        return next(iter(model.query_embed(query)))
    except Exception:
        return None


def embed_passages(texts: List[str]) -> Optional[np.ndarray]:
    """Embed a list of passages, returning an (n, dim) matrix (or None)."""
    model = _get_model()
    if model is None or not texts:
        return None
    try:
        return np.asarray(list(model.embed(texts)))
    except Exception:
        return None


def warm_up() -> None:
    """Trigger model download/load ahead of the first user query."""
    _get_model()
