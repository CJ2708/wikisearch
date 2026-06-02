"""Cross-encoder reranking — the precision step on top of hybrid retrieval.

Hybrid retrieval (BM25 + embeddings) is a fast "recall" stage: it cheaply scores
every passage independently and surfaces a good candidate set. A cross-encoder is
slower but far more accurate: it reads the query and a passage *together* and
judges true relevance, catching nuances bi-encoders miss. So we use the classic
two-stage pattern — retrieve a candidate set cheaply, then rerank the top few
with the cross-encoder to nail the ordering.

Powered by fastembed's ONNX TextCrossEncoder (no PyTorch), lazy-loaded, with a
graceful fallback to the hybrid order if the model is unavailable.
"""
from typing import List, Optional

from . import config

_model = None
_load_failed = False


def _get_model():
    global _model, _load_failed
    if _model is not None or _load_failed:
        return _model
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        _model = TextCrossEncoder(model_name=config.RERANK_MODEL)
    except Exception:
        _load_failed = True
        _model = None
    return _model


def available() -> bool:
    if not config.USE_RERANKER:
        return False
    return _get_model() is not None


def rerank_scores(query: str, documents: List[str]) -> Optional[List[float]]:
    """Return one relevance score per document (higher = more relevant), or None."""
    model = _get_model()
    if model is None or not documents:
        return None
    try:
        return list(model.rerank(query, documents))
    except Exception:
        return None


def warm_up() -> None:
    _get_model()
