"""Hybrid retrieval: BM25 + dense embeddings with configurable weighting."""

import numpy as np

from src.config import settings
from src.documents.schemas import JobProfile, ResumeProfile
from src.storage.vector import _text_for_embedding, similarity_search
from src.storage.bm25 import bm25_search


def _normalize_scores(scores: list[tuple[str, float]]) -> dict[str, float]:
    """Min-max normalize scores to [0, 1]."""
    if not scores:
        return {}
    vals = [s[1] for s in scores]
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {s[0]: 1.0 for s in scores}
    return {s[0]: (s[1] - lo) / (hi - lo) for s in scores}


def hybrid_search(
    query_profile: ResumeProfile | JobProfile,
    kind: str = "job",
    top_k: int = 10,
    alpha: float | None = None,
) -> list[tuple[str, float]]:
    """
    Hybrid search: alpha * BM25 + (1-alpha) * dense.
    Returns list of (document_id, combined_score) sorted by score desc.
    """
    alpha = alpha if alpha is not None else settings.hybrid_alpha
    query_text = _text_for_embedding(query_profile)

    # BM25 (only for job search)
    bm25_results = bm25_search(query_text, top_k=top_k * 2) if kind == "job" else []
    bm25_norm = _normalize_scores(bm25_results)

    # Dense
    dense_results = similarity_search(query_profile, kind=kind, top_k=top_k * 2)
    dense_norm = _normalize_scores(dense_results)

    # Combine
    all_ids = set(bm25_norm) | set(dense_norm)
    combined = []
    for doc_id in all_ids:
        b = bm25_norm.get(doc_id, 0.0)
        d = dense_norm.get(doc_id, 0.0)
        score = alpha * b + (1 - alpha) * d
        combined.append((doc_id, score))
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:top_k]
