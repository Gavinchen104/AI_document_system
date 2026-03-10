"""BM25 lexical index for job profiles."""

import re
from pathlib import Path

from src.config import settings
from src.documents.schemas import JobProfile, ResumeProfile
from src.storage.vector import _text_for_embedding

_bm25_index = None
_bm25_doc_ids: list[str] = []


def _tokenize(text: str) -> list[str]:
    """Simple tokenization: lowercase, alphanumeric tokens."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def _get_index():
    """Build or return BM25 index over job profiles from Chroma."""
    global _bm25_index, _bm25_doc_ids
    if _bm25_index is not None:
        return _bm25_index, _bm25_doc_ids
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise ImportError("BM25 requires: pip install rank-bm25")
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    coll = client.get_or_create_collection("profiles")
    results = coll.get(where={"kind": "job"}, include=["documents", "metadatas"])
    if not results or not results["ids"]:
        _bm25_index = BM25Okapi([[]])
        _bm25_doc_ids = []
        return _bm25_index, _bm25_doc_ids
    doc_ids = results["ids"]
    documents = results["documents"] or [""] * len(doc_ids)
    tokenized = [_tokenize(d or "") for d in documents]
    _bm25_index = BM25Okapi(tokenized)
    _bm25_doc_ids = doc_ids
    return _bm25_index, _bm25_doc_ids


def bm25_search(query_text: str, top_k: int = 10) -> list[tuple[str, float]]:
    """
    Search job profiles by BM25. Returns list of (document_id, score).
    Higher score = better match.
    """
    index, doc_ids = _get_index()
    if not doc_ids:
        return []
    tokens = _tokenize(query_text)
    scores = index.get_scores(tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(doc_ids[i], float(scores[i])) for i in ranked if scores[i] > 0]


def invalidate_bm25_cache() -> None:
    """Invalidate BM25 index cache (call when adding new job profiles)."""
    global _bm25_index, _bm25_doc_ids
    _bm25_index = None
    _bm25_doc_ids = []
