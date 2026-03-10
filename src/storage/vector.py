"""Vector store for resume and job embeddings (Chroma)."""

from pathlib import Path

from src.config import settings
from src.documents.schemas import JobProfile, ResumeProfile

# Lazy init to avoid importing heavy libs at module load
_chroma_client = None
_collection = None


def _text_for_embedding(profile: ResumeProfile | JobProfile) -> str:
    """Build a single searchable text blob from a profile."""
    if isinstance(profile, ResumeProfile):
        parts = [
            profile.summary,
            " ".join(profile.skills),
            " ".join(profile.domains),
            str(profile.years_experience) if profile.years_experience else "",
        ]
        for ex in profile.experience:
            parts.extend([ex.title, ex.company, ex.description, " ".join(ex.skills_used)])
        for ed in profile.education:
            parts.extend([ed.degree, ed.institution, ed.field])
    else:
        parts = [
            profile.title,
            profile.company,
            profile.summary,
            " ".join(profile.skills),
            " ".join(profile.preferred_skills),
            " ".join(profile.domains),
            str(profile.years_experience) if profile.years_experience else "",
        ]
        for req in profile.requirements:
            parts.append(req.text)
    return "\n".join(p for p in parts if p).strip() or profile.raw_text[:8000]


def _get_collection():
    global _chroma_client, _collection
    if _collection is not None:
        return _collection
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    _chroma_client = chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    _collection = _chroma_client.get_or_create_collection(
        name="profiles",
        metadata={"description": "Resume and job profiles for matching"},
    )
    return _collection


def _embedding_function():
    """Return embedding function: OpenAI or local sentence-transformers."""
    if settings.use_local_embeddings:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=settings.local_embedding_model,
            model_kwargs={"device": "cpu"},
        )
    from langchain_openai import OpenAIEmbeddings
    kwargs = {"model": settings.embedding_model, "api_key": settings.openai_api_key}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return OpenAIEmbeddings(**kwargs)


def add_profile(document_id: str, profile: ResumeProfile | JobProfile) -> None:
    """Add or update a profile in the vector store."""
    coll = _get_collection()
    emb_fn = _embedding_function()
    text = _text_for_embedding(profile)
    # Chroma can use its default embedding; we use our own for consistency
    vectors = emb_fn.embed_documents([text])
    meta = {"document_id": document_id}
    if isinstance(profile, ResumeProfile):
        meta["kind"] = "resume"
    else:
        meta["kind"] = "job"
        try:
            from src.storage.bm25 import invalidate_bm25_cache
            invalidate_bm25_cache()
        except Exception:
            pass
    # Upsert: remove existing then add (Chroma doesn't have native update by id)
    try:
        coll.delete(ids=[document_id])
    except Exception:
        pass
    coll.add(ids=[document_id], embeddings=vectors, documents=[text], metadatas=[meta])


def similarity_search(
    query_profile: ResumeProfile | JobProfile,
    kind: str,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """
    Search for similar profiles. kind is 'resume' or 'job'.
    Returns list of (document_id, score) where higher score = more similar.
    """
    coll = _get_collection()
    emb_fn = _embedding_function()
    query_text = _text_for_embedding(query_profile)
    query_vec = emb_fn.embed_query(query_text)
    results = coll.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        where={"kind": kind},
        include=["metadatas", "distances"],
    )
    if not results or not results["ids"] or not results["ids"][0]:
        return []
    # Chroma returns distances (lower = more similar); convert to similarity-like score 0-1
    ids = results["ids"][0]
    distances = results["distances"][0]
    doc_ids = [m["document_id"] for m in results["metadatas"][0]]
    # Simple conversion: 1 / (1 + distance) so higher = better
    scores = [1.0 / (1.0 + d) for d in distances]
    return list(zip(doc_ids, scores))
