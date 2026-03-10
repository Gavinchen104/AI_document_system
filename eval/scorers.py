"""Scoring strategies for evaluation: embedding-only, keyword, LLM-only, full pipeline."""

import numpy as np

from src.documents.schemas import JobProfile, ResumeProfile
from src.storage.vector import _embedding_function, _text_for_embedding


def _embed_text(text: str) -> list[float]:
    """Embed text using the configured embedding function."""
    emb_fn = _embedding_function()
    return emb_fn.embed_query(text)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Returns value in [-1, 1]."""
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))


def _scale_to_100(s: float) -> float:
    """Scale a similarity score from [-1,1] or [0,1] to 0-100."""
    if s < 0:
        return max(0, 50 + s * 50)
    return min(100, s * 100)


def score_embedding_only(resume_profile: ResumeProfile, job_profile: JobProfile) -> float:
    """Cosine similarity between resume and job embeddings. No LLM."""
    r_text = _text_for_embedding(resume_profile)
    j_text = _text_for_embedding(job_profile)
    r_vec = _embed_text(r_text)
    j_vec = _embed_text(j_text)
    sim = _cosine_similarity(r_vec, j_vec)
    return _scale_to_100(sim)


def score_keyword_overlap(resume_profile: ResumeProfile, job_profile: JobProfile) -> float:
    """Jaccard-like overlap on skills + requirements. Returns 0-100."""
    r_skills = set(s.lower() for s in resume_profile.skills)
    r_domains = set(d.lower() for d in resume_profile.domains)
    j_skills = set(s.lower() for s in job_profile.skills)
    j_pref = set(s.lower() for s in job_profile.preferred_skills)
    j_req_texts = set(r.text.lower() for r in job_profile.requirements if r.text)
    all_job = j_skills | j_pref | j_req_texts
    all_resume = r_skills | r_domains
    if not all_job:
        return 50.0
    overlap = len(all_resume & all_job) / len(all_job)
    return min(100, overlap * 100)


def score_llm_only(resume_profile: ResumeProfile, job_profile: JobProfile) -> int:
    """LLM scoring only. Uses matching module."""
    from src.agents.matching import _score_with_llm
    result = _score_with_llm(resume_profile, job_profile)
    return result.score


def score_full_pipeline(resume_profile: ResumeProfile, job_profile: JobProfile) -> int:
    """Full pipeline: same as LLM-only (current production flow)."""
    return score_llm_only(resume_profile, job_profile)
