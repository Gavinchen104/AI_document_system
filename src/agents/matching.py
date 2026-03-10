"""Matching and scoring agents: compare resume vs job and produce score + explanations."""

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import settings
from src.documents.schemas import JobProfile, ResumeProfile
from src.agents.prompts import SCORING_SYSTEM, CRITIC_SYSTEM
from src.storage.store import save_match_score
from src.storage.vector import add_profile, similarity_search
from src.retrieval.hybrid import hybrid_search


def _score_embedding_fallback(resume_profile: ResumeProfile, job_profile: JobProfile) -> int:
    """Embedding-only score (0-100) when LLM is unavailable."""
    import numpy as np
    from src.storage.vector import _text_for_embedding, _embedding_function
    emb_fn = _embedding_function()
    r_text = _text_for_embedding(resume_profile)
    j_text = _text_for_embedding(job_profile)
    r_vec = emb_fn.embed_query(r_text)
    j_vec = emb_fn.embed_query(j_text)
    va, vb = np.array(r_vec), np.array(j_vec)
    sim = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
    scaled = min(100, max(0, int(50 + sim * 50)))
    return scaled


class MatchResult(BaseModel):
    """Result of matching a resume to a job."""

    score: int = Field(ge=0, le=100)
    explanations: list[str] = Field(default_factory=list)
    revised: bool = False
    revision_reason: str | None = None


class CriticReview(BaseModel):
    """Critic's review of score consistency."""

    consistent: bool = True
    revised_score: int | None = None
    reason: str = ""


def _llm():
    kwargs = {"model": settings.llm_model, "api_key": settings.openai_api_key}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return ChatOpenAI(**kwargs)


def _score_with_llm(resume_profile: ResumeProfile, job_profile: JobProfile) -> MatchResult:
    """Use LLM to produce fit score and explanations."""
    llm = _llm()
    structured_llm = llm.with_structured_output(MatchResult)
    resume_summary = (
        f"Summary: {resume_profile.summary}\n"
        f"Skills: {', '.join(resume_profile.skills)}\n"
        f"Experience (years): {resume_profile.years_experience}\n"
        f"Domains: {', '.join(resume_profile.domains)}"
    )
    job_summary = (
        f"Title: {job_profile.title}\n"
        f"Summary: {job_profile.summary}\n"
        f"Required skills: {', '.join(job_profile.skills)}\n"
        f"Preferred: {', '.join(job_profile.preferred_skills)}\n"
        f"Experience (years): {job_profile.years_experience}\n"
        f"Requirements: {[r.text for r in job_profile.requirements]}"
    )
    response = structured_llm.invoke([
        SystemMessage(content=SCORING_SYSTEM),
        HumanMessage(
            content=f"Candidate profile:\n{resume_summary}\n\nJob profile:\n{job_summary}\n\nProvide fit score (0-100) and 3-5 bullet explanations."
        ),
    ])
    if isinstance(response, MatchResult):
        return response
    return MatchResult.model_validate(response)


def _critic_review(score: int, explanations: list[str]) -> CriticReview:
    """Check if score and explanations are consistent."""
    llm = _llm()
    structured_llm = llm.with_structured_output(CriticReview)
    content = (
        f"Score: {score}\nExplanations:\n" + "\n".join(f"- {e}" for e in explanations)
        + "\n\nIs this consistent? If not, suggest revised_score and reason."
    )
    response = structured_llm.invoke([
        SystemMessage(content=CRITIC_SYSTEM),
        HumanMessage(content=content),
    ])
    if isinstance(response, CriticReview):
        return response
    return CriticReview.model_validate(response)


def run_match(
    resume_id: str,
    job_id: str,
    resume_profile: ResumeProfile,
    job_profile: JobProfile,
    persist: bool = True,
) -> MatchResult:
    """
    Run matching + scoring agents for one resume vs one job.
    Includes reflection: critic checks consistency and may trigger one revision.
    Returns score and explanations. Optionally persists to store.
    """
    result = _score_with_llm(resume_profile, job_profile)
    critic = _critic_review(result.score, result.explanations)
    if not critic.consistent and critic.revised_score is not None:
        delta = abs(critic.revised_score - result.score)
        if delta > 10:
            # Re-run scorer once with critic feedback
            llm = _llm()
            structured_llm = llm.with_structured_output(MatchResult)
            resume_summary = (
                f"Summary: {resume_profile.summary}\n"
                f"Skills: {', '.join(resume_profile.skills)}\n"
                f"Experience (years): {resume_profile.years_experience}\n"
                f"Domains: {', '.join(resume_profile.domains)}"
            )
            job_summary = (
                f"Title: {job_profile.title}\n"
                f"Summary: {job_profile.summary}\n"
                f"Required skills: {', '.join(job_profile.skills)}\n"
                f"Preferred: {', '.join(job_profile.preferred_skills)}\n"
                f"Experience (years): {job_profile.years_experience}\n"
                f"Requirements: {[r.text for r in job_profile.requirements]}"
            )
            revision_note = f"\n\nConsider this feedback: {critic.reason}. Revise your score if needed."
            response = structured_llm.invoke([
                SystemMessage(content=SCORING_SYSTEM),
                HumanMessage(
                    content=f"Candidate profile:\n{resume_summary}\n\nJob profile:\n{job_summary}"
                    + revision_note
                ),
            ])
            result = MatchResult.model_validate(response) if not isinstance(response, MatchResult) else response
            result.revised = True
            result.revision_reason = critic.reason
    if persist:
        save_match_score(
            resume_id,
            job_id,
            result.score,
            result.explanations,
        )
    return result


def get_ranked_jobs_for_resume(
    resume_id: str,
    resume_profile: ResumeProfile,
    job_ids: list[str] | None = None,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """
    Use hybrid (BM25 + dense) retrieval to rank jobs for a resume.
    Returns list of (job_id, similarity_score).
    """
    return hybrid_search(resume_profile, kind="job", top_k=top_k)
