"""Extraction agent: raw document text -> structured profile (Resume or Job)."""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import settings
from src.documents.schemas import DocumentType, JobProfile, ResumeProfile
from src.agents.prompts import RESUME_EXTRACTION_SYSTEM, JOB_EXTRACTION_SYSTEM
from src.storage.store import save_profile


def _llm():
    kwargs = {"model": settings.llm_model, "api_key": settings.openai_api_key}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return ChatOpenAI(**kwargs)


def _extract_resume(raw_text: str) -> ResumeProfile:
    llm = _llm()
    structured_llm = llm.with_structured_output(ResumeProfile)
    response = structured_llm.invoke([
        SystemMessage(content=RESUME_EXTRACTION_SYSTEM),
        HumanMessage(content=f"Extract structured profile from this resume:\n\n{raw_text}"),
    ])
    if isinstance(response, ResumeProfile):
        profile = response
    else:
        profile = ResumeProfile.model_validate(response)
    profile.raw_text = raw_text
    return profile


def _extract_job(raw_text: str) -> JobProfile:
    llm = _llm()
    structured_llm = llm.with_structured_output(JobProfile)
    response = structured_llm.invoke([
        SystemMessage(content=JOB_EXTRACTION_SYSTEM),
        HumanMessage(content=f"Extract structured profile from this job description:\n\n{raw_text}"),
    ])
    if isinstance(response, JobProfile):
        profile = response
    else:
        profile = JobProfile.model_validate(response)
    profile.raw_text = raw_text
    return profile


def extract_profile(
    document_id: str,
    doc_type: DocumentType,
    raw_text: str,
    persist: bool = True,
) -> ResumeProfile | JobProfile:
    """
    Run extraction agent on raw text and return the structured profile.
    Persists the profile to storage unless persist=False (e.g. for eval).
    """
    if doc_type == DocumentType.RESUME:
        profile = _extract_resume(raw_text)
    else:
        profile = _extract_job(raw_text)
    if persist:
        save_profile(document_id, profile)
    return profile
