"""Tests for extraction and scoring agents."""

import pytest

from src.documents.schemas import DocumentType, JobProfile, ResumeProfile
from src.agents.extraction import _extract_resume, _extract_job, extract_profile
from src.agents.prompts import RESUME_EXTRACTION_SYSTEM, JOB_EXTRACTION_SYSTEM, SCORING_SYSTEM


def test_resume_extraction_prompt_defined():
    """Extraction prompts are non-empty."""
    assert "resume" in RESUME_EXTRACTION_SYSTEM.lower()
    assert "skills" in RESUME_EXTRACTION_SYSTEM.lower()


def test_job_extraction_prompt_defined():
    """Job extraction prompt is defined."""
    assert "job" in JOB_EXTRACTION_SYSTEM.lower()
    assert "requirements" in JOB_EXTRACTION_SYSTEM.lower()


def test_scoring_prompt_defined():
    """Scoring prompt asks for score and explanations."""
    assert "score" in SCORING_SYSTEM.lower()
    assert "0" in SCORING_SYSTEM and "100" in SCORING_SYSTEM


@pytest.mark.skipif(
    True,  # Skip unless OPENAI_API_KEY set; run with: pytest -k "extraction" --run-openai
    reason="Requires OPENAI_API_KEY for live LLM call",
)
def test_extract_resume_live():
    """Live call: _extract_resume returns ResumeProfile with raw_text set."""
    raw = "John Doe. Software Engineer. 5 years Python, FastAPI. MIT BS CS."
    profile = _extract_resume(raw)
    assert isinstance(profile, ResumeProfile)
    assert profile.raw_text == raw


@pytest.mark.skipif(
    True,
    reason="Requires OPENAI_API_KEY for live LLM call",
)
def test_extract_job_live():
    """Live call: _extract_job returns JobProfile."""
    raw = "Senior Software Engineer. We need 5+ years Python, FastAPI. BS in CS required."
    profile = _extract_job(raw)
    assert isinstance(profile, JobProfile)
    assert profile.raw_text == raw
