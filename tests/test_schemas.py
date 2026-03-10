"""Tests for document schemas."""

from src.documents.schemas import (
    DocumentType,
    JobProfile,
    JobRequirement,
    ResumeProfile,
    ExperienceEntry,
    EducationEntry,
)


def test_resume_profile_defaults():
    """ResumeProfile has sensible defaults."""
    p = ResumeProfile()
    assert p.raw_text == ""
    assert p.skills == []
    assert p.experience == []
    assert p.education == []


def test_resume_profile_from_extraction():
    """ResumeProfile can hold extracted data."""
    p = ResumeProfile(
        raw_text="Full resume text...",
        summary="SWE with 5 years Python",
        skills=["Python", "FastAPI"],
        experience=[
            ExperienceEntry(title="Engineer", company="Acme", duration="2 years"),
        ],
        education=[EducationEntry(degree="BS", institution="MIT", field="CS")],
        years_experience=5,
    )
    assert p.years_experience == 5
    assert len(p.experience) == 1
    assert p.experience[0].company == "Acme"


def test_job_profile_defaults():
    """JobProfile has sensible defaults."""
    p = JobProfile()
    assert p.title == ""
    assert p.requirements == []
    assert p.skills == []


def test_job_profile_from_extraction():
    """JobProfile can hold extracted data."""
    p = JobProfile(
        title="Senior Software Engineer",
        company="Tech Co",
        summary="Backend role",
        requirements=[
            JobRequirement(text="5+ years Python", required=True, category="experience"),
        ],
        skills=["Python", "SQL"],
        preferred_skills=["Kubernetes"],
        years_experience=5,
    )
    assert p.years_experience == 5
    assert len(p.requirements) == 1
    assert p.requirements[0].text == "5+ years Python"


def test_document_type_enum():
    """DocumentType has resume and job_description."""
    assert DocumentType.RESUME.value == "resume"
    assert DocumentType.JOB_DESCRIPTION.value == "job_description"
