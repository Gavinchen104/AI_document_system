"""Structured profiles for resume and job descriptions."""

from enum import Enum
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Type of uploaded document."""

    RESUME = "resume"
    JOB_DESCRIPTION = "job_description"


class ExperienceEntry(BaseModel):
    """Single work experience entry."""

    title: str = ""
    company: str = ""
    duration: str = ""
    description: str = ""
    skills_used: list[str] = Field(default_factory=list)


class EducationEntry(BaseModel):
    """Single education entry."""

    degree: str = ""
    institution: str = ""
    year: str = ""
    field: str = ""


class ResumeProfile(BaseModel):
    """Structured profile extracted from a resume."""

    raw_text: str = ""
    summary: str = ""
    skills: list[str] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)
    years_experience: int | None = None
    domains: list[str] = Field(default_factory=list)  # e.g. backend, ML, etc.


class JobRequirement(BaseModel):
    """Single requirement from a job description."""

    text: str = ""
    required: bool = True
    category: str = ""  # e.g. skill, experience, education


class JobProfile(BaseModel):
    """Structured profile extracted from a job description."""

    raw_text: str = ""
    title: str = ""
    company: str = ""
    summary: str = ""
    requirements: list[JobRequirement] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    years_experience: int | None = None
    domains: list[str] = Field(default_factory=list)
