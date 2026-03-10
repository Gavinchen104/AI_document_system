"""Request/response Pydantic models for the API."""

from pydantic import BaseModel, Field

from src.documents.schemas import DocumentType


class DocumentCreateResponse(BaseModel):
    """Response after creating a document."""

    id: str
    type: str


class ExtractResponse(BaseModel):
    """Response after running extraction (summary of profile)."""

    document_id: str
    summary: str
    skills: list[str] = Field(default_factory=list)
    extraction_confidence: int | None = Field(None, ge=0, le=100)


class MatchRequest(BaseModel):
    """Request body for match endpoint."""

    resume_id: str
    job_id: str


class MatchResponse(BaseModel):
    """Response for match endpoint."""

    resume_id: str
    job_id: str
    score: int
    explanations: list[str] = Field(default_factory=list)
    revised: bool = False
    revision_reason: str | None = None
    score_confidence: int | None = Field(None, ge=0, le=100)
    vague_job_note: str | None = None


class JobSuggestionItem(BaseModel):
    """One item in job suggestions list."""

    job_id: str
    score: int | None = None
    explanations: list[str] = Field(default_factory=list)


class JobSuggestionsResponse(BaseModel):
    """Response for GET /jobs/suggestions."""

    resume_id: str
    suggestions: list[JobSuggestionItem] = Field(default_factory=list)
