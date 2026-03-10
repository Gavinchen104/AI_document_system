"""API routes for documents, extraction, match, and suggestions."""

import logging

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request

logger = logging.getLogger("ai_document_system")
from fastapi.responses import PlainTextResponse

from src.limiter import limiter
from src.config import settings

from src.documents.schemas import DocumentType
from src.documents.parser import parse_document, parse_text_input
from src.agents.orchestrator import route_document, run_ocr_on_pdf
from src.storage.store import (
    create_document,
    get_document,
    list_documents,
    get_profile,
    get_match_scores,
)
from src.storage.vector import add_profile
from src.agents.extraction import extract_profile
from src.agents.matching import run_match
from src.api.models import (
    DocumentCreateResponse,
    ExtractResponse,
    MatchRequest,
    MatchResponse,
    JobSuggestionsResponse,
    JobSuggestionItem,
)

router = APIRouter(prefix="/api", tags=["api"])


@router.get("/health")
async def api_health():
    """Health check with dependency status (embedding, LLM)."""
    from src.config import settings
    status = {"status": "ok"}
    status["embedding"] = "ok" if settings.use_local_embeddings else ("ok" if settings.openai_api_key else "no_key")
    status["llm"] = "ok" if settings.openai_api_key else "no_key"
    return status


@router.post("/documents", response_model=DocumentCreateResponse)
@limiter.limit("60/minute")
async def post_document(
    request: Request,
    file: UploadFile | None = File(None),
    type: str = Form(...),
    text: str | None = Form(None),
):
    """
    Upload a document (PDF/DOCX file) or submit raw text.
    - type: "resume" or "job_description"
    - Either provide `file` or `text`.
    """
    try:
        doc_type = DocumentType(type)
    except ValueError:
        raise HTTPException(400, detail="type must be 'resume' or 'job_description'")
    raw_text: str
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if file and file.filename:
        contents = await file.read()
        if len(contents) > max_bytes:
            raise HTTPException(
                400,
                detail=f"File too large. Max {settings.max_file_size_mb} MB.",
            )
        suffix = (file.filename or "").split(".")[-1].lower()
        if suffix == "pdf":
            import tempfile
            from pathlib import Path
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(contents)
                path = f.name
            try:
                try:
                    raw_text = parse_document(path)
                except (ValueError, FileNotFoundError) as e:
                    raise HTTPException(
                        400,
                        detail="Unsupported or unreadable document. Try pasting text instead.",
                    ) from e
                # Orchestrator: if parse yielded little text (e.g. scanned PDF), consider OCR
                if len(raw_text.strip()) < 100:
                    decision = route_document(file_suffix="pdf", text_preview=raw_text)
                    if decision.path in ("ocr", "both"):
                        try:
                            ocr_text = run_ocr_on_pdf(path)
                            raw_text = (raw_text + "\n\n" + ocr_text).strip() if raw_text else ocr_text
                        except Exception:
                            pass  # Fall back to parse-only
            finally:
                Path(path).unlink(missing_ok=True)
        elif suffix in ("docx", "doc"):
            import tempfile
            from pathlib import Path
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
                f.write(contents)
                path = f.name
            try:
                try:
                    raw_text = parse_document(path)
                except (ValueError, FileNotFoundError) as e:
                    raise HTTPException(
                        400,
                        detail="Unsupported or unreadable document. Try pasting text instead.",
                    ) from e
            finally:
                Path(path).unlink(missing_ok=True)
        else:
            raise HTTPException(400, detail="Unsupported file type. Use .pdf or .docx")
    elif text is not None:
        raw_text = parse_text_input(text)
        if not raw_text:
            raise HTTPException(400, detail="text cannot be empty")
        if len(raw_text) > settings.max_text_length:
            raise HTTPException(
                400,
                detail=f"Text too long. Max {settings.max_text_length} characters.",
            )
    else:
        raise HTTPException(400, detail="Provide either file or text")
    doc_id = create_document(doc_type, raw_text)
    logger.info("document_ingest type=%s id=%s len=%d", doc_type.value, doc_id, len(raw_text))
    return DocumentCreateResponse(id=doc_id, type=doc_type.value)


@router.post("/documents/{document_id}/extract", response_model=ExtractResponse)
@limiter.limit("60/minute")
async def post_extract(request: Request, document_id: str):
    """Run extraction agent on a document. Returns summary and skills."""
    doc = get_document(document_id)
    if not doc:
        raise HTTPException(404, detail="Document not found")
    doc_type_str, raw_text = doc
    doc_type = DocumentType(doc_type_str)
    logger.info("extract_start document_id=%s", document_id)
    try:
        profile = extract_profile(document_id, doc_type, raw_text)
        logger.info("extract_end document_id=%s", document_id)
    except Exception as e:
        logger.warning("extract_failed document_id=%s error=%s", document_id, str(e))
        raise HTTPException(
            503,
            detail="Extraction failed; try again or simplify the document.",
        ) from e
    from src.documents.schemas import ResumeProfile, JobProfile
    add_profile(document_id, profile)
    summary = profile.summary
    skills = profile.skills if hasattr(profile, "skills") else []
    # Heuristic confidence: skills count + summary length
    conf = min(100, len(skills) * 5 + min(50, len(summary) // 10)) if summary or skills else None
    return ExtractResponse(
        document_id=document_id,
        summary=summary,
        skills=skills,
        extraction_confidence=conf,
    )


@router.post("/match", response_model=MatchResponse)
@limiter.limit("60/minute")
async def post_match(request: Request, body: MatchRequest):
    """Run matching + scoring for a resume and a job. Both must have been extracted."""
    resume_profile = get_profile(body.resume_id)
    job_profile = get_profile(body.job_id)
    if not resume_profile:
        raise HTTPException(404, detail="Resume profile not found; run extract first")
    if not job_profile:
        raise HTTPException(404, detail="Job profile not found; run extract first")
    from src.documents.schemas import ResumeProfile, JobProfile
    if not isinstance(resume_profile, ResumeProfile):
        raise HTTPException(400, detail="resume_id must refer to a resume document")
    if not isinstance(job_profile, JobProfile):
        raise HTTPException(400, detail="job_id must refer to a job document")
    logger.info("match_start resume_id=%s job_id=%s", body.resume_id, body.job_id)
    try:
        result = run_match(
            body.resume_id,
            body.job_id,
            resume_profile,
            job_profile,
            persist=True,
        )
    except Exception as e:
        logger.warning("match_failed resume_id=%s job_id=%s error=%s", body.resume_id, body.job_id, str(e))
        # Fallback: use embedding-only score when LLM fails
        try:
            from src.agents.matching import _score_embedding_fallback
            fallback_score = _score_embedding_fallback(resume_profile, job_profile)
            return MatchResponse(
                resume_id=body.resume_id,
                job_id=body.job_id,
                score=fallback_score,
                explanations=["Score from embedding similarity (LLM unavailable)."],
                score_confidence=50,
                vague_job_note=None,
            )
        except Exception:
            raise HTTPException(
                503,
                detail="Match failed; try again later.",
            ) from e
    # Vague job: short text or few requirements
    vague_note = None
    if isinstance(job_profile, JobProfile):
        if len(job_profile.raw_text or "") < 200:
            vague_note = "Job description is short; score may be less reliable."
        elif len(job_profile.requirements) < 2 and len(job_profile.skills) < 3:
            vague_note = "Job description is vague; score may be less reliable."
    logger.info("match_end resume_id=%s job_id=%s score=%d", body.resume_id, body.job_id, result.score)
    score_conf = 90 if not result.revised else 75
    return MatchResponse(
        resume_id=body.resume_id,
        job_id=body.job_id,
        score=result.score,
        explanations=result.explanations,
        revised=result.revised,
        revision_reason=result.revision_reason,
        score_confidence=score_conf,
        vague_job_note=vague_note,
    )


@router.get("/jobs/suggestions", response_model=JobSuggestionsResponse)
async def get_jobs_suggestions(resume_id: str):
    """
    Return jobs ranked by match score for the given resume.
    Requires that match has been run for this resume against jobs (via POST /match).
    """
    doc = get_document(resume_id)
    if not doc:
        raise HTTPException(404, detail="Resume document not found")
    if doc[0] != DocumentType.RESUME.value:
        raise HTTPException(400, detail="resume_id must refer to a resume")
    scores = get_match_scores(resume_id)
    suggestions = [
        JobSuggestionItem(
            job_id=s["job_id"],
            score=s["score"],
            explanations=s["explanations"],
        )
        for s in scores
    ]
    return JobSuggestionsResponse(resume_id=resume_id, suggestions=suggestions)


@router.get("/documents")
async def get_documents_list(type: str | None = None):
    """List document ids and types. Optional filter: type=resume or type=job_description."""
    doc_type = None
    if type is not None:
        try:
            doc_type = DocumentType(type)
        except ValueError:
            raise HTTPException(400, detail="type must be 'resume' or 'job_description'")
    docs = list_documents(doc_type)
    return {"documents": docs}
