"""Orchestrator agent: decides processing path (OCR, parse, or both) for document ingestion."""

from pathlib import Path

from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import settings


class RouteDecision(BaseModel):
    """Orchestrator output: which path to use."""

    path: str  # "ocr" | "parse" | "both"
    reason: str = ""


ORCHESTRATOR_SYSTEM = """You decide how to process a document for text extraction.
Given metadata about an uploaded file (type, size hint) and optionally a short preview of extracted text, output:
- path: one of "ocr", "parse", "both"
  - "parse": use standard PDF/DOCX text extraction (structured, digital documents)
  - "ocr": use OCR (scanned PDFs, images, low-quality scans)
  - "both": try both and merge (when uncertain)
- reason: one sentence explaining why

Use "parse" for normal PDFs/DOCX. Use "ocr" when the document appears to be a scan or image-based. Use "both" when you cannot tell."""


def _llm():
    kwargs = {"model": settings.llm_model, "api_key": settings.openai_api_key}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return ChatOpenAI(**kwargs)


def route_document(
    file_path: str | Path | None = None,
    file_suffix: str = "",
    text_preview: str = "",
) -> RouteDecision:
    """
    Decide processing path for a document.
    If no API key, defaults to "parse".
    """
    if not settings.openai_api_key:
        return RouteDecision(path="parse", reason="No LLM; using default parse path")

    llm = _llm()
    structured_llm = llm.with_structured_output(RouteDecision)
    preview = (text_preview or "")[:500]
    meta = f"File type: {file_suffix or 'unknown'}. Preview length: {len(text_preview or '')} chars."
    if preview:
        meta += f"\nPreview: {preview[:300]}..."
    response = structured_llm.invoke([
        SystemMessage(content=ORCHESTRATOR_SYSTEM),
        HumanMessage(content=meta),
    ])
    if isinstance(response, RouteDecision):
        return response
    return RouteDecision.model_validate(response)


def run_ocr_on_pdf(pdf_path: str | Path) -> str:
    """
    Run OCR on a PDF (scanned pages). Requires pytesseract and pdf2image.
    Returns extracted text.
    """
    try:
        import pdf2image
        import pytesseract
    except ImportError:
        raise ImportError("OCR requires: pip install pdf2image pytesseract")
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    images = pdf2image.convert_from_path(str(path))
    parts = []
    for img in images:
        parts.append(pytesseract.image_to_string(img))
    return "\n".join(parts).strip()
