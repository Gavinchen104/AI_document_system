"""Parse PDF and DOCX documents to raw text."""

from pathlib import Path

import fitz  # PyMuPDF
from docx import Document as DocxDocument


def extract_text_from_pdf(path: str | Path) -> str:
    """Extract text from a PDF file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    doc = fitz.open(path)
    parts = []
    for page in doc:
        parts.append(page.get_text())
    doc.close()
    return "\n".join(parts).strip()


def extract_text_from_docx(path: str | Path) -> str:
    """Extract text from a DOCX file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DOCX not found: {path}")
    doc = DocxDocument(path)
    parts = [p.text for p in doc.paragraphs]
    return "\n".join(parts).strip()


def parse_document(path: str | Path) -> str:
    """
    Parse a document (PDF or DOCX) and return raw text.
    Raises ValueError if the format is not supported.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix in (".docx", ".doc"):
        return extract_text_from_docx(path)
    raise ValueError(f"Unsupported document format: {suffix}. Use .pdf or .docx")


def parse_text_input(text: str) -> str:
    """Normalize pasted text (e.g. from a job description)."""
    return (text or "").strip()
