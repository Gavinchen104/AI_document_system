"""Tests for API routes."""

import tempfile
import os

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.storage.store import create_document, get_profile, save_profile
from src.documents.schemas import DocumentType, ResumeProfile, JobProfile

client = TestClient(app)


def test_health():
    """Health endpoint returns ok."""
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"


def test_post_document_text():
    """POST /api/documents with text creates a document."""
    r = client.post(
        "/api/documents",
        data={"type": "resume", "text": "John Doe. Software Engineer. Python, FastAPI. 5 years."},
    )
    assert r.status_code == 200
    data = r.json()
    assert "id" in data
    assert data["type"] == "resume"


def test_post_document_invalid_type():
    """POST /api/documents with invalid type returns 400."""
    r = client.post("/api/documents", data={"type": "invalid", "text": "hello"})
    assert r.status_code == 400


def test_post_document_no_file_no_text():
    """POST /api/documents without file or text returns 400."""
    r = client.post("/api/documents", data={"type": "resume"})
    assert r.status_code == 400


def test_get_documents_list():
    """GET /api/documents returns list of documents."""
    r = client.get("/api/documents")
    assert r.status_code == 200
    data = r.json()
    assert "documents" in data


def test_extract_404():
    """POST /api/documents/{id}/extract with bad id returns 404."""
    r = client.post("/api/documents/00000000-0000-0000-0000-000000000000/extract")
    assert r.status_code == 404


def test_match_404():
    """POST /api/match with missing resume profile returns 404."""
    r = client.post(
        "/api/match",
        json={"resume_id": "00000000-0000-0000-0000-000000000000", "job_id": "00000000-0000-0000-0000-000000000001"},
    )
    assert r.status_code == 404


def test_jobs_suggestions_404():
    """GET /api/jobs/suggestions with bad resume_id returns 404."""
    r = client.get("/api/jobs/suggestions?resume_id=00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404
