"""Tests for structured store."""

import tempfile
import os

import pytest

from src.storage.store import (
    create_document,
    get_document,
    list_documents,
    save_profile,
    get_profile,
    save_match_score,
    get_match_scores,
)
from src.documents.schemas import DocumentType, JobProfile, ResumeProfile


@pytest.fixture
def temp_db(monkeypatch):
    """Use a temporary SQLite file for tests."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    try:
        monkeypatch.setattr("src.storage.store.settings", type("S", (), {"sqlite_path": path, "data_dir": os.path.dirname(path)})())
        yield path
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_create_and_get_document(temp_db):
    """Creating a document returns id; get_document returns type and raw_text."""
    from src.storage import store
    store.settings.sqlite_path = temp_db
    store.settings.data_dir = os.path.dirname(temp_db)
    doc_id = create_document(DocumentType.RESUME, "Hello resume text")
    assert doc_id
    out = get_document(doc_id)
    assert out is not None
    typ, text = out
    assert typ == DocumentType.RESUME.value
    assert text == "Hello resume text"


def test_list_documents(temp_db):
    """list_documents returns created documents."""
    from src.storage import store
    store.settings.sqlite_path = temp_db
    store.settings.data_dir = os.path.dirname(temp_db)
    id1 = create_document(DocumentType.RESUME, "Resume 1")
    id2 = create_document(DocumentType.JOB_DESCRIPTION, "Job 1")
    all_docs = list_documents()
    assert len(all_docs) >= 2
    ids = {d["id"] for d in all_docs}
    assert id1 in ids and id2 in ids
    resumes = list_documents(DocumentType.RESUME)
    assert any(d["id"] == id1 for d in resumes)


def test_save_and_get_profile(temp_db):
    """save_profile and get_profile round-trip ResumeProfile and JobProfile."""
    from src.storage import store
    store.settings.sqlite_path = temp_db
    store.settings.data_dir = os.path.dirname(temp_db)
    doc_id = create_document(DocumentType.RESUME, "Resume text")
    profile = ResumeProfile(summary="SWE", skills=["Python"], years_experience=3)
    save_profile(doc_id, profile)
    loaded = get_profile(doc_id)
    assert isinstance(loaded, ResumeProfile)
    assert loaded.summary == "SWE"
    assert loaded.skills == ["Python"]
    assert loaded.years_experience == 3


def test_save_and_get_match_score(temp_db):
    """save_match_score and get_match_scores work."""
    from src.storage import store
    store.settings.sqlite_path = temp_db
    store.settings.data_dir = os.path.dirname(temp_db)
    r_id = create_document(DocumentType.RESUME, "R")
    j_id = create_document(DocumentType.JOB_DESCRIPTION, "J")
    save_match_score(r_id, j_id, 85, ["Good fit", "Strong Python"])
    scores = get_match_scores(r_id)
    assert len(scores) == 1
    assert scores[0]["job_id"] == j_id
    assert scores[0]["score"] == 85
    assert "Good fit" in scores[0]["explanations"]
