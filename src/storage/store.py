"""Structured store for documents and extracted profiles (SQLite)."""

import json
import sqlite3
import uuid
from pathlib import Path

from src.config import settings
from src.documents.schemas import DocumentType, JobProfile, ResumeProfile


def _ensure_data_dir():
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)


def _get_conn() -> sqlite3.Connection:
    _ensure_data_dir()
    conn = sqlite3.connect(settings.sqlite_path)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            raw_text TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS profiles (
            document_id TEXT PRIMARY KEY,
            profile_kind TEXT NOT NULL,
            profile_json TEXT NOT NULL,
            updated_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (document_id) REFERENCES documents(id)
        );
        CREATE TABLE IF NOT EXISTS match_scores (
            id TEXT PRIMARY KEY,
            resume_id TEXT NOT NULL,
            job_id TEXT NOT NULL,
            score INTEGER NOT NULL,
            explanations_json TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (resume_id) REFERENCES documents(id),
            FOREIGN KEY (job_id) REFERENCES documents(id)
        );
    """)


def create_document(doc_type: DocumentType, raw_text: str) -> str:
    """Insert a document and return its id."""
    _ensure_data_dir()
    doc_id = str(uuid.uuid4())
    with _get_conn() as conn:
        _init_db(conn)
        conn.execute(
            "INSERT INTO documents (id, type, raw_text) VALUES (?, ?, ?)",
            (doc_id, doc_type.value, raw_text),
        )
    return doc_id


def get_document(doc_id: str) -> tuple[str, str] | None:
    """Return (type, raw_text) for a document, or None."""
    with _get_conn() as conn:
        _init_db(conn)
        row = conn.execute(
            "SELECT type, raw_text FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
    if not row:
        return None
    return (row["type"], row["raw_text"])


def list_documents(doc_type: DocumentType | None = None) -> list[dict]:
    """List documents, optionally filtered by type. Each dict has id, type, created_at."""
    with _get_conn() as conn:
        _init_db(conn)
        if doc_type is None:
            rows = conn.execute(
                "SELECT id, type, created_at FROM documents ORDER BY created_at DESC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, type, created_at FROM documents WHERE type = ? ORDER BY created_at DESC",
                (doc_type.value,),
            ).fetchall()
    return [dict(r) for r in rows]


def save_profile(document_id: str, profile: ResumeProfile | JobProfile) -> None:
    """Persist an extracted profile for a document."""
    kind = "resume" if isinstance(profile, ResumeProfile) else "job"
    with _get_conn() as conn:
        _init_db(conn)
        conn.execute(
            """INSERT OR REPLACE INTO profiles (document_id, profile_kind, profile_json, updated_at)
               VALUES (?, ?, ?, datetime('now'))""",
            (document_id, kind, profile.model_dump_json()),
        )


def get_profile(document_id: str) -> ResumeProfile | JobProfile | None:
    """Load profile for a document. Returns ResumeProfile or JobProfile or None."""
    with _get_conn() as conn:
        _init_db(conn)
        row = conn.execute(
            "SELECT profile_kind, profile_json FROM profiles WHERE document_id = ?",
            (document_id,),
        ).fetchone()
    if not row:
        return None
    data = json.loads(row["profile_json"])
    if row["profile_kind"] == "resume":
        return ResumeProfile.model_validate(data)
    return JobProfile.model_validate(data)


def save_match_score(resume_id: str, job_id: str, score: int, explanations: list[str]) -> str:
    """Store a match result; return the match id."""
    _ensure_data_dir()
    match_id = str(uuid.uuid4())
    with _get_conn() as conn:
        _init_db(conn)
        conn.execute(
            """INSERT INTO match_scores (id, resume_id, job_id, score, explanations_json)
               VALUES (?, ?, ?, ?, ?)""",
            (match_id, resume_id, job_id, score, json.dumps(explanations)),
        )
    return match_id


def get_match_scores(resume_id: str) -> list[dict]:
    """Return list of {job_id, score, explanations} for a resume, ordered by score desc."""
    with _get_conn() as conn:
        _init_db(conn)
        rows = conn.execute(
            """SELECT job_id, score, explanations_json FROM match_scores
               WHERE resume_id = ? ORDER BY score DESC""",
            (resume_id,),
        ).fetchall()
    return [
        {
            "job_id": r["job_id"],
            "score": r["score"],
            "explanations": json.loads(r["explanations_json"]),
        }
        for r in rows
    ]
