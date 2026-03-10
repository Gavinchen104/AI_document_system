"""CLI for document system: add resume, add job, match, list suggestions."""

from pathlib import Path

import typer

from src.config import settings
from src.documents.schemas import DocumentType
from src.documents.parser import parse_document, parse_text_input
from src.storage.store import (
    create_document,
    get_document,
    list_documents,
    get_profile,
    get_match_scores,
)
from src.agents.extraction import extract_profile
from src.agents.matching import run_match
from src.storage.vector import add_profile

app = typer.Typer(help="Multi-AI-Agent Document System: resume & job matching")


@app.command()
def add_resume(
    path: str = typer.Argument(..., help="Path to resume PDF or DOCX"),
):
    """Add a resume from a file."""
    p = Path(path)
    if not p.exists():
        typer.echo(f"File not found: {path}", err=True)
        raise typer.Exit(1)
    try:
        raw_text = parse_document(p)
    except ValueError as e:
        typer.echo(f"Parse error: {e}", err=True)
        raise typer.Exit(1)
    doc_id = create_document(DocumentType.RESUME, raw_text)
    typer.echo(f"Added resume: {doc_id}")


@app.command()
def add_job(
    path: str | None = typer.Argument(None, help="Path to job description PDF/DOCX"),
    text: str | None = typer.Option(None, "--text", "-t", help="Paste job description text"),
):
    """Add a job description from a file or pasted text."""
    if path:
        p = Path(path)
        if not p.exists():
            typer.echo(f"File not found: {path}", err=True)
            raise typer.Exit(1)
        try:
            raw_text = parse_document(p)
        except ValueError as e:
            typer.echo(f"Parse error: {e}", err=True)
            raise typer.Exit(1)
    elif text:
        raw_text = parse_text_input(text)
        if not raw_text:
            typer.echo("Text is empty.", err=True)
            raise typer.Exit(1)
    else:
        typer.echo("Provide either path or --text.", err=True)
        raise typer.Exit(1)
    doc_id = create_document(DocumentType.JOB_DESCRIPTION, raw_text)
    typer.echo(f"Added job: {doc_id}")


@app.command()
def extract(
    document_id: str = typer.Argument(..., help="Document ID from add-resume or add-job"),
):
    """Run extraction agent on a document."""
    doc = get_document(document_id)
    if not doc:
        typer.echo(f"Document not found: {document_id}", err=True)
        raise typer.Exit(1)
    typ, raw_text = doc
    doc_type = DocumentType(typ)
    try:
        profile = extract_profile(document_id, doc_type, raw_text)
        add_profile(document_id, profile)
    except Exception as e:
        typer.echo(f"Extraction failed: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Extracted profile for {document_id}")
    typer.echo(f"  Summary: {profile.summary[:200]}..." if len(profile.summary) > 200 else f"  Summary: {profile.summary}")


@app.command()
def match(
    resume_id: str = typer.Argument(..., help="Resume document ID"),
    job_id: str = typer.Argument(..., help="Job document ID"),
):
    """Run matching + scoring for a resume and a job. Both must be extracted first."""
    resume_profile = get_profile(resume_id)
    job_profile = get_profile(job_id)
    if not resume_profile:
        typer.echo(f"Resume profile not found: {resume_id}. Run extract first.", err=True)
        raise typer.Exit(1)
    if not job_profile:
        typer.echo(f"Job profile not found: {job_id}. Run extract first.", err=True)
        raise typer.Exit(1)
    from src.documents.schemas import ResumeProfile, JobProfile
    if not isinstance(resume_profile, ResumeProfile):
        typer.echo(f"{resume_id} is not a resume.", err=True)
        raise typer.Exit(1)
    if not isinstance(job_profile, JobProfile):
        typer.echo(f"{job_id} is not a job.", err=True)
        raise typer.Exit(1)
    try:
        result = run_match(resume_id, job_id, resume_profile, job_profile, persist=True)
    except Exception as e:
        typer.echo(f"Match failed: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Score: {result.score}/100")
    for ex in result.explanations:
        typer.echo(f"  - {ex}")


@app.command()
def suggestions(
    resume_id: str = typer.Argument(..., help="Resume document ID"),
    top: int = typer.Option(10, "--top", "-n", help="Max number to show"),
):
    """Show job suggestions (by match score) for a resume."""
    scores = get_match_scores(resume_id)
    if not scores:
        typer.echo("No match results yet. Run 'match <resume_id> <job_id>' for each job first.")
        raise typer.Exit(0)
    for i, s in enumerate(scores[:top], 1):
        typer.echo(f"{i}. Job {s['job_id']} — Score: {s['score']}")
        for ex in s["explanations"][:3]:
            typer.echo(f"   - {ex}")


@app.command()
def list_docs(
    type: str | None = typer.Option(None, "--type", "-t", help="Filter: resume or job_description"),
):
    """List stored documents."""
    doc_type = None
    if type:
        try:
            doc_type = DocumentType(type)
        except ValueError:
            typer.echo("--type must be 'resume' or 'job_description'", err=True)
            raise typer.Exit(1)
    docs = list_documents(doc_type)
    if not docs:
        typer.echo("No documents.")
        return
    for d in docs:
        typer.echo(f"  {d['id']}  {d['type']}  {d['created_at']}")


def main():
    app()


if __name__ == "__main__":
    main()
