#!/usr/bin/env python3
"""Run evaluation benchmark: compare scoring strategies vs human labels."""

import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.documents.schemas import DocumentType, JobProfile, ResumeProfile
from src.agents.extraction import extract_profile
from src.config import settings
from eval.scorers import (
    score_embedding_only,
    score_keyword_overlap,
    score_llm_only,
    score_full_pipeline,
)

# Common tech words for heuristic extraction when API key is missing
SKILLS_PATTERN = re.compile(
    r"\b(Python|Java|JavaScript|TypeScript|Go|Rust|C\+\+|C#|SQL|React|Node\.?js|"
    r"FastAPI|Django|Flask|Kubernetes|AWS|Docker|Terraform|PostgreSQL|MongoDB|"
    r"TensorFlow|PyTorch|ML|NLP|Kafka|Spark|Airflow|Git)\b",
    re.I,
)


def _heuristic_profile(resume_text: str, is_resume: bool) -> ResumeProfile | JobProfile:
    """Build minimal profile from raw text when API key missing (for embedding/keyword only)."""
    skills = list(set(SKILLS_PATTERN.findall(resume_text)))
    years_match = re.search(r"(\d+)\s* years?", resume_text, re.I)
    years = int(years_match.group(1)) if years_match else None
    if is_resume:
        return ResumeProfile(
            raw_text=resume_text,
            summary=resume_text[:300],
            skills=skills,
            years_experience=years,
        )
    return JobProfile(
        raw_text=resume_text,
        summary=resume_text[:300],
        skills=skills,
        years_experience=years,
    )


def load_pairs(data_path: Path) -> list[dict]:
    """Load eval pairs: [{resume_text, job_text, human_score}, ...]."""
    with open(data_path) as f:
        data = json.load(f)
    return data


def run_eval(data_path: Path, results_dir: Path, use_llm: bool = True) -> dict:
    """Run all strategies and compute correlations."""
    pairs = load_pairs(data_path)
    human_scores = np.array([p["human_score"] for p in pairs])
    has_api_key = bool(settings.openai_api_key)

    strategies = {
        "embedding_only": [],
        "keyword_overlap": [],
        "llm_only": [],
        "full_pipeline": [],
    }

    for i, pair in enumerate(pairs):
        resume_text = pair["resume_text"]
        job_text = pair["job_text"]
        print(f"  Pair {i+1}/{len(pairs)}: ", end="", flush=True)

        if has_api_key and use_llm:
            print("extracting...", end=" ", flush=True)
            try:
                resume_profile = extract_profile(
                    "_eval_r", DocumentType.RESUME, resume_text, persist=False
                )
                job_profile = extract_profile(
                    "_eval_j", DocumentType.JOB_DESCRIPTION, job_text, persist=False
                )
            except Exception as e:
                print(f"Extraction failed: {e}")
                raise
            resume_profile.raw_text = resume_text
            job_profile.raw_text = job_text
        else:
            # No API key or --no-llm: use heuristic profiles (embedding + keyword only)
            resume_profile = _heuristic_profile(resume_text, is_resume=True)
            job_profile = _heuristic_profile(job_text, is_resume=False)

        print("scoring...", end=" ", flush=True)

        strategies["embedding_only"].append(score_embedding_only(resume_profile, job_profile))
        strategies["keyword_overlap"].append(score_keyword_overlap(resume_profile, job_profile))
        if has_api_key and use_llm:
            strategies["llm_only"].append(score_llm_only(resume_profile, job_profile))
            strategies["full_pipeline"].append(score_full_pipeline(resume_profile, job_profile))
        else:
            strategies["llm_only"].append(np.nan)
            strategies["full_pipeline"].append(np.nan)

        print("done")

    # Compute correlations (skip LLM strategies if no scores)
    results = []
    for name, scores in strategies.items():
        scores_arr = np.array(scores, dtype=float)
        valid = ~np.isnan(scores_arr)
        if valid.sum() < 2:
            results.append({
                "strategy": name,
                "pearson_r": np.nan,
                "pearson_p": np.nan,
                "spearman_r": np.nan,
                "spearman_p": np.nan,
                "rmse": np.nan,
            })
            continue
        h, s = human_scores[valid], scores_arr[valid]
        pearson_r, pearson_p = pearsonr(h, s)
        spearman_r, spearman_p = spearmanr(h, s)
        rmse = np.sqrt(np.mean((h - s) ** 2))
        results.append({
            "strategy": name,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "rmse": rmse,
        })

    return {"results": results, "n_pairs": len(pairs)}


def write_table(results: dict, out_path: Path) -> None:
    """Write markdown table to results dir."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Evaluation Results",
        "",
        f"**Dataset:** {results['n_pairs']} resume-job pairs with human-labeled fit scores (0-100)",
        "",
        "| Strategy | Pearson r | Spearman r | RMSE |",
        "|----------|-----------|------------|------|",
    ]
    for r in results["results"]:
        pr = f"{r['pearson_r']:.3f}" if not np.isnan(r["pearson_r"]) else "N/A"
        sr = f"{r['spearman_r']:.3f}" if not np.isnan(r["spearman_r"]) else "N/A"
        rm = f"{r['rmse']:.1f}" if not np.isnan(r["rmse"]) else "N/A"
        lines.append(f"| {r['strategy']} | {pr} | {sr} | {rm} |")
    lines.extend(["", "Lower RMSE = better. Higher correlation = better.", ""])
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM strategies (embedding + keyword only)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "eval" / "data" / "sample_pairs.json"
    results_dir = project_root / "eval" / "results"
    results_path = results_dir / "eval_table.md"

    use_llm = not args.no_llm
    print("Running evaluation benchmark...")
    print(f"  Data: {data_path}")
    if not use_llm or not settings.openai_api_key:
        print("  Mode: embedding + keyword only (set OPENAI_API_KEY for full eval)")
    results = run_eval(data_path, results_dir, use_llm=use_llm)
    write_table(results, results_path)
    print(f"\nResults written to {results_path}")
    for r in results["results"]:
        pr = f"{r['pearson_r']:.3f}" if not np.isnan(r["pearson_r"]) else "N/A"
        sr = f"{r['spearman_r']:.3f}" if not np.isnan(r["spearman_r"]) else "N/A"
        rm = f"{r['rmse']:.1f}" if not np.isnan(r["rmse"]) else "N/A"
        print(f"  {r['strategy']}: Pearson={pr}, Spearman={sr}, RMSE={rm}")


if __name__ == "__main__":
    main()
