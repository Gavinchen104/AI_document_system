#!/usr/bin/env python3
"""Ablation study: compare configs (embedding-only, LLM-only, full) and cost analysis."""

import json
import time
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.documents.schemas import DocumentType, JobProfile, ResumeProfile
from src.agents.extraction import extract_profile
from src.config import settings
from eval.run_eval import load_pairs, _heuristic_profile
from eval.scorers import score_embedding_only, score_keyword_overlap, score_llm_only

# Approximate pricing (USD per 1M tokens) - update as needed
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}


def _estimate_tokens(text: str) -> int:
    """Rough estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def run_ablation(data_path: Path, max_pairs: int | None = None) -> list[dict]:
    """Run ablation: embedding-only, LLM-only, full. Record latency and cost."""
    pairs = load_pairs(data_path)
    if max_pairs:
        pairs = pairs[:max_pairs]
    human_scores = np.array([p["human_score"] for p in pairs])
    has_api_key = bool(settings.openai_api_key)

    configs = []

    # 1. Embedding-only (no LLM)
    print("  Config: embedding_only...")
    scores, latencies = [], []
    for pair in pairs:
        rp = _heuristic_profile(pair["resume_text"], True)
        jp = _heuristic_profile(pair["job_text"], False)
        t0 = time.perf_counter()
        s = score_embedding_only(rp, jp)
        latencies.append(time.perf_counter() - t0)
        scores.append(s)
    scores_arr = np.array(scores)
    pearson_r, _ = pearsonr(human_scores, scores_arr)
    configs.append({
        "config": "embedding_only",
        "pearson_r": pearson_r,
        "avg_latency_s": np.mean(latencies),
        "cost_per_100_pairs_usd": 0.0,
        "notes": "No LLM; local or OpenAI embeddings only",
    })

    # 2. LLM-only (extraction + scoring, no embedding for match)
    if has_api_key:
        print("  Config: llm_only...")
        scores, latencies, total_input_tok, total_output_tok = [], [], 0, 0
        for i, pair in enumerate(pairs):
            t0 = time.perf_counter()
            rp = extract_profile("_ab_r", DocumentType.RESUME, pair["resume_text"], persist=False)
            jp = extract_profile("_ab_j", DocumentType.JOB_DESCRIPTION, pair["job_text"], persist=False)
            rp.raw_text, jp.raw_text = pair["resume_text"], pair["job_text"]
            s = score_llm_only(rp, jp)
            latencies.append(time.perf_counter() - t0)
            scores.append(s)
            total_input_tok += _estimate_tokens(pair["resume_text"] + pair["job_text"]) * 3  # extract x2 + score
            total_output_tok += 500  # rough per pair
        scores_arr = np.array(scores)
        pearson_r, _ = pearsonr(human_scores, scores_arr)
        price = PRICING.get(settings.llm_model, PRICING["gpt-4o-mini"])
        cost = (total_input_tok * price["input"] + total_output_tok * price["output"]) / 1e6
        cost_per_100 = cost * (100 / len(pairs)) if pairs else 0
        configs.append({
            "config": "llm_only",
            "pearson_r": pearson_r,
            "avg_latency_s": np.mean(latencies),
            "cost_per_100_pairs_usd": round(cost_per_100, 2),
            "notes": f"Extraction + scoring ({settings.llm_model})",
        })
    else:
        configs.append({
            "config": "llm_only",
            "pearson_r": np.nan,
            "avg_latency_s": np.nan,
            "cost_per_100_pairs_usd": np.nan,
            "notes": "Skipped (no OPENAI_API_KEY)",
        })

    # 3. Full pipeline (same as llm_only for scoring; embeddings used elsewhere)
    if has_api_key:
        configs.append({
            "config": "full_pipeline",
            "pearson_r": configs[-1]["pearson_r"],
            "avg_latency_s": configs[-1]["avg_latency_s"],
            "cost_per_100_pairs_usd": configs[-1]["cost_per_100_pairs_usd"],
            "notes": "LLM + embeddings (production)",
        })
    else:
        configs.append({
            "config": "full_pipeline",
            "pearson_r": np.nan,
            "avg_latency_s": np.nan,
            "cost_per_100_pairs_usd": np.nan,
            "notes": "Skipped (no OPENAI_API_KEY)",
        })

    return configs


def write_table(configs: list[dict], out_path: Path) -> None:
    """Write ablation table to markdown."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Ablation & Cost Analysis",
        "",
        "| Config | Pearson r | Avg Latency (s) | Cost per 100 pairs (USD) | Notes |",
        "|--------|-----------|-----------------|--------------------------|-------|",
    ]
    for c in configs:
        pr = f"{c['pearson_r']:.3f}" if not np.isnan(c["pearson_r"]) else "N/A"
        lat = f"{c['avg_latency_s']:.2f}" if not np.isnan(c.get("avg_latency_s", np.nan)) else "N/A"
        cost = f"{c['cost_per_100_pairs_usd']:.2f}" if not np.isnan(c.get("cost_per_100_pairs_usd", np.nan)) else "N/A"
        lines.append(f"| {c['config']} | {pr} | {lat} | {cost} | {c['notes']} |")
    lines.extend(["", "Run with OPENAI_API_KEY set for full results.", ""])
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "eval" / "data" / "sample_pairs.json"
    results_path = project_root / "eval" / "results" / "ablation_table.md"

    print("Running ablation study...")
    configs = run_ablation(data_path, max_pairs=5)
    write_table(configs, results_path)
    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    main()
