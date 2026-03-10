# Ablation & Cost Analysis

| Config | Pearson r | Avg Latency (s) | Cost per 100 pairs (USD) | Notes |
|--------|-----------|-----------------|--------------------------|-------|
| embedding_only | 0.993 | 2.94 | 0.00 | No LLM; local or OpenAI embeddings only |
| llm_only | N/A | N/A | N/A | Skipped (no OPENAI_API_KEY) |
| full_pipeline | N/A | N/A | N/A | Skipped (no OPENAI_API_KEY) |

Run with OPENAI_API_KEY set for full results.
