# Evaluation Results

**Dataset:** 15 resume-job pairs with human-labeled fit scores (0-100)

| Strategy | Pearson r | Spearman r | RMSE |
|----------|-----------|------------|------|
| embedding_only | 0.720 | 0.376 | 19.3 |
| keyword_overlap | 0.417 | 0.210 | 33.3 |
| llm_only | N/A | N/A | N/A |
| full_pipeline | N/A | N/A | N/A |

Lower RMSE = better. Higher correlation = better.
