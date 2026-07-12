# T3 (E4): Production TOXICITY re-score + Perspective API documentation

- Attribute originally used in the paper: `TOXICITY_EXPERIMENTAL` (see `ToxicityEvaluator`, diagnosis_toxicity.py)
- Re-scored attribute: production `TOXICITY`, Perspective API v1alpha1 (commentanalyzer.googleapis.com)
- Query date of this re-score: 2026-07-11
- Cached per-example scores: rebuttal/data/rtp_perspective_prod.json (verifiable offline)

| Model | Method | Paper (EXPERIMENTAL) | Production TOXICITY | n scored |
|---|---|---|---|---|
| llama3.2-1B | Direct | 0.315 | **0.318** | 296 |
| llama3.2-1B | Heuristic | 0.429 | **0.426** | 494 |
| llama3.2-1B | CoT | 0.056 | **0.041** | 297 |
| llama3.2-1B | Light | 0.038 | **0.045** | 297 |
| llama3.2-1B | Heavy | 0.057 | **0.057** | 296 |
| llama3.2-3B | Direct | 0.187 | **0.188** | 297 |
| llama3.2-3B | Heuristic | 0.491 | **0.496** | 499 |
| llama3.2-3B | CoT | 0.039 | **0.039** | 298 |
| llama3.2-3B | Light | 0.037 | **0.037** | 297 |
| llama3.2-3B | Heavy | 0.045 | **0.045** | 298 |

Per-example correlation production-TOXICITY vs TOXICITY_EXPERIMENTAL (pooled n=3369): Pearson r = 0.996, Spearman rho = 1.000.
