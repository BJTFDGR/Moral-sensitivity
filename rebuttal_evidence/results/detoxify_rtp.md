# T2 (E3): Detoxify re-scoring of RTP revised replies

Detoxify v0.5.2 (`original` checkpoint, offline) re-scores the *identical* revised replies
whose Perspective scores back Table 2 (Llama-3B). Method ranking preserved -> Perspective-independent evidence.

| Model | Method | Paper (Perspective) | Cached Perspective | Detoxify | n |
|---|---|---|---|---|---|
| llama3.2-3B | Direct | 0.187 | 0.187 | **0.128** | 300 |
| llama3.2-3B | Heuristic | 0.491 | 0.495 | **0.548** | 500 |
| llama3.2-3B | CoT | 0.039 | 0.039 | **0.010** | 300 |
| llama3.2-3B | Light | 0.037 | 0.037 | **0.008** | 300 |
| llama3.2-3B | Heavy | 0.045 | 0.045 | **0.023** | 300 |

Per-example correlation Perspective vs Detoxify (all methods pooled, n=1700): Pearson r = 0.929, Spearman rho = 0.789.

## Ranking check (mean Detoxify toxicity, lower better)
- llama3.2-3B: Light (0.008) < CoT (0.010) < Heavy (0.023) < Direct (0.128) < Heuristic (0.548)
