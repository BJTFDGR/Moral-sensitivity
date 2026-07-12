# T4 (E5): Significance tests

## RTP / Table 2 (per-example Perspective toxicity, lower better)
Paired Wilcoxon signed-rank; paired bootstrap 95% CI of mean(A-B). Negative = A less toxic.

| Model | A vs B | n pairs | mean A | mean B | mean diff [95% CI] | Wilcoxon p |
|---|---|---|---|---|---|---|
| llama3.2-1B | Light vs CoT | 300 | 0.038 | 0.041 | -0.003 [-0.018, +0.012] | 9.95e-01 |
| llama3.2-1B | Light vs Heavy | 300 | 0.038 | 0.057 | -0.019 [-0.039, -0.001] | 3.07e-01 |
| llama3.2-1B | Light vs Direct | 300 | 0.038 | 0.315 | -0.277 [-0.315, -0.239] | 1.11e-34 **sig** |
| llama3.2-1B | Light vs Heuristic | 300 | 0.038 | 0.428 | -0.390 [-0.428, -0.353] | 2.60e-41 **sig** |
| llama3.2-1B | Heavy vs CoT | 300 | 0.057 | 0.041 | +0.016 [+0.001, +0.033] | 1.08e-01 |
| llama3.2-3B | Light vs CoT | 300 | 0.037 | 0.039 | -0.001 [-0.008, +0.005] | 8.77e-01 |
| llama3.2-3B | Light vs Heavy | 300 | 0.037 | 0.045 | -0.008 [-0.021, +0.004] | 7.37e-01 |
| llama3.2-3B | Light vs Direct | 300 | 0.037 | 0.187 | -0.150 [-0.176, -0.125] | 1.79e-32 **sig** |
| llama3.2-3B | Light vs Heuristic | 300 | 0.037 | 0.498 | -0.461 [-0.496, -0.425] | 3.37e-43 **sig** |
| llama3.2-3B | Heavy vs CoT | 300 | 0.045 | 0.039 | +0.007 [-0.004, +0.018] | 6.84e-01 |

## BBQ / Table 3 (Llama-3B, binary is_correct; exact McNemar)
b = Heavy right & baseline wrong; c = Heavy wrong & baseline right.

> CAVEAT: the cached Light(semantic) BBQ files come from a later judging round than Table 3
> (cached acc .98 vs paper .769-.894): Light often rewrites the reply into an evasive non-answer
> (e.g. 'I was in a computer store the other day.'), which the option-matching judge maps to
> 'cannot be determined' (gold for ambiguous items). Heavy-vs-Light rows below inherit this
> leniency; the multi-judge re-scoring (T5) and error analysis (T7) address it. Heavy-vs-CoT and
> Heavy-vs-Heuristics rows are unaffected (those runs match the logged evaluation round).

| Bias | Heavy vs | n pairs | acc Heavy | acc base | b | c | McNemar p |
|---|---|---|---|---|---|---|---|
| gender | Light | 500 | 0.938 | 0.982 | 9 | 31 | 6.80e-04 **sig** |
| gender | Heuristics | 500 | 0.938 | 0.872 | 58 | 25 | 3.78e-04 **sig** |
| gender | CoT | 500 | 0.938 | 0.794 | 91 | 19 | 1.87e-12 **sig** |
| nationality | Light | 300 | 0.913 | 0.983 | 5 | 26 | 1.92e-04 **sig** |
| nationality | Heuristics | 300 | 0.913 | 0.887 | 29 | 21 | 3.22e-01 |
| nationality | CoT | 300 | 0.913 | 0.713 | 75 | 15 | 9.18e-11 **sig** |
| nationality | Direct | 300 | 0.913 | 0.600 | 116 | 22 | 1.28e-16 **sig** |
| disability | Light | 149 | 0.946 | 0.993 | 1 | 8 | 3.91e-02 **sig** |
| disability | Heuristics | 149 | 0.946 | 0.879 | 16 | 6 | 5.25e-02 |
| disability | CoT | 149 | 0.946 | 0.772 | 31 | 5 | 1.29e-05 **sig** |
| disability | Direct | 149 | 0.946 | 0.805 | 27 | 6 | 3.24e-04 **sig** |

## Jailbreak / Table 4 (Llama-3B, binary llm_correct; exact McNemar)

| Light+Heavy vs | n pairs | acc L+H | acc base | b | c | McNemar p |
|---|---|---|---|---|---|---|
| Heavy | 420 | 0.905 | 0.883 | 30 | 21 | 2.62e-01 |
| Light | 420 | 0.905 | 0.714 | 95 | 15 | 2.14e-15 **sig** |
| Heuristic | 420 | 0.905 | 0.871 | 40 | 26 | 1.09e-01 |
| CoT | 420 | 0.905 | 0.712 | 95 | 14 | 5.77e-16 **sig** |
