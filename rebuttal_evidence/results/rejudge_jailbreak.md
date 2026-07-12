# T6 (E2): Multi-judge re-scoring of jailbreak corrections (Llama-3B)

Judges: DeepSeek (original, cached), Qwen3-14B, Mistral-24B (local judges from
additional model families; GPT/Gemini API keys unavailable on this machine).
All judges score the IDENTICAL revised replies with the paper's own judging prompt.

| Task | Method | Paper | DeepSeek (cached) | Qwen3-14B | Mistral-24B |
|---|---|---|---|---|---|
| JB | CoT | 0.702 | 0.712 | 0.798 | 0.719 |
| JB | Heuristic | 0.883 | 0.871 | 0.821 | 0.821 |
| JB | Light | 0.714 | 0.714 | 0.743 | 0.695 |
| JB | Heavy | 0.883 | 0.883 | 0.840 | 0.848 |
| JB | Light+Heavy | 0.905 | 0.905 | 0.814 | 0.817 |

## Inter-judge agreement (per-example correct/incorrect, pooled)

- DS-QW: raw agreement = 0.803, Cohen's kappa = 0.361 (n=2100)
- DS-Mistral-24B: raw agreement = 0.817, Cohen's kappa = 0.433 (n=2100)
- QW-Mistral-24B: raw agreement = 0.836, Cohen's kappa = 0.502 (n=2100)
- All three judges: Fleiss' kappa = 0.433 (n=2100)

## Ranking preservation (method order by accuracy)

- JB [DeepSeek]: Light+Heavy > Heavy > Heuristic > Light > CoT
- JB [Qwen3-14B]: Heavy > Heuristic > Light+Heavy > CoT > Light
- JB [Mistral-24B]: Heavy > Heuristic > Light+Heavy > CoT > Light
  Spearman(method-accuracy) DS-QW=0.60, DS-Mistral-24B=0.60

