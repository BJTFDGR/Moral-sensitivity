# T5 (E1): Multi-judge re-scoring of BBQ corrections (Llama-3B)

Judges: DeepSeek (original, cached), Qwen3-14B, phi-4 (local judges from
additional model families; GPT/Gemini API keys unavailable on this machine).
All judges score the IDENTICAL revised replies with the paper's own judging prompt.

Note on judge validation: Mistral-Small-24B was rejected as a BBQ judge — under the option-matching prompt it answers the question itself rather than matching the reply (on items DeepSeek scored wrong, Mistral returned the gold option 97/103 times; Cohen's kappa vs DeepSeek 0.15). This is itself evidence that LLM judges need validation; phi-4 (Microsoft) is used as the second family instead. Mistral remains a well-behaved judge for the binary jailbreak task.

| Task | Method | Paper | DeepSeek (cached) | Qwen3-14B | phi-4 |
|---|---|---|---|---|---|
| BBQ/gender | CoT | 0.446 | 0.794 | 0.822 | 0.866 |
| BBQ/gender | Heuristics | 0.77 | 0.872 | 0.838 | 0.872 |
| BBQ/gender | Light | 0.769 | 0.984 | 0.993 | 0.975 |
| BBQ/gender | Heavy | 0.909 | 0.938 | 0.950 | 0.952 |
| BBQ/nationality | Direct | 0.64 | 0.600 | 0.607 | 0.810 |
| BBQ/nationality | CoT | 0.467 | 0.713 | 0.747 | 0.893 |
| BBQ/nationality | Heuristics | 0.847 | 0.887 | 0.780 | 0.967 |
| BBQ/nationality | Light | 0.783 | 0.983 | 0.967 | 0.997 |
| BBQ/nationality | Heavy | 0.927 | 0.913 | 0.900 | 0.970 |
| BBQ/disability | Direct | 0.757 | 0.796 | 0.757 | 0.875 |
| BBQ/disability | CoT | 0.461 | 0.757 | 0.743 | 0.842 |
| BBQ/disability | Heuristics | 0.809 | 0.882 | 0.757 | 0.862 |
| BBQ/disability | Light | 0.894 | 0.993 | 0.895 | 0.974 |
| BBQ/disability | Heavy | 0.947 | 0.947 | 0.914 | 0.967 |

## Inter-judge agreement (per-example correct/incorrect, pooled)

- DS-QW: raw agreement = 0.940, Cohen's kappa = 0.753 (n=4310)
- DS-phi-4: raw agreement = 0.910, Cohen's kappa = 0.533 (n=4310)
- QW-phi-4: raw agreement = 0.897, Cohen's kappa = 0.500 (n=4310)
- All three judges: Fleiss' kappa = 0.604 (n=4310)

## Ranking preservation (method order by accuracy)

- BBQ/gender [DeepSeek]: Light > Heavy > Heuristics > CoT
- BBQ/gender [Qwen3-14B]: Light > Heavy > Heuristics > CoT
- BBQ/gender [phi-4]: Light > Heavy > Heuristics > CoT
  Spearman(method-accuracy) DS-QW=1.00, DS-phi-4=1.00

- BBQ/nationality [DeepSeek]: Light > Heavy > Heuristics > CoT > Direct
- BBQ/nationality [Qwen3-14B]: Light > Heavy > Heuristics > CoT > Direct
- BBQ/nationality [phi-4]: Light > Heavy > Heuristics > CoT > Direct
  Spearman(method-accuracy) DS-QW=1.00, DS-phi-4=1.00

- BBQ/disability [DeepSeek]: Light > Heavy > Heuristics > Direct > CoT
- BBQ/disability [Qwen3-14B]: Heavy > Light > Direct > Heuristics > CoT
- BBQ/disability [phi-4]: Light > Heavy > Direct > Heuristics > CoT
  Spearman(method-accuracy) DS-QW=0.87, DS-phi-4=0.90

