# Final multi-judge re-scoring (GPT-4.1, Gemini-2.5-flash, Qwen3-14B, phi-4/Mistral-24B)

Gemini judged a deterministic random 100-item subsample per set (API quota);
all other judges cover every item.

## BBQ (Llama-3B)

| Task | Method | Paper | DeepSeek | GPT-4.1 | Gemini-2.5 | Qwen3-14B | phi-4 |
|---|---|---|---|---|---|---|
| BBQ/gender | CoT | 0.446 | 0.794 | 0.758 | 0.830 | 0.822 | 0.866 |
| BBQ/gender | Heuristics | 0.77 | 0.872 | 0.804 | 0.870 | 0.838 | 0.872 |
| BBQ/gender | Light | 0.769 | 0.984 | 0.984 | 0.970 | 0.993 | 0.975 |
| BBQ/gender | Heavy | 0.909 | 0.938 | 0.876 | 0.890 | 0.950 | 0.952 |
| BBQ/nationality | Direct | 0.64 | 0.600 | 0.653 | 0.760 | 0.607 | 0.810 |
| BBQ/nationality | CoT | 0.467 | 0.713 | 0.770 | 0.830 | 0.747 | 0.893 |
| BBQ/nationality | Heuristics | 0.847 | 0.887 | 0.860 | 0.930 | 0.780 | 0.967 |
| BBQ/nationality | Light | 0.783 | 0.983 | 0.973 | 0.950 | 0.967 | 0.997 |
| BBQ/nationality | Heavy | 0.927 | 0.913 | 0.923 | 0.970 | 0.900 | 0.970 |
| BBQ/disability | Direct | 0.757 | 0.796 | 0.809 | 0.860 | 0.757 | 0.875 |
| BBQ/disability | CoT | 0.461 | 0.757 | 0.750 | 0.820 | 0.743 | 0.842 |
| BBQ/disability | Heuristics | 0.809 | 0.882 | 0.796 | 0.930 | 0.757 | 0.862 |
| BBQ/disability | Light | 0.894 | 0.993 | 0.993 | 0.970 | 0.895 | 0.974 |
| BBQ/disability | Heavy | 0.947 | 0.947 | 0.947 | 0.950 | 0.914 | 0.967 |

### Agreement with DeepSeek (per-example, each judge's own coverage)

- DeepSeek vs GPT-4.1: raw=0.921, kappa=0.673 (n=4310)
- DeepSeek vs Gemini-2.5: raw=0.917, kappa=0.625 (n=1400)
- DeepSeek vs Qwen3-14B: raw=0.940, kappa=0.753 (n=4310)
- DeepSeek vs phi-4: raw=0.910, kappa=0.533 (n=4310)
- Fleiss' kappa, all 5 judges on common items: 0.565 (n=1400)

### Rankings per judge

- BBQ/gender [DeepSeek]: Light > Heavy > Heuristics > CoT
- BBQ/gender [GPT-4.1]: Light > Heavy > Heuristics > CoT
- BBQ/gender [Gemini-2.5]: Light > Heavy > Heuristics > CoT
- BBQ/gender [Qwen3-14B]: Light > Heavy > Heuristics > CoT
- BBQ/gender [phi-4]: Light > Heavy > Heuristics > CoT
  Spearman vs DeepSeek: GPT-4.1=1.00, Gemini-2.5=1.00, Qwen3-14B=1.00, phi-4=1.00

- BBQ/nationality [DeepSeek]: Light > Heavy > Heuristics > CoT > Direct
- BBQ/nationality [GPT-4.1]: Light > Heavy > Heuristics > CoT > Direct
- BBQ/nationality [Gemini-2.5]: Heavy > Light > Heuristics > CoT > Direct
- BBQ/nationality [Qwen3-14B]: Light > Heavy > Heuristics > CoT > Direct
- BBQ/nationality [phi-4]: Light > Heavy > Heuristics > CoT > Direct
  Spearman vs DeepSeek: GPT-4.1=1.00, Gemini-2.5=0.90, Qwen3-14B=1.00, phi-4=1.00

- BBQ/disability [DeepSeek]: Light > Heavy > Heuristics > Direct > CoT
- BBQ/disability [GPT-4.1]: Light > Heavy > Direct > Heuristics > CoT
- BBQ/disability [Gemini-2.5]: Light > Heavy > Heuristics > Direct > CoT
- BBQ/disability [Qwen3-14B]: Heavy > Light > Direct > Heuristics > CoT
- BBQ/disability [phi-4]: Light > Heavy > Direct > Heuristics > CoT
  Spearman vs DeepSeek: GPT-4.1=0.90, Gemini-2.5=1.00, Qwen3-14B=0.87, phi-4=0.90

## JailbreakBench (Llama-3B)

| Task | Method | Paper | DeepSeek | GPT-4.1 | Gemini-2.5 | Qwen3-14B | Mistral-24B |
|---|---|---|---|---|---|---|
| JB | CoT | 0.702 | 0.712 | 0.910 | 0.950 | 0.798 | 0.719 |
| JB | Heuristic | 0.883 | 0.871 | 0.924 | 0.950 | 0.821 | 0.821 |
| JB | Light | 0.714 | 0.714 | 0.888 | 0.940 | 0.743 | 0.695 |
| JB | Heavy | 0.883 | 0.883 | 0.962 | 0.940 | 0.840 | 0.848 |
| JB | Light+Heavy | 0.905 | 0.905 | 0.940 | 0.950 | 0.814 | 0.817 |

### Agreement with DeepSeek (per-example, each judge's own coverage)

- DeepSeek vs GPT-4.1: raw=0.852, kappa=0.360 (n=2100)
- DeepSeek vs Gemini-2.5: raw=0.846, kappa=0.203 (n=500)
- DeepSeek vs Qwen3-14B: raw=0.803, kappa=0.361 (n=2100)
- DeepSeek vs Mistral-24B: raw=0.817, kappa=0.433 (n=2100)
- Fleiss' kappa, all 5 judges on common items: 0.289 (n=500)

### Rankings per judge

- JB [DeepSeek]: Light+Heavy > Heavy > Heuristic > Light > CoT
- JB [GPT-4.1]: Heavy > Light+Heavy > Heuristic > CoT > Light
- JB [Gemini-2.5]: CoT > Heuristic > Light+Heavy > Light > Heavy
- JB [Qwen3-14B]: Heavy > Heuristic > Light+Heavy > CoT > Light
- JB [Mistral-24B]: Heavy > Heuristic > Light+Heavy > CoT > Light
  Spearman vs DeepSeek: GPT-4.1=0.80, Gemini-2.5=0.00, Qwen3-14B=0.60, Mistral-24B=0.60

