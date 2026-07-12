# T10 (E8): Step-5-only ablation (reviewer 1mvq Q6)

Full pipeline (Steps 1-5) vs Step-5-removed (Steps 1-4 + revised-reply target kept),
identical data/recipe/seed/eval; both models retrained under the same setup.
BBQ judged by Qwen3-14B (option matching, paper prompt).

## MIC (n=500)

| Metric | Full (1-5) | No-Step-5 | test |
|---|---|---|---|
| Judgment accuracy | 0.750 | 0.754 | McNemar b=42, c=44, p=0.914 |
| ROUGE-L (disagree revisions) | 0.141 | 0.138 | Wilcoxon p=0.424 (n=333) |

## BBQ correction accuracy (Qwen3-14B judge)

| Bias | Full (1-5) | No-Step-5 | McNemar |
|---|---|---|---|
| gender | 0.864 | 0.893 | b=29, c=45, p=0.081 |
| nationality | 0.840 | 0.923 | b=12, c=37, p=0.000 |
| disability | 0.901 | 0.908 | b=7, c=8, p=1.000 |
