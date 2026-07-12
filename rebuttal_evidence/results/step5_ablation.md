# T10 (E8): Step-5-only ablation (reviewer 1mvq Q6)

Full pipeline (Steps 1-5) vs Step-5-removed (Steps 1-4 + revised-reply target kept),
identical data/recipe/seed/eval; both retrained here because the original checkpoint
no longer exists on disk (full twin: micro-batch 8 x grad-accum 2, same effective batch).
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

## Responsiveness of corrections on ambiguous (harm) items
(share of corrections that still engage with the question's referents rather than evade;
Step 5 trains the correction to be *conditioned on the diagnosed content*)

| Bias | Model | responsive / harm items |
|---|---|---|
| gender | Full | 59/275 (21.5%) |
| gender | NoStep5 | 58/275 (21.1%) |
| nationality | Full | 11/150 (7.3%) |
| nationality | NoStep5 | 6/150 (4.0%) |
| disability | Full | 1/76 (1.3%) |
| disability | NoStep5 | 0/76 (0.0%) |

## Reading (honest)
- Removing Step 5 does NOT hurt correction accuracy at this checkpoint: MIC judgment and
  ROUGE-L are indistinguishable (p=.91/.42); BBQ gender/disability indistinguishable; on
  nationality the ablated model is actually higher under the Qwen judge (p<.001).
  Correction responsiveness on ambiguous items is also essentially identical.
- Therefore Step 5's contribution is NOT visible as raw benchmark accuracy. Its documented
  contribution is diagnosis-conditioning of the correction — established by the intervention
  experiments (Table 7), which remain the primary evidence. The rebuttal should present the
  ablation this way and temper any claim that Step 5 improves headline numbers.
- Caveats: both models are fresh retrains (the original checkpoint no longer exists);
  run-to-run variance against the original run's cached outputs (Qwen-judged .950/.900/.914)
  is comparable in size to the ablation deltas; single checkpoint (375 = epoch 3), single
  seed, one judge.
