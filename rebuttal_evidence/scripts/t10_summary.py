"""T10 (E8) summary: Step-5-only ablation head-to-head.

Both models trained by us with the identical recipe/seed on the identical 2,000
MIC instances; the ONLY difference is that the ablated variant removes the
Step-5 text ('(5) ... explanation tying the correction to the diagnosis')
from every disagree instance, keeping Steps 1-4 and the revised-reply target.
Evaluated through the identical pipeline (MIC-500; BBQ generation + Qwen3-14B
option-matching judge). Includes McNemar tests and a responsiveness/evasion
analysis of the BBQ corrections.
Writes rebuttal/results/step5_ablation.md
"""
import json
import re
import numpy as np
from scipy.stats import binomtest, wilcoxon

B = "."
full = json.load(open(f"{B}/rebuttal/data/eval_pragmaticFull.json"))
abl = json.load(open(f"{B}/rebuttal/data/eval_noStep5.json"))


def mcnemar(pairs):
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if not x and y)
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p


def is_unknown(t):
    t = t.lower()
    return any(u in t for u in ["cannot", "can't", "unknown", "undetermined", "not answerable",
                                "not enough", "not known", "no answer"])


def person_options(choices):
    return [t.strip().rstrip('.').strip() for _, t in re.findall(r'\(([a-c])\)\s*([^(]+)', choices)
            if not is_unknown(t)]


lines = ["# T10 (E8): Step-5-only ablation (reviewer 1mvq Q6)", "",
         "Full pipeline (Steps 1-5) vs Step-5-removed (Steps 1-4 + revised-reply target kept),",
         "identical data/recipe/seed/eval; both retrained here because the original checkpoint",
         "no longer exists on disk (full twin: micro-batch 8 x grad-accum 2, same effective batch).",
         "BBQ judged by Qwen3-14B (option matching, paper prompt).", ""]

# MIC
fm, am = full["MIC"], abl["MIC"]
jp = [(int(r1["pred"] == r1["gold"]), int(r2["pred"] == r2["gold"]))
      for r1, r2 in zip(fm["records"], am["records"])]
b, c, p = mcnemar(jp)
rl_pairs = [(r1["rougeL"], r2["rougeL"]) for r1, r2 in zip(fm["records"], am["records"])
            if r1["rougeL"] is not None and r2["rougeL"] is not None]
wp = wilcoxon([a for a, _ in rl_pairs], [b_ for _, b_ in rl_pairs]).pvalue if rl_pairs else float("nan")
lines += ["## MIC (n=500)", "",
          "| Metric | Full (1-5) | No-Step-5 | test |",
          "|---|---|---|---|",
          f"| Judgment accuracy | {fm['judgment_acc']:.3f} | {am['judgment_acc']:.3f} | McNemar b={b}, c={c}, p={p:.3f} |",
          f"| ROUGE-L (disagree revisions) | {fm['rougeL_disagree']:.3f} | {am['rougeL_disagree']:.3f} | Wilcoxon p={wp:.3f} (n={len(rl_pairs)}) |",
          ""]

# BBQ
lines += ["## BBQ correction accuracy (Qwen3-14B judge)", "",
          "| Bias | Full (1-5) | No-Step-5 | McNemar |",
          "|---|---|---|---|"]
resp_stats = []
for bias in ["gender", "nationality", "disability"]:
    fr = full[f"BBQ_gen/{bias}"]
    ar = abl[f"BBQ_gen/{bias}"]
    pairs = [(r1["judge_correct"], r2["judge_correct"]) for r1, r2 in zip(fr, ar)]
    accF = np.mean([x for x, _ in pairs]); accA = np.mean([y for _, y in pairs])
    b, c, p = mcnemar(pairs)
    lines.append(f"| {bias} | {accF:.3f} | {accA:.3f} | b={b}, c={c}, p={p:.3f} |")
    for tag, recs in [("Full", fr), ("NoStep5", ar)]:
        harm = [r for r in recs if r["flag"] == "harm"]
        resp = sum(1 for r in harm
                   if any(p_.lower()[:14] in r["extracted_reply"].lower() for p_ in person_options(r["choices"])))
        resp_stats.append((bias, tag, resp, len(harm)))
lines += ["", "## Responsiveness of corrections on ambiguous (harm) items",
          "(share of corrections that still engage with the question's referents rather than evade;",
          "Step 5 trains the correction to be *conditioned on the diagnosed content*)", "",
          "| Bias | Model | responsive / harm items |", "|---|---|---|"]
for bias, tag, resp, n in resp_stats:
    lines.append(f"| {bias} | {tag} | {resp}/{n} ({resp/n:.1%}) |")

lines += ["", "## Reading (honest)",
          "- Removing Step 5 does NOT hurt correction accuracy at this checkpoint: MIC judgment and",
          "  ROUGE-L are indistinguishable (p=.91/.42); BBQ gender/disability indistinguishable; on",
          "  nationality the ablated model is actually higher under the Qwen judge (p<.001).",
          "  Correction responsiveness on ambiguous items is also essentially identical.",
          "- Therefore Step 5's contribution is NOT visible as raw benchmark accuracy. Its documented",
          "  contribution is diagnosis-conditioning of the correction — established by the intervention",
          "  experiments (Table 7), which remain the primary evidence. The rebuttal should present the",
          "  ablation this way and temper any claim that Step 5 improves headline numbers.",
          "- Caveats: both models are fresh retrains (the original checkpoint no longer exists);",
          "  run-to-run variance against the original run's cached outputs (Qwen-judged .950/.900/.914)",
          "  is comparable in size to the ablation deltas; single checkpoint (375 = epoch 3), single",
          "  seed, one judge."]

open(f"{B}/rebuttal/results/step5_ablation.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
