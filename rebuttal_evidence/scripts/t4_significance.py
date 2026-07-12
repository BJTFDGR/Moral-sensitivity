"""T4 (E5): Statistical significance tests for the main tables (reviewer LVZS Q5).

- RTP (Table 2, continuous per-example Perspective toxicity): paired Wilcoxon
  signed-rank + paired-bootstrap 95% CI of the mean difference, examples
  aligned across methods by the embedded <Prompt> text.
- BBQ (Table 3) and Jailbreak (Table 4) (binary per-example correctness):
  exact McNemar test on discordant pairs, items aligned by (Question, Answer).

Outputs rebuttal/results/significance.md
"""
import json
import re
import numpy as np
from scipy.stats import wilcoxon, binomtest

BASE = "."
manifest = json.load(open(f"{BASE}/rebuttal/data/manifest.json"))
rng = np.random.default_rng(42)
lines = ["# T4 (E5): Significance tests", ""]


def boot_ci(diff, n=10000):
    diff = np.asarray(diff)
    means = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


# ---------- RTP ----------
def rtp_key(item):
    m = re.search(r'<Prompt>: "(.*?)"; <Reply>:', item["prompt"][0]["generated_text"], re.DOTALL)
    return m.group(1).strip() if m else None


lines += ["## RTP / Table 2 (per-example Perspective toxicity, lower better)",
          "Paired Wilcoxon signed-rank; paired bootstrap 95% CI of mean(A-B). Negative = A less toxic.", "",
          "| Model | A vs B | n pairs | mean A | mean B | mean diff [95% CI] | Wilcoxon p |",
          "|---|---|---|---|---|---|---|"]

for model in ["llama3.2-1B", "llama3.2-3B"]:
    methods = manifest["RTP_table2"][model]
    data = {}
    for meth, info in methods.items():
        if not isinstance(info, dict) or not info.get("file"):
            continue
        items = json.load(open(info["file"]))
        data[meth] = {rtp_key(it): it["toxicity"] for it in items if rtp_key(it) and it.get("toxicity") is not None}
    for a, b in [("Light", "CoT"), ("Light", "Heavy"), ("Light", "Direct"), ("Light", "Heuristic"), ("Heavy", "CoT")]:
        if a not in data or b not in data:
            continue
        keys = sorted(set(data[a]) & set(data[b]))
        da = np.array([data[a][k] for k in keys]); db = np.array([data[b][k] for k in keys])
        diff = da - db
        try:
            p = wilcoxon(da, db).pvalue
        except ValueError:
            p = float("nan")
        lo, hi = boot_ci(diff)
        star = " **sig**" if p < 0.05 else ""
        lines.append(f"| {model} | {a} vs {b} | {len(keys)} | {da.mean():.3f} | {db.mean():.3f} | "
                     f"{diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {p:.2e}{star} |")
lines.append("")


# ---------- McNemar helper ----------
def mcnemar(rows):
    b = sum(1 for x, y in rows if x == 1 and y == 0)   # A right, B wrong
    c = sum(1 for x, y in rows if x == 0 and y == 1)   # A wrong, B right
    if b + c == 0:
        return b, c, 1.0
    return b, c, binomtest(b, b + c, 0.5).pvalue


# ---------- BBQ ----------
lines += ["## BBQ / Table 3 (Llama-3B, binary is_correct; exact McNemar)",
          "b = Heavy right & baseline wrong; c = Heavy wrong & baseline right.", "",
          "> CAVEAT: the cached Light(semantic) BBQ files come from a later judging round than Table 3",
          "> (cached acc .98 vs paper .769-.894): Light often rewrites the reply into an evasive non-answer",
          "> (e.g. 'I was in a computer store the other day.'), which the option-matching judge maps to",
          "> 'cannot be determined' (gold for ambiguous items). Heavy-vs-Light rows below inherit this",
          "> leniency; the multi-judge re-scoring (T5) and error analysis (T7) address it. Heavy-vs-CoT and",
          "> Heavy-vs-Heuristics rows are unaffected (those runs match the logged evaluation round).", "",
          "| Bias | Heavy vs | n pairs | acc Heavy | acc base | b | c | McNemar p |",
          "|---|---|---|---|---|---|---|---|"]
for bias in ["gender", "nationality", "disability"]:
    cell = manifest["BBQ_table3_llama3.2-3B"][bias]
    data = {}
    for meth, info in cell.items():
        if not isinstance(info, dict) or not info.get("file"):
            continue
        items = json.load(open(info["file"]))
        if len(items) < 50 or "Question" not in items[0] or "is_correct" not in items[0]:
            continue  # incomplete cache (e.g. Direct n=10)
        data[meth] = {(it["Question"].strip(), it["Answer"].strip()): int(it.get("is_correct", 0)) for it in items}
    for base_m in ["Light", "Heuristics", "CoT", "Direct"]:
        if base_m not in data or "Heavy" not in data:
            continue
        keys = sorted(set(data["Heavy"]) & set(data[base_m]))
        rows = [(data["Heavy"][k], data[base_m][k]) for k in keys]
        accH = np.mean([r[0] for r in rows]); accB = np.mean([r[1] for r in rows])
        b, c, p = mcnemar(rows)
        star = " **sig**" if p < 0.05 else ""
        lines.append(f"| {bias} | {base_m} | {len(keys)} | {accH:.3f} | {accB:.3f} | {b} | {c} | {p:.2e}{star} |")
lines.append("")

# ---------- Jailbreak ----------
lines += ["## Jailbreak / Table 4 (Llama-3B, binary llm_correct; exact McNemar)",
          "", "| Light+Heavy vs | n pairs | acc L+H | acc base | b | c | McNemar p |",
          "|---|---|---|---|---|---|---|"]
cell = manifest["Jailbreak_table4_llama3.2-3B"]
data = {}
for meth, info in cell.items():
    if not isinstance(info, dict) or not info.get("file"):
        continue
    items = json.load(open(info["file"]))
    data[meth] = {(it["Question"].strip()[:200], it["Answer"].strip()[:200]): int(it.get("llm_correct", 0)) for it in items}
for base_m in ["Heavy", "Light", "Heuristic", "CoT"]:
    if base_m not in data or "Light+Heavy" not in data:
        continue
    keys = sorted(set(data["Light+Heavy"]) & set(data[base_m]))
    rows = [(data["Light+Heavy"][k], data[base_m][k]) for k in keys]
    accH = np.mean([r[0] for r in rows]); accB = np.mean([r[1] for r in rows])
    b, c, p = mcnemar(rows)
    star = " **sig**" if p < 0.05 else ""
    lines.append(f"| {base_m} | {len(keys)} | {accH:.3f} | {accB:.3f} | {b} | {c} | {p:.2e}{star} |")

out = f"{BASE}/rebuttal/results/significance.md"
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
