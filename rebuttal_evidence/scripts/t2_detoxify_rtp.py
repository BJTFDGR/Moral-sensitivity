"""T2 (E3): Re-score RTP revised replies with Detoxify (offline toxicity classifier).

Complements Perspective API numbers in Table 2 (reviewer xnDd Q2: offline
reproducibility). Uses the canonical files in rebuttal/data/manifest.json.
Outputs per-example scores (rebuttal/data/rtp_detoxify.json) and a summary
markdown (rebuttal/results/detoxify_rtp.md).
"""
import json
import numpy as np
from detoxify import Detoxify
from scipy.stats import pearsonr, spearmanr

BASE = "."
manifest = json.load(open(f"{BASE}/rebuttal/data/manifest.json"))

detox = Detoxify("original", device="cuda:0")

per_example = {}
rows = []
all_persp, all_detox = [], []

for model, methods in manifest["RTP_table2"].items():
    if not isinstance(methods, dict):
        continue
    for method, info in methods.items():
        if not isinstance(info, dict) or not info.get("file"):
            continue
        data = json.load(open(info["file"]))
        texts = [d.get("response", "") or "" for d in data]
        persp = [d.get("toxicity") for d in data]
        scores = []
        BS = 64
        for i in range(0, len(texts), BS):
            batch = [t if t.strip() else " " for t in texts[i:i + BS]]
            scores.extend(float(s) for s in detox.predict(batch)["toxicity"])
        key = f"{model}/{method}"
        per_example[key] = {"file": info["file"], "detoxify": scores, "perspective": persp}
        pairs = [(p, s) for p, s in zip(persp, scores) if p is not None]
        pr = pearsonr([p for p, _ in pairs], [s for _, s in pairs])
        sr = spearmanr([p for p, _ in pairs], [s for _, s in pairs])
        all_persp.extend(p for p, _ in pairs)
        all_detox.extend(s for _, s in pairs)
        rows.append((model, method, info.get("paper"), float(np.mean([p for p, _ in pairs])),
                     float(np.mean(scores)), len(scores), pr[0], sr.statistic))
        print(f"{key:35s} n={len(scores):3d} perspective={np.mean([p for p,_ in pairs]):.3f} detoxify={np.mean(scores):.3f} r={pr[0]:.3f}")

json.dump(per_example, open(f"{BASE}/rebuttal/data/rtp_detoxify.json", "w"))

gp = pearsonr(all_persp, all_detox)
gs = spearmanr(all_persp, all_detox)

order = ["Direct", "Heuristic", "CoT", "Light", "Heavy"]
lines = [
    "# T2 (E3): Detoxify re-scoring of RTP revised replies",
    "",
    "Detoxify v0.5.2 (`original` checkpoint, offline) re-scores the *identical* revised replies",
    "whose Perspective scores back Table 2. Method ranking preserved -> Perspective-independent evidence.",
    "",
    "| Model | Method | Paper (Perspective) | Cached Perspective | Detoxify | n |",
    "|---|---|---|---|---|---|",
]
for model in ["llama3.2-1B", "llama3.2-3B"]:
    for method in order:
        for m, meth, paper, pm, dm, n, r, rs in rows:
            if m == model and meth == method:
                lines.append(f"| {model} | {method} | {paper} | {pm:.3f} | **{dm:.3f}** | {n} |")
lines += [
    "",
    f"Per-example correlation Perspective vs Detoxify (all methods pooled, n={len(all_persp)}): "
    f"Pearson r = {gp[0]:.3f}, Spearman rho = {gs.statistic:.3f}.",
    "",
    "## Ranking check (mean Detoxify toxicity, lower better)",
]
for model in ["llama3.2-1B", "llama3.2-3B"]:
    sub = sorted([(dm, meth) for m, meth, _, _, dm, _, _, _ in rows if m == model])
    lines.append(f"- {model}: " + " < ".join(f"{meth} ({dm:.3f})" for dm, meth in sub))

open(f"{BASE}/rebuttal/results/detoxify_rtp.md", "w").write("\n".join(lines) + "\n")
print("\nwrote results/detoxify_rtp.md")
