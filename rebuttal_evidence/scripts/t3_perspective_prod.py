import os
"""T3 (E4): Re-score RTP revised replies with the PRODUCTION Perspective TOXICITY attribute.

The paper's numbers used TOXICITY_EXPERIMENTAL (see ToxicityEvaluator in
diagnosis_toxicity.py). This re-scores the identical responses with the
stable production attribute and documents attribute/version/query date,
so every reported number stays verifiable offline (cached per-example scores).

API key is read at runtime from the project's own diagnosis_toxicity.py
(never copied anywhere).
"""
import ast
import json
import time
import urllib.request
import urllib.error
from datetime import date

BASE = "."


KEY = os.environ["PERSPECTIVE_API_KEY"]
URL = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={KEY}"


def score(text, retries=5):
    payload = {
        "comment": {"text": text[:20000]},
        "requestedAttributes": {"TOXICITY": {}},
        "doNotStore": True,
        "languages": ["en"],
    }
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
            return resp["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(2 ** attempt)
            elif e.code == 400:
                return None  # unscorable (empty/lang)
            else:
                time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return None


manifest = json.load(open(f"{BASE}/rebuttal/data/manifest.json"))
out_path = f"{BASE}/rebuttal/data/rtp_perspective_prod.json"
try:
    results = json.load(open(out_path))
except Exception:
    results = {}

for model, methods in manifest["RTP_table2"].items():
    if not isinstance(methods, dict):
        continue
    for method, info in methods.items():
        if not isinstance(info, dict) or not info.get("file"):
            continue
        key = f"{model}/{method}"
        data = json.load(open(info["file"]))
        texts = [d.get("response", "") or "" for d in data]
        exp = [d.get("toxicity") for d in data]
        prev = results.get(key, {}).get("toxicity_prod", [])
        scores = prev + [None] * (len(texts) - len(prev))
        t0 = time.time()
        for i, t in enumerate(texts):
            if scores[i] is not None:
                continue
            scores[i] = score(t) if t.strip() else None
            time.sleep(0.35)
            if i % 50 == 0:
                results[key] = {"file": info["file"], "toxicity_prod": scores,
                                "toxicity_experimental_cached": exp}
                json.dump(results, open(out_path, "w"))
                done = sum(s is not None for s in scores)
                print(f"{key}: {done}/{len(texts)} ({time.time()-t0:.0f}s)", flush=True)
        results[key] = {"file": info["file"], "toxicity_prod": scores,
                        "toxicity_experimental_cached": exp}
        json.dump(results, open(out_path, "w"))
        ok = [s for s in scores if s is not None]
        print(f"DONE {key}: n={len(ok)} mean_prod={sum(ok)/len(ok):.4f}", flush=True)

# summary
from statistics import mean
lines = [
    "# T3 (E4): Production TOXICITY re-score + Perspective API documentation",
    "",
    f"- Attribute originally used in the paper: `TOXICITY_EXPERIMENTAL` (see `ToxicityEvaluator`, diagnosis_toxicity.py)",
    f"- Re-scored attribute: production `TOXICITY`, Perspective API v1alpha1 (commentanalyzer.googleapis.com)",
    f"- Query date of this re-score: {date.today().isoformat()}",
    "- Cached per-example scores: rebuttal/data/rtp_perspective_prod.json (verifiable offline)",
    "",
    "| Model | Method | Paper (EXPERIMENTAL) | Production TOXICITY | n scored |",
    "|---|---|---|---|---|",
]
manifest_rows = []
for model, methods in manifest["RTP_table2"].items():
    if not isinstance(methods, dict):
        continue
    for method, info in methods.items():
        if not isinstance(info, dict) or not info.get("file"):
            continue
        key = f"{model}/{method}"
        r = results.get(key)
        if not r:
            continue
        ok = [s for s in r["toxicity_prod"] if s is not None]
        pairs = [(p, e) for p, e in zip(r["toxicity_prod"], r["toxicity_experimental_cached"]) if p is not None and e is not None]
        lines.append(f"| {model} | {method} | {info.get('paper')} | **{mean(ok):.3f}** | {len(ok)} |")
        manifest_rows.extend(pairs)

if manifest_rows:
    from scipy.stats import pearsonr, spearmanr
    pr = pearsonr([a for a, _ in manifest_rows], [b for _, b in manifest_rows])
    sr = spearmanr([a for a, _ in manifest_rows], [b for _, b in manifest_rows])
    lines += ["", f"Per-example correlation production-TOXICITY vs TOXICITY_EXPERIMENTAL "
                  f"(pooled n={len(manifest_rows)}): Pearson r = {pr[0]:.3f}, Spearman rho = {sr.statistic:.3f}."]

open(f"{BASE}/rebuttal/results/perspective_doc.md", "w").write("\n".join(lines) + "\n")
print("wrote results/perspective_doc.md")
