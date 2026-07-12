"""Final multi-judge summary across ALL judges.
BBQ: DeepSeek (cached) + GPT-4.1 + Gemini-2.5-flash(subsample) + Qwen3-14B + phi-4
JB:  DeepSeek (cached) + GPT-4.1 + Gemini-2.5-flash(subsample) + Qwen3-14B + Mistral-24B
Gemini judged a deterministic 100-item subsample per set (free-tier quota);
its accuracy/agreement are computed on that subsample; Fleiss' kappa uses items
judged by every judge (i.e. the Gemini subsample) plus a full-coverage variant.
Writes rebuttal/results/final_judges.md
"""
import json
import os
import numpy as np
from scipy.stats import spearmanr

B = "."
man = json.load(open(f"{B}/rebuttal/data/manifest.json"))
FILES = {"GPT-4.1": "rejudge_gpt-4.1.json", "Gemini-2.5": "rejudge_gemini-2.5-flash.json",
         "Qwen3-14B": "rejudge_Qwen3-14B.json", "phi-4": "rejudge_phi-4.json",
         "Mistral-24B": "rejudge_Mistral-Small-24B-Instruct-2501.json"}
data = {n: json.load(open(f"{B}/rebuttal/data/{f}")) for n, f in FILES.items() if os.path.exists(f"{B}/rebuttal/data/{f}")}


def recs_map(recs):
    """src_idx -> (judge_correct, deepseek_correct)"""
    return {r.get("src_idx", i): (r["judge_correct"], r["deepseek_correct"]) for i, r in enumerate(recs)}


def cohen(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pe = sum(((a == c).mean()) * ((b == c).mean()) for c in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def fleiss(votes):
    from collections import Counter
    n = len(votes[0]); N = len(votes)
    cats = sorted({v for r in votes for v in r})
    pj = {c: sum(r.count(c) for r in votes) / (N * n) for c in cats}
    Pi = [(sum(v * v for v in Counter(r).values()) - n) / (n * (n - 1)) for r in votes]
    Pe = sum(p * p for p in pj.values())
    return (sum(Pi) / N - Pe) / (1 - Pe) if Pe < 1 else 1.0


def block(keys, judges, title):
    judges = [j for j in judges if j in data]
    lines = [f"## {title}", "", "| Task | Method | Paper | DeepSeek | " + " | ".join(judges) + " |",
             "|" + "---|" * (len(judges) + 3)]
    kap = {j: ([], []) for j in judges}
    ranks = {}
    votes_all, votes_full = [], []
    for key, meta in keys:
        maps = {j: recs_map(data[j][key]) for j in judges if key in data[j]}
        if not maps:
            continue
        anchor = maps[judges[0]]
        ds_full = {i: dsc for i, (_, dsc) in anchor.items()}
        row = [np.mean([d for _, d in ds_full.items()])]
        for j in judges:
            m = maps.get(j, {})
            row.append(np.mean([jc for jc, _ in m.values()]) if m else float("nan"))
            kap[j][0].extend(dsc for _, dsc in m.values())
            kap[j][1].extend(jc for jc, _ in m.values())
        # all-judge Fleiss on intersection of indices
        common = set(ds_full)
        for j in judges:
            common &= set(maps.get(j, {}))
        for i in sorted(common):
            votes_all.append([ds_full[i]] + [maps[j][i][0] for j in judges])
        # full-coverage Fleiss (exclude subsampled judges)
        fullj = [j for j in judges if len(maps.get(j, {})) == len(anchor)]
        for i in sorted(ds_full):
            votes_full.append([ds_full[i]] + [maps[j][i][0] for j in fullj if i in maps[j]])
        task, meth = key.rsplit("/", 1)
        ranks.setdefault(task, []).append((meth, row))
        lines.append(f"| {task} | {meth} | {meta.get('paper','-')} | " +
                     " | ".join(f"{a:.3f}" for a in row) + " |")
    lines += ["", "### Agreement with DeepSeek (per-example, each judge's own coverage)", ""]
    for j, (a, b) in kap.items():
        if a:
            lines.append(f"- DeepSeek vs {j}: raw={np.mean(np.array(a)==np.array(b)):.3f}, "
                         f"kappa={cohen(a,b):.3f} (n={len(a)})")
    if votes_all:
        lines.append(f"- Fleiss' kappa, all {len(judges)+1} judges on common items: "
                     f"{fleiss(votes_all):.3f} (n={len(votes_all)})")
    lines += ["", "### Rankings per judge", ""]
    for task, rows in ranks.items():
        names = ["DeepSeek"] + judges
        for ji, jn in enumerate(names):
            order = sorted(rows, key=lambda x: -(x[1][ji] if not np.isnan(x[1][ji]) else -1))
            lines.append(f"- {task} [{jn}]: " + " > ".join(m for m, _ in order))
        if len(rows) > 2:
            ds_acc = [r[1][0] for r in rows]
            cors = ", ".join(f"{jn}={spearmanr(ds_acc, [r[1][ji+1] for r in rows]).statistic:.2f}"
                             for ji, jn in enumerate(judges))
            lines.append(f"  Spearman vs DeepSeek: {cors}")
        lines.append("")
    return lines


bbq_keys = [(f"BBQ/{bias}/{meth}", man["BBQ_table3_llama3.2-3B"][bias].get(meth, {}))
            for bias in ["gender", "nationality", "disability"]
            for meth in ["Direct", "CoT", "Heuristics", "Light", "Heavy"]
            if isinstance(man["BBQ_table3_llama3.2-3B"][bias].get(meth), dict)]
jb_keys = [(f"JB/{meth}", man["Jailbreak_table4_llama3.2-3B"].get(meth, {}))
           for meth in ["CoT", "Heuristic", "Light", "Heavy", "Light+Heavy"]]

lines = ["# Final multi-judge re-scoring (GPT-4.1, Gemini-2.5-flash, Qwen3-14B, phi-4/Mistral-24B)",
         "", "Gemini judged a deterministic random 100-item subsample per set (API quota);",
         "all other judges cover every item.", ""]
lines += block(bbq_keys, ["GPT-4.1", "Gemini-2.5", "Qwen3-14B", "phi-4"], "BBQ (Llama-3B)")
lines += block(jb_keys, ["GPT-4.1", "Gemini-2.5", "Qwen3-14B", "Mistral-24B"], "JailbreakBench (Llama-3B)")
open(f"{B}/rebuttal/results/final_judges.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
