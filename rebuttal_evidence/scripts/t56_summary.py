"""T5/T6 summary (E1/E2): multi-judge accuracy tables + inter-judge agreement.

Judges: DeepSeek (cached, original) + Qwen3-14B + Mistral-Small-24B (both local,
different families). Reports per-method accuracy under each judge, pairwise
Cohen's kappa on per-example correctness, Fleiss' kappa across the 3 judges,
and whether the method ranking is preserved.
Writes rebuttal/results/rejudge_bbq.md and rejudge_jailbreak.md
"""
import json
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr

import os
B = "."
J1 = json.load(open(f"{B}/rebuttal/data/rejudge_Qwen3-14B.json"))
# Second family per benchmark: Mistral-24B answers the BBQ question itself instead of
# option-matching the reply (97/103 'correct' on DeepSeek-wrong items) -> degenerate for BBQ;
# it is a well-behaved jailbreak judge. phi-4 is the second family for BBQ.
J2_JB = json.load(open(f"{B}/rebuttal/data/rejudge_Mistral-Small-24B-Instruct-2501.json"))
_phi = f"{B}/rebuttal/data/rejudge_phi-4.json"
J2_BBQ = json.load(open(_phi)) if os.path.exists(_phi) else None
man = json.load(open(f"{B}/rebuttal/data/manifest.json"))


def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pe = sum(((a == c).mean()) * ((b == c).mean()) for c in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def fleiss(votes):
    """votes: list over items of list of labels (equal #raters)."""
    from collections import Counter
    n = len(votes[0]); N = len(votes)
    cats = sorted({v for row in votes for v in row})
    pj = {c: sum(row.count(c) for row in votes) / (N * n) for c in cats}
    Pi = [(sum(v * v for v in Counter(r).values()) - n) / (n * (n - 1)) for r in votes]
    Pbar = sum(Pi) / N
    Pe = sum(p * p for p in pj.values())
    return (Pbar - Pe) / (1 - Pe) if Pe < 1 else 1.0


def summarize(bench_keys, title, J2, j2_name, note=""):
    lines = [f"# {title}", "",
             f"Judges: DeepSeek (original, cached), Qwen3-14B, {j2_name} (local judges from",
             "additional model families; GPT/Gemini API keys unavailable on this machine).",
             "All judges score the IDENTICAL revised replies with the paper's own judging prompt."]
    if note:
        lines += ["", note]
    lines += ["", f"| Task | Method | Paper | DeepSeek (cached) | Qwen3-14B | {j2_name} |",
              "|---|---|---|---|---|---|"]
    kappa_pool = {"DS-QW": [[], []], "DS-J2": [[], []], "QW-J2": [[], []]}
    fleiss_votes = []
    ranks = {}
    for key, meta in bench_keys:
        r1 = J1.get(key)
        r2 = J2.get(key) if J2 else None
        if not r1:
            continue
        ds = [r["deepseek_correct"] for r in r1]
        q = [r["judge_correct"] for r in r1]
        m = [r["judge_correct"] for r in r2] if r2 else None
        task, meth = key.rsplit("/", 1)
        m_cell = f"{np.mean(m):.3f}" if m else "pending"
        lines.append(f"| {task} | {meth} | {meta.get('paper','-')} | {np.mean(ds):.3f} | {np.mean(q):.3f} | {m_cell} |")
        ranks.setdefault(task, []).append((meth, np.mean(ds), np.mean(q), np.mean(m) if m else None))
        kappa_pool["DS-QW"][0] += ds; kappa_pool["DS-QW"][1] += q
        if m:
            kappa_pool["DS-J2"][0] += ds; kappa_pool["DS-J2"][1] += m
            kappa_pool["QW-J2"][0] += q;  kappa_pool["QW-J2"][1] += m
            fleiss_votes += [[a, b, c] for a, b, c in zip(ds, q, m)]
    lines += ["", "## Inter-judge agreement (per-example correct/incorrect, pooled)", ""]
    for name, (a, b) in kappa_pool.items():
        if not a:
            continue
        nm = name.replace("J2", j2_name)
        lines.append(f"- {nm}: raw agreement = {np.mean(np.array(a)==np.array(b)):.3f}, "
                     f"Cohen's kappa = {cohen_kappa(a, b):.3f} (n={len(a)})")
    if fleiss_votes:
        lines.append(f"- All three judges: Fleiss' kappa = {fleiss(fleiss_votes):.3f} (n={len(fleiss_votes)})")
    lines += ["", "## Ranking preservation (method order by accuracy)", ""]
    judges = [(1, "DeepSeek"), (2, "Qwen3-14B")] + ([(3, j2_name)] if fleiss_votes else [])
    for task, rows in ranks.items():
        for j, jn in judges:
            lines.append(f"- {task} [{jn}]: " + " > ".join(v[0] for v in sorted(rows, key=lambda x: -(x[j] or 0))))
        ds_rank = [v[1] for v in rows]; q_rank = [v[2] for v in rows]
        if len(rows) > 2:
            s = f"  Spearman(method-accuracy) DS-QW={spearmanr(ds_rank, q_rank).statistic:.2f}"
            if fleiss_votes:
                m_rank = [v[3] for v in rows]
                s += f", DS-{j2_name}={spearmanr(ds_rank, m_rank).statistic:.2f}"
            lines.append(s)
        lines.append("")
    return lines


bbq_keys = []
for bias in ["gender", "nationality", "disability"]:
    for meth in ["Direct", "CoT", "Heuristics", "Light", "Heavy"]:
        info = man["BBQ_table3_llama3.2-3B"][bias].get(meth, {})
        bbq_keys.append((f"BBQ/{bias}/{meth}", info if isinstance(info, dict) else {}))
BBQ_NOTE = ("Note on judge validation: Mistral-Small-24B was rejected as a BBQ judge — under the "
            "option-matching prompt it answers the question itself rather than matching the reply "
            "(on items DeepSeek scored wrong, Mistral returned the gold option 97/103 times; "
            "Cohen's kappa vs DeepSeek 0.15). This is itself evidence that LLM judges need "
            "validation; phi-4 (Microsoft) is used as the second family instead. Mistral remains "
            "a well-behaved judge for the binary jailbreak task.")
open(f"{B}/rebuttal/results/rejudge_bbq.md", "w").write(
    "\n".join(summarize(bbq_keys, "T5 (E1): Multi-judge re-scoring of BBQ corrections (Llama-3B)",
                        J2_BBQ, "phi-4", BBQ_NOTE)) + "\n")

jb_keys = [(f"JB/{meth}", man["Jailbreak_table4_llama3.2-3B"].get(meth, {}))
           for meth in ["CoT", "Heuristic", "Light", "Heavy", "Light+Heavy"]]
open(f"{B}/rebuttal/results/rejudge_jailbreak.md", "w").write(
    "\n".join(summarize(jb_keys, "T6 (E2): Multi-judge re-scoring of jailbreak corrections (Llama-3B)",
                        J2_JB, "Mistral-24B")) + "\n")
print("wrote rejudge_bbq.md + rejudge_jailbreak.md")
