"""Compute inter-annotator agreement (Fleiss' kappa + raw agreement) and
human-vs-automated-judge agreement for the rebuttal human-eval packets.

Usage:
  python compute_agreement.py --packet bbq  --files bbq_annotation.A1.csv bbq_annotation.A2.csv ...
  python compute_agreement.py --packet jb   --files jailbreak_annotation.A*.csv
  python compute_agreement.py --packet audit --files training_audit_annotation.A*.csv
  python compute_agreement.py --packet a4   --files a4_labels.A*.csv   # columns: item_id,preferred
"""
import argparse
import csv
import os
from collections import Counter, defaultdict

QUESTIONS = {
    "bbq": ["biased_yes_no", "evasive_yes_no"],
    "jb": ["harmful_yes_no"],
    "audit": ["steps_factually_consistent_yes_no", "revision_resolves_issue_yes_no_na",
              "fluent_appropriate_yes_no"],
    "a4": ["preferred"],
}


def fleiss_kappa(mat):
    """mat: list of Counters {category: n_votes} per item, equal raters per item."""
    n = sum(mat[0].values())
    cats = sorted({c for m in mat for c in m})
    N = len(mat)
    p_j = {c: sum(m.get(c, 0) for m in mat) / (N * n) for c in cats}
    P_i = [(sum(v * v for v in m.values()) - n) / (n * (n - 1)) for m in mat]
    P_bar = sum(P_i) / N
    P_e = sum(p * p for p in p_j.values())
    return (P_bar - P_e) / (1 - P_e) if P_e < 1 else 1.0


def pairwise_raw(labels_by_annotator):
    anns = list(labels_by_annotator)
    tot = agree = 0
    for i in range(len(anns)):
        for j in range(i + 1, len(anns)):
            for iid in labels_by_annotator[anns[i]]:
                a, b = labels_by_annotator[anns[i]][iid], labels_by_annotator[anns[j]].get(iid)
                if b is None:
                    continue
                tot += 1
                agree += int(a == b)
    return agree / tot if tot else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet", required=True, choices=list(QUESTIONS))
    ap.add_argument("--files", nargs="+", required=True)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    for q in QUESTIONS[args.packet]:
        labels = defaultdict(dict)  # item_id -> annotator -> label
        for fp in args.files:
            for row in csv.DictReader(open(fp)):
                v = row.get(q, "").strip().lower()
                if v:
                    labels[row["item_id"]][fp] = v
        complete = {iid: d for iid, d in labels.items() if len(d) == len(args.files)}
        if not complete:
            print(f"[{q}] no complete items yet")
            continue
        mat = [Counter(d.values()) for d in complete.values()]
        kappa = fleiss_kappa(mat)
        by_ann = defaultdict(dict)
        for iid, d in complete.items():
            for a, v in d.items():
                by_ann[a][iid] = v
        raw = pairwise_raw(by_ann)
        print(f"[{q}] items={len(complete)} raters={len(args.files)} "
              f"raw pairwise agreement={raw:.3f} Fleiss kappa={kappa:.3f}")

        # human majority vs automated judge (packets with a key + judge labels)
        keyfile = {"bbq": "bbq_key.csv", "jb": "jailbreak_key.csv"}.get(args.packet)
        if keyfile and os.path.exists(os.path.join(here, keyfile)):
            key = {r["item_id"]: r for r in csv.DictReader(open(os.path.join(here, keyfile)))}
            tot = agree = 0
            for iid, d in complete.items():
                maj = Counter(d.values()).most_common(1)[0][0]
                k = key.get(iid, {})
                if args.packet == "bbq" and q == "biased_yes_no" and k.get("deepseek_correct") not in (None, ""):
                    judge_ok = str(k["deepseek_correct"]) in ("1", "1.0", "True")
                    human_ok = maj == "no"          # not biased = corrected
                    tot += 1; agree += int(judge_ok == human_ok)
                if args.packet == "jb" and q == "harmful_yes_no" and k.get("deepseek_flag"):
                    tot += 1; agree += int(k["deepseek_flag"].strip().lower() == maj)
            if tot:
                print(f"    human-majority vs DeepSeek-judge agreement: {agree/tot:.3f} (n={tot})")


if __name__ == "__main__":
    main()
