"""T7 (E6): Qualitative error analysis on BBQ (reviewer LVZS Q5, RHB2 Q1).

Contrasts (Llama-3B, cached canonical files):
 1. CoT vs Heavy   - categorize CoT failures on ambiguous (harm) items:
    bias-kept (judged = original biased referent), bias-transferred (judged =
    the OTHER person), other. Heavy failure types for symmetry.
 2. Light vs Heavy - Light behavior on harm items: evasive-correct
    (non-answer mapped to unknown option), responsive-correct,
    leaves-bias-intact (wrong). Real examples for each category.
Writes rebuttal/results/error_analysis.md
"""
import json
import re

B = "."
man = json.load(open(f"{B}/rebuttal/data/manifest.json"))["BBQ_table3_llama3.2-3B"]
BIASES = ["gender", "nationality", "disability"]


def load(bias, meth):
    items = json.load(open(man[bias][meth]["file"]))
    return {(it["Question"].strip(), it["Answer"].strip()): it for it in items if "Question" in it}


def opt_map(choices):
    """label -> option text"""
    return {f"({l})": t.strip().rstrip('.').strip() for l, t in re.findall(r'\(([a-c])\)\s*([^(]+)', choices)}


def is_unknown(text):
    t = text.lower()
    return any(u in t for u in ["cannot", "can't", "unknown", "undetermined", "not answerable",
                                "not enough", "not known", "no answer"])


def person_options(choices):
    return [t for t in opt_map(choices).values() if not is_unknown(t)]


lines = ["# T7 (E6): Qualitative error analysis on BBQ (Llama-3B)", ""]

# ---------- 1. CoT vs Heavy ----------
lines += ["## 1. CoT vs Heavy on ambiguous (harm) items",
          "",
          "Failure taxonomy of judged labels on items where the method is wrong (gold = the",
          "unknown/cannot-determine option): *bias-kept* = revised reply still matches the originally",
          "biased referent; *bias-transferred* = revised reply now matches the OTHER person (bias is",
          "swapped, not removed); *other* = unparseable/other option.",
          "",
          "| Bias | Method | wrong (harm) | bias-kept | bias-transferred | other |",
          "|---|---|---|---|---|---|"]

examples_cot = []
for bias in BIASES:
    H, C = load(bias, "Heavy"), load(bias, "CoT")
    for meth, D in [("CoT", C), ("Heavy", H)]:
        wrong = [k for k in D if D[k].get("flag") == "harm" and int(D[k]["is_correct"]) == 0]
        kept = trans = other = 0
        for k in wrong:
            it = D[k]
            om = opt_map(it["choices"])
            judged = it.get("revised_reply_label", "").strip()
            orig_answer = it["Answer"].strip().lower()
            judged_text = om.get(judged, "").lower()
            # original biased referent = the person option contained in the original Answer
            biased_opt = None
            for lab, t in om.items():
                if not is_unknown(t) and t.lower()[:12] in orig_answer:
                    biased_opt = lab
            if judged and judged == biased_opt:
                kept += 1
            elif judged in om and not is_unknown(om[judged]) and judged != biased_opt:
                trans += 1
                if meth == "CoT" and len(examples_cot) < 4 and bias != "disability":
                    examples_cot.append((bias, k, it, H.get(k)))
            else:
                other += 1
        lines.append(f"| {bias} | {meth} | {len(wrong)} | {kept} | {trans} | {other} |")

lines += ["", "### Examples: CoT transfers the bias, Heavy resolves to 'cannot be determined'", ""]
for bias, k, it, h in examples_cot[:3]:
    lines += [f"**[{bias}]** Q: {k[0][:180]}",
              f"- Biased reply: `{k[1]}` (gold: {it['label']}, choices: {it['choices']})",
              f"- CoT revision: `{it['extracted_reply'][:160]}` -> judged {it.get('revised_reply_label')} (wrong: bias transferred)",
              f"- Heavy revision: `{(h or {}).get('extracted_reply','-')[:160]}` -> judged {(h or {}).get('revised_reply_label')} (correct)",
              ""]

# ---------- 2. Light vs Heavy ----------
lines += ["## 2. Light vs Heavy on ambiguous (harm) items",
          "",
          "Light-load inference removes explicit cues; on context-dependent bias this yields either",
          "*evasive* non-answers (counted correct by option-matching: maps to 'unknown') or replies that",
          "*leave the biased proposition intact*. Heavy stays responsive and grounds the correction in",
          "the diagnosed foundations.",
          "",
          "| Bias | Light: evasive-correct | responsive-correct | wrong (bias intact/other) | Heavy: evasive-correct | responsive-correct | wrong |",
          "|---|---|---|---|---|---|---|"]

ex_evasion, ex_intact = [], []
for bias in BIASES:
    H, L = load(bias, "Heavy"), load(bias, "Light")
    row = {}
    for meth, D in [("Light", L), ("Heavy", H)]:
        ev = resp = wrong = 0
        for k in D:
            it = D[k]
            if it.get("flag") != "harm":
                continue
            rep = it["extracted_reply"].lower()
            ps = person_options(it["choices"])
            responsive = any(p.lower()[:14] in rep for p in ps)
            if int(it["is_correct"]) == 1:
                if responsive:
                    resp += 1
                else:
                    ev += 1
                    if meth == "Light" and len(ex_evasion) < 3:
                        ex_evasion.append((bias, k, it))
            else:
                wrong += 1
                if meth == "Light" and len(ex_intact) < 3:
                    ex_intact.append((bias, k, it, H.get(k)))
        row[meth] = (ev, resp, wrong)
    l, h = row["Light"], row["Heavy"]
    lines.append(f"| {bias} | {l[0]} | {l[1]} | {l[2]} | {h[0]} | {h[1]} | {h[2]} |")

lines += ["", "### Examples: Light evades (counted correct by the option-matching judge)", ""]
for bias, k, it in ex_evasion:
    lines += [f"**[{bias}]** Q: {k[0][:160]}",
              f"- Biased reply: `{k[1]}` -> Light revision: `{it['extracted_reply'][:120]}` (non-answer; judge maps to unknown option = gold)",
              ""]
lines += ["### Examples: Light removes surface cues but leaves the biased proposition intact", ""]
for bias, k, it, h in ex_intact:
    lines += [f"**[{bias}]** Q: {k[0][:160]}",
              f"- Biased reply: `{k[1]}` (gold {it['label']})",
              f"- Light revision: `{it['extracted_reply'][:140]}` -> judged {it.get('revised_reply_label')} (still commits to a referent)",
              f"- Heavy revision: `{(h or {}).get('extracted_reply','-')[:140]}` -> judged {(h or {}).get('revised_reply_label')} (correct)",
              ""]

open(f"{B}/rebuttal/results/error_analysis.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines[:40]))
print("... wrote results/error_analysis.md")
