"""T9 (E10/E11/E12): Build human-evaluation packets.

A) BBQ stratified sample     (100 items) - label revised replies: biased? evasive?
B) Jailbreak stratified      (100 items) - label revised replies: harmful?
C) Training-data audit       (100 items) - heavy+light load instances: steps sound? revision ok?
Each packet: annotator CSV (method-blind, order-randomized), hidden key CSV, instructions.
Plus compute_agreement.py: Fleiss kappa + raw agreement + human-judge agreement,
usable both for these packets and for the existing A.4 300-item raw labels.
"""
import csv
import json
import random
import glob

B = "."
HE = f"{B}/rebuttal/human_eval"
rng = random.Random(42)
man = json.load(open(f"{B}/rebuttal/data/manifest.json"))

# ---------- A) BBQ ----------
rows, key_rows = [], []
per_bias = {"gender": 40, "nationality": 34, "disability": 26}
methods = ["Heavy", "Light", "CoT", "Heuristics"]
for bias, n_bias in per_bias.items():
    pool = []
    for meth in methods:
        info = man["BBQ_table3_llama3.2-3B"][bias].get(meth)
        items = json.load(open(info["file"]))
        for i, it in enumerate(items):
            if "Question" not in it or it.get("flag") != "harm":
                continue
            pool.append((meth, i, it))
    rng.shuffle(pool)
    # stratify: equal per method
    per_m = n_bias // len(methods)
    chosen = []
    for meth in methods:
        chosen += [p for p in pool if p[0] == meth][:per_m]
    for meth, i, it in chosen:
        iid = f"bbq_{len(rows):03d}"
        rows.append({"item_id": iid, "question": it["Question"], "choices": it["choices"],
                     "original_reply": it["Answer"], "revised_reply": it["extracted_reply"],
                     "biased_yes_no": "", "evasive_yes_no": ""})
        key_rows.append({"item_id": iid, "bias": bias, "method": meth, "source_index": i,
                         "gold_label": it["label"], "deepseek_correct": it.get("is_correct")})
rng.shuffle(rows)
with open(f"{HE}/bbq_annotation.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
with open(f"{HE}/bbq_key.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(key_rows[0].keys())); w.writeheader(); w.writerows(key_rows)
print(f"BBQ packet: {len(rows)} items")

# ---------- B) Jailbreak ----------
rows, key_rows = [], []
jmethods = ["Heavy", "Light", "CoT", "Heuristic", "Light+Heavy"]
for meth in jmethods:
    info = man["Jailbreak_table4_llama3.2-3B"].get(meth)
    if not isinstance(info, dict) or not info.get("file"):
        continue
    items = json.load(open(info["file"]))
    harm = [(i, it) for i, it in enumerate(items) if it.get("harmful_flag") == "Yes"]
    rng.shuffle(harm)
    for i, it in harm[:20]:
        iid = f"jb_{len(rows):03d}"
        rows.append({"item_id": iid, "prompt": it["Question"][:500], "original_reply": it["Answer"][:500],
                     "revised_reply": it["extracted_reply"][:800], "harmful_yes_no": ""})
        key_rows.append({"item_id": iid, "method": meth, "source_index": i,
                         "deepseek_flag": it.get("llm_harmful_flag"), "deepseek_correct": it.get("llm_correct")})
rng.shuffle(rows)
with open(f"{HE}/jailbreak_annotation.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
with open(f"{HE}/jailbreak_key.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(key_rows[0].keys())); w.writeheader(); w.writerows(key_rows)
print(f"Jailbreak packet: {len(rows)} items")

# ---------- C) Training-data audit ----------
rows, key_rows = [], []
train_lines = open(f"{B}/experiments/MIC.Correct.pragmatic.llama3.2-3B.2000.train.1.txt").read().split("\n")
heavy_dis = [l for l in train_lines if "Therefore, the <Revised Reply> is" in l]
heavy_agr = [l for l in train_lines if "Therefore, the <Revised Reply> is" not in l]
sema = []
for fp in sorted(glob.glob(f"{B}/experiments/deepseek/realtoxicity/Correct/semantic/*.json")):
    d = json.load(open(fp)); sema.append(d if isinstance(d, dict) else d[0])
sema_dis = [x for x in sema if x["judg2reply"].strip() == "disagree"]
sema_agr = [x for x in sema if x["judg2reply"].strip() == "agree"]
rng.shuffle(heavy_dis); rng.shuffle(heavy_agr); rng.shuffle(sema_dis); rng.shuffle(sema_agr)

def add_audit(text, load, judg):
    iid = f"audit_{len(rows):03d}"
    rows.append({"item_id": iid, "training_instance": text[:4000],
                 "steps_factually_consistent_yes_no": "", "revision_resolves_issue_yes_no_na": "",
                 "fluent_appropriate_yes_no": ""})
    key_rows.append({"item_id": iid, "load": load, "judgment": judg})

for l in heavy_dis[:25]:
    add_audit(l, "heavy", "disagree")
for l in heavy_agr[:25]:
    add_audit(l, "heavy", "agree")
for x in sema_dis[:25]:
    add_audit(f"<Prompt>: \"{x['Question'].strip()}\"; <Reply>: \"{x['Answer'].strip()}\". ###Diagnosis: "
              + x["inference4semantic"][0]["response"].strip().replace("\n", " ")
              + f' Therefore, the <Revised Reply> is "{x.get("worker_answer","").strip()}"', "light", "disagree")
for x in sema_agr[:25]:
    add_audit(f"<Prompt>: \"{x['Question'].strip()}\"; <Reply>: \"{x['Answer'].strip()}\". ###Diagnosis: "
              + x["inference4semantic"][0]["response"].strip().replace("\n", " "), "light", "agree")
rng.shuffle(rows)
with open(f"{HE}/training_audit_annotation.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
with open(f"{HE}/training_audit_key.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(key_rows[0].keys())); w.writeheader(); w.writerows(key_rows)
print(f"Training-audit packet: {len(rows)} items")
