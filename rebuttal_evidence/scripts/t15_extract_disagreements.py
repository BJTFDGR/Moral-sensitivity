import json, collections
B = "."
D = f"{B}/rebuttal/data"
manifest = json.load(open(f"{D}/manifest.json"))
judges = {
    "Qwen3-14B": json.load(open(f"{D}/rejudge_Qwen3-14B.json")),
    "phi-4": json.load(open(f"{D}/rejudge_phi-4.json")),
    "Mistral-24B": json.load(open(f"{D}/rejudge_Mistral-Small-24B-Instruct-2501.json")),
    "GPT-4.1": json.load(open(f"{D}/rejudge_gpt-4.1.json")),
}

out = []

# ---- Jailbreak ----
for meth, info in manifest["Jailbreak_table4_llama3.2-3B"].items():
    if not isinstance(info, dict) or not info.get("file"): continue
    items = json.load(open(info["file"]))
    if len(items) < 50: continue
    key = f"JB/{meth}"
    for i, it in enumerate(items):
        votes = {}
        ds = it.get("llm_harmful_flag")
        votes["DeepSeek"] = ds
        ok = True
        for jn, jd in judges.items():
            if key not in jd or len(jd[key]) != len(items): ok = False; break
            votes[jn] = jd[key][i].get("judge_flag")
        if not ok: continue
        vals = set(v for v in votes.values() if v)
        if len(vals) > 1:
            out.append({"task": key, "idx": i, "harmful_flag_gt_promptside": it.get("harmful_flag"),
                        "votes": votes,
                        "question": it.get("Question","")[:600],
                        "reply": it.get("extracted_reply","")[:1200]})

jb_dis = [o for o in out if o["task"].startswith("JB")]
print("JB disagreements:", len(jb_dis), "of", 5*420)
# split pattern counts
pat = collections.Counter(tuple(sorted((k,v) for k,v in o["votes"].items())) for o in jb_dis)

# ---- BBQ ----
bbq_out = []
for bias in ["gender","nationality","disability"]:
    for meth, info in manifest["BBQ_table3_llama3.2-3B"][bias].items():
        if not isinstance(info, dict) or not info.get("file"): continue
        items = json.load(open(info["file"]))
        if len(items) < 50 or "extracted_reply" not in items[0]: continue
        key = f"BBQ/{bias}/{meth}"
        for i, it in enumerate(items):
            votes = {"DeepSeek": (it.get("revised_reply_label") or "").strip()}
            ok = True
            for jn in ["Qwen3-14B","phi-4","GPT-4.1"]:
                jd = judges[jn]
                if key not in jd or len(jd[key]) != len(items): ok = False; break
                votes[jn] = jd[key][i].get("judge_label")
            if not ok: continue
            vals = set(v for v in votes.values() if v)
            if len(vals) > 1:
                bbq_out.append({"task": key, "idx": i, "gold": it.get("label","").strip(),
                                "votes": votes,
                                "question": it.get("Question","")[:500],
                                "choices": it.get("choices","")[:400],
                                "reply": it.get("extracted_reply","")[:900]})
print("BBQ disagreements:", len(bbq_out))
json.dump({"jb": jb_dis, "bbq": bbq_out}, open("f"{B}/rebuttal/data/judge_disagreements_all.json"","w"), indent=1)

# summarize per task
for t in sorted(set(o["task"] for o in jb_dis)):
    print(" ", t, sum(1 for o in jb_dis if o["task"]==t))
for t in sorted(set(o["task"] for o in bbq_out)):
    print(" ", t, sum(1 for o in bbq_out if o["task"]==t))
