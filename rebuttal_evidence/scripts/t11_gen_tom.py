"""T11 (E9): Generate a SimToM-style Theory-of-Mind inference baseline dataset
(reviewer LVZS Q3) through the SAME pipeline as the paper's methods:
same 2000 MIC items (1000 agree / 1000 disagree), same gold labels
(judgment + human revised reply), teacher verbalizes intermediate inference.

NOTE: the paper's teacher (DeepSeek) API key is no longer valid, so the
teacher here is a local Qwen3-14B. This substitution must be stated wherever
these numbers are reported.

SimToM-style steps (perspective inference BEFORE judgment/correction):
 (1) speaker's perspective (beliefs/intentions behind the reply)
 (2) perspective of the listener / affected parties (how they perceive it)
 (3) moral judgment explained from those perspectives
 (4) (disagree only) how to revise the reply so all perspectives are respected

Output: rebuttal/data/MIC.Correct.ToM.llama3.2-3B.2000.train.1.txt
Usage: python t11_gen_tom.py --gpu 1
"""
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="1")
parser.add_argument("--batch", type=int, default=24)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

B = "."
TEACHER = "Qwen/Qwen3-14B"
MIC_agree2label = {"agree": "agree", "disagree": "disagree", "neutral": "agree"}

data = json.load(open(f"{B}/dataset/mic/MIC.Correct.pragmatic.json"))
print(f"{len(data)} MIC items")

tok = AutoTokenizer.from_pretrained(TEACHER)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(TEACHER, torch_dtype=torch.bfloat16, device_map="cuda:0")
model.eval()


def teacher_prompt(it):
    q = it["Question"].strip().replace("\n", " ")
    a = it["Answer"].strip().replace("\n", " ")
    judg = MIC_agree2label[it["judg2reply"].strip()]
    p = (f'<Prompt>: {q}\n<Reply>: {a}\n'
         f'The moral judgment of the <Reply> is: {judg}.\n\n'
         "(1) Take the perspective of the speaker of the <Reply>: what does the speaker believe, "
         "intend, or assume? Your answer should start with \"(1) The speaker\".\n"
         "(2) Take the perspective of the listener and of any people affected by the <Reply>: how "
         "would they perceive or be affected by it?\n"
         f"(3) Based on these perspectives, explain why the moral judgment of the <Reply> is {judg}.\n")
    if judg == "disagree":
        wa = it["worker_answer"].strip().replace("\n", " ")
        p += (f'(4) There is a <Revised Reply>: "{wa}". Explain how the <Reply> should be revised, '
              "considering the perspectives above, to become the <Revised Reply>.\n")
    p += "\nPlease ensure your answer would not be more than 20 sentences."
    return p


out_path = f"{B}/rebuttal/data/tom_inference.json"
results = json.load(open(out_path)) if os.path.exists(out_path) else {}

todo = [(i, it) for i, it in enumerate(data) if str(i) not in results]
print(f"{len(todo)} to generate")
for s in range(0, len(todo), args.batch):
    batch = todo[s:s + args.batch]
    texts = [tok.apply_chat_template([{"role": "user", "content": teacher_prompt(it)}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
             for _, it in batch]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=512, do_sample=False, pad_token_id=tok.pad_token_id)
    for j, (i, it) in enumerate(batch):
        resp = tok.decode(gen[j, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        results[str(i)] = resp
    if (s // args.batch) % 5 == 0:
        json.dump(results, open(out_path, "w"))
        print(f"{s + len(batch)}/{len(todo)}", flush=True)
json.dump(results, open(out_path, "w"))

# assemble the final training txt with the same contract as other settings
lines = []
for i, it in enumerate(data):
    q = it["Question"].strip().replace("\n", " ")
    a = it["Answer"].strip().replace("\n", " ")
    judg = MIC_agree2label[it["judg2reply"].strip()]
    inf = results[str(i)].strip().replace("\n", " ")
    text = f'<Prompt>: "{q}"; <Reply>: "{a}". ###Diagnosis: {inf} '
    if judg == "agree":
        if "(3)" in text:
            text = text  # keep perspective steps + judgment explanation for agree
    else:
        wa = it["worker_answer"].strip().replace("\n", " ")
        text += f'Therefore, the <Revised Reply> is "{wa}"'
    lines.append(text)
train_path = f"{B}/rebuttal/data/MIC.Correct.ToM.llama3.2-3B.2000.train.1.txt"
open(train_path, "w").write("\n".join(lines))
print("wrote", train_path)
