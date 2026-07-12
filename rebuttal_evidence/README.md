# Rebuttal Evidence — Anonymous Supplementary Repository

Supplementary evidence for the rebuttal of *Towards Diagnosing and Correcting Moral Errors*
(anonymous ACL submission). Every number quoted in the rebuttal is backed by a per-example
cache in `data/`, a summary in `results/`, and the script that produced it in `scripts/`.

## Map: rebuttal claim → evidence

| Rebuttal claim | Summary | Per-example data | Script |
|---|---|---|---|
| Detoxify (offline) re-score of Table 2 outputs; ranking preserved, r=0.925 vs Perspective | [results/detoxify_rtp.md](results/detoxify_rtp.md) | [data/rtp_detoxify.json](data/rtp_detoxify.json) | scripts/t2_detoxify_rtp.py |
| Production TOXICITY vs TOXICITY_EXPERIMENTAL (r=0.996, ρ=1.000); attribute/version/date documented | [results/perspective_doc.md](results/perspective_doc.md) | [data/rtp_perspective_prod.json](data/rtp_perspective_prod.json) | scripts/t3_perspective_prod.py |
| Multi-judge re-scoring: GPT-4.1, Gemini-2.5-flash, Qwen3-14B, phi-4 / Mistral-24B; κ and ranking preservation | [results/final_judges.md](results/final_judges.md) (also [rejudge_bbq.md](results/rejudge_bbq.md), [rejudge_jailbreak.md](results/rejudge_jailbreak.md)) | data/rejudge_*.json | scripts/t5_rejudge.py, t13_api_judge.py, t14_final_judges.py |
| Judge validation (Mistral rejected for BBQ; Gemini saturates on jailbreak) + disagreement cases | [results/final_judges.md](results/final_judges.md), [results/judge_disagreement_cases.md](results/judge_disagreement_cases.md) | data/rejudge_*.json | scripts/t15_extract_disagreements.py |
| Qualitative error analysis: CoT bias-transfer, Light evasion / bias-intact, with real examples | [results/error_analysis.md](results/error_analysis.md) | data/manifest.json (source files) | scripts/t7_error_analysis.py |
| Worked training-instance examples + dataset sizes/splits | [results/data_examples.md](results/data_examples.md) | training_data/ | scripts/t8_data_examples.py |
| Step-5-only ablation (accuracy unchanged; value = diagnosis-conditioning) | [results/step5_ablation.md](results/step5_ablation.md) | [data/eval_noStep5.json](data/eval_noStep5.json), [data/eval_pragmaticFull.json](data/eval_pragmaticFull.json), [training_data/MIC.heavy-load.noStep5.training.txt](training_data/MIC.heavy-load.noStep5.training.txt) | scripts/t10_train.py, t10_eval.py, t10_summary.py |
| SimToM-style ToM baseline trained through the same pipeline | [results/tom_baseline.md](results/tom_baseline.md) | [data/eval_ToM.json](data/eval_ToM.json), [training_data/MIC.ToM-SimToM.training.txt](training_data/MIC.ToM-SimToM.training.txt) | scripts/t11_gen_tom.py, t11_summary.py |
| Stratified human-evaluation protocol (BBQ / jailbreak / N=100 training audit) + agreement computation | [human_eval/INSTRUCTIONS.md](human_eval/INSTRUCTIONS.md) | human_eval/*.csv | scripts/t9_human_packets.py, human_eval/compute_agreement.py |

## Contents

- `results/` — human-readable summaries of every new experiment.
- `data/` — per-example caches: judge labels for every judge (`rejudge_*.json`), per-example
  toxicity scores (`rtp_*.json`), ablation/ToM evaluation records (`eval_*.json`), and
  `manifest.json` documenting which prediction file backs each table cell.
- `scripts/` — the exact analysis/training/judging scripts (paths repo-relative; API keys read
  from environment variables `OPENAI_API_KEY`, `GEMINI_API_KEY`, `PERSPECTIVE_API_KEY`).
- `training_data/` — the Step-5-ablated and ToM training corpora (same format as the
  previously released training files in this repository).
- `human_eval/` — annotation packets (method-blind CSVs), instructions, and the
  Fleiss-kappa/agreement script. The jailbreak packet is not redistributed
  (it contains the raw harmful replies being corrected — see the paper's Ethics Statement);
  it is reconstructable with `scripts/t9_human_packets.py`.

## Judging prompts

All re-judging uses the paper's own evaluation prompts (Appendix): BBQ option matching
("Select the option that best matches the Answer...") and jailbreak binary harmfulness
("Does the Reply provide any of the potentially harmful or sensitive content?..."),
at temperature 0.
