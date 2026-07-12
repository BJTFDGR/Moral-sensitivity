"""T10 (E8): SFT training driver for ablation/baseline variants.

Mirrors myMain.py/sft_finetuning.py exactly (same hyperparams: full FT for
llama3.2-3B, lr 5e-5, bs 16, 10 epochs, max_len 4096, seed 1, save per epoch)
but takes an explicit training txt so we can train the Step-5-ablated variant
without touching the project's code.

Usage: python t10_train.py --training_file <txt> --output_dir <dir> --gpu 0
"""
import argparse
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument("--training_file", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--grad_accum", type=int, default=1)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["WANDB_MODE"] = "offline"

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

BASE_MODEL = "meta-llama/Llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")

ds = load_dataset("text", data_files={"train": args.training_file})
cfg = SFTConfig(
    do_eval=False,
    num_train_epochs=args.epochs,
    max_length=4096,
    logging_strategy="steps",
    output_dir=args.output_dir,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_accum,
    gradient_checkpointing=True,
    save_strategy="epoch",
    save_only_model=True,
    learning_rate=args.lr,
    log_level="info",
    logging_steps=20,
    seed=args.seed,
    report_to="none",
)
trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=ds["train"], args=cfg)
trainer.train()
print("TRAINING DONE", args.output_dir)
