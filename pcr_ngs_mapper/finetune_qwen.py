import argparse
import json
import os
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel


LABELS = ["Normal", "Trisomy21", "Trisomy18", "Trisomy13"]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def format_prompt(row: Dict) -> str:
    return (
        f"PCR: FF={row['Fetal_Fraction']:.2f}, Ct21={row['Ct_21']:.2f}, "
        f"Ct18={row['Ct_18']:.2f}, Ct13={row['Ct_13']:.2f}, "
        f"CtRef={row['Ct_Ref']:.2f}, QC={int(row['QC_Flag'])} -> NGS Label:"
    )


def build_dataset(csv_path: str, seed: int = 42) -> DatasetDict:
    df = pd.read_csv(csv_path)
    # Deterministic 70/20/10 split by label
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    # Stratify-like manual split per class for exact sizes
    parts = []
    for label in LABELS:
        sub = df[df["Label"] == label]
        n = len(sub)
        n_test = max(1, int(round(0.10 * n)))
        n_val = max(1, int(round(0.20 * n)))
        test = sub.iloc[:n_test]
        val = sub.iloc[n_test:n_test + n_val]
        train = sub.iloc[n_test + n_val:]
        parts.append((train, val, test))
    train_df = pd.concat([p[0] for p in parts]).sample(frac=1.0, random_state=seed)
    val_df = pd.concat([p[1] for p in parts]).sample(frac=1.0, random_state=seed)
    test_df = pd.concat([p[2] for p in parts]).sample(frac=1.0, random_state=seed)

    def to_records(df_: pd.DataFrame) -> List[Dict]:
        recs = []
        for _, r in df_.iterrows():
            prompt = format_prompt(r)
            label = r["Label"]
            recs.append({"prompt": prompt, "label": label})
        return recs

    dsd = DatasetDict(
        {
            "train": Dataset.from_list(to_records(train_df)),
            "validation": Dataset.from_list(to_records(val_df)),
            "test": Dataset.from_list(to_records(test_df)),
        }
    )
    return dsd


def tokenize_for_causal_lm(batch, tokenizer, eos_token_id):
    # Build supervised labels where only the label tokens are trained
    records = []
    for prompt, label in zip(batch["prompt"], batch["label"]):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_text = " " + label
        target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"] + [eos_token_id]
        input_ids = prompt_ids + target_ids
        labels_ids = [-100] * len(prompt_ids) + target_ids.copy()
        attention_mask = [1] * len(input_ids)
        records.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_ids,
        })
    # Manual padding to max length
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    max_len = max(len(r["input_ids"]) for r in records)
    for r in records:
        pad_needed = max_len - len(r["input_ids"])
        if pad_needed > 0:
            r["input_ids"] += [pad_id] * pad_needed
            r["attention_mask"] += [0] * pad_needed
            r["labels"] += [-100] * pad_needed
    # Convert to lists of lists
    batch_padded = {
        "input_ids": [r["input_ids"] for r in records],
        "attention_mask": [r["attention_mask"] for r in records],
        "labels": [r["labels"] for r in records],
    }
    return batch_padded


def prepare_model_and_tokenizer(model_id: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.eos_token_id is None and tokenizer.eos_token:
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
    )
    model.to(device)
    # LoRA config (lightweight)
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    return model, tokenizer


def evaluate_generation(model, tokenizer, ds, device) -> Dict:
    model.eval()
    preds = []
    trues = []
    for ex in ds:
        prompt = ex["prompt"]
        true_label = ex["label"]
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=3,
                do_sample=False,
            )
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        # crude parse: take first word token
        pred = gen.strip().split()[0] if gen.strip() else ""
        # Normalize to one of LABELS (simple mapping by startswith)
        mapped = None
        for lab in LABELS:
            if pred.lower().startswith(lab.lower()):
                mapped = lab
                break
        if mapped is None:
            # fallback: choose label with highest overlap
            mapped = min(LABELS, key=lambda l: len(l))
        preds.append(mapped)
        trues.append(true_label)
    acc = accuracy_score(trues, preds)
    f1m = f1_score(trues, preds, average="macro")
    cm = confusion_matrix(trues, preds, labels=LABELS).tolist()
    report = classification_report(trues, preds, labels=LABELS, output_dict=True)
    return {"accuracy": acc, "f1_macro": f1m, "confusion_matrix": cm, "report": report}


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune a causal LM (e.g., Qwen) for PCR→NGS label prediction on Apple MPS")
    parser.add_argument("--csv", default="pcr_ngs_mapper/data/synthetic_pcr_ngs_dataset.csv")
    parser.add_argument("--model_id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="e.g., Qwen/Qwen2.5-1.8B-Instruct")
    parser.add_argument("--out_dir", default="models/qwen_lora")
    parser.add_argument("--results_dir", default="results_qwen")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--bsz", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = get_device()
    ds = build_dataset(args.csv, seed=args.seed)
    model, tokenizer = prepare_model_and_tokenizer(args.model_id, device)

    def tok_fn(batch):
        return tokenize_for_causal_lm(batch, tokenizer, tokenizer.eos_token_id)

    ds_tok = ds.map(tok_fn, batched=True, remove_columns=["prompt", "label"])  # now has input_ids, attention_mask, labels

    # Trainer
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        gradient_accumulation_steps=2,
        fp16=False,  # MPS: keep False; model is float16 internally
        bf16=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save LoRA adapter and tokenizer
    trainer.model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    # Evaluate by generation on test set
    metrics = evaluate_generation(trainer.model, tokenizer, ds["test"], device)
    with open(os.path.join(args.results_dir, "metrics_qwen.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
