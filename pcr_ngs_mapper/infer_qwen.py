import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

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
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
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
            recs.append({"prompt": format_prompt(r), "label": r["Label"]})
        return recs

    return DatasetDict({
        "train": Dataset.from_list(to_records(train_df)),
        "validation": Dataset.from_list(to_records(val_df)),
        "test": Dataset.from_list(to_records(test_df)),
    })


def main():
    parser = argparse.ArgumentParser(description="Inference with Qwen + LoRA adapter")
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--csv", default="pcr_ngs_mapper/data/synthetic_pcr_ngs_dataset.csv")
    parser.add_argument("--results_dir", default="results_qwen")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.to(device)
    model.eval()

    ds = build_dataset(args.csv, seed=args.seed)
    test = ds["test"]
    outputs = []
    for i, ex in enumerate(test):
        if i >= args.n:
            break
        prompt = ex["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=3, do_sample=False)
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        token = gen.strip().split()[0] if gen.strip() else ""
        mapped = None
        for lab in LABELS:
            if token.lower().startswith(lab.lower()):
                mapped = lab
                break
        mapped = mapped or token or ""
        rec = {"prompt": prompt, "pred": mapped, "true": ex["label"]}
        outputs.append(rec)
        print(rec)

    with open(os.path.join(args.results_dir, "preds_sample.jsonl"), "w") as f:
        for rec in outputs:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()

