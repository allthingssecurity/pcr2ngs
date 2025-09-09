**PCR → NGS Mapping Engine (Toy Prototype)**

- Goal: learn a mapping from PCR features (Ct values, fetal fraction, QC) to NGS-like outcomes: classification (Normal, Trisomy21/18/13) or regression (z-scores for chr21/18/13).
- This repo ships a synthetic demo: data generator + two training paths:
  - A compact transformer for tabular features (PyTorch, MPS-ready on Apple Silicon)
  - Optional Qwen LoRA fine-tuning that emits labels from prompt-formatted PCR features

This is a toy prototype for exploration, not a clinical tool.

**Repo Layout**
- `pcr_ngs_mapper/generate_data.py` — creates a synthetic dataset (100 rows, 25/class)
- `pcr_ngs_mapper/train.py` — tabular mini-transformer trainer (classification or regression)
- `pcr_ngs_mapper/finetune_qwen.py` — LoRA fine-tunes Qwen to output labels given PCR prompts
- `pcr_ngs_mapper/infer_qwen.py` — runs inference with a trained LoRA adapter
- `pcr_ngs_mapper/config.yaml` — training config (mode, epochs, lr, paths)
- `pcr_ngs_mapper/requirements.txt` — Python dependencies

**Quickstart (Apple Silicon)**
1) Create venv and install deps
- `python3 -m venv pcr_ngs_mapper/.venv`
- `source pcr_ngs_mapper/.venv/bin/activate`
- `pip install -r pcr_ngs_mapper/requirements.txt`

2) Generate synthetic data
- `python pcr_ngs_mapper/generate_data.py --out pcr_ngs_mapper/data/synthetic_pcr_ngs_dataset.csv`

3) Train mini-transformer (default classification)
- Adjust `pcr_ngs_mapper/config.yaml` if needed
- `python pcr_ngs_mapper/train.py --config pcr_ngs_mapper/config.yaml`
- Outputs: `models/model_classification.pt`, `results/metrics.json`, `results/confusion_matrix.png`

4) Optional: LoRA fine-tune Qwen
- Qwen (small): `python pcr_ngs_mapper/finetune_qwen.py --csv pcr_ngs_mapper/data/synthetic_pcr_ngs_dataset.csv \
  --model_id Qwen/Qwen2.5-0.5B-Instruct --out_dir models/qwen2p5_0p5b_lora --results_dir results_qwen --epochs 3 --bsz 1`
- For larger models (more RAM): `--model_id Qwen/Qwen2.5-1.8B-Instruct`

5) Qwen inference (post-finetuning)
- `python pcr_ngs_mapper/infer_qwen.py --adapter_dir models/qwen2p5_0p5b_lora \
  --model_id Qwen/Qwen2.5-0.5B-Instruct --csv pcr_ngs_mapper/data/synthetic_pcr_ngs_dataset.csv --n 5`

**Assumptions and Simulation**
- Trisomy reduces the affected chromosome Ct by ~1–2 cycles; z-score ≈ 3 ± 0.5 for affected, ≈ 0 ± 0.5 for others.
- Gaussian noise on Ct (σ=0.5). QC is independent with 90% pass.
- Balanced classes (25 each): Normal, T21, T18, T13.

**Limitations**
- Synthetic, tiny dataset (100 samples): expect modest accuracy and limited generalization.
- No lab batch effects or real assay drift modeled.
- Qwen LoRA uses minimal epochs and small batch sizes by default.

**How to Improve**
- More data: scale synthetic set (e.g., 10k+) with richer noise models.
- More training: increase epochs (e.g., 20–50) and tune LR/weight decay.
- Stronger prompts for Qwen: explicitly list allowable labels and use a short chain-of-thought rationale (if permitted) or few-shot examples.
- Model capacity: try slightly larger tabular transformers (careful with overfitting).
- Better targets: combine classification and regression with multi-task heads.
- Robustness: inject lab-like variability (batch, instrument, reagent lots) during simulation.

**License & Use**
- For research and prototyping only; not for clinical use.

