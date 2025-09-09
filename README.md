**PCR → NGS Mapping Engine (Synthetic Prototype)**

This repository demonstrates a toy, end‑to‑end pipeline that learns to map PCR features (Ct values, fetal fraction, QC) to NGS‑like outcomes. It includes:
- A compact tabular transformer (PyTorch) for classification or regression.
- An optional LoRA fine‑tuning path for Qwen to generate discrete trisomy labels from prompt‑formatted PCR inputs.

This is a synthetic proof‑of‑concept for experimentation only — not for clinical use.

**Repository Structure**
- `pcr_ngs_mapper/generate_data.py`: Synthetic dataset (100 rows; 25/class).
- `pcr_ngs_mapper/train.py`: Tabular mini‑transformer (classification or regression).
- `pcr_ngs_mapper/finetune_qwen.py`: LoRA fine‑tune Qwen on text prompts.
- `pcr_ngs_mapper/infer_qwen.py`: Post‑finetuning inference for Qwen.
- `pcr_ngs_mapper/config.yaml`: Training configuration.
- `pcr_ngs_mapper/requirements.txt`: Dependencies.

**Getting Started (Apple Silicon)**
1) Environment
- `python3 -m venv pcr_ngs_mapper/.venv`
- `source pcr_ngs_mapper/.venv/bin/activate`
- `pip install -r pcr_ngs_mapper/requirements.txt`

2) Data Generation
- `python pcr_ngs_mapper/generate_data.py --out pcr_ngs_mapper/data/synthetic_pcr_ngs_dataset.csv`

3) Tabular Model: Train & Evaluate
- Configure: edit `pcr_ngs_mapper/config.yaml` (default `mode: classification`).
- Train: `python pcr_ngs_mapper/train.py --config pcr_ngs_mapper/config.yaml`
- Artifacts:
  - `models/model_classification.pt`
  - `results/metrics.json`, `results/history.json`, `results/confusion_matrix.png`

Example `results/metrics.json` (values vary):
```
{
  "mode": "classification",
  "test_accuracy": 0.40,
  "test_f1_macro": 0.41,
  "classification_report": { "Normal": {"precision": ...}, ... }
}
```

Example predictions (printed by `train.py`, first 5):
```
{'true': 0, 'pred': 1, 'input': [...]}
...
```
Use `results/label_mapping.json` to map indices to class names.

4) Qwen LoRA: Fine‑tune & Inference
- Fine‑tune (small model shown; increase model_id/epochs for better results):
```
python pcr_ngs_mapper/finetune_qwen.py \
  --csv pcr_ngs_mapper/data/synthetic_pcr_ngs_dataset.csv \
  --model_id Qwen/Qwen2.5-0.5B-Instruct \
  --out_dir models/qwen2p5_0p5b_lora --results_dir results_qwen \
  --epochs 3 --bsz 1
```
- Inference (post‑finetune):
```
python pcr_ngs_mapper/infer_qwen.py \
  --adapter_dir models/qwen2p5_0p5b_lora \
  --model_id Qwen/Qwen2.5-0.5B-Instruct \
  --csv pcr_ngs_mapper/data/synthetic_pcr_ngs_dataset.csv \
  --n 5
```
- Example output (one line per sample):
```
{"prompt": "PCR: FF=7.15, Ct21=24.92, Ct18=24.41, Ct13=23.24, CtRef=24.63, QC=1 -> NGS Label:",
 "pred": "Trisomy", "true": "Trisomy13"}
```
Tip: For crisper outputs, constrain the model with a choice list in the prompt, e.g.,
`... -> NGS Label (choose one: Normal, Trisomy21, Trisomy18, Trisomy13):`

**Assumptions (Synthetic Simulator)**
- Trisomy reduces the affected chromosome Ct by ~1–2 cycles; z‑score ≈ 3 ± 0.5 (others ≈ 0 ± 0.5).
- Gaussian noise on Ct (σ = 0.5). QC pass rate = 90% (independent of label).
- Balanced classes: Normal, Trisomy21, Trisomy18, Trisomy13 (25 each).

**Limitations**
- Small synthetic data (100 samples) → modest accuracy, limited generalization.
- No batch effects or lab variability simulated.
- LoRA defaults are minimal; accuracy benefits from more epochs and larger models.

**How To Improve**
- Data scale: increase to thousands of samples; add richer noise and batch effects.
- Training: raise epochs (e.g., 20–50), tune LR/weight decay, add early stopping.
- Prompting (Qwen): provide explicit choice list, few‑shot demonstrations, and slightly longer generation length.
- Modeling: multi‑task heads (classification + z‑scores), uncertainty via MC dropout.

**Disclaimer**
Research prototype only. Not validated for clinical use.
