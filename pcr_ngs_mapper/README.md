**PCR → NGS Mapping Engine (Synthetic Prototype)**

- **Goal:** Train a small transformer to predict NGS-like outcomes (trisomy label or z-scores) from PCR features.
- **Apple hardware:** Uses `torch` with MPS acceleration on Apple Silicon when available.

**Repo Layout**
- `generate_data.py`: Creates `data/synthetic_pcr_ngs_dataset.csv` with 100 samples (25/class).
- `train.py`: Loads data → splits → scales → trains transformer → evaluates.
- `finetune_qwen.py`: LoRA fine-tunes a causal LM (e.g., Qwen) to emit labels from prompt-formatted PCR features.
- `config.yaml`: Training/configuration defaults (mode, epochs, lr, seed, paths).
- `models/`: Saved model checkpoints.
- `results/`: Metrics (`metrics.json`, `history.json`) and `confusion_matrix.png`.

**Installation (Apple Silicon)**
- Python 3.10+ recommended.
- Create/activate a virtualenv, then:
  - `pip install --upgrade pip`
- `pip install torch --extra-index-url https://download.pytorch.org/whl/cpu` (pip wheels include MPS; CPU index is fine; MPS is used automatically if available)
- `pip install numpy pandas scikit-learn matplotlib pyyaml transformers peft datasets accelerate`

If you already have PyTorch installed with MPS support, you can skip the first torch line. The code auto-selects `mps`, then `cuda`, then `cpu`.

**Assumptions**
- Trisomy reduces the Ct value (earlier amplification) by 1–2 cycles for the affected chromosome.
- Corresponding NGS z-score for the affected chromosome is ~3 ± 0.5; others ~0 ± 0.5.
- Gaussian noise on Ct (σ=0.5) simulates measurement variability.
- QC flag is independent (90% pass) and not correlated with trisomy.

**Generate Data**
- From the project root:
- `python pcr_ngs_mapper/generate_data.py --out pcr_ngs_mapper/data/synthetic_pcr_ngs_dataset.csv`

**Train (default: classification)**
- Adjust `pcr_ngs_mapper/config.yaml` as needed.
- `python pcr_ngs_mapper/train.py --config pcr_ngs_mapper/config.yaml`
- Outputs:
  - `models/model_classification.pt`
  - `results/metrics.json`, `results/history.json`, and `results/confusion_matrix.png`

**Regression mode (z-scores)**
- Edit `mode: regression` in `config.yaml`, then rerun training.

**LoRA Fine-tune Qwen (optional, Apple MPS)**
- This uses the CSV to create text prompts ("PCR: ... -> NGS Label:") and trains a LoRA adapter to generate the label.
- Example (TinyLlama default for low RAM; swap to Qwen when resources allow):
  - `python pcr_ngs_mapper/finetune_qwen.py --csv pcr_ngs_mapper/data/synthetic_pcr_ngs_dataset.csv \
     --model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --out_dir models/qwen_lora --results_dir results_qwen`
- For Qwen on Apple Silicon, ensure enough memory and use an instruct model id, e.g.:
  - `--model_id Qwen/Qwen2.5-1.8B-Instruct` (≈1.8B) or `Qwen/Qwen2.5-3B-Instruct` (≈3B). 2–3B may require 24–48GB unified memory.
- Outputs: adapter weights in `models/qwen_lora/` and `results_qwen/metrics_qwen.json` with accuracy/F1 and confusion matrix.

**Example Input–Output**
- Input: `FF=7.8, Ct21=23.9, Ct18=25.0, Ct13=25.1, CtRef=24.8, QC=1`
- Output (classification): `Label = Trisomy21`
- Output (regression): `Z_Score_21 ≈ 3.1, others ≈ 0`

**Limitations**
- Synthetic, small dataset; real-world batch effects and lab variability are not modeled.
- Simple transformer; not hyperparameter-optimized.
- QC behavior not tied to outcomes.

**Extension Plan (high level)**
- Add real paired PCR–NGS datasets (with necessary approvals).
- Retrain/fine-tune on real data; add uncertainty estimates (MC dropout).
- Package as a FastAPI microservice; add monitoring for dataset drift.
