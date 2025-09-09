import argparse
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class GenerationConfig:
    n_per_class: int = 25
    seed: int = 42
    noise_sd_ct: float = 0.5
    noise_sd_z: float = 0.5
    ff_mean: float = 8.0
    ff_sd: float = 1.0


CLASSES = ["Normal", "Trisomy21", "Trisomy18", "Trisomy13"]


def generate_class_samples(label: str, n: int, cfg: GenerationConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + hash(label) % 10000)
    # Baselines
    ff = rng.normal(cfg.ff_mean, cfg.ff_sd, size=n)
    ct21 = rng.normal(25.0, 0.3, size=n)
    ct18 = rng.normal(25.0, 0.3, size=n)
    ct13 = rng.normal(25.0, 0.3, size=n)
    ctref = rng.normal(25.0, 0.2, size=n)

    # Trisomy effects: lower Ct by 1.0–2.0 cycles for affected chromosome
    if label == "Trisomy21":
        delta = rng.uniform(1.0, 2.0, size=n)
        ct21 = ct21 - delta
    elif label == "Trisomy18":
        delta = rng.uniform(1.0, 2.0, size=n)
        ct18 = ct18 - delta
    elif label == "Trisomy13":
        delta = rng.uniform(1.0, 2.0, size=n)
        ct13 = ct13 - delta

    # Add Gaussian measurement noise to all Ct values
    ct21 = ct21 + rng.normal(0.0, cfg.noise_sd_ct, size=n)
    ct18 = ct18 + rng.normal(0.0, cfg.noise_sd_ct, size=n)
    ct13 = ct13 + rng.normal(0.0, cfg.noise_sd_ct, size=n)
    ctref = ctref + rng.normal(0.0, cfg.noise_sd_ct, size=n)

    # Z-scores: set affected chromosome to ~3 ± 0.5, others ~0 ± 0.5
    z21 = rng.normal(0.0, cfg.noise_sd_z, size=n)
    z18 = rng.normal(0.0, cfg.noise_sd_z, size=n)
    z13 = rng.normal(0.0, cfg.noise_sd_z, size=n)
    if label == "Trisomy21":
        z21 = rng.normal(3.0, cfg.noise_sd_z, size=n)
    elif label == "Trisomy18":
        z18 = rng.normal(3.0, cfg.noise_sd_z, size=n)
    elif label == "Trisomy13":
        z13 = rng.normal(3.0, cfg.noise_sd_z, size=n)

    # QC Flag: 90% pass, independent from label
    qc = rng.choice([1, 0], size=n, p=[0.9, 0.1])

    df = pd.DataFrame(
        {
            "Fetal_Fraction": np.clip(ff, 2.0, 15.0),
            "Ct_21": ct21,
            "Ct_18": ct18,
            "Ct_13": ct13,
            "Ct_Ref": ctref,
            "QC_Flag": qc,
            "Z_Score_21": z21,
            "Z_Score_18": z18,
            "Z_Score_13": z13,
            "Label": label,
        }
    )
    return df


def generate_dataset(cfg: GenerationConfig) -> pd.DataFrame:
    frames = [generate_class_samples(label, cfg.n_per_class, cfg) for label in CLASSES]
    df = pd.concat(frames, ignore_index=True)
    # Shuffle deterministically
    df = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic PCR-NGS dataset")
    parser.add_argument("--out", default="data/synthetic_pcr_ngs_dataset.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = GenerationConfig(seed=args.seed)
    df = generate_dataset(cfg)

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()

