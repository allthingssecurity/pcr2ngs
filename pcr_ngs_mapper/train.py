import argparse
import json
import os
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MiniTabTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        seq_len: int = 6,
        out_dims: Dict[str, int] = {"classification": 4, "regression": 3},
        mode: str = "classification",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.mode = mode

        # Learnable feature (column) embeddings, one per input feature token
        self.col_embeddings = nn.Embedding(seq_len, d_model)
        # Project scalar value -> d_model, shared across tokens
        self.value_proj = nn.Linear(1, d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        if mode == "classification":
            self.head = nn.Linear(d_model, out_dims["classification"])
        else:
            self.head = nn.Linear(d_model, out_dims["regression"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len) continuous features already scaled
        B, S = x.shape
        assert S == self.seq_len
        # Convert to tokens per feature: (B, S, 1) -> value_proj -> (B, S, d_model)
        v = self.value_proj(x.unsqueeze(-1))
        # Add column embeddings
        col_ids = torch.arange(self.seq_len, device=x.device).unsqueeze(0).expand(B, -1)
        e = self.col_embeddings(col_ids)
        h = self.dropout(v + e)
        # Transformer encoder
        z = self.encoder(h)  # (B, S, d_model)
        # Mean pooling
        z = z.mean(dim=1)
        z = self.norm(z)
        out = self.head(z)
        return out


@dataclass
class TrainConfig:
    mode: str
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    data_path: str
    model_dir: str
    results_dir: str
    num_workers: int = 0


INPUT_COLS = [
    "Fetal_Fraction",
    "Ct_21",
    "Ct_18",
    "Ct_13",
    "Ct_Ref",
    "QC_Flag",
]
REG_TARGETS = ["Z_Score_21", "Z_Score_18", "Z_Score_13"]
CLASS_COL = "Label"


def load_config(path: str) -> TrainConfig:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return TrainConfig(**cfg)


def split_and_scale(df: pd.DataFrame, seed: int) -> Tuple:
    # Split 10% test, then 20/70 val/train from remaining
    df_trainval, df_test = train_test_split(df, test_size=0.1, random_state=seed, stratify=df[CLASS_COL])
    df_train, df_val = train_test_split(
        df_trainval, test_size=2 / 9, random_state=seed, stratify=df_trainval[CLASS_COL]
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[INPUT_COLS].values)
    X_val = scaler.transform(df_val[INPUT_COLS].values)
    X_test = scaler.transform(df_test[INPUT_COLS].values)

    return (df_train, df_val, df_test, X_train, X_val, X_test, scaler)


def prepare_targets(cfg: TrainConfig, df_train, df_val, df_test):
    if cfg.mode == "classification":
        le = LabelEncoder()
        y_train = le.fit_transform(df_train[CLASS_COL])
        y_val = le.transform(df_val[CLASS_COL])
        y_test = le.transform(df_test[CLASS_COL])
        return (y_train, y_val, y_test, le)
    else:
        y_train = df_train[REG_TARGETS].values.astype(np.float32)
        y_val = df_val[REG_TARGETS].values.astype(np.float32)
        y_test = df_test[REG_TARGETS].values.astype(np.float32)
        return (y_train, y_val, y_test, None)


def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, cfg: TrainConfig):
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    if cfg.mode == "classification":
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        y_val_t = torch.tensor(y_val, dtype=torch.long)
        y_test_t = torch.tensor(y_test, dtype=torch.long)
    else:
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    val_ds = torch.utils.data.TensorDataset(X_val_t, y_val_t)
    test_ds = torch.utils.data.TensorDataset(X_test_t, y_test_t)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cv(cfg.batch_size, 1), shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    return train_loader, val_loader, test_loader


def cv(v, minimum):
    return max(v, minimum)


def train_one_epoch(model, loader, criterion, optimizer, device, mode: str):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, mode: str):
    model.eval()
    total_loss = 0.0
    all_y = []
    all_pred = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        if mode == "classification":
            pred = torch.argmax(logits, dim=1)
        else:
            pred = logits
        all_y.append(yb.detach().cpu())
        all_pred.append(pred.detach().cpu())

    y = torch.cat(all_y)
    pred = torch.cat(all_pred)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, y, pred


def plot_confusion(cm, classes, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes, ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def save_metrics(metrics: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train a mini transformer on synthetic PCR->NGS")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    set_seed(cfg.seed)

    # Load data
    df = pd.read_csv(cfg.data_path)
    (df_train, df_val, df_test, X_train, X_val, X_test, scaler) = split_and_scale(df, cfg.seed)
    (y_train, y_val, y_test, label_encoder) = prepare_targets(cfg, df_train, df_val, df_test)

    # Save label mapping if classification
    os.makedirs(cfg.results_dir, exist_ok=True)
    if cfg.mode == "classification" and label_encoder is not None:
        mapping = {int(i): c for i, c in enumerate(label_encoder.classes_)}
        with open(os.path.join(cfg.results_dir, "label_mapping.json"), "w") as f:
            json.dump(mapping, f, indent=2)

    # Data loaders
    train_loader, val_loader, test_loader = make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, cfg)

    # Model
    model = MiniTabTransformer(
        d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1, seq_len=len(INPUT_COLS), mode=cfg.mode
    ).to(device)

    # Loss/optimizer
    if cfg.mode == "classification":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # Training loop with early stopping
    best_val = float('inf')
    best_state = None
    patience = 5
    wait = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, cfg.mode)
        val_loss, y_val_t, y_val_pred_t = evaluate(model, val_loader, criterion, device, cfg.mode)

        log = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}

        if cfg.mode == "classification":
            y_true = y_val_t.numpy()
            y_pred = y_val_pred_t.numpy()
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            log.update({"val_accuracy": acc, "val_f1_macro": f1})

        history.append(log)
        print(json.dumps(log))

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test evaluation
    val_loss, y_test_t, y_test_pred_t = evaluate(model, test_loader, criterion, device, cfg.mode)

    metrics = {"mode": cfg.mode, "val_loss": val_loss}
    if cfg.mode == "classification":
        y_true = y_test_t.numpy()
        y_pred = y_test_pred_t.numpy()
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics.update({
            "test_accuracy": acc,
            "test_f1_macro": f1_macro,
            "classification_report": report,
        })
        # Save confusion matrix
        classes = label_encoder.classes_.tolist() if label_encoder is not None else ["C0", "C1", "C2", "C3"]
        plot_confusion(cm, classes, os.path.join(cfg.results_dir, "confusion_matrix.png"))
    else:
        y_true = y_test_t.numpy()
        y_pred = y_test_pred_t.numpy()
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        metrics.update({
            "test_mse": float(mse),
            "test_mae": float(mae),
            "test_rmse": rmse,
        })

    # Save model and scaler
    os.makedirs(cfg.model_dir, exist_ok=True)
    model_path = os.path.join(cfg.model_dir, f"model_{cfg.mode}.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "mode": cfg.mode,
        "input_cols": INPUT_COLS,
        "scaler_mean_": scaler.mean_.tolist(),
        "scaler_scale_": scaler.scale_.tolist(),
    }, model_path)

    # Save metrics and history
    save_metrics(metrics, os.path.join(cfg.results_dir, "metrics.json"))
    save_metrics({"history": history}, os.path.join(cfg.results_dir, "history.json"))

    # Print a few example predictions
    print("Example test predictions (first 5):")
    if cfg.mode == "classification":
        for i in range(min(5, len(y_true))):
            print({
                "true": int(y_true[i]),
                "pred": int(y_pred[i]),
                "input": X_test[i].round(2).tolist(),
            })
    else:
        for i in range(min(5, len(y_true))):
            print({
                "true": [float(x) for x in y_true[i]],
                "pred": [float(x) for x in y_pred[i]],
                "input": X_test[i].round(2).tolist(),
            })


if __name__ == "__main__":
    main()

