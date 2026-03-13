"""
PRISM-CTG Pre-training

Usage:
    cd training/
    python run_pretraining.py --data_path /path/to/data.npz --save_dir /path/to/checkpoints

The .npz file should contain:
    - fhr_segments:   [N, seq_len]   FHR signal segments
    - toco_segments:  [N, seq_len]   TOCO signal segments
    - gest_age:        [N]            Gestational age (raw, will be normalised)
    - maternal_age:    [N]            Maternal age (raw, will be normalised)
    - time_to_birth:   [N]            Time to birth (raw, will be normalised)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from config import ModelConfig
from model import PRISMCTG


# ------------------------------------------------------------------
# Data loading & preprocessing
# ------------------------------------------------------------------

def normalise_clinical_var(arr):
    """Min-max normalise to [0, 1], replacing NaN with -1 (missing marker)."""
    arr = arr.astype(np.float64)
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    arr = (arr - vmin) / (vmax - vmin + 1e-8)
    arr = np.nan_to_num(arr, nan=-1.0)
    return arr.astype(np.float32)


def load_data(data_path):
    """Load and preprocess CTG data from a single .npz file."""
    data = np.load(data_path)

    fhr_segments = data["fhr_segments"].astype(np.float32)
    toco_segments = data["toco_segments"].astype(np.float32)
    gest_age = normalise_clinical_var(data["gest_age"])
    maternal_age = normalise_clinical_var(data["maternal_age"])
    time_to_birth = normalise_clinical_var(data["time_to_birth"])

    # [N, 2, seq_len]
    ctg_signal = np.stack([fhr_segments, toco_segments], axis=1)
    # [N, 3]
    clinical_vars = np.stack([gest_age, maternal_age, time_to_birth], axis=1)

    print(f"Loaded {ctg_signal.shape[0]} samples, signal shape {ctg_signal.shape[1:]}")
    print(f"Clinical vars — gest_age: [{gest_age.min():.2f}, {gest_age.max():.2f}], "
          f"maternal_age: [{maternal_age.min():.2f}, {maternal_age.max():.2f}], "
          f"time_to_birth: [{time_to_birth.min():.2f}, {time_to_birth.max():.2f}]")
    print(f"Any NaN remaining — fhr: {np.isnan(fhr_segments).any()}, "
          f"toco: {np.isnan(toco_segments).any()}, "
          f"clinical: {np.isnan(clinical_vars).any()}")

    return torch.from_numpy(ctg_signal), torch.from_numpy(clinical_vars)


def train_val_split(ctg_signal, clinical_vars, val_ratio=0.05):
    """Random train/val split."""
    n = ctg_signal.size(0)
    n_val = int(n * val_ratio)
    n_train = n - n_val

    indices = torch.randperm(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    return (
        ctg_signal[train_idx], clinical_vars[train_idx],
        ctg_signal[val_idx], clinical_vars[val_idx],
    )


def create_dataloader(ctg, clinical_vars, batch_size, shuffle):
    dataset = TensorDataset(ctg, clinical_vars)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        drop_last=shuffle,
    )


# ------------------------------------------------------------------
# Save directory from config
# ------------------------------------------------------------------

def make_save_dir(base_dir, config):
    config_params = (
        f"ps{config.patch_size}_ed{config.embed_dim}_nh{config.num_heads}_"
        f"el{config.encoder_layers}_mr{int(config.mask_ratio * 100)}_"
        f"snt{config.signal_n_tokens}_fnt{config.feature_n_tokens}_dl{config.decoder_layers}"
    )
    save_dir = os.path.join(base_dir, f"prism_ctg_{config_params}")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    metrics_accum = {}
    num_batches = 0

    for batch_idx, (ctg_batch, vars_batch) in enumerate(dataloader):
        ctg_batch = ctg_batch.to(device)
        vars_batch = vars_batch.to(device)

        optimizer.zero_grad()
        loss, metrics = model(ctg_batch, vars_batch)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  Warning: NaN/Inf loss at batch {batch_idx}, skipping")
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        for k, v in metrics.items():
            metrics_accum[k] = metrics_accum.get(k, 0.0) + v
        num_batches += 1

    if num_batches == 0:
        return None, None

    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in metrics_accum.items()}
    return avg_loss, avg_metrics


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    metrics_accum = {}
    num_batches = 0

    for ctg_batch, vars_batch in dataloader:
        ctg_batch = ctg_batch.to(device)
        vars_batch = vars_batch.to(device)

        loss, metrics = model(ctg_batch, vars_batch)

        total_loss += loss.item()
        for k, v in metrics.items():
            metrics_accum[k] = metrics_accum.get(k, 0.0) + v
        num_batches += 1

    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in metrics_accum.items()}
    return avg_loss, avg_metrics


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PRISM-CTG Pre-training")
    parser.add_argument("--data_path", type=str, required=True, help="Path to .npz data file")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Base directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    ctg_signal, clinical_vars = load_data(args.data_path)
    train_ctg, train_vars, val_ctg, val_vars = train_val_split(
        ctg_signal, clinical_vars, val_ratio=args.val_ratio
    )
    train_loader = create_dataloader(train_ctg, train_vars, args.batch_size, shuffle=False)
    val_loader = create_dataloader(val_ctg, val_vars, args.batch_size, shuffle=False)
    print(f"Train: {len(train_ctg)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_ctg)} samples, {len(val_loader)} batches")

    # Model
    config = ModelConfig()
    model = PRISMCTG(config).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,}")

    # Save directory
    save_dir = make_save_dir(args.save_dir, config)
    print(f"Models will be saved to: {save_dir}")

    # Optimizer & scheduler (cosine decay with linear warmup)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, total_iters=args.warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs],
    )

    # Training loop
    best_val_loss = float("inf")

    for epoch in tqdm(range(args.epochs), desc="Training"):
        avg_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, device
        )

        if avg_loss is None:
            print(f"Warning: No valid batches in epoch {epoch}")
            continue

        scheduler.step()

        avg_val_loss, val_metrics = validate(model, val_loader, device)

        print(f"\nEpoch {epoch}:")
        print(f"  Train - Loss: {avg_loss:.4f}")
        print(f"          Recon: {train_metrics['recon_loss']:.4f}, "
              f"Var: {train_metrics['var_loss']:.4f}, "
              f"Feature: {train_metrics['feature_loss']:.4f}")
        print(f"          VarMAE: {train_metrics['var_mae']:.4f}, "
              f"FeatureAcc: {train_metrics['feature_acc']:.4f}")
        print(f"  Val   - Loss: {avg_val_loss:.4f}")
        print(f"          Recon: {val_metrics['recon_loss']:.4f}, "
              f"Var: {val_metrics['var_loss']:.4f}, "
              f"Feature: {val_metrics['feature_loss']:.4f}")
        print(f"          VarMAE: {val_metrics['var_mae']:.4f}, "
              f"FeatureAcc: {val_metrics['feature_acc']:.4f}")
        print(f"  Learned Weights - Recon: {train_metrics['recon_weight']:.4f}, "
              f"Var: {train_metrics['var_weight']:.4f}, "
              f"Feature: {train_metrics['feature_weight']:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        model_path = os.path.join(save_dir, f"model_epoch_{epoch}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"  Saved: {model_path}")

    print(f"\nPre-training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
