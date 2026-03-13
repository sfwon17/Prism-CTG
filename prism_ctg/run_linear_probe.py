"""
PRISM-CTG Linear Probing

Expects data_dir to contain:
    - X_train.npy  [N, 2, 1200]
    - y_train.npy  [N]
    - X_test.npy   [N, 2, 1200]
    - y_test.npy   [N]

Train is split 90/10 into train/val. Early stopping on val AUC.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score

from config import ModelConfig
from model import PRISMCTG


# ------------------------------------------------------------------
# Classifier
# ------------------------------------------------------------------

class DownstreamClassifier(nn.Module):
    def __init__(self, encoder, num_classes=2, pooling="concat_cls"):
        super().__init__()
        embed_dim = encoder.config.embed_dim
        self.encoder = encoder
        self.encoder.eval()
        self.pooling = pooling

        for p in self.encoder.parameters():
            p.requires_grad = False

        input_dim = embed_dim * 3 if pooling == "concat_cls" else embed_dim
        self.head = nn.Linear(input_dim, num_classes)

    def forward(self, ctg):
        with torch.no_grad():
            enc = self.encoder.encode(ctg)

        if self.pooling == "cls_mean":
            pooled = (enc["cls_recon"] + enc["cls_var"] + enc["cls_feature"]) / 3
        elif self.pooling == "patches_mean":
            pooled = enc["patches"].mean(dim=1)
        elif self.pooling == "concat_cls":
            pooled = torch.cat([enc["cls_recon"], enc["cls_var"], enc["cls_feature"]], dim=-1)
        else:
            pooled = enc[self.pooling]

        return self.head(pooled)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--pooling", type=str, default="concat_cls")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load encoder
    config = ModelConfig()
    encoder = PRISMCTG(config)
    encoder.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    encoder.to(device).eval()

    classifier = DownstreamClassifier(encoder, pooling=args.pooling).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.head.parameters(), lr=0.01)

    # Load data
    X_train = np.load(os.path.join(args.data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(args.data_dir, "y_train.npy"))
    X_test = np.load(os.path.join(args.data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(args.data_dir, "y_test.npy"))

    # 90/10 split on train for val
    n = len(X_train)
    idx = np.random.permutation(n)
    split = int(n * 0.9)
    train_idx, val_idx = idx[:split], idx[split:]

    def make_loader(X, y, shuffle):
        ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
        return DataLoader(ds, batch_size=32, shuffle=shuffle)

    train_loader = make_loader(X_train[train_idx], y_train[train_idx], shuffle=True)
    val_loader = make_loader(X_train[val_idx], y_train[val_idx], shuffle=False)
    test_loader = make_loader(X_test, y_test, shuffle=False)

    # Train with early stopping on val AUC
    best_auc = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(100):
        classifier.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(classifier(x), y).backward()
            optimizer.step()

        # Val AUC
        classifier.eval()
        probs, labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                p = F.softmax(classifier(x), dim=1)[:, 1]
                probs.extend(p.cpu().numpy())
                labels.extend(y.numpy())
        auc = roc_auc_score(labels, probs)

        if auc > best_auc:
            best_auc = auc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= 3:
            break

    # Test with best model
    classifier.load_state_dict(best_state)
    classifier.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            p = F.softmax(classifier(x), dim=1)[:, 1]
            probs.extend(p.cpu().numpy())
            labels.extend(y.numpy())

    test_auc = roc_auc_score(labels, probs)
    print(f"Test AUC: {test_auc:.4f}")


if __name__ == "__main__":
    main()
