#!/usr/bin/env python3
"""
run_040: FT-Transformer (Feature Tokenizer Transformer) on run_012 features.

Projects 191 continuous features into a d_model space, prepends a CLS token,
and runs a TransformerEncoder. 
Targeting high decorrelation from tree models for blending.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import copy
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore'); np.seterr(all='ignore')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_manager import PROCESSED_DIR, RunManager
from run_trees_260k import (
    pairwise_diagnostic,
)

RANDOM_STATE = 42
CV_FOLDS = 5
EPOCHS = 15
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-4

class FeatureTokenizer(nn.Module):
    def __init__(self, num_features: int, d_model: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_features, d_model) * 0.01)
        self.b = nn.Parameter(torch.zeros(num_features, d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_features)
        # out: (B, num_features, d_model)
        out = x.unsqueeze(-1) * self.W.unsqueeze(0) + self.b.unsqueeze(0)
        return out

class FTTransformer(nn.Module):
    def __init__(self, num_features: int, d_model: int = 64, n_heads: int = 4, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.01)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        tokens = self.tokenizer(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_seq = torch.cat([cls_tokens, tokens], dim=1)
        
        out_seq = self.transformer(x_seq)
        
        cls_out = out_seq[:, 0, :]
        cls_out = self.ln(cls_out)
        return self.head(cls_out)

def train_fold(X_tr, y_tr, X_va, y_va, fold: int):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Scale features
    scaler = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE)
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_va_s = scaler.transform(X_va).astype(np.float32)
    
    train_ds = TensorDataset(torch.tensor(X_tr_s), torch.tensor(y_tr - 1, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_va_s), torch.tensor(y_va - 1, dtype=torch.long))
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)
    
    model = FTTransformer(num_features=X_tr.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=len(train_dl), epochs=EPOCHS
    )
    criterion = nn.CrossEntropyLoss()
    
    best_f1 = 0.0
    best_weights = None
    
    for epoch in range(EPOCHS):
        model.train()
        for X_b, y_b in train_dl:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        model.eval()
        preds = []
        with torch.no_grad():
            for X_b, _ in val_dl:
                X_b = X_b.to(device)
                out = model(X_b)
                preds.append(torch.softmax(out, dim=1).cpu().numpy())
        
        preds = np.concatenate(preds)
        f1 = f1_score(y_va, preds.argmax(axis=1) + 1, average="micro", labels=[1, 2, 3])
        
        if f1 > best_f1:
            best_f1 = f1
            best_weights = copy.deepcopy(model.state_dict())
            
    print(f"  Fold {fold} Best F1: {best_f1:.4f}")
    model.load_state_dict(best_weights)
    model.eval()
    
    val_preds = []
    with torch.no_grad():
        for X_b, _ in val_dl:
            X_b = X_b.to(device)
            val_preds.append(torch.softmax(model(X_b), dim=1).cpu().numpy())
    
    return np.concatenate(val_preds), model, scaler


def main():
    t0 = time.time()
    
    print("── Loading Data ──")
    X = pd.read_csv(PROCESSED_DIR / "X_train_run012.csv").fillna(0)
    y = pd.read_csv(PROCESSED_DIR / "y_train_full.csv")["damage_grade"].to_numpy()
    X_test = pd.read_csv(PROCESSED_DIR / "X_test_run012.csv").fillna(0)
    
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    oof_proba = np.zeros((len(y), 3), dtype=np.float32)
    test_folds_proba = []
    scores = []
    
    print(f"\n── Training FT-Transformer (5-fold CV) ──")
    for fold, (tri, vai) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X.iloc[tri].values, y[tri]
        X_va, y_va = X.iloc[vai].values, y[vai]
        
        val_preds, model, scaler = train_fold(X_tr, y_tr, X_va, y_va, fold)
        oof_proba[vai] = val_preds
        
        device = next(model.parameters()).device
        X_te_s = scaler.transform(X_test.values).astype(np.float32)
        te_ds = TensorDataset(torch.tensor(X_te_s))
        te_dl = DataLoader(te_ds, batch_size=BATCH_SIZE * 2, shuffle=False)
        
        te_preds = []
        with torch.no_grad():
            for X_b, in te_dl:
                X_b = X_b.to(device)
                te_preds.append(torch.softmax(model(X_b), dim=1).cpu().numpy())
        test_folds_proba.append(np.concatenate(te_preds))
        
        scores.append(f1_score(y_va, val_preds.argmax(axis=1) + 1, average="micro", labels=[1, 2, 3]))

    mean_f1 = float(np.mean(scores))
    std_f1 = float(np.std(scores, ddof=1))
    
    print(f"\nFT-Transformer CV = {mean_f1:.4f} ± {std_f1:.4f}")
    
    test_proba = np.mean(test_folds_proba, axis=0)
    
    rm = RunManager()
    run_id = rm.get_next_run_id()
    
    rm.create_run(
        description=f"FT-Transformer on run_012 features",
        model_type="PyTorch_Transformer",
        feature_set=f"run_012_features",
        params={"epochs": EPOCHS, "lr": LR, "d_model": 64, "n_layers": 3},
        run_id=run_id,
        objective="multiclass",
        n_features=X.shape[1],
        cv_folds=CV_FOLDS,
        cv_metric="micro_f1",
    )
    rm.save_cv_scores(run_id, scores, mean_f1, std_f1)
    run_dir = rm.run_path(run_id)
    np.save(run_dir / "oof_proba.npy", oof_proba.astype(np.float32))
    np.save(run_dir / "test_proba.npy", test_proba.astype(np.float32))
    
    sub = pd.DataFrame({"building_id": pd.read_csv(ROOT / "data" / "driven_data" / "test_values.csv")["building_id"].values,
                        "damage_grade": test_proba.argmax(axis=1) + 1})
    rm.save_submission(run_id, sub)
    
    print("\nEvaluating Blend vs run_026 (SOTA Blend):")
    p26_oof = np.load(ROOT / "runs" / "run_026" / "oof_proba.npy").astype(np.float64)
    pairwise_diagnostic("run_026", p26_oof, run_id, oof_proba.astype(np.float64), y)
    
    print(f"\nRegistered {run_id} in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
