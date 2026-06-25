#!/usr/bin/env python3
"""
run_041_gnn: Transductive Spatial GNN on geo_level_2 cliques (Embedding Extractor)

Trains the macro-GNN on geo_level_2 and saves the 256-dimensional node embeddings 
(Out-Of-Fold for train, averaged for test) to be stacked into a LightGBM model.
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

warnings.filterwarnings('ignore'); np.seterr(all='ignore')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_manager import PROCESSED_DIR, RunManager
from run_trees_260k import (
    _oof_f1,
    build_blend_submission,
    pairwise_diagnostic,
    threeway_optimize,
)

RANDOM_STATE = 42
CV_FOLDS = 5
EPOCHS = 100
LR = 3e-3
WEIGHT_DECAY = 1e-4

class GeoSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim * 2, out_dim)
        self.norm = nn.BatchNorm1d(out_dim)
        
    def forward(self, x: torch.Tensor, geo3_idx: torch.Tensor, num_geo3: int) -> torch.Tensor:
        # x is (N, in_dim), geo3_idx is (N,)
        geo3_sum = torch.zeros(num_geo3, x.shape[1], device=x.device, dtype=x.dtype)
        geo3_sum.scatter_add_(0, geo3_idx.unsqueeze(1).expand(-1, x.shape[1]), x)
        
        geo3_count = torch.zeros(num_geo3, 1, device=x.device, dtype=x.dtype)
        geo3_count.scatter_add_(0, geo3_idx.unsqueeze(1), torch.ones_like(geo3_idx.unsqueeze(1), dtype=x.dtype))
        
        # Neighbor sum excluding self
        neighbor_sum = geo3_sum[geo3_idx] - x
        neighbor_count = geo3_count[geo3_idx] - 1.0
        
        # Mean aggregation
        neighbor_mean = neighbor_sum / neighbor_count.clamp(min=1.0)
        
        # If node has no neighbors, neighbor_mean is 0, which is fine
        out = self.proj(torch.cat([x, neighbor_mean], dim=1))
        out = self.norm(out)
        return F.gelu(out)

class TransductiveGNN(nn.Module):
    def __init__(self, in_features: int, num_geo3: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.num_geo3 = num_geo3
        
        self.embed = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.sage1 = GeoSAGELayer(hidden_dim, hidden_dim)
        self.drop1 = nn.Dropout(dropout)
        
        self.sage2 = GeoSAGELayer(hidden_dim, hidden_dim // 2)
        self.drop2 = nn.Dropout(dropout)
        
        self.head = nn.Linear(hidden_dim // 2, 3)
        
    def forward_embed(self, x: torch.Tensor, geo3_idx: torch.Tensor):
        h = self.embed(x)
        h = self.sage1(h, geo3_idx, self.num_geo3)
        h = self.drop1(h)
        h = self.sage2(h, geo3_idx, self.num_geo3)
        h = self.drop2(h)
        return h
        
    def forward(self, x: torch.Tensor, geo3_idx: torch.Tensor):
        h = self.forward_embed(x, geo3_idx)
        return self.head(h)


def train_transductive(X_full_s, y_train, geo3_idx_full, num_geo3, fold: int, train_idx, val_idx):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # We load everything into memory and onto device
    X_t = torch.tensor(X_full_s, dtype=torch.float32).to(device)
    geo_t = torch.tensor(geo3_idx_full, dtype=torch.long).to(device)
    y_t = torch.tensor(y_train - 1, dtype=torch.long).to(device) # Only first N rows are valid
    
    model = TransductiveGNN(in_features=X_full_s.shape[1], num_geo3=num_geo3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=1, epochs=EPOCHS
    )
    criterion = nn.CrossEntropyLoss()
    
    best_f1 = 0.0
    best_weights = None
    
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(X_t, geo_t)
        
        # Only compute loss on train mask
        loss = criterion(out[train_idx], y_t[train_idx])
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            out_val = out[val_idx]
            preds = torch.softmax(out_val, dim=1).cpu().numpy()
            f1 = f1_score(y_train[val_idx], preds.argmax(axis=1) + 1, average="micro", labels=[1, 2, 3])
            
        if f1 > best_f1:
            best_f1 = f1
            best_weights = copy.deepcopy(model.state_dict())
            
    print(f"  Fold {fold} Best F1: {best_f1:.4f}")
    model.load_state_dict(best_weights)
    model.eval()
    
    with torch.no_grad():
        out_all_embeds = model.forward_embed(X_t, geo_t).cpu().numpy()
        
    val_embeds = out_all_embeds[val_idx]
    test_embeds = out_all_embeds[len(y_train):]
    
    return val_embeds, test_embeds


def main():
    t0 = time.time()
    
    print("── Loading Data ──")
    X_train = pd.read_csv(PROCESSED_DIR / "X_train_run012.csv").fillna(0)
    y_train = pd.read_csv(PROCESSED_DIR / "y_train_full.csv")["damage_grade"].to_numpy()
    X_test = pd.read_csv(PROCESSED_DIR / "X_test_run012.csv").fillna(0)
    
    # We need geo_level_2_id from the original data
    df_tr_raw = pd.read_csv(ROOT / "data" / "driven_data" / "train_values.csv")
    df_te_raw = pd.read_csv(ROOT / "data" / "driven_data" / "test_values.csv")
    geo3_raw = np.concatenate([df_tr_raw["geo_level_2_id"].values, df_te_raw["geo_level_2_id"].values])
    
    # Remap geo3 to contiguous 0..num_geo3-1
    unique_geo3, geo3_idx_full = np.unique(geo3_raw, return_inverse=True)
    num_geo3 = len(unique_geo3)
    
    X_full = np.vstack([X_train.values, X_test.values])
    
    print("── Scaling Features ──")
    scaler = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE)
    X_full_s = scaler.fit_transform(X_full).astype(np.float32)
    
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    oof_embeds = np.zeros((len(y_train), 256), dtype=np.float32)
    test_folds_embeds = []
    
    print(f"\n── Training Transductive GNN ({num_geo3} unique geo2, 5-fold CV) for Feature Extraction ──")
    for fold, (tri, vai) in enumerate(skf.split(X_train, y_train), start=1):
        
        val_embeds, test_embeds = train_transductive(X_full_s, y_train, geo3_idx_full, num_geo3, fold, tri, vai)
        oof_embeds[vai] = val_embeds
        test_folds_embeds.append(test_embeds)

    test_embeds_avg = np.mean(test_folds_embeds, axis=0)
    
    print(f"\nSaving embeddings to {PROCESSED_DIR}")
    np.save(PROCESSED_DIR / "X_train_gnn2_embeds.npy", oof_embeds)
    np.save(PROCESSED_DIR / "X_test_gnn2_embeds.npy", test_embeds_avg)
    
    print(f"\nExtraction complete in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
