#!/usr/bin/env python3
"""
run_047 (spec'd as "run_045"): GNN GENERALIZATION FIX — DropEdge + Neighbor Feature Masking.

run_039 (Transductive GeoSAGE GNN) hit 0.7684 OOF but is expected to collapse on
public: with ~22 buildings/cell × 191 features, the leave-one-out neighbor mean over
a geo3 clique is a near-unique per-cell FINGERPRINT, so the model memorizes
cell -> label instead of learning neighborhood physics.

Fix: destroy the fingerprint during training with two stochastic mechanisms, applied
ONLY in train mode (full neighborhood used at eval):
  1. DropEdge (adapted to clique aggregation): per forward pass, include each node in
     the neighbor aggregation with prob (1-DROP_P). At DROP_P=0.5 each node sees ~half
     its cell-mates (~11 of 22) — the spec's "~11 random neighbors" behaviour. There is
     no explicit edge list in run_039 (scatter over geo3 cliques), so per-node inclusion
     is the efficient, faithful clique analogue of dropping edges.
  2. Neighbor Feature Masking: per forward pass, zero out MASK_Q of feature dimensions
     of the NEIGHBOR aggregation path only (the node's own features stay intact).

Accept LOWER OOF than run_039 in exchange for public generalization. Gate on the blend
+ within-cell-variance fingerprint test, NOT on OOF (see run_039's optimism note).

NOTE: registered as run_047 because runs/run_045 and runs/run_046 are already taken by
unrelated (failed) experiments. Run from project root:
    env/bin/python src/run_047.py [--quick] [--drop-p 0.5] [--mask-q 0.3] [--epochs 200] [--lr 0.0021]
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer

warnings.filterwarnings("ignore"); np.seterr(all="ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_manager import PROCESSED_DIR, RunManager
from run_trees_260k import _oof_f1, pairwise_diagnostic

RANDOM_STATE = 42
CV_FOLDS = 5
# run_039 used EPOCHS=100, LR=3e-3. Stochastic noise -> noisier gradients -> train 2x
# longer at 0.7x LR (spec).
EPOCHS = 200
LR = 2.1e-3
WEIGHT_DECAY = 1e-4
HIDDEN = 512
DROPOUT = 0.3
DROP_P = 0.5   # DropEdge: P(drop a node from neighbor aggregation) per forward pass
MASK_Q = 0.3   # Neighbor Feature Masking: fraction of neighbor feature dims zeroed


class GeoSAGELayer(nn.Module):
    """Leave-one-out clique mean aggregation with train-only DropEdge + neighbor masking."""

    def __init__(self, in_dim: int, out_dim: int, drop_p: float, mask_q: float):
        super().__init__()
        self.proj = nn.Linear(in_dim * 2, out_dim)
        self.norm = nn.BatchNorm1d(out_dim)
        self.drop_p = drop_p
        self.mask_q = mask_q

    def forward(self, x: torch.Tensor, geo3_idx: torch.Tensor, num_geo3: int) -> torch.Tensor:
        d = x.shape[1]

        # --- neighbor feature masking (train only): mask the NEIGHBOR path, keep self intact
        if self.training and self.mask_q > 0:
            feat_mask = (torch.rand(d, device=x.device) >= self.mask_q).to(x.dtype)
            x_neigh = x * feat_mask
        else:
            x_neigh = x

        # --- DropEdge (train only): per-node inclusion in the aggregation
        if self.training and self.drop_p > 0:
            keep = (torch.rand(x.shape[0], 1, device=x.device) >= self.drop_p).to(x.dtype)
        else:
            keep = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)

        xk = x_neigh * keep  # masked + kept contribution of each node

        geo3_sum = torch.zeros(num_geo3, d, device=x.device, dtype=x.dtype)
        geo3_sum.scatter_add_(0, geo3_idx.unsqueeze(1).expand(-1, d), xk)
        geo3_cnt = torch.zeros(num_geo3, 1, device=x.device, dtype=x.dtype)
        geo3_cnt.scatter_add_(0, geo3_idx.unsqueeze(1), keep)

        # leave-one-out: remove self's own (masked, kept) contribution
        neighbor_sum = geo3_sum[geo3_idx] - xk
        neighbor_cnt = geo3_cnt[geo3_idx] - keep
        neighbor_mean = neighbor_sum / neighbor_cnt.clamp(min=1.0)

        out = self.proj(torch.cat([x, neighbor_mean], dim=1))  # self path uses UNMASKED x
        return F.gelu(self.norm(out))


class TransductiveGNN(nn.Module):
    def __init__(self, in_features: int, num_geo3: int, hidden_dim: int = HIDDEN,
                 dropout: float = DROPOUT, drop_p: float = DROP_P, mask_q: float = MASK_Q):
        super().__init__()
        self.num_geo3 = num_geo3
        self.embed = nn.Sequential(
            nn.Linear(in_features, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.sage1 = GeoSAGELayer(hidden_dim, hidden_dim, drop_p, mask_q)
        self.drop1 = nn.Dropout(dropout)
        self.sage2 = GeoSAGELayer(hidden_dim, hidden_dim // 2, drop_p, mask_q)
        self.drop2 = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim // 2, 3)

    def forward(self, x, geo3_idx):
        h = self.embed(x)
        h = self.drop1(self.sage1(h, geo3_idx, self.num_geo3))
        h = self.drop2(self.sage2(h, geo3_idx, self.num_geo3))
        return self.head(h)


def train_transductive(X_full_s, y_train, geo3_idx_full, num_geo3, fold, train_idx, val_idx,
                       epochs, lr, drop_p, mask_q):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    X_t = torch.tensor(X_full_s, dtype=torch.float32).to(device)
    geo_t = torch.tensor(geo3_idx_full, dtype=torch.long).to(device)
    y_t = torch.tensor(y_train - 1, dtype=torch.long).to(device)

    model = TransductiveGNN(X_full_s.shape[1], num_geo3, drop_p=drop_p, mask_q=mask_q).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=1, epochs=epochs)
    criterion = nn.CrossEntropyLoss()

    best_f1, best_weights = 0.0, None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_t, geo_t)
        loss = criterion(out[train_idx], y_t[train_idx])
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            out_val = model(X_t, geo_t)[val_idx]  # eval pass: full neighborhood, no drop/mask
            f1 = f1_score(y_train[val_idx], out_val.argmax(1).cpu().numpy() + 1,
                          average="micro", labels=[1, 2, 3])
        if f1 > best_f1:
            best_f1, best_weights = f1, copy.deepcopy(model.state_dict())

    print(f"  Fold {fold} Best F1: {best_f1:.4f}")
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        out_all = torch.softmax(model(X_t, geo_t), dim=1).cpu().numpy()
    return out_all[val_idx], out_all[len(y_train):]


def within_cell_variance(proba: np.ndarray, cell_ids: np.ndarray) -> float:
    """Mean over geo3 cells (size>=2) of the per-cell variance of predicted probs (avg over 3 classes)."""
    df = pd.DataFrame(proba, columns=["p1", "p2", "p3"]); df["cell"] = cell_ids
    g = df.groupby("cell")[["p1", "p2", "p3"]].var()  # sample var per cell per class (NaN if size 1)
    cell_var = g.mean(axis=1).dropna()
    return float(cell_var.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--drop-p", type=float, default=DROP_P)
    ap.add_argument("--mask-q", type=float, default=MASK_Q)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    args = ap.parse_args()
    t0 = time.time()

    print("── Loading Data ──")
    X_train = pd.read_csv(PROCESSED_DIR / "X_train_run012.csv").fillna(0)
    y_train = pd.read_csv(PROCESSED_DIR / "y_train_full.csv")["damage_grade"].to_numpy()
    X_test = pd.read_csv(PROCESSED_DIR / "X_test_run012.csv").fillna(0)
    df_tr_raw = pd.read_csv(ROOT / "data" / "driven_data" / "train_values.csv")
    df_te_raw = pd.read_csv(ROOT / "data" / "driven_data" / "test_values.csv")

    geo3_raw = np.concatenate([df_tr_raw["geo_level_3_id"].values, df_te_raw["geo_level_3_id"].values])
    unique_geo3, geo3_idx_full = np.unique(geo3_raw, return_inverse=True)
    num_geo3 = len(unique_geo3)
    geo3_train = geo3_idx_full[: len(y_train)]

    X_full = np.vstack([X_train.values, X_test.values])
    print("── Scaling Features (QuantileTransformer) ──")
    scaler = QuantileTransformer(output_distribution="normal", random_state=RANDOM_STATE)
    X_full_s = scaler.fit_transform(X_full).astype(np.float32)

    epochs = 5 if args.quick else args.epochs
    folds = 1 if args.quick else CV_FOLDS
    print(f"\n── Training run_047 GNN  (drop_p={args.drop_p}, mask_q={args.mask_q}, "
          f"epochs={epochs}, lr={args.lr}, {num_geo3} geo3 cells) ──")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros((len(y_train), 3), dtype=np.float32)
    test_folds, scores = [], []
    splits = list(skf.split(X_train, y_train))
    for fold, (tri, vai) in enumerate(splits[:folds], start=1):
        vp, tp = train_transductive(X_full_s, y_train, geo3_idx_full, num_geo3, fold, tri, vai,
                                    epochs, args.lr, args.drop_p, args.mask_q)
        oof_proba[vai] = vp
        test_folds.append(tp)
        scores.append(f1_score(y_train[vai], vp.argmax(1) + 1, average="micro", labels=[1, 2, 3]))

    if args.quick:
        print(f"\n[quick] fold1 F1={scores[0]:.4f} in {time.time()-t0:.0f}s")
        return

    mean_f1, std_f1 = float(np.mean(scores)), float(np.std(scores, ddof=1))
    test_proba = np.mean(test_folds, axis=0)
    print(f"\nrun_047 GNN CV (solo OOF) = {mean_f1:.4f} ± {std_f1:.4f}")

    # ===================== DIAGNOSTICS =====================
    p26 = np.load(ROOT / "runs" / "run_026" / "oof_proba.npy").astype(np.float64)
    p26_test = np.load(ROOT / "runs" / "run_026" / "test_proba.npy").astype(np.float64)
    f26 = _oof_f1(p26, y_train)
    g = oof_proba.astype(np.float64)

    print("\n" + "=" * 64)
    print("DIAGNOSTIC 1 (MOST IMPORTANT) — WITHIN-CELL VARIANCE FINGERPRINT TEST")
    print("=" * 64)
    wcv_047 = within_cell_variance(g, geo3_train)
    line = f"  run_047 within-cell variance: {wcv_047:.6f}"
    p039_path = ROOT / "runs" / "run_039" / "oof_proba.npy"
    if p039_path.exists():
        wcv_039 = within_cell_variance(np.load(p039_path).astype(np.float64), geo3_train)
        ratio = wcv_047 / wcv_039 if wcv_039 > 0 else float("nan")
        print(f"  run_039 within-cell variance: {wcv_039:.6f}")
        print(line)
        print(f"  ratio run_047/run_039: {ratio:.2f}x  "
              f"({'HIGHER → less fingerprinting → GOOD' if ratio > 1.05 else 'NOT higher → fingerprint NOT broken'})")
    else:
        print(line + "  (run_039 OOF not found for comparison)")

    print("\n" + "=" * 64)
    print("DIAGNOSTIC 2 — loss correlation / disagreement / per-grade vs run_026")
    print("=" * 64)
    pairwise_diagnostic("run_026", p26, "run_047", g, y_train)

    print("\n" + "=" * 64)
    print("DIAGNOSTIC 3 — GNN unique-wins grade profile (GNN right, run_026 wrong)")
    print("=" * 64)
    gnn_pred, p26_pred = g.argmax(1) + 1, p26.argmax(1) + 1
    uw = (gnn_pred == y_train) & (p26_pred != y_train)
    lw = (gnn_pred != y_train) & (p26_pred == y_train)
    print(f"  GNN-unique wins: {uw.sum()}   run_026-unique wins: {lw.sum()}   net: {uw.sum()-lw.sum():+d}")
    for grade in [1, 2, 3]:
        m = uw & (y_train == grade)
        print(f"    win grade {grade}: {m.sum():>6}  ({100*m.sum()/max(uw.sum(),1):.1f}% of wins)")

    print("\n" + "=" * 64)
    print("DIAGNOSTIC 4 — BLEND CURVE vs run_026 (proba space)")
    print("=" * 64)
    print(f"  run_026 solo OOF: {f26:.4f}")
    best_w, best_f1b = 0.0, f26
    for w in [0.0, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30]:
        bl = (1 - w) * p26 + w * g
        fb = _oof_f1(bl, y_train)
        mark = ""
        if fb > best_f1b:
            best_f1b, best_w = fb, w
        print(f"    w={w:.2f} GNN : OOF={fb:.4f}  (Δ {fb-f26:+.4f})")
    print(f"  --> best blend weight {best_w:.2f}, OOF {best_f1b:.4f} (Δ vs run_026 {best_f1b-f26:+.4f})")

    # ===================== SAVE =====================
    rm = RunManager()
    run_id = rm.get_next_run_id()
    rm.create_run(
        description=f"GNN generalization fix (DropEdge p={args.drop_p} + neighbor mask q={args.mask_q}); spec'd run_045",
        model_type="PyTorch_GNN",
        feature_set="run_012_features+geo3_graph+dropedge+neighbor_mask",
        params={"epochs": epochs, "lr": args.lr, "hidden": HIDDEN, "dropout": DROPOUT,
                "drop_p": args.drop_p, "mask_q": args.mask_q, "base_run": "run_039"},
        run_id=run_id, objective="multiclass", n_features=X_train.shape[1],
        cv_folds=CV_FOLDS, cv_metric="micro_f1",
        notes=f"within_cell_var={wcv_047:.6f}; best_blend_w={best_w:.2f}; best_blend_oof={best_f1b:.4f}",
    )
    rm.save_cv_scores(run_id, scores, mean_f1, std_f1)
    run_dir = rm.run_path(run_id)
    np.save(run_dir / "oof_proba.npy", oof_proba.astype(np.float32))
    np.save(run_dir / "test_proba.npy", test_proba.astype(np.float32))
    rm.save_submission(run_id, pd.DataFrame(
        {"building_id": df_te_raw["building_id"].values, "damage_grade": test_proba.argmax(1) + 1}))

    # blend submission CSVs at the requested weights (proba space) — NOT auto-submitted
    for w in [0.03, 0.05, 0.07, 0.10, 0.15]:
        bl_test = (1 - w) * p26_test + w * test_proba.astype(np.float64)
        pd.DataFrame({"building_id": df_te_raw["building_id"].values,
                      "damage_grade": bl_test.argmax(1) + 1}).to_csv(
            run_dir / f"blend_run026_w{int(w*100):02d}.csv", index=False)

    print(f"\nRegistered {run_id} ({run_dir}) in {time.time()-t0:.1f}s")
    print(f"Blend CSVs saved at w=3/5/7/10/15%. NOTHING SUBMITTED.")


if __name__ == "__main__":
    main()
