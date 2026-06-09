#!/usr/bin/env python3
"""
run_013: ResNet-style MLP trained end-to-end on full 260k rows.

Uses run_012's unweighted entity embeddings as warm-start, then jointly
optimises embeddings + ResNet body with label smoothing and no class weights.

Architecture:
  Entity embeds (warm-started) + numeric bin embeds + 51 tabular features
  → Linear(231, 256) → LayerNorm → GELU → 3× ResBlock(256) → 3-class head

Training:
  5-fold stratified CV, AdamW (body lr=5e-4, embed lr=1e-4), cosine LR,
  label smoothing 0.05, early stopping patience=15.

Run from project root:
    python src/run_013.py
"""

from __future__ import annotations

import copy
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib

from embed import (
    EMBED_COLS,
    NUMERIC_COLS,
    RANDOM_STATE,
    apply_age_sentinel,
    get_device,
    load_raw_data,
    set_seed,
)
from mlp_model import (
    CAT_COLS,
    NUM_BINS,
    NepalResNetMLP,
    NumericBins,
    warm_start_embeddings,
)
from run_manager import PREPROCESSING_DIR, PROCESSED_DIR, RunManager

print = partial(print, flush=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
EMBED_UW_DIR = ROOT / "models" / "embedding_unweighted"
UW_CKPT = EMBED_UW_DIR / "embedding_network.pt"

# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE = 2048
MAX_EPOCHS = 100
PATIENCE = 15
LR_BODY = 5e-4
LR_EMBED = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
CV_FOLDS = 5
SEED = 42

# Tabular feature columns from X_train_run012.csv (non-embedding cols)
_ALL_COLS: list[str] | None = None


def _tab_cols() -> list[str]:
    global _ALL_COLS
    if _ALL_COLS is None:
        sample = pd.read_csv(PROCESSED_DIR / "X_train_run012.csv", nrows=0)
        _ALL_COLS = [c for c in sample.columns if "_emb_" not in c]
    return _ALL_COLS


# ── Dataset ────────────────────────────────────────────────────────────────────

class NepalDataset(Dataset):
    def __init__(
        self,
        cat_matrix: np.ndarray,    # (N, 11) int64
        num_bins: np.ndarray,      # (N, 5) int64
        tabular: np.ndarray,       # (N, D) float32
        y: np.ndarray | None = None,  # (N,) int64 — 0-indexed (grade-1)
    ):
        self.cat = torch.from_numpy(cat_matrix)
        self.bins = torch.from_numpy(num_bins)
        self.tab = torch.from_numpy(tabular.astype(np.float32))
        self.y = torch.from_numpy(y) if y is not None else None

    def __len__(self) -> int:
        return len(self.tab)

    def __getitem__(self, i: int):
        if self.y is not None:
            return self.cat[i], self.bins[i], self.tab[i], self.y[i]
        return self.cat[i], self.bins[i], self.tab[i]


# ── Data preparation ───────────────────────────────────────────────────────────

def load_categorical_indices() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    """Encode all 260k training rows and test rows using run_012's label encoders."""
    from sklearn.preprocessing import LabelEncoder

    train_full, test_raw = load_raw_data()
    train_full, test_raw = apply_age_sentinel(train_full, test_raw)
    train_full = train_full.reset_index(drop=True)
    test_raw = test_raw.reset_index(drop=True)

    encoders: dict[str, LabelEncoder] = joblib.load(EMBED_UW_DIR / "label_encoders.pkl")

    cat_train: dict[str, np.ndarray] = {}
    cat_test: dict[str, np.ndarray] = {}
    n_emb: dict[str, int] = {}

    for col in EMBED_COLS:
        le: LabelEncoder = encoders[col]

        tr_str = train_full[col].astype(str).values
        tr = np.zeros(len(train_full), dtype=np.int64)
        known = np.isin(tr_str, le.classes_)
        if known.any():
            tr[known] = le.transform(tr_str[known]) + 1
        cat_train[col] = tr

        te_str = test_raw[col].astype(str).values
        te = np.zeros(len(test_raw), dtype=np.int64)
        known = np.isin(te_str, le.classes_)
        if known.any():
            te[known] = le.transform(te_str[known]) + 1
        cat_test[col] = te

        n_emb[col] = len(le.classes_) + 1

    return cat_train, cat_test, n_emb


def load_numeric_raw() -> tuple[np.ndarray, np.ndarray]:
    """Return scaled numerics (5 cols, no age_unknown) for bin-fitting."""
    scaler = joblib.load(PREPROCESSING_DIR / "scaler.pkl")
    train_full, test_raw = load_raw_data()
    train_full, test_raw = apply_age_sentinel(train_full, test_raw)
    return (
        scaler.transform(train_full[NUMERIC_COLS]).astype(np.float32),
        scaler.transform(test_raw[NUMERIC_COLS]).astype(np.float32),
    )


def load_tabular() -> tuple[np.ndarray, np.ndarray]:
    """Load non-embedding tabular features from X_train/test_run012.csv."""
    cols = _tab_cols()
    X_tr = pd.read_csv(PROCESSED_DIR / "X_train_run012.csv", usecols=cols)
    X_te = pd.read_csv(PROCESSED_DIR / "X_test_run012.csv", usecols=cols)
    return X_tr.values.astype(np.float32), X_te.values.astype(np.float32)


def load_labels() -> np.ndarray:
    """Returns 0-indexed labels: grade 1→0, 2→1, 3→2."""
    y = pd.read_csv(PROCESSED_DIR / "y_train_full.csv")["damage_grade"].to_numpy()
    return (y - 1).astype(np.int64)


# ── Training helpers ───────────────────────────────────────────────────────────

def _make_loader(ds: NepalDataset, shuffle: bool, bs: int) -> DataLoader:
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0, pin_memory=False)


def predict_proba(model: NepalResNetMLP, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Returns (N, 3) softmax probabilities."""
    model.eval()
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            cat, bins_, tab = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            logits = model(cat, bins_, tab)
            parts.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(parts, axis=0)


def train_fold(
    model: NepalResNetMLP,
    train_ds: NepalDataset,
    val_ds: NepalDataset,
    device: torch.device,
    fold: int,
) -> tuple[NepalResNetMLP, float]:
    """Train one CV fold; returns model loaded with best checkpoint and best val F1."""
    torch.manual_seed(SEED + fold)

    train_dl = _make_loader(train_ds, shuffle=True, bs=BATCH_SIZE)
    val_dl = _make_loader(val_ds, shuffle=False, bs=BATCH_SIZE * 2)

    embed_params = list(model.embeddings.parameters()) + list(model.bin_embeds.parameters())
    body_params = [
        p for n, p in model.named_parameters()
        if not (n.startswith("embeddings") or n.startswith("bin_embeds"))
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": body_params, "lr": LR_BODY, "weight_decay": WEIGHT_DECAY},
            {"params": embed_params, "lr": LR_EMBED, "weight_decay": WEIGHT_DECAY},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    best_val_f1 = 0.0
    best_state: dict | None = None
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_dl:
            cat, bins_, tab, y = [b.to(device) for b in batch]
            optimizer.zero_grad()
            loss = criterion(model(cat, bins_, tab), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        val_proba = predict_proba(model, val_dl, device)
        val_preds = val_proba.argmax(axis=1)
        # retrieve ground truth from dataset
        y_val = val_ds.y.numpy()
        val_f1 = float(f1_score(y_val, val_preds, average="micro"))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"    epoch {epoch:3d}: loss={epoch_loss/n_batches:.4f} "
                f"val_F1={val_f1:.4f}  best={best_val_f1:.4f}"
            )

        if no_improve >= PATIENCE:
            print(f"    early stop at epoch {epoch}  best val F1={best_val_f1:.4f}")
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, best_val_f1


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    set_seed()
    t0 = time.time()
    device = get_device()
    print(f"Device: {device}")

    # ── 1. Load all data ──────────────────────────────────────────────────────
    print("\n── Loading data ──")
    cat_train, cat_test, n_emb = load_categorical_indices()
    num_train_raw, num_test_raw = load_numeric_raw()
    tabular_train, tabular_test = load_tabular()
    y = load_labels()
    tabular_dim = tabular_train.shape[1]
    print(f"  Training rows:  {len(y):,}")
    print(f"  Tabular dim:    {tabular_dim}")
    print(f"  Total NN input: {sum(EMBED_COLS.values()) + len(NUMERIC_COLS)*8 + tabular_dim}")

    # ── 2. Fit numeric quantile bins on all training data ─────────────────────
    print("\n── Fitting numeric quantile bins ──")
    binner = NumericBins(n_bins=NUM_BINS)
    binner.fit(num_train_raw)
    num_bins_train = binner.transform(num_train_raw).astype(np.int64)
    num_bins_test = binner.transform(num_test_raw).astype(np.int64)
    print(f"  Bins fitted on {len(num_train_raw):,} rows, {len(NUMERIC_COLS)} numerics")

    # ── 3. Build cat matrices (N, 11) ─────────────────────────────────────────
    cat_matrix_train = np.column_stack([cat_train[c] for c in CAT_COLS])
    cat_matrix_test = np.column_stack([cat_test[c] for c in CAT_COLS])

    # ── 4. 5-Fold CV ──────────────────────────────────────────────────────────
    print("\n── 5-Fold CV ──")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros((len(y), 3), dtype=np.float32)
    test_proba_folds: list[np.ndarray] = []
    fold_scores: list[float] = []

    test_ds = NepalDataset(cat_matrix_test, num_bins_test, tabular_test)
    test_dl = _make_loader(test_ds, shuffle=False, bs=BATCH_SIZE * 2)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(cat_matrix_train, y), start=1):
        print(f"\n  ── Fold {fold}/{CV_FOLDS} ──")

        train_ds = NepalDataset(
            cat_matrix_train[tr_idx], num_bins_train[tr_idx],
            tabular_train[tr_idx], y[tr_idx],
        )
        val_ds = NepalDataset(
            cat_matrix_train[va_idx], num_bins_train[va_idx],
            tabular_train[va_idx], y[va_idx],
        )

        model = NepalResNetMLP(n_emb, tabular_dim).to(device)
        warm_start_embeddings(model, UW_CKPT, device)

        model, best_f1 = train_fold(model, train_ds, val_ds, device, fold)
        fold_scores.append(best_f1)
        print(f"  Fold {fold} best val F1: {best_f1:.4f}")

        # OOF probabilities
        val_dl = _make_loader(val_ds, shuffle=False, bs=BATCH_SIZE * 2)
        oof_proba[va_idx] = predict_proba(model, val_dl, device)

        # Test probabilities for this fold
        test_proba_folds.append(predict_proba(model, test_dl, device))

        print(f"  Fold {fold} OOF saved")

    mean_f1 = float(np.mean(fold_scores))
    std_f1 = float(np.std(fold_scores, ddof=1))
    print(f"\nCV micro F1: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Fold scores: {[f'{s:.4f}' for s in fold_scores]}")

    # ── 5. Average test predictions ───────────────────────────────────────────
    test_proba_avg = np.mean(test_proba_folds, axis=0)  # (N_test, 3)
    test_grades = test_proba_avg.argmax(axis=1) + 1      # back to 1-indexed

    # ── 6. Register run & save outputs ────────────────────────────────────────
    print("\n── Registering run_013 ──")
    rm = RunManager()
    run_id = rm.get_next_run_id()
    print(f"  Assigned run ID: {run_id}")

    rm.create_run(
        description="ResNet MLP end-to-end, run_012 unweighted embed warm-start, full 260k",
        model_type="NeuralNet",
        feature_set="entity_embeds+numeric_bins+tabular_51",
        params={
            "hidden": 256,
            "n_blocks": 3,
            "dropout": 0.25,
            "lr_body": LR_BODY,
            "lr_embed": LR_EMBED,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "num_bins": NUM_BINS,
            "bin_embed_dim": 8,
            "geo3_embed_dropout": 0.10,
            "warm_start": "run_012_unweighted",
        },
        run_id=run_id,
        objective="multiclass",
        n_features=sum(EMBED_COLS.values()) + len(NUMERIC_COLS) * 8 + tabular_dim,
        cv_folds=CV_FOLDS,
        cv_metric="micro_f1",
        notes="Warm-start entity embeds from models/embedding_unweighted/. "
              "Numeric bins (32 bins, 8-dim embed). geo3 embedding dropout 10%.",
    )
    rm.save_cv_scores(run_id, fold_scores, mean_f1, std_f1)

    # Save OOF / test probability arrays
    run_dir = rm.run_path(run_id)
    np.save(run_dir / "oof_proba.npy", oof_proba)
    np.save(run_dir / "test_proba.npy", test_proba_avg)
    print(f"  Saved oof_proba.npy ({oof_proba.shape})")
    print(f"  Saved test_proba.npy ({test_proba_avg.shape})")

    # OOF micro F1 (on 0-indexed labels)
    oof_f1 = float(f1_score(y, oof_proba.argmax(axis=1), average="micro"))
    print(f"  OOF (full) micro F1: {oof_f1:.4f}")

    # Save submission
    test_csv = pd.read_csv(ROOT / "data" / "driven_data" / "test_values.csv")
    building_ids = test_csv["building_id"].values
    sub_df = pd.DataFrame({
        "building_id": building_ids,
        "damage_grade": test_grades,
    })
    rm.save_submission(run_id, sub_df)
    print(f"  Submission saved → runs/{run_id}/submission.csv")

    # Grade distribution
    grades, counts = np.unique(test_grades, return_counts=True)
    for g, c in zip(grades, counts):
        print(f"  grade {g}: {c:,} ({c/len(test_grades)*100:.1f}%)")

    rm.print_all_runs()
    print(f"\nTotal wall-clock time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
