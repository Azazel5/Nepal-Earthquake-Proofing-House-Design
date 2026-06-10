#!/usr/bin/env python3
"""
run_021: Transfer learning — fine-tune run_020 AE encoder with classification head.

Loads pretrained TabularAE weights, freezes encoder for warmup epochs, then
unfreezes top encoder layer. 5-fold stratified CV, 3 seeds per fold (like run_018).

Run from project root:
    python src/run_021.py [--quick]
    python src/run_020.py   # must run first to create models/tabular_ae_run020.pt
"""

from __future__ import annotations

import argparse
import copy
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

from embed import get_device, set_seed
from run_020 import AE_CKPT, build_arrays
from run_020_model import LATENT_DIM, TabularAE, TabularAEClassifier
from run_manager import RunManager
from run_trees_260k import _oof_f1, build_blend_submission, pairwise_diagnostic, threeway_optimize

print = partial(print, flush=True)

RANDOM_STATE = 42
BATCH_SIZE = 1024
MAX_EPOCHS = 60
PATIENCE = 12
WARMUP_EPOCHS = 5
LR_HEAD = 1e-3
LR_ENCODER = 3e-4
LABEL_SMOOTHING = 0.05
LGBM_CV_REF = 0.7588
THRESHOLD = LGBM_CV_REF + 0.0016

LGBM_OOF = ROOT / "runs" / "run_012" / "oof_proba.npy"
LGBM_TEST = ROOT / "runs" / "run_012" / "test_proba.npy"
XGB_OOF = ROOT / "runs" / "run_015" / "oof_proba.npy"
XGB_TEST = ROOT / "runs" / "run_015" / "test_proba.npy"


class ClsDataset(Dataset):
    def __init__(self, cat: np.ndarray, numeric: np.ndarray, binary: np.ndarray, y: np.ndarray | None = None):
        self.cat = torch.from_numpy(cat)
        self.num = torch.from_numpy(numeric)
        self.bin = torch.from_numpy(binary)
        self.y = torch.from_numpy(y) if y is not None else None

    def __len__(self) -> int:
        return len(self.cat)

    def __getitem__(self, i: int):
        if self.y is not None:
            return self.cat[i], self.num[i], self.bin[i], self.y[i]
        return self.cat[i], self.num[i], self.bin[i]


def load_pretrained_classifier(data: dict, device: torch.device) -> TabularAEClassifier:
    if not AE_CKPT.exists():
        raise FileNotFoundError(f"Missing {AE_CKPT}. Run: python src/run_020.py")
    ae = TabularAE(data["n_cat"], data["n_binary"], latent_dim=LATENT_DIM, noise_std=0.0)
    ae.load_state_dict(torch.load(AE_CKPT, map_location="cpu", weights_only=True))
    model = TabularAEClassifier(ae, n_classes=3).to(device)
    return model


@torch.no_grad()
def predict_proba(model: TabularAEClassifier, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    for batch in loader:
        cat, num, binv = [b.to(device) for b in batch[:3]]
        logits = model(cat, num, binv)
        parts.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(parts, axis=0)


def train_fold(
    model: TabularAEClassifier,
    train_ds: ClsDataset,
    val_ds: ClsDataset,
    device: torch.device,
    max_epochs: int,
    patience: int,
    seed: int,
) -> tuple[TabularAEClassifier, float]:
    set_seed(seed)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    model.freeze_encoder()
    opt = torch.optim.AdamW(model.head.parameters(), lr=LR_HEAD, weight_decay=1e-4)
    best_f1, best_state, no_imp = -1.0, None, 0

    for epoch in range(1, max_epochs + 1):
        if epoch == WARMUP_EPOCHS + 1:
            model.unfreeze_encoder_top()
            opt = torch.optim.AdamW([
                {"params": model.head.parameters(), "lr": LR_HEAD},
                {"params": [p for p in model.ae.parameters() if p.requires_grad], "lr": LR_ENCODER},
            ], weight_decay=1e-4)
            print(f"      unfreezing encoder top at epoch {epoch}")

        model.train()
        for cat, num, binv, y in train_dl:
            cat, num, binv, y = cat.to(device), num.to(device), binv.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(cat, num, binv), y)
            loss.backward()
            opt.step()

        model.eval()
        val_parts, val_y = [], []
        with torch.no_grad():
            for cat, num, binv, y in val_dl:
                cat, num, binv = cat.to(device), num.to(device), binv.to(device)
                val_parts.append(model(cat, num, binv).cpu())
                val_y.append(y)
        logits = torch.cat(val_parts)
        y_true = torch.cat(val_y).numpy()
        val_f1 = f1_score(y_true + 1, logits.argmax(dim=1).numpy() + 1, average="micro", labels=[1, 2, 3])

        if val_f1 > best_f1:
            best_f1, best_state, no_imp = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            no_imp += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"      epoch {epoch}: val_F1={val_f1:.4f}  best={best_f1:.4f}")
        if no_imp >= patience:
            print(f"      early stop epoch {epoch}  best={best_f1:.4f}")
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, best_f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=str, default="42,142,242")
    args = parser.parse_args()
    seeds = [42] if args.quick else [int(s) for s in args.seeds.split(",")]
    max_epochs = 3 if args.quick else MAX_EPOCHS
    patience = 99 if args.quick else PATIENCE
    n_folds = 1 if args.quick else 5

    set_seed(RANDOM_STATE)
    device = get_device()
    t0 = time.time()
    data = build_arrays()
    y0 = (data["y"] - 1).astype(np.int64)
    n_test = len(data["cat_test"])

    test_ds = ClsDataset(data["cat_test"], data["numeric_test"], data["binary_test"])
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros((len(y0), 3), dtype=np.float64)
    test_accum = np.zeros((n_test, 3), dtype=np.float64)
    fold_scores: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(data["cat_train"], y0), start=1):
        if fold > n_folds:
            break
        print(f"\n── Fold {fold}/5 ──")
        train_ds = ClsDataset(
            data["cat_train"][tr_idx], data["numeric_train"][tr_idx],
            data["binary_train"][tr_idx], y0[tr_idx],
        )
        val_ds = ClsDataset(
            data["cat_train"][va_idx], data["numeric_train"][va_idx],
            data["binary_train"][va_idx], y0[va_idx],
        )

        seed_val, seed_test = [], []
        for seed in seeds:
            print(f"  seed {seed}")
            model = load_pretrained_classifier(data, device)
            model, best = train_fold(model, train_ds, val_ds, device, max_epochs, patience, seed * 1000 + fold)
            seed_val.append(predict_proba(model, DataLoader(val_ds, batch_size=BATCH_SIZE * 2), device))
            seed_test.append(predict_proba(model, test_dl, device))
            print(f"    best val F1: {best:.4f}")

        oof[va_idx] = np.mean(seed_val, axis=0)
        test_accum += np.mean(seed_test, axis=0)
        fold_f1 = f1_score(y0[va_idx], oof[va_idx].argmax(axis=1), average="micro")
        fold_scores.append(float(fold_f1))
        print(f"  fold {fold} OOF F1: {fold_f1:.4f}")

    if args.quick:
        print(f"Quick done in {time.time() - t0:.1f}s")
        return

    test_avg = test_accum / n_folds
    mean_f1 = float(np.mean(fold_scores))
    std_f1 = float(np.std(fold_scores, ddof=1))

    rm = RunManager()
    run_id = rm.get_next_run_id()
    rm.create_run(
        description="Fine-tuned run_020 AE encoder + classifier head, 3-seed averaged, 5-fold",
        model_type="NeuralNet",
        feature_set="ae_transfer_latent48",
        params={
            "latent_dim": LATENT_DIM, "warmup_epochs": WARMUP_EPOCHS,
            "lr_head": LR_HEAD, "lr_encoder": LR_ENCODER,
            "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "seeds": seeds,
            "pretrained": str(AE_CKPT),
        },
        run_id=run_id,
        cv_folds=5,
        cv_metric="micro_f1",
        notes="Transfer from run_020 TabularAE checkpoint",
    )
    rm.save_cv_scores(run_id, fold_scores, mean_f1, std_f1)
    run_dir = rm.run_path(run_id)
    np.save(run_dir / "oof_proba.npy", oof.astype(np.float32))
    np.save(run_dir / "test_proba.npy", test_avg.astype(np.float32))
    test_csv = pd.read_csv(ROOT / "data" / "driven_data" / "test_values.csv")
    rm.save_submission(run_id, pd.DataFrame({
        "building_id": test_csv["building_id"].values,
        "damage_grade": test_avg.argmax(axis=1) + 1,
    }))

    y = data["y"]
    lgbm_oof = np.load(LGBM_OOF).astype(np.float64)
    lgbm_test = np.load(LGBM_TEST).astype(np.float64)
    xgb_oof = np.load(XGB_OOF).astype(np.float64)
    xgb_test = np.load(XGB_TEST).astype(np.float64)
    print(f"\n{run_id} solo OOF: {_oof_f1(oof, y):.4f}")
    r = pairwise_diagnostic("lgbm_012", lgbm_oof, run_id, oof, y)
    f3, w_lg, w_xg, w_new = threeway_optimize(lgbm_oof, xgb_oof, oof, y)
    best = max(r["best_score"], f3)
    if best > THRESHOLD:
        if f3 >= r["best_score"]:
            build_blend_submission(
                {"lgbm": lgbm_test, "xgb": xgb_test, run_id: test_avg},
                {"lgbm": w_lg, "xgb": w_xg, run_id: w_new},
                space="proba", tag=f"{run_id}_3way",
            )

    print(f"\nRegistered {run_id}  CV={mean_f1:.4f}±{std_f1:.4f}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
