#!/usr/bin/env python3
"""
run_020: Tabular autoencoder pretrain (train+test) → latent features → LGBM.

Stage 1: Unsupervised AE on all rows (no labels).
Stage 2: Extract 48-dim latent codes.
Stage 3: LGBM on concat(latent, geo_rates, binaries) with 5-fold OOF.

Run from project root:
    python src/run_020.py [--quick] [--skip-pretrain]
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
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embed import NUMERIC_COLS, apply_age_sentinel, get_device, load_raw_data, set_seed
from retrain import EARLY_STOPPING_ROUNDS, TRIAL_66_PARAMS
from run_020_model import LATENT_DIM, TabularAE
from run_manager import PROCESSED_DIR, RunManager
from run_trees_260k import _oof_f1, build_blend_submission, pairwise_diagnostic, threeway_optimize

print = partial(print, flush=True)

RANDOM_STATE = 42
CV_FOLDS = 5
BATCH_SIZE = 2048
AE_EPOCHS = 40
AE_PATIENCE = 8
AE_LR = 1e-3
LGBM_CV_REF = 0.7588
THRESHOLD = LGBM_CV_REF + 0.0016
AE_CKPT = ROOT / "models" / "tabular_ae_run020.pt"

GEO_COLS = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]
LOW_CARD_COLS = [
    "foundation_type", "roof_type", "land_surface_condition", "ground_floor_type",
    "other_floor_type", "position", "plan_configuration", "legal_ownership_status",
]
GEO_RATE_COLS = [
    "geo1_p_grade1", "geo1_p_grade2", "geo1_p_grade3",
    "geo2_p_grade1", "geo2_p_grade2", "geo2_p_grade3",
    "geo3_p_grade1", "geo3_p_grade2", "geo3_p_grade3",
]

LGBM_OOF = ROOT / "runs" / "run_012" / "oof_proba.npy"
LGBM_TEST = ROOT / "runs" / "run_012" / "test_proba.npy"
XGB_OOF = ROOT / "runs" / "run_015" / "oof_proba.npy"
XGB_TEST = ROOT / "runs" / "run_015" / "test_proba.npy"


def build_arrays() -> dict:
    train_full, test_raw = load_raw_data()
    train_full, test_raw = apply_age_sentinel(train_full, test_raw)
    train_full = train_full.reset_index(drop=True)
    test_raw = test_raw.reset_index(drop=True)

    all_cat = GEO_COLS + LOW_CARD_COLS
    n_cat: dict[str, int] = {}
    cat_train = np.zeros((len(train_full), len(all_cat)), dtype=np.int64)
    cat_test = np.zeros((len(test_raw), len(all_cat)), dtype=np.int64)
    for j, col in enumerate(all_cat):
        le = LabelEncoder()
        le.fit(train_full[col].astype(str))
        cat_train[:, j] = le.transform(train_full[col].astype(str).values) + 1
        te_str = test_raw[col].astype(str).values
        te = np.zeros(len(test_raw), dtype=np.int64)
        known = np.isin(te_str, le.classes_)
        if known.any():
            te[known] = le.transform(te_str[known]) + 1
        cat_test[:, j] = te
        n_cat[col] = len(le.classes_) + 1

    bin_cols = [c for c in pd.read_csv(PROCESSED_DIR / "X_train_run012.csv", nrows=0).columns
                if c.startswith("has_superstructure_") or c.startswith("has_secondary_use")]
    bin_cols = bin_cols + ["age_unknown"]
    x012_tr = pd.read_csv(PROCESSED_DIR / "X_train_run012.csv", usecols=GEO_RATE_COLS + bin_cols)
    x012_te = pd.read_csv(PROCESSED_DIR / "X_test_run012.csv", usecols=GEO_RATE_COLS + bin_cols)

    num_tr = train_full[NUMERIC_COLS].values.astype(np.float64)
    num_te = test_raw[NUMERIC_COLS].values.astype(np.float64)
    age_unknown_tr = (train_full["age"] == 995).astype(np.float64).values.reshape(-1, 1)
    age_unknown_te = (test_raw["age"] == 995).astype(np.float64).values.reshape(-1, 1)
    numeric_tr = np.hstack([num_tr, age_unknown_tr])
    numeric_te = np.hstack([num_te, age_unknown_te])

    scaler = StandardScaler()
    numeric_tr = scaler.fit_transform(numeric_tr).astype(np.float32)
    numeric_te = scaler.transform(numeric_te).astype(np.float32)

    binary_tr = x012_tr[bin_cols].values.astype(np.float32)
    binary_te = x012_te[bin_cols].values.astype(np.float32)
    geo_rates_tr = x012_tr[GEO_RATE_COLS].values.astype(np.float32)
    geo_rates_te = x012_te[GEO_RATE_COLS].values.astype(np.float32)

    y = pd.read_csv(PROCESSED_DIR / "y_train_full.csv")["damage_grade"].to_numpy()

    cat_all = np.vstack([cat_train, cat_test])
    numeric_all = np.vstack([numeric_tr, numeric_te])
    binary_all = np.vstack([binary_tr, binary_te])

    return {
        "cat_train": cat_train, "cat_test": cat_test, "cat_all": cat_all,
        "numeric_train": numeric_tr, "numeric_test": numeric_te, "numeric_all": numeric_all,
        "binary_train": binary_tr, "binary_test": binary_te, "binary_all": binary_all,
        "geo_rates_tr": geo_rates_tr, "geo_rates_te": geo_rates_te,
        "y": y, "n_cat": n_cat, "n_binary": binary_tr.shape[1],
        "num_scaler": scaler,
    }


class AEDataset(Dataset):
    def __init__(self, cat: np.ndarray, numeric: np.ndarray, binary: np.ndarray):
        self.cat = torch.from_numpy(cat)
        self.num = torch.from_numpy(numeric)
        self.bin = torch.from_numpy(binary)

    def __len__(self) -> int:
        return len(self.cat)

    def __getitem__(self, i: int):
        return self.cat[i], self.num[i], self.bin[i]


def pretrain_ae(data: dict, device: torch.device, epochs: int, patience: int) -> TabularAE:
    AE_CKPT.parent.mkdir(parents=True, exist_ok=True)
    model = TabularAE(data["n_cat"], data["n_binary"], latent_dim=LATENT_DIM, noise_std=0.05).to(device)
    if AE_CKPT.exists():
        print(f"  Loading AE checkpoint {AE_CKPT}")
        state = torch.load(AE_CKPT, map_location=device, weights_only=True)
        model.load_state_dict(state)
        return model

    ds = AEDataset(data["cat_all"], data["numeric_all"], data["binary_all"])
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=AE_LR, weight_decay=1e-4)
    best_loss, best_state, no_imp = float("inf"), None, 0

    print(f"  Pretraining AE on {len(ds):,} rows ({epochs} epochs max)...")
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for cat, num, binv in dl:
            cat, num, binv = cat.to(device), num.to(device), binv.to(device)
            opt.zero_grad()
            _, recon = model(cat, num, binv)
            loss = model.reconstruction_loss(cat, num, binv, recon)
            loss.backward()
            opt.step()
            total += loss.item() * len(cat)
            n += len(cat)
        avg = total / n
        if avg < best_loss:
            best_loss, best_state, no_imp = avg, copy.deepcopy(model.state_dict()), 0
        else:
            no_imp += 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"    epoch {epoch}: recon_loss={avg:.5f}  best={best_loss:.5f}")
        if no_imp >= patience:
            print(f"    early stop at epoch {epoch}")
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), AE_CKPT)
    return model


@torch.no_grad()
def extract_latent(model: TabularAE, cat: np.ndarray, numeric: np.ndarray, binary: np.ndarray,
                   device: torch.device, batch: int = 4096) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    n = len(cat)
    for i in range(0, n, batch):
        c = torch.from_numpy(cat[i:i + batch]).to(device)
        num = torch.from_numpy(numeric[i:i + batch]).to(device)
        binv = torch.from_numpy(binary[i:i + batch]).to(device)
        z = model.encode(c, num, binv).cpu().numpy()
        parts.append(z)
    return np.concatenate(parts, axis=0).astype(np.float32)


def lgbm_on_latent(
    z_tr: np.ndarray,
    geo_rates: np.ndarray,
    binary: np.ndarray,
    y: np.ndarray,
    z_te: np.ndarray,
    geo_rates_te: np.ndarray,
    binary_te: np.ndarray,
    *,
    quick: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    X = np.hstack([z_tr, geo_rates, binary])
    X_test = np.hstack([z_te, geo_rates_te, binary_te])
    feat_names = [f"z{i}" for i in range(z_tr.shape[1])] + \
        [f"g{i}" for i in range(geo_rates.shape[1])] + \
        [f"b{i}" for i in range(binary.shape[1])]
    X_df = pd.DataFrame(X, columns=feat_names)
    X_test_df = pd.DataFrame(X_test, columns=feat_names)

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros((len(y), 3), dtype=np.float32)
    test_folds: list[np.ndarray] = []
    scores: list[float] = []
    n_folds = 1 if quick else CV_FOLDS

    for fold, (tr, va) in enumerate(skf.split(X_df, y), start=1):
        if fold > n_folds:
            break
        model = LGBMClassifier(**{**TRIAL_66_PARAMS, "n_jobs": 1})
        model.fit(
            X_df.iloc[tr], y[tr],
            eval_set=[(X_df.iloc[va], y[va])],
            eval_metric="multi_logloss",
            callbacks=[early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        oof[va] = model.predict_proba(X_df.iloc[va]).astype(np.float32)
        test_folds.append(model.predict_proba(X_test_df).astype(np.float32))
        f1 = f1_score(y[va], oof[va].argmax(axis=1) + 1, average="micro", labels=[1, 2, 3])
        scores.append(f1)
        print(f"  LGBM fold {fold}: F1={f1:.4f}")

    return oof, np.mean(test_folds, axis=0), scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--force-pretrain", action="store_true", help="Ignore saved AE checkpoint")
    args = parser.parse_args()
    if args.quick:
        args.skip_pretrain = False
    if args.force_pretrain and AE_CKPT.exists():
        AE_CKPT.unlink()
    t0 = time.time()
    set_seed(RANDOM_STATE)
    device = get_device()

    data = build_arrays()
    epochs = 3 if args.quick else AE_EPOCHS
    patience = 99 if args.quick else AE_PATIENCE

    print("\n── Stage 1: AE pretrain ──")
    if args.skip_pretrain and AE_CKPT.exists():
        model = TabularAE(data["n_cat"], data["n_binary"], latent_dim=LATENT_DIM).to(device)
        model.load_state_dict(torch.load(AE_CKPT, map_location=device, weights_only=True))
    else:
        model = pretrain_ae(data, device, epochs, patience)

    print("\n── Stage 2: Extract latents ──")
    z_tr = extract_latent(model, data["cat_train"], data["numeric_train"], data["binary_train"], device)
    z_te = extract_latent(model, data["cat_test"], data["numeric_test"], data["binary_test"], device)
    print(f"  latent train: {z_tr.shape}  test: {z_te.shape}")

    print("\n── Stage 3: LGBM on latent + geo_rates + binary ──")
    oof, test_p, scores = lgbm_on_latent(
        z_tr, data["geo_rates_tr"], data["binary_train"], data["y"],
        z_te, data["geo_rates_te"], data["binary_test"],
        quick=args.quick,
    )

    if args.quick:
        print(f"Quick done in {time.time() - t0:.1f}s")
        return

    mean_f1 = float(np.mean(scores))
    std_f1 = float(np.std(scores, ddof=1))
    print(f"\nrun_020 CV: {mean_f1:.4f} ± {std_f1:.4f}")

    rm = RunManager()
    run_id = rm.get_next_run_id()
    rm.create_run(
        description="Tabular AE pretrain (train+test) → 48-dim latent + geo_rates + binary → LGBM",
        model_type="LightGBM",
        feature_set=f"ae_latent{LATENT_DIM}+geo_rates+binary",
        params={"latent_dim": LATENT_DIM, "ae_epochs": AE_EPOCHS, "ae_ckpt": str(AE_CKPT), **TRIAL_66_PARAMS},
        run_id=run_id,
        objective="multiclass",
        n_features=LATENT_DIM + len(GEO_RATE_COLS) + data["n_binary"],
        cv_folds=CV_FOLDS,
        cv_metric="micro_f1",
    )
    rm.save_cv_scores(run_id, scores, mean_f1, std_f1)
    run_dir = rm.run_path(run_id)
    np.save(run_dir / "oof_proba.npy", oof)
    np.save(run_dir / "test_proba.npy", test_p)
    test_csv = pd.read_csv(ROOT / "data" / "driven_data" / "test_values.csv")
    rm.save_submission(run_id, pd.DataFrame({
        "building_id": test_csv["building_id"].values,
        "damage_grade": test_p.argmax(axis=1) + 1,
    }))

    lgbm_oof = np.load(LGBM_OOF).astype(np.float64)
    lgbm_test = np.load(LGBM_TEST).astype(np.float64)
    xgb_oof = np.load(XGB_OOF).astype(np.float64)
    xgb_test = np.load(XGB_TEST).astype(np.float64)
    pairwise_diagnostic("lgbm_012", lgbm_oof, run_id, oof.astype(np.float64), data["y"])
    f3, w_lg, w_xg, w_new = threeway_optimize(lgbm_oof, xgb_oof, oof.astype(np.float64), data["y"])
    if max(_oof_f1(oof, data["y"]), f3) > THRESHOLD and f3 >= _oof_f1(oof, data["y"]):
        build_blend_submission(
            {"lgbm": lgbm_test, "xgb": xgb_test, run_id: test_p},
            {"lgbm": w_lg, "xgb": w_xg, run_id: w_new},
            space="proba", tag=f"{run_id}_3way",
        )

    print(f"\nRegistered {run_id} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
