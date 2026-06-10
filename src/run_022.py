#!/usr/bin/env python3
"""
run_022: Hybrid PCA (numerics) + categorical AE → LGBM.

  Stage 1 — Cat AE (unsupervised, train ∪ test):
    11 categoricals → embeddings → latent z (32-d)
    Optional Shoumik-style geo1/geo2 rollup auxiliary loss from z.

  Stage 2 — Per-fold OOF features for LGBM:
    PCA on 6 numeric cols (5 numerics + age_unknown), k ∈ {3,4,5}, fit on train-fold only
    + fixed cat latent z + raw geo_rates (9) + binaries (23)

Run from project root:
    python src/run_022.py [--quick] [--geo-rollup] [--force-pretrain]
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
from lightgbm import LGBMClassifier, early_stopping
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embed import NUMERIC_COLS, apply_age_sentinel, get_device, load_raw_data, set_seed
from retrain import EARLY_STOPPING_ROUNDS, TRIAL_66_PARAMS
from run_022_model import CAT_LATENT_DIM, CategoricalAE
from run_manager import PROCESSED_DIR, RunManager
from run_trees_260k import _oof_f1, build_blend_submission, pairwise_diagnostic, threeway_optimize

print = partial(print, flush=True)

RANDOM_STATE = 42
CV_FOLDS = 5
BATCH_SIZE = 2048
AE_EPOCHS = 40
AE_PATIENCE = 8
AE_LR = 1e-3
ROLLUP_WEIGHT = 0.2
K_GRID = [3, 4, 5]
LGBM_CV_REF = 0.7588
THRESHOLD = LGBM_CV_REF + 0.0016
CAT_AE_CKPT = ROOT / "models" / "cat_ae_run022.pt"

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
RUN019_OOF = ROOT / "runs" / "run_019" / "oof_proba.npy"


def build_data() -> dict:
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

    bin_cols = [
        c for c in pd.read_csv(PROCESSED_DIR / "X_train_run012.csv", nrows=0).columns
        if c.startswith("has_superstructure_") or c.startswith("has_secondary_use")
    ] + ["age_unknown"]
    x012_tr = pd.read_csv(PROCESSED_DIR / "X_train_run012.csv", usecols=GEO_RATE_COLS + bin_cols)
    x012_te = pd.read_csv(PROCESSED_DIR / "X_test_run012.csv", usecols=GEO_RATE_COLS + bin_cols)

    num_tr = train_full[NUMERIC_COLS].values.astype(np.float64)
    num_te = test_raw[NUMERIC_COLS].values.astype(np.float64)
    age_tr = (train_full["age"] == 995).astype(np.float64).values.reshape(-1, 1)
    age_te = (test_raw["age"] == 995).astype(np.float64).values.reshape(-1, 1)
    numeric_tr = np.hstack([num_tr, age_tr])
    numeric_te = np.hstack([num_te, age_te])

    y = pd.read_csv(PROCESSED_DIR / "y_train_full.csv")["damage_grade"].to_numpy()

    return {
        "cat_train": cat_train,
        "cat_test": cat_test,
        "cat_all": np.vstack([cat_train, cat_test]),
        "numeric_tr": numeric_tr,
        "numeric_te": numeric_te,
        "geo_rates_tr": x012_tr[GEO_RATE_COLS].values.astype(np.float32),
        "geo_rates_te": x012_te[GEO_RATE_COLS].values.astype(np.float32),
        "binary_tr": x012_tr[bin_cols].values.astype(np.float32),
        "binary_te": x012_te[bin_cols].values.astype(np.float32),
        "y": y,
        "n_cat": n_cat,
        "n_binary": len(bin_cols),
    }


class CatDataset(Dataset):
    def __init__(self, cat: np.ndarray):
        self.cat = torch.from_numpy(cat)

    def __len__(self) -> int:
        return len(self.cat)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.cat[i]


def pretrain_cat_ae(
    cat_all: np.ndarray,
    n_cat: dict[str, int],
    device: torch.device,
    epochs: int,
    patience: int,
    geo_rollup: bool,
) -> CategoricalAE:
    CAT_AE_CKPT.parent.mkdir(parents=True, exist_ok=True)
    model = CategoricalAE(n_cat, latent_dim=CAT_LATENT_DIM).to(device)
    if CAT_AE_CKPT.exists():
        print(f"  Loading cat AE from {CAT_AE_CKPT}")
        model.load_state_dict(torch.load(CAT_AE_CKPT, map_location=device, weights_only=True))
        return model

    dl = DataLoader(CatDataset(cat_all), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=AE_LR, weight_decay=1e-4)
    best_loss, best_state, no_imp = float("inf"), None, 0
    print(f"  Pretraining cat AE on {len(cat_all):,} rows ({epochs} epochs max)...")

    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for cat in dl:
            cat = cat.to(device)
            opt.zero_grad()
            z, recon = model(cat)
            loss = model.reconstruction_loss(cat, recon)
            if geo_rollup:
                loss = loss + ROLLUP_WEIGHT * model.rollup_loss(cat, z)
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
            print(f"    epoch {epoch}: loss={avg:.5f}  best={best_loss:.5f}")
        if no_imp >= patience:
            print(f"    early stop at epoch {epoch}")
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), CAT_AE_CKPT)
    return model


@torch.no_grad()
def extract_cat_latent(model: CategoricalAE, cat: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    for i in range(0, len(cat), 4096):
        c = torch.from_numpy(cat[i:i + 4096]).to(device)
        parts.append(model.encode(c).cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float32)


def _pca_numeric(
    num_tr: np.ndarray,
    num_va: np.ndarray,
    num_te: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    pca = PCA(n_components=k, random_state=RANDOM_STATE)

    def _clean(arr: np.ndarray) -> np.ndarray:
        return np.clip(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)

    tr_s = _clean(scaler.fit_transform(num_tr))
    va_s = _clean(scaler.transform(num_va))
    te_s = _clean(scaler.transform(num_te))
    return (
        pca.fit_transform(tr_s).astype(np.float32),
        pca.transform(va_s).astype(np.float32),
        pca.transform(te_s).astype(np.float32),
    )


def _build_fold_matrix(
    pca_tr: np.ndarray,
    z_tr: np.ndarray,
    geo_tr: np.ndarray,
    bin_tr: np.ndarray,
) -> np.ndarray:
    return np.hstack([pca_tr, z_tr, geo_tr, bin_tr])


def _pick_k(
    data: dict,
    z_tr: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    k: int,
) -> float:
    p_tr, p_va, _ = _pca_numeric(
        data["numeric_tr"][tr_idx], data["numeric_tr"][va_idx], data["numeric_te"][:1], k,
    )
    X_tr = _build_fold_matrix(p_tr, z_tr[tr_idx], data["geo_rates_tr"][tr_idx], data["binary_tr"][tr_idx])
    X_va = _build_fold_matrix(p_va, z_tr[va_idx], data["geo_rates_tr"][va_idx], data["binary_tr"][va_idx])
    model = LGBMClassifier(**{**TRIAL_66_PARAMS, "n_jobs": 1})
    model.fit(
        X_tr, data["y"][tr_idx],
        eval_set=[(X_va, data["y"][va_idx])],
        eval_metric="multi_logloss",
        callbacks=[early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    proba = model.predict_proba(X_va)
    return float(f1_score(data["y"][va_idx], proba.argmax(axis=1) + 1, average="micro", labels=[1, 2, 3]))


def run_hybrid_cv(
    data: dict,
    z_tr: np.ndarray,
    z_te: np.ndarray,
    *,
    quick: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[float], int]:
    y = data["y"]
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(z_tr, y))
    tr_idx, va_idx = splits[0]

    if quick:
        k = 4
        print(f"\n── quick mode: k={k} ──")
    else:
        print("\n── selecting numeric PCA k on fold 1 ──")
        best_k, best_f1 = K_GRID[0], -1.0
        for k in K_GRID:
            f1 = _pick_k(data, z_tr, tr_idx, va_idx, k)
            print(f"    k={k}: fold-1 F1={f1:.4f}")
            if f1 > best_f1:
                best_f1, best_k = f1, k
        k = best_k
        print(f"  Selected k={k}")

    oof = np.zeros((len(y), 3), dtype=np.float32)
    test_folds: list[np.ndarray] = []
    scores: list[float] = []
    n_folds = 1 if quick else CV_FOLDS

    for fold, (tri, vai) in enumerate(splits[:n_folds], start=1):
        t0 = time.time()
        p_tr, p_va, p_te = _pca_numeric(
            data["numeric_tr"][tri], data["numeric_tr"][vai], data["numeric_te"], k,
        )
        X_tr = _build_fold_matrix(p_tr, z_tr[tri], data["geo_rates_tr"][tri], data["binary_tr"][tri])
        X_va = _build_fold_matrix(p_va, z_tr[vai], data["geo_rates_tr"][vai], data["binary_tr"][vai])
        X_te = _build_fold_matrix(
            p_te, z_te, data["geo_rates_te"], data["binary_te"],
        )
        model = LGBMClassifier(**{**TRIAL_66_PARAMS, "n_jobs": 1})
        model.fit(
            X_tr, y[tri],
            eval_set=[(X_va, y[vai])],
            eval_metric="multi_logloss",
            callbacks=[early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        oof[vai] = model.predict_proba(X_va).astype(np.float32)
        test_folds.append(model.predict_proba(X_te).astype(np.float32))
        f1 = f1_score(y[vai], oof[vai].argmax(axis=1) + 1, average="micro", labels=[1, 2, 3])
        scores.append(f1)
        print(f"  fold {fold}: F1={f1:.4f}  ({time.time() - t0:.0f}s)")

    return oof, np.mean(test_folds, axis=0), scores, k


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--geo-rollup", action="store_true", help="Shoumik-style geo1/geo2 aux loss")
    parser.add_argument("--force-pretrain", action="store_true")
    args = parser.parse_args()
    if args.force_pretrain and CAT_AE_CKPT.exists():
        CAT_AE_CKPT.unlink()

    set_seed(RANDOM_STATE)
    device = get_device()
    t0 = time.time()
    data = build_data()

    print("\n── Stage 1: categorical AE pretrain ──")
    epochs = 3 if args.quick else AE_EPOCHS
    patience = 99 if args.quick else AE_PATIENCE
    cat_ae = pretrain_cat_ae(
        data["cat_all"], data["n_cat"], device, epochs, patience, args.geo_rollup,
    )

    print("\n── Stage 2: extract cat latents ──")
    z_tr = extract_cat_latent(cat_ae, data["cat_train"], device)
    z_te = extract_cat_latent(cat_ae, data["cat_test"], device)
    print(f"  z_train: {z_tr.shape}  z_test: {z_te.shape}")

    print("\n── Stage 3: OOF PCA(numeric) + z_cat + raw → LGBM ──")
    oof, test_p, scores, k = run_hybrid_cv(data, z_tr, z_te, quick=args.quick)

    if args.quick:
        print(f"\nQuick done in {time.time() - t0:.1f}s")
        return

    mean_f1 = float(np.mean(scores))
    std_f1 = float(np.std(scores, ddof=1))
    n_feat = k + CAT_LATENT_DIM + len(GEO_RATE_COLS) + data["n_binary"]
    print(f"\nrun_022 CV: {mean_f1:.4f} ± {std_f1:.4f}  (numeric_pca_k={k})")

    rm = RunManager()
    run_id = rm.get_next_run_id()
    rm.create_run(
        description=f"Hybrid PCA numerics k={k} + cat AE latent {CAT_LATENT_DIM} + geo_rates + binary → LGBM",
        model_type="LightGBM",
        feature_set=f"pca_numeric_k{k}+cat_ae{CAT_LATENT_DIM}+geo_rates+binary",
        params={
            "numeric_pca_k": k,
            "cat_latent_dim": CAT_LATENT_DIM,
            "geo_rollup": args.geo_rollup,
            "cat_ae_ckpt": str(CAT_AE_CKPT),
            **TRIAL_66_PARAMS,
        },
        run_id=run_id,
        objective="multiclass",
        n_features=n_feat,
        cv_folds=CV_FOLDS,
        cv_metric="micro_f1",
        notes=f"Builds on run_019 SOTA insight: split repr by feature type.",
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
    print(f"\n{run_id} solo OOF: {_oof_f1(oof, data['y']):.4f}")
    r = pairwise_diagnostic("lgbm_012", lgbm_oof, run_id, oof.astype(np.float64), data["y"])
    if RUN019_OOF.exists():
        pairwise_diagnostic("run_019", np.load(RUN019_OOF).astype(np.float64), run_id, oof.astype(np.float64), data["y"])
    f3, w_lg, w_xg, w_new = threeway_optimize(lgbm_oof, xgb_oof, oof.astype(np.float64), data["y"])
    best = max(r["best_score"], f3)
    if best > THRESHOLD:
        if f3 >= r["best_score"]:
            build_blend_submission(
                {"lgbm": lgbm_test, "xgb": xgb_test, run_id: test_p},
                {"lgbm": w_lg, "xgb": w_xg, run_id: w_new},
                space="proba", tag=f"{run_id}_3way",
            )
        else:
            build_blend_submission(
                {"lgbm": lgbm_test, run_id: test_p},
                {"lgbm": r["best_alpha"], run_id: 1.0 - r["best_alpha"]},
                space=r["best_space"], tag=f"{run_id}_2way",
            )
    else:
        print(f"  No blend clears threshold {THRESHOLD:.4f}")

    print(f"\nRegistered {run_id} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
