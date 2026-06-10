#!/usr/bin/env python3
"""
run_018: MLP rebuild — quality play on the decorrelated architecture.

run_013's MLP (0.7427 solo, 0.887 loss-corr / 9.3% disagreement vs run_012
LGBM) is the most decorrelated model found so far but too weak to blend.
This rebuild keeps the full feature signal (incl. Laplace-smoothed CV geo
rates as PLE inputs — the raw-feature-view experiment showed trees re-derive
these regardless, so removing them doesn't buy decorrelation) and focuses on
quality:

  - Hierarchical residual geo embeddings: emb(geo2)=proj(emb(geo1))+Delta2,
    emb(geo3)=emb(geo2)+Delta3. Delta tables zero-init, embedding dropout on
    Delta2/Delta3 (0.05 / 0.15).
  - Piecewise-linear (PLE) embeddings (48 bins) for the 5 numerics + 9 geo
    rates instead of raw floats / simple bin-index embeddings.
  - 8 low-card categoricals: small learned embeddings (dims 2-6).
  - 22 binary flags + age_unknown: raw.
  - 3 x ResBlock(448), GELU, LayerNorm, dropout 0.3.
  - Plain CrossEntropyLoss(label_smoothing=0.05) — NO class weights anywhere.
  - AdamW lr=1e-3, cosine decay to 0 over max_epochs, batch 1024, early stop
    on val accuracy (=micro F1).
  - Same 5-fold StratifiedKFold(seed=42) split as run_012. 3 seeds per fold,
    seed-averaged probabilities.

Run from project root:
    python src/run_018.py                  # full run (5 folds x 3 seeds)
    python src/run_018.py --quick           # smoke test (1 fold x 1 seed, 3 epochs)
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
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embed import NUMERIC_COLS, apply_age_sentinel, get_device, load_raw_data, set_seed
from run_018_model import NepalRun018MLP, PiecewiseLinearEncoder, PLE_BINS, PLE_EMBED_DIM
from run_manager import PROCESSED_DIR, RunManager
from run_trees_260k import _oof_f1, pairwise_diagnostic, build_blend_submission, threeway_optimize

print = partial(print, flush=True)

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
GEO_COLS = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]
LOW_CARD_COLS = [
    "foundation_type", "roof_type", "land_surface_condition", "ground_floor_type",
    "other_floor_type", "position", "plan_configuration", "legal_ownership_status",
]
CAT_DIMS = {
    "foundation_type": 4,
    "roof_type": 3,
    "land_surface_condition": 3,
    "ground_floor_type": 4,
    "other_floor_type": 3,
    "position": 3,
    "plan_configuration": 5,
    "legal_ownership_status": 3,
}
GEO_RATE_COLS = [
    "geo1_p_grade1", "geo1_p_grade2", "geo1_p_grade3",
    "geo2_p_grade1", "geo2_p_grade2", "geo2_p_grade3",
    "geo3_p_grade1", "geo3_p_grade2", "geo3_p_grade3",
]
PLE_COLS = NUMERIC_COLS + GEO_RATE_COLS  # 5 + 9 = 14

BATCH_SIZE = 1024
LABEL_SMOOTHING = 0.05
LR = 1e-3
WD_OTHER = 1e-4
WD_DELTA2 = 5e-4
WD_DELTA3 = 1e-3

LGBM_OOF  = ROOT / "runs" / "run_012" / "oof_proba.npy"
LGBM_TEST = ROOT / "runs" / "run_012" / "test_proba.npy"
XGB_OOF   = ROOT / "runs" / "run_015" / "oof_proba.npy"
XGB_TEST  = ROOT / "runs" / "run_015" / "test_proba.npy"
LGBM_CV_REF = 0.7588
NOISE = 0.0016
THRESHOLD = LGBM_CV_REF + NOISE
KILL_SOLO = 0.748
KILL_GAIN = 0.0016


# ── Data ──────────────────────────────────────────────────────────────────────

def build_data():
    train_full, test_raw = load_raw_data()
    train_full, test_raw = apply_age_sentinel(train_full, test_raw)
    train_full = train_full.reset_index(drop=True)
    test_raw = test_raw.reset_index(drop=True)

    all_cat_cols = GEO_COLS + LOW_CARD_COLS
    n_cat: dict[str, int] = {}
    idx_train: dict[str, np.ndarray] = {}
    idx_test: dict[str, np.ndarray] = {}

    for col in all_cat_cols:
        le = LabelEncoder()
        le.fit(train_full[col].astype(str))
        tr = le.transform(train_full[col].astype(str).values) + 1
        te_str = test_raw[col].astype(str).values
        te = np.zeros(len(test_raw), dtype=np.int64)
        known = np.isin(te_str, le.classes_)
        if known.any():
            te[known] = le.transform(te_str[known]) + 1
        idx_train[col] = tr.astype(np.int64)
        idx_test[col] = te.astype(np.int64)
        n_cat[col] = len(le.classes_) + 1

    geo_idx_train = np.column_stack([idx_train[c] for c in GEO_COLS])
    geo_idx_test = np.column_stack([idx_test[c] for c in GEO_COLS])
    cat_idx_train = np.column_stack([idx_train[c] for c in LOW_CARD_COLS])
    cat_idx_test = np.column_stack([idx_test[c] for c in LOW_CARD_COLS])

    # Binary flags + geo rates from X_train_run012 / X_test_run012 (same row order)
    bin_cols = [c for c in pd.read_csv(PROCESSED_DIR / "X_train_run012.csv", nrows=0).columns
                 if c.startswith("has_superstructure_") or c.startswith("has_secondary_use")]
    bin_cols = bin_cols + ["age_unknown"]

    usecols = GEO_RATE_COLS + bin_cols
    x012_train = pd.read_csv(PROCESSED_DIR / "X_train_run012.csv", usecols=usecols)
    x012_test = pd.read_csv(PROCESSED_DIR / "X_test_run012.csv", usecols=usecols)

    ple_raw_train = np.column_stack(
        [train_full[c].values.astype(np.float64) for c in NUMERIC_COLS]
        + [x012_train[c].values.astype(np.float64) for c in GEO_RATE_COLS]
    )
    ple_raw_test = np.column_stack(
        [test_raw[c].values.astype(np.float64) for c in NUMERIC_COLS]
        + [x012_test[c].values.astype(np.float64) for c in GEO_RATE_COLS]
    )

    binary_train = x012_train[bin_cols].values.astype(np.float32)
    binary_test = x012_test[bin_cols].values.astype(np.float32)

    y = pd.read_csv(PROCESSED_DIR / "y_train_full.csv")["damage_grade"].to_numpy()
    y0 = (y - 1).astype(np.int64)

    return {
        "geo_idx_train": geo_idx_train, "geo_idx_test": geo_idx_test,
        "cat_idx_train": cat_idx_train, "cat_idx_test": cat_idx_test,
        "ple_raw_train": ple_raw_train, "ple_raw_test": ple_raw_test,
        "binary_train": binary_train, "binary_test": binary_test,
        "y0": y0, "n_cat": n_cat, "n_binary": binary_train.shape[1],
    }


class NepalDataset018(Dataset):
    def __init__(self, geo_idx, cat_idx, ple, binary, y=None):
        self.geo = torch.from_numpy(geo_idx)
        self.cat = torch.from_numpy(cat_idx)
        self.ple = torch.from_numpy(ple)
        self.bin = torch.from_numpy(binary)
        self.y = torch.from_numpy(y) if y is not None else None

    def __len__(self) -> int:
        return len(self.bin)

    def __getitem__(self, i: int):
        if self.y is not None:
            return self.geo[i], self.cat[i], self.ple[i], self.bin[i], self.y[i]
        return self.geo[i], self.cat[i], self.ple[i], self.bin[i]


# ── Training helpers ──────────────────────────────────────────────────────────

def predict_proba(model: NepalRun018MLP, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            geo, cat, ple, binv = [b.to(device) for b in batch[:4]]
            logits = model(geo, cat, ple, binv)
            parts.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(parts, axis=0)


def train_fold_seed(
    model: NepalRun018MLP,
    train_ds: NepalDataset018,
    val_ds: NepalDataset018,
    device: torch.device,
    max_epochs: int,
    patience: int,
    seed: int,
) -> tuple[NepalRun018MLP, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)

    delta2_params = list(model.geo_embed.delta2.parameters())
    delta3_params = list(model.geo_embed.delta3.parameters())
    delta_ids = {id(p) for p in delta2_params + delta3_params}
    other_params = [p for p in model.parameters() if id(p) not in delta_ids]

    optimizer = torch.optim.AdamW([
        {"params": other_params, "lr": LR, "weight_decay": WD_OTHER},
        {"params": delta2_params, "lr": LR, "weight_decay": WD_DELTA2},
        {"params": delta3_params, "lr": LR, "weight_decay": WD_DELTA3},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)  # plain CE, no class weights

    best_f1 = 0.0
    best_state = None
    no_improve = 0
    y_val = val_ds.y.numpy()

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_dl:
            geo, cat, ple, binv, y = [b.to(device) for b in batch]
            optimizer.zero_grad()
            loss = criterion(model(geo, cat, ple, binv), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        val_proba = predict_proba(model, val_dl, device)
        val_f1 = float(f1_score(y_val, val_proba.argmax(axis=1), average="micro"))

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"      epoch {epoch:3d}: loss={epoch_loss/n_batches:.4f}  "
                  f"val_F1={val_f1:.4f}  best={best_f1:.4f}")

        if no_improve >= patience:
            print(f"      early stop at epoch {epoch}  best val F1={best_f1:.4f}")
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, best_f1


def build_model(data: dict, device: torch.device) -> NepalRun018MLP:
    n_cat = data["n_cat"]
    return NepalRun018MLP(
        n_geo1=n_cat["geo_level_1_id"],
        n_geo2=n_cat["geo_level_2_id"],
        n_geo3=n_cat["geo_level_3_id"],
        n_cat={c: n_cat[c] for c in LOW_CARD_COLS},
        cat_dims=CAT_DIMS,
        n_ple_features=len(PLE_COLS),
        n_binary=data["n_binary"],
    ).to(device)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smoke test: 1 fold x 1 seed, 3 epochs")
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=str, default="42,142,242")
    args = parser.parse_args()

    if args.quick:
        args.folds = 1
        args.max_epochs = 3
        args.patience = 99
        seeds = [42]
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    set_seed(RANDOM_STATE)
    t0 = time.time()
    device = get_device()
    print(f"Device: {device}")

    print("\n── Loading data ──")
    data = build_data()
    print(f"  geo_idx_train: {data['geo_idx_train'].shape}  cat_idx_train: {data['cat_idx_train'].shape}")
    print(f"  ple_raw_train: {data['ple_raw_train'].shape}  binary_train: {data['binary_train'].shape}")
    print(f"  n_cat: {data['n_cat']}")

    print("\n── Fitting PLE quantile bins ──")
    ple_enc = PiecewiseLinearEncoder(n_bins=PLE_BINS)
    ple_enc.fit(data["ple_raw_train"])
    ple_train = ple_enc.transform(data["ple_raw_train"]).astype(np.float32)
    ple_test = ple_enc.transform(data["ple_raw_test"]).astype(np.float32)
    print(f"  ple_train: {ple_train.shape}  ple_test: {ple_test.shape}")

    y0 = data["y0"]
    n = len(y0)
    n_test = data["geo_idx_test"].shape[0]

    test_ds = NepalDataset018(data["geo_idx_test"], data["cat_idx_test"], ple_test, data["binary_test"])
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros((n, 3), dtype=np.float64)
    test_proba_accum = np.zeros((n_test, 3), dtype=np.float64)
    fold_scores: list[float] = []

    n_folds_to_run = args.folds if args.quick else 5
    for fold, (tr_idx, va_idx) in enumerate(skf.split(data["geo_idx_train"], y0), start=1):
        if fold > n_folds_to_run:
            break
        print(f"\n── Fold {fold}/5 ──")
        t_fold = time.time()

        train_ds = NepalDataset018(
            data["geo_idx_train"][tr_idx], data["cat_idx_train"][tr_idx],
            ple_train[tr_idx], data["binary_train"][tr_idx], y0[tr_idx],
        )
        val_ds = NepalDataset018(
            data["geo_idx_train"][va_idx], data["cat_idx_train"][va_idx],
            ple_train[va_idx], data["binary_train"][va_idx], y0[va_idx],
        )
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)

        seed_val_probas = []
        seed_test_probas = []
        for seed in seeds:
            print(f"  ── seed {seed} ──")
            model = build_model(data, device)
            model, best_f1 = train_fold_seed(
                model, train_ds, val_ds, device, args.max_epochs, args.patience,
                seed=seed * 1000 + fold,
            )
            seed_val_probas.append(predict_proba(model, val_dl, device))
            seed_test_probas.append(predict_proba(model, test_dl, device))
            print(f"    seed {seed} best val F1: {best_f1:.4f}")

        oof_proba[va_idx] = np.mean(seed_val_probas, axis=0)
        test_proba_accum += np.mean(seed_test_probas, axis=0)

        fold_f1 = float(f1_score(y0[va_idx], oof_proba[va_idx].argmax(axis=1), average="micro"))
        fold_scores.append(fold_f1)
        print(f"  Fold {fold} seed-averaged OOF F1: {fold_f1:.4f}  ({time.time()-t_fold:.0f}s)")

    if args.quick:
        print(f"\nQuick smoke test done in {time.time()-t0:.1f}s. Not registering a run.")
        return

    test_proba_avg = test_proba_accum / n_folds_to_run
    mean_f1 = float(np.mean(fold_scores))
    std_f1 = float(np.std(fold_scores, ddof=1))
    print(f"\nrun_018 CV: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Fold scores: {[f'{s:.4f}' for s in fold_scores]}")

    # ── Register run ──────────────────────────────────────────────────────────
    rm = RunManager()
    run_id = rm.get_next_run_id()
    print(f"\n── Registering {run_id} ──")
    rm.create_run(
        description="MLP rebuild: hierarchical residual geo embeddings + PLE numeric/geo-rate "
                     "encodings, 3-seed averaged, 5-fold",
        model_type="NeuralNet",
        feature_set="hier_geo_emb+ple48+lowcard_emb+binary",
        params={
            "hidden": 448, "n_blocks": 3, "dropout": 0.3,
            "geo_dims": [16, 32, 32], "delta2_dropout": 0.05, "delta3_dropout": 0.15,
            "ple_bins": PLE_BINS, "ple_embed_dim": PLE_EMBED_DIM,
            "cat_dims": CAT_DIMS,
            "lr": LR, "wd_other": WD_OTHER, "wd_delta2": WD_DELTA2, "wd_delta3": WD_DELTA3,
            "label_smoothing": LABEL_SMOOTHING, "batch_size": BATCH_SIZE,
            "max_epochs": args.max_epochs, "patience": args.patience,
            "seeds": seeds,
        },
        run_id=run_id,
        objective="multiclass",
        n_features=None,
        cv_folds=5,
        cv_metric="micro_f1",
        notes="Trained from scratch (no warm-start). Geo rates kept as PLE inputs "
              "per raw-feature-view experiment finding (trees re-derive them anyway).",
    )
    rm.save_cv_scores(run_id, fold_scores, mean_f1, std_f1)

    run_dir = rm.run_path(run_id)
    np.save(run_dir / "oof_proba.npy", oof_proba.astype(np.float32))
    np.save(run_dir / "test_proba.npy", test_proba_avg.astype(np.float32))

    test_csv = pd.read_csv(ROOT / "data" / "driven_data" / "test_values.csv")
    sub_df = pd.DataFrame({"building_id": test_csv["building_id"].values,
                            "damage_grade": test_proba_avg.argmax(axis=1) + 1})
    rm.save_submission(run_id, sub_df)
    print(f"  Registered {run_id}")

    # ── Diagnostics vs run_012 LGBM ──────────────────────────────────────────
    lgbm_oof = np.load(LGBM_OOF).astype(np.float64)
    lgbm_test = np.load(LGBM_TEST).astype(np.float64)
    xgb_oof = np.load(XGB_OOF).astype(np.float64)
    xgb_test = np.load(XGB_TEST).astype(np.float64)

    print("\n" + "═" * 60)
    print("PAIRWISE BLEND DIAGNOSTICS (reference LGBM run_012 OOF = 0.7588)")
    print("═" * 60)
    r = pairwise_diagnostic("lgbm_012", lgbm_oof, "run_018", oof_proba, y0 + 1)

    print("\n" + "═" * 60)
    print("3-WAY BLEND: run_012 + run_015 XGB + run_018")
    print("═" * 60)
    f3, w_lg, w_xg, w_018 = threeway_optimize(lgbm_oof, xgb_oof, oof_proba, y0 + 1)
    print(f"  Best 3-way OOF F1: {f3:.4f}")
    print(f"  Weights — LGBM: {w_lg:.3f}  XGB: {w_xg:.3f}  run_018: {w_018:.3f}")

    # ── Summary & decision ───────────────────────────────────────────────────
    solo_f1 = _oof_f1(oof_proba, y0 + 1)
    best_2way = r["best_score"]
    best_3way = f3
    best_overall = max(best_2way, best_3way)

    print("\n" + "═" * 60)
    print("SUMMARY")
    print("═" * 60)
    print(f"  LGBM run_012 solo: {LGBM_CV_REF:.4f}  (reference)")
    print(f"  run_018 solo:      {solo_f1:.4f}")
    print(f"  2-way lgbm+run_018: {best_2way:.4f}  α={r['best_alpha']:.3f} ({r['best_space']})  "
          f"disagree={r['disagree_rate']:.3f}  corr={r['loss_corr']:.3f}  gain={best_2way-LGBM_CV_REF:+.4f}")
    print(f"  3-way blend:        {best_3way:.4f}  gain={best_3way-LGBM_CV_REF:+.4f}")
    print(f"  Best overall:       {best_overall:.4f}  gain={best_overall-LGBM_CV_REF:+.4f}  "
          f"{'>>> CLEARS THRESHOLD <<<' if best_overall > THRESHOLD else f'noise (threshold={THRESHOLD:.4f})'}")

    print(f"\n  KILL CRITERION CHECK:")
    print(f"    solo OOF >= {KILL_SOLO}: {'PASS' if solo_f1 >= KILL_SOLO else 'FAIL'} ({solo_f1:.4f})")
    print(f"    best blend gain >= +{KILL_GAIN}: {'PASS' if (best_overall-LGBM_CV_REF) >= KILL_GAIN else 'FAIL'} "
          f"({best_overall-LGBM_CV_REF:+.4f})")
    if solo_f1 < KILL_SOLO or (best_overall - LGBM_CV_REF) < KILL_GAIN:
        print("    => KILL: pivot to pseudo-labeling with current best ensemble as teacher.")
    else:
        print("    => CONTINUE: MLP rebuild meets revised kill criterion.")

    if best_overall > THRESHOLD:
        if best_3way >= best_2way:
            print("\n── Building 3-way blend submission ──")
            build_blend_submission(
                {"lgbm": lgbm_test, "xgb": xgb_test, "run_018": test_proba_avg},
                {"lgbm": w_lg, "xgb": w_xg, "run_018": w_018},
                space="proba",
                tag=f"run018_3way_lg{w_lg:.2f}_xg{w_xg:.2f}_018_{w_018:.2f}",
            )
        else:
            print(f"\n── Building 2-way blend submission ({r['best_space']}) ──")
            alpha = r["best_alpha"]
            build_blend_submission(
                {"lgbm": lgbm_test, "run_018": test_proba_avg},
                {"lgbm": alpha, "run_018": 1.0 - alpha},
                space=r["best_space"],
                tag=f"run018_2way_a{alpha:.3f}",
            )
    else:
        print("\n  No blend clears the noise threshold. NOT submitting.")

    print(f"\nTotal wall-clock time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
