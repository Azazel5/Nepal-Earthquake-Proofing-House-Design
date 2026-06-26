#!/usr/bin/env python3
"""Definitive memorization-vs-physics test for the run_047 GNN.

Retrains run_047's architecture (DropEdge + neighbor masking) under GroupKFold by
geo_level_3: every held-out validation cell has ZERO training members, so any
"predict the cell's memorized label via its train cell-mates" benefit is IMPOSSIBLE.

Read it as:
  - solo geo-grouped F1 ≈ run_047 random-KFold F1 (0.7644)  -> the GNN learned
    transferable neighborhood physics; the OOF gain is real.
  - solo geo-grouped F1 craters toward run_026 (0.7546) or below  -> the gain was
    geo3-cell memorization that cannot survive unseen cells (== run_039's failure).

Run:  env/bin/python scripts/geo_grouped_gnn_validation.py [--epochs 200]
"""
from __future__ import annotations
import argparse, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import QuantileTransformer
warnings.filterwarnings("ignore"); np.seterr(all="ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_manager import PROCESSED_DIR
from run_047 import train_transductive, within_cell_variance, DROP_P, MASK_Q, LR

RANDOM_STATE = 42; CV_FOLDS = 5


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args(); t0 = time.time()

    X_train = pd.read_csv(PROCESSED_DIR / "X_train_run012.csv").fillna(0)
    y = pd.read_csv(PROCESSED_DIR / "y_train_full.csv")["damage_grade"].to_numpy()
    X_test = pd.read_csv(PROCESSED_DIR / "X_test_run012.csv").fillna(0)
    df_tr = pd.read_csv(ROOT / "data/driven_data/train_values.csv")
    df_te = pd.read_csv(ROOT / "data/driven_data/test_values.csv")
    geo3_raw = np.concatenate([df_tr["geo_level_3_id"].values, df_te["geo_level_3_id"].values])
    _, geo3_idx_full = np.unique(geo3_raw, return_inverse=True)
    num_geo3 = len(_)
    geo3_train = geo3_idx_full[: len(y)]

    X_full_s = QuantileTransformer(output_distribution="normal", random_state=RANDOM_STATE
                                   ).fit_transform(np.vstack([X_train.values, X_test.values])).astype(np.float32)

    print(f"── GEO-GROUPED (GroupKFold by geo3) GNN validation  "
          f"(drop_p={DROP_P}, mask_q={MASK_Q}, epochs={args.epochs}) ──")
    gkf = GroupKFold(n_splits=CV_FOLDS)
    oof = np.zeros((len(y), 3), dtype=np.float32); scores = []
    for fold, (tri, vai) in enumerate(gkf.split(X_train, y, groups=geo3_train), start=1):
        # confirm zero cell overlap
        overlap = len(set(geo3_train[tri]) & set(geo3_train[vai]))
        vp, _tp = train_transductive(X_full_s, y, geo3_idx_full, num_geo3, fold, tri, vai,
                                     args.epochs, LR, DROP_P, MASK_Q)
        oof[vai] = vp
        f = f1_score(y[vai], vp.argmax(1) + 1, average="micro", labels=[1, 2, 3])
        scores.append(f)
        print(f"  fold {fold}: F1={f:.4f}  (train/val geo3 overlap={overlap})")

    geo_f1 = float(np.mean(scores))
    p26 = np.load(ROOT / "runs/run_026/oof_proba.npy").astype(np.float64)
    f26 = f1_score(y, p26.argmax(1) + 1, average="micro")
    print("\n" + "=" * 60)
    print(f"  run_047 GNN — RANDOM KFold solo F1 : 0.7644  (inflated reference)")
    print(f"  run_047 GNN — GEO-GROUPED solo F1  : {geo_f1:.4f} ± {np.std(scores,ddof=1):.4f}")
    print(f"  run_026 blend (random KFold)       : {f26:.4f}")
    drop = 0.7644 - geo_f1
    print(f"\n  Drop from random→geo-grouped: {drop:+.4f}")
    if geo_f1 >= 0.760:
        verdict = "PHYSICS — generalizes to unseen cells. Gain may be real; consider a public probe."
    elif geo_f1 >= f26:
        verdict = "MIXED — above run_026 but well below its random-KFold OOF; gain is partly memorization."
    else:
        verdict = "MEMORIZATION — collapses below run_026 on unseen cells. OOF gain will NOT transfer (== run_039)."
    print(f"  VERDICT: {verdict}")

    # stratify the geo-grouped GNN accuracy by cell size (all val cells are unseen here)
    n = pd.Series(geo3_train).map(pd.Series(geo3_train).value_counts()).to_numpy()
    print(f"\n  geo-grouped GNN accuracy vs run_026 by geo3 cell size:")
    for lo, hi, lab in [(0,20,'<20'),(20,100,'20-100'),(100,500,'100-500'),(500,10**9,'500+')]:
        m = (n >= lo) & (n < hi)
        if m.sum() == 0: continue
        ag = (oof.argmax(1)[m] + 1 == y[m]).mean()
        a26 = (p26.argmax(1)[m] + 1 == y[m]).mean()
        print(f"    {lab:<8} n={m.sum():>7}  GNN_geo={ag:.4f}  run026={a26:.4f}  Δ={ag-a26:+.4f}")

    np.save(ROOT / "outputs" / "run_047_geogrouped_oof.npy", oof)
    print(f"\n  Saved geo-grouped OOF to outputs/run_047_geogrouped_oof.npy")
    print(f"  Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
