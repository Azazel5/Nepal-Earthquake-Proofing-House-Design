#!/usr/bin/env python3
"""Residual-signal screen for the grade-2 ↔ grade-3 boundary.

For candidate features, measures whether they predict true-G3-vs-G2 BEYOND what
the base model's own 2-vs-3 log-odds already encodes. If a feature is already
captured by the model, its residual lift is ~0; only ORTHOGONAL features lift CV
AUC. Includes a planted positive control to prove the test is sensitive.

Conclusion as of 2026-06-25 (base = run_026 OOF): every per-building / categorical
feature tried gives ΔAUC ≈ 0.0000 — the model is a sufficient statistic for them.
The 2↔3 errors are irreducible from the existing feature columns. See project
memory "Grade-2/3 boundary residual-signal screen".

Run:  env/bin/python scripts/boundary_residual_screen.py [--base runs/run_026/oof_proba.npy]
"""
from __future__ import annotations
import argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def _cv_auc(X: np.ndarray, z: np.ndarray) -> float:
    X = np.ascontiguousarray(X, dtype=np.float64)
    pr = cross_val_predict(LogisticRegression(max_iter=2000), X, z, cv=5,
                           method="predict_proba")[:, 1]
    return roc_auc_score(z, pr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="runs/run_026/oof_proba.npy")
    args = ap.parse_args()

    tv = pd.read_csv(ROOT / "data/driven_data/train_values.csv")
    lab = pd.read_csv(ROOT / "data/driven_data/train_labels.csv")
    d = tv.merge(lab, on="building_id")
    y = d["damage_grade"].to_numpy()
    p = np.load(ROOT / args.base)

    mask = np.isin(y, [2, 3]); z = (y[mask] == 3).astype(int); eps = 1e-6
    base_lo = (np.log(p[mask, 2] + eps) - np.log(p[mask, 1] + eps)).reshape(-1, 1)
    d = d[mask].reset_index(drop=True)
    auc0 = _cv_auc(base_lo, z)
    print(f"boundary pool n={mask.sum()} (G2={int((z==0).sum())}, G3={int((z==1).sum())})")
    print(f"BASE 2-vs-3 AUC = {auc0:.4f}\n")

    cf = d["count_floors_pre_eq"].astype(float).to_numpy()
    ht = d["height_percentage"].astype(float).to_numpy()
    ar = d["area_percentage"].astype(float).to_numpy()
    fam = d["count_families"].astype(float).to_numpy()
    ms = d["has_superstructure_mud_mortar_stone"].astype(float).to_numpy()
    rng = np.random.default_rng(0)

    feats: dict[str, np.ndarray] = {
        "POSCTRL_weak_target": (z * 0.3 + rng.normal(0, 1, len(z))).reshape(-1, 1),
        "slenderness=h/area": (ht / (ar + 1)).reshape(-1, 1),
        "avg_floor_height=h/floors": (ht / (cf + 0.5)).reshape(-1, 1),
        "area_per_family=area/fam": (ar / (fam + 1)).reshape(-1, 1),
        "mudstone_x_floors": (ms * cf).reshape(-1, 1),
        "raw_height": ht.reshape(-1, 1),
    }
    for c in ["land_surface_condition", "roof_type", "ground_floor_type",
              "other_floor_type", "position", "plan_configuration",
              "legal_ownership_status", "foundation_type"]:
        feats["cat:" + c] = pd.get_dummies(d[c]).to_numpy().astype(float)
    feats["secondary_use_block"] = d[[c for c in d.columns
                                      if c.startswith("has_secondary_use")]].to_numpy().astype(float)

    rows = []
    for name, F in feats.items():
        F = np.nan_to_num(np.asarray(F, dtype=np.float64))
        if F.shape[1] == 1:
            F = StandardScaler().fit_transform(F)
        rows.append((name, _cv_auc(np.hstack([base_lo, F]), z) - auc0))

    print(f"{'feature':<30}{'ΔAUC':>9}")
    print("-" * 39)
    for name, da in sorted(rows, key=lambda r: -r[1]):
        flag = "  <-- ORTHOGONAL SIGNAL" if da > 0.001 else ""
        print(f"{name:<30}{da:>+9.4f}{flag}")


if __name__ == "__main__":
    main()
