#!/usr/bin/env python3
"""
run_019: OOF-safe PCA on embedded features + LightGBM.

Fits StandardScaler + PCA inside each CV fold (no leakage), compresses the
191-dim run_012 feature matrix, then trains LGBM with trial-66 hyperparams.

k is chosen on fold 1 from {32, 48, 64, 80}. Also evaluates an embed-only
PCA variant (PCA on _emb_ columns, keep geo rates + binaries raw).

Run from project root:
    python src/run_019.py [--quick]
"""

from __future__ import annotations

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retrain import EARLY_STOPPING_ROUNDS, TRIAL_66_PARAMS
from run_manager import PROCESSED_DIR, RunManager
from run_trees_260k import (
    _oof_f1,
    build_blend_submission,
    pairwise_diagnostic,
    threeway_optimize,
)

print = partial(print, flush=True)

RANDOM_STATE = 42
CV_FOLDS = 5
K_GRID = [32, 48, 64, 80]
LGBM_CV_REF = 0.7588
NOISE = 0.0016
THRESHOLD = LGBM_CV_REF + NOISE

LGBM_OOF = ROOT / "runs" / "run_012" / "oof_proba.npy"
LGBM_TEST = ROOT / "runs" / "run_012" / "test_proba.npy"
XGB_OOF = ROOT / "runs" / "run_015" / "oof_proba.npy"
XGB_TEST = ROOT / "runs" / "run_015" / "test_proba.npy"
RUN018_OOF = ROOT / "runs" / "run_018" / "oof_proba.npy"
RUN018_TEST = ROOT / "runs" / "run_018" / "test_proba.npy"


def _embed_cols(cols: list[str]) -> list[str]:
    return [c for c in cols if "_emb_" in c]


def _non_embed_cols(cols: list[str]) -> list[str]:
    return [c for c in cols if "_emb_" not in c]


def _transform_pca(
    X_tr: pd.DataFrame,
    X_va: pd.DataFrame,
    X_te: pd.DataFrame,
    pca_cols: list[str],
    pass_cols: list[str],
    n_components: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE, svd_solver="randomized")

    def _scale_train(df: pd.DataFrame) -> np.ndarray:
        arr = scaler.fit_transform(df[pca_cols].to_numpy(dtype=np.float64))
        return np.clip(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)

    def _scale_apply(df: pd.DataFrame) -> np.ndarray:
        arr = scaler.transform(df[pca_cols].to_numpy(dtype=np.float64))
        return np.clip(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)

    z_tr = pca.fit_transform(_scale_train(X_tr))
    z_va = pca.transform(_scale_apply(X_va))
    z_te = pca.transform(_scale_apply(X_te))

    pca_names = [f"pca_{i}" for i in range(n_components)]
    if pass_cols:
        z_tr = np.hstack([z_tr, X_tr[pass_cols].to_numpy()])
        z_va = np.hstack([z_va, X_va[pass_cols].to_numpy()])
        z_te = np.hstack([z_te, X_te[pass_cols].to_numpy()])
        col_names = pca_names + pass_cols
    else:
        col_names = pca_names
    df_tr = pd.DataFrame(z_tr, columns=col_names)
    df_va = pd.DataFrame(z_va, columns=col_names)
    df_te = pd.DataFrame(z_te, columns=col_names)
    return df_tr, df_va, df_te


def _fit_lgbm_fold(X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray):
    model = LGBMClassifier(**TRIAL_66_PARAMS)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="multi_logloss",
        callbacks=[early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model


def _pick_k(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    pca_cols: list[str],
    pass_cols: list[str],
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
) -> int:
    best_k, best_f1 = K_GRID[0], -1.0
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    for k in K_GRID:
        if k >= len(pca_cols):
            continue
        df_tr, df_va, _ = _transform_pca(X_tr, X_va, X_test.iloc[:1], pca_cols, pass_cols, k)
        model = _fit_lgbm_fold(df_tr, y_tr, df_va, y_va)
        proba = model.predict_proba(df_va)
        f1 = f1_score(y_va, proba.argmax(axis=1) + 1, average="micro", labels=[1, 2, 3])
        print(f"    k={k}: fold-1 F1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_k = f1, k
    return best_k


def run_pca_cv(
    variant: str,
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    *,
    quick: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[float], int]:
    cols = list(X.columns)
    if variant == "full":
        pca_cols, pass_cols = cols, []
    else:
        pca_cols, pass_cols = _embed_cols(cols), _non_embed_cols(cols)

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(X, y))
    tr_idx, va_idx = splits[0]
    if quick:
        k = min(48, len(pca_cols) - 1)
        print(f"\n── {variant}: quick mode k={k} ──")
    else:
        print(f"\n── {variant}: selecting k on fold 1 ──")
        k = _pick_k(X, y, X_test, pca_cols, pass_cols, tr_idx, va_idx)
        print(f"  Selected k={k}")

    oof = np.zeros((len(y), 3), dtype=np.float32)
    test_folds: list[np.ndarray] = []
    scores: list[float] = []
    n_folds = 1 if quick else CV_FOLDS

    for fold, (tri, vai) in enumerate(splits[:n_folds], start=1):
        t0 = time.time()
        X_tr, X_va = X.iloc[tri], X.iloc[vai]
        df_tr, df_va, df_te = _transform_pca(X_tr, X_va, X_test, pca_cols, pass_cols, k)
        model = _fit_lgbm_fold(df_tr, y[tri], df_va, y[vai])
        oof[vai] = model.predict_proba(df_va).astype(np.float32)
        test_folds.append(model.predict_proba(df_te).astype(np.float32))
        f1 = f1_score(y[vai], oof[vai].argmax(axis=1) + 1, average="micro", labels=[1, 2, 3])
        scores.append(f1)
        print(f"  fold {fold}: F1={f1:.4f}  ({time.time() - t0:.0f}s)")

    test_avg = np.mean(test_folds, axis=0)
    return oof, test_avg, scores, k


def _blend_diagnostics(run_id: str, oof: np.ndarray, test: np.ndarray, y: np.ndarray) -> float:
    lgbm_oof = np.load(LGBM_OOF).astype(np.float64)
    lgbm_test = np.load(LGBM_TEST).astype(np.float64)
    xgb_oof = np.load(XGB_OOF).astype(np.float64)
    xgb_test = np.load(XGB_TEST).astype(np.float64)
    run018_oof = np.load(RUN018_OOF).astype(np.float64)
    run018_test = np.load(RUN018_TEST).astype(np.float64)

    solo = _oof_f1(oof, y)
    print(f"\n{run_id} solo OOF F1: {solo:.4f}")
    r = pairwise_diagnostic("lgbm_012", lgbm_oof, run_id, oof, y)
    f3, w_lg, w_xg, w_new = threeway_optimize(lgbm_oof, xgb_oof, oof, y)
    print(f"  3-way (LGBM+XGB+{run_id}): {f3:.4f}")
    best = max(r["best_score"], f3)
    if best > THRESHOLD:
        if f3 >= r["best_score"]:
            build_blend_submission(
                {"lgbm": lgbm_test, "xgb": xgb_test, run_id: test},
                {"lgbm": w_lg, "xgb": w_xg, run_id: w_new},
                space="proba",
                tag=f"{run_id}_3way",
            )
        else:
            build_blend_submission(
                {"lgbm": lgbm_test, run_id: test},
                {"lgbm": r["best_alpha"], run_id: 1.0 - r["best_alpha"]},
                space=r["best_space"],
                tag=f"{run_id}_2way",
            )
    else:
        print(f"  No blend clears threshold {THRESHOLD:.4f}")
    return solo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    t0 = time.time()

    X = pd.read_csv(PROCESSED_DIR / "X_train_run012.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test_run012.csv")
    y = pd.read_csv(PROCESSED_DIR / "y_train_full.csv")["damage_grade"].to_numpy()
    print(f"Features: {X.shape[1]}  train: {len(y):,}")

    results: dict[str, tuple] = {}
    variants = ("full",) if args.quick else ("full", "embed_only")
    for variant in variants:
        oof, test_p, scores, k = run_pca_cv(variant, X, y, X_test, quick=args.quick)
        mean_f1 = float(np.mean(scores))
        std_f1 = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
        results[variant] = (oof, test_p, scores, mean_f1, std_f1, k)

    if args.quick:
        print(f"\nQuick test done in {time.time() - t0:.1f}s")
        return

    best_variant = max(results, key=lambda v: results[v][3])
    oof, test_p, scores, mean_f1, std_f1, k = results[best_variant]
    print(f"\nBest variant: {best_variant}  CV={mean_f1:.4f} ± {std_f1:.4f}  k={k}")

    rm = RunManager()
    run_id = rm.get_next_run_id()
    rm.create_run(
        description=f"OOF PCA ({best_variant}, k={k}) on run_012 features + LGBM trial-66",
        model_type="LightGBM",
        feature_set=f"pca_{best_variant}_k{k}+lgbm",
        params={"pca_variant": best_variant, "n_components": k, "k_grid": K_GRID, **TRIAL_66_PARAMS},
        run_id=run_id,
        objective="multiclass",
        n_features=k + (0 if best_variant == "full" else len(_non_embed_cols(list(X.columns)))),
        cv_folds=CV_FOLDS,
        cv_metric="micro_f1",
        notes=f"Compared variants: { {v: round(results[v][3], 4) for v in results} }",
    )
    rm.save_cv_scores(run_id, scores, mean_f1, std_f1)
    run_dir = rm.run_path(run_id)
    np.save(run_dir / "oof_proba.npy", oof.astype(np.float32))
    np.save(run_dir / "test_proba.npy", test_p.astype(np.float32))
    test_csv = pd.read_csv(ROOT / "data" / "driven_data" / "test_values.csv")
    sub = pd.DataFrame({"building_id": test_csv["building_id"].values,
                        "damage_grade": test_p.argmax(axis=1) + 1})
    rm.save_submission(run_id, sub)
    _blend_diagnostics(run_id, oof.astype(np.float64), test_p.astype(np.float64), y)
    print(f"\nRegistered {run_id} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
