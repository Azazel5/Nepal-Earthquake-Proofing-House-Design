#!/usr/bin/env python3
"""
run_024: Shoumik #29 replication — geo DR autoencoder + geo3 rollup + XGBoost.

Pipeline (from Shoumik-Gandre/Richter-s-Predictor-Modeling-Earthquake-Damage):
  1. Train geo DR AE and geo3→geo1/2 rollup on train∪test (unsupervised).
  2. Build features: OHE categoricals, raw geo ids, DR latent (32d), rollup latent
     (16d), numerics + binaries (XGB notebook layout; CatBoostEncoder omitted).
  3. 5-fold stratified CV with Optuna-tuned XGB hyperparams.
  4. Full-train refit for submission (Shoumik style).

Geo training runs in a torch-only subprocess (run_024_geo.py) to avoid a
torch/xgboost OpenMP conflict on macOS.

Run from project root:
    python src/run_024.py [--quick] [--force-geo]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

# xgboost must load before any torch import in this process
import xgboost as xgb  # noqa: E402

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_manager import PROCESSED_DIR, RunManager
from run_024_features import ShoumikFeatureBuilder, geo_latents_ready, load_frames, load_geo_latents
from run_trees_260k import (
    _oof_f1,
    build_blend_submission,
    pairwise_diagnostic,
    threeway_optimize,
)

print = partial(print, flush=True)

RANDOM_STATE = 42
CV_FOLDS = 5
RUN_ID = "run_024"
LGBM_CV_REF = 0.7588
NOISE = 0.0016
THRESHOLD = LGBM_CV_REF + NOISE

LGBM_OOF = ROOT / "runs" / "run_012" / "oof_proba.npy"
LGBM_TEST = ROOT / "runs" / "run_012" / "test_proba.npy"
XGB_OOF = ROOT / "runs" / "run_015" / "oof_proba.npy"
XGB_TEST = ROOT / "runs" / "run_015" / "test_proba.npy"
RUN019_OOF = ROOT / "runs" / "run_019" / "oof_proba.npy"
RUN019_TEST = ROOT / "runs" / "run_019" / "test_proba.npy"

SHOUMIK_XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
    "learning_rate": 0.01306,
    "max_depth": 15,
    "n_estimators": 756,
    "gamma": 1.37,
    "min_child_weight": 7,
    "reg_alpha": 0.018,
    "reg_lambda": 0.059,
    "subsample": 0.808,
    "colsample_bytree": 0.528,
    "colsample_bylevel": 0.835,
    "colsample_bynode": 0.564,
}
EARLY_STOP = 50


def _micro_f1(y_true: np.ndarray, proba: np.ndarray) -> float:
    return f1_score(y_true, proba.argmax(axis=1) + 1, average="micro", labels=[1, 2, 3])


def _fit_xgb_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
) -> tuple[xgb.XGBClassifier, int]:
    model = xgb.XGBClassifier(**SHOUMIK_XGB_PARAMS, early_stopping_rounds=EARLY_STOP)
    model.fit(
        X_tr,
        y_tr - 1,
        eval_set=[(X_va, y_va - 1)],
        verbose=False,
    )
    best_iter = model.best_iteration if model.best_iteration is not None else SHOUMIK_XGB_PARAMS["n_estimators"] - 1
    return model, int(best_iter) + 1


def run_cv(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    geo_train: np.ndarray,
    geo_test: np.ndarray,
    dr_train: np.ndarray,
    dr_test: np.ndarray,
    ru_train: np.ndarray,
    ru_test: np.ndarray,
    *,
    quick: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[float], list[int]]:
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(train, y))
    n_folds = 1 if quick else CV_FOLDS

    oof = np.zeros((len(y), 3), dtype=np.float32)
    test_folds: list[np.ndarray] = []
    scores: list[float] = []
    best_iters: list[int] = []

    for fold, (tri, vai) in enumerate(splits[:n_folds], start=1):
        t0 = time.time()
        fb = ShoumikFeatureBuilder()
        fb.fit(train.iloc[tri])
        X_tr = fb.transform(train.iloc[tri], geo_train[tri], dr_train[tri], ru_train[tri])
        X_va = fb.transform(train.iloc[vai], geo_train[vai], dr_train[vai], ru_train[vai])
        X_te = fb.transform(test, geo_test, dr_test, ru_test)

        model, n_iter = _fit_xgb_fold(X_tr, y[tri], X_va, y[vai])
        oof[vai] = model.predict_proba(X_va).astype(np.float32)
        test_folds.append(model.predict_proba(X_te).astype(np.float32))
        best_iters.append(n_iter)
        f1 = _micro_f1(y[vai], oof[vai])
        scores.append(f1)
        print(f"  fold {fold}: F1={f1:.4f}  best_iter={n_iter}  ({time.time() - t0:.0f}s)")

    test_avg = np.mean(test_folds, axis=0)
    return oof, test_avg, scores, best_iters


def fit_full_train(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    geo_train: np.ndarray,
    geo_test: np.ndarray,
    dr_train: np.ndarray,
    dr_test: np.ndarray,
    ru_train: np.ndarray,
    ru_test: np.ndarray,
    n_estimators: int,
) -> tuple[xgb.XGBClassifier, np.ndarray]:
    fb = ShoumikFeatureBuilder()
    fb.fit(train)
    X_full = fb.transform(train, geo_train, dr_train, ru_train)
    X_te = fb.transform(test, geo_test, dr_test, ru_test)
    params = {**SHOUMIK_XGB_PARAMS, "n_estimators": n_estimators}
    model = xgb.XGBClassifier(**params)
    model.fit(X_full, y - 1, verbose=False)
    return model, model.predict_proba(X_te).astype(np.float32)


def _ensure_geo_latents(*, quick: bool, force: bool) -> None:
    if geo_latents_ready() and not force:
        print("Geo latents cached — skipping geo subprocess")
        return
    cmd = [sys.executable, str(ROOT / "src" / "run_024_geo.py")]
    if quick:
        cmd.append("--quick")
    if force:
        cmd.append("--force")
    print("Running geo training subprocess:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _blend_diagnostics(run_id: str, oof: np.ndarray, test: np.ndarray, y: np.ndarray) -> float:
    solo = _oof_f1(oof, y)
    print(f"\n{run_id} solo OOF F1: {solo:.4f}")

    lgbm_oof = np.load(LGBM_OOF).astype(np.float64)
    lgbm_test = np.load(LGBM_TEST).astype(np.float64)
    xgb_oof = np.load(XGB_OOF).astype(np.float64)
    xgb_test = np.load(XGB_TEST).astype(np.float64)
    r19_oof = np.load(RUN019_OOF).astype(np.float64)
    r19_test = np.load(RUN019_TEST).astype(np.float64)

    r12 = pairwise_diagnostic("lgbm_012", lgbm_oof, run_id, oof, y)
    r19 = pairwise_diagnostic("run_019", r19_oof, run_id, oof, y)
    f3, w_lg, w_xg, w_new = threeway_optimize(lgbm_oof, xgb_oof, oof, y)
    print(f"  3-way (LGBM+XGB+{run_id}): {f3:.4f}")
    best = max(r12["best_score"], r19["best_score"], f3)
    if best > THRESHOLD:
        if f3 >= max(r12["best_score"], r19["best_score"]):
            build_blend_submission(
                {"lgbm": lgbm_test, "xgb": xgb_test, run_id: test},
                {"lgbm": w_lg, "xgb": w_xg, run_id: w_new},
                space="proba",
                tag=f"{run_id}_3way",
            )
        elif r12["best_score"] >= r19["best_score"]:
            build_blend_submission(
                {"lgbm": lgbm_test, run_id: test},
                {"lgbm": r12["best_alpha"], run_id: 1.0 - r12["best_alpha"]},
                space=r12["best_space"],
                tag=f"{run_id}_2way_lgbm",
            )
        else:
            build_blend_submission(
                {run_id: test, "run_019": r19_test},
                {run_id: r19["best_alpha"], "run_019": 1.0 - r19["best_alpha"]},
                space=r19["best_space"],
                tag=f"{run_id}_2way_r19",
            )
    else:
        print(f"  No blend clears threshold {THRESHOLD:.4f}")
    return solo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="1 fold, 2 geo epochs")
    parser.add_argument("--force-geo", action="store_true", help="Retrain geo encoders")
    args = parser.parse_args()
    t0 = time.time()

    train, test = load_frames()
    y = pd.read_csv(PROCESSED_DIR / "y_train_full.csv")["damage_grade"].to_numpy()
    assert len(train) == len(y)
    print(f"Train: {len(y):,}  test: {len(test):,}")

    _ensure_geo_latents(quick=args.quick, force=args.force_geo)
    geo_train, geo_test, dr_train, dr_test, ru_train, ru_test = load_geo_latents()
    print(f"Geo latents: DR {dr_train.shape[1]}d  rollup {ru_train.shape[1]}d")

    print("\n── XGBoost CV ──")
    oof, test_cv_avg, scores, best_iters = run_cv(
        train, test, y,
        geo_train, geo_test, dr_train, dr_test, ru_train, ru_test,
        quick=args.quick,
    )
    mean_f1 = float(np.mean(scores))
    std_f1 = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    print(f"  CV mean F1: {mean_f1:.4f} ± {std_f1:.4f}")

    if args.quick:
        print(f"\nQuick test done in {time.time() - t0:.1f}s")
        return

    n_est_full = int(np.mean(best_iters))
    print(f"\n── Full-train XGB (n_estimators={n_est_full}) ──")
    _, test_proba = fit_full_train(
        train, test, y,
        geo_train, geo_test, dr_train, dr_test, ru_train, ru_test,
        n_est_full,
    )

    rm = RunManager()
    run_path = rm.run_path(RUN_ID)
    if run_path.exists():
        print(f"Warning: {RUN_ID} already exists — overwriting artifacts")
    else:
        rm.create_run(
            description="Shoumik replication: geo DR AE + geo3 rollup + Optuna XGB",
            model_type="XGBoost",
            feature_set="shoumik_ohe_geo_dr_rollup_passthrough",
            params={**SHOUMIK_XGB_PARAMS, "geo_dr_latent": 32, "geo_rollup_latent": 16},
            run_id=RUN_ID,
            objective="multiclass",
            cv_folds=CV_FOLDS,
            cv_metric="micro_f1",
            notes="CatBoostEncoder omitted (CV-safe). Submission = full-train refit.",
        )

    rm.save_cv_scores(RUN_ID, scores, mean_f1, std_f1)
    np.save(run_path / "oof_proba.npy", oof)
    np.save(run_path / "test_proba.npy", test_proba)
    test_csv = pd.read_csv(ROOT / "data" / "driven_data" / "test_values.csv")
    sub = pd.DataFrame({
        "building_id": test_csv["building_id"].values,
        "damage_grade": test_proba.argmax(axis=1) + 1,
    })
    rm.save_submission(RUN_ID, sub)
    _blend_diagnostics(RUN_ID, oof.astype(np.float64), test_proba.astype(np.float64), y)
    print(f"\nRegistered {RUN_ID} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
