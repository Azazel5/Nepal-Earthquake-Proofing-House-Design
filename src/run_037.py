#!/usr/bin/env python3
"""
run_037: Ordinal LightGBM (Regression-then-Threshold) on PCA embedded features.

Mimics run_019 (PCA compression) but uses a regression objective instead of multiclass.
Thresholds [t1, t2] are tuned per-fold to maximize micro-F1.
"""

from __future__ import annotations

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from scipy.optimize import minimize
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
LGBM_CV_REF = 0.7588
NOISE = 0.0016
THRESHOLD = LGBM_CV_REF + NOISE

LGBM_OOF = ROOT / "runs" / "run_012" / "oof_proba.npy"
LGBM_TEST = ROOT / "runs" / "run_012" / "test_proba.npy"
XGB_OOF = ROOT / "runs" / "run_015" / "oof_proba.npy"
XGB_TEST = ROOT / "runs" / "run_015" / "test_proba.npy"

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

def _fit_lgbm_reg_fold(X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray):
    params = dict(TRIAL_66_PARAMS)
    params["objective"] = "regression"
    params["metric"] = "rmse"
    params.pop("num_class", None)
    
    model = LGBMRegressor(**params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model

def optimize_thresholds(y_pred_cont: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """Find t1, t2 to maximize F1 score."""
    def neg_f1(t):
        if t[0] >= t[1]:
            return 0.0
        preds = np.ones_like(y_pred_cont, dtype=int)
        preds[y_pred_cont > t[0]] = 2
        preds[y_pred_cont > t[1]] = 3
        return -f1_score(y_true, preds, average="micro", labels=[1, 2, 3])

    res = minimize(neg_f1, x0=np.array([1.5, 2.5]), method="Nelder-Mead")
    return res.x[0], res.x[1]

def apply_thresholds_to_proba(y_pred_cont: np.ndarray, t1: float, t2: float) -> np.ndarray:
    """Convert continuous predictions to pseudo-probabilities based on distance to thresholds."""
    # A simple mapping to create a probability distribution:
    # We want argmax(proba) to match the hard thresholding, but we also want a smooth space for blending.
    # Let's use a softmax over negative distances squared, scaled.
    # We can represent the class centers roughly as:
    c1 = t1 - 0.5
    c2 = (t1 + t2) / 2
    c3 = t2 + 0.5
    
    d1 = -np.abs(y_pred_cont - c1)
    d2 = -np.abs(y_pred_cont - c2)
    d3 = -np.abs(y_pred_cont - c3)
    
    # Exponentiate and normalize (softmax)
    logits = np.stack([d1, d2, d3], axis=1) * 3.0  # temperature
    exp_l = np.exp(logits)
    return exp_l / exp_l.sum(axis=1, keepdims=True)

def run_pca_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    cols = list(X.columns)
    pca_cols, pass_cols = cols, [] # full variant, we know it's good from run_019
    k = 80 # we know 80 is good from run_019

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(X, y))

    print(f"\n── Ordinal LGBM: k={k} ──")

    oof_proba = np.zeros((len(y), 3), dtype=np.float32)
    test_folds_proba: list[np.ndarray] = []
    scores: list[float] = []
    
    thresholds_list = []

    for fold, (tri, vai) in enumerate(splits, start=1):
        t0 = time.time()
        X_tr, X_va = X.iloc[tri], X.iloc[vai]
        df_tr, df_va, df_te = _transform_pca(X_tr, X_va, X_test, pca_cols, pass_cols, k)
        
        model = _fit_lgbm_reg_fold(df_tr, y[tri], df_va, y[vai])
        
        # Predict continuous
        val_pred_cont = model.predict(df_va)
        test_pred_cont = model.predict(df_te)
        
        # Optimize thresholds on validation
        t1, t2 = optimize_thresholds(val_pred_cont, y[vai])
        thresholds_list.append((t1, t2))
        
        # Convert to proba
        val_proba = apply_thresholds_to_proba(val_pred_cont, t1, t2)
        test_proba = apply_thresholds_to_proba(test_pred_cont, t1, t2)
        
        oof_proba[vai] = val_proba.astype(np.float32)
        test_folds_proba.append(test_proba.astype(np.float32))
        
        f1 = f1_score(y[vai], val_proba.argmax(axis=1) + 1, average="micro", labels=[1, 2, 3])
        scores.append(f1)
        print(f"  fold {fold}: F1={f1:.4f} thresholds=[{t1:.3f}, {t2:.3f}] ({time.time() - t0:.0f}s)")

    test_avg_proba = np.mean(test_folds_proba, axis=0)
    print(f"  Mean thresholds: {np.mean(thresholds_list, axis=0)}")
    return oof_proba, test_avg_proba, scores

def main() -> None:
    t0 = time.time()

    X = pd.read_csv(PROCESSED_DIR / "X_train_run012.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test_run012.csv")
    y = pd.read_csv(PROCESSED_DIR / "y_train_full.csv")["damage_grade"].to_numpy()
    print(f"Features: {X.shape[1]}  train: {len(y):,}")

    oof_proba, test_proba, scores = run_pca_cv(X, y, X_test)
    mean_f1 = float(np.mean(scores))
    std_f1 = float(np.std(scores, ddof=1))

    print(f"\nOrdinal Regression CV={mean_f1:.4f} ± {std_f1:.4f}")

    rm = RunManager()
    run_id = rm.get_next_run_id()
    
    rm.create_run(
        description=f"Ordinal LGBM (Regression+Thresholds) on full PCA k=80",
        model_type="LightGBM_Ordinal",
        feature_set=f"pca_full_k80+lgbm_ordinal",
        params={"objective": "regression", "n_components": 80, **TRIAL_66_PARAMS},
        run_id=run_id,
        objective="ordinal",
        n_features=80,
        cv_folds=CV_FOLDS,
        cv_metric="micro_f1",
        notes=f"Tuned thresholds on val fold.",
    )
    rm.save_cv_scores(run_id, scores, mean_f1, std_f1)
    run_dir = rm.run_path(run_id)
    np.save(run_dir / "oof_proba.npy", oof_proba.astype(np.float32))
    np.save(run_dir / "test_proba.npy", test_proba.astype(np.float32))
    
    test_csv = pd.read_csv(ROOT / "data" / "driven_data" / "test_values.csv")
    sub = pd.DataFrame({"building_id": test_csv["building_id"].values,
                        "damage_grade": test_proba.argmax(axis=1) + 1})
    rm.save_submission(run_id, sub)
    
    # Evaluate blend with run_024 (Shoumik XGB) because it's the other part of the SOTA blend run_026
    p24_oof = np.load(ROOT / "runs" / "run_024" / "oof_proba.npy").astype(np.float64)
    print("\nEvaluating Blend vs run_024:")
    pairwise_diagnostic("run_024", p24_oof, run_id, oof_proba.astype(np.float64), y)

    print(f"\nRegistered {run_id} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
