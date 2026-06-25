#!/usr/bin/env python3
"""
run_042: LightGBM on (Original Features + Transductive LOO Spatial Features).

The "Poor Man's GNN". We load the standard 191 continuous features and concatenate
the 7 Leave-One-Out spatial features representing the mean age, area, and superstructure
of the building's geo_level_3 neighbors.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_manager import PROCESSED_DIR, RunManager
from run_trees_260k import pairwise_diagnostic

RANDOM_STATE = 42
CV_FOLDS = 5

# Robust standard LightGBM parameters (similar to run_019)
LGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "num_leaves": 128,
    "max_depth": 10,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 50,
    "n_estimators": 1500,
    "verbose": -1,
    "n_jobs": 4, # prevent segfaults on MPS
    "random_state": RANDOM_STATE
}

def main():
    t0 = time.time()
    
    print("── Loading Original Features ──")
    X_orig = pd.read_csv(PROCESSED_DIR / "X_train_run012.csv").fillna(0).values
    X_test_orig = pd.read_csv(PROCESSED_DIR / "X_test_run012.csv").fillna(0).values
    y = pd.read_csv(PROCESSED_DIR / "y_train_full.csv")["damage_grade"].to_numpy()
    
    print("── Loading LOO Spatial Features ──")
    X_loo = pd.read_csv(PROCESSED_DIR / "X_train_spatial_loo.csv").values
    X_test_loo = pd.read_csv(PROCESSED_DIR / "X_test_spatial_loo.csv").values
    
    print(f"Original shape: {X_orig.shape}, LOO shape: {X_loo.shape}")
    X = np.hstack([X_orig, X_loo])
    X_test = np.hstack([X_test_orig, X_test_loo])
    print(f"Stacked shape: {X.shape}")
    
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    oof_proba = np.zeros((len(y), 3), dtype=np.float32)
    test_folds_proba = []
    scores = []
    
    print("\n── Training LightGBM on Stacked Features ──")
    for fold, (tri, vai) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X[tri], y[tri]
        X_va, y_va = X[vai], y[vai]
        
        clf = lgb.LGBMClassifier(**LGBM_PARAMS)
        clf.fit(
            X_tr, y_tr - 1,
            eval_set=[(X_va, y_va - 1)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        )
        
        val_preds = clf.predict_proba(X_va)
        oof_proba[vai] = val_preds
        
        test_preds = clf.predict_proba(X_test)
        test_folds_proba.append(test_preds)
        
        f1 = f1_score(y_va, val_preds.argmax(axis=1) + 1, average="micro")
        print(f"  Fold {fold} F1: {f1:.5f} (iters: {clf.best_iteration_})")
        scores.append(f1)

    mean_f1 = float(np.mean(scores))
    std_f1 = float(np.std(scores, ddof=1))
    print(f"\nLightGBM LOO Spatial CV = {mean_f1:.4f} ± {std_f1:.4f}")
    
    test_proba = np.mean(test_folds_proba, axis=0)
    
    rm = RunManager()
    run_id = rm.get_next_run_id()
    
    rm.create_run(
        description=f"LightGBM on run_012 features + LOO geo_level_3 spatial features",
        model_type="LightGBM",
        feature_set=f"run_012+loo_spatial",
        params=LGBM_PARAMS,
        run_id=run_id,
        objective="multiclass",
        n_features=X.shape[1],
        cv_folds=CV_FOLDS,
        cv_metric="micro_f1",
    )
    rm.save_cv_scores(run_id, scores, mean_f1, std_f1)
    run_dir = rm.run_path(run_id)
    np.save(run_dir / "oof_proba.npy", oof_proba.astype(np.float32))
    np.save(run_dir / "test_proba.npy", test_proba.astype(np.float32))
    
    sub = pd.DataFrame({"building_id": pd.read_csv(ROOT / "data" / "driven_data" / "test_values.csv")["building_id"].values,
                        "damage_grade": test_proba.argmax(axis=1) + 1})
    rm.save_submission(run_id, sub)
    
    print("\nEvaluating Blend vs run_026 (SOTA Blend):")
    p26_oof = np.load(ROOT / "runs" / "run_026" / "oof_proba.npy").astype(np.float64)
    pairwise_diagnostic("run_026", p26_oof, run_id, oof_proba.astype(np.float64), y)
    
    print(f"\nRegistered {run_id} in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
