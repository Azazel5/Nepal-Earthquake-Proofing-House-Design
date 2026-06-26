#!/usr/bin/env python3
"""
run_043: LightGBM on (Original Features + Transductive Geo2 LOO Spatial Features).

Tests if the transductive LOO spatial signal generalizes safely when aggregated
at the much broader geo_level_2 grain (mean 182 buildings/cell).
Includes 3-way blend evaluation against run_026 and run_024.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')
np.seterr(all='ignore')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_manager import PROCESSED_DIR, RunManager
from run_trees_260k import pairwise_diagnostic

RANDOM_STATE = 42
CV_FOLDS = 5

# Same identical hyperparameters as run_026 base (run_019)
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
    "n_jobs": 4, # MPS safe
    "random_state": RANDOM_STATE
}

def get_3way_blend(p43, p26, p24, y):
    print("\n── 3-Way Blend Optimization ──")
    l43 = np.log(np.clip(p43, 1e-15, 1 - 1e-15))
    l26 = np.log(np.clip(p26, 1e-15, 1 - 1e-15))
    l24 = np.log(np.clip(p24, 1e-15, 1 - 1e-15))
    
    def loss_func(weights):
        w = np.exp(weights) / np.sum(np.exp(weights)) # softmax to ensure sum to 1
        blend_logits = w[0] * l43 + w[1] * l26 + w[2] * l24
        preds = blend_logits.argmax(axis=1) + 1
        return -f1_score(y, preds, average="micro")
        
    res = minimize(loss_func, [0.0, 0.0, 0.0], method="Nelder-Mead")
    w_opt = np.exp(res.x) / np.sum(np.exp(res.x))
    best_f1 = -res.fun
    
    print(f"Optimal 3-Way Logit Weights:")
    print(f"  run_043 (Geo2 LOO) : {w_opt[0]:.3f}")
    print(f"  run_026 (SOTA Blend): {w_opt[1]:.3f}")
    print(f"  run_024 (XGB Base) : {w_opt[2]:.3f}")
    print(f"  3-Way Blend F1     : {best_f1:.5f}")

def main():
    t0 = time.time()
    
    print("── Loading Original Features ──")
    X_orig = pd.read_csv(PROCESSED_DIR / "X_train_run012.csv").fillna(0).values
    X_test_orig = pd.read_csv(PROCESSED_DIR / "X_test_run012.csv").fillna(0).values
    y = pd.read_csv(PROCESSED_DIR / "y_train_full.csv")["damage_grade"].to_numpy()
    
    print("── Loading Geo2 LOO Spatial Features ──")
    X_loo = pd.read_csv(PROCESSED_DIR / "X_train_spatial_loo_geo2.csv").values
    X_test_loo = pd.read_csv(PROCESSED_DIR / "X_test_spatial_loo_geo2.csv").values
    
    X = np.hstack([X_orig, X_loo])
    X_test = np.hstack([X_test_orig, X_test_loo])
    print(f"Stacked shape: {X.shape}")
    
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    oof_proba = np.zeros((len(y), 3), dtype=np.float32)
    test_folds_proba = []
    scores = []
    
    print("\n── Training LightGBM run_043 ──")
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
    print(f"\nrun_043 (Geo2 LOO) CV = {mean_f1:.4f} ± {std_f1:.4f}")
    
    test_proba = np.mean(test_folds_proba, axis=0)
    
    rm = RunManager()
    run_id = rm.get_next_run_id()
    
    rm.create_run(
        description=f"LightGBM on run_012 + Geo2 LOO spatial features",
        model_type="LightGBM",
        feature_set=f"run_012+loo_spatial_geo2",
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
    
    print("\nEvaluating 2-Way Blend vs run_026 (SOTA Blend):")
    p26_oof = np.load(ROOT / "runs" / "run_026" / "oof_proba.npy").astype(np.float64)
    pairwise_diagnostic("run_026", p26_oof, run_id, oof_proba.astype(np.float64), y)
    
    # 3-Way blend
    p24_oof = np.load(ROOT / "runs" / "run_024" / "oof_proba.npy").astype(np.float64)
    get_3way_blend(oof_proba.astype(np.float64), p26_oof, p24_oof, y)
    
    print(f"\nRegistered {run_id} in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
