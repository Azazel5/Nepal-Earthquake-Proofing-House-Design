#!/usr/bin/env python3
"""
Tune micro-F1 aware decision rule (probability multipliers).

This script takes run_026 OOF probabilities and tunes 3 multipliers [w1, w2, w3]
such that pred = argmax(w * proba).
It evaluates the tuning using the canonical 5-fold CV to ensure we aren't
overfitting the OOF predictions.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_manager import PROCESSED_DIR

def _oof_f1(proba: np.ndarray, y: np.ndarray) -> float:
    return float(f1_score(y, proba.argmax(axis=1) + 1, average="micro", labels=[1, 2, 3]))

def optimize_multipliers(p_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Finds best [w1, w2, w3] for the training split."""
    def neg_f1(w):
        w = np.clip(w, 0, 10)
        # We can anchor one weight to 1.0 without loss of generality
        # but let's just optimize all 3, or fix w2=1.0. Let's fix w2=1.0 for stability.
        full_w = np.array([w[0], 1.0, w[1]])
        blend = p_train * full_w
        return -_oof_f1(blend, y_train)

    # Grid search for initial point
    best_val = 1.0
    best_w = (1.0, 1.0)
    for w1 in np.linspace(0.5, 2.0, 16):
        for w3 in np.linspace(0.5, 2.0, 16):
            v = neg_f1([w1, w3])
            if v < best_val:
                best_val = v
                best_w = (w1, w3)

    res = minimize(neg_f1, x0=np.array(best_w),
                   method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 1000})
    return np.array([res.x[0], 1.0, res.x[1]])

def main() -> None:
    y = pd.read_csv(PROCESSED_DIR / "y_train_full.csv")["damage_grade"].to_numpy()
    p_oof = np.load(ROOT / "runs" / "run_026" / "oof_proba.npy").astype(np.float64)

    baseline_f1 = _oof_f1(p_oof, y)
    print(f"run_026 baseline OOF F1: {baseline_f1:.5f}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    tuned_oof = np.zeros_like(p_oof)
    
    fold_baseline_scores = []
    fold_tuned_scores = []
    learned_weights = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(p_oof, y), start=1):
        p_tr, y_tr = p_oof[tr_idx], y[tr_idx]
        p_va, y_va = p_oof[va_idx], y[va_idx]
        
        baseline_va_f1 = _oof_f1(p_va, y_va)
        fold_baseline_scores.append(baseline_va_f1)
        
        w = optimize_multipliers(p_tr, y_tr)
        learned_weights.append(w)
        
        tuned_p_va = p_va * w
        tuned_va_f1 = _oof_f1(tuned_p_va, y_va)
        fold_tuned_scores.append(tuned_va_f1)
        
        tuned_oof[va_idx] = tuned_p_va
        
        print(f"Fold {fold}: baseline={baseline_va_f1:.5f} -> tuned={tuned_va_f1:.5f} (diff={tuned_va_f1 - baseline_va_f1:+.5f}) [w={w.round(3)}]")

    overall_tuned_f1 = _oof_f1(tuned_oof, y)
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Overall baseline OOF: {baseline_f1:.5f}")
    print(f"Overall tuned OOF:    {overall_tuned_f1:.5f}")
    print(f"Gain vs baseline:     {overall_tuned_f1 - baseline_f1:+.5f}")
    
    positive_folds = sum(1 for b, t in zip(fold_baseline_scores, fold_tuned_scores) if t > b)
    print(f"Positive folds:       {positive_folds}/5")
    
    mean_w = np.mean(learned_weights, axis=0)
    print(f"Mean learned weights: {mean_w.round(3)}")

if __name__ == "__main__":
    main()
