#!/usr/bin/env python3
"""
Blend the GNN (run_039) with the SOTA tree models (run_019, run_024).

Because run_039 suffers from a spatial CV leak (CV=0.7684 but LB=0.7450), 
we CANNOT trust OOF optimization to find the optimal blend weight (it will give 
run_039 ~95% weight).

Instead, this script:
1. Calculates loss correlations between the GNN and the trees to verify decorrelation.
2. Creates submissions using fixed, small weights for the GNN (5%, 10%, 15%, 20%)
   mixed into the SOTA run_026 blend.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_manager import PROCESSED_DIR
from run_trees_260k import _oof_f1, build_blend_submission

def main():
    print("── Loading Probas ──")
    y = pd.read_csv(PROCESSED_DIR / "y_train_full.csv")["damage_grade"].to_numpy()
    
    p19_oof = np.load(ROOT / "runs/run_019/oof_proba.npy").astype(np.float64)
    p24_oof = np.load(ROOT / "runs/run_024/oof_proba.npy").astype(np.float64)
    p39_oof = np.load(ROOT / "runs/run_039/oof_proba.npy").astype(np.float64)
    p26_oof = np.load(ROOT / "runs/run_026/oof_proba.npy").astype(np.float64)
    
    p19_test = np.load(ROOT / "runs/run_019/test_proba.npy").astype(np.float64)
    p24_test = np.load(ROOT / "runs/run_024/test_proba.npy").astype(np.float64)
    p39_test = np.load(ROOT / "runs/run_039/test_proba.npy").astype(np.float64)
    p26_test = np.load(ROOT / "runs/run_026/test_proba.npy").astype(np.float64)
    
    # 1. Loss Correlation
    eps = 1e-7
    idx = y - 1
    N = len(y)
    
    loss_19 = -np.log(p19_oof[np.arange(N), idx] + eps)
    loss_24 = -np.log(p24_oof[np.arange(N), idx] + eps)
    loss_39 = -np.log(p39_oof[np.arange(N), idx] + eps)
    loss_26 = -np.log(p26_oof[np.arange(N), idx] + eps)
    
    print("\n── Loss Correlations (Lower is better / more decorrelated) ──")
    print(f"run_019 (LGBM) vs run_024 (XGB):  {np.corrcoef(loss_19, loss_24)[0, 1]:.4f}")
    print(f"run_019 (LGBM) vs run_039 (GNN):  {np.corrcoef(loss_19, loss_39)[0, 1]:.4f}")
    print(f"run_024 (XGB)  vs run_039 (GNN):  {np.corrcoef(loss_24, loss_39)[0, 1]:.4f}")
    print(f"run_026 (SOTA) vs run_039 (GNN):  {np.corrcoef(loss_26, loss_39)[0, 1]:.4f}")
    
    # 2. Fixed-weight blends
    print("\n── Fixed-Weight Blends (GNN + run_026) ──")
    test_csv = pd.read_csv(ROOT / "data/driven_data/test_values.csv")
    
    for w_gnn in [0.02, 0.05, 0.10, 0.15, 0.20]:
        w_26 = 1.0 - w_gnn
        
        blend_oof = w_26 * p26_oof + w_gnn * p39_oof
        blend_test = w_26 * p26_test + w_gnn * p39_test
        
        f1 = _oof_f1(blend_oof, y)
        print(f"Weight GNN={w_gnn:.2f}, run_026={w_26:.2f} -> Leaky OOF F1: {f1:.5f}")
        
        # Save submission
        sub_df = pd.DataFrame({
            "building_id": test_csv["building_id"].values,
            "damage_grade": blend_test.argmax(axis=1) + 1,
        })
        out_path = ROOT / "outputs" / f"submission_blend_gnn_{int(w_gnn*100)}pct.csv"
        sub_df.to_csv(out_path, index=False)
    
    print(f"\nSubmissions saved to {ROOT}/outputs/")
    
if __name__ == "__main__":
    main()
