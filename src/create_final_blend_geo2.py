#!/usr/bin/env python3
"""
create_final_blend_geo2.py
Generates the final Kaggle submission by blending run_026 (SOTA) and run_043 (Geo2 LOO).
Uses the optimal Logit Blend weights found during cross-validation.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main():
    print("Loading test probabilities...")
    p26 = np.load(ROOT / "runs" / "run_026" / "test_proba.npy")
    p43 = np.load(ROOT / "runs" / "run_043" / "test_proba.npy")
    
    # Convert to logits
    p26_logits = np.log(np.clip(p26, 1e-15, 1 - 1e-15))
    p43_logits = np.log(np.clip(p43, 1e-15, 1 - 1e-15))
    
    # Optimal logit blend weights from pairwise_diagnostic
    # alpha=0.366 on run_026
    alpha = 0.366
    print(f"Blending Logits: {alpha} * run_026 + {1 - alpha} * run_043")
    
    blend_logits = (alpha * p26_logits) + ((1 - alpha) * p43_logits)
    final_preds = blend_logits.argmax(axis=1) + 1
    
    # Save submission
    sub = pd.DataFrame({"building_id": pd.read_csv(ROOT / "data" / "driven_data" / "test_values.csv")["building_id"].values})
    sub["damage_grade"] = final_preds
    
    out_path = ROOT / "outputs" / "submission_final_blend_run043_run026.csv"
    sub.to_csv(out_path, index=False)
    
    print(f"\nSaved final blended submission to:")
    print(out_path)

if __name__ == "__main__":
    main()
