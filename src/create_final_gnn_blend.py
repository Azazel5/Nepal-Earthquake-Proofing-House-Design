#!/usr/bin/env python3
"""
create_final_gnn_blend.py
Generates the final kaggle submission by blending 5% GNN (run_039) and 95% SOTA (run_026).
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main():
    print("Loading test probabilities...")
    p26 = np.load(ROOT / "runs" / "run_026" / "test_proba.npy")
    p39 = np.load(ROOT / "runs" / "run_039" / "test_proba.npy")
    
    # Convert to logits
    l26 = np.log(np.clip(p26, 1e-15, 1 - 1e-15))
    l39 = np.log(np.clip(p39, 1e-15, 1 - 1e-15))
    
    # 5% GNN Blend
    print("Blending Logits: 0.95 * run_026 + 0.05 * run_039")
    blend_logits = (0.95 * l26) + (0.05 * l39)
    final_preds = blend_logits.argmax(axis=1) + 1
    
    # Save submission
    sub = pd.DataFrame({"building_id": pd.read_csv(ROOT / "data" / "driven_data" / "test_values.csv")["building_id"].values})
    sub["damage_grade"] = final_preds
    
    out_path = ROOT / "outputs" / "submission_final_blend_5pct_gnn.csv"
    sub.to_csv(out_path, index=False)
    
    print(f"\nSaved final blended submission to:")
    print(out_path)

if __name__ == "__main__":
    main()
