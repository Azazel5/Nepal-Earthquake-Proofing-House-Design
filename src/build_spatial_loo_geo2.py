#!/usr/bin/env python3
"""
build_spatial_loo_geo2.py

Builds Leave-One-Out (LOO) transductive spatial features based on `geo_level_2`.
Combines train and test sets to compute the LOO mean of basic structural features
for each building's neighbors, preventing "fingerprinting" memorization while
capturing robust macro-regional physics.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_manager import PROCESSED_DIR

def main():
    t0 = time.time()
    print("── Loading Raw Data for Geo2 LOO Aggregation ──")
    
    train_raw = pd.read_csv(ROOT / "data" / "driven_data" / "train_values.csv")
    test_raw = pd.read_csv(ROOT / "data" / "driven_data" / "test_values.csv")
    
    n_train = len(train_raw)
    full = pd.concat([train_raw, test_raw], axis=0).reset_index(drop=True)
    
    features = [
        "age",
        "area_percentage",
        "height_percentage",
        "count_floors_pre_eq",
        "has_superstructure_mud_mortar_stone",
        "has_superstructure_cement_mortar_brick",
        "has_superstructure_timber"
    ]
    
    print(f"Computing Transductive LOO Mean for {len(features)} basic features across geo_level_2...")
    
    loo_features = []
    
    for f in features:
        # Sum and count across the entire transductive geo_level_2 cell
        cell_sums = full.groupby("geo_level_2_id")[f].transform('sum')
        cell_counts = full.groupby("geo_level_2_id")[f].transform('count')
        
        # Subtract the building's own feature to prevent leakage
        sum_others = cell_sums - full[f]
        count_others = cell_counts - 1
        
        # Compute LOO mean. If count_others == 0, fallback to its own feature
        loo_mean = np.where(count_others > 0, sum_others / count_others, full[f])
        loo_features.append(loo_mean)
        
    df_loo = pd.DataFrame(np.column_stack(loo_features), columns=[f"loo_geo2_{f}" for f in features])
    
    print("Splitting and saving...")
    df_loo_train = df_loo.iloc[:n_train].reset_index(drop=True)
    df_loo_test = df_loo.iloc[n_train:].reset_index(drop=True)
    
    df_loo_train.to_csv(PROCESSED_DIR / "X_train_spatial_loo_geo2.csv", index=False)
    df_loo_test.to_csv(PROCESSED_DIR / "X_test_spatial_loo_geo2.csv", index=False)
    
    print(f"Done in {time.time() - t0:.1f}s.")
    print(f"Saved shapes: Train {df_loo_train.shape}, Test {df_loo_test.shape}")

if __name__ == "__main__":
    main()
