#!/usr/bin/env python3
"""
build_block_b.py

Builds Transductive Leave-One-Out (LOO) structural proportions (Block B features)
for geo_level_1, geo_level_2, and geo_level_3. 

Features:
- mean_age
- prop_weak (has_superstructure_adobe_mud | mud_mortar_stone | bamboo)
- prop_rc (rc_engineered | rc_non_engineered)
- prop_multistory (count_floors_pre_eq > 2)

All features are explicitly computed TRANSDUCTIVELY (Train + Test combined) 
and use LOO subtraction to prevent self-leakage.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_manager import PROCESSED_DIR

def print_variance_ratio(full: pd.DataFrame, level: str, feature: str):
    global_mean = full[feature].mean()
    groups = full.groupby(level)[feature]
    group_means = groups.mean()
    group_sizes = groups.count()
    between_var = np.sum(group_sizes * (group_means - global_mean)**2) / (len(full) - 1)
    within_var = np.sum(groups.apply(lambda g: np.sum((g - g.mean())**2))) / (len(full) - len(group_sizes))
    ratio = between_var / within_var if within_var > 0 else np.nan
    print(f"  {feature:<15} B/W Ratio: {ratio:.4f}")

def main():
    t0 = time.time()
    print("── Loading Raw Data for Block B Aggregation ──")
    
    train_raw = pd.read_csv(ROOT / "data" / "driven_data" / "train_values.csv")
    test_raw = pd.read_csv(ROOT / "data" / "driven_data" / "test_values.csv")
    
    n_train = len(train_raw)
    
    # EXACT CONFIRMATION: Combining Train + Test transductively before LOO
    full = pd.concat([train_raw, test_raw], axis=0).reset_index(drop=True)
    
    # Define Block B base features
    WEAK_COLS = [
        "has_superstructure_adobe_mud",
        "has_superstructure_mud_mortar_stone",
        "has_superstructure_bamboo",
    ]
    RC_COLS = [
        "has_superstructure_rc_engineered",
        "has_superstructure_rc_non_engineered",
    ]
    
    full["prop_weak"] = full[WEAK_COLS].max(axis=1).values
    full["prop_rc"] = full[RC_COLS].max(axis=1).values
    full["prop_multistory"] = (full["count_floors_pre_eq"] > 2).astype(int).values
    full["mean_age"] = full["age"].values
    
    features = ["mean_age", "prop_weak", "prop_rc", "prop_multistory"]
    
    for level in ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]:
        print(f"\nComputing Transductive LOO for {level}...")
        
        # Variance Check
        for f in features:
            print_variance_ratio(full, level, f)
            
        loo_features = []
        for f in features:
            cell_sums = full.groupby(level)[f].transform('sum')
            cell_counts = full.groupby(level)[f].transform('count')
            
            # Leave-One-Out subtraction
            sum_others = cell_sums - full[f]
            count_others = cell_counts - 1
            
            # Fallback to building's own feature if isolated (count_others == 0)
            loo_mean = np.where(count_others > 0, sum_others / count_others, full[f])
            loo_features.append(loo_mean)
            
        df_loo = pd.DataFrame(np.column_stack(loo_features), columns=[f"loo_{level}_{f}" for f in features])
        
        df_loo_train = df_loo.iloc[:n_train].reset_index(drop=True)
        df_loo_test = df_loo.iloc[n_train:].reset_index(drop=True)
        
        train_path = PROCESSED_DIR / f"X_train_block_b_{level}.csv"
        test_path = PROCESSED_DIR / f"X_test_block_b_{level}.csv"
        
        df_loo_train.to_csv(train_path, index=False)
        df_loo_test.to_csv(test_path, index=False)
        
    print(f"\nDone in {time.time() - t0:.1f}s.")

if __name__ == "__main__":
    main()
