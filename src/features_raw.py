"""
Raw feature view for ensemble decorrelation experiments (run_014 raw-cat experiment).

Deliberately excludes everything engineered for run_012:
  - no Laplace-smoothed geo rates
  - no entity-embedding columns
  - no engineered interactions (age_x_height, etc.)

Includes only:
  - geo_level_1/2/3_id as raw categoricals (native handling, no precomputed rates)
  - 8 low-cardinality categoricals, raw
  - 5 raw numerics
  - 22 binary superstructure / secondary-use flags

Categorical columns are emitted as pandas 'category' dtype with categories
aligned across train+test (via union) so LGBM/CatBoost see consistent codes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embed import EMBED_COLS, NUMERIC_COLS, apply_age_sentinel, load_raw_data
from preprocess import binary_columns

RAW_CAT_COLS: list[str] = list(EMBED_COLS.keys())  # geo1/2/3 + 8 low-card cats (11 total)
RAW_NUM_COLS: list[str] = list(NUMERIC_COLS)         # 5 numerics


def build_raw_features() -> tuple[pd.DataFrame, pd.DataFrame, "pd.Series[int]"]:
    """Returns (X_train, X_test, y) with categorical cols as pandas 'category' dtype."""
    train, test = load_raw_data()
    train, test = apply_age_sentinel(train, test)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    bin_cols = binary_columns(train)
    cols = RAW_CAT_COLS + RAW_NUM_COLS + bin_cols

    X_train = train[cols].copy()
    X_test = test[cols].copy()

    # Align categorical codes across train+test (union of categories)
    for c in RAW_CAT_COLS:
        combined = pd.concat([X_train[c], X_test[c]], axis=0).astype(str)
        cats = pd.Categorical(combined).categories
        X_train[c] = pd.Categorical(X_train[c].astype(str), categories=cats)
        X_test[c] = pd.Categorical(X_test[c].astype(str), categories=cats)

    y = train["damage_grade"].to_numpy()
    return X_train, X_test, y


if __name__ == "__main__":
    X_train, X_test, y = build_raw_features()
    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"Categorical cols ({len(RAW_CAT_COLS)}): {RAW_CAT_COLS}")
    print(f"Numeric cols ({len(RAW_NUM_COLS)}): {RAW_NUM_COLS}")
    print(f"Binary cols ({X_train.shape[1] - len(RAW_CAT_COLS) - len(RAW_NUM_COLS)})")
    print(f"\ndtypes:\n{X_train.dtypes}")
