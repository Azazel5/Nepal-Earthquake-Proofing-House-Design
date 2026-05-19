# pip install category_encoders if missing
"""
Preprocess DrivenData Nepal earthquake training data for survival classification.

Loads train_values + train_labels, engineers features, encodes categoricals,
scales numerics, and writes stratified train/val splits for modeling.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from category_encoders.wrapper import NestedCVWrapper
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "driven_data"
PROCESSED_DIR = ROOT / "data" / "processed"
PREPROCESSING_DIR = ROOT / "models" / "preprocessing"

GEO_COLS = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]
GEO_ENC_COLS = ["geo_1_enc", "geo_2_enc", "geo_3_enc"]

CATEGORICAL_COLS = [
    "land_surface_condition",
    "foundation_type",
    "roof_type",
    "ground_floor_type",
    "other_floor_type",
    "position",
    "plan_configuration",
    "legal_ownership_status",
]

NUMERIC_COLS = [
    "count_floors_pre_eq",
    "age",
    "area_percentage",
    "height_percentage",
    "count_families",
]

AGE_SENTINEL = 995
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_and_merge() -> pd.DataFrame:
    """Load CSVs and merge on building_id; drop ID so it cannot leak into features."""
    values = pd.read_csv(DATA_DIR / "train_values.csv")
    labels = pd.read_csv(DATA_DIR / "train_labels.csv")
    df = values.merge(labels, on="building_id", how="inner", validate="one_to_one")
    df = df.drop(columns=["building_id"])
    return df


def create_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Binary survived (grade 1–2) vs severe damage (grade 3); keep damage_grade for multiclass."""
    df = df.copy()
    df["survived"] = (df["damage_grade"] <= 2).astype(int)
    return df


def binary_columns(df: pd.DataFrame) -> list[str]:
    """Superstructure and secondary-use flags are already 0/1 — leave unchanged."""
    return [
        c
        for c in df.columns
        if c.startswith("has_superstructure_") or c.startswith("has_secondary_use")
    ]


def stratified_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """80/20 stratified split on survived so both sets keep similar class ratios."""
    feature_cols = [c for c in df.columns if c not in ("survived", "damage_grade")]
    X = df[feature_cols]
    y_survived = df["survived"]
    y_multiclass = df["damage_grade"]

    X_train, X_val, y_train, y_val, y_train_mc, y_val_mc = train_test_split(
        X,
        y_survived,
        y_multiclass,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_survived,
    )
    return X_train, X_val, y_train, y_val, y_train_mc, y_val_mc


def process_age(
    train: pd.DataFrame, val: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """995 = unknown: flag it, then impute age from train median only (no val leakage)."""
    train = train.copy()
    val = val.copy()

    train["age_unknown"] = (train["age"] == AGE_SENTINEL).astype(int)
    val["age_unknown"] = (val["age"] == AGE_SENTINEL).astype(int)

    train_age = train["age"].replace(AGE_SENTINEL, np.nan)
    median_age = train_age.median()
    train["age"] = train_age.fillna(median_age).astype(float)
    val["age"] = val["age"].replace(AGE_SENTINEL, np.nan).fillna(median_age).astype(float)

    return train, val


def target_encode_geo(
    train: pd.DataFrame,
    val: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, TargetEncoder]:
    """
    Replace geo IDs with mean(survived) per region.
    NestedCVWrapper(TargetEncoder, cv=5) yields out-of-fold encodings on train;
    val is encoded with a TargetEncoder fit on the full train split (saved for inference).
    """
    oof_encoder = NestedCVWrapper(TargetEncoder(cols=GEO_COLS), cv=5)
    train_enc = oof_encoder.fit_transform(train[GEO_COLS], y_train)
    train_enc.columns = GEO_ENC_COLS

    inference_encoder = TargetEncoder(cols=GEO_COLS)
    inference_encoder.fit(train[GEO_COLS], y_train)
    val_enc = inference_encoder.transform(val[GEO_COLS])
    val_enc.columns = GEO_ENC_COLS

    train = train.drop(columns=GEO_COLS)
    val = val.drop(columns=GEO_COLS)
    train[GEO_ENC_COLS] = train_enc
    val[GEO_ENC_COLS] = val_enc
    return train, val, inference_encoder


def one_hot_encode(
    train: pd.DataFrame, val: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot nominal categoricals (drop_first); val columns aligned to train to avoid leakage."""
    train_dummies = pd.get_dummies(train[CATEGORICAL_COLS], drop_first=True)
    val_dummies = pd.get_dummies(val[CATEGORICAL_COLS], drop_first=True)
    val_dummies = val_dummies.reindex(columns=train_dummies.columns, fill_value=0)

    train = train.drop(columns=CATEGORICAL_COLS)
    val = val.drop(columns=CATEGORICAL_COLS)
    train = pd.concat([train, train_dummies], axis=1)
    val = pd.concat([val, val_dummies], axis=1)
    return train, val


def scale_numeric(
    train: pd.DataFrame, val: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """StandardScaler on numeric cols for logistic regression; fit on train only."""
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train[NUMERIC_COLS])
    val_scaled = scaler.transform(val[NUMERIC_COLS])

    train[NUMERIC_COLS] = train_scaled
    val[NUMERIC_COLS] = val_scaled
    return train, val, scaler


def build_feature_matrix(
    train: pd.DataFrame, val: pd.DataFrame, binary_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble final column order: geo enc, numerics, binaries, then one-hot dummies."""
    dummy_cols = [c for c in train.columns if c not in binary_cols + GEO_ENC_COLS + NUMERIC_COLS + ["age_unknown"]]
    ordered = GEO_ENC_COLS + NUMERIC_COLS + ["age_unknown"] + binary_cols + sorted(dummy_cols)
    return train[ordered], val[ordered]


def save_outputs(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_train_mc: pd.Series,
    y_val_mc: pd.Series,
) -> None:
    """Write processed feature matrices and label CSVs for downstream training."""
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_val.to_csv(PROCESSED_DIR / "X_val.csv", index=False)
    y_train.to_frame("survived").to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_val.to_frame("survived").to_csv(PROCESSED_DIR / "y_val.csv", index=False)
    y_train_mc.to_frame("damage_grade").to_csv(
        PROCESSED_DIR / "y_train_multiclass.csv", index=False
    )
    y_val_mc.to_frame("damage_grade").to_csv(
        PROCESSED_DIR / "y_val_multiclass.csv", index=False
    )


def print_class_distribution(series: pd.Series, title: str) -> None:
    counts = series.value_counts().sort_index()
    total = len(series)
    print(f"\n{title}")
    for label, n in counts.items():
        print(f"  {label}: {n:,} ({100 * n / total:.2f}%)")


def main() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(PREPROCESSING_DIR, exist_ok=True)

    # --- Load and define targets ---
    df = load_and_merge()
    df = create_targets(df)

    print("Class distribution (survived) after remapping:")
    print_class_distribution(df["survived"], "Full training set")

    # --- Train/val split before any fitting to prevent leakage ---
    X_train, X_val, y_train, y_val, y_train_mc, y_val_mc = stratified_split(df)
    bin_cols = binary_columns(X_train)

    # --- Age sentinel handling (train median only) ---
    X_train, X_val = process_age(X_train, X_val)

    # --- Geographic target encoding (high cardinality → compact risk signal) ---
    X_train, X_val, target_encoder = target_encode_geo(X_train, X_val, y_train)
    joblib.dump(target_encoder, PREPROCESSING_DIR / "target_encoder.pkl")

    # --- Nominal categoricals → one-hot (low cardinality, no order) ---
    X_train, X_val = one_hot_encode(X_train, X_val)

    # --- Scale numerics for linear models; tree models can ignore scale if needed ---
    X_train, X_val, scaler = scale_numeric(X_train, X_val)
    joblib.dump(scaler, PREPROCESSING_DIR / "scaler.pkl")

    # --- Final feature matrix (binaries untouched) ---
    X_train, X_val = build_feature_matrix(X_train, X_val, bin_cols)

    save_outputs(X_train, X_val, y_train, y_val, y_train_mc, y_val_mc)

    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_val shape:   {X_val.shape}")

    print("\nFinal feature columns:")
    for col in X_train.columns:
        print(f"  {col}")

    print_class_distribution(y_train, "y_train class balance")
    print_class_distribution(y_val, "y_val class balance")

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
