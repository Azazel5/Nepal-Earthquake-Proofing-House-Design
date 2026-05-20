"""
Shared test-set preprocessing and multiclass prediction for submissions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from preprocess import (
    AGE_SENTINEL,
    CATEGORICAL_COLS,
    DATA_DIR,
    GEO_COLS,
    GEO_ENC_COLS,
    NUMERIC_COLS,
    PROCESSED_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    binary_columns,
    build_feature_matrix,
)
from run_manager import PREPROCESSING_DIR, ROOT

ARTIFACT_CANDIDATES = {
    "scaler": [PREPROCESSING_DIR / "scaler.pkl", ROOT / "models" / "scaler.pkl"],
    "target_encoder": [
        PREPROCESSING_DIR / "target_encoder.pkl",
        ROOT / "models" / "target_encoder.pkl",
    ],
}


def load_artifact(name: str, candidates: list[Path]):
    """Load first existing joblib artifact from candidate paths."""
    import joblib

    for path in candidates:
        if path.exists():
            return joblib.load(path)
    paths = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Missing {name}. Tried:\n  {paths}")


def load_train_age_median() -> float:
    """Median age from training split (exclude sentinel 995)."""
    values = pd.read_csv(DATA_DIR / "train_values.csv")
    labels = pd.read_csv(DATA_DIR / "train_labels.csv")
    df = values.merge(labels, on="building_id")
    df["survived"] = (df["damage_grade"] <= 2).astype(int)
    train_df, _ = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["survived"],
    )
    return float(train_df["age"].replace(AGE_SENTINEL, np.nan).median())


def preprocess_test(
    test_df: pd.DataFrame,
    target_encoder,
    scaler,
    train_columns: list[str],
    age_median: float,
) -> pd.DataFrame:
    """Apply fitted preprocessing; column order matches X_train.csv."""
    df = test_df.copy()
    bin_cols = binary_columns(df)

    df["age_unknown"] = (df["age"] == AGE_SENTINEL).astype(int)
    df["age"] = df["age"].replace(AGE_SENTINEL, np.nan).fillna(age_median).astype(float)

    geo_enc = target_encoder.transform(df[GEO_COLS])
    geo_enc.columns = GEO_ENC_COLS
    df = df.drop(columns=GEO_COLS)
    df[GEO_ENC_COLS] = geo_enc

    dummies = pd.get_dummies(df[CATEGORICAL_COLS], drop_first=True)
    dummy_train_cols = [
        c
        for c in train_columns
        if c not in bin_cols + GEO_ENC_COLS + NUMERIC_COLS + ["age_unknown"]
    ]
    dummies = dummies.reindex(columns=dummy_train_cols, fill_value=0)
    df = df.drop(columns=CATEGORICAL_COLS)
    df = pd.concat([df, dummies], axis=1)

    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])
    df, _ = build_feature_matrix(df, df, bin_cols)
    return df[train_columns]


def predict_multiclass(model: LGBMClassifier, X_test: pd.DataFrame) -> np.ndarray:
    """Predict damage grades 1, 2, or 3."""
    proba = model.predict_proba(X_test)
    class_idx = np.argmax(proba, axis=1)
    return model.classes_[class_idx].astype(int)


def build_test_matrix() -> tuple[pd.Series, pd.DataFrame]:
    """Load test_values.csv, preprocess, return building_ids and feature matrix."""
    scaler = load_artifact("scaler", ARTIFACT_CANDIDATES["scaler"])
    target_encoder = load_artifact("target_encoder", ARTIFACT_CANDIDATES["target_encoder"])
    train_columns = list(pd.read_csv(PROCESSED_DIR / "X_train.csv").columns)

    test_raw = pd.read_csv(DATA_DIR / "test_values.csv")
    building_ids = test_raw["building_id"].copy()
    test_features = test_raw.drop(columns=["building_id"])
    X_test = preprocess_test(
        test_features,
        target_encoder,
        scaler,
        train_columns,
        load_train_age_median(),
    )
    return building_ids, X_test


def grades_to_submission_df(building_ids: pd.Series, grades: np.ndarray) -> pd.DataFrame:
    """Validate grades and build submission DataFrame."""
    unique = set(np.unique(grades))
    if not unique.issubset({1, 2, 3}):
        raise ValueError(f"Invalid damage_grade predictions: {unique}")
    return pd.DataFrame({"building_id": building_ids, "damage_grade": grades.astype(int)})


def print_grade_distribution(grades: np.ndarray, title: str) -> None:
    """Print % per damage grade."""
    total = len(grades)
    print(f"\n{title}")
    for grade in (1, 2, 3):
        n = int(np.sum(grades == grade))
        print(f"  grade {grade}: {100 * n / total:.1f}%")
