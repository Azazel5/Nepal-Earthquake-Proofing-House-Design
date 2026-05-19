# pip install lightgbm category_encoders if missing
"""
Generate DrivenData submissions from test_values.csv.

Version 1: binary LightGBM → mapped damage grades (2 or 3 only).
Version 2: multiclass LightGBM trained on damage_grade 1/2/3.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import f1_score
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

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"

# Artifact search paths (current layout + legacy paths from earlier runs)
ARTIFACT_CANDIDATES = {
    "lgbm_binary": [
        ROOT / "models" / "lightGBM" / "lgbm_model.pkl",
        ROOT / "models" / "lgbm_model.pkl",
    ],
    "scaler": [
        ROOT / "models" / "preprocessing" / "scaler.pkl",
        ROOT / "models" / "scaler.pkl",
    ],
    "target_encoder": [
        ROOT / "models" / "preprocessing" / "target_encoder.pkl",
        ROOT / "models" / "target_encoder.pkl",
    ],
}

FINAL_N_ESTIMATORS = 645  # mean best iteration from binary CV in train.py
N_SPLITS = 5
EARLY_STOPPING_ROUNDS = 50


def load_artifact(name: str, candidates: list[Path]):
    """Load a joblib artifact; raise FileNotFoundError with clear message if missing."""
    for path in candidates:
        if path.exists():
            return joblib.load(path)
    paths = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Missing {name} artifact. Tried:\n  {paths}\n"
        f"Run src/preprocess.py and src/train.py first."
    )


def load_train_age_median() -> float:
    """Match preprocess.py: median age from train split only (exclude sentinel 995)."""
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
    return train_df["age"].replace(AGE_SENTINEL, np.nan).median()


def preprocess_test(
    test_df: pd.DataFrame,
    target_encoder,
    scaler,
    train_columns: list[str],
    age_median: float,
) -> pd.DataFrame:
    """
    Apply fitted preprocessing to raw test rows (no refitting).
    Column order and names must match data/processed/X_train.csv exactly.
    """
    df = test_df.copy()
    bin_cols = binary_columns(df)

    # Age sentinel: same rules as training
    df["age_unknown"] = (df["age"] == AGE_SENTINEL).astype(int)
    df["age"] = df["age"].replace(AGE_SENTINEL, np.nan).fillna(age_median).astype(float)

    # Geographic target encoding (fitted on training split)
    geo_enc = target_encoder.transform(df[GEO_COLS])
    geo_enc.columns = GEO_ENC_COLS
    df = df.drop(columns=GEO_COLS)
    df[GEO_ENC_COLS] = geo_enc

    # One-hot categoricals; align to training dummy columns
    dummies = pd.get_dummies(df[CATEGORICAL_COLS], drop_first=True)
    dummy_train_cols = [
        c
        for c in train_columns
        if c not in bin_cols + GEO_ENC_COLS + NUMERIC_COLS + ["age_unknown"]
    ]
    dummies = dummies.reindex(columns=dummy_train_cols, fill_value=0)
    df = df.drop(columns=CATEGORICAL_COLS)
    df = pd.concat([df, dummies], axis=1)

    # Scale numerics with training scaler
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

    # Same column order as X_train.csv
    df, _ = build_feature_matrix(df, df, bin_cols)
    return df[train_columns]


def print_grade_distribution(grades: pd.Series, title: str) -> None:
    """Print % per damage grade for submission QA."""
    counts = grades.value_counts().sort_index()
    total = len(grades)
    print(f"\n{title}")
    for grade in (1, 2, 3):
        n = counts.get(grade, 0)
        print(f"  grade {grade}: {100 * n / total:.1f}%")


def save_submission(building_ids: pd.Series, damage_grade: np.ndarray, path: Path) -> None:
    """Write building_id + damage_grade CSV without index."""
    out = pd.DataFrame({"building_id": building_ids, "damage_grade": damage_grade.astype(int)})
    out.to_csv(path, index=False)


def predict_binary_mapped(model, X_test: pd.DataFrame) -> np.ndarray:
    """
    Binary survived probability → damage_grade 2 or 3 only.
    prob >= 0.5 → grade 2; prob < 0.5 → grade 3.
    """
    proba = model.predict_proba(X_test)[:, 1]
    return np.where(proba >= 0.5, 2, 3)


def build_multiclass_params() -> dict:
    """Same LightGBM settings as train.py, but multiclass (no scale_pos_weight)."""
    return {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        "verbose": -1,
    }


def run_multiclass_cv(
    X: pd.DataFrame, y: np.ndarray, params: dict
) -> tuple[list[float], list[int]]:
    """Stratified 5-fold CV with micro F1 on damage grades 1–3."""
    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    micro_f1_scores: list[float] = []
    best_iterations: list[int] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        print("=" * 60)
        print(f"Multiclass fold {fold_idx} / {N_SPLITS}")
        print("=" * 60)

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = LGBMClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="multi_logloss",
            callbacks=[early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        best_iterations.append(model.best_iteration_)

        y_pred = model.predict(X_val)
        micro_f1 = f1_score(y_val, y_pred, average="micro", labels=[1, 2, 3])
        micro_f1_scores.append(micro_f1)
        print(f"  Micro F1: {micro_f1:.4f}  (best iteration: {model.best_iteration_})")

    return micro_f1_scores, best_iterations


def predict_multiclass(model: LGBMClassifier, X_test: pd.DataFrame) -> np.ndarray:
    """argmax on class probabilities → damage_grade in {1, 2, 3}."""
    proba = model.predict_proba(X_test)
    class_idx = np.argmax(proba, axis=1)
    return model.classes_[class_idx].astype(int)


def main() -> None:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    try:
        binary_model = load_artifact("binary LightGBM", ARTIFACT_CANDIDATES["lgbm_binary"])
        scaler = load_artifact("scaler", ARTIFACT_CANDIDATES["scaler"])
        target_encoder = load_artifact(
            "target encoder", ARTIFACT_CANDIDATES["target_encoder"]
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # Reference schema from processed training features
    X_train_ref = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    train_columns = list(X_train_ref.columns)

    # --- Load and preprocess test set ---
    test_raw = pd.read_csv(DATA_DIR / "test_values.csv")
    building_ids = test_raw["building_id"].copy()
    test_features = test_raw.drop(columns=["building_id"])

    age_median = load_train_age_median()
    X_test = preprocess_test(
        test_features,
        target_encoder,
        scaler,
        train_columns,
        age_median,
    )

    print(f"Test set shape after preprocessing: {X_test.shape}")

    # --- Version 1: binary model → grades 2 or 3 ---
    grades_binary = predict_binary_mapped(binary_model, X_test)
    binary_path = OUTPUTS_DIR / "submission_binary_mapped.csv"
    save_submission(building_ids, grades_binary, binary_path)
    print(f"Submission saved to {binary_path.relative_to(ROOT)}")
    print_grade_distribution(pd.Series(grades_binary), "Binary-mapped prediction distribution")

    # --- Version 2: train multiclass on full processed train, predict test ---
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y_mc = pd.read_csv(PROCESSED_DIR / "y_train_multiclass.csv")["damage_grade"].to_numpy()

    mc_params = build_multiclass_params()
    print("\nMulticlass cross-validation (Micro F1):")
    micro_f1_scores, _ = run_multiclass_cv(X_train, y_mc, mc_params)

    mean_f1 = np.mean(micro_f1_scores)
    std_f1 = np.std(micro_f1_scores, ddof=1)
    print(f"\nMicro F1: {mean_f1:.4f} ± {std_f1:.4f}")

    final_mc_params = {**mc_params, "n_estimators": FINAL_N_ESTIMATORS}
    multiclass_model = LGBMClassifier(**final_mc_params)
    multiclass_model.fit(X_train, y_mc)

    grades_mc = predict_multiclass(multiclass_model, X_test)
    mc_path = OUTPUTS_DIR / "submission_multiclass.csv"
    save_submission(building_ids, grades_mc, mc_path)
    print(f"\nSubmission saved to {mc_path.relative_to(ROOT)}")
    print_grade_distribution(pd.Series(grades_mc), "Multiclass prediction distribution")


if __name__ == "__main__":
    main()
