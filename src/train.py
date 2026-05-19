# pip install lightgbm if missing
"""
Train a LightGBM binary classifier for Nepal earthquake building survival.

Uses stratified 5-fold CV for evaluation, then refits on full X_train and
saves the model artifact for inference.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs"
LGBM_MODEL_PATH = ROOT / "models" / "lightGBM" / "lgbm_model.pkl"

N_SPLITS = 5
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 50

METRIC_NAMES = ("f1_macro", "auc_roc", "accuracy", "precision", "recall")


def load_training_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Load preprocessed features and binary survived labels."""
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")["survived"].to_numpy()
    return X_train, y_train


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """Weight positive class in LightGBM: count(0) / count(1) for 2:1 imbalance."""
    n_neg = np.sum(y == 0)
    n_pos = np.sum(y == 1)
    return n_neg / n_pos


def build_params(scale_pos_weight: float) -> dict:
    """Default LightGBM hyperparameters (not tuned yet)."""
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "scale_pos_weight": scale_pos_weight,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        "verbose": -1,
    }


def evaluate_fold(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, float]:
    """Compute classification metrics on one held-out fold."""
    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "auc_roc": roc_auc_score(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def run_cross_validation(
    X: pd.DataFrame, y: np.ndarray, params: dict
) -> tuple[dict[str, list[float]], list[int]]:
    """
    Stratified 5-fold CV with early stopping on each fold's validation split.
    Returns per-metric fold scores and best iteration per fold.
    """
    cv = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    fold_metrics: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    best_iterations: list[int] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        print("=" * 60)
        print(f"Fold {fold_idx} / {N_SPLITS}")
        print("=" * 60)

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = LGBMClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
        )

        best_iter = model.best_iteration_
        best_iterations.append(best_iter)
        print(f"Best iteration: {best_iter}")

        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = evaluate_fold(y_val, y_pred, y_proba)

        for name in METRIC_NAMES:
            fold_metrics[name].append(metrics[name])
            print(f"  {name}: {metrics[name]:.4f}")

    return fold_metrics, best_iterations


def print_cv_summary(fold_metrics: dict[str, list[float]]) -> None:
    """Print mean ± std for each metric across folds."""
    print("\n" + "=" * 60)
    print("Cross-validation summary (mean ± std)")
    print("=" * 60)
    labels = {
        "f1_macro": "F1 macro",
        "auc_roc": "AUC-ROC",
        "accuracy": "Accuracy",
        "precision": "Precision (macro)",
        "recall": "Recall (macro)",
    }
    for name in METRIC_NAMES:
        values = fold_metrics[name]
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        print(f"  {labels[name]}: {mean:.4f} ± {std:.4f}")


def fit_final_model(
    X: pd.DataFrame, y: np.ndarray, params: dict, n_estimators: int
) -> LGBMClassifier:
    """Refit on all training data with fixed n_estimators (mean best iteration from CV)."""
    final_params = {**params, "n_estimators": n_estimators}
    model = LGBMClassifier(**final_params)
    model.fit(X, y)
    return model


def print_top_features(model: LGBMClassifier, feature_names: list[str], top_n: int = 20) -> None:
    """Print ranked feature importances by gain."""
    gain = model.booster_.feature_importance(importance_type="gain")
    ranked = sorted(zip(feature_names, gain), key=lambda x: x[1], reverse=True)

    print(f"\nTop {top_n} features by importance (gain):")
    for rank, (name, score) in enumerate(ranked[:top_n], start=1):
        print(f"  {rank:2d}. {name:<35} {score:.1f}")


def save_feature_importance(model: LGBMClassifier, feature_names: list[str]) -> None:
    """Write full gain and split importances to outputs/feature_importance.csv."""
    gain = model.booster_.feature_importance(importance_type="gain")
    split = model.booster_.feature_importance(importance_type="split")
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_gain": gain,
            "importance_split": split,
        }
    ).sort_values("importance_gain", ascending=False)
    out_path = OUTPUTS_DIR / "feature_importance.csv"
    df.to_csv(out_path, index=False)
    print(f"Feature importance saved to {out_path.relative_to(ROOT)}")


def main() -> None:
    t0 = time.time()

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(LGBM_MODEL_PATH.parent, exist_ok=True)

    # --- Load preprocessed training data ---
    X_train, y_train = load_training_data()
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    scale_pos_weight = compute_scale_pos_weight(y_train)
    print(f"scale_pos_weight: {scale_pos_weight:.6f}")

    params = build_params(scale_pos_weight)

    # --- Stratified K-fold CV with early stopping ---
    fold_metrics, best_iterations = run_cross_validation(X_train, y_train, params)
    print_cv_summary(fold_metrics)

    mean_best_iter = int(round(np.mean(best_iterations)))
    print(f"\nMean best iteration across folds: {mean_best_iter}")

    # --- Final model on full training set (no early stopping) ---
    final_model = fit_final_model(X_train, y_train, params, n_estimators=mean_best_iter)
    joblib.dump(final_model, LGBM_MODEL_PATH)
    print(f"\nFinal model saved to {LGBM_MODEL_PATH.relative_to(ROOT)}")

    # --- Feature importance ---
    feature_names = list(X_train.columns)
    print_top_features(final_model, feature_names)
    save_feature_importance(final_model, feature_names)

    elapsed = time.time() - t0
    print(f"\nTotal training time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
