# pip install lightgbm if missing
"""
Train a LightGBM binary classifier for Nepal earthquake building survival.

Uses stratified 5-fold CV for evaluation, then refits on full X_train.
Artifacts are stored under runs/run_NNN/ via RunManager.
"""

from __future__ import annotations

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

from run_manager import PROCESSED_DIR, RunManager

N_SPLITS = 5
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 50
FEATURE_SET = "baseline_61_features"

METRIC_NAMES = ("f1_macro", "auc_roc", "accuracy", "precision", "recall")


def load_training_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Load preprocessed features and binary survived labels."""
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")["survived"].to_numpy()
    return X_train, y_train


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """Weight positive class: count(0) / count(1)."""
    return float(np.sum(y == 0) / np.sum(y == 1))


def build_params(scale_pos_weight: float) -> dict:
    """Default LightGBM hyperparameters for binary classification."""
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


def evaluate_fold(y_true, y_pred, y_proba) -> dict[str, float]:
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
    """Stratified 5-fold CV with early stopping."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = {name: [] for name in METRIC_NAMES}
    best_iterations: list[int] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        print("=" * 60)
        print(f"Fold {fold_idx} / {N_SPLITS}")
        print("=" * 60)

        model = LGBMClassifier(**params)
        model.fit(
            X.iloc[train_idx],
            y[train_idx],
            eval_set=[(X.iloc[val_idx], y[val_idx])],
            eval_metric="auc",
            callbacks=[early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        best_iterations.append(model.best_iteration_)
        y_proba = model.predict_proba(X.iloc[val_idx])[:, 1]
        metrics = evaluate_fold(y[val_idx], (y_proba >= 0.5).astype(int), y_proba)
        for name in METRIC_NAMES:
            fold_metrics[name].append(metrics[name])
            print(f"  {name}: {metrics[name]:.4f}")

    return fold_metrics, best_iterations


def print_cv_summary(fold_metrics: dict[str, list[float]]) -> None:
    labels = {
        "f1_macro": "F1 macro",
        "auc_roc": "AUC-ROC",
        "accuracy": "Accuracy",
        "precision": "Precision (macro)",
        "recall": "Recall (macro)",
    }
    print("\n" + "=" * 60)
    print("Cross-validation summary (mean ± std)")
    print("=" * 60)
    for name in METRIC_NAMES:
        values = fold_metrics[name]
        print(f"  {labels[name]}: {np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}")


def main() -> None:
    t0 = time.time()
    rm = RunManager()
    X_train, y_train = load_training_data()
    print(f"X_train shape: {X_train.shape}")

    scale_pos_weight = compute_scale_pos_weight(y_train)
    params = build_params(scale_pos_weight)

    run_id, _ = rm.create_run(
        description="binary survived LightGBM default params",
        model_type="LightGBM",
        feature_set=FEATURE_SET,
        params=params,
        objective="binary",
        n_features=X_train.shape[1],
        cv_folds=N_SPLITS,
        cv_metric="f1_macro",
    )

    fold_metrics, best_iterations = run_cross_validation(X_train, y_train, params)
    print_cv_summary(fold_metrics)

    mean_best_iter = int(round(np.mean(best_iterations)))
    final_model = LGBMClassifier(**{**params, "n_estimators": mean_best_iter})
    final_model.fit(X_train, y_train)

    rm.save_model(run_id, final_model)
    rm.save_cv_scores(
        run_id,
        fold_metrics["f1_macro"],
        float(np.mean(fold_metrics["f1_macro"])),
        float(np.std(fold_metrics["f1_macro"], ddof=1)),
    )

    gain = final_model.booster_.feature_importance(importance_type="gain")
    split = final_model.booster_.feature_importance(importance_type="split")
    rm.save_feature_importance(
        run_id,
        pd.DataFrame(
            {
                "feature": list(X_train.columns),
                "importance_gain": gain,
                "importance_split": split,
            }
        ).sort_values("importance_gain", ascending=False),
    )

    rm.update_best_run()
    print(f"\nRun saved: runs/{run_id}/")
    rm.print_run_summary(run_id)
    print(f"\nTotal training time: {time.time() - t0:.1f} seconds")


if __name__ == "__main__":
    main()
