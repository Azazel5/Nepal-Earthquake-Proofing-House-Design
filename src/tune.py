# pip install optuna optuna[visualization] lightgbm
"""
Optuna hyperparameter tuning for multiclass LightGBM (damage_grade 1/2/3).

Uses SQLite study storage and a JSON checkpoint so runs can resume safely.
Only overwrites models/lgbm_tuned.pkl when a trial beats the checkpoint score.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from predict import (
    ARTIFACT_CANDIDATES,
    DATA_DIR,
    PROCESSED_DIR,
    load_artifact,
    load_train_age_median,
    preprocess_test,
    print_grade_distribution,
    save_submission,
)
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
CHECKPOINT_PATH = OUTPUTS_DIR / "best_checkpoint.json"
BEST_PARAMS_PATH = OUTPUTS_DIR / "best_params.json"
TUNED_MODEL_PATH = ROOT / "models" / "lgbm_tuned.pkl"
SUBMISSION_PATH = OUTPUTS_DIR / "submission_tuned.csv"
OPTUNA_DB = OUTPUTS_DIR / "optuna_study.db"
HISTORY_PLOT_PATH = OUTPUTS_DIR / "optuna_history.png"

RANDOM_STATE = 42
N_TRIALS = 100
CV_FOLDS_TUNING = 3
CV_FOLDS_FINAL = 5
EARLY_STOPPING_ROUNDS = 50
STUDY_NAME = "lgbm_nepal_v1"

BASELINE_SCORE = 0.7417
BASELINE_PARAMS: dict[str, Any] = {
    "num_leaves": 63,
    "max_depth": 8,
    "min_child_samples": 20,
    "learning_rate": 0.05,
    "n_estimators": 1000,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 1e-8,
    "reg_lambda": 1e-8,
    "min_split_gain": 0.0,
}

FIXED_LGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}


def default_checkpoint() -> dict[str, Any]:
    """Initial checkpoint before any tuning run (multiclass CV baseline)."""
    return {
        "best_score": BASELINE_SCORE,
        "best_params": BASELINE_PARAMS.copy(),
        "best_model_path": str(TUNED_MODEL_PATH.relative_to(ROOT)),
        "submission_path": str(SUBMISSION_PATH.relative_to(ROOT)),
        "trial_number": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def load_checkpoint() -> dict[str, Any]:
    """Load checkpoint JSON or initialize from baseline."""
    if not CHECKPOINT_PATH.exists():
        checkpoint = default_checkpoint()
        save_checkpoint(checkpoint)
        print(f"No checkpoint found — initialized baseline CV score = {BASELINE_SCORE:.4f}")
        return checkpoint

    try:
        with CHECKPOINT_PATH.open(encoding="utf-8") as f:
            checkpoint = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Could not parse {CHECKPOINT_PATH}. Delete or fix the file and retry."
        ) from exc

    print(f"Loaded checkpoint: best CV score so far = {checkpoint['best_score']:.4f}")
    return checkpoint


def save_checkpoint(checkpoint: dict[str, Any]) -> None:
    """Persist checkpoint state to disk."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    checkpoint["timestamp"] = datetime.now(timezone.utc).isoformat()
    with CHECKPOINT_PATH.open("w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)


def build_lgbm_classifier(trial_params: dict[str, Any]) -> LGBMClassifier:
    """Merge fixed multiclass settings with tunable hyperparameters."""
    return LGBMClassifier(**FIXED_LGBM_PARAMS, **trial_params)


def cv_micro_f1(
    X: pd.DataFrame,
    y: np.ndarray,
    trial_params: dict[str, Any],
    n_splits: int,
    trial: optuna.Trial | None = None,
) -> tuple[float, list[int], list[float]]:
    """Run stratified CV; return mean micro F1, best iterations, and per-fold scores."""
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    )
    scores: list[float] = []
    best_iters: list[int] = []

    for fold_i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = build_lgbm_classifier(trial_params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="multi_logloss",
            callbacks=[
                early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
            ],
        )
        best_iters.append(model.best_iteration_)

        y_pred = model.predict(X_val)
        score = f1_score(y_val, y_pred, average="micro", labels=[1, 2, 3])
        scores.append(score)

        if trial is not None:
            trial.report(score, fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return float(np.mean(scores)), best_iters, scores


def fit_full_model(
    X: pd.DataFrame, y: np.ndarray, trial_params: dict[str, Any], n_estimators: int
) -> LGBMClassifier:
    """Train on all rows with a fixed tree count (no early stopping)."""
    params = {**trial_params, "n_estimators": n_estimators}
    model = build_lgbm_classifier(params)
    model.fit(X, y)
    return model


def maybe_update_checkpoint(
    score: float,
    trial_params: dict[str, Any],
    trial_number: int,
    checkpoint: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_estimators: int,
) -> None:
    """Save model and checkpoint only when score strictly beats the previous best."""
    if score <= checkpoint["best_score"]:
        return

    previous = checkpoint["best_score"]
    print(f"New best! Trial {trial_number}: {score:.4f} (previous best: {previous:.4f})")

    os.makedirs(TUNED_MODEL_PATH.parent, exist_ok=True)
    model = fit_full_model(X_train, y_train, trial_params, n_estimators)
    joblib.dump(model, TUNED_MODEL_PATH)

    checkpoint["best_score"] = score
    checkpoint["best_params"] = trial_params.copy()
    checkpoint["trial_number"] = trial_number
    checkpoint["best_model_path"] = str(TUNED_MODEL_PATH.relative_to(ROOT))
    checkpoint["submission_path"] = str(SUBMISSION_PATH.relative_to(ROOT))
    save_checkpoint(checkpoint)


def sample_trial_params(trial: optuna.Trial) -> dict[str, Any]:
    """Optuna search space for LightGBM hyperparameters."""
    return {
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.3, log=True
        ),
        "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
    }


def create_objective(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    checkpoint: dict[str, Any],
):
    """Build Optuna objective with checkpoint updates after each trial."""

    def objective(trial: optuna.Trial) -> float:
        trial_params = sample_trial_params(trial)
        score, best_iters, _ = cv_micro_f1(
            X_train, y_train, trial_params, CV_FOLDS_TUNING, trial=trial
        )
        mean_n_estimators = int(round(np.mean(best_iters)))

        print(
            f"Trial {trial.number}: score={score:.4f}  "
            f"best so far={max(checkpoint['best_score'], score):.4f}"
        )

        maybe_update_checkpoint(
            score,
            trial_params,
            trial.number,
            checkpoint,
            X_train,
            y_train,
            mean_n_estimators,
        )

        if trial.number > 0 and trial.number % 10 == 0:
            print(f"  [checkpoint best params] {checkpoint['best_params']}")

        return score

    return objective


def save_optimization_plot(study: optuna.Study) -> None:
    """Save Optuna optimization history as PNG."""
    from optuna.visualization.matplotlib import plot_optimization_history

    fig = plot_optimization_history(study)
    fig.figure.savefig(HISTORY_PLOT_PATH, bbox_inches="tight", dpi=120)
    plt.close(fig.figure)
    print(f"Optimization history saved to {HISTORY_PLOT_PATH.relative_to(ROOT)}")


def generate_submission(model: LGBMClassifier) -> None:
    """Preprocess test data with saved artifacts and write submission CSV."""
    from predict import predict_multiclass

    scaler = load_artifact("scaler", ARTIFACT_CANDIDATES["scaler"])
    target_encoder = load_artifact(
        "target encoder", ARTIFACT_CANDIDATES["target_encoder"]
    )

    train_columns = list(pd.read_csv(PROCESSED_DIR / "X_train.csv").columns)
    test_raw = pd.read_csv(DATA_DIR / "test_values.csv")
    building_ids = test_raw["building_id"]
    test_features = test_raw.drop(columns=["building_id"])

    X_test = preprocess_test(
        test_features,
        target_encoder,
        scaler,
        train_columns,
        load_train_age_median(),
    )

    grades = predict_multiclass(model, X_test)
    unique = set(np.unique(grades))
    if not unique.issubset({1, 2, 3}):
        raise ValueError(f"Invalid damage_grade predictions: {unique}")

    save_submission(building_ids, grades, SUBMISSION_PATH)
    print_grade_distribution(pd.Series(grades), "Tuned submission grade distribution")
    print(f"Ready to submit: {SUBMISSION_PATH.relative_to(ROOT)}")


def main() -> None:
    t0 = time.time()
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(TUNED_MODEL_PATH.parent, exist_ok=True)

    # --- Checkpoint (resume-safe) ---
    try:
        checkpoint = load_checkpoint()
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # --- Training data ---
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train_multiclass.csv")[
        "damage_grade"
    ].to_numpy()
    print(f"X_train shape: {X_train.shape}")

    # --- Optuna study (SQLite persists across restarts) ---
    storage_url = f"sqlite:///{OPTUNA_DB.resolve()}"
    sampler = TPESampler(seed=RANDOM_STATE)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    # Warm-start: enqueue baseline only on a fresh study
    if len(study.trials) == 0:
        study.enqueue_trial(BASELINE_PARAMS)

    objective = create_objective(X_train, y_train, checkpoint)
    trials_to_run = max(0, N_TRIALS - len(study.trials))
    if trials_to_run > 0:
        print(f"Running {trials_to_run} Optuna trials ({len(study.trials)} already in study)...")
        study.optimize(objective, n_trials=trials_to_run, show_progress_bar=False)
    else:
        print(f"Study already has {len(study.trials)} trials — skipping optimization.")

    # --- Study summary ---
    print("\n" + "=" * 60)
    print("Optuna tuning complete")
    print("=" * 60)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best CV micro F1 (3-fold, in-study): {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    best_params_out = {
        "best_trial": study.best_trial.number,
        "best_cv_micro_f1_3fold": study.best_value,
        "best_params": study.best_params,
    }
    with BEST_PARAMS_PATH.open("w", encoding="utf-8") as f:
        json.dump(best_params_out, f, indent=2)
    print(f"Saved {BEST_PARAMS_PATH.relative_to(ROOT)}")

    save_optimization_plot(study)

    # --- Honest 5-fold estimate with study-best params ---
    print("\nFinal 5-fold evaluation (study best params):")
    mean_5, best_iters_5, cv5_scores = cv_micro_f1(
        X_train, y_train, study.best_params, CV_FOLDS_FINAL, trial=None
    )
    std_f1 = float(np.std(cv5_scores, ddof=1))
    print(f"  Micro F1 (5-fold): {mean_5:.4f} ± {std_f1:.4f}")

    n_estimators_final = int(round(np.mean(best_iters_5)))
    final_model = fit_full_model(
        X_train, y_train, study.best_params, n_estimators_final
    )

    # Save model only if 5-fold score beats checkpoint; always record 5-fold in checkpoint
    maybe_update_checkpoint(
        mean_5,
        study.best_params,
        study.best_trial.number,
        checkpoint,
        X_train,
        y_train,
        n_estimators_final,
    )
    checkpoint["final_5fold_micro_f1"] = mean_5
    checkpoint["final_5fold_std"] = std_f1
    checkpoint["best_params"] = study.best_params.copy()
    checkpoint["trial_number"] = study.best_trial.number
    save_checkpoint(checkpoint)

    if not TUNED_MODEL_PATH.exists():
        joblib.dump(final_model, TUNED_MODEL_PATH)
        print(
            f"No tuned model on disk yet — saved final fit to "
            f"{TUNED_MODEL_PATH.relative_to(ROOT)}"
        )

    # --- Submission from best saved model ---
    print("\nGenerating submission...")
    model = joblib.load(TUNED_MODEL_PATH)
    generate_submission(model)

    elapsed = time.time() - t0
    print(f"\nTotal wall-clock time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
