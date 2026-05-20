# pip install optuna optuna[visualization] lightgbm
"""
Optuna hyperparameter tuning for multiclass LightGBM (damage_grade 1/2/3).

Study state: outputs/optuna/optuna_study.db
Register production runs via retrain.py / RunManager (runs/run_NNN/).
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from run_manager import OPTUNA_DIR, PROCESSED_DIR, RunManager

RANDOM_STATE = 42
N_TRIALS = 100
CV_FOLDS_TUNING = 3
CV_FOLDS_FINAL = 5
EARLY_STOPPING_ROUNDS = 50
STUDY_NAME = "lgbm_nepal_v1"
FEATURE_SET = "baseline_61_features"

CHECKPOINT_PATH = OPTUNA_DIR / "best_checkpoint.json"
BEST_PARAMS_PATH = OPTUNA_DIR / "best_params.json"
OPTUNA_DB = OPTUNA_DIR / "optuna_study.db"
HISTORY_PLOT_PATH = OPTUNA_DIR / "optuna_history.png"

# Measured 5-fold baseline (run_001); 3-fold trials compare against this floor
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
    return {
        "best_score": BASELINE_SCORE,
        "best_params": BASELINE_PARAMS.copy(),
        "trial_number": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "notes": "Floor = run_001 5-fold CV; trials use 3-fold micro F1",
    }


def load_checkpoint() -> dict[str, Any]:
    if not CHECKPOINT_PATH.exists():
        cp = default_checkpoint()
        save_checkpoint(cp)
        print(f"No checkpoint — initialized baseline CV floor = {BASELINE_SCORE:.4f}")
        return cp
    try:
        with CHECKPOINT_PATH.open(encoding="utf-8") as f:
            cp = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {CHECKPOINT_PATH}") from exc
    print(f"Loaded checkpoint: best CV score so far = {cp['best_score']:.4f}")
    return cp


def save_checkpoint(checkpoint: dict[str, Any]) -> None:
    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint["timestamp"] = datetime.now(timezone.utc).isoformat()
    with CHECKPOINT_PATH.open("w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)


def build_lgbm_classifier(trial_params: dict[str, Any]) -> LGBMClassifier:
    return LGBMClassifier(**FIXED_LGBM_PARAMS, **trial_params)


def cv_micro_f1(
    X: pd.DataFrame,
    y: np.ndarray,
    trial_params: dict[str, Any],
    n_splits: int,
    trial: optuna.Trial | None = None,
) -> tuple[float, list[int], list[float]]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores: list[float] = []
    best_iters: list[int] = []

    for fold_i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        model = build_lgbm_classifier(trial_params)
        model.fit(
            X.iloc[train_idx],
            y[train_idx],
            eval_set=[(X.iloc[val_idx], y[val_idx])],
            eval_metric="multi_logloss",
            callbacks=[early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        best_iters.append(model.best_iteration_)
        y_pred = model.predict(X.iloc[val_idx])
        score = f1_score(y[val_idx], y_pred, average="micro", labels=[1, 2, 3])
        scores.append(score)
        if trial is not None:
            trial.report(score, fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return float(np.mean(scores)), best_iters, scores


def sample_trial_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
    }


def create_objective(X_train: pd.DataFrame, y_train: np.ndarray, checkpoint: dict[str, Any]):
    def objective(trial: optuna.Trial) -> float:
        trial_params = sample_trial_params(trial)
        score, _, _ = cv_micro_f1(X_train, y_train, trial_params, CV_FOLDS_TUNING, trial=trial)
        print(
            f"Trial {trial.number}: score={score:.4f}  "
            f"best so far={max(checkpoint['best_score'], score):.4f}"
        )
        if score > checkpoint["best_score"]:
            previous = checkpoint["best_score"]
            print(f"New best! Trial {trial.number}: {score:.4f} (previous: {previous:.4f})")
            checkpoint["best_score"] = score
            checkpoint["best_params"] = trial_params.copy()
            checkpoint["trial_number"] = trial.number
            save_checkpoint(checkpoint)
        if trial.number > 0 and trial.number % 10 == 0:
            print(f"  [checkpoint best params] {checkpoint['best_params']}")
        return score

    return objective


def save_optimization_plot(study: optuna.Study) -> None:
    from optuna.visualization.matplotlib import plot_optimization_history

    fig = plot_optimization_history(study)
    fig.figure.savefig(HISTORY_PLOT_PATH, bbox_inches="tight", dpi=120)
    plt.close(fig.figure)
    print(f"Optimization history saved to {HISTORY_PLOT_PATH}")


def main() -> None:
    t0 = time.time()
    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        checkpoint = load_checkpoint()
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train_multiclass.csv")["damage_grade"].to_numpy()
    print(f"X_train shape: {X_train.shape}")

    storage_url = f"sqlite:///{OPTUNA_DB.resolve()}"
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_STATE),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    )

    if len(study.trials) == 0:
        study.enqueue_trial(BASELINE_PARAMS)

    trials_to_run = max(0, N_TRIALS - len(study.trials))
    if trials_to_run > 0:
        print(f"Running {trials_to_run} trials ({len(study.trials)} in study)...")
        study.optimize(
            create_objective(X_train, y_train, checkpoint),
            n_trials=trials_to_run,
            show_progress_bar=False,
        )
    else:
        print(f"Study has {len(study.trials)} trials — skipping optimization.")

    print("\n" + "=" * 60)
    print("Optuna tuning complete")
    print("=" * 60)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best 3-fold micro F1: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    with BEST_PARAMS_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_trial": study.best_trial.number,
                "best_cv_micro_f1_3fold": study.best_value,
                "best_params": study.best_params,
            },
            f,
            indent=2,
        )

    save_optimization_plot(study)

    print("\nFinal 5-fold evaluation (study best params):")
    mean_5, best_iters_5, cv5_scores = cv_micro_f1(
        X_train, y_train, study.best_params, CV_FOLDS_FINAL
    )
    std_f1 = float(np.std(cv5_scores, ddof=1))
    print(f"  Micro F1 (5-fold): {mean_5:.4f} ± {std_f1:.4f}")
    print(
        f"\nTo register as a production run, use src/retrain.py with these params "
        f"(see runs/run_002 for trial 66)."
    )

    RunManager().update_best_run()
    RunManager().print_all_runs()

    print(f"\nTotal wall-clock time: {time.time() - t0:.1f} seconds")


if __name__ == "__main__":
    main()
