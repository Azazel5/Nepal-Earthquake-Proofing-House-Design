#!/usr/bin/env python3
"""
Optuna hyperparameter search for raw-categorical LGBM and CatBoost models
(run_014 ensemble-decorrelation experiment — see features_raw.py).

3-fold stratified CV per trial (matches the original trial_66 search protocol),
early stopping, micro F1 objective. Studies are persisted to SQLite so they
can be resumed / monitored.

Run from project root:
    python src/tune_raw_models.py --model lgbm --n-trials 30
    python src/tune_raw_models.py --model catboost --n-trials 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features_raw import RAW_CAT_COLS, build_raw_features

print = partial(print, flush=True)

RANDOM_STATE = 42
SEARCH_FOLDS = 3
ES_ROUNDS = 50
SEARCH_ITERATIONS = 2000

OPTUNA_DIR = ROOT / "outputs" / "optuna_raw"
OPTUNA_DIR.mkdir(parents=True, exist_ok=True)


# ── LGBM objective ───────────────────────────────────────────────────────────

def lgbm_objective(trial: optuna.Trial, X, y) -> float:
    from lightgbm import LGBMClassifier, early_stopping

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 63, 200),
        "max_depth": trial.suggest_int("max_depth", 6, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 500),
        "cat_smooth": trial.suggest_float("cat_smooth", 10, 200),
        "cat_l2": trial.suggest_float("cat_l2", 1, 50, log=True),
        "min_data_per_group": trial.suggest_int("min_data_per_group", 50, 500),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 1,
        "n_estimators": SEARCH_ITERATIONS,
        # NOTE: n_jobs=1 — LGBM 4.6.0 segfaults with categorical_feature +
        # multithreading on this platform. ~47s/fold single-threaded.
        "n_jobs": 1,
        "random_state": RANDOM_STATE,
        "verbose": -1,
    }

    skf = StratifiedKFold(n_splits=SEARCH_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for tr_idx, va_idx in skf.split(X, y):
        model = LGBMClassifier(**params)
        model.fit(
            X.iloc[tr_idx], y[tr_idx],
            eval_set=[(X.iloc[va_idx], y[va_idx])],
            eval_metric="multi_logloss",
            categorical_feature=RAW_CAT_COLS,
            callbacks=[early_stopping(stopping_rounds=ES_ROUNDS, verbose=False)],
        )
        preds = model.predict(X.iloc[va_idx])
        scores.append(f1_score(y[va_idx], preds, average="micro", labels=[1, 2, 3]))

    return float(np.mean(scores))


# ── CatBoost objective ────────────────────────────────────────────────────────

def catboost_objective(trial: optuna.Trial, X, y) -> float:
    from catboost import CatBoostClassifier

    params = {
        "loss_function": "MultiClass",
        "eval_metric": "Accuracy",
        "iterations": SEARCH_ITERATIONS,
        "depth": trial.suggest_int("depth", 6, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.5, 5.0),
        "one_hot_max_size": 10,
        "random_seed": RANDOM_STATE,
        "verbose": 0,
        "early_stopping_rounds": ES_ROUNDS,
        "thread_count": -1,
    }

    skf = StratifiedKFold(n_splits=SEARCH_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for tr_idx, va_idx in skf.split(X, y):
        model = CatBoostClassifier(**params)
        model.fit(
            X.iloc[tr_idx], y[tr_idx],
            eval_set=(X.iloc[va_idx], y[va_idx]),
            cat_features=RAW_CAT_COLS,
        )
        preds = model.predict(X.iloc[va_idx]).ravel()
        scores.append(f1_score(y[va_idx], preds, average="micro", labels=[1, 2, 3]))

    return float(np.mean(scores))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lgbm", "catboost"], required=True)
    parser.add_argument("--n-trials", type=int, default=30)
    args = parser.parse_args()

    t0 = time.time()
    print(f"── Loading raw feature matrix ──")
    X, X_test, y = build_raw_features()
    print(f"  X: {X.shape}")

    storage = f"sqlite:///{OPTUNA_DIR}/{args.model}_raw.db"
    study = optuna.create_study(
        study_name=f"{args.model}_raw",
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    n_done = len(study.trials)
    n_remaining = max(0, args.n_trials - n_done)
    print(f"  Study has {n_done} trials, running {n_remaining} more (target {args.n_trials})")

    if args.model == "lgbm":
        objective = partial(lgbm_objective, X=X, y=y)
    else:
        objective = partial(catboost_objective, X=X, y=y)

    if n_remaining > 0:
        study.optimize(objective, n_trials=n_remaining, show_progress_bar=False,
                       callbacks=[lambda study, trial: print(
                           f"    trial {trial.number:3d}: F1={trial.value:.4f}  "
                           f"(best so far: {study.best_value:.4f})")])

    print(f"\n  Best {args.model} (3-fold search): {study.best_value:.4f}")
    print(f"  Best params: {json.dumps(study.best_params, indent=2)}")

    out_path = OPTUNA_DIR / f"{args.model}_raw_best_params.json"
    with out_path.open("w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=2)
    print(f"  Saved → {out_path.relative_to(ROOT)}")
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
