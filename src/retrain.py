#!/usr/bin/env python3
"""
Retrain multiclass LightGBM with Optuna trial 66 params and register as a new run.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from inference import (
    build_test_matrix,
    grades_to_submission_df,
    predict_multiclass,
    print_grade_distribution,
)
from preprocess import DATA_DIR
from run_manager import PROCESSED_DIR, ROOT, RunManager

RANDOM_STATE = 42
CV_FOLDS = 5
EARLY_STOPPING_ROUNDS = 50
FEATURE_SET = "baseline_61_features"
EMBEDDED_FEATURE_SET = "embedded_192_features"

# Optuna trial 66 — best of 97-trial study (3-fold search micro F1 ≈ 0.7407)
TRIAL_66_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 200,
    "max_depth": 11,
    "min_child_samples": 35,
    "learning_rate": 0.022600692673331986,
    "n_estimators": 1909,
    "feature_fraction": 0.6437872058657726,
    "bagging_fraction": 0.6858442495559679,
    "bagging_freq": 1,
    "reg_alpha": 5.577336583845032e-07,
    "reg_lambda": 2.4266765087309108e-08,
    "min_split_gain": 0.3279274666048547,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}

RUN_001_ID = "run_001"
RUN_002_ID = "run_002"
RUN_003_ID = "run_003"
RUN_004_ID = "run_004"

BASELINE_MULTICLASS_PARAMS = {
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


def run_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    params: dict,
    n_splits: int = CV_FOLDS,
) -> tuple[list[float], list[int]]:
    """Stratified CV with early stopping; return per-fold micro F1 and best iterations."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores: list[float] = []
    best_iters: list[int] = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        model = LGBMClassifier(**params)
        model.fit(
            X.iloc[train_idx],
            y[train_idx],
            eval_set=[(X.iloc[val_idx], y[val_idx])],
            eval_metric="multi_logloss",
            callbacks=[
                early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
            ],
        )
        best_iters.append(model.best_iteration_)
        y_pred = model.predict(X.iloc[val_idx])
        fold_f1 = f1_score(y[val_idx], y_pred, average="micro", labels=[1, 2, 3])
        scores.append(fold_f1)
        print(f"    fold {fold}: {fold_f1:.4f}")

    return scores, best_iters


def save_feature_importance(run_id: str, model: LGBMClassifier, feature_names: list[str], rm: RunManager) -> None:
    """Export gain/split importances for the run."""
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)
    rm.save_feature_importance(run_id, df)


def migrate_run_001(rm: RunManager) -> None:
    """Register baseline multiclass run from existing submission and known scores."""
    if rm.run_path(RUN_001_ID).exists():
        print(f"Migration: {RUN_001_ID} already exists — skipping.")
        return

    rm.create_run(
        description="baseline multiclass LightGBM default params",
        model_type="LightGBM",
        feature_set=FEATURE_SET,
        params=BASELINE_MULTICLASS_PARAMS,
        run_id=RUN_001_ID,
        objective="multiclass",
        n_features=61,
        cv_folds=5,
        cv_metric="micro_f1",
        notes="first public submission; model.pkl not archived (retrain to reproduce)",
    )

    legacy_sub = ROOT / "outputs" / "submission_multiclass.csv"
    if legacy_sub.exists():
        shutil.copy(legacy_sub, rm.run_path(RUN_001_ID) / "submission.csv")

    legacy_fi = ROOT / "outputs" / "feature_importance.csv"
    if legacy_fi.exists():
        shutil.copy(legacy_fi, rm.run_path(RUN_001_ID) / "feature_importance.csv")

    rm.save_cv_scores(RUN_001_ID, [], mean=0.7417, std=0.0019)
    meta = rm.load_metadata(RUN_001_ID)
    meta["notes"] = "CV from predict.py 5-fold; per-fold scores not archived"
    meta["cv_scores_per_fold"] = None
    RunManager._write_json(rm.run_path(RUN_001_ID) / "metadata.json", meta)

    rm.update_public_score(RUN_001_ID, 0.7370, submitted=True, submission_date="2026-05-19")
    print(f"Migration: created {RUN_001_ID} (public score 0.7370, CV 0.7417)")


def train_multiclass_run(
    rm: RunManager,
    run_id: str,
    *,
    x_train_path: Path | None = None,
    x_test_path: Path | None = None,
    y_train_path: Path | None = None,
    description: str | None = None,
    feature_set: str | None = None,
) -> str:
    """Train trial 66 params, CV, full fit, test submission for any feature matrix."""
    run_dir = rm.run_path(run_id)
    model_path = run_dir / "model.pkl"
    cv_path = run_dir / "cv_scores.json"

    x_train_file = x_train_path or (PROCESSED_DIR / "X_train.csv")
    y_train_file = y_train_path or (PROCESSED_DIR / "y_train_multiclass.csv")
    X_train = pd.read_csv(x_train_file)
    y_train = pd.read_csv(y_train_file)["damage_grade"].to_numpy()

    fs = feature_set or (
        EMBEDDED_FEATURE_SET if x_train_path else FEATURE_SET
    )
    desc = description or (
        "Entity embeddings + geo rates + interactions"
        if run_id == RUN_003_ID
        else "Optuna trial 66 best params, 99 trials, 3-fold search"
    )

    if not run_dir.exists():
        rm.create_run(
            description=desc,
            model_type="LightGBM",
            feature_set=fs,
            params=TRIAL_66_PARAMS,
            run_id=run_id,
            objective="multiclass",
            n_features=X_train.shape[1],
            cv_folds=CV_FOLDS,
            cv_metric="micro_f1",
            notes=f"Retrained from {x_train_file.name}",
        )

    n_estimators = TRIAL_66_PARAMS["n_estimators"]
    if cv_path.exists():
        import json

        with cv_path.open(encoding="utf-8") as f:
            cv_data = json.load(f)
        print(f"\nUsing saved CV scores for {run_id}: {cv_data['mean']:.4f} ± {cv_data['std']:.4f}")
        n_estimators = TRIAL_66_PARAMS["n_estimators"]
    else:
        print(f"\n{run_id} {CV_FOLDS}-fold CV:")
        fold_scores, best_iters = run_cv(X_train, y_train, TRIAL_66_PARAMS)
        mean_f1 = float(np.mean(fold_scores))
        std_f1 = float(np.std(fold_scores, ddof=1))
        print(f"  mean micro F1: {mean_f1:.4f} ± {std_f1:.4f}")
        rm.save_cv_scores(run_id, fold_scores, mean_f1, std_f1)
        n_estimators = int(round(np.mean(best_iters)))

    if model_path.exists() and model_path.stat().st_size < 100_000:
        model_path.unlink()

    if not model_path.exists() or model_path.stat().st_size < 100_000:
        print(f"  Refitting on full train ({n_estimators} trees)...")
        model = LGBMClassifier(**{**TRIAL_66_PARAMS, "n_estimators": n_estimators})
        model.fit(X_train, y_train)
        rm.save_model(run_id, model)
        save_feature_importance(run_id, model, list(X_train.columns), rm)
    else:
        import joblib

        model = joblib.load(model_path)
        print(f"{run_id} model.pkl OK — skipping fit.")

    if x_test_path:
        building_ids = pd.read_csv(DATA_DIR / "test_values.csv")["building_id"].values
        X_test = pd.read_csv(x_test_path)
    else:
        building_ids, X_test = build_test_matrix()

    grades = predict_multiclass(model, X_test)
    rm.save_submission(run_id, grades_to_submission_df(building_ids, grades))
    print_grade_distribution(grades, f"{run_id} submission grade distribution")
    return run_id


def train_run_002(rm: RunManager) -> str:
    """Train trial 66 on baseline 61-feature matrix → run_002."""
    return train_multiclass_run(rm, RUN_002_ID)


def migrate_optuna_outputs() -> None:
    """Move legacy Optuna artifacts into outputs/optuna/."""
    optuna_dir = ROOT / "outputs" / "optuna"
    optuna_dir.mkdir(parents=True, exist_ok=True)
    for name in ("optuna_study.db", "optuna_history.png", "best_params.json"):
        src = ROOT / "outputs" / name
        dst = optuna_dir / name
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
            print(f"Moved {name} → outputs/optuna/")


def cleanup_legacy_artifacts() -> None:
    """Remove duplicate model paths outside runs/ and models/preprocessing/."""
    to_remove = [
        ROOT / "models" / "lightGBM",
        ROOT / "models" / "lgbm_model.pkl",
        ROOT / "models" / "lgbm_tuned.pkl",
        ROOT / "models" / "scaler.pkl",
        ROOT / "models" / "target_encoder.pkl",
        ROOT / "outputs" / "submission_multiclass.csv",
        ROOT / "outputs" / "submission_tuned.csv",
        ROOT / "outputs" / "submission_binary_mapped.csv",
        ROOT / "outputs" / "feature_importance.csv",
        ROOT / "outputs" / "best_checkpoint.json",
    ]
    for path in to_remove:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            print(f"Removed directory {path.relative_to(ROOT)}")
        elif path.exists():
            path.unlink()
            print(f"Removed {path.relative_to(ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain multiclass LightGBM")
    parser.add_argument("--run-id", type=str, default=None, help="e.g. run_003")
    parser.add_argument("--x-train", type=str, default=None, help="Path to X_train CSV")
    parser.add_argument("--x-test", type=str, default=None, help="Path to X_test CSV")
    parser.add_argument("--y-train", type=str, default=None, help="Path to y_train CSV (damage_grade column)")
    parser.add_argument("--description", type=str, default=None, help="Run description")
    return parser.parse_args()


def main() -> None:
    t0 = time.time()
    args = parse_args()
    rm = RunManager()

    if args.run_id:
        run_id = args.run_id
        print("=" * 60)
        print(f"Retrain {run_id}")
        print("=" * 60)
        train_multiclass_run(
            rm,
            run_id,
            x_train_path=Path(args.x_train) if args.x_train else None,
            x_test_path=Path(args.x_test) if args.x_test else None,
            y_train_path=Path(args.y_train) if args.y_train else None,
            description=args.description,
        )
        rm.print_all_runs()
        print(f"\nTotal wall-clock time: {time.time() - t0:.1f} seconds")
        return

    print("=" * 60)
    print("Artifact migration & run_002 retrain")
    print("=" * 60)

    migrate_optuna_outputs()
    cleanup_legacy_artifacts()
    migrate_run_001(rm)
    train_run_002(rm)
    rm.update_best_run(recommended_run=RUN_002_ID)
    rm.print_all_runs()

    elapsed = time.time() - t0
    print(f"\nTotal wall-clock time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
