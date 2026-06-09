#!/usr/bin/env python3
"""
Two-stage multiclass ensemble — stacking, blending, and evaluation report.

Reads OOF / test probability arrays from the ensemble base run produced by
src/train_ensemble.py, then:
  1. Trains a multinomial logistic regression meta-model (stacker) on the
     concatenated OOF probabilities using 5-fold CV (same fold assignments as
     base models → no leakage).
  2. Optionally trains a small LightGBM stacker as an alternative.
  3. Computes a uniform weighted blend as a simple fallback.
  4. Prints an evaluation report comparing all approaches (including historical
     single-model and run_003-style LightGBM scores from existing run metadata).
  5. Registers a new run via RunManager and saves two submissions:
       submission_stacked.csv  — from the logistic meta-model
       submission_blend.csv    — from the uniform-weight blend
       submission.csv          — best of the two (stacked, unless overfit)

Usage
-----
    python src/stack.py --base-run-id run_005
    python src/stack.py --base-run-id run_005 --meta-model lgbm
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold

from features import GRADES, load_run003_features
from inference import grades_to_submission_df, print_grade_distribution
from run_manager import ROOT, RUNS_DIR, RunManager

print = partial(print, flush=True)

RANDOM_STATE = 42
CV_FOLDS = 5

# ── OOF / test proba loaders ──────────────────────────────────────────────────


def load_base_artifacts(
    base_run_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    """Load all oof_proba_*.npy and test_proba_*.npy from a base run directory.

    Returns
    -------
    oof_probas  : {model_name: (n_train, 3)}
    test_probas : {model_name: (n_test,  3)}
    model_names : sorted list of available model names
    """
    oof_probas: dict[str, np.ndarray] = {}
    test_probas: dict[str, np.ndarray] = {}

    for p in sorted(base_run_dir.glob("oof_proba_*.npy")):
        name = p.stem.replace("oof_proba_", "")
        test_path = base_run_dir / f"test_proba_{name}.npy"
        if test_path.exists():
            oof_probas[name] = np.load(p)
            test_probas[name] = np.load(test_path)

    if not oof_probas:
        raise FileNotFoundError(
            f"No oof_proba_*.npy files found in {base_run_dir}.\n"
            "Run src/train_ensemble.py first."
        )

    model_names = sorted(oof_probas)
    return oof_probas, test_probas, model_names


# ── Stacking helpers ──────────────────────────────────────────────────────────


def build_stacking_matrix(
    proba_dict: dict[str, np.ndarray], model_names: list[str]
) -> np.ndarray:
    """Concatenate OOF/test proba arrays horizontally."""
    return np.hstack([proba_dict[n] for n in model_names])


def proba_to_grades(proba: np.ndarray) -> np.ndarray:
    """Argmax → damage grade label 1/2/3."""
    return np.array(GRADES)[np.argmax(proba, axis=1)]


def grades_micro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="micro", labels=GRADES))


# ── Meta-model training ───────────────────────────────────────────────────────


def train_logreg_stacker(
    S_train: np.ndarray, y_train: np.ndarray, n_splits: int = CV_FOLDS
) -> tuple[LogisticRegression, list[float], np.ndarray]:
    """5-fold CV of logistic regression stacker.

    Returns fitted meta-model, per-fold F1 scores, and OOF predictions.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    meta = LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE, solver="lbfgs")
    oof_pred = np.zeros(len(y_train), dtype=int)
    scores: list[float] = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(S_train, y_train), start=1):
        m = clone(meta)
        m.fit(S_train[tr_idx], y_train[tr_idx])
        val_pred = m.predict(S_train[val_idx])
        oof_pred[val_idx] = val_pred
        f1 = grades_micro_f1(y_train[val_idx], val_pred)
        scores.append(f1)
        print(f"    stacker fold {fold}/{n_splits}: {f1:.4f}")

    meta.fit(S_train, y_train)
    return meta, scores, oof_pred


def train_lgbm_stacker(
    S_train: np.ndarray, y_train: np.ndarray, n_splits: int = CV_FOLDS
) -> tuple[Any, list[float], np.ndarray]:
    """5-fold CV of a small LightGBM meta-model."""
    from lightgbm import LGBMClassifier, early_stopping

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "num_leaves": 16,
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 300,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        "verbose": -1,
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_pred = np.zeros(len(y_train), dtype=int)
    scores: list[float] = []
    best_iters: list[int] = []
    S_df = pd.DataFrame(S_train)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(S_train, y_train), start=1):
        model = LGBMClassifier(**params)
        model.fit(
            S_df.iloc[tr_idx], y_train[tr_idx],
            eval_set=[(S_df.iloc[val_idx], y_train[val_idx])],
            eval_metric="multi_logloss",
            callbacks=[early_stopping(stopping_rounds=20, verbose=False)],
        )
        best_iters.append(model.best_iteration_)
        val_pred = model.predict(S_df.iloc[val_idx])
        oof_pred[val_idx] = val_pred
        f1 = grades_micro_f1(y_train[val_idx], val_pred)
        scores.append(f1)
        print(f"    lgbm-stacker fold {fold}/{n_splits}: {f1:.4f}")

    mean_bi = int(round(float(np.mean(best_iters))))
    final = LGBMClassifier(**{**params, "n_estimators": mean_bi})
    final.fit(S_df, y_train)
    return final, scores, oof_pred


# ── Blend helpers ─────────────────────────────────────────────────────────────


def blend_proba(
    proba_dict: dict[str, np.ndarray],
    model_names: list[str],
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Weighted average of class probability arrays."""
    if weights is None:
        w = np.ones(len(model_names)) / len(model_names)
    else:
        w = np.array([weights.get(n, 1.0) for n in model_names], dtype=float)
        w /= w.sum()
    return sum(float(w[i]) * proba_dict[n] for i, n in enumerate(model_names))


def blend_oof_cv(
    oof_probas: dict[str, np.ndarray],
    y_train: np.ndarray,
    model_names: list[str],
) -> tuple[float, np.ndarray]:
    """Compute OOF micro F1 for the uniform blend and return blend predictions."""
    blended = blend_proba(oof_probas, model_names)
    pred = proba_to_grades(blended)
    return grades_micro_f1(y_train, pred), pred


# ── Evaluation report ─────────────────────────────────────────────────────────


def load_historical_cv(rm: RunManager, run_ids: list[str]) -> dict[str, float]:
    """Load cv_mean for a list of historical run_ids that exist."""
    out: dict[str, float] = {}
    for rid in run_ids:
        try:
            meta = rm.load_metadata(rid)
            v = meta.get("cv_mean")
            if v is not None:
                out[rid] = float(v)
        except Exception:
            pass
    return out


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Per-grade F1, precision, recall."""
    report = classification_report(
        y_true, y_pred, labels=GRADES, output_dict=True, zero_division=0
    )
    out: dict[str, float] = {}
    for g in GRADES:
        key = str(g)
        out[f"g{g}_f1"] = report[key]["f1-score"]
        out[f"g{g}_recall"] = report[key]["recall"]
        out[f"g{g}_precision"] = report[key]["precision"]
    return out


def print_evaluation_report(
    historical: dict[str, float],
    base_model_cv: dict[str, dict],
    stacker_scores: list[float],
    stacker_oof_pred: np.ndarray,
    blend_f1: float,
    blend_oof_pred: np.ndarray,
    oof_probas: dict[str, np.ndarray],
    y_train: np.ndarray,
    model_names: list[str],
) -> None:
    """Print full comparison table with per-class breakdown."""
    print("\n" + "=" * 70)
    print("ENSEMBLE EVALUATION REPORT")
    print("=" * 70)

    # Historical single-model baselines
    if historical:
        print("\n── Historical single-model baselines ──────────────────────────────────")
        for rid, cv in sorted(historical.items()):
            print(f"  {rid:<12} CV micro F1 = {cv:.4f}")

    # Per-model base learner CV
    print("\n── Base learner CV (run_003 feature matrix) ───────────────────────────")
    print(f"  {'Model':<12} {'mean':>8}  {'std':>7}  {'g1_rec':>8}  {'g2_rec':>8}  {'g3_rec':>8}")
    print(f"  {'-'*12}  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}")
    for name in sorted(base_model_cv, key=lambda n: -base_model_cv[n]["mean"]):
        m = base_model_cv[name]
        oof_pred = proba_to_grades(oof_probas[name])
        pm = per_class_metrics(y_train, oof_pred)
        print(
            f"  {name:<12} {m['mean']:>8.4f}  {m['std']:>7.4f}"
            f"  {pm['g1_recall']:>8.3f}  {pm['g2_recall']:>8.3f}  {pm['g3_recall']:>8.3f}"
        )

    # Stacker
    stk_mean = float(np.mean(stacker_scores))
    stk_std = float(np.std(stacker_scores, ddof=1))
    stk_pm = per_class_metrics(y_train, stacker_oof_pred)
    print("\n── Stacking layer ─────────────────────────────────────────────────────")
    print(
        f"  {'stacker':<12} {stk_mean:>8.4f}  {stk_std:>7.4f}"
        f"  {stk_pm['g1_recall']:>8.3f}  {stk_pm['g2_recall']:>8.3f}  {stk_pm['g3_recall']:>8.3f}"
    )

    # Blend
    blend_pm = per_class_metrics(y_train, blend_oof_pred)
    print(
        f"  {'blend(uniform)':<12} {blend_f1:>8.4f}  {'—':>7}"
        f"  {blend_pm['g1_recall']:>8.3f}  {blend_pm['g2_recall']:>8.3f}  {blend_pm['g3_recall']:>8.3f}"
    )

    # Best run summary
    best_base = max(base_model_cv.items(), key=lambda kv: kv[1]["mean"])
    best_overall = max(
        [("stacked", stk_mean), ("blend", blend_f1), (best_base[0], best_base[1]["mean"])],
        key=lambda kv: kv[1],
    )
    print("\n── Summary ────────────────────────────────────────────────────────────")
    print(f"  Best base learner : {best_base[0]}  (CV {best_base[1]['mean']:.4f})")
    print(f"  Stacked ensemble  : CV {stk_mean:.4f}")
    print(f"  Weighted blend    : CV {blend_f1:.4f}")
    print(f"  Winner            : {best_overall[0]}  (CV {best_overall[1]:.4f})")
    print("=" * 70)


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stacking / blending over ensemble base learners")
    parser.add_argument("--base-run-id", type=str, default=None,
                        help="Run ID of the ensemble_base artifacts (default: auto-detect)")
    parser.add_argument("--meta-model", choices=["logreg", "lgbm"], default="logreg",
                        help="Meta-model type (default: logreg)")
    return parser.parse_args()


def _find_base_run(rm: RunManager, explicit: str | None) -> str:
    if explicit:
        return explicit
    for rid in reversed(rm.list_runs()):
        try:
            meta = rm.load_metadata(rid)
        except Exception:
            continue
        if meta.get("model_type") == "ensemble_base":
            return rid
    raise FileNotFoundError(
        "No ensemble_base run found. Run src/train_ensemble.py first, "
        "or pass --base-run-id explicitly."
    )


def main() -> None:
    t0 = time.time()
    args = parse_args()
    rm = RunManager()

    base_run_id = _find_base_run(rm, args.base_run_id)
    base_run_dir = rm.run_path(base_run_id)
    print(f"Base run: {base_run_id}  ({base_run_dir})")

    # Load features and base artifacts
    X_train, y_train, X_test, building_ids = load_run003_features()
    oof_probas, test_probas, model_names = load_base_artifacts(base_run_dir)
    print(f"Base models available: {model_names}")

    # Load per-model CV metrics from base run metadata
    base_meta = rm.load_metadata(base_run_id)
    base_model_cv: dict[str, dict] = base_meta.get("per_model_cv", {})
    # Patch in OOF-derived scores for any models that are loaded but not in metadata
    for name in model_names:
        if name not in base_model_cv:
            oof_pred = proba_to_grades(oof_probas[name])
            f1 = grades_micro_f1(y_train, oof_pred)
            base_model_cv[name] = {"mean": f1, "std": 0.0, "per_fold": []}

    # Build stacking matrices
    S_train = build_stacking_matrix(oof_probas, model_names)
    S_test = build_stacking_matrix(test_probas, model_names)
    print(f"Stacking feature matrix: train {S_train.shape}  test {S_test.shape}")

    # ── Train meta-model ──────────────────────────────────────────────────────
    print(f"\n── {args.meta_model.upper()} stacker CV ──")
    if args.meta_model == "logreg":
        meta_model, stacker_scores, stacker_oof_pred = train_logreg_stacker(S_train, y_train)
    else:
        meta_model, stacker_scores, stacker_oof_pred = train_lgbm_stacker(S_train, y_train)

    stk_mean = float(np.mean(stacker_scores))
    stk_std = float(np.std(stacker_scores, ddof=1))
    print(f"\nStacker CV micro F1: {stk_mean:.4f} ± {stk_std:.4f}")

    # ── Weighted blend ────────────────────────────────────────────────────────
    print("\n── Uniform-weight blend (OOF eval) ──")
    blend_f1, blend_oof_pred = blend_oof_cv(oof_probas, y_train, model_names)
    print(f"Blend OOF micro F1: {blend_f1:.4f}")

    # ── Evaluation report ─────────────────────────────────────────────────────
    historical = load_historical_cv(rm, ["run_001", "run_002", "run_003", "run_004"])
    print_evaluation_report(
        historical, base_model_cv, stacker_scores, stacker_oof_pred,
        blend_f1, blend_oof_pred, oof_probas, y_train, model_names,
    )

    # ── Generate test predictions ─────────────────────────────────────────────
    if args.meta_model == "logreg":
        stacked_proba = meta_model.predict_proba(S_test)
        # align columns: LogReg classes_ might be [1,2,3] in order already
        from train_ensemble import align_proba
        stacked_grades = proba_to_grades(
            align_proba(stacked_proba, meta_model.classes_)
        )
    else:
        stacked_grades = np.array(GRADES)[
            np.argmax(meta_model.predict_proba(pd.DataFrame(S_test)), axis=1)
        ]

    blend_test_proba = blend_proba(test_probas, model_names)
    blend_grades = proba_to_grades(blend_test_proba)

    # ── Register stacking run ─────────────────────────────────────────────────
    stack_run_id = rm.get_next_run_id()
    rm.create_run(
        description=f"Stacked ensemble ({args.meta_model} meta) over {base_run_id}",
        model_type=f"ensemble_stack_{args.meta_model}",
        feature_set="oof_proba_stacking",
        params={
            "base_run_id": base_run_id,
            "meta_model": args.meta_model,
            "base_models": model_names,
        },
        run_id=stack_run_id,
        objective="multiclass",
        n_features=S_train.shape[1],
        cv_folds=CV_FOLDS,
        cv_metric="micro_f1",
        notes=f"Stacker on OOF proba from {base_run_id}",
    )

    # Save meta-model and both submissions
    stack_dir = rm.run_path(stack_run_id)
    joblib.dump(meta_model, stack_dir / "meta_model.pkl")

    sub_stacked = grades_to_submission_df(building_ids, stacked_grades)
    sub_blend = grades_to_submission_df(building_ids, blend_grades)
    sub_stacked.to_csv(stack_dir / "submission_stacked.csv", index=False)
    sub_blend.to_csv(stack_dir / "submission_blend.csv", index=False)

    # Primary submission = stacker if it beats blend, else blend
    if stk_mean >= blend_f1:
        primary_grades = stacked_grades
        primary_label = "stacked"
    else:
        primary_grades = blend_grades
        primary_label = "blend"
    rm.save_submission(stack_run_id, grades_to_submission_df(building_ids, primary_grades))

    rm.save_cv_scores(stack_run_id, stacker_scores, stk_mean, stk_std)

    # Persist blend score in metadata
    stack_meta = rm.load_metadata(stack_run_id)
    stack_meta["blend_oof_f1"] = blend_f1
    stack_meta["primary_submission"] = primary_label
    stack_meta["stacker_scores_per_fold"] = [float(s) for s in stacker_scores]
    RunManager._write_json(stack_dir / "metadata.json", stack_meta)

    print_grade_distribution(primary_grades, f"\n{stack_run_id} ({primary_label}) grade distribution")

    rm.update_best_run()
    rm.print_all_runs()
    print(f"\nStack run saved: runs/{stack_run_id}/")
    print(f"  submission_stacked.csv  (meta={args.meta_model}, CV {stk_mean:.4f})")
    print(f"  submission_blend.csv    (uniform blend, OOF {blend_f1:.4f})")
    print(f"  submission.csv          → {primary_label} (selected as primary)")
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
