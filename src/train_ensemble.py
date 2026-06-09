#!/usr/bin/env python3
"""
Two-stage multiclass ensemble — base learner training.

Trains five base models on the run_003-style embedded features using
stratified 5-fold CV.  Saves out-of-fold (OOF) class probabilities and
fold-averaged test probabilities for the stacking step (src/stack.py).

Base models
-----------
  lgbm      LightGBM with Optuna trial-66 params
  xgb       XGBoost with comparable depth/shrinkage
  catboost  CatBoost (skipped silently if not installed)
  et        ExtraTrees (non-boosted diversity)
  hgb       HistGradientBoosting (sklearn native, fast baseline)

Artifacts written to runs/run_NNN/ (auto-incremented, or --run-id)
---------------------------------------------------------------------------
  oof_proba_{model}.npy   (n_train, 3)  — OOF class probabilities
  test_proba_{model}.npy  (n_test,  3)  — test probabilities avg over folds
  model_{model}.pkl                     — full-data fitted model
  metadata.json                         — per_model_cv key with CV metrics

Usage
-----
    python src/train_ensemble.py                   # auto run-id, all models
    python src/train_ensemble.py --run-id run_005  # resume or reuse
    python src/train_ensemble.py --models lgbm xgb # subset
"""

from __future__ import annotations

import argparse
import time
from functools import partial
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier, early_stopping

from features import GRADES, load_run003_features
from retrain import TRIAL_66_PARAMS
from run_manager import ROOT, RunManager

print = partial(print, flush=True)

RANDOM_STATE = 42
CV_FOLDS = 5
EARLY_STOPPING_ROUNDS = 50

# ── Base model parameter blocks ───────────────────────────────────────────────

_XGB_BASE = {
    "objective": "multi:softprob",
    "n_estimators": 1500,
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "eval_metric": "mlogloss",
    "verbosity": 0,
}

_CATBOOST_BASE = {
    "iterations": 1500,
    "learning_rate": 0.05,
    "depth": 7,
    "l2_leaf_reg": 3,
    "loss_function": "MultiClass",
    "random_seed": RANDOM_STATE,
    "verbose": 0,
}

_ET_PARAMS = {
    "n_estimators": 500,
    "max_features": "sqrt",
    "min_samples_leaf": 4,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

_HGB_PARAMS = {
    "max_iter": 500,
    "learning_rate": 0.05,
    "max_leaf_nodes": 127,
    "min_samples_leaf": 20,
    "l2_regularization": 0.1,
    "random_state": RANDOM_STATE,
}

# ── Model factories ───────────────────────────────────────────────────────────


def _build_lgbm(n_iter: int | None = None) -> LGBMClassifier:
    p = dict(TRIAL_66_PARAMS)
    if n_iter is not None:
        p["n_estimators"] = n_iter
    return LGBMClassifier(**p)


def _build_xgb(n_iter: int | None = None, with_es: bool = True):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("Install xgboost: pip install xgboost") from exc
    p = dict(_XGB_BASE)
    if n_iter is not None:
        p["n_estimators"] = n_iter
    if with_es:
        p["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    return XGBClassifier(**p)


def _build_catboost(n_iter: int | None = None, with_es: bool = True):
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError("Install catboost: pip install catboost") from exc
    p = dict(_CATBOOST_BASE)
    if n_iter is not None:
        p["iterations"] = n_iter
    if with_es:
        p["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    return CatBoostClassifier(**p)


def _build_et(_n_iter: int | None = None, **_kw) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(**_ET_PARAMS)


def _build_hgb(_n_iter: int | None = None, **_kw) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**_HGB_PARAMS)


_BUILDERS = {
    "lgbm": _build_lgbm,
    "xgb": _build_xgb,
    "catboost": _build_catboost,
    "et": _build_et,
    "hgb": _build_hgb,
}

# Which models use an eval_set during fold fitting
_USES_EVAL = {"lgbm", "xgb", "catboost"}


def _fit_fold(name: str, model: Any, X_tr, y_tr, X_val, y_val) -> None:
    """Fit one fold, dispatching eval-set handling per model family."""
    if name == "lgbm":
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="multi_logloss",
            callbacks=[early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
        )
    elif name == "xgb":
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    elif name == "catboost":
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), silent=True)
    else:
        model.fit(X_tr, y_tr)


def _get_best_iter(name: str, model: Any) -> int | None:
    """Extract best iteration from a fitted model; None if unavailable."""
    bi = getattr(model, "best_iteration_", None)
    if bi is None:
        bi = getattr(model, "best_iteration", None)
    if bi is None and hasattr(model, "get_best_iteration"):
        try:
            bi = model.get_best_iteration()
        except Exception:
            pass
    return int(bi) if bi is not None else None


# ── Probability alignment ─────────────────────────────────────────────────────


def align_proba(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Map predict_proba output to fixed columns [P(grade=1), P(grade=2), P(grade=3)].

    Handles any class ordering returned by the model.
    """
    out = np.zeros((len(proba), 3), dtype=np.float64)
    for j, cls in enumerate(classes):
        out[:, int(cls) - 1] = proba[:, j]
    return out


# ── Main CV loop ──────────────────────────────────────────────────────────────


def train_model_cv(
    name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    n_splits: int = CV_FOLDS,
) -> tuple[np.ndarray, np.ndarray, list[float], int | None]:
    """Run stratified K-fold CV for one model.

    Returns
    -------
    oof_proba   : (n_train, 3)
    test_proba  : (n_test, 3)  — fold-averaged
    scores      : per-fold micro F1 list
    mean_best_iter : mean best_iteration across folds (None for non-ES models)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    n_train, n_test = len(y), len(X_test)

    oof_proba = np.zeros((n_train, 3), dtype=np.float64)
    test_probas: list[np.ndarray] = []
    scores: list[float] = []
    best_iters: list[int] = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = _BUILDERS[name]()
        _fit_fold(name, model, X_tr, y_tr, X_val, y_val)

        bi = _get_best_iter(name, model)
        if bi is not None:
            best_iters.append(bi)

        val_proba = align_proba(model.predict_proba(X_val), model.classes_)
        oof_proba[val_idx] = val_proba

        test_probas.append(align_proba(model.predict_proba(X_test), model.classes_))

        val_pred = np.array(GRADES)[np.argmax(val_proba, axis=1)]
        f1 = f1_score(y_val, val_pred, average="micro", labels=GRADES)
        scores.append(f1)
        print(f"    fold {fold}/{n_splits}: micro_F1={f1:.4f}")

    mean_bi = int(round(float(np.mean(best_iters)))) if best_iters else None
    test_proba_avg = np.mean(test_probas, axis=0)
    return oof_proba, test_proba_avg, scores, mean_bi


def train_full_model(name: str, X: pd.DataFrame, y: np.ndarray, best_iter: int | None) -> Any:
    """Train on the entire dataset (no eval set, fixed n_estimators from CV mean).

    lgbm: no early-stopping callback, just fixed n_estimators.
    xgb/catboost: also strip early_stopping_rounds from constructor.
    et/hgb: use configured params unchanged.
    """
    if name in ("xgb", "catboost"):
        model = _BUILDERS[name](n_iter=best_iter, with_es=False)
    else:
        model = _BUILDERS[name](n_iter=best_iter)
    model.fit(X, y)
    return model


# ── Helpers ───────────────────────────────────────────────────────────────────


def find_ensemble_base_run(rm: RunManager) -> str | None:
    """Return the most recent run with model_type='ensemble_base', or None."""
    for rid in reversed(rm.list_runs()):
        try:
            meta = rm.load_metadata(rid)
        except Exception:
            continue
        if meta.get("model_type") == "ensemble_base":
            return rid
    return None


def print_model_table(cv_metrics: dict[str, dict]) -> None:
    print("\n" + "=" * 55)
    print("Base learner CV summary")
    print("=" * 55)
    print(f"{'Model':<12} {'CV micro F1 (mean ± std)':>26}")
    print("-" * 40)
    for name, m in sorted(cv_metrics.items(), key=lambda kv: -kv[1]["mean"]):
        per_fold = m.get("per_fold", [])
        if per_fold:
            print(f"  {name:<10} {m['mean']:.4f} ± {m['std']:.4f}")
        else:
            print(f"  {name:<10} {m['mean']:.4f}  (cached OOF, no per-fold)")
    print("=" * 55)


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ensemble base learners")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Existing run to resume (default: auto-increment)")
    parser.add_argument("--models", nargs="+",
                        default=["lgbm", "xgb", "catboost", "et", "hgb"],
                        choices=list(_BUILDERS),
                        help="Which base models to train")
    return parser.parse_args()


def main() -> None:
    t0 = time.time()
    args = parse_args()
    rm = RunManager()

    X_train, y_train, X_test, building_ids = load_run003_features()
    print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")

    # ── Run registration ──────────────────────────────────────────────────────
    run_id = args.run_id or find_ensemble_base_run(rm)
    if run_id and rm.run_path(run_id).exists():
        print(f"Resuming ensemble base run: {run_id}")
    else:
        run_id = rm.get_next_run_id()
        rm.create_run(
            description="Ensemble base: LGBM+XGB+CatBoost+ET+HGB on run_003 features",
            model_type="ensemble_base",
            feature_set="embedded_192_features",
            params={"base_models": args.models, "cv_folds": CV_FOLDS},
            run_id=run_id,
            objective="multiclass",
            n_features=X_train.shape[1],
            cv_folds=CV_FOLDS,
            cv_metric="micro_f1",
        )
        print(f"Created ensemble base run: {run_id}")

    run_dir = rm.run_path(run_id)
    all_cv_metrics: dict[str, dict] = {}

    # Load any existing per_model_cv from metadata
    meta = rm.load_metadata(run_id)
    saved_metrics: dict = meta.get("per_model_cv", {})

    # ── Per-model training ────────────────────────────────────────────────────
    for name in args.models:
        oof_path = run_dir / f"oof_proba_{name}.npy"
        test_path = run_dir / f"test_proba_{name}.npy"

        if oof_path.exists() and test_path.exists():
            print(f"\n[{name}] loading cached OOF/test probas")
            oof_p = np.load(oof_path)
            oof_pred = np.array(GRADES)[np.argmax(oof_p, axis=1)]
            f1 = f1_score(y_train, oof_pred, average="micro", labels=GRADES)
            all_cv_metrics[name] = saved_metrics.get(name, {"mean": f1, "std": 0.0, "per_fold": []})
            continue

        print(f"\n{'='*55}")
        print(f"[{name}] 5-fold CV")
        print(f"{'='*55}")

        # Check if this model's builder is available
        try:
            _BUILDERS[name]()
        except ImportError as exc:
            print(f"  SKIP {name}: {exc}")
            continue

        oof_p, test_p, scores, mean_bi = train_model_cv(name, X_train, y_train, X_test)

        np.save(oof_path, oof_p)
        np.save(test_path, test_p)

        mean_f1 = float(np.mean(scores))
        std_f1 = float(np.std(scores, ddof=1))
        all_cv_metrics[name] = {
            "mean": mean_f1,
            "std": std_f1,
            "per_fold": [float(s) for s in scores],
            "mean_best_iter": mean_bi,
        }
        print(f"  [{name}] CV micro F1: {mean_f1:.4f} ± {std_f1:.4f}"
              + (f"  (best_iter={mean_bi})" if mean_bi else ""))

        # Save best model as the one for the best-mean fold, or just train full
        model_path = run_dir / f"model_{name}.pkl"
        if not model_path.exists():
            print(f"  [{name}] fitting full model ({mean_bi or 'default'} iters)...")
            full_model = train_full_model(name, X_train, y_train, mean_bi)
            joblib.dump(full_model, model_path)
            print(f"  [{name}] saved model_{name}.pkl")

    # ── Update metadata ───────────────────────────────────────────────────────
    meta = rm.load_metadata(run_id)
    meta["per_model_cv"] = all_cv_metrics
    # Set run-level CV to best base model's mean (for best_run comparison)
    if all_cv_metrics:
        best_model_mean = max(m["mean"] for m in all_cv_metrics.values())
        best_model_std = next(
            m["std"] for m in all_cv_metrics.values() if m["mean"] == best_model_mean
        )
        rm.save_cv_scores(
            run_id,
            [],
            mean=best_model_mean,
            std=best_model_std,
        )
    RunManager._write_json(run_dir / "metadata.json", meta)

    print_model_table(all_cv_metrics)
    print(f"\nEnsemble base artifacts: runs/{run_id}/")
    print(f"Next step: python src/stack.py --base-run-id {run_id}")
    print(f"\nTotal training time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
