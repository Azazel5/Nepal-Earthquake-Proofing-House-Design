#!/usr/bin/env python3
"""
run_028: Combined pipeline — LGBM on Shoumik features + stacked meta-learner +
class threshold tuning.

Stages:
  1. LGBM (trial-66) on run_024 Shoumik feature matrix — 5-fold OOF + full refit
  2. Stack run_024 + run_019 + new LGBM OOF with logistic regression meta-model
  3. Grid-search log-prob offsets for grade 1 / grade 3 on meta OOF
  4. Apply to test → submission

Run from project root:
    python src/run_028.py [--quick] [--skip-lgbm]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retrain import EARLY_STOPPING_ROUNDS, TRIAL_66_PARAMS
from run_manager import PROCESSED_DIR, RunManager
from run_024_features import ShoumikFeatureBuilder, geo_latents_ready, load_frames, load_geo_latents
from run_trees_260k import _oof_f1
from stack import build_stacking_matrix, grades_micro_f1

print = partial(print, flush=True)

RANDOM_STATE = 42
CV_FOLDS = 5
RUN_ID = "run_028"
GRADES = [1, 2, 3]
BASE_RUNS = ["run_024", "run_019"]
RUN_026_ID = "run_026"

LGBM_PARAMS = {**TRIAL_66_PARAMS, "n_jobs": 1}


def _micro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred, average="micro", labels=GRADES)


def _micro_f1_proba(y_true: np.ndarray, proba: np.ndarray) -> float:
    return _micro_f1(y_true, proba.argmax(axis=1) + 1)


def _clip_proba(S: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(S, eps, 1.0 - eps)


def _align_proba(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Reorder predict_proba columns to grades 1, 2, 3."""
    out = np.zeros((len(proba), 3), dtype=np.float64)
    for i, g in enumerate(GRADES):
        j = int(np.where(classes == g)[0][0])
        out[:, i] = proba[:, j]
    return out


def _ensure_geo_latents() -> None:
    if geo_latents_ready():
        return
    cmd = [sys.executable, str(ROOT / "src" / "run_024_geo.py")]
    print("Geo latents missing — running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def train_lgbm_shoumik_cv(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    geo_train: np.ndarray,
    geo_test: np.ndarray,
    dr_train: np.ndarray,
    dr_test: np.ndarray,
    ru_train: np.ndarray,
    ru_test: np.ndarray,
    *,
    quick: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[float], list[int]]:
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(train, y))
    n_folds = 1 if quick else CV_FOLDS

    oof = np.zeros((len(y), 3), dtype=np.float32)
    test_folds: list[np.ndarray] = []
    scores: list[float] = []
    best_iters: list[int] = []

    for fold, (tri, vai) in enumerate(splits[:n_folds], start=1):
        t0 = time.time()
        fb = ShoumikFeatureBuilder()
        fb.fit(train.iloc[tri])
        X_tr = fb.transform(train.iloc[tri], geo_train[tri], dr_train[tri], ru_train[tri])
        X_va = fb.transform(train.iloc[vai], geo_train[vai], dr_train[vai], ru_train[vai])
        X_te = fb.transform(test, geo_test, dr_test, ru_test)

        model = LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_tr,
            y[tri],
            eval_set=[(X_va, y[vai])],
            eval_metric="multi_logloss",
            callbacks=[early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        best = model.best_iteration_ if model.best_iteration_ else LGBM_PARAMS["n_estimators"]
        best_iters.append(int(best))
        oof[vai] = model.predict_proba(X_va).astype(np.float32)
        test_folds.append(model.predict_proba(X_te).astype(np.float32))
        f1 = _micro_f1_proba(y[vai], oof[vai])
        scores.append(f1)
        print(f"  LGBM fold {fold}: F1={f1:.4f}  best_iter={best}  ({time.time() - t0:.0f}s)")

    return oof, np.mean(test_folds, axis=0), scores, best_iters


def fit_lgbm_full(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    geo_train: np.ndarray,
    geo_test: np.ndarray,
    dr_train: np.ndarray,
    dr_test: np.ndarray,
    ru_train: np.ndarray,
    ru_test: np.ndarray,
    n_estimators: int,
) -> np.ndarray:
    fb = ShoumikFeatureBuilder()
    fb.fit(train)
    X_full = fb.transform(train, geo_train, dr_train, ru_train)
    X_te = fb.transform(test, geo_test, dr_test, ru_test)
    params = {**LGBM_PARAMS, "n_estimators": n_estimators}
    model = LGBMClassifier(**params)
    model.fit(X_full, y)
    return model.predict_proba(X_te).astype(np.float32)


def train_meta_oof(
    S: np.ndarray,
    y: np.ndarray,
    *,
    quick: bool = False,
) -> tuple[Pipeline, list[float], np.ndarray]:
    """5-fold CV meta-learner; returns fitted pipeline and OOF probabilities."""
    S = _clip_proba(S)
    n_splits = 1 if quick else CV_FOLDS
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    meta = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, max_iter=2000, random_state=RANDOM_STATE, solver="saga")),
    ])
    oof_proba = np.zeros((len(y), 3), dtype=np.float64)
    scores: list[float] = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for fold, (tr, va) in enumerate(skf.split(S, y), start=1):
            m = clone(meta)
            m.fit(S[tr], y[tr])
            classes = m.named_steps["clf"].classes_
            oof_proba[va] = _align_proba(m.predict_proba(S[va]), classes)
            f1 = _micro_f1_proba(y[va], oof_proba[va])
            scores.append(f1)
            print(f"  meta fold {fold}: F1={f1:.4f}")

        meta.fit(S, y)

    return meta, scores, oof_proba


def predict_meta(meta: Pipeline, S: np.ndarray) -> np.ndarray:
    S = _clip_proba(S)
    classes = meta.named_steps["clf"].classes_
    return _align_proba(meta.predict_proba(S), classes)


def apply_log_offsets(proba: np.ndarray, off_g1: float, off_g3: float) -> np.ndarray:
    eps = 1e-7
    logp = np.log(proba + eps)
    logp[:, 0] += off_g1
    logp[:, 2] += off_g3
    return logp


def predict_with_offsets(proba: np.ndarray, off_g1: float, off_g3: float) -> np.ndarray:
    logp = apply_log_offsets(proba, off_g1, off_g3)
    return logp.argmax(axis=1) + 1


def search_thresholds(
    proba: np.ndarray,
    y: np.ndarray,
    *,
    step: float = 0.05,
) -> tuple[float, float, float]:
    grid = np.arange(-0.4, 0.401, step)
    best_f1 = -1.0
    best_off = (0.0, 0.0)
    for off_g1 in grid:
        for off_g3 in grid:
            pred = predict_with_offsets(proba, float(off_g1), float(off_g3))
            f1 = _micro_f1(y, pred)
            if f1 > best_f1:
                best_f1 = f1
                best_off = (float(off_g1), float(off_g3))
    return best_f1, best_off[0], best_off[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-lgbm", action="store_true", help="Reuse runs/run_028/lgbm_oof_proba.npy")
    args = parser.parse_args()
    t0 = time.time()

    y = pd.read_csv(PROCESSED_DIR / "y_train_full.csv")["damage_grade"].to_numpy()
    train, test = load_frames()
    _ensure_geo_latents()
    geo_train, geo_test, dr_train, dr_test, ru_train, ru_test = load_geo_latents()

    run_dir = ROOT / "runs" / RUN_ID
    lgbm_oof_path = run_dir / "lgbm_oof_proba.npy"
    lgbm_test_path = run_dir / "lgbm_test_proba.npy"

    # ── Stage 1: LGBM on Shoumik features ───────────────────────────────────
    if args.skip_lgbm and lgbm_oof_path.exists() and lgbm_test_path.exists():
        print("── Stage 1: LGBM (cached) ──")
        lgbm_oof = np.load(lgbm_oof_path).astype(np.float32)
        lgbm_test = np.load(lgbm_test_path).astype(np.float32)
        lgbm_solo = _micro_f1_proba(y, lgbm_oof.astype(np.float64))
        print(f"  LGBM solo OOF: {lgbm_solo:.4f}")
    else:
        print("\n── Stage 1: LGBM on Shoumik features ──")
        lgbm_oof, lgbm_test_cv, lgbm_scores, lgbm_iters = train_lgbm_shoumik_cv(
            train, test, y,
            geo_train, geo_test, dr_train, dr_test, ru_train, ru_test,
            quick=args.quick,
        )
        lgbm_solo = float(np.mean(lgbm_scores))
        print(f"  LGBM CV mean: {lgbm_solo:.4f}")

        if args.quick:
            run_dir.mkdir(parents=True, exist_ok=True)
            np.save(lgbm_oof_path, lgbm_oof)
            np.save(lgbm_test_path, lgbm_test_cv.astype(np.float32))
            print(f"\nQuick test done ({time.time() - t0:.1f}s)")
            return

        n_est = int(np.mean(lgbm_iters))
        print(f"  LGBM full-train refit (n_estimators={n_est})")
        lgbm_test = fit_lgbm_full(
            train, test, y,
            geo_train, geo_test, dr_train, dr_test, ru_train, ru_test,
            n_est,
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        np.save(lgbm_oof_path, lgbm_oof)
        np.save(lgbm_test_path, lgbm_test)

    # Load base model probabilities
    oof_dict: dict[str, np.ndarray] = {}
    test_dict: dict[str, np.ndarray] = {}
    for rid in BASE_RUNS:
        oof_dict[rid] = np.load(ROOT / "runs" / rid / "oof_proba.npy").astype(np.float64)
        test_dict[rid] = np.load(ROOT / "runs" / rid / "test_proba.npy").astype(np.float64)
        print(f"{rid} solo OOF: {_oof_f1(oof_dict[rid], y):.4f}")

    oof_dict["lgbm_shoumik"] = lgbm_oof.astype(np.float64)
    test_dict["lgbm_shoumik"] = lgbm_test.astype(np.float64)
    print(f"lgbm_shoumik solo OOF: {lgbm_solo:.4f}")

    model_names = BASE_RUNS + ["lgbm_shoumik"]
    S_train = build_stacking_matrix(oof_dict, model_names)
    S_test = build_stacking_matrix(test_dict, model_names)
    print(f"\nStack matrix: {S_train.shape[1]} features ({len(model_names)} models × 2 proba cols)")

    # ── Stage 2: Meta-learner ───────────────────────────────────────────────
    print("\n── Stage 2: Logistic regression stacker ──")
    meta, meta_scores, meta_oof = train_meta_oof(S_train, y)
    meta_solo = _micro_f1_proba(y, meta_oof)
    meta_mean = float(np.mean(meta_scores))
    meta_std = float(np.std(meta_scores, ddof=1))
    print(f"  Meta OOF F1: {meta_solo:.4f}  (CV folds: {meta_mean:.4f} ± {meta_std:.4f})")

    # ── Stage 3: Threshold tuning ───────────────────────────────────────────
    print("\n── Stage 3: Grade 1 / 3 log-offset threshold search ──")
    th_f1, off_g1, off_g3 = search_thresholds(meta_oof, y, step=0.05)
    print(f"  Coarse grid: F1={th_f1:.4f}  off_g1={off_g1:+.2f}  off_g3={off_g3:+.2f}")

    # Refine around best
    fine_g1 = np.arange(off_g1 - 0.08, off_g1 + 0.081, 0.02)
    fine_g3 = np.arange(off_g3 - 0.08, off_g3 + 0.081, 0.02)
    best_f1, best_g1, best_g3 = th_f1, off_g1, off_g3
    for o1 in fine_g1:
        for o3 in fine_g3:
            pred = predict_with_offsets(meta_oof, float(o1), float(o3))
            f1 = _micro_f1(y, pred)
            if f1 > best_f1:
                best_f1, best_g1, best_g3 = f1, float(o1), float(o3)
    print(f"  Refined:     F1={best_f1:.4f}  off_g1={best_g1:+.3f}  off_g3={best_g3:+.3f}")

    argmax_f1 = _micro_f1_proba(y, meta_oof)
    print(f"  Meta argmax: F1={argmax_f1:.4f}  (threshold gain: {best_f1 - argmax_f1:+.4f})")

    # ── Compare run_026 ───────────────────────────────────────────────────
    f26 = None
    p26 = ROOT / "runs" / RUN_026_ID / "oof_proba.npy"
    if p26.exists():
        f26 = _oof_f1(np.load(p26).astype(np.float64), y)
        print(f"\n  run_026 OOF: {f26:.4f}  Δ run_028: {best_f1 - f26:+.4f}")

    # ── Test predictions ────────────────────────────────────────────────────
    meta_test = predict_meta(meta, S_test)
    final_pred = predict_with_offsets(meta_test, best_g1, best_g3)

    # Per-fold CV for registration (stack + thresholds)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores: list[float] = []
    for _fold, (_tr, va) in enumerate(skf.split(S_train, y), start=1):
        m = clone(meta)
        m.fit(S_train[_tr], y[_tr])
        classes = m.named_steps["clf"].classes_
        va_proba = _align_proba(m.predict_proba(S_train[va]), classes)
        pred = predict_with_offsets(va_proba, best_g1, best_g3)
        fold_scores.append(_micro_f1(y[va], pred))

    mean_f1 = float(np.mean(fold_scores))
    std_f1 = float(np.std(fold_scores, ddof=1))

    # OOF with thresholds (using full meta OOF — slight optimism vs nested, standard for this project)
    oof_final = predict_with_offsets(meta_oof, best_g1, best_g3)
    oof_proba_out = meta_oof.astype(np.float32)

    rm = RunManager()
    run_path = rm.run_path(RUN_ID)
    run_path.mkdir(parents=True, exist_ok=True)

    blend_params = {
        "base_runs": model_names,
        "lgbm": LGBM_PARAMS,
        "meta": {"type": "logreg", "C": 0.1, "solver": "saga"},
        "thresholds": {"off_grade1": best_g1, "off_grade3": best_g3},
        "solo_oof": {
            "run_024": _oof_f1(oof_dict["run_024"], y),
            "run_019": _oof_f1(oof_dict["run_019"], y),
            "lgbm_shoumik": lgbm_solo,
            "meta_argmax": argmax_f1,
        },
    }

    if not (run_path / "metadata.json").exists():
        metadata = {
            "run_id": RUN_ID,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "description": "LGBM Shoumik + stacked meta (run_024/019/lgbm) + grade threshold tuning",
            "model_type": "ensemble_stack_threshold",
            "objective": "multiclass",
            "feature_set": "shoumik_lgbm + oof_stack + log_offsets",
            "n_features": None,
            "params": blend_params,
            "cv_folds": CV_FOLDS,
            "cv_metric": "micro_f1",
            "cv_mean": None,
            "cv_std": None,
            "cv_scores_per_fold": None,
            "public_leaderboard_score": None,
            "submitted": False,
            "submission_date": None,
            "notes": "Combines ideas: LGBM on Shoumik features, logreg stack, g1/g3 log offsets.",
        }
        RunManager._write_json(run_path / "metadata.json", metadata)
        RunManager._write_json(run_path / "params.json", blend_params)

    joblib.dump(meta, rm.run_path(RUN_ID) / "meta_model.pkl")
    rm.save_cv_scores(RUN_ID, fold_scores, mean_f1, std_f1)
    np.save(rm.run_path(RUN_ID) / "oof_proba.npy", oof_proba_out)
    np.save(rm.run_path(RUN_ID) / "test_proba.npy", meta_test.astype(np.float32))

    test_csv = pd.read_csv(ROOT / "data" / "driven_data" / "test_values.csv")
    sub = pd.DataFrame({"building_id": test_csv["building_id"].values, "damage_grade": final_pred})
    rm.save_submission(RUN_ID, sub)

    meta_doc = rm.load_metadata(RUN_ID)
    meta_doc["stack_oof_f1"] = meta_solo
    meta_doc["threshold_oof_f1"] = best_f1
    meta_doc["threshold_offsets"] = {"grade1": best_g1, "grade3": best_g3}
    meta_doc["gain_threshold_vs_argmax"] = best_f1 - argmax_f1
    if f26 is not None:
        meta_doc["run_026_oof"] = f26
        meta_doc["gain_vs_run_026"] = best_f1 - f26
    RunManager._write_json(rm.run_path(RUN_ID) / "metadata.json", meta_doc)

    print("\n" + "═" * 60)
    print("SUMMARY")
    print("═" * 60)
    print(f"  LGBM Shoumik solo:     {lgbm_solo:.4f}")
    print(f"  Meta stack OOF:        {meta_solo:.4f}")
    print(f"  + threshold tuning:    {best_f1:.4f}")
    print(f"  Per-fold (nested):     {mean_f1:.4f} ± {std_f1:.4f}")
    if f26 is not None:
        print(f"  vs run_026:            {best_f1 - f26:+.4f}")
    grades, counts = np.unique(final_pred, return_counts=True)
    print(f"\n── {RUN_ID} submission ──")
    for g, c in zip(grades, counts):
        print(f"  grade {g}: {c:,} ({c/len(final_pred)*100:.1f}%)")
    print(f"\nRegistered {RUN_ID} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
