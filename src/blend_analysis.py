#!/usr/bin/env python3
"""
OOF blend analysis: LGBM (run_012) × MLP (run_013).

Steps:
  1. Regenerate LGBM OOF probabilities via 5-fold CV on X_train_run012.csv
     (same StratifiedKFold seed as retrain.py — results are cached to disk).
  2. Load MLP OOF probabilities from runs/run_013/oof_proba.npy.
  3. Grid search α in proba space:  α·LGBM + (1-α)·MLP
  4. scipy-optimize α in proba space.
  5. Repeat 3-4 in logit space.
  6. Diagnostics: argmax disagreement rate, per-row loss correlation.
  7. Print final recommendation and, if blend CV > 0.7588+noise, generate submission.

Run from project root:
    python src/blend_analysis.py
"""

from __future__ import annotations

import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retrain import TRIAL_66_PARAMS, CV_FOLDS, EARLY_STOPPING_ROUNDS
from run_manager import PROCESSED_DIR, RunManager

print = partial(print, flush=True)

RANDOM_STATE = 42
LGBM_OOF_PATH = ROOT / "runs" / "run_012" / "oof_proba.npy"
MLP_OOF_PATH  = ROOT / "runs" / "run_013" / "oof_proba.npy"
MLP_TEST_PATH = ROOT / "runs" / "run_013" / "test_proba.npy"
LGBM_TEST_PATH = ROOT / "runs" / "run_012" / "test_proba.npy"


# ── LGBM OOF generation (cached) ──────────────────────────────────────────────

def generate_lgbm_oof() -> np.ndarray:
    """
    5-fold LGBM OOF on X_train_run012 features.
    Cached to runs/run_012/oof_proba.npy after first run.
    Also saves test_proba averaged across folds.
    """
    if LGBM_OOF_PATH.exists():
        print(f"  Loading cached LGBM OOF from {LGBM_OOF_PATH.name}")
        return np.load(LGBM_OOF_PATH)

    print("  Generating LGBM OOF (5-fold CV on X_train_run012.csv)...")
    X = pd.read_csv(PROCESSED_DIR / "X_train_run012.csv")
    y = pd.read_csv(PROCESSED_DIR / "y_train_full.csv")["damage_grade"].to_numpy()
    X_test = pd.read_csv(PROCESSED_DIR / "X_test_run012.csv")

    oof = np.zeros((len(y), 3), dtype=np.float32)
    test_folds: list[np.ndarray] = []

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        model = LGBMClassifier(**TRIAL_66_PARAMS)
        model.fit(
            X.iloc[tr_idx], y[tr_idx],
            eval_set=[(X.iloc[va_idx], y[va_idx])],
            eval_metric="multi_logloss",
            callbacks=[early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        oof[va_idx] = model.predict_proba(X.iloc[va_idx]).astype(np.float32)
        test_folds.append(model.predict_proba(X_test).astype(np.float32))

        f1 = f1_score(y[va_idx], oof[va_idx].argmax(axis=1) + 1,
                      average="micro", labels=[1, 2, 3])
        print(f"    fold {fold}: F1={f1:.4f}  best_iter={model.best_iteration_}")

    np.save(LGBM_OOF_PATH, oof)
    np.save(LGBM_TEST_PATH, np.mean(test_folds, axis=0))
    print(f"  Saved {LGBM_OOF_PATH.name} and {LGBM_TEST_PATH.name}")

    oof_f1 = f1_score(y, oof.argmax(axis=1) + 1, average="micro", labels=[1, 2, 3])
    print(f"  LGBM OOF micro F1: {oof_f1:.4f}")
    return oof


# ── Blend scoring ──────────────────────────────────────────────────────────────

def proba_blend_f1(alpha: float, lgbm_p: np.ndarray, mlp_p: np.ndarray, y_true: np.ndarray) -> float:
    """Score α·LGBM + (1-α)·MLP in probability space."""
    blend = alpha * lgbm_p + (1.0 - alpha) * mlp_p
    preds = blend.argmax(axis=1) + 1  # back to 1-indexed grades
    return float(f1_score(y_true, preds, average="micro", labels=[1, 2, 3]))


def logit_blend_f1(alpha: float, lgbm_p: np.ndarray, mlp_p: np.ndarray, y_true: np.ndarray) -> float:
    """Score α·log(LGBM) + (1-α)·log(MLP) in logit space (then softmax implicitly via argmax)."""
    eps = 1e-7
    logits = alpha * np.log(lgbm_p + eps) + (1.0 - alpha) * np.log(mlp_p + eps)
    preds = logits.argmax(axis=1) + 1
    return float(f1_score(y_true, preds, average="micro", labels=[1, 2, 3]))


# ── Diagnostics ────────────────────────────────────────────────────────────────

def compute_diagnostics(lgbm_p: np.ndarray, mlp_p: np.ndarray, y_true: np.ndarray) -> None:
    lgbm_pred = lgbm_p.argmax(axis=1) + 1
    mlp_pred  = mlp_p.argmax(axis=1) + 1

    disagree_mask = lgbm_pred != mlp_pred
    disagree_rate = disagree_mask.mean()
    print(f"\n── Diagnostics ──────────────────────────────────────────────────")
    print(f"  Argmax disagreement:     {disagree_rate:.3f}  ({disagree_mask.sum():,} / {len(y_true):,} rows)")

    # Among disagreements, who's right?
    lgbm_right = (lgbm_pred == y_true).sum()
    mlp_right  = (mlp_pred  == y_true).sum()
    lgbm_right_where_disagree = ((lgbm_pred == y_true) & disagree_mask).sum()
    mlp_right_where_disagree  = ((mlp_pred  == y_true) & disagree_mask).sum()
    print(f"  On disagreed rows — LGBM correct: {lgbm_right_where_disagree:,}  "
          f"MLP correct: {mlp_right_where_disagree:,}  "
          f"neither: {disagree_mask.sum() - lgbm_right_where_disagree - mlp_right_where_disagree:,}")

    # Per-row cross-entropy loss
    eps = 1e-7
    idx = y_true - 1  # 0-indexed
    N = len(y_true)
    lgbm_loss = -np.log(lgbm_p[np.arange(N), idx] + eps)
    mlp_loss  = -np.log(mlp_p[np.arange(N), idx] + eps)
    loss_corr = float(np.corrcoef(lgbm_loss, mlp_loss)[0, 1])
    print(f"  Per-row loss correlation: {loss_corr:.4f}")
    print(f"  LGBM mean loss: {lgbm_loss.mean():.4f}   MLP mean loss: {mlp_loss.mean():.4f}")

    # Grade-level accuracy breakdown
    print(f"\n  Per-grade accuracy breakdown:")
    for g in [1, 2, 3]:
        mask = y_true == g
        lgbm_acc = (lgbm_pred[mask] == g).mean()
        mlp_acc  = (mlp_pred[mask] == g).mean()
        print(f"    grade {g} (n={mask.sum():,}): LGBM={lgbm_acc:.4f}  MLP={mlp_acc:.4f}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    # ── 1. Load / generate OOF probabilities ─────────────────────────────────
    print("── Loading OOF probabilities ──")
    lgbm_oof = generate_lgbm_oof()
    mlp_oof  = np.load(MLP_OOF_PATH).astype(np.float64)
    lgbm_oof = lgbm_oof.astype(np.float64)

    y = pd.read_csv(PROCESSED_DIR / "y_train_full.csv")["damage_grade"].to_numpy()
    assert len(lgbm_oof) == len(mlp_oof) == len(y), "OOF length mismatch"

    lgbm_cv = proba_blend_f1(1.0, lgbm_oof, mlp_oof, y)
    mlp_cv  = proba_blend_f1(0.0, lgbm_oof, mlp_oof, y)
    print(f"\n  LGBM solo OOF micro F1: {lgbm_cv:.4f}")
    print(f"  MLP  solo OOF micro F1: {mlp_cv:.4f}")

    # ── 2. Diagnostics ────────────────────────────────────────────────────────
    compute_diagnostics(lgbm_oof, mlp_oof, y)

    # ── 3. Grid search (proba space) ─────────────────────────────────────────
    print("\n── Grid search α (proba space: α·LGBM + (1-α)·MLP) ──")
    alphas = np.linspace(0, 1, 101)
    grid_scores = [proba_blend_f1(a, lgbm_oof, mlp_oof, y) for a in alphas]
    best_alpha_grid = alphas[np.argmax(grid_scores)]
    best_score_grid = max(grid_scores)
    print(f"  Best α (grid, step=0.01): {best_alpha_grid:.2f}  F1={best_score_grid:.4f}")

    # Print nearby grid scores
    best_i = int(np.argmax(grid_scores))
    print("  α sweep around optimum:")
    for i in range(max(0, best_i - 5), min(len(alphas), best_i + 6)):
        marker = " ←" if i == best_i else ""
        print(f"    α={alphas[i]:.2f}  F1={grid_scores[i]:.4f}{marker}")

    # ── 4. scipy optimize (proba space) ──────────────────────────────────────
    res_p = minimize_scalar(
        lambda a: -proba_blend_f1(a, lgbm_oof, mlp_oof, y),
        bounds=(0.0, 1.0), method="bounded",
        options={"xatol": 1e-4},
    )
    best_alpha_opt = float(res_p.x)
    best_score_opt = -float(res_p.fun)
    print(f"\n  scipy optimum:  α={best_alpha_opt:.4f}  F1={best_score_opt:.4f}")

    # ── 5. Grid + optimize (logit space) ─────────────────────────────────────
    print("\n── Grid search α (logit space) ──")
    logit_scores = [logit_blend_f1(a, lgbm_oof, mlp_oof, y) for a in alphas]
    best_alpha_logit_grid = alphas[np.argmax(logit_scores)]
    best_score_logit = max(logit_scores)
    print(f"  Best α (grid): {best_alpha_logit_grid:.2f}  F1={best_score_logit:.4f}")

    res_l = minimize_scalar(
        lambda a: -logit_blend_f1(a, lgbm_oof, mlp_oof, y),
        bounds=(0.0, 1.0), method="bounded",
        options={"xatol": 1e-4},
    )
    best_alpha_logit_opt = float(res_l.x)
    best_score_logit_opt = -float(res_l.fun)
    print(f"  scipy optimum:  α={best_alpha_logit_opt:.4f}  F1={best_score_logit_opt:.4f}")

    # ── 6. Summary ────────────────────────────────────────────────────────────
    LGBM_CV_REFERENCE = 0.7588
    NOISE_THRESHOLD   = 0.7588 + 0.0016  # must beat by at least fold noise

    best_overall = max(best_score_opt, best_score_logit_opt)
    best_space   = "proba" if best_score_opt >= best_score_logit_opt else "logit"
    best_alpha_f = best_alpha_opt if best_space == "proba" else best_alpha_logit_opt

    print("\n" + "═" * 60)
    print("BLEND SUMMARY")
    print("═" * 60)
    print(f"  LGBM solo OOF:        {lgbm_cv:.4f}  (reference)")
    print(f"  MLP  solo OOF:        {mlp_cv:.4f}")
    print(f"  Best proba blend:     {best_score_opt:.4f}  (α={best_alpha_opt:.3f})")
    print(f"  Best logit blend:     {best_score_logit_opt:.4f}  (α={best_alpha_logit_opt:.3f})")
    print(f"  Gain over LGBM:       {best_overall - LGBM_CV_REFERENCE:+.4f}")
    print(f"  Beat noise threshold: {'YES ✓' if best_overall > NOISE_THRESHOLD else 'NO — blend unlikely to help'}")

    # ── 7. Build blend submission if it beats threshold ───────────────────────
    if best_overall > NOISE_THRESHOLD:
        print(f"\n── Building blend submission ({best_space} space, α={best_alpha_f:.3f}) ──")

        lgbm_test = np.load(LGBM_TEST_PATH).astype(np.float64)
        mlp_test  = np.load(MLP_TEST_PATH).astype(np.float64)

        if best_space == "proba":
            blend_test = best_alpha_f * lgbm_test + (1.0 - best_alpha_f) * mlp_test
        else:
            eps = 1e-7
            blend_test = (best_alpha_f * np.log(lgbm_test + eps)
                          + (1.0 - best_alpha_f) * np.log(mlp_test + eps))

        test_grades = blend_test.argmax(axis=1) + 1

        test_csv = pd.read_csv(ROOT / "data" / "driven_data" / "test_values.csv")
        sub_df = pd.DataFrame({
            "building_id": test_csv["building_id"].values,
            "damage_grade": test_grades,
        })

        out = ROOT / "outputs" / f"submission_blend_lgbm_mlp.csv"
        out.parent.mkdir(exist_ok=True)
        sub_df.to_csv(out, index=False)
        print(f"  Saved → {out.relative_to(ROOT)}")

        grades, counts = np.unique(test_grades, return_counts=True)
        for g, c in zip(grades, counts):
            print(f"  grade {g}: {c:,} ({c/len(test_grades)*100:.1f}%)")
    else:
        print("\n  → MLP adds no measurable signal at current quality.")
        print("    Recommendation: improve MLP before blending (wider, deeper, or different features).")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
