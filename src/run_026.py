#!/usr/bin/env python3
"""
run_026: OOF-optimized blend of run_024 (Shoumik XGB, public SOTA 0.7523) and
run_019 (PCA embed-only + LGBM, public 0.7520).

No retraining — loads saved OOF/test probabilities, searches blend weight in
proba and logit space, registers blend CV and submission.

Run from project root:
    python src/run_026.py
"""

from __future__ import annotations

import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_manager import PROCESSED_DIR, RunManager
from run_trees_260k import _blend_logit, _blend_proba, _oof_f1, pairwise_diagnostic

print = partial(print, flush=True)

RANDOM_STATE = 42
CV_FOLDS = 5
RUN_024_ID = "run_024"
RUN_019_ID = "run_019"
RUN_024_CV_REF = 0.7518
RUN_019_CV_REF = 0.7528
NOISE = 0.0016
THRESHOLD = RUN_024_CV_REF + NOISE


def _per_fold_scores(
    p_a: np.ndarray,
    p_b: np.ndarray,
    y: np.ndarray,
    alpha: float,
    space: str,
) -> list[float]:
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores: list[float] = []
    blend_fn = _blend_proba if space == "proba" else _blend_logit
    for _fold, (_tr, va) in enumerate(skf.split(p_a, y), start=1):
        blended = blend_fn(alpha, p_a[va], p_b[va])
        f1 = f1_score(y[va], blended.argmax(axis=1) + 1, average="micro", labels=[1, 2, 3])
        scores.append(float(f1))
    return scores


def _blend(p_a: np.ndarray, p_b: np.ndarray, alpha: float, space: str) -> np.ndarray:
    if space == "proba":
        return _blend_proba(alpha, p_a, p_b)
    return _blend_logit(alpha, p_a, p_b)


def main() -> None:
    t0 = time.time()
    y = pd.read_csv(PROCESSED_DIR / "y_train_full.csv")["damage_grade"].to_numpy()

    p24_oof = np.load(ROOT / "runs" / RUN_024_ID / "oof_proba.npy").astype(np.float64)
    p19_oof = np.load(ROOT / "runs" / RUN_019_ID / "oof_proba.npy").astype(np.float64)
    p24_test = np.load(ROOT / "runs" / RUN_024_ID / "test_proba.npy").astype(np.float64)
    p19_test = np.load(ROOT / "runs" / RUN_019_ID / "test_proba.npy").astype(np.float64)

    f24 = _oof_f1(p24_oof, y)
    f19 = _oof_f1(p19_oof, y)
    print(f"run_024 solo OOF: {f24:.4f}")
    print(f"run_019 solo OOF: {f19:.4f}")

    print("\n" + "═" * 60)
    print("PAIRWISE BLEND: run_024 + run_019")
    print("═" * 60)
    r = pairwise_diagnostic(RUN_024_ID, p24_oof, RUN_019_ID, p19_oof, y)

    best_score = r["best_score"]
    best_alpha = r["best_alpha"]
    best_space = r["best_space"]
    gain_vs_024 = best_score - f24
    gain_vs_019 = best_score - f19

    print("\n" + "═" * 60)
    print("SUMMARY")
    print("═" * 60)
    print(f"  Best blend OOF:     {best_score:.4f}  ({best_space}, α={best_alpha:.3f} on {RUN_024_ID})")
    print(f"  Weight run_019:     {1.0 - best_alpha:.3f}")
    print(f"  Gain vs run_024:    {gain_vs_024:+.4f}")
    print(f"  Gain vs run_019:    {gain_vs_019:+.4f}")
    print(f"  Noise threshold:    {THRESHOLD:.4f}  "
          f"{'CLEARS ✓' if best_score > THRESHOLD else 'below — may still beat public solos'}")

    fold_scores = _per_fold_scores(p24_oof, p19_oof, y, best_alpha, best_space)
    mean_f1 = float(np.mean(fold_scores))
    std_f1 = float(np.std(fold_scores, ddof=1))
    print(f"  Per-fold blend F1:  {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"  Folds: {[f'{s:.4f}' for s in fold_scores]}")

    oof_blend = _blend(p24_oof, p19_oof, best_alpha, best_space).astype(np.float32)
    test_blend = _blend(p24_test, p19_test, best_alpha, best_space).astype(np.float32)

    rm = RunManager()
    run_id = "run_026"
    run_dir = rm.run_path(run_id)
    blend_params = {
        "base_runs": [RUN_024_ID, RUN_019_ID],
        "alpha_run_024": best_alpha,
        "weight_run_019": 1.0 - best_alpha,
        "blend_space": best_space,
        "solo_oof": {RUN_024_ID: f24, RUN_019_ID: f19},
    }
    if not run_dir.exists():
        rm.create_run(
            description="OOF blend run_024 Shoumik XGB + run_019 PCA-embed LGBM",
            model_type="ensemble_blend",
            feature_set="oof_proba_blend",
            params=blend_params,
            run_id=run_id,
            objective="multiclass",
            n_features=None,
            cv_folds=CV_FOLDS,
            cv_metric="micro_f1",
            notes=f"Both models generalize well on public. Gain vs run_024 {gain_vs_024:+.4f}, vs run_019 {gain_vs_019:+.4f}.",
        )
    else:
        print(f"\n  Updating existing {run_id}/")

    rm.save_cv_scores(run_id, fold_scores, mean_f1, std_f1)
    np.save(run_dir / "oof_proba.npy", oof_blend)
    np.save(run_dir / "test_proba.npy", test_blend)

    test_csv = pd.read_csv(ROOT / "data" / "driven_data" / "test_values.csv")
    sub_df = pd.DataFrame({
        "building_id": test_csv["building_id"].values,
        "damage_grade": test_blend.argmax(axis=1) + 1,
    })
    rm.save_submission(run_id, sub_df)

    meta = rm.load_metadata(run_id)
    meta["blend_oof_f1"] = best_score
    meta["blend_space"] = best_space
    meta["blend_alpha_run_024"] = best_alpha
    meta["weight_run_019"] = 1.0 - best_alpha
    meta["solo_oof_run_024"] = f24
    meta["solo_oof_run_019"] = f19
    meta["gain_vs_run_024"] = gain_vs_024
    meta["gain_vs_run_019"] = gain_vs_019
    meta["disagree_rate"] = r["disagree_rate"]
    meta["loss_corr"] = r["loss_corr"]
    RunManager._write_json(run_dir / "metadata.json", meta)
    RunManager._write_json(run_dir / "params.json", meta["params"])

    grades, counts = np.unique(sub_df["damage_grade"], return_counts=True)
    print(f"\n── {run_id} submission grade distribution ──")
    for g, c in zip(grades, counts):
        print(f"  grade {g}: {c:,} ({c/len(sub_df)*100:.1f}%)")

    print(f"\nRegistered {run_id}  submission → runs/{run_id}/submission.csv")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
