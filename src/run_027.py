#!/usr/bin/env python3
"""
run_027: OOF-optimized 3-way blend of run_024 (Shoumik XGB), run_019 (PCA+LGBM),
and run_025 (Shoumik + CatBoostEncoder geo).

No retraining — loads saved OOF/test probabilities, searches weights on the
simplex (proba and logit space), registers blend CV and submission.

Run from project root:
    python src/run_027.py
"""

from __future__ import annotations

import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_manager import PROCESSED_DIR, RunManager
from run_trees_260k import _oof_f1, pairwise_diagnostic, threeway_optimize

print = partial(print, flush=True)

RANDOM_STATE = 42
CV_FOLDS = 5
RUN_024_ID = "run_024"
RUN_019_ID = "run_019"
RUN_025_ID = "run_025"
RUN_026_ID = "run_026"
BASE_RUNS = [RUN_024_ID, RUN_019_ID, RUN_025_ID]
RUN_026_CV_REF = 0.7546
NOISE = 0.0016
THRESHOLD = RUN_026_CV_REF + NOISE


def _threeway_logit(
    p_a: np.ndarray,
    p_b: np.ndarray,
    p_c: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float, float]:
    eps = 1e-7
    log_a = np.log(p_a + eps)
    log_b = np.log(p_b + eps)
    log_c = np.log(p_c + eps)

    def neg_f1(w: np.ndarray) -> float:
        w = np.clip(w, 0, 1)
        w = w / w.sum()
        blend = w[0] * log_a + w[1] * log_b + w[2] * log_c
        return -_oof_f1(blend, y)

    best_val = 1.0
    best_w = (1.0, 0.0, 0.0)
    for i in range(21):
        for j in range(21 - i):
            k = 20 - i - j
            w = np.array([i, j, k], dtype=float) / 20
            v = neg_f1(w)
            if v < best_val:
                best_val = v
                best_w = tuple(w)

    res = minimize(
        neg_f1,
        x0=np.array(best_w),
        method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 5000},
    )
    w = np.clip(res.x, 0, 1)
    w = w / w.sum()
    return float(-res.fun), float(w[0]), float(w[1]), float(w[2])


def _blend_threeway(
    probs: list[np.ndarray],
    weights: tuple[float, float, float],
    space: str,
) -> np.ndarray:
    w0, w1, w2 = weights
    if space == "proba":
        return w0 * probs[0] + w1 * probs[1] + w2 * probs[2]
    eps = 1e-7
    return (
        w0 * np.log(probs[0] + eps)
        + w1 * np.log(probs[1] + eps)
        + w2 * np.log(probs[2] + eps)
    )


def _per_fold_scores(
    probs: list[np.ndarray],
    y: np.ndarray,
    weights: tuple[float, float, float],
    space: str,
) -> list[float]:
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores: list[float] = []
    for _fold, (_tr, va) in enumerate(skf.split(probs[0], y), start=1):
        fold_probs = [p[va] for p in probs]
        blended = _blend_threeway(fold_probs, weights, space)
        f1 = f1_score(y[va], blended.argmax(axis=1) + 1, average="micro", labels=[1, 2, 3])
        scores.append(float(f1))
    return scores


def _load_run_probs(run_id: str) -> tuple[np.ndarray, np.ndarray]:
    run_dir = ROOT / "runs" / run_id
    oof = np.load(run_dir / "oof_proba.npy").astype(np.float64)
    test = np.load(run_dir / "test_proba.npy").astype(np.float64)
    return oof, test


def main() -> None:
    t0 = time.time()
    y = pd.read_csv(PROCESSED_DIR / "y_train_full.csv")["damage_grade"].to_numpy()

    oof: dict[str, np.ndarray] = {}
    test: dict[str, np.ndarray] = {}
    solo: dict[str, float] = {}
    for rid in BASE_RUNS:
        oof[rid], test[rid] = _load_run_probs(rid)
        solo[rid] = _oof_f1(oof[rid], y)
        print(f"{rid} solo OOF: {solo[rid]:.4f}")

    p24, p19, p25 = oof[RUN_024_ID], oof[RUN_019_ID], oof[RUN_025_ID]

    print("\n" + "═" * 60)
    print("PAIRWISE (diagnostics)")
    print("═" * 60)
    for a, b in [(RUN_024_ID, RUN_019_ID), (RUN_024_ID, RUN_025_ID), (RUN_019_ID, RUN_025_ID)]:
        pairwise_diagnostic(a, oof[a], b, oof[b], y)

    print("\n" + "═" * 60)
    print("3-WAY BLEND: run_024 + run_019 + run_025")
    print("═" * 60)

    f_proba, w24_p, w19_p, w25_p = threeway_optimize(p24, p19, p25, y)
    f_logit, w24_l, w19_l, w25_l = _threeway_logit(p24, p19, p25, y)

    print(f"  Best proba:  F1={f_proba:.4f}  w=({w24_p:.3f}, {w19_p:.3f}, {w25_p:.3f})")
    print(f"  Best logit:  F1={f_logit:.4f}  w=({w24_l:.3f}, {w19_l:.3f}, {w25_l:.3f})")

    if f_proba >= f_logit:
        best_score, best_space = f_proba, "proba"
        weights = (w24_p, w19_p, w25_p)
    else:
        best_score, best_space = f_logit, "logit"
        weights = (w24_l, w19_l, w25_l)

    w24, w19, w25 = weights
    probs_oof = [p24, p19, p25]
    probs_test = [test[RUN_024_ID], test[RUN_019_ID], test[RUN_025_ID]]

    # Compare to run_026 2-way
    f26 = None
    run26_path = ROOT / "runs" / RUN_026_ID / "oof_proba.npy"
    if run26_path.exists():
        f26 = _oof_f1(np.load(run26_path).astype(np.float64), y)
        print(f"\n  run_026 (2-way) OOF: {f26:.4f}  Δ 3-way: {best_score - f26:+.4f}")

    gain_vs_best_solo = best_score - max(solo.values())
    print("\n" + "═" * 60)
    print("SUMMARY")
    print("═" * 60)
    print(f"  Best 3-way OOF:     {best_score:.4f}  ({best_space})")
    print(f"  Weights:            run_024={w24:.3f}  run_019={w19:.3f}  run_025={w25:.3f}")
    print(f"  Gain vs best solo:  {gain_vs_best_solo:+.4f}")
    if f26 is not None:
        print(f"  Gain vs run_026:    {best_score - f26:+.4f}")
    print(f"  Threshold ({RUN_026_CV_REF}+{NOISE}): {THRESHOLD:.4f}  "
          f"{'CLEARS ✓' if best_score > THRESHOLD else 'below'}")

    fold_scores = _per_fold_scores(probs_oof, y, weights, best_space)
    mean_f1 = float(np.mean(fold_scores))
    std_f1 = float(np.std(fold_scores, ddof=1))
    print(f"  Per-fold blend F1:  {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"  Folds: {[f'{s:.4f}' for s in fold_scores]}")

    oof_blend = _blend_threeway(probs_oof, weights, best_space).astype(np.float32)
    test_blend = _blend_threeway(probs_test, weights, best_space).astype(np.float32)

    rm = RunManager()
    run_id = "run_027"
    run_dir = rm.run_path(run_id)
    blend_params = {
        "base_runs": BASE_RUNS,
        "weights": {RUN_024_ID: w24, RUN_019_ID: w19, RUN_025_ID: w25},
        "blend_space": best_space,
        "solo_oof": solo,
        "proba_search": {"f1": f_proba, "weights": {RUN_024_ID: w24_p, RUN_019_ID: w19_p, RUN_025_ID: w25_p}},
        "logit_search": {"f1": f_logit, "weights": {RUN_024_ID: w24_l, RUN_019_ID: w19_l, RUN_025_ID: w25_l}},
    }
    if not run_dir.exists():
        rm.create_run(
            description="OOF 3-way blend run_024 + run_019 + run_025",
            model_type="ensemble_blend",
            feature_set="oof_proba_blend_3way",
            params=blend_params,
            run_id=run_id,
            objective="multiclass",
            n_features=None,
            cv_folds=CV_FOLDS,
            cv_metric="micro_f1",
            notes=f"Extends run_026 with run_025. Gain vs best solo {gain_vs_best_solo:+.4f}.",
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
    meta["weights"] = blend_params["weights"]
    meta["solo_oof"] = solo
    meta["gain_vs_best_solo"] = gain_vs_best_solo
    if f26 is not None:
        meta["solo_oof_run_026"] = f26
        meta["gain_vs_run_026"] = best_score - f26
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
