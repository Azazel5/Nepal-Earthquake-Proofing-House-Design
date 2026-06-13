#!/usr/bin/env python3
"""
run_030: G2 vs G3 binary specialist + run_026 blend.

Trains an LGBM specialist on grade-2/3 rows with foundation × geo ×
superstructure features (recall-oriented). Blends specialist conditional
P(G3 | not G1) with run_026 OOF probabilities, searches blend weight on OOF.

Run from project root:
    python src/run_030.py [--quick]
"""

from __future__ import annotations

import sys
import time
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g23_features import G23FeatureBuilder, load_g23_frames
from run_manager import PROCESSED_DIR, RunManager
from run_trees_260k import _oof_f1

print = partial(print, flush=True)

RANDOM_STATE = 42
CV_FOLDS = 5
RUN_ID = "run_030"
RUN_026_ID = "run_026"
GRADES = [1, 2, 3]
RECALL_BOOST = 1.35  # scale_pos_weight multiplier vs class ratio

LGBM_G23_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "max_depth": 8,
    "learning_rate": 0.05,
    "n_estimators": 800,
    "min_child_samples": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_jobs": 1,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}
EARLY_STOP = 50


def _micro_f1(y: np.ndarray, proba: np.ndarray) -> float:
    return f1_score(y, proba.argmax(axis=1) + 1, average="micro", labels=GRADES)


def _g3_recall(y: np.ndarray, proba: np.ndarray) -> float:
    pred = proba.argmax(axis=1) + 1
    return float(recall_score(y == 3, pred == 3))


def _combine_probs(
    p26: np.ndarray, sp_g3: np.ndarray, boost: float, tau: float,
) -> np.ndarray:
    """Keep run_026; upgrade G2→G3 when specialist exceeds threshold (recall-safe)."""
    out = np.asarray(p26, dtype=np.float32).copy()
    if boost <= 0.0:
        return out.astype(np.float64)
    sp = np.asarray(sp_g3, dtype=np.float32)
    pred = out.argmax(axis=1) + 1
    mass = out[:, 1] + out[:, 2]
    mask = (pred == 2) & (mass > 0.5) & (sp >= tau)
    if not np.any(mask):
        return out.astype(np.float64)
    m = mass[mask]
    q26 = out[mask, 2] / m
    q_new = q26 + boost * (sp[mask] - q26)
    q_new = np.clip(q_new, q26, 1.0)
    out[mask, 2] = m * q_new
    out[mask, 1] = m * (1.0 - q_new)
    out /= np.clip(out.sum(axis=1, keepdims=True), 1e-9, None)
    return out.astype(np.float64)


def search_blend_params(
    p26: np.ndarray,
    sp: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float, float]:
    best_boost, best_tau, best_f1, best_g3 = 0.0, 0.55, -1.0, 0.0
    for tau in np.linspace(0.45, 0.70, 11):
        for boost in np.linspace(0.05, 1.0, 20):
            proba = _combine_probs(p26, sp, float(boost), float(tau))
            f1 = _micro_f1(y, proba)
            g3 = _g3_recall(y, proba)
            if f1 > best_f1:
                best_f1, best_boost, best_tau, best_g3 = f1, float(boost), float(tau), g3
    return best_boost, best_tau, best_f1, best_g3


def train_specialist_oof(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    *,
    quick: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[float], G23FeatureBuilder]:
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(train, y))
    n_folds = 1 if quick else CV_FOLDS

    oof_sp = np.zeros(len(y), dtype=np.float32)
    test_folds: list[np.ndarray] = []
    scores: list[float] = []
    g23_mask = np.isin(y, [2, 3])

    for fold, (tri, vai) in enumerate(splits[:n_folds], start=1):
        t0 = time.time()
        fb = G23FeatureBuilder()
        fb.fit(train.iloc[tri], y[tri])

        tr_g23 = tri[np.isin(y[tri], [2, 3])]
        X_tr = fb.transform(train.iloc[tr_g23])
        y_tr = (y[tr_g23] == 3).astype(np.int8)
        n_pos = int(y_tr.sum())
        n_neg = len(y_tr) - n_pos
        spw = (n_neg / max(n_pos, 1)) * RECALL_BOOST

        X_va = fb.transform(train.iloc[vai])
        X_te = fb.transform(test)
        y_va_bin = (y[vai] == 3).astype(np.int8)
        va_g23 = np.isin(y[vai], [2, 3])

        model = LGBMClassifier(**LGBM_G23_PARAMS, scale_pos_weight=spw)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va[va_g23], y_va_bin[va_g23])],
            callbacks=[early_stopping(stopping_rounds=EARLY_STOP, verbose=False)],
        )
        oof_sp[vai] = model.predict_proba(X_va)[:, 1].astype(np.float32)
        test_folds.append(model.predict_proba(X_te)[:, 1].astype(np.float32))

        g3_rec = recall_score(y[vai] == 3, (oof_sp[vai] >= 0.5).astype(int))
        scores.append(g3_rec)
        print(f"  specialist fold {fold}: G3 recall@0.5={g3_rec:.4f}  ({time.time() - t0:.0f}s)")

    fb_full = G23FeatureBuilder()
    fb_full.fit(train, y)
    return oof_sp, np.mean(test_folds, axis=0), scores, fb_full


def fit_specialist_full(
    fb: G23FeatureBuilder,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    n_estimators: int,
) -> np.ndarray:
    g23 = np.isin(y, [2, 3])
    idx = np.where(g23)[0]
    X_tr = fb.transform(train.iloc[idx])
    y_tr = (y[idx] == 3).astype(np.int8)
    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    spw = (n_neg / max(n_pos, 1)) * RECALL_BOOST
    params = {**LGBM_G23_PARAMS, "n_estimators": n_estimators}
    model = LGBMClassifier(**params, scale_pos_weight=spw)
    model.fit(X_tr, y_tr)
    return model.predict_proba(fb.transform(test))[:, 1].astype(np.float32)


def _per_fold_blend_scores(
    p26: np.ndarray,
    sp: np.ndarray,
    y: np.ndarray,
    boost: float,
    tau: float,
) -> list[float]:
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores: list[float] = []
    for _fold, (_tr, va) in enumerate(skf.split(p26, y), start=1):
        proba = _combine_probs(p26[va], sp[va], boost, tau)
        scores.append(_micro_f1(y[va], proba))
    return scores


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--skip-lgbm",
        action="store_true",
        help="Reuse runs/run_030/oof_specialist_g3.npy if present",
    )
    args = parser.parse_args()
    t0 = time.time()

    train, test = load_g23_frames()
    y = pd.read_csv(PROCESSED_DIR / "y_train_full.csv")["damage_grade"].to_numpy()

    p26_oof = np.load(ROOT / "runs" / RUN_026_ID / "oof_proba.npy").astype(np.float64)
    p26_test = np.load(ROOT / "runs" / RUN_026_ID / "test_proba.npy").astype(np.float64)
    f26 = _oof_f1(p26_oof, y)
    g3_26 = _g3_recall(y, p26_oof)
    print(f"run_026 OOF: {f26:.4f}  G3 recall: {g3_26:.4f}")

    print("\n── G2 vs G3 specialist (OOF) ──")
    run_path = ROOT / "runs" / RUN_ID
    oof_sp_path = run_path / "oof_specialist_g3.npy"
    if args.skip_lgbm and oof_sp_path.exists():
        print(f"  Reusing {oof_sp_path}")
        oof_sp = np.load(oof_sp_path).astype(np.float32)
        test_sp_cv = np.zeros(len(test), dtype=np.float32)
        spec_scores = []
        fb_full = G23FeatureBuilder()
        fb_full.fit(train, y)
    else:
        oof_sp, test_sp_cv, spec_scores, fb_full = train_specialist_oof(
            train, test, y, quick=args.quick,
        )
    oof_sp64 = oof_sp.astype(np.float64)

    print("\n── Blend search (run_026 + specialist upgrades) ──")
    best_boost, best_tau, blend_f1, blend_g3 = search_blend_params(p26_oof, oof_sp64, y)
    print(f"  Best boost={best_boost:.2f}  tau={best_tau:.2f}")
    print(f"  Blended OOF F1: {blend_f1:.4f}  G3 recall: {blend_g3:.4f}")
    print(f"  Δ vs run_026:   F1 {blend_f1 - f26:+.4f}  G3 {blend_g3 - g3_26:+.4f}")

    if args.quick:
        print(f"\nQuick done ({time.time() - t0:.1f}s)")
        return

    # Refit specialist on all G2|G3 with avg best_iter from folds
    g23 = np.isin(y, [2, 3])
    idx = np.where(g23)[0]
    fb_full.fit(train, y)
    X_tr = fb_full.transform(train.iloc[idx])
    y_tr = (y[idx] == 3).astype(np.int8)
    spw = ((len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)) * RECALL_BOOST
    spec_model = LGBMClassifier(**LGBM_G23_PARAMS, scale_pos_weight=spw)
    spec_model.fit(X_tr, y_tr)
    test_sp = spec_model.predict_proba(fb_full.transform(test))[:, 1].astype(np.float32)

    test_proba = _combine_probs(p26_test, test_sp.astype(np.float64), best_boost, best_tau).astype(np.float32)
    oof_proba = _combine_probs(p26_oof, oof_sp64, best_boost, best_tau).astype(np.float32)

    fold_scores = _per_fold_blend_scores(p26_oof, oof_sp64, y, best_boost, best_tau)
    mean_f1 = float(np.mean(fold_scores))
    std_f1 = float(np.std(fold_scores, ddof=1))

    rm = RunManager()
    run_path = rm.run_path(RUN_ID)
    if not (run_path / "metadata.json").exists():
        rm.create_run(
            description="run_026 + G2|G3 specialist (foundation/geo/superstructure features)",
            model_type="ensemble_blend_specialist",
            feature_set="g23_engineered+lgbm",
            params={
                "base_run": RUN_026_ID,
                "upgrade_boost": best_boost,
                "upgrade_tau": best_tau,
                "recall_boost": RECALL_BOOST,
                "lgbm_g23": LGBM_G23_PARAMS,
            },
            run_id=RUN_ID,
            objective="multiclass",
            cv_folds=CV_FOLDS,
            cv_metric="micro_f1",
            notes="Specialist trained on G2|G3 only; run_026 P(G1) preserved.",
        )

    joblib.dump(spec_model, run_path / "g23_specialist.pkl")
    joblib.dump(fb_full, run_path / "g23_feature_builder.pkl")
    rm.save_cv_scores(RUN_ID, fold_scores, mean_f1, std_f1)
    np.save(run_path / "oof_proba.npy", oof_proba)
    np.save(run_path / "oof_specialist_g3.npy", oof_sp)
    np.save(run_path / "test_proba.npy", test_proba)

    test_csv = pd.read_csv(ROOT / "data" / "driven_data" / "test_values.csv")
    sub = pd.DataFrame({
        "building_id": test_csv["building_id"].values,
        "damage_grade": test_proba.argmax(axis=1) + 1,
    })
    rm.save_submission(RUN_ID, sub)

    meta = rm.load_metadata(RUN_ID)
    meta["run_026_oof"] = f26
    meta["run_026_g3_recall"] = g3_26
    meta["blend_oof_f1"] = blend_f1
    meta["blend_g3_recall"] = blend_g3
    meta["gain_f1_vs_run_026"] = blend_f1 - f26
    meta["gain_g3_recall_vs_run_026"] = blend_g3 - g3_26
    meta["blend_boost"] = best_boost
    meta["blend_tau"] = best_tau
    RunManager._write_json(run_path / "metadata.json", meta)

    print("\n" + "═" * 60)
    print(f"  run_026:        F1={f26:.4f}  G3 recall={g3_26:.4f}")
    print(f"  run_030 blend:  F1={blend_f1:.4f}  G3 recall={blend_g3:.4f}")
    print(f"  Per-fold:       {mean_f1:.4f} ± {std_f1:.4f}")
    grades, counts = np.unique(sub["damage_grade"], return_counts=True)
    print(f"\n── {RUN_ID} submission ──")
    for g, c in zip(grades, counts):
        print(f"  grade {g}: {c:,} ({c/len(sub)*100:.1f}%)")
    print(f"\nRegistered {RUN_ID} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
