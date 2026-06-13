#!/usr/bin/env python3
"""
run_031: Residual G2|G3 specialist — trained only where run_026 predicts G2.

Uses residual-specific features (plan/position, weak-foundation × geo combos,
surprise vs run_026 conditional) and AUC-optimized LGBM. Blends upgrades into
run_026 OOF/test probabilities.

Run from project root:
    python src/run_031.py [--quick] [--skip-lgbm]
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
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from residual_g23_features import ResidualG23FeatureBuilder, load_residual_frames
from run_manager import PROCESSED_DIR, RunManager
from run_trees_260k import _oof_f1

print = partial(print, flush=True)

RANDOM_STATE = 42
CV_FOLDS = 5
RUN_ID = "run_031"
RUN_026_ID = "run_026"
GRADES = [1, 2, 3]

LGBM_RESIDUAL_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": 6,
    "learning_rate": 0.04,
    "n_estimators": 1200,
    "min_child_samples": 60,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "reg_alpha": 0.2,
    "reg_lambda": 0.2,
    "n_jobs": 1,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}
EARLY_STOP = 60


def _micro_f1(y: np.ndarray, proba: np.ndarray) -> float:
    return f1_score(y, proba.argmax(axis=1) + 1, average="micro", labels=GRADES)


def _g3_recall(y: np.ndarray, proba: np.ndarray) -> float:
    pred = proba.argmax(axis=1) + 1
    return float(recall_score(y == 3, pred == 3, zero_division=0))


from scipy.stats import rankdata


def _pred_g2(proba: np.ndarray) -> np.ndarray:
    return proba.argmax(axis=1) + 1 == 2


def _rank_on_mask(sp: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Percentile rank on masked rows; 0.5 elsewhere (blend ignores non-G2)."""
    out = np.full(len(sp), 0.5, dtype=np.float32)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return out
    out[idx] = (rankdata(sp[idx]) / len(idx)).astype(np.float32)
    return out


def _residual_mask(proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    return _pred_g2(proba) & np.isin(y, [2, 3])


def _combine_probs(
    p26: np.ndarray,
    sp_g3: np.ndarray,
    boost: float,
    tau: float,
    delta_min: float,
) -> np.ndarray:
    """Upgrade G2→G3 when residual specialist exceeds tau and beats run_026."""
    out = np.asarray(p26, dtype=np.float32).copy()
    if boost <= 0.0:
        return out.astype(np.float64)
    sp = np.asarray(sp_g3, dtype=np.float32)
    pred = out.argmax(axis=1) + 1
    mass = out[:, 1] + out[:, 2]
    q26 = np.divide(out[:, 2], mass, out=np.full(len(out), 0.5, dtype=np.float32), where=mass > 1e-9)
    mask = (pred == 2) & (mass > 0.5) & (sp >= tau) & (sp >= q26 + delta_min)
    if not np.any(mask):
        return out.astype(np.float64)
    m = mass[mask]
    q_new = q26[mask] + boost * (sp[mask] - q26[mask])
    q_new = np.clip(q_new, q26[mask], 1.0)
    out[mask, 2] = m * q_new
    out[mask, 1] = m * (1.0 - q_new)
    out /= np.clip(out.sum(axis=1, keepdims=True), 1e-9, None)
    return out.astype(np.float64)


def search_blend_params(
    p26: np.ndarray,
    sp: np.ndarray,
    y: np.ndarray,
    f26: float,
) -> tuple[float, float, float, float, float, float]:
    g3_base = _g3_recall(y, p26)
    best_boost, best_tau, best_delta = 0.0, 1.0, 0.0
    best_f1, best_g3 = f26, g3_base
    for delta_min in (0.0, 0.03, 0.06, 0.10):
        for tau in np.linspace(0.70, 0.98, 15):
            for boost in np.linspace(0.05, 0.50, 10):
                proba = _combine_probs(p26, sp, float(boost), float(tau), float(delta_min))
                f1 = _micro_f1(y, proba)
                g3 = _g3_recall(y, proba)
                if f1 > best_f1 + 1e-9 or (abs(f1 - best_f1) < 1e-6 and g3 > best_g3 + 1e-9):
                    best_f1, best_boost, best_tau, best_delta, best_g3 = (
                        f1, float(boost), float(tau), float(delta_min), g3,
                    )
    return best_boost, best_tau, best_delta, best_f1, best_g3, best_f1 - f26


def train_residual_specialist_oof(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    p26_oof: np.ndarray,
    p26_test: np.ndarray,
    *,
    quick: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict], ResidualG23FeatureBuilder]:
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(train, y))
    n_folds = 1 if quick else CV_FOLDS

    pred26 = _pred_g2(p26_oof)
    oof_sp = np.full(len(y), np.nan, dtype=np.float32)
    test_folds: list[np.ndarray] = []
    fold_metrics: list[dict] = []

    for fold, (tri, vai) in enumerate(splits[:n_folds], start=1):
        t0 = time.time()
        tri_res = tri[pred26[tri] & np.isin(y[tri], [2, 3])]
        if len(tri_res) < 100:
            raise RuntimeError(f"fold {fold}: too few residual training rows ({len(tri_res)})")

        fb = ResidualG23FeatureBuilder()
        tri_mask = np.zeros(len(y), dtype=bool)
        tri_mask[tri] = True
        fb.fit(train, y, residual_mask=tri_mask & pred26)

        X_tr = fb.transform(train.iloc[tri_res], p26_oof[tri_res])
        y_tr = (y[tri_res] == 3).astype(np.int8)

        va_res = vai[pred26[vai] & np.isin(y[vai], [2, 3])]
        X_va = fb.transform(train.iloc[va_res], p26_oof[va_res])
        y_va = (y[va_res] == 3).astype(np.int8)

        spw = (len(y_tr) - y_tr.sum()) / max(int(y_tr.sum()), 1)
        model = LGBMClassifier(**LGBM_RESIDUAL_PARAMS, scale_pos_weight=spw)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[early_stopping(stopping_rounds=EARLY_STOP, verbose=False)],
        )

        oof_sp[vai] = model.predict_proba(
            fb.transform(train.iloc[vai], p26_oof[vai]),
        )[:, 1].astype(np.float32)
        test_folds.append(
            model.predict_proba(fb.transform(test, p26_test))[:, 1].astype(np.float32),
        )

        va_auc = float(roc_auc_score(y_va, oof_sp[va_res])) if len(np.unique(y_va)) > 1 else 0.5
        miss = (y[vai] == 3) & (p26_oof[vai].argmax(1) + 1 == 2)
        hit = miss & (oof_sp[vai] >= 0.5)
        miss_auc = (
            float(roc_auc_score(y[va_res] == 3, oof_sp[va_res]))
            if len(va_res) > 0 and len(np.unique(y[va_res])) > 1
            else 0.5
        )
        fold_metrics.append({
            "fold": fold,
            "auc_residual": va_auc,
            "auc_all_residual": miss_auc,
            "n_train": len(tri_res),
            "g3_miss_recall@0.5": float(hit.sum() / max(miss.sum(), 1)),
            "n_miss_g3": int(miss.sum()),
        })
        print(
            f"  fold {fold}: AUC={va_auc:.4f}  train_n={len(tri_res):,}  "
            f"G3-miss recall@0.5={fold_metrics[-1]['g3_miss_recall@0.5']:.3f}  "
            f"({time.time() - t0:.0f}s)",
        )

    fb_full = ResidualG23FeatureBuilder()
    full_mask = pred26 & np.isin(y, [2, 3])
    fb_full.fit(train, y, residual_mask=full_mask)
    return oof_sp, np.mean(test_folds, axis=0), fold_metrics, fb_full


def _per_fold_blend_scores(
    p26: np.ndarray,
    sp: np.ndarray,
    y: np.ndarray,
    boost: float,
    tau: float,
    delta_min: float,
) -> list[float]:
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores: list[float] = []
    for _fold, (_tr, va) in enumerate(skf.split(p26, y), start=1):
        proba = _combine_probs(p26[va], sp[va], boost, tau, delta_min)
        scores.append(_micro_f1(y[va], proba))
    return scores


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-lgbm", action="store_true")
    args = parser.parse_args()
    t0 = time.time()

    train, test = load_residual_frames()
    y = pd.read_csv(PROCESSED_DIR / "y_train_full.csv")["damage_grade"].to_numpy()

    p26_oof = np.load(ROOT / "runs" / RUN_026_ID / "oof_proba.npy").astype(np.float64)
    p26_test = np.load(ROOT / "runs" / RUN_026_ID / "test_proba.npy").astype(np.float64)
    f26 = _oof_f1(p26_oof, y)
    g3_26 = _g3_recall(y, p26_oof)

    pred26 = _pred_g2(p26_oof)
    res_mask = _residual_mask(p26_oof, y)
    print(f"run_026 OOF: {f26:.4f}  G3 recall: {g3_26:.4f}")
    print(f"Residual pool (pred G2, true G2|G3): {res_mask.sum():,} rows")
    print(f"  true G3 in pool: {(res_mask & (y == 3)).sum():,}  true G2: {(res_mask & (y == 2)).sum():,}")

    run_path = ROOT / "runs" / RUN_ID
    oof_sp_path = run_path / "oof_specialist_residual.npy"

    print("\n── Residual specialist (OOF) ──")
    if args.skip_lgbm and oof_sp_path.exists():
        print(f"  Reusing {oof_sp_path}")
        oof_sp = np.load(oof_sp_path).astype(np.float32)
        fold_metrics = []
        fb_full = ResidualG23FeatureBuilder()
        fb_full.fit(train, y, residual_mask=pred26 & np.isin(y, [2, 3]))
        test_sp_cv = np.zeros(len(test), dtype=np.float32)
    else:
        oof_sp, test_sp_cv, fold_metrics, fb_full = train_residual_specialist_oof(
            train, test, y, p26_oof, p26_test, quick=args.quick,
        )

    valid = ~np.isnan(oof_sp)
    res_oof = res_mask & valid
    if res_oof.sum() and len(np.unique(y[res_oof])) > 1:
        auc_pool = roc_auc_score(y[res_oof] == 3, oof_sp[res_oof])
        print(f"\nOOF AUC on residual pool: {auc_pool:.4f}")
        miss = (y == 3) & pred26
        print(f"G3→G2 errors with sp>=0.5: {(miss & (oof_sp >= 0.5)).sum()}/{miss.sum()}")

    oof_sp64 = np.nan_to_num(oof_sp, nan=0.5).astype(np.float64)
    oof_sp_blend = _rank_on_mask(oof_sp64.astype(np.float32), pred26).astype(np.float64)

    print("\n── Blend search (run_026 + residual rank upgrades) ──")
    boost, tau, delta_min, blend_f1, blend_g3, gain_f1 = search_blend_params(
        p26_oof, oof_sp_blend, y, f26,
    )
    print(f"  Best boost={boost:.2f}  tau={tau:.2f}  delta_min={delta_min:.2f}")
    print(f"  Blended OOF F1: {blend_f1:.4f}  G3 recall: {blend_g3:.4f}")
    print(f"  Δ vs run_026:   F1 {blend_f1 - f26:+.4f}  G3 {blend_g3 - g3_26:+.4f}")

    if args.quick:
        print(f"\nQuick done ({time.time() - t0:.1f}s)")
        return

    # Full refit on all residual rows
    idx = np.where(pred26 & np.isin(y, [2, 3]))[0]
    X_full = fb_full.transform(train.iloc[idx], p26_oof[idx])
    y_full = (y[idx] == 3).astype(np.int8)
    spw = (len(y_full) - y_full.sum()) / max(int(y_full.sum()), 1)
    spec_model = LGBMClassifier(**LGBM_RESIDUAL_PARAMS, scale_pos_weight=spw)
    spec_model.fit(X_full, y_full)

    test_pred_g2 = _pred_g2(p26_test)
    test_sp = np.full(len(test), 0.5, dtype=np.float32)
    te_p = spec_model.predict_proba(fb_full.transform(test, p26_test))[:, 1].astype(np.float32)
    test_sp[test_pred_g2] = te_p[test_pred_g2]
    test_sp_blend = _rank_on_mask(test_sp, test_pred_g2).astype(np.float64)

    test_proba = _combine_probs(
        p26_test, test_sp_blend, boost, tau, delta_min,
    ).astype(np.float32)
    oof_proba = _combine_probs(
        p26_oof, oof_sp_blend, boost, tau, delta_min,
    ).astype(np.float32)

    fold_scores = _per_fold_blend_scores(p26_oof, oof_sp_blend, y, boost, tau, delta_min)
    mean_f1 = float(np.mean(fold_scores))
    std_f1 = float(np.std(fold_scores, ddof=1))

    rm = RunManager()
    run_path = rm.run_path(RUN_ID)
    if not (run_path / "metadata.json").exists():
        rm.create_run(
            description="run_026 + residual G2|G3 specialist (pred-G2 only)",
            model_type="ensemble_residual_specialist",
            feature_set="residual_g23_engineered+lgbm",
            params={
                "base_run": RUN_026_ID,
                "upgrade_boost": boost,
                "upgrade_tau": tau,
                "delta_min": delta_min,
                "lgbm": LGBM_RESIDUAL_PARAMS,
            },
            run_id=RUN_ID,
            objective="multiclass",
            cv_folds=CV_FOLDS,
            cv_metric="micro_f1",
            notes="Specialist trained only on run_026 pred-G2 rows; AUC objective.",
        )

    joblib.dump(spec_model, run_path / "residual_specialist.pkl")
    joblib.dump(fb_full, run_path / "residual_feature_builder.pkl")
    rm.save_cv_scores(RUN_ID, fold_scores, mean_f1, std_f1)
    np.save(run_path / "oof_proba.npy", oof_proba)
    np.save(oof_sp_path, oof_sp)
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
    meta["blend_boost"] = boost
    meta["blend_tau"] = tau
    meta["delta_min"] = delta_min
    meta["residual_pool_n"] = int(res_mask.sum())
    meta["specialist_fold_metrics"] = fold_metrics
    if res_mask.sum():
        meta["residual_auc_oof"] = float(roc_auc_score(y[res_mask] == 3, oof_sp[res_mask]))
    meta["blend_uses_rank"] = True
    RunManager._write_json(run_path / "metadata.json", meta)

    print("\n" + "═" * 60)
    print(f"  run_026:        F1={f26:.4f}  G3 recall={g3_26:.4f}")
    print(f"  run_031 blend:  F1={blend_f1:.4f}  G3 recall={blend_g3:.4f}")
    print(f"  Per-fold:       {mean_f1:.4f} ± {std_f1:.4f}")
    grades, counts = np.unique(sub["damage_grade"], return_counts=True)
    print(f"\n── {RUN_ID} submission ──")
    for g, c in zip(grades, counts):
        print(f"  grade {g}: {c:,} ({c/len(sub)*100:.1f}%)")
    print(f"\nRegistered {RUN_ID} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
