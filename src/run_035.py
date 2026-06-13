#!/usr/bin/env python3
"""
run_035: CV-safe foundation×geo3 rate features + run_026 retrain (no PL).

Adds Laplace-smoothed P(grade | geo3, foundation_type) and weak×G3-rate
interaction to run_024 XGB and run_019 LGBM, then blends like run_026.

Run from project root:
    python src/run_035.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import xgboost as xgb  # noqa: E402

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from found_geo_rates import FOUND_GEO_FEATURE_COLS, cv_found_geo_rates, fulltrain_found_geo_rates, rates_to_array
from retrain import EARLY_STOPPING_ROUNDS, TRIAL_66_PARAMS
from run_024_features import ShoumikFeatureBuilder, load_frames, load_geo_latents
from run_manager import PROCESSED_DIR, RunManager
from run_trees_260k import _blend_logit, _blend_proba, _oof_f1, pairwise_diagnostic

print = partial(print, flush=True)

RUN_ID = "run_035"
SOTA_RUN = "run_026"
SOTA_OOF = 0.7546
NOISE = 0.0016
THRESHOLD = SOTA_OOF + NOISE
RANDOM_STATE = 42
CV_FOLDS = 5
PCA_K = 80

SHOUMIK_XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
    "learning_rate": 0.01306,
    "max_depth": 15,
    "n_estimators": 756,
    "gamma": 1.37,
    "min_child_weight": 7,
    "reg_alpha": 0.018,
    "reg_lambda": 0.059,
    "subsample": 0.808,
    "colsample_bytree": 0.528,
    "colsample_bylevel": 0.835,
    "colsample_bynode": 0.564,
}
EARLY_STOP = 50


class ShoumikFoundGeoFeatureBuilder(ShoumikFeatureBuilder):
    def transform(
        self,
        df: pd.DataFrame,
        geo_idx: np.ndarray,
        geo_dr: np.ndarray,
        geo_ru: np.ndarray,
        found_rates: np.ndarray,
    ) -> np.ndarray:
        base = super().transform(df, geo_idx, geo_dr, geo_ru)
        return np.hstack([base, found_rates]).astype(np.float32)


def _micro_f1(y_true: np.ndarray, proba: np.ndarray) -> float:
    return f1_score(y_true, proba.argmax(axis=1) + 1, average="micro", labels=[1, 2, 3])


def _per_grade_table(y: np.ndarray, proba: np.ndarray) -> dict:
    pred = proba.argmax(axis=1) + 1
    p, r, f1, sup = precision_recall_fscore_support(
        y, pred, labels=[1, 2, 3], zero_division=0,
    )
    return {
        str(g): {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]), "support": int(sup[i])}
        for i, g in enumerate([1, 2, 3])
    }


def _fit_xgb_fold(
    X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray,
) -> tuple[xgb.XGBClassifier, int]:
    model = xgb.XGBClassifier(**SHOUMIK_XGB_PARAMS, early_stopping_rounds=EARLY_STOP)
    model.fit(X_tr, y_tr - 1, eval_set=[(X_va, y_va - 1)], verbose=False)
    best_iter = model.best_iteration if model.best_iteration is not None else SHOUMIK_XGB_PARAMS["n_estimators"] - 1
    return model, int(best_iter) + 1


def run_xgb_cv(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    geo_train: np.ndarray,
    geo_test: np.ndarray,
    dr_train: np.ndarray,
    dr_test: np.ndarray,
    ru_train: np.ndarray,
    ru_test: np.ndarray,
    rates_train: np.ndarray,
    rates_test: np.ndarray,
    *,
    quick: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(train, y))
    n_folds = 1 if quick else CV_FOLDS
    oof = np.zeros((len(y), 3), dtype=np.float32)
    test_folds: list[np.ndarray] = []
    scores: list[float] = []

    for fold, (tri, vai) in enumerate(splits[:n_folds], start=1):
        t0 = time.time()
        fb = ShoumikFoundGeoFeatureBuilder()
        fb.fit(train.iloc[tri])
        X_tr = fb.transform(train.iloc[tri], geo_train[tri], dr_train[tri], ru_train[tri], rates_train[tri])
        X_va = fb.transform(train.iloc[vai], geo_train[vai], dr_train[vai], ru_train[vai], rates_train[vai])
        X_te = fb.transform(test, geo_test, dr_test, ru_test, rates_test)
        model, _ = _fit_xgb_fold(X_tr, y[tri], X_va, y[vai])
        oof[vai] = model.predict_proba(X_va).astype(np.float32)
        test_folds.append(model.predict_proba(X_te).astype(np.float32))
        f1 = _micro_f1(y[vai], oof[vai])
        scores.append(f1)
        print(f"  XGB fold {fold}: F1={f1:.4f}  ({time.time() - t0:.0f}s)")

    return oof, np.mean(test_folds, axis=0), scores


def _embed_cols(cols: list[str]) -> list[str]:
    return [c for c in cols if "_emb_" in c]


def _non_embed_cols(cols: list[str]) -> list[str]:
    return [c for c in cols if "_emb_" not in c]


def _transform_pca(
    X_tr: pd.DataFrame, X_va: pd.DataFrame, X_te: pd.DataFrame,
    pca_cols: list[str], pass_cols: list[str], n_components: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE, svd_solver="randomized")

    def _scale_train(df: pd.DataFrame) -> np.ndarray:
        arr = scaler.fit_transform(df[pca_cols].to_numpy(dtype=np.float64))
        return np.clip(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)

    def _scale_apply(df: pd.DataFrame) -> np.ndarray:
        arr = scaler.transform(df[pca_cols].to_numpy(dtype=np.float64))
        return np.clip(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)

    z_tr = pca.fit_transform(_scale_train(X_tr))
    z_va = pca.transform(_scale_apply(X_va))
    z_te = pca.transform(_scale_apply(X_te))
    pca_names = [f"pca_{i}" for i in range(n_components)]
    if pass_cols:
        z_tr = np.hstack([z_tr, X_tr[pass_cols].to_numpy()])
        z_va = np.hstack([z_va, X_va[pass_cols].to_numpy()])
        z_te = np.hstack([z_te, X_te[pass_cols].to_numpy()])
        col_names = pca_names + pass_cols
    else:
        col_names = pca_names
    return (
        pd.DataFrame(z_tr, columns=col_names),
        pd.DataFrame(z_va, columns=col_names),
        pd.DataFrame(z_te, columns=col_names),
    )


def _fit_lgbm_fold(X_tr: pd.DataFrame, y_tr: np.ndarray, X_va: pd.DataFrame, y_va: np.ndarray) -> LGBMClassifier:
    model = LGBMClassifier(**TRIAL_66_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="multi_logloss",
        callbacks=[early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model


def run_lgbm_cv(
    X: pd.DataFrame, X_test: pd.DataFrame, y: np.ndarray, *, quick: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    cols = list(X.columns)
    pca_cols, pass_cols = _embed_cols(cols), _non_embed_cols(cols)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(X, y))
    n_folds = 1 if quick else CV_FOLDS
    oof = np.zeros((len(y), 3), dtype=np.float32)
    test_folds: list[np.ndarray] = []
    scores: list[float] = []

    for fold, (tri, vai) in enumerate(splits[:n_folds], start=1):
        t0 = time.time()
        df_tr, df_va, df_te = _transform_pca(
            X.iloc[tri], X.iloc[vai], X_test, pca_cols, pass_cols, PCA_K,
        )
        model = _fit_lgbm_fold(df_tr, y[tri], df_va, y[vai])
        oof[vai] = model.predict_proba(df_va).astype(np.float32)
        test_folds.append(model.predict_proba(df_te).astype(np.float32))
        f1 = _micro_f1(y[vai], oof[vai])
        scores.append(f1)
        print(f"  LGBM fold {fold}: F1={f1:.4f}  ({time.time() - t0:.0f}s)")

    return oof, np.mean(test_folds, axis=0), scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    t0 = time.time()

    train, test = load_frames()
    y = pd.read_csv(PROCESSED_DIR / "y_train_full.csv")["damage_grade"].to_numpy()

    print("═" * 72)
    print("run_035 — CV-safe foundation×geo3 rate features (no PL)")
    print("═" * 72)
    print(f"  Features: {FOUND_GEO_FEATURE_COLS}")

    rates_train = rates_to_array(cv_found_geo_rates(train, y))
    rates_test = rates_to_array(fulltrain_found_geo_rates(train, test, y))
    print(f"  Rate stats — foundg3_p_grade3: mean={rates_train[:, 2].mean():.3f}  "
          f"weak×p3: mean={rates_train[:, 3].mean():.3f}")

    print("\n── XGB (run_024 + found-geo rates) ──")
    geo_train, geo_test, dr_train, dr_test, ru_train, ru_test = load_geo_latents()
    xgb_oof, xgb_test, xgb_scores = run_xgb_cv(
        train, test, y, geo_train, geo_test, dr_train, dr_test, ru_train, ru_test,
        rates_train, rates_test, quick=args.quick,
    )
    print(f"  XGB solo OOF: {_oof_f1(xgb_oof, y):.4f}")

    print("\n── LGBM (run_019 embed PCA + found-geo rates) ──")
    X = pd.read_csv(PROCESSED_DIR / "X_train_run012.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test_run012.csv")
    rate_df_tr = cv_found_geo_rates(train, y)
    rate_df_te = fulltrain_found_geo_rates(train, test, y)
    X = pd.concat([X.reset_index(drop=True), rate_df_tr.reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_test.reset_index(drop=True), rate_df_te.reset_index(drop=True)], axis=1)
    print(f"  Matrix: {X.shape[1]} cols (+{len(FOUND_GEO_FEATURE_COLS)} rate features)")
    lgbm_oof, lgbm_test, lgbm_scores = run_lgbm_cv(X, X_test, y, quick=args.quick)
    print(f"  LGBM solo OOF: {_oof_f1(lgbm_oof, y):.4f}")

    if args.quick:
        print(f"\nQuick done in {time.time() - t0:.1f}s")
        return

    p26 = np.load(ROOT / "runs" / SOTA_RUN / "oof_proba.npy")
    f26 = _oof_f1(p26, y)
    print("\n── Blend (run_026 harness) ──")
    print(f"  run_026 baseline: {f26:.4f}")
    blend = pairwise_diagnostic("run_035_xgb", xgb_oof.astype(np.float64), "run_035_lgbm", lgbm_oof.astype(np.float64), y)
    alpha, space = blend["best_alpha"], blend["best_space"]
    blend_fn = _blend_proba if space == "proba" else _blend_logit
    oof_blend = blend_fn(alpha, xgb_oof.astype(np.float64), lgbm_oof.astype(np.float64)).astype(np.float32)
    test_blend = blend_fn(alpha, xgb_test.astype(np.float64), lgbm_test.astype(np.float64)).astype(np.float32)
    blend_f1 = _oof_f1(oof_blend, y)
    delta = blend_f1 - f26

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = [
        _micro_f1(y[va], blend_fn(alpha, xgb_oof[va].astype(np.float64), lgbm_oof[va].astype(np.float64)))
        for _tr, va in skf.split(y, y)
    ]

    print(f"\n  run_035 blend OOF: {blend_f1:.4f}  (Δ vs run_026: {delta:+.4f})")
    print(f"  Threshold {THRESHOLD:.4f}: {'SUBMIT ✓' if blend_f1 > THRESHOLD else 'hold'}")

    grade_26 = _per_grade_table(y, p26)
    grade_35 = _per_grade_table(y, oof_blend)
    print("\n  Per-grade (run_026 → run_035):")
    for g in ["1", "2", "3"]:
        a, b = grade_26[g], grade_35[g]
        print(f"    G{g}: prec {a['precision']:.3f}→{b['precision']:.3f}  rec {a['recall']:.3f}→{b['recall']:.3f}")

    rm = RunManager()
    rm.create_run(
        description="Foundation×geo3 CV-smoothed rates + run_026 retrain (no PL)",
        model_type="ensemble_blend",
        feature_set="found_geo3_rates+run_024_xgb+run_019_lgbm",
        params={
            "rate_features": FOUND_GEO_FEATURE_COLS,
            "laplace_alpha": 5,
            "blend_alpha_xgb": alpha,
            "blend_space": space,
            "solo_oof": {"xgb": _oof_f1(xgb_oof, y), "lgbm": _oof_f1(lgbm_oof, y), SOTA_RUN: f26},
        },
        run_id=RUN_ID,
        objective="multiclass",
        n_features=int(X.shape[1]),
        cv_folds=CV_FOLDS,
        cv_metric="micro_f1",
        notes="Last live signal hypothesis; precision-safe rate features only.",
    )
    rm.save_cv_scores(RUN_ID, fold_scores, float(np.mean(fold_scores)), float(np.std(fold_scores, ddof=1)))
    run_dir = rm.run_path(RUN_ID)
    np.save(run_dir / "oof_proba.npy", oof_blend)
    np.save(run_dir / "test_proba.npy", test_blend)
    np.save(run_dir / "oof_xgb.npy", xgb_oof)
    np.save(run_dir / "oof_lgbm.npy", lgbm_oof)
    with open(run_dir / "eval_report.json", "w") as f:
        json.dump({
            "blend_oof_f1": blend_f1,
            "delta_vs_run_026": delta,
            "threshold": THRESHOLD,
            "submit": blend_f1 > THRESHOLD,
            "per_grade_run_026": grade_26,
            "per_grade_run_035": grade_35,
        }, f, indent=2)

    test_csv = pd.read_csv(ROOT / "data" / "driven_data" / "test_values.csv")
    sub = pd.DataFrame({"building_id": test_csv["building_id"].values, "damage_grade": test_blend.argmax(axis=1) + 1})
    rm.save_submission(RUN_ID, sub)
    print(f"\nRegistered {RUN_ID} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
