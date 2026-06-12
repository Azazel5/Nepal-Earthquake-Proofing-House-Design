#!/usr/bin/env python3
"""
run_025: Full Shoumik stack — run_024 + OOF CatBoostEncoder on geo only.

Same geo AE latents, XGB hyperparams, and CV protocol as run_024. Adds
CatBoostEncoder(cols=geo_level_*) fitted per fold (OOF-safe).

Reuses geo latents from models/shoumik_run024/ (run_024_geo.py).

Run from project root:
    python src/run_025.py [--quick] [--xgb-verbose]

Logs every sub-step with timestamps so hangs are easy to spot. CatBoostEncoder
on high-cardinality geo_level_3_id is often the slow step (~minutes per fold).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import xgboost as xgb  # noqa: E402 — before torch via run_024_features

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_manager import PROCESSED_DIR, RunManager
from run_024_features import (
    ShoumikFullFeatureBuilder,
    geo_latents_ready,
    load_frames,
    load_geo_latents,
)
from run_trees_260k import (
    _oof_f1,
    build_blend_submission,
    pairwise_diagnostic,
    threeway_optimize,
)

print = partial(print, flush=True)

RANDOM_STATE = 42
CV_FOLDS = 5
RUN_ID = "run_025"
LGBM_CV_REF = 0.7588
NOISE = 0.0016
THRESHOLD = LGBM_CV_REF + NOISE

LGBM_OOF = ROOT / "runs" / "run_012" / "oof_proba.npy"
LGBM_TEST = ROOT / "runs" / "run_012" / "test_proba.npy"
RUN024_OOF = ROOT / "runs" / "run_024" / "oof_proba.npy"
RUN024_TEST = ROOT / "runs" / "run_024" / "test_proba.npy"
RUN019_OOF = ROOT / "runs" / "run_019" / "oof_proba.npy"
RUN019_TEST = ROOT / "runs" / "run_019" / "test_proba.npy"

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

_T0 = time.time()


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str, *, fold: int | None = None, since: float | None = None) -> None:
    prefix = f"[{_ts()}]"
    if fold is not None:
        prefix += f" fold {fold}"
    if since is not None:
        prefix += f" +{time.time() - since:.1f}s"
    prefix += f" +{time.time() - _T0:.0f}s total"
    print(f"{prefix} | {msg}")


def _micro_f1(y_true: np.ndarray, proba: np.ndarray) -> float:
    return f1_score(y_true, proba.argmax(axis=1) + 1, average="micro", labels=[1, 2, 3])


def _fit_xgb_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    *,
    fold: int,
    xgb_verbose: bool,
) -> tuple[xgb.XGBClassifier, int]:
    log(
        f"XGB fit  train={X_tr.shape}  val={X_va.shape}  n_estimators={SHOUMIK_XGB_PARAMS['n_estimators']}",
        fold=fold,
    )
    t_fit = time.time()
    model = xgb.XGBClassifier(**SHOUMIK_XGB_PARAMS, early_stopping_rounds=EARLY_STOP)
    model.fit(
        X_tr,
        y_tr - 1,
        eval_set=[(X_va, y_va - 1)],
        verbose=10 if xgb_verbose else False,
    )
    best_iter = model.best_iteration if model.best_iteration is not None else SHOUMIK_XGB_PARAMS["n_estimators"] - 1
    log(f"XGB fit done  best_iter={best_iter + 1}", fold=fold, since=t_fit)
    return model, int(best_iter) + 1


def run_cv(
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
    xgb_verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[float], list[int]]:
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(train, y))
    n_folds = 1 if quick else CV_FOLDS
    log(f"CV: {n_folds} fold(s)  train={len(y):,}  test={len(test):,}")

    oof = np.zeros((len(y), 3), dtype=np.float32)
    test_folds: list[np.ndarray] = []
    scores: list[float] = []
    best_iters: list[int] = []

    for fold, (tri, vai) in enumerate(splits[:n_folds], start=1):
        t_fold = time.time()
        log(f"── fold {fold} start  train={len(tri):,}  val={len(vai):,}", fold=fold)

        fb = ShoumikFullFeatureBuilder()
        fold_log = lambda m, f=fold: log(m, fold=f, since=t_fold)

        t_feat = time.time()
        log("feature fit start", fold=fold)
        fb.fit(train.iloc[tri], y[tri], log_fn=fold_log)
        log("feature fit done", fold=fold, since=t_feat)

        t_tr = time.time()
        X_tr = fb.transform(
            train.iloc[tri], geo_train[tri], dr_train[tri], ru_train[tri],
            log_fn=fold_log, label="train",
        )
        log("train transform done", fold=fold, since=t_tr)

        t_va = time.time()
        X_va = fb.transform(
            train.iloc[vai], geo_train[vai], dr_train[vai], ru_train[vai],
            log_fn=fold_log, label="val",
        )
        log("val transform done", fold=fold, since=t_va)

        t_te = time.time()
        X_te = fb.transform(
            test, geo_test, dr_test, ru_test,
            log_fn=fold_log, label="test",
        )
        log("test transform done", fold=fold, since=t_te)

        model, n_iter = _fit_xgb_fold(
            X_tr, y[tri], X_va, y[vai], fold=fold, xgb_verbose=xgb_verbose,
        )

        t_pred = time.time()
        log("predict_proba val", fold=fold)
        oof[vai] = model.predict_proba(X_va).astype(np.float32)
        log("predict_proba test", fold=fold)
        test_folds.append(model.predict_proba(X_te).astype(np.float32))
        log("predict done", fold=fold, since=t_pred)

        best_iters.append(n_iter)
        f1 = _micro_f1(y[vai], oof[vai])
        scores.append(f1)
        log(f"fold {fold} complete  F1={f1:.4f}  best_iter={n_iter}", fold=fold, since=t_fold)

    test_avg = np.mean(test_folds, axis=0)
    return oof, test_avg, scores, best_iters


def fit_full_train(
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
) -> tuple[xgb.XGBClassifier, np.ndarray]:
    log("── full-train refit start")
    fb = ShoumikFullFeatureBuilder()
    full_log = lambda m: log(m, since=_T0)
    t0 = time.time()
    fb.fit(train, y, log_fn=full_log)
    log("full-train feature fit done", since=t0)
    X_full = fb.transform(train, geo_train, dr_train, ru_train, log_fn=full_log, label="full")
    X_te = fb.transform(test, geo_test, dr_test, ru_test, log_fn=full_log, label="test")
    params = {**SHOUMIK_XGB_PARAMS, "n_estimators": n_estimators}
    log(f"XGB full fit  n_estimators={n_estimators}  X={X_full.shape}")
    t_fit = time.time()
    model = xgb.XGBClassifier(**params)
    model.fit(X_full, y - 1, verbose=False)
    log("XGB full fit done", since=t_fit)
    t_pred = time.time()
    test_proba = model.predict_proba(X_te).astype(np.float32)
    log("full-train predict done", since=t_pred)
    return model, test_proba


def _ensure_geo_latents() -> None:
    if geo_latents_ready():
        log("Geo latents cached (run_024) — skipping geo subprocess")
        return
    cmd = [sys.executable, str(ROOT / "src" / "run_024_geo.py")]
    log("Geo latents missing — running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _blend_diagnostics(run_id: str, oof: np.ndarray, test: np.ndarray, y: np.ndarray) -> float:
    solo = _oof_f1(oof, y)
    print(f"\n{run_id} solo OOF F1: {solo:.4f}")

    r24_oof = np.load(RUN024_OOF).astype(np.float64)
    r24_test = np.load(RUN024_TEST).astype(np.float64)
    r19_oof = np.load(RUN019_OOF).astype(np.float64)
    r19_test = np.load(RUN019_TEST).astype(np.float64)
    lgbm_oof = np.load(LGBM_OOF).astype(np.float64)
    lgbm_test = np.load(LGBM_TEST).astype(np.float64)

    r24 = pairwise_diagnostic("run_024", r24_oof, run_id, oof, y)
    r19 = pairwise_diagnostic("run_019", r19_oof, run_id, oof, y)
    f3, w_lg, w_r24, w_new = threeway_optimize(lgbm_oof, r24_oof, oof, y)
    print(f"  3-way (LGBM+run_024+{run_id}): {f3:.4f}")
    best = max(r24["best_score"], r19["best_score"], f3)
    if best > THRESHOLD:
        if f3 >= max(r24["best_score"], r19["best_score"]):
            build_blend_submission(
                {"lgbm": lgbm_test, "run_024": r24_test, run_id: test},
                {"lgbm": w_lg, "run_024": w_r24, run_id: w_new},
                space="proba",
                tag=f"{run_id}_3way",
            )
        elif r24["best_score"] >= r19["best_score"]:
            build_blend_submission(
                {"run_024": r24_test, run_id: test},
                {"run_024": r24["best_alpha"], run_id: 1.0 - r24["best_alpha"]},
                space=r24["best_space"],
                tag=f"{run_id}_2way_r24",
            )
        else:
            build_blend_submission(
                {run_id: test, "run_019": r19_test},
                {run_id: r19["best_alpha"], "run_019": 1.0 - r19["best_alpha"]},
                space=r19["best_space"],
                tag=f"{run_id}_2way_r19",
            )
    else:
        print(f"  No blend clears threshold {THRESHOLD:.4f}")
    return solo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="1 fold smoke test")
    parser.add_argument("--xgb-verbose", action="store_true", help="XGBoost training log every 10 trees")
    args = parser.parse_args()
    t0 = time.time()

    log("run_025 start")
    train, test = load_frames()
    y = pd.read_csv(PROCESSED_DIR / "y_train_full.csv")["damage_grade"].to_numpy()
    assert len(train) == len(y)
    log(f"loaded data  train={len(y):,}  test={len(test):,}")

    _ensure_geo_latents()
    t_lat = time.time()
    geo_train, geo_test, dr_train, dr_test, ru_train, ru_test = load_geo_latents()
    log(f"geo latents loaded  DR={dr_train.shape[1]}d  rollup={ru_train.shape[1]}d", since=t_lat)

    print("\n── XGBoost CV (full Shoumik stack) ──")
    oof, _, scores, best_iters = run_cv(
        train, test, y,
        geo_train, geo_test, dr_train, dr_test, ru_train, ru_test,
        quick=args.quick,
        xgb_verbose=args.xgb_verbose,
    )
    mean_f1 = float(np.mean(scores))
    std_f1 = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    log(f"CV mean F1: {mean_f1:.4f} ± {std_f1:.4f}")

    if args.quick:
        log(f"quick test done in {time.time() - t0:.1f}s")
        return

    n_est_full = int(np.mean(best_iters))
    _, test_proba = fit_full_train(
        train, test, y,
        geo_train, geo_test, dr_train, dr_test, ru_train, ru_test,
        n_est_full,
    )

    rm = RunManager()
    run_path = rm.run_path(RUN_ID)
    if not run_path.exists():
        rm.create_run(
            description="Full Shoumik stack: run_024 + OOF CatBoostEncoder on geo",
            model_type="XGBoost",
            feature_set="shoumik_ohe_geo_dr_rollup_cbcat_passthrough",
            params={**SHOUMIK_XGB_PARAMS, "geo_dr_latent": 32, "geo_rollup_latent": 16, "geo_catboost_encoder": True},
            run_id=RUN_ID,
            objective="multiclass",
            cv_folds=CV_FOLDS,
            cv_metric="micro_f1",
            notes="CatBoostEncoder on geo_level_* only, fit per CV fold. Submission = full-train refit.",
        )

    rm.save_cv_scores(RUN_ID, scores, mean_f1, std_f1)
    np.save(run_path / "oof_proba.npy", oof)
    np.save(run_path / "test_proba.npy", test_proba)
    test_csv = pd.read_csv(ROOT / "data" / "driven_data" / "test_values.csv")
    sub = pd.DataFrame({
        "building_id": test_csv["building_id"].values,
        "damage_grade": test_proba.argmax(axis=1) + 1,
    })
    rm.save_submission(RUN_ID, sub)
    _blend_diagnostics(RUN_ID, oof.astype(np.float64), test_proba.astype(np.float64), y)
    log(f"registered {RUN_ID} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
