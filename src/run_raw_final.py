#!/usr/bin/env python3
"""
Final raw-categorical LGBM and CatBoost models (run_014 ensemble-decorrelation
experiment — see features_raw.py / tune_raw_models.py).

Trains both with best Optuna params on the EXACT 5-fold StratifiedKFold(seed=42)
split used by run_012, saves oof_proba.npy / test_proba.npy, registers runs via
RunManager, then runs pairwise + 4-way blend diagnostics against
{run_012 LGBM, run_015 XGB}.

Run from project root:
    python src/run_raw_final.py [--skip-lgbm] [--skip-catboost]
"""

from __future__ import annotations

import argparse
import itertools
import json
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

from features_raw import RAW_CAT_COLS, build_raw_features
from run_manager import RunManager
from run_trees_260k import _oof_f1, _blend_proba, _blend_logit, pairwise_diagnostic, build_blend_submission

print = partial(print, flush=True)

RANDOM_STATE = 42
CV_FOLDS = 5
ES_ROUNDS = 50

# Best params from outputs/optuna_raw/{model}_raw_best_params.json (3-fold search)
LGBM_RAW_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 142,
    "max_depth": 11,
    "learning_rate": 0.011008538980473884,
    "min_data_in_leaf": 196,
    "cat_smooth": 13.8564604093017,
    "cat_l2": 1.6378046861443067,
    "min_data_per_group": 185,
    "feature_fraction": 0.6297040066191009,
    "bagging_fraction": 0.6830638266104458,
    "bagging_freq": 1,
    "n_estimators": 2000,
    "n_jobs": 1,  # categorical_feature + multithreading segfaults on this platform
    "random_state": RANDOM_STATE,
    "verbose": -1,
}

CATBOOST_RAW_PARAMS = {
    "loss_function": "MultiClass",
    "eval_metric": "Accuracy",
    "iterations": 2000,
    "depth": 8,
    "learning_rate": 0.12077532307464126,
    "l2_leaf_reg": 2.1731885645782896,
    "random_strength": 3.8678024044744825,
    "one_hot_max_size": 10,
    "random_seed": RANDOM_STATE,
    "verbose": 0,
    "early_stopping_rounds": ES_ROUNDS,
    "thread_count": -1,
}

LGBM_OOF  = ROOT / "runs" / "run_012" / "oof_proba.npy"
LGBM_TEST = ROOT / "runs" / "run_012" / "test_proba.npy"
XGB_OOF   = ROOT / "runs" / "run_015" / "oof_proba.npy"
XGB_TEST  = ROOT / "runs" / "run_015" / "test_proba.npy"

LGBM_CV_REF = 0.7588
NOISE = 0.0016
THRESHOLD = LGBM_CV_REF + NOISE


# ── 5-fold CV training ───────────────────────────────────────────────────────

def cv_and_oof_raw(model_fn, X, y, X_test, name: str) -> tuple[np.ndarray, np.ndarray, list[float]]:
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros((len(y), 3), dtype=np.float32)
    test_folds: list[np.ndarray] = []
    scores: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        t_fold = time.time()
        model = model_fn(fold)

        if name == "lgbm":
            from lightgbm import early_stopping
            model.fit(
                X.iloc[tr_idx], y[tr_idx],
                eval_set=[(X.iloc[va_idx], y[va_idx])],
                eval_metric="multi_logloss",
                categorical_feature=RAW_CAT_COLS,
                callbacks=[early_stopping(stopping_rounds=ES_ROUNDS, verbose=False)],
            )
            best_iter = model.best_iteration_
        elif name == "catboost":
            model.fit(
                X.iloc[tr_idx], y[tr_idx],
                eval_set=(X.iloc[va_idx], y[va_idx]),
                cat_features=RAW_CAT_COLS,
            )
            best_iter = model.best_iteration_
        else:
            raise ValueError(name)

        oof[va_idx] = model.predict_proba(X.iloc[va_idx]).astype(np.float32)
        test_folds.append(model.predict_proba(X_test).astype(np.float32))

        f1 = f1_score(y[va_idx], oof[va_idx].argmax(axis=1) + 1, average="micro", labels=[1, 2, 3])
        scores.append(f1)
        print(f"    fold {fold}: F1={f1:.4f}  best_iter={best_iter}  ({time.time()-t_fold:.0f}s)")

    test_avg = np.mean(test_folds, axis=0)
    return oof, test_avg, scores


def build_lgbm(_fold: int):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**LGBM_RAW_PARAMS)


def build_catboost(_fold: int):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(**CATBOOST_RAW_PARAMS)


# ── N-way blend optimizer ────────────────────────────────────────────────────

def nway_optimize(probas: dict[str, np.ndarray], y: np.ndarray, space: str, step: float = 0.05):
    names = list(probas.keys())
    arrs = [probas[k] for k in names]
    n = len(names)
    eps = 1e-7

    def blend(w):
        if space == "logit":
            out = np.zeros_like(arrs[0], dtype=np.float64)
            for wi, a in zip(w, arrs):
                out += wi * np.log(a + eps)
            return out
        out = np.zeros_like(arrs[0], dtype=np.float64)
        for wi, a in zip(w, arrs):
            out += wi * a
        return out

    def neg_f1(w):
        w = np.clip(w, 0, None)
        s = w.sum()
        if s == 0:
            w = np.ones(n) / n
        else:
            w = w / s
        return -_oof_f1(blend(w), y)

    g = int(round(1 / step))
    best_val = 1.0
    best_w = np.ones(n) / n

    # grid search over the simplex
    for combo in itertools.product(range(g + 1), repeat=n - 1):
        if sum(combo) > g:
            continue
        last = g - sum(combo)
        w = np.array(list(combo) + [last], dtype=float) / g
        v = neg_f1(w)
        if v < best_val:
            best_val = v
            best_w = w

    res = minimize(neg_f1, x0=best_w, method="Nelder-Mead",
                   options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 10000})
    w = np.clip(res.x, 0, None)
    w = w / w.sum()
    return float(-res.fun), dict(zip(names, w.tolist()))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-lgbm", action="store_true")
    parser.add_argument("--skip-catboost", action="store_true")
    args = parser.parse_args()
    t0 = time.time()

    print("── Loading raw feature matrix ──")
    X, X_test, y = build_raw_features()
    print(f"  X: {X.shape}  X_test: {X_test.shape}")

    rm = RunManager()
    test_csv = pd.read_csv(ROOT / "data" / "driven_data" / "test_values.csv")

    oof_store: dict[str, np.ndarray] = {}
    test_store: dict[str, np.ndarray] = {}

    # ── raw-LGBM ──────────────────────────────────────────────────────────────
    raw_lgbm_run = None
    for run_dir in sorted(ROOT.glob("runs/run_0*/")):
        meta_path = run_dir / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("feature_set") == "raw_categorical_38" and meta.get("model_type") == "LGBM":
                raw_lgbm_run = run_dir
                break

    if raw_lgbm_run is not None and args.skip_lgbm:
        print(f"\n── raw-LGBM OOF found at {raw_lgbm_run.name}, loading ──")
        oof_store["raw_lgbm"] = np.load(raw_lgbm_run / "oof_proba.npy").astype(np.float64)
        test_store["raw_lgbm"] = np.load(raw_lgbm_run / "test_proba.npy").astype(np.float64)
        print(f"  raw-LGBM OOF F1: {_oof_f1(oof_store['raw_lgbm'], y):.4f}")
    elif not args.skip_lgbm:
        print("\n── Training raw-LGBM (5-fold, raw categorical view) ──")
        lg_oof, lg_test, lg_scores = cv_and_oof_raw(build_lgbm, X, y, X_test, "lgbm")
        mean_f1, std_f1 = float(np.mean(lg_scores)), float(np.std(lg_scores, ddof=1))
        print(f"  raw-LGBM CV: {mean_f1:.4f} ± {std_f1:.4f}")

        run_id = rm.get_next_run_id()
        rm.create_run(
            description="LGBM on raw-categorical 38-feature view (no precomputed geo rates) — ensemble decorrelation",
            model_type="LGBM",
            feature_set="raw_categorical_38",
            params=LGBM_RAW_PARAMS,
            run_id=run_id,
            objective="multiclass",
            n_features=X.shape[1],
            cv_folds=CV_FOLDS,
            cv_metric="micro_f1",
        )
        rm.save_cv_scores(run_id, lg_scores, mean_f1, std_f1)
        run_dir = rm.run_path(run_id)
        np.save(run_dir / "oof_proba.npy", lg_oof)
        np.save(run_dir / "test_proba.npy", lg_test)
        sub_df = pd.DataFrame({"building_id": test_csv["building_id"].values,
                                "damage_grade": lg_test.argmax(axis=1) + 1})
        rm.save_submission(run_id, sub_df)
        print(f"  Registered {run_id}")

        oof_store["raw_lgbm"] = lg_oof.astype(np.float64)
        test_store["raw_lgbm"] = lg_test.astype(np.float64)

    # ── raw-CatBoost ──────────────────────────────────────────────────────────
    raw_cb_run = None
    for run_dir in sorted(ROOT.glob("runs/run_0*/")):
        meta_path = run_dir / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("feature_set") == "raw_categorical_38" and meta.get("model_type") == "CatBoost":
                raw_cb_run = run_dir
                break

    if raw_cb_run is not None and args.skip_catboost:
        print(f"\n── raw-CatBoost OOF found at {raw_cb_run.name}, loading ──")
        oof_store["raw_catboost"] = np.load(raw_cb_run / "oof_proba.npy").astype(np.float64)
        test_store["raw_catboost"] = np.load(raw_cb_run / "test_proba.npy").astype(np.float64)
        print(f"  raw-CatBoost OOF F1: {_oof_f1(oof_store['raw_catboost'], y):.4f}")
    elif not args.skip_catboost:
        print("\n── Training raw-CatBoost (5-fold, raw categorical view) ──")
        cb_oof, cb_test, cb_scores = cv_and_oof_raw(build_catboost, X, y, X_test, "catboost")
        mean_f1, std_f1 = float(np.mean(cb_scores)), float(np.std(cb_scores, ddof=1))
        print(f"  raw-CatBoost CV: {mean_f1:.4f} ± {std_f1:.4f}")

        run_id = rm.get_next_run_id()
        rm.create_run(
            description="CatBoost on raw-categorical 38-feature view, ordered TS for geo (no precomputed rates) — ensemble decorrelation",
            model_type="CatBoost",
            feature_set="raw_categorical_38",
            params=CATBOOST_RAW_PARAMS,
            run_id=run_id,
            objective="multiclass",
            n_features=X.shape[1],
            cv_folds=CV_FOLDS,
            cv_metric="micro_f1",
        )
        rm.save_cv_scores(run_id, cb_scores, mean_f1, std_f1)
        run_dir = rm.run_path(run_id)
        np.save(run_dir / "oof_proba.npy", cb_oof)
        np.save(run_dir / "test_proba.npy", cb_test)
        sub_df = pd.DataFrame({"building_id": test_csv["building_id"].values,
                                "damage_grade": cb_test.argmax(axis=1) + 1})
        rm.save_submission(run_id, sub_df)
        print(f"  Registered {run_id}")

        oof_store["raw_catboost"] = cb_oof.astype(np.float64)
        test_store["raw_catboost"] = cb_test.astype(np.float64)

    # ── Reference models ──────────────────────────────────────────────────────
    lgbm_oof  = np.load(LGBM_OOF).astype(np.float64)
    lgbm_test = np.load(LGBM_TEST).astype(np.float64)
    xgb_oof   = np.load(XGB_OOF).astype(np.float64)
    xgb_test  = np.load(XGB_TEST).astype(np.float64)

    all_oof  = {"lgbm_012": lgbm_oof, "xgb_015": xgb_oof, **oof_store}
    all_test = {"lgbm_012": lgbm_test, "xgb_015": xgb_test, **test_store}

    # ── Pairwise diagnostics vs run_012 LGBM ─────────────────────────────────
    print("\n" + "═" * 60)
    print("PAIRWISE BLEND DIAGNOSTICS (reference LGBM run_012 OOF = 0.7588)")
    print("═" * 60)

    pair_results = []
    for name, oof in oof_store.items():
        r = pairwise_diagnostic("lgbm_012", lgbm_oof, name, oof, y)
        pair_results.append(r)

    # ── N-way blend over all available models ───────────────────────────────
    print("\n" + "═" * 60)
    print(f"N-WAY BLEND OPTIMIZATION ({len(all_oof)} models: {list(all_oof.keys())})")
    print("═" * 60)

    f1_proba, w_proba = nway_optimize(all_oof, y, "proba")
    f1_logit, w_logit = nway_optimize(all_oof, y, "logit")

    print(f"  Proba-space best F1: {f1_proba:.4f}")
    for k, v in w_proba.items():
        flag = "  <- DEAD (<3%)" if v < 0.03 else ""
        print(f"    {k:14s}: {v:.3f}{flag}")
    print(f"  Logit-space best F1: {f1_logit:.4f}")
    for k, v in w_logit.items():
        flag = "  <- DEAD (<3%)" if v < 0.03 else ""
        print(f"    {k:14s}: {v:.3f}{flag}")

    best_f1, best_w, best_space = (f1_proba, w_proba, "proba") if f1_proba >= f1_logit else (f1_logit, w_logit, "logit")

    # ── Summary & decision ───────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("SUMMARY")
    print("═" * 60)
    print(f"  LGBM run_012 solo: {LGBM_CV_REF:.4f}  (reference)")
    for name, oof in oof_store.items():
        print(f"  {name} solo:       {_oof_f1(oof, y):.4f}")
    for r in pair_results:
        gain = r["best_score"] - LGBM_CV_REF
        print(f"  2-way lgbm_012+{r['name_b']:12s}: {r['best_score']:.4f}  "
              f"α={r['best_alpha']:.3f} ({r['best_space']:5s})  "
              f"disagree={r['disagree_rate']:.3f}  corr={r['loss_corr']:.3f}  gain={gain:+.4f}")
    gain = best_f1 - LGBM_CV_REF
    print(f"  N-way blend ({best_space}): {best_f1:.4f}  gain={gain:+.4f}  "
          f"{'>>> CLEARS THRESHOLD <<<' if best_f1 > THRESHOLD else 'noise (threshold=' + f'{THRESHOLD:.4f})'}")

    if best_f1 > THRESHOLD:
        print("\n── Building N-way blend submission ──")
        tag = "_".join(f"{k.split('_')[0]}{v:.2f}" for k, v in best_w.items())
        build_blend_submission(all_test, best_w, space=best_space, tag=f"nway_{tag}")
    else:
        print("\n  No blend clears the noise threshold. NOT submitting.")

    print(f"\nTotal wall-clock time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
