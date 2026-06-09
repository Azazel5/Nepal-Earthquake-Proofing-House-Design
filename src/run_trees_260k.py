#!/usr/bin/env python3
"""
Train CatBoost and XGBoost on X_train_run012.csv (191 features, 260k rows).
Saves OOF/test probabilities then runs blend diagnostics against LGBM (run_012).

Run from project root:
    python src/run_trees_260k.py [--skip-xgb] [--skip-catboost]
"""

from __future__ import annotations

import argparse
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

from run_manager import PROCESSED_DIR, RunManager

print = partial(print, flush=True)

RANDOM_STATE = 42
CV_FOLDS = 5
ES_ROUNDS = 50

CATBOOST_PARAMS = {
    "iterations": 2000,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 3,
    "loss_function": "MultiClass",
    "eval_metric": "Accuracy",
    "random_seed": RANDOM_STATE,
    "verbose": 0,
    "early_stopping_rounds": ES_ROUNDS,
    "thread_count": -1,
}

XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "n_estimators": 2000,
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
    "early_stopping_rounds": ES_ROUNDS,
}

LGBM_OOF  = ROOT / "runs" / "run_012" / "oof_proba.npy"
LGBM_TEST = ROOT / "runs" / "run_012" / "test_proba.npy"


# ── Generic 5-fold CV helper ───────────────────────────────────────────────────

def cv_and_oof(model_fn, X, y, X_test, name: str) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    5-fold stratified CV.
    model_fn(fold) → freshly constructed model.
    Returns (oof_proba (N,3), test_proba (N_test,3), per_fold_f1).
    """
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros((len(y), 3), dtype=np.float32)
    test_folds: list[np.ndarray] = []
    scores: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        t_fold = time.time()
        model = model_fn(fold)

        if name == "catboost":
            model.fit(
                X.iloc[tr_idx], y[tr_idx],
                eval_set=(X.iloc[va_idx], y[va_idx]),
            )
        elif name == "xgb":
            model.fit(
                X.iloc[tr_idx], y[tr_idx] - 1,        # 0-indexed for XGB
                eval_set=[(X.iloc[va_idx], y[va_idx] - 1)],
                verbose=False,
            )
        else:
            raise ValueError(name)

        if name == "catboost":
            oof[va_idx] = model.predict_proba(X.iloc[va_idx]).astype(np.float32)
            test_folds.append(model.predict_proba(X_test).astype(np.float32))
        else:
            oof[va_idx] = model.predict_proba(X.iloc[va_idx]).astype(np.float32)
            test_folds.append(model.predict_proba(X_test).astype(np.float32))

        f1 = f1_score(y[va_idx], oof[va_idx].argmax(axis=1) + 1,
                      average="micro", labels=[1, 2, 3])
        best_iter = getattr(model, "best_iteration_", None) or getattr(model, "best_iteration", None)
        scores.append(f1)
        print(f"    fold {fold}: F1={f1:.4f}  best_iter={best_iter}  ({time.time()-t_fold:.0f}s)")

    test_avg = np.mean(test_folds, axis=0)
    return oof, test_avg, scores


# ── Model factories ────────────────────────────────────────────────────────────

def build_catboost(_fold: int):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(**CATBOOST_PARAMS)


def build_xgb(_fold: int):
    from xgboost import XGBClassifier
    p = dict(XGB_PARAMS)
    return XGBClassifier(**p)


# ── Blend analysis ─────────────────────────────────────────────────────────────

def _oof_f1(proba: np.ndarray, y: np.ndarray) -> float:
    return float(f1_score(y, proba.argmax(axis=1) + 1, average="micro", labels=[1, 2, 3]))


def _blend_proba(alpha: float, p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    return alpha * p_a + (1.0 - alpha) * p_b


def _blend_logit(alpha: float, p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    eps = 1e-7
    return alpha * np.log(p_a + eps) + (1.0 - alpha) * np.log(p_b + eps)


def pairwise_diagnostic(name_a: str, p_a: np.ndarray,
                         name_b: str, p_b: np.ndarray,
                         y: np.ndarray) -> dict:
    """Full pairwise blend analysis. Returns dict with best alpha, score, space."""
    f1_a = _oof_f1(p_a, y)
    f1_b = _oof_f1(p_b, y)

    pred_a = p_a.argmax(axis=1) + 1
    pred_b = p_b.argmax(axis=1) + 1
    disagree = pred_a != pred_b
    disagree_rate = disagree.mean()

    eps = 1e-7
    idx = y - 1
    N = len(y)
    loss_a = -np.log(p_a[np.arange(N), idx] + eps)
    loss_b = -np.log(p_b[np.arange(N), idx] + eps)
    loss_corr = float(np.corrcoef(loss_a, loss_b)[0, 1])

    a_right_disagree = ((pred_a == y) & disagree).sum()
    b_right_disagree = ((pred_b == y) & disagree).sum()

    print(f"\n  {'─'*55}")
    print(f"  {name_a} OOF F1: {f1_a:.4f}   {name_b} OOF F1: {f1_b:.4f}")
    print(f"  Argmax disagreement: {disagree_rate:.3f} ({disagree.sum():,} rows)")
    print(f"  On disagreed rows — {name_a} correct: {a_right_disagree:,}  {name_b} correct: {b_right_disagree:,}")
    print(f"  Per-row loss correlation: {loss_corr:.4f}")

    # Grid + optimize proba space
    alphas = np.linspace(0, 1, 101)
    proba_scores = [_oof_f1(_blend_proba(a, p_a, p_b), y) for a in alphas]
    logit_scores = [_oof_f1(_blend_logit(a, p_a, p_b), y) for a in alphas]

    best_proba_i = int(np.argmax(proba_scores))
    best_logit_i = int(np.argmax(logit_scores))

    from scipy.optimize import minimize_scalar
    res_p = minimize_scalar(lambda a: -_oof_f1(_blend_proba(a, p_a, p_b), y),
                            bounds=(0, 1), method="bounded", options={"xatol": 1e-4})
    res_l = minimize_scalar(lambda a: -_oof_f1(_blend_logit(a, p_a, p_b), y),
                            bounds=(0, 1), method="bounded", options={"xatol": 1e-4})

    best_score = max(-res_p.fun, -res_l.fun)
    best_space = "proba" if -res_p.fun >= -res_l.fun else "logit"
    best_alpha = float(res_p.x) if best_space == "proba" else float(res_l.x)

    print(f"  Best proba blend: α={res_p.x:.3f} (weight on {name_a})  F1={-res_p.fun:.4f}")
    print(f"  Best logit blend: α={res_l.x:.3f} (weight on {name_a})  F1={-res_l.fun:.4f}")

    # Per-grade breakdown
    print(f"  Per-grade accuracy:")
    for g in [1, 2, 3]:
        mask = y == g
        print(f"    grade {g} (n={mask.sum():,}): {name_a}={( pred_a[mask]==g).mean():.4f}  {name_b}={(pred_b[mask]==g).mean():.4f}")

    return {
        "name_a": name_a, "name_b": name_b,
        "f1_a": f1_a, "f1_b": f1_b,
        "disagree_rate": float(disagree_rate),
        "loss_corr": float(loss_corr),
        "best_alpha": best_alpha, "best_score": best_score, "best_space": best_space,
    }


def threeway_optimize(p_lgbm: np.ndarray, p_cb: np.ndarray, p_xgb: np.ndarray,
                      y: np.ndarray) -> tuple[float, float, float, float]:
    """Optimize w_lgbm, w_cb, w_xgb s.t. sum=1, all>=0."""
    def neg_f1(w):
        w = np.clip(w, 0, 1)
        w = w / w.sum()
        blend = w[0] * p_lgbm + w[1] * p_cb + w[2] * p_xgb
        return -_oof_f1(blend, y)

    best_val = 1.0
    best_w = (1.0, 0.0, 0.0)
    # Grid search over simplex (step 0.05)
    for i in range(21):
        for j in range(21 - i):
            k = 20 - i - j
            w = np.array([i, j, k], dtype=float) / 20
            v = neg_f1(w)
            if v < best_val:
                best_val = v
                best_w = tuple(w)

    # Refine with scipy
    res = minimize(neg_f1, x0=np.array(best_w),
                   method="Nelder-Mead",
                   options={"xatol": 1e-5, "fatol": 1e-5, "maxiter": 5000})
    w = np.clip(res.x, 0, 1)
    w = w / w.sum()
    return float(-res.fun), float(w[0]), float(w[1]), float(w[2])


# ── Build blend submission ─────────────────────────────────────────────────────

def build_blend_submission(
    test_probas: dict[str, np.ndarray],
    weights: dict[str, float],
    space: str,
    tag: str,
) -> None:
    eps = 1e-7
    blend = np.zeros_like(next(iter(test_probas.values())), dtype=np.float64)
    for name, w in weights.items():
        p = test_probas[name].astype(np.float64)
        if space == "logit":
            blend += w * np.log(p + eps)
        else:
            blend += w * p

    test_grades = blend.argmax(axis=1) + 1
    test_csv = pd.read_csv(ROOT / "data" / "driven_data" / "test_values.csv")
    sub_df = pd.DataFrame({"building_id": test_csv["building_id"].values,
                           "damage_grade": test_grades})
    out = ROOT / "outputs" / f"submission_{tag}.csv"
    out.parent.mkdir(exist_ok=True)
    sub_df.to_csv(out, index=False)
    print(f"  Saved → outputs/{out.name}")
    grades, counts = np.unique(test_grades, return_counts=True)
    for g, c in zip(grades, counts):
        print(f"    grade {g}: {c:,} ({c/len(test_grades)*100:.1f}%)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-catboost", action="store_true")
    parser.add_argument("--skip-xgb", action="store_true")
    args = parser.parse_args()
    t0 = time.time()

    print("── Loading data ──")
    X = pd.read_csv(PROCESSED_DIR / "X_train_run012.csv")
    y = pd.read_csv(PROCESSED_DIR / "y_train_full.csv")["damage_grade"].to_numpy()
    X_test = pd.read_csv(PROCESSED_DIR / "X_test_run012.csv")
    print(f"  X_train: {X.shape}  X_test: {X_test.shape}  y range: [{y.min()},{y.max()}]")

    rm = RunManager()
    oof_store: dict[str, np.ndarray] = {}
    test_store: dict[str, np.ndarray] = {}

    # ── CatBoost ──────────────────────────────────────────────────────────────
    cb_run = ROOT / "runs" / "run_014"
    cb_oof_path  = cb_run / "oof_proba.npy"
    cb_test_path = cb_run / "test_proba.npy"

    if cb_oof_path.exists() and args.skip_catboost:
        print("\n── CatBoost OOF found, loading ──")
        oof_store["catboost"] = np.load(cb_oof_path).astype(np.float64)
        test_store["catboost"] = np.load(cb_test_path).astype(np.float64)
        f1 = _oof_f1(oof_store["catboost"], y)
        print(f"  CatBoost OOF F1: {f1:.4f}")
    elif not args.skip_catboost:
        print("\n── Training CatBoost (5-fold, 260k rows) ──")
        cb_oof, cb_test, cb_scores = cv_and_oof(build_catboost, X, y, X_test, "catboost")
        mean_f1 = float(np.mean(cb_scores))
        std_f1  = float(np.std(cb_scores, ddof=1))
        print(f"  CatBoost CV: {mean_f1:.4f} ± {std_f1:.4f}")

        run_id = rm.get_next_run_id()
        rm.create_run(
            description="CatBoost on full 260k / run_012 191-feature matrix",
            model_type="CatBoost",
            feature_set="embedded_191_features_260k",
            params=CATBOOST_PARAMS,
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

        test_csv = pd.read_csv(ROOT / "data" / "driven_data" / "test_values.csv")
        test_grades = cb_test.argmax(axis=1) + 1
        sub_df = pd.DataFrame({"building_id": test_csv["building_id"].values,
                                "damage_grade": test_grades})
        rm.save_submission(run_id, sub_df)
        print(f"  Registered {run_id}")

        oof_store["catboost"] = cb_oof.astype(np.float64)
        test_store["catboost"] = cb_test.astype(np.float64)

    # ── XGBoost ───────────────────────────────────────────────────────────────
    # Check for existing XGBoost run (by model_type in metadata)
    xgb_oof_path  = Path("/dev/null")
    xgb_test_path = Path("/dev/null")
    for run_dir in sorted(ROOT.glob("runs/run_0*/")):
        meta_path = run_dir / "metadata.json"
        if meta_path.exists():
            with meta_path.open() as f:
                meta = json.load(f)
            if meta.get("model_type") == "XGBoost" and (run_dir / "oof_proba.npy").exists():
                xgb_oof_path  = run_dir / "oof_proba.npy"
                xgb_test_path = run_dir / "test_proba.npy"
                print(f"\n  Found existing XGBoost OOF at {run_dir.name}")
                break

    if xgb_oof_path.exists() and args.skip_xgb:
        print("\n── XGBoost OOF found, loading ──")
        oof_store["xgb"] = np.load(xgb_oof_path).astype(np.float64)
        test_store["xgb"] = np.load(xgb_test_path).astype(np.float64)
        f1 = _oof_f1(oof_store["xgb"], y)
        print(f"  XGBoost OOF F1: {f1:.4f}")
    elif not args.skip_xgb:
        print("\n── Training XGBoost (5-fold, 260k rows) ──")
        xgb_oof, xgb_test, xgb_scores = cv_and_oof(build_xgb, X, y, X_test, "xgb")
        mean_f1 = float(np.mean(xgb_scores))
        std_f1  = float(np.std(xgb_scores, ddof=1))
        print(f"  XGBoost CV: {mean_f1:.4f} ± {std_f1:.4f}")

        run_id = rm.get_next_run_id()
        rm.create_run(
            description="XGBoost on full 260k / run_012 191-feature matrix",
            model_type="XGBoost",
            feature_set="embedded_191_features_260k",
            params=XGB_PARAMS,
            run_id=run_id,
            objective="multiclass",
            n_features=X.shape[1],
            cv_folds=CV_FOLDS,
            cv_metric="micro_f1",
        )
        rm.save_cv_scores(run_id, xgb_scores, mean_f1, std_f1)

        run_dir = rm.run_path(run_id)
        np.save(run_dir / "oof_proba.npy", xgb_oof)
        np.save(run_dir / "test_proba.npy", xgb_test)

        test_csv = pd.read_csv(ROOT / "data" / "driven_data" / "test_values.csv")
        test_grades = xgb_test.argmax(axis=1) + 1
        sub_df = pd.DataFrame({"building_id": test_csv["building_id"].values,
                                "damage_grade": test_grades})
        rm.save_submission(run_id, sub_df)
        print(f"  Registered {run_id}")

        oof_store["xgb"] = xgb_oof.astype(np.float64)
        test_store["xgb"] = xgb_test.astype(np.float64)

    # ── Blend diagnostics ─────────────────────────────────────────────────────
    lgbm_oof  = np.load(LGBM_OOF).astype(np.float64)
    lgbm_test = np.load(LGBM_TEST).astype(np.float64)
    oof_store["lgbm"] = lgbm_oof
    test_store["lgbm"] = lgbm_test

    LGBM_CV = 0.7588
    NOISE    = 0.0016
    THRESHOLD = LGBM_CV + NOISE

    print("\n" + "═" * 60)
    print("PAIRWISE BLEND DIAGNOSTICS (reference LGBM OOF = 0.7588)")
    print("═" * 60)

    results: list[dict] = []

    if "catboost" in oof_store:
        r = pairwise_diagnostic("lgbm", lgbm_oof, "catboost", oof_store["catboost"], y)
        results.append(r)

    if "xgb" in oof_store:
        r = pairwise_diagnostic("lgbm", lgbm_oof, "xgb", oof_store["xgb"], y)
        results.append(r)

    # Three-way if both available
    if "catboost" in oof_store and "xgb" in oof_store:
        print(f"\n── 3-way blend: LGBM + CatBoost + XGB ──")
        f3, w_lg, w_cb, w_xg = threeway_optimize(lgbm_oof, oof_store["catboost"], oof_store["xgb"], y)
        print(f"  Best 3-way OOF F1: {f3:.4f}")
        print(f"  Weights — LGBM: {w_lg:.3f}  CatBoost: {w_cb:.3f}  XGB: {w_xg:.3f}")
        results.append({"name_a": "3way", "best_score": f3,
                         "w_lg": w_lg, "w_cb": w_cb, "w_xg": w_xg})

    # ── Summary & submission decision ─────────────────────────────────────────
    print("\n" + "═" * 60)
    print("SUMMARY")
    print("═" * 60)
    print(f"  LGBM solo:    {LGBM_CV:.4f}  (run_012 reference)")
    for r in results:
        if r["name_a"] == "3way":
            gain = r["best_score"] - LGBM_CV
            print(f"  3-way blend:  {r['best_score']:.4f}  gain={gain:+.4f}"
                  f"  {'✓ SUBMIT' if r['best_score'] > THRESHOLD else '✗ noise'}")
        else:
            gain = r["best_score"] - LGBM_CV
            print(f"  LGBM+{r['name_b']:9s}: {r['best_score']:.4f}  "
                  f"α={r['best_alpha']:.3f} ({r['best_space']:5s})  "
                  f"disagree={r['disagree_rate']:.3f}  corr={r['loss_corr']:.3f}  "
                  f"gain={gain:+.4f}  {'✓ SUBMIT' if r['best_score'] > THRESHOLD else '✗ noise'}")

    # Build submissions for anything that clears the threshold
    any_submitted = False
    for r in results:
        if r.get("best_score", 0) <= THRESHOLD:
            continue
        any_submitted = True

        if r["name_a"] == "3way":
            print(f"\n── Building 3-way blend submission ──")
            build_blend_submission(
                test_store,
                {"lgbm": r["w_lg"], "catboost": r["w_cb"], "xgb": r["w_xg"]},
                space="proba",
                tag=f"3way_lg{r['w_lg']:.2f}_cb{r['w_cb']:.2f}_xg{r['w_xg']:.2f}",
            )
        else:
            nb = r["name_b"]
            print(f"\n── Building LGBM+{nb} blend submission ({r['best_space']}) ──")
            alpha = r["best_alpha"]
            build_blend_submission(
                {"lgbm": lgbm_test, nb: test_store[nb]},
                {"lgbm": alpha, nb: 1.0 - alpha},
                space=r["best_space"],
                tag=f"lgbm_{nb}_a{alpha:.3f}",
            )

    if not any_submitted:
        print("\n  No blend clears the noise threshold. Best solo candidate to submit:")
        for name, oof in oof_store.items():
            if name == "lgbm":
                continue
            f1 = _oof_f1(oof, y)
            print(f"    {name}: {f1:.4f}")

    print(f"\nTotal wall-clock time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
