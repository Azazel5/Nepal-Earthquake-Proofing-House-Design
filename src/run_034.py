#!/usr/bin/env python3
"""
run_034: Geo coverage gate + foundation×geo interactions + pseudo-label retrain.

Teacher = run_026 blend. Retrains run_024-style XGB + run_019-style LGBM on
train ∪ high-confidence pseudo-labeled test rows (soft weights). OOF is measured
only on held-out original-train rows (anti-leak).

Run from project root:
    python src/run_034.py [--quick] [--phase0-only]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import xgboost as xgb  # noqa: E402 — before any torch import

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pl_interactions import PL_INTERACTION_NAMES, compute_pl_interactions
from retrain import EARLY_STOPPING_ROUNDS, TRIAL_66_PARAMS
from run_024_features import ShoumikFeatureBuilder, load_frames, load_geo_latents
from run_manager import PROCESSED_DIR, RunManager
from run_trees_260k import _blend_logit, _blend_proba, _oof_f1, pairwise_diagnostic

print = partial(print, flush=True)

RUN_ID = "run_034"
TEACHER_RUN = "run_026"
SOTA_OOF = 0.7546
NOISE = 0.0016
THRESHOLD = SOTA_OOF + NOISE
RANDOM_STATE = 42
CV_FOLDS = 5
PL_TAU = 0.85
G2_DOMINANCE = 0.70
CLASS_PRIORS = np.array([0.10, 0.57, 0.33], dtype=np.float64)
PCA_K = 80
PCA_VARIANT = "embed_only"

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


class ShoumikPLFeatureBuilder(ShoumikFeatureBuilder):
    """run_024 matrix + explicit foundation×geo/age interactions."""

    def transform(
        self,
        df: pd.DataFrame,
        geo_idx: np.ndarray,
        geo_dr: np.ndarray,
        geo_ru: np.ndarray,
    ) -> np.ndarray:
        base = super().transform(df, geo_idx, geo_dr, geo_ru)
        pl = compute_pl_interactions(df)
        return np.hstack([base, pl]).astype(np.float32)


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


def _g3_fn_taxonomy(y: np.ndarray, proba: np.ndarray) -> dict:
    pred = proba.argmax(axis=1) + 1
    g3_fn_g2 = (y == 3) & (pred == 2)
    mass = proba[:, 1] + proba[:, 2]
    q26 = np.divide(proba[:, 2], mass, out=np.zeros(len(y)), where=mass > 1e-9)
    n_fn = int(g3_fn_g2.sum())
    border = g3_fn_g2 & (q26 > 0.35) & (q26 < 0.45)
    conf = g3_fn_g2 & (q26 < 0.25)
    g3_rec = float(((y == 3) & (pred == 3)).sum() / max((y == 3).sum(), 1))
    return {
        "n_g3_fn_g2": n_fn,
        "borderline_pct": float(border.sum() / max(n_fn, 1) * 100),
        "confident_g2_pct": float(conf.sum() / max(n_fn, 1) * 100),
        "g3_recall": g3_rec,
    }


def phase0_geo_coverage(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    p26_oof: np.ndarray,
) -> dict:
    """Train vs test geo frequency + borderline FN vs geo3 support."""
    print("\n" + "═" * 72)
    print("PHASE 0 — GEO COVERAGE CHECK")
    print("═" * 72)

    for level in ("geo_level_2_id", "geo_level_3_id"):
        tr_cnt = train[level].value_counts()
        te_vals = test[level].values
        support = tr_cnt.reindex(te_vals).fillna(0).astype(int).to_numpy()
        n_te = len(test)
        lt5 = float((support < 5).mean() * 100)
        lt20 = float((support < 20).mean() * 100)
        print(f"\n  [{level}] test rows with train support <5: {lt5:.2f}%  <20: {lt20:.2f}%")
        print(f"    median support: {np.median(support):.0f}  min: {support.min()}  max: {support.max()}")

    g3_support = train.groupby("geo_level_3_id").size()
    pred = p26_oof.argmax(axis=1) + 1
    g3_fn = (y == 3) & (pred == 2)
    mass = p26_oof[:, 1] + p26_oof[:, 2]
    q26 = np.divide(p26_oof[:, 2], mass, out=np.zeros(len(y)), where=mass > 1e-9)
    border_fn = g3_fn & (q26 > 0.35) & (q26 < 0.45)

    tr_geo3 = train["geo_level_3_id"].values
    sup_map = g3_support.to_dict()
    all_fn_sup = np.array([sup_map.get(g, 0) for g in tr_geo3[g3_fn]])
    border_sup = np.array([sup_map.get(g, 0) for g in tr_geo3[border_fn]])

    print("\n  Borderline G3→G2 FN (0.35<q26<0.45) vs geo3 train support:")
    print(f"    n borderline FN: {border_fn.sum():,}")
    if border_fn.sum() > 0:
        for thr in (5, 20):
            pct_b = float((border_sup < thr).mean() * 100)
            pct_a = float((all_fn_sup < thr).mean() * 100)
            print(f"    support<{thr}: borderline FN {pct_b:.1f}%  all G3→G2 FN {pct_a:.1f}%")

    lt5_g3 = float((test["geo_level_3_id"].map(g3_support).fillna(0) < 5).mean() * 100)
    gate = "PROCEED" if lt5_g3 > 3.0 else "LOW_UPSIDE"
    print(f"\n  GATE: {gate}  (test geo3 <5 train support: {lt5_g3:.2f}%)")
    if gate == "PROCEED":
        print("  → Sparse test geo3 cells present; PL has a real mechanism.")
    else:
        print("  → Geo support near-uniform; PL upside limited — interactions still worth testing.")

    return {
        "test_geo3_pct_support_lt5": lt5_g3,
        "test_geo3_pct_support_lt20": float(
            (test["geo_level_3_id"].map(g3_support).fillna(0) < 20).mean() * 100
        ),
        "borderline_fn_pct_support_lt5": float((border_sup < 5).mean() * 100) if border_fn.sum() else 0.0,
        "borderline_fn_pct_support_lt20": float((border_sup < 20).mean() * 100) if border_fn.sum() else 0.0,
        "all_fn_pct_support_lt5": float((all_fn_sup < 5).mean() * 100) if g3_fn.sum() else 0.0,
        "gate": gate,
    }


def select_pseudo_labels(
    proba: np.ndarray,
    *,
    tau: float = PL_TAU,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Return mask, hard labels (1-indexed), soft weights, selection mode."""
    maxp = proba.max(axis=1)
    pred0 = proba.argmax(axis=1)
    confident = maxp >= tau
    mode = "global_tau"

    if confident.sum() > 0 and (pred0[confident] == 1).mean() > G2_DOMINANCE:
        mode = "per_class_quota"
        target_n = int(confident.sum()) or max(int(0.12 * len(proba)), 1000)
        confident = np.zeros(len(proba), dtype=bool)
        for c in range(3):
            m = (pred0 == c) & (maxp >= tau * 0.80)
            idx = np.where(m)[0]
            n_take = max(1, int(CLASS_PRIORS[c] * target_n))
            order = idx[np.argsort(-maxp[idx])]
            confident[order[: min(n_take, len(order))]] = True

    y_hard = pred0 + 1
    weights = maxp.astype(np.float32)
    return confident, y_hard, weights, mode


def _fit_xgb_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    sw_tr: np.ndarray | None = None,
) -> tuple[xgb.XGBClassifier, int]:
    model = xgb.XGBClassifier(**SHOUMIK_XGB_PARAMS, early_stopping_rounds=EARLY_STOP)
    model.fit(
        X_tr,
        y_tr - 1,
        sample_weight=sw_tr,
        eval_set=[(X_va, y_va - 1)],
        verbose=False,
    )
    best_iter = model.best_iteration if model.best_iteration is not None else SHOUMIK_XGB_PARAMS["n_estimators"] - 1
    return model, int(best_iter) + 1


def run_xgb_pl_cv(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    geo_train: np.ndarray,
    geo_test: np.ndarray,
    dr_train: np.ndarray,
    dr_test: np.ndarray,
    ru_train: np.ndarray,
    ru_test: np.ndarray,
    pl_mask: np.ndarray,
    y_pl: np.ndarray,
    sw_pl: np.ndarray,
    *,
    quick: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(train, y))
    n_folds = 1 if quick else CV_FOLDS
    pl_idx = np.where(pl_mask)[0]

    oof = np.zeros((len(y), 3), dtype=np.float32)
    test_folds: list[np.ndarray] = []
    scores: list[float] = []

    for fold, (tri, vai) in enumerate(splits[:n_folds], start=1):
        t0 = time.time()
        fb = ShoumikPLFeatureBuilder()
        fb.fit(train.iloc[tri])

        X_tr = fb.transform(train.iloc[tri], geo_train[tri], dr_train[tri], ru_train[tri])
        X_va = fb.transform(train.iloc[vai], geo_train[vai], dr_train[vai], ru_train[vai])
        X_te = fb.transform(test, geo_test, dr_test, ru_test)
        X_pl = fb.transform(
            test.iloc[pl_idx], geo_test[pl_idx], dr_test[pl_idx], ru_test[pl_idx],
        )

        X_tr_aug = np.vstack([X_tr, X_pl])
        y_tr_aug = np.concatenate([y[tri], y_pl[pl_idx]])
        sw_tr_aug = np.concatenate([np.ones(len(tri), dtype=np.float32), sw_pl[pl_idx]])

        model, n_iter = _fit_xgb_fold(X_tr_aug, y_tr_aug, X_va, y[vai], sw_tr=sw_tr_aug)
        oof[vai] = model.predict_proba(X_va).astype(np.float32)
        test_folds.append(model.predict_proba(X_te).astype(np.float32))
        f1 = _micro_f1(y[vai], oof[vai])
        scores.append(f1)
        print(f"  XGB fold {fold}: F1={f1:.4f}  n_pl={len(pl_idx):,}  ({time.time() - t0:.0f}s)")

    return oof, np.mean(test_folds, axis=0), scores


def _embed_cols(cols: list[str]) -> list[str]:
    return [c for c in cols if "_emb_" in c]


def _non_embed_cols(cols: list[str]) -> list[str]:
    return [c for c in cols if "_emb_" not in c]


def _transform_pca(
    X_tr: pd.DataFrame,
    X_va: pd.DataFrame,
    X_te: pd.DataFrame,
    pca_cols: list[str],
    pass_cols: list[str],
    n_components: int,
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


def _fit_lgbm_fold(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    sw_tr: np.ndarray | None = None,
) -> LGBMClassifier:
    model = LGBMClassifier(**TRIAL_66_PARAMS)
    model.fit(
        X_tr,
        y_tr,
        sample_weight=sw_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="multi_logloss",
        callbacks=[early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model


def _append_pl_to_matrix(X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    pl = compute_pl_interactions(df)
    pl_df = pd.DataFrame(pl, columns=PL_INTERACTION_NAMES, index=df.index)
    return pd.concat([X.reset_index(drop=True), pl_df], axis=1)


def run_lgbm_pl_cv(
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    y: np.ndarray,
    pl_mask: np.ndarray,
    y_pl: np.ndarray,
    sw_pl: np.ndarray,
    *,
    quick: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    cols = list(X.columns)
    pca_cols, pass_cols = _embed_cols(cols), _non_embed_cols(cols)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(X, y))
    n_folds = 1 if quick else CV_FOLDS
    pl_idx = np.where(pl_mask)[0]

    oof = np.zeros((len(y), 3), dtype=np.float32)
    test_folds: list[np.ndarray] = []
    scores: list[float] = []

    for fold, (tri, vai) in enumerate(splits[:n_folds], start=1):
        t0 = time.time()
        X_tr = X.iloc[tri]
        X_va = X.iloc[vai]
        X_pl = X_test.iloc[pl_idx]

        df_tr, df_va, df_te = _transform_pca(X_tr, X_va, X_test, pca_cols, pass_cols, PCA_K)
        _, _, df_pl = _transform_pca(X_tr, X_va, X_pl, pca_cols, pass_cols, PCA_K)

        df_tr_aug = pd.concat([df_tr, df_pl], axis=0, ignore_index=True)
        y_tr_aug = np.concatenate([y[tri], y_pl[pl_idx]])
        sw_tr_aug = np.concatenate([np.ones(len(tri), dtype=np.float32), sw_pl[pl_idx]])

        model = _fit_lgbm_fold(df_tr_aug, y_tr_aug, df_va, y[vai], sw_tr=sw_tr_aug)
        oof[vai] = model.predict_proba(df_va).astype(np.float32)
        test_folds.append(model.predict_proba(df_te).astype(np.float32))
        f1 = _micro_f1(y[vai], oof[vai])
        scores.append(f1)
        print(f"  LGBM fold {fold}: F1={f1:.4f}  n_pl={len(pl_idx):,}  ({time.time() - t0:.0f}s)")

    return oof, np.mean(test_folds, axis=0), scores


def _per_fold_blend_scores(
    p_a: np.ndarray,
    p_b: np.ndarray,
    y: np.ndarray,
    alpha: float,
    space: str,
) -> list[float]:
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    blend_fn = _blend_proba if space == "proba" else _blend_logit
    scores: list[float] = []
    for _tr, va in skf.split(p_a, y):
        blended = blend_fn(alpha, p_a[va], p_b[va])
        scores.append(_micro_f1(y[va], blended))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="1-fold CV smoke test")
    parser.add_argument("--phase0-only", action="store_true")
    args = parser.parse_args()
    t0 = time.time()

    train, test = load_frames()
    y = pd.read_csv(PROCESSED_DIR / "y_train_full.csv")["damage_grade"].to_numpy()
    p26_oof = np.load(ROOT / "runs" / TEACHER_RUN / "oof_proba.npy").astype(np.float64)
    teacher_test = np.load(ROOT / "runs" / TEACHER_RUN / "test_proba.npy").astype(np.float64)

    run_dir = ROOT / "runs" / RUN_ID

    geo_report = phase0_geo_coverage(train, test, y, p26_oof)

    if args.phase0_only:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "geo_coverage.json", "w") as f:
            json.dump(geo_report, f, indent=2)
        print(f"\nPhase 0 only — saved {run_dir / 'geo_coverage.json'}")
        return

    print("\n" + "═" * 72)
    print("PHASE 2 — PSEUDO-LABEL SELECTION (teacher = run_026)")
    print("═" * 72)
    pl_mask, y_pl, sw_pl, pl_mode = select_pseudo_labels(teacher_test)
    n_pl = int(pl_mask.sum())
    pred_pl = y_pl[pl_mask]
    print(f"  Selection mode: {pl_mode}  tau={PL_TAU}")
    print(f"  Pseudo-labeled rows: {n_pl:,} / {len(test):,} ({n_pl/len(test)*100:.1f}%)")
    for g in [1, 2, 3]:
        ng = int((pred_pl == g).sum())
        print(f"    class G{g}: {ng:,} ({ng/max(n_pl,1)*100:.1f}%)")

    print("\n" + "═" * 72)
    print("PHASE 1+2 — RETRAIN XGB (run_024 + PL interactions + pseudo)")
    print("═" * 72)
    geo_train, geo_test, dr_train, dr_test, ru_train, ru_test = load_geo_latents()
    xgb_oof, xgb_test, xgb_scores = run_xgb_pl_cv(
        train, test, y, geo_train, geo_test, dr_train, dr_test, ru_train, ru_test,
        pl_mask, y_pl, sw_pl, quick=args.quick,
    )
    print(f"  XGB solo OOF: {_oof_f1(xgb_oof, y):.4f}  folds={[f'{s:.4f}' for s in xgb_scores]}")

    print("\n" + "═" * 72)
    print("PHASE 1+2 — RETRAIN LGBM (run_019 embed PCA + PL interactions + pseudo)")
    print("═" * 72)
    X = pd.read_csv(PROCESSED_DIR / "X_train_run012.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test_run012.csv")
    X = _append_pl_to_matrix(X, train)
    X_test = _append_pl_to_matrix(X_test, test)
    print(f"  Features: {X.shape[1]}  (+{len(PL_INTERACTION_NAMES)} PL interactions)")
    lgbm_oof, lgbm_test, lgbm_scores = run_lgbm_pl_cv(
        X, X_test, y, pl_mask, y_pl, sw_pl, quick=args.quick,
    )
    print(f"  LGBM solo OOF: {_oof_f1(lgbm_oof, y):.4f}  folds={[f'{s:.4f}' for s in lgbm_scores]}")

    if args.quick:
        print(f"\nQuick mode done in {time.time() - t0:.1f}s")
        return

    print("\n" + "═" * 72)
    print("BLEND — run_034 XGB + LGBM (run_026 harness)")
    print("═" * 72)
    f_xgb = _oof_f1(xgb_oof, y)
    f_lgbm = _oof_f1(lgbm_oof, y)
    f26 = _oof_f1(p26_oof, y)
    print(f"  run_026 teacher OOF: {f26:.4f}")
    print(f"  run_034 XGB solo:      {f_xgb:.4f}")
    print(f"  run_034 LGBM solo:     {f_lgbm:.4f}")

    blend = pairwise_diagnostic("run_034_xgb", xgb_oof.astype(np.float64), "run_034_lgbm", lgbm_oof.astype(np.float64), y)
    best_alpha = blend["best_alpha"]
    best_space = blend["best_space"]
    oof_blend = (
        _blend_proba(best_alpha, xgb_oof, lgbm_oof)
        if best_space == "proba"
        else _blend_logit(best_alpha, xgb_oof.astype(np.float64), lgbm_oof.astype(np.float64))
    ).astype(np.float32)
    test_blend = (
        _blend_proba(best_alpha, xgb_test, lgbm_test)
        if best_space == "proba"
        else _blend_logit(best_alpha, xgb_test.astype(np.float64), lgbm_test.astype(np.float64))
    ).astype(np.float32)
    blend_f1 = _oof_f1(oof_blend, y)
    fold_scores = _per_fold_blend_scores(
        xgb_oof.astype(np.float64), lgbm_oof.astype(np.float64), y, best_alpha, best_space,
    )

    print("\n" + "═" * 72)
    print("EVALUATION")
    print("═" * 72)
    delta = blend_f1 - f26
    print(f"  run_034 blend OOF:  {blend_f1:.4f}  (Δ vs run_026: {delta:+.4f})")
    print(f"  Threshold:          {THRESHOLD:.4f}  "
          f"{'SUBMIT ✓' if blend_f1 > THRESHOLD else 'hold — below SOTA+noise'}")
    print(f"  Per-fold blend:     {np.mean(fold_scores):.4f} ± {np.std(fold_scores, ddof=1):.4f}")

    grade_26 = _per_grade_table(y, p26_oof)
    grade_34 = _per_grade_table(y, oof_blend)
    print("\n  Per-grade (run_026 → run_034):")
    for g in ["1", "2", "3"]:
        a, b = grade_26[g], grade_34[g]
        print(
            f"    G{g}: prec {a['precision']:.3f}→{b['precision']:.3f}  "
            f"rec {a['recall']:.3f}→{b['recall']:.3f}  "
            f"Δrec {b['recall']-a['recall']:+.3f}"
        )

    tax_26 = _g3_fn_taxonomy(y, p26_oof)
    tax_34 = _g3_fn_taxonomy(y, oof_blend)
    print("\n  G3 FN taxonomy (run_026 → run_034):")
    print(f"    G3 recall:        {tax_26['g3_recall']:.4f} → {tax_34['g3_recall']:.4f}")
    print(f"    borderline FN %:  {tax_26['borderline_pct']:.1f}% → {tax_34['borderline_pct']:.1f}%")
    print(f"    confident G2 %:   {tax_26['confident_g2_pct']:.1f}% → {tax_34['confident_g2_pct']:.1f}%")

    rm = RunManager()
    rm.create_run(
        description="Geo PL + foundation×geo interactions; XGB+LGBM retrain on pseudo-labeled test",
        model_type="ensemble_pl_retrain",
        feature_set="run_024_xgb+run_019_lgbm+pl_interactions+pseudo_labels",
        params={
            "teacher": TEACHER_RUN,
            "pl_tau": PL_TAU,
            "pl_mode": pl_mode,
            "n_pseudo": n_pl,
            "pl_class_counts": {str(g): int((pred_pl == g).sum()) for g in [1, 2, 3]},
            "blend_alpha_xgb": best_alpha,
            "blend_space": best_space,
            "pca_k": PCA_K,
            "pca_variant": PCA_VARIANT,
            "geo_gate": geo_report["gate"],
            "solo_oof": {"xgb": f_xgb, "lgbm": f_lgbm, "run_026": f26},
        },
        run_id=RUN_ID,
        objective="multiclass",
        n_features=int(X.shape[1]),
        cv_folds=CV_FOLDS,
        cv_metric="micro_f1",
        notes="Soft PL via sample_weight=max_prob; OOF on original train only.",
    )

    rm.save_cv_scores(RUN_ID, fold_scores, float(np.mean(fold_scores)), float(np.std(fold_scores, ddof=1)))
    with open(run_dir / "geo_coverage.json", "w") as f:
        json.dump(geo_report, f, indent=2)
    np.save(run_dir / "pl_mask.npy", pl_mask)
    np.save(run_dir / "pl_soft_weights.npy", sw_pl)
    np.save(run_dir / "pl_hard_labels.npy", y_pl)
    np.save(run_dir / "oof_proba.npy", oof_blend)
    np.save(run_dir / "test_proba.npy", test_blend)
    np.save(run_dir / "oof_xgb.npy", xgb_oof)
    np.save(run_dir / "oof_lgbm.npy", lgbm_oof)

    eval_report = {
        "blend_oof_f1": blend_f1,
        "delta_vs_run_026": delta,
        "threshold": THRESHOLD,
        "submit": blend_f1 > THRESHOLD,
        "per_grade_run_026": grade_26,
        "per_grade_run_034": grade_34,
        "g3_fn_taxonomy_run_026": tax_26,
        "g3_fn_taxonomy_run_034": tax_34,
        "geo_coverage": geo_report,
    }
    with open(run_dir / "eval_report.json", "w") as f:
        json.dump(eval_report, f, indent=2)

    test_csv = pd.read_csv(ROOT / "data" / "driven_data" / "test_values.csv")
    sub = pd.DataFrame({"building_id": test_csv["building_id"].values, "damage_grade": test_blend.argmax(axis=1) + 1})
    rm.save_submission(RUN_ID, sub)

    print(f"\nRegistered {RUN_ID} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
