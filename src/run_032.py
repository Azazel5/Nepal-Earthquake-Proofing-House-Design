#!/usr/bin/env python3
"""
run_032: Segment residual specialists + F1-constrained blend with run_026.

Segments (where run_026 predicts G2, true G2|G3):
  - weak_found: foundation w/i/u (highest G3 miss rate)
  - other_found: all other foundations

Blend: maximize G3 recall subject to OOF micro F1 >= run_026.

Run from project root:
    python src/run_032.py [--quick] [--skip-lgbm]
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
from scipy.stats import rankdata
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g23_features import FOUNDATION_WEAK
from residual_g23_features import ResidualG23FeatureBuilder, load_residual_frames
from run_manager import PROCESSED_DIR, RunManager

print = partial(print, flush=True)

RANDOM_STATE = 42
CV_FOLDS = 5
RUN_ID = "run_032"
RUN_026_ID = "run_026"
GRADES = [1, 2, 3]
F1_EPS = 1e-6
F1_TOLERANCE = 0.0002  # max allowed drop vs run_026 when maximizing G3

SEGMENTS = {
    "weak_found": lambda ft: ft.isin(FOUNDATION_WEAK),
    "other_found": lambda ft: ~ft.isin(FOUNDATION_WEAK),
}

LGBM_RESIDUAL_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": 6,
    "learning_rate": 0.04,
    "n_estimators": 1200,
    "min_child_samples": 40,
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


def _pred_g2(proba: np.ndarray) -> np.ndarray:
    return proba.argmax(axis=1) + 1 == 2


def _segment_rank(sp: np.ndarray, pred_g2: np.ndarray, seg_mask: np.ndarray) -> np.ndarray:
    out = np.full(len(sp), 0.5, dtype=np.float32)
    idx = np.where(pred_g2 & seg_mask)[0]
    if len(idx):
        out[idx] = (rankdata(sp[idx]) / len(idx)).astype(np.float32)
    return out


def _combine_segmented(
    p26: np.ndarray,
    sp_rank: np.ndarray,
    weak_row: np.ndarray,
    boost_w: float,
    tau_w: float,
    boost_o: float,
    tau_o: float,
    delta_min: float,
) -> np.ndarray:
    out = np.asarray(p26, dtype=np.float32).copy()
    pred = out.argmax(axis=1) + 1
    mass = out[:, 1] + out[:, 2]
    q26 = np.divide(out[:, 2], mass, out=np.full(len(out), 0.5, dtype=np.float32), where=mass > 1e-9)
    sp = np.asarray(sp_rank, dtype=np.float32)
    base_g2 = (pred == 2) & (mass > 0.5)

    for boost, tau, seg in (
        (boost_w, tau_w, weak_row),
        (boost_o, tau_o, ~weak_row),
    ):
        if boost <= 0.0:
            continue
        mask = base_g2 & seg & (sp >= tau)
        if delta_min > 0.0:
            mask = mask & (sp >= q26 + delta_min)
        if not np.any(mask):
            continue
        m = mass[mask]
        q_new = q26[mask] + boost * (sp[mask] - q26[mask])
        q_new = np.clip(q_new, q26[mask], 1.0)
        out[mask, 2] = m * q_new
        out[mask, 1] = m * (1.0 - q_new)

    out /= np.clip(out.sum(axis=1, keepdims=True), 1e-9, None)
    return out.astype(np.float64)


def search_f1_constrained_blend(
    p26: np.ndarray,
    sp_rank: np.ndarray,
    weak_row: np.ndarray,
    y: np.ndarray,
    f26_floor: float,
) -> dict:
    g3_base = _g3_recall(y, p26)
    floor = f26_floor - F1_TOLERANCE
    best = {
        "boost_w": 0.0, "tau_w": 1.0, "boost_o": 0.0, "tau_o": 1.0,
        "delta_min": 0.0, "f1": f26_floor, "g3": g3_base,
    }
    for delta_min in (0.0, 0.03):
        for tau_w in (0.82, 0.85, 0.88, 0.92, 0.96, 0.99):
            for tau_o in (0.88, 0.90, 0.95, 0.98, 0.99):
                for boost_w in (0.05, 0.10, 0.15, 0.20, 0.30):
                    for boost_o in (0.05, 0.10, 0.15, 0.20):
                        proba = _combine_segmented(
                            p26, sp_rank, weak_row,
                            boost_w, tau_w, boost_o, tau_o, delta_min,
                        )
                        f1 = _micro_f1(y, proba)
                        if f1 + F1_EPS < floor:
                            continue
                        g3 = _g3_recall(y, proba)
                        if g3 > best["g3"] + 1e-9 or (
                            abs(g3 - best["g3"]) < 1e-6 and f1 > best["f1"] + 1e-9
                        ):
                            best = {
                                "boost_w": boost_w, "tau_w": tau_w,
                                "boost_o": boost_o, "tau_o": tau_o,
                                "delta_min": delta_min, "f1": f1, "g3": g3,
                            }
    return best


def train_segment_oof(
    name: str,
    row_mask: np.ndarray,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    p26_oof: np.ndarray,
    p26_test: np.ndarray,
    pred26: np.ndarray,
    *,
    quick: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict], ResidualG23FeatureBuilder]:
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(train, y))
    n_folds = 1 if quick else CV_FOLDS

    oof_sp = np.full(len(y), np.nan, dtype=np.float32)
    test_folds: list[np.ndarray] = []
    fold_metrics: list[dict] = []

    for fold, (tri, vai) in enumerate(splits[:n_folds], start=1):
        t0 = time.time()
        tri_mask = np.zeros(len(y), dtype=bool)
        tri_mask[tri] = True
        fit_mask = tri_mask & pred26 & row_mask

        tri_res = tri[pred26[tri] & row_mask[tri] & np.isin(y[tri], [2, 3])]
        if len(tri_res) < 80:
            print(f"  [{name}] fold {fold}: skip (n_train={len(tri_res)})")
            continue

        fb = ResidualG23FeatureBuilder()
        fb.fit(train, y, residual_mask=fit_mask)

        X_tr = fb.transform(train.iloc[tri_res], p26_oof[tri_res])
        y_tr = (y[tri_res] == 3).astype(np.int8)

        va_res = vai[pred26[vai] & row_mask[vai] & np.isin(y[vai], [2, 3])]
        X_va = fb.transform(train.iloc[va_res], p26_oof[va_res]) if len(va_res) else X_tr[:1]
        y_va = (y[va_res] == 3).astype(np.int8) if len(va_res) else y_tr[:1]

        spw = (len(y_tr) - y_tr.sum()) / max(int(y_tr.sum()), 1)
        model = LGBMClassifier(**LGBM_RESIDUAL_PARAMS, scale_pos_weight=spw)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[early_stopping(stopping_rounds=EARLY_STOP, verbose=False)],
        )

        oof_sp[vai] = model.predict_proba(
            fb.transform(train.iloc[vai], p26_oof[vai]),
        )[:, 1].astype(np.float32)
        test_folds.append(model.predict_proba(fb.transform(test, p26_test))[:, 1].astype(np.float32))

        auc = (
            float(roc_auc_score(y_va, oof_sp[va_res]))
            if len(va_res) > 1 and len(np.unique(y_va)) > 1
            else float("nan")
        )
        miss = (y[vai] == 3) & pred26[vai] & row_mask[vai]
        fold_metrics.append({
            "segment": name, "fold": fold, "auc": auc, "n_train": len(tri_res),
            "g3_miss_recall@0.5": float(((miss) & (oof_sp[vai] >= 0.5)).sum() / max(miss.sum(), 1)),
        })
        print(
            f"  [{name}] fold {fold}: AUC={auc:.4f}  n={len(tri_res):,}  "
            f"miss_rec@0.5={fold_metrics[-1]['g3_miss_recall@0.5']:.3f}  ({time.time() - t0:.0f}s)",
        )

    fb_full = ResidualG23FeatureBuilder()
    fb_full.fit(train, y, residual_mask=pred26 & row_mask & np.isin(y, [2, 3]))
    test_sp = np.mean(test_folds, axis=0) if test_folds else np.full(len(test), 0.5, np.float32)
    return oof_sp, test_sp, fold_metrics, fb_full


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-lgbm", action="store_true")
    args = parser.parse_args()
    t0 = time.time()

    train, test = load_residual_frames()
    y = pd.read_csv(PROCESSED_DIR / "y_train_full.csv")["damage_grade"].to_numpy()
    ft = train["foundation_type"].astype(str)
    weak_row = ft.isin(FOUNDATION_WEAK).to_numpy()

    p26_oof = np.load(ROOT / "runs" / RUN_026_ID / "oof_proba.npy").astype(np.float64)
    p26_test = np.load(ROOT / "runs" / RUN_026_ID / "test_proba.npy").astype(np.float64)
    pred26 = _pred_g2(p26_oof)
    f26 = _micro_f1(y, p26_oof)
    g3_26 = _g3_recall(y, p26_oof)

    res_mask = pred26 & np.isin(y, [2, 3])
    print(f"run_026 OOF: {f26:.4f}  G3 recall: {g3_26:.4f}")
    print(f"Residual pool: {res_mask.sum():,}  weak={((res_mask) & weak_row).sum():,}  other={((res_mask) & ~weak_row).sum():,}")

    run_path = ROOT / "runs" / RUN_ID

    segment_oof: dict[str, np.ndarray] = {}
    segment_test: dict[str, np.ndarray] = {}
    segment_fb: dict[str, ResidualG23FeatureBuilder] = {}
    all_metrics: list[dict] = []

    for name, fn in SEGMENTS.items():
        seg_mask = fn(ft).to_numpy()
        oof_path = run_path / f"oof_{name}.npy"
        print(f"\n── Segment: {name} ({seg_mask.sum():,} rows) ──")
        if args.skip_lgbm and oof_path.exists():
            print(f"  Reusing {oof_path}")
            segment_oof[name] = np.load(oof_path).astype(np.float32)
            segment_fb[name] = ResidualG23FeatureBuilder()
            segment_fb[name].fit(train, y, residual_mask=pred26 & seg_mask & np.isin(y, [2, 3]))
            segment_test[name] = np.full(len(test), 0.5, np.float32)
        else:
            oof, te, metrics, fb = train_segment_oof(
                name, seg_mask, train, test, y, p26_oof, p26_test, pred26, quick=args.quick,
            )
            segment_oof[name] = oof
            segment_test[name] = te
            segment_fb[name] = fb
            all_metrics.extend(metrics)
            oof_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(oof_path, oof)

    # Merge: segment model scores on its rows; 0.5 elsewhere
    oof_merged = np.full(len(y), 0.5, dtype=np.float32)
    test_merged = np.full(len(test), 0.5, dtype=np.float32)
    test_weak = test["foundation_type"].astype(str).isin(FOUNDATION_WEAK).to_numpy()
    for name, fn in SEGMENTS.items():
        m = fn(ft).to_numpy()
        v = ~np.isnan(segment_oof[name])
        oof_merged[m & v] = segment_oof[name][m & v]
        tm = fn(test["foundation_type"].astype(str)).to_numpy()
        oof_merged[m & ~v] = 0.5
        test_merged[tm] = segment_test[name][tm]

    oof_rank = np.full(len(y), 0.5, dtype=np.float32)
    for name, fn in SEGMENTS.items():
        m = fn(ft).to_numpy()
        r = _segment_rank(oof_merged, pred26, m)
        oof_rank[m] = r[m]

    pool = res_mask
    for name, fn in SEGMENTS.items():
        m = pool & fn(ft).to_numpy()
        if m.sum() > 10 and len(np.unique(y[m])) > 1:
            auc = roc_auc_score(y[m] == 3, oof_merged[m])
            print(f"  OOF AUC [{name}]: {auc:.4f}  (n={m.sum():,})")

    print(f"\n── F1-constrained blend (floor={f26:.4f}, tol={F1_TOLERANCE}) ──")
    best = search_f1_constrained_blend(p26_oof, oof_rank, weak_row, y, f26)
    print(
        f"  weak: boost={best['boost_w']:.2f} tau={best['tau_w']:.2f}  "
        f"other: boost={best['boost_o']:.2f} tau={best['tau_o']:.2f}  "
        f"delta={best['delta_min']:.2f}",
    )
    print(f"  OOF F1: {best['f1']:.4f}  G3 recall: {best['g3']:.4f}")
    print(f"  Δ vs run_026: F1 {best['f1'] - f26:+.4f}  G3 {best['g3'] - g3_26:+.4f}")

    if args.quick:
        print(f"\nQuick done ({time.time() - t0:.1f}s)")
        return

    # Full refit per segment
    models: dict[str, LGBMClassifier] = {}
    for name, fn in SEGMENTS.items():
        m = pred26 & fn(ft).to_numpy() & np.isin(y, [2, 3])
        idx = np.where(m)[0]
        fb = segment_fb[name]
        X = fb.transform(train.iloc[idx], p26_oof[idx])
        y_bin = (y[idx] == 3).astype(np.int8)
        spw = (len(y_bin) - y_bin.sum()) / max(int(y_bin.sum()), 1)
        model = LGBMClassifier(**LGBM_RESIDUAL_PARAMS, scale_pos_weight=spw)
        model.fit(X, y_bin)
        models[name] = model
        joblib.dump(model, run_path / f"specialist_{name}.pkl")
        joblib.dump(fb, run_path / f"feature_builder_{name}.pkl")

    test_pred_g2 = _pred_g2(p26_test)
    test_sp = np.full(len(test), 0.5, dtype=np.float32)
    for name, fn in SEGMENTS.items():
        tm = fn(test["foundation_type"].astype(str)).to_numpy()
        if not tm.any():
            continue
        p = models[name].predict_proba(
            segment_fb[name].transform(test, p26_test),
        )[:, 1].astype(np.float32)
        test_sp[tm] = p[tm]

    test_rank = np.full(len(test), 0.5, dtype=np.float32)
    for name, fn in SEGMENTS.items():
        tm = fn(test["foundation_type"].astype(str)).to_numpy()
        r = _segment_rank(test_sp, test_pred_g2, tm)
        test_rank[tm] = r[tm]

    blend_kw = dict(
        boost_w=best["boost_w"], tau_w=best["tau_w"],
        boost_o=best["boost_o"], tau_o=best["tau_o"],
        delta_min=best["delta_min"],
    )
    oof_proba = _combine_segmented(p26_oof, oof_rank, weak_row, **blend_kw).astype(np.float32)
    test_proba = _combine_segmented(p26_test, test_rank, test_weak, **blend_kw).astype(np.float32)

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = [
        _micro_f1(y[va], _combine_segmented(p26_oof[va], oof_rank[va], weak_row[va], **blend_kw))
        for _f, (_tr, va) in enumerate(skf.split(p26_oof, y), start=1)
    ]

    rm = RunManager()
    rp = rm.run_path(RUN_ID)
    rp.mkdir(parents=True, exist_ok=True)
    if not (rp / "metadata.json").exists():
        try:
            rm.create_run(
                description="run_026 + segment residual specialists, F1-constrained blend",
                model_type="ensemble_segment_residual",
                feature_set="residual_g23_per_segment",
                params={"base_run": RUN_026_ID, "segments": list(SEGMENTS), "f1_tolerance": F1_TOLERANCE, **blend_kw, "f1_floor": f26},
                run_id=RUN_ID,
                objective="multiclass",
                cv_folds=CV_FOLDS,
                cv_metric="micro_f1",
                notes="Maximize G3 recall with F1 >= run_026 - tolerance.",
            )
        except FileExistsError:
            from datetime import datetime, timezone
            init = {
                "run_id": RUN_ID,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "description": "run_026 + segment residual specialists, F1-constrained blend",
                "model_type": "ensemble_segment_residual",
                "feature_set": "residual_g23_per_segment",
                "params": {"base_run": RUN_026_ID, "segments": list(SEGMENTS), **blend_kw},
                "cv_folds": CV_FOLDS,
                "cv_metric": "micro_f1",
                "notes": "Maximize G3 recall with F1 >= run_026 - tolerance.",
            }
            RunManager._write_json(rp / "metadata.json", init)
            RunManager._write_json(rp / "params.json", init["params"])

    rm.save_cv_scores(RUN_ID, fold_scores, float(np.mean(fold_scores)), float(np.std(fold_scores, ddof=1)))
    np.save(rp / "oof_proba.npy", oof_proba)
    np.save(rp / "oof_merged_sp.npy", oof_merged)
    np.save(rp / "test_proba.npy", test_proba)

    sub = pd.DataFrame({
        "building_id": pd.read_csv(ROOT / "data" / "driven_data" / "test_values.csv")["building_id"],
        "damage_grade": test_proba.argmax(axis=1) + 1,
    })
    rm.save_submission(RUN_ID, sub)

    meta = rm.load_metadata(RUN_ID)
    meta.update({
        "run_026_oof": f26, "run_026_g3_recall": g3_26,
        "blend_oof_f1": best["f1"], "blend_g3_recall": best["g3"],
        "gain_f1_vs_run_026": best["f1"] - f26,
        "gain_g3_recall_vs_run_026": best["g3"] - g3_26,
        "segment_metrics": all_metrics,
        **{f"blend_{k}": v for k, v in blend_kw.items()},
    })
    RunManager._write_json(rp / "metadata.json", meta)

    print("\n" + "═" * 60)
    print(f"  run_026:       F1={f26:.4f}  G3={g3_26:.4f}")
    print(f"  run_032:       F1={best['f1']:.4f}  G3={best['g3']:.4f}")
    print(f"  Per-fold:      {np.mean(fold_scores):.4f} ± {np.std(fold_scores, ddof=1):.4f}")
    flips = ((_pred_g2(p26_oof)) & (oof_proba.argmax(1) + 1 == 3)).sum()
    print(f"  G2→G3 flips:   {flips:,}")
    for g, c in zip(*np.unique(sub["damage_grade"], return_counts=True)):
        print(f"  submission grade {g}: {c:,} ({c/len(sub)*100:.1f}%)")
    print(f"\nRegistered {RUN_ID} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
