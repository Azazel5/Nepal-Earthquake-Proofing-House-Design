#!/usr/bin/env python3
"""
run_033: G3 false-negative error analysis on run_026 OOF.

Profiles rows where the ensemble predicts G2 but truth is G3 (G3→G2),
compares to true G2 among pred-G2, and scores candidate interaction segments
for feature-gap vs boundary-shift potential.

Run from project root:
    python src/run_033.py
"""

from __future__ import annotations

import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g23_features import FOUNDATION_STRONG, FOUNDATION_WEAK, load_g23_frames
from run_manager import PROCESSED_DIR, RunManager

print = partial(print, flush=True)

RUN_ID = "run_033"
BASE_RUN = "run_026"
GRADES = [1, 2, 3]

SEGMENT_COLS = [
    "foundation_type",
    "geo_level_1_id",
    "geo_level_3_id",
    "plan_configuration",
    "position",
    "roof_type",
    "land_surface_condition",
    "ground_floor_type",
]
NUMERIC_BINS = {
    "age": [0, 5, 15, 30, 100],
    "count_floors_pre_eq": [0, 1, 2, 4, 20],
    "height_percentage": [0, 3, 5, 8, 100],
}
WEAK_SUPER = [
    "has_superstructure_adobe_mud",
    "has_superstructure_mud_mortar_stone",
    "has_superstructure_bamboo",
]
STRONG_SUPER = [
    "has_superstructure_rc_engineered",
    "has_superstructure_cement_mortar_brick",
    "has_superstructure_cement_mortar_stone",
]


def _segment_table(
    df: pd.DataFrame,
    col: str,
    mask_fn: np.ndarray,
    ref_fn: np.ndarray,
    ref_g2: np.ndarray,
    min_n: int = 30,
) -> pd.DataFrame:
    """Per-level FN rate among true G3 and enrichment vs pred-G2 true G2."""
    rows = []
    for val in df[col].astype(str).unique():
        m = (df[col].astype(str) == val).to_numpy()
        n_g3 = int((m & (df["y"].to_numpy() == 3)).sum())
        if n_g3 < min_n:
            continue
        fn = m & mask_fn
        g3 = m & (df["y"].to_numpy() == 3)
        fn_rate = fn.sum() / max(g3.sum(), 1)
        g2_pred = m & ref_g2
        g3_among_g2pred = (g2_pred & (df["y"].to_numpy() == 3)).sum()
        g3_rate_among_g2pred = g3_among_g2pred / max(g2_pred.sum(), 1)
        rows.append({
            "segment": col,
            "value": val,
            "n_true_g3": n_g3,
            "n_fn_g3_g2": int(fn.sum()),
            "fn_rate": float(fn_rate),
            "n_pred_g2": int(g2_pred.sum()),
            "g3_rate_among_pred_g2": float(g3_rate_among_g2pred),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("fn_rate", ascending=False)
    return out


def _combo_lift(
    df: pd.DataFrame,
    keys: list[str],
    fn_mask: np.ndarray,
    pred_g2: np.ndarray,
    min_n: int = 25,
) -> pd.DataFrame:
    y = df["y"].to_numpy()
    combo = df[keys[0]].astype(str)
    for k in keys[1:]:
        combo = combo + "|" + df[k].astype(str)
    df = df.copy()
    df["_combo"] = combo
    rows = []
    global_fn = fn_mask.sum() / max((y == 3).sum(), 1)
    global_g3_in_g2pred = ((pred_g2) & (y == 3)).sum() / max(pred_g2.sum(), 1)
    for val, grp in df.groupby("_combo"):
        g3 = grp["y"].to_numpy() == 3
        if g3.sum() < min_n:
            continue
        fn = grp.index.to_numpy()
        fn_m = fn_mask[fn]
        pg2 = pred_g2[fn]
        fn_rate = fn_m.sum() / max(g3.sum(), 1)
        g3_in_pg2 = (pg2 & (grp["y"].to_numpy() == 3)).sum() / max(pg2.sum(), 1)
        rows.append({
            "combo": val,
            "keys": "+".join(keys),
            "n_true_g3": int(g3.sum()),
            "fn_rate": float(fn_rate),
            "fn_lift_vs_global": float(fn_rate / max(global_fn, 1e-9)),
            "g3_rate_pred_g2": float(g3_in_pg2),
            "lift_pred_g2": float(g3_in_pg2 / max(global_g3_in_g2pred, 1e-9)),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("fn_lift_vs_global", ascending=False)
    return out


def main() -> None:
    t0 = time.time()
    train, _ = load_g23_frames()
    y = pd.read_csv(PROCESSED_DIR / "y_train_full.csv")["damage_grade"].to_numpy()
    p26 = np.load(ROOT / "runs" / BASE_RUN / "oof_proba.npy")

    pred = p26.argmax(axis=1) + 1
    f26 = f1_score(y, pred, average="micro", labels=GRADES)
    g3_rec = recall_score(y == 3, pred == 3, zero_division=0)

    g3_fn_g2 = (y == 3) & (pred == 2)
    g3_fn_g1 = (y == 3) & (pred == 1)
    g3_ok = (y == 3) & (pred == 3)
    pred_g2 = pred == 2
    true_g2_pred_g2 = (y == 2) & pred_g2

    mass = p26[:, 1] + p26[:, 2]
    q26 = np.divide(p26[:, 2], mass, out=np.zeros(len(y)), where=mass > 1e-9)

    df = train.copy()
    df["y"] = y
    df["pred"] = pred
    df["p1"] = p26[:, 0]
    df["p2"] = p26[:, 1]
    df["p3"] = p26[:, 2]
    df["q26_g3"] = q26
    df["margin_23"] = p26[:, 1] - p26[:, 2]
    df["weak_found"] = df["foundation_type"].astype(str).isin(FOUNDATION_WEAK)
    df["strong_found"] = df["foundation_type"].astype(str).isin(FOUNDATION_STRONG)
    if all(c in df.columns for c in WEAK_SUPER):
        df["weak_super"] = df[WEAK_SUPER].max(axis=1).astype(bool)
        df["strong_super"] = df[STRONG_SUPER].max(axis=1).astype(bool)
    else:
        df["weak_super"] = False
        df["strong_super"] = False

    for col, bins in NUMERIC_BINS.items():
        df[f"{col}_bin"] = pd.cut(df[col], bins=bins, labels=False, include_lowest=True)

    print("=" * 72)
    print("run_033 — G3 FALSE-NEGATIVE ERROR ANALYSIS (run_026 OOF)")
    print("=" * 72)
    print(f"Base OOF F1: {f26:.4f}  G3 recall: {g3_rec:.4f}")
    print(f"True G3: {(y==3).sum():,}  G3→G2 FN: {g3_fn_g2.sum():,} ({g3_fn_g2.sum()/(y==3).sum()*100:.1f}%)")
    print(f"G3→G1: {g3_fn_g1.sum():,} ({g3_fn_g1.sum()/(y==3).sum()*100:.1f}%)  G3 correct: {g3_ok.sum():,}")

    print("\n── 1. Error taxonomy (all true G3) ──")
    for label, m in [
        ("G3→G2 (main FN)", g3_fn_g2),
        ("G3→G1 (skip grade)", g3_fn_g1),
        ("G3 correct", g3_ok),
    ]:
        print(f"  {label:22s} {m.sum():6,}  mean P(G2)={p26[m,1].mean():.3f}  P(G3)={p26[m,2].mean():.3f}")

    print("\n── 2. Pred-G2 pool: true G3 vs true G2 (discriminability) ──")
    pool = pred_g2 & np.isin(y, [2, 3])
    print(f"  Pool size: {pool.sum():,}  true G3: {(pool & (y==3)).sum():,}  true G2: {(pool & (y==2)).sum():,}")
    print(f"  Global G3 rate in pool: {(pool & (y==3)).sum()/pool.sum()*100:.1f}%")
    for label, m in [("true G3 in pool", pool & (y==3)), ("true G2 in pool", pool & (y==2))]:
        print(f"  {label}: q26={q26[m].mean():.3f}  margin(p2-p3)={df.loc[m, 'margin_23'].mean():.3f}")

    print("\n── 3. Univariate segments (worst G3→G2 FN rate) ──")
    uni_tables: dict[str, list] = {}
    for col in SEGMENT_COLS + ["weak_found", "strong_found", "weak_super", "strong_super"]:
        if col not in df.columns:
            continue
        ser = df[col].astype(str) if df[col].dtype == bool else df[col]
        tab = _segment_table(
            df.assign(_seg=ser), "_seg", g3_fn_g2, g3_fn_g2, pred_g2, min_n=40,
        )
        if len(tab):
            uni_tables[col] = tab.head(12).to_dict(orient="records")
            print(f"\n  [{col}] top FN rates:")
            for _, r in tab.head(5).iterrows():
                print(
                    f"    {r['value']:20s}  FN={r['fn_rate']*100:5.1f}%  "
                    f"n_g3={int(r['n_true_g3']):5d}  g3|pred_g2={r['g3_rate_among_pred_g2']*100:5.1f}%",
                )

    print("\n── 4. Numeric bins ──")
    for col in NUMERIC_BINS:
        bcol = f"{col}_bin"
        tab = _segment_table(df, bcol, g3_fn_g2, g3_fn_g2, pred_g2, min_n=40)
        if len(tab):
            print(f"  [{col}]")
            for _, r in tab.iterrows():
                val = r["value"]
                try:
                    val_s = str(int(float(val)))
                except (TypeError, ValueError):
                    val_s = "nan"
                print(f"    bin {val_s}  FN={r['fn_rate']*100:.1f}%  n_g3={int(r['n_true_g3'])}")

    print("\n── 5. Interaction combos (FN lift vs global) ──")
    combo_specs = [
        ["foundation_type", "geo_level_1_id"],
        ["foundation_type", "geo_level_3_id"],
        ["foundation_type", "plan_configuration"],
        ["geo_level_1_id", "plan_configuration"],
        ["geo_level_3_id", "weak_super"],
        ["foundation_type", "age_bin"],
        ["foundation_type", "count_floors_pre_eq_bin"],
    ]
    combo_tables: dict[str, list] = {}
    global_fn = g3_fn_g2.sum() / (y == 3).sum()
    for keys in combo_specs:
        if not all(k in df.columns for k in keys):
            continue
        tab = _combo_lift(df, keys, g3_fn_g2, pred_g2, min_n=30)
        if len(tab):
            combo_tables["+".join(keys)] = tab.head(15).to_dict(orient="records")
            print(f"\n  [{' + '.join(keys)}] top lifts:")
            for _, r in tab.head(4).iterrows():
                print(
                    f"    {str(r['combo'])[:45]:45s}  FN={r['fn_rate']*100:5.1f}%  "
                    f"lift={r['fn_lift_vs_global']:.2f}x  g3|g2pred={r['g3_rate_pred_g2']*100:.1f}%",
                )

    print("\n── 6. Borderline vs confident FN ──")
    fn_idx = np.where(g3_fn_g2)[0]
    border = g3_fn_g2 & (q26 > 0.35) & (q26 < 0.45)
    conf = g3_fn_g2 & (q26 < 0.25)
    print(f"  Borderline FN (0.35<q26<0.45): {border.sum():,} ({border.sum()/g3_fn_g2.sum()*100:.1f}% of FN)")
    print(f"  Confident G2 FN (q26<0.25):      {conf.sum():,} ({conf.sum()/g3_fn_g2.sum()*100:.1f}% of FN)")
    print(f"  Recoverable by small q26 shift (+0.05): {(g3_fn_g2 & (q26 > 0.45)).sum():,} already near flip")

    print("\n── 7. Go / no-go signals ──")
    weak_fn = g3_fn_g2 & df["weak_found"].to_numpy()
    other_fn = g3_fn_g2 & ~df["weak_found"].to_numpy()
    print(f"  FN in weak foundation: {weak_fn.sum():,} ({weak_fn.sum()/g3_fn_g2.sum()*100:.1f}% of all G3→G2)")
    print(f"  FN in other foundation: {other_fn.sum():,}")
    # separability: mean q26 difference in pred_g2 pool
    d_q = q26[pool & (y==3)].mean() - q26[pool & (y==2)].mean()
    print(f"  q26 gap (G3 vs G2 in pred-G2 pool): {d_q:.4f}  (small gap → boundary problem)")
    print(f"  G3→G1 errors (ordinal skip): {g3_fn_g1.sum():,} — ordinal head may help separately")

    recommendations = []
    if weak_fn.sum() > 0.2 * g3_fn_g2.sum():
        recommendations.append(
            "FEATURE: weak-foundation × geo interaction rates (w/i/u); "
            f"{weak_fn.sum():,} FN rows ({weak_fn.sum()/g3_fn_g2.sum()*100:.0f}%)"
        )
    if g3_fn_g1.sum() > 500:
        recommendations.append(
            f"ORDINAL: {g3_fn_g1.sum():,} G3→G1 skips — cumulative/ordinal loss worth testing"
        )
    if abs(d_q) < 0.05:
        recommendations.append(
            "BOUNDARY: q26 barely separates G3 vs G2 in pred-G2 pool — "
            "recall boosts likely hurt micro F1; prefer pseudo-label or new features"
        )
    else:
        recommendations.append(
            "BLEND: q26 gap exists — targeted upgrades on high-lift combos may work with F1 floor"
        )
    recommendations.append(
        "PSEUDO-LABEL: rare geo3 in test — check train vs test geo3 coverage before PL round"
    )

    print("\n── 8. Recommendations ──")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")

    run_dir = ROOT / "runs" / RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "base_run": BASE_RUN,
        "oof_f1": float(f26),
        "g3_recall": float(g3_rec),
        "n_true_g3": int((y == 3).sum()),
        "n_g3_fn_g2": int(g3_fn_g2.sum()),
        "n_g3_fn_g1": int(g3_fn_g1.sum()),
        "fn_rate_global": float(global_fn),
        "pred_g2_pool_n": int(pool.sum()),
        "pred_g2_g3_rate": float((pool & (y == 3)).sum() / pool.sum()),
        "q26_gap_g3_vs_g2_in_pool": float(d_q),
        "borderline_fn_n": int(border.sum()),
        "confident_fn_n": int(conf.sum()),
        "weak_found_fn_n": int(weak_fn.sum()),
        "univariate": uni_tables,
        "interactions": combo_tables,
        "recommendations": recommendations,
        "fn_prob_summary": {
            "g3_fn_g2": {"mean_p2": float(p26[g3_fn_g2, 1].mean()), "mean_p3": float(p26[g3_fn_g2, 2].mean()), "mean_q26": float(q26[g3_fn_g2].mean())},
            "true_g2_pred_g2": {"mean_q26": float(q26[true_g2_pred_g2].mean())},
            "true_g3_pred_g2": {"mean_q26": float(q26[g3_fn_g2].mean())},
        },
    }

    with (run_dir / "g3_fn_report.json").open("w") as f:
        json.dump(report, f, indent=2)

    md_lines = [
        "# run_033 G3 False-Negative Report",
        "",
        f"Base: **{BASE_RUN}** OOF F1={f26:.4f}, G3 recall={g3_rec:.4f}",
        "",
        "## Summary",
        f"- G3→G2 errors: **{g3_fn_g2.sum():,}** / {(y==3).sum():,} true G3 ({g3_fn_g2.sum()/(y==3).sum()*100:.1f}%)",
        f"- G3→G1 errors: {g3_fn_g1.sum():,}",
        f"- Pred-G2 pool G3 rate: {(pool & (y==3)).sum()/pool.sum()*100:.1f}%",
        f"- q26 gap (G3 vs G2 in pool): {d_q:.4f}",
        "",
        "## Recommendations",
    ]
    for rec in recommendations:
        md_lines.append(f"- {rec}")
    (run_dir / "g3_fn_report.md").write_text("\n".join(md_lines) + "\n")

    rm = RunManager()
    rp = rm.run_path(RUN_ID)
    if not (rp / "metadata.json").exists():
        try:
            rm.create_run(
                description="G3 false-negative error analysis on run_026 OOF",
                model_type="analysis",
                feature_set="g3_fn_report",
                params={"base_run": BASE_RUN},
                run_id=RUN_ID,
                objective="analysis",
                notes="No model trained; diagnostic report for feature/PL/ordinal decisions.",
            )
        except FileExistsError:
            pass

    meta_path = rp / "metadata.json"
    if meta_path.exists():
        meta = rm.load_metadata(RUN_ID)
    else:
        meta = {"run_id": RUN_ID}
    meta.update({
        "g3_fn_g2": int(g3_fn_g2.sum()),
        "g3_fn_g1": int(g3_fn_g1.sum()),
        "recommendations": recommendations,
        "oof_f1_base": float(f26),
    })
    RunManager._write_json(rp / "metadata.json", meta)

    print(f"\nSaved runs/{RUN_ID}/g3_fn_report.json and g3_fn_report.md")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
