#!/usr/bin/env python3
"""
Evaluate blended OOF probabilities for the ensemble base run.

Prints per-model OOF metrics, blend metrics, and a comparison table
against historical single-model runs (run_003 / run_004).

Run from the project root:
    python scripts/eval_blend_oof.py [--base-run run_005] [--weights lgbm=2,xgb=1]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
PROCESSED_DIR = ROOT / "data" / "processed"

GRADES = [1, 2, 3]


# ── Metrics ──────────────────────────────────────────────────────────────────


def compute_metrics(y: np.ndarray, oof_proba: np.ndarray) -> dict:
    preds = oof_proba.argmax(axis=1) + 1
    micro = f1_score(y, preds, average="micro", labels=GRADES)
    report = classification_report(y, preds, labels=GRADES, output_dict=True, zero_division=0)
    return {
        "micro_f1": micro,
        "g1_f1": report["1"]["f1-score"],
        "g2_f1": report["2"]["f1-score"],
        "g3_f1": report["3"]["f1-score"],
        "g1_recall": report["1"]["recall"],
        "g2_recall": report["2"]["recall"],
        "g3_recall": report["3"]["recall"],
        "g1_prec": report["1"]["precision"],
        "g2_prec": report["2"]["precision"],
        "g3_prec": report["3"]["precision"],
    }


def load_cv_mean(run_id: str) -> float | None:
    path = RUNS_DIR / run_id / "cv_scores.json"
    if not path.exists():
        path = RUNS_DIR / run_id / "metadata.json"
    if not path.exists():
        return None
    with path.open() as f:
        d = json.load(f)
    return d.get("mean") or d.get("cv_mean")


# ── Main ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-run", default="run_005")
    p.add_argument("--weights", default=None,
                   help="Comma-separated model=weight pairs, e.g. lgbm=2,xgb=1")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_run = args.base_run
    run_dir = RUNS_DIR / base_run

    # ── Load OOF arrays ───────────────────────────────────────────────────────
    oof_paths = sorted(run_dir.glob("oof_proba_*.npy"))
    if not oof_paths:
        sys.exit(f"No oof_proba files found in runs/{base_run}/. "
                 "Run src/train_ensemble.py first.")

    model_names = [p.stem.replace("oof_proba_", "") for p in oof_paths]
    oofs = {n: np.load(p) for n, p in zip(model_names, oof_paths)}

    y_path = PROCESSED_DIR / "y_train_multiclass.csv"
    if not y_path.exists():
        sys.exit(f"Missing {y_path}. Run src/preprocess.py first.")
    y = pd.read_csv(y_path)["damage_grade"].to_numpy()

    # ── Parse blend weights ───────────────────────────────────────────────────
    weights: dict[str, float] = {}
    if args.weights:
        for token in args.weights.split(","):
            if "=" in token:
                k, v = token.split("=", 1)
                weights[k.strip()] = float(v.strip())

    ws = np.array([weights.get(n, 1.0) for n in model_names], dtype=float)
    ws /= ws.sum()
    blend_proba = sum(float(ws[i]) * oofs[n] for i, n in enumerate(model_names))

    # ── Load per_model_cv from metadata ──────────────────────────────────────
    meta_path = run_dir / "metadata.json"
    saved_cv: dict = {}
    if meta_path.exists():
        with meta_path.open() as f:
            saved_cv = json.load(f).get("per_model_cv", {})

    # ── Print report ─────────────────────────────────────────────────────────
    print(f"\nBase run : {base_run}  |  models: {model_names}")
    if weights:
        print(f"Weights  : {dict(zip(model_names, ws.round(3)))}")
    print()

    hdr = f"  {'Source':<18} {'micro_F1':>9}  {'g1_F1':>7}  {'g2_F1':>7}  {'g3_F1':>7}  {'g1_rec':>7}  {'g2_rec':>7}  {'g3_rec':>7}"
    sep = "  " + "-" * (len(hdr) - 2)
    print(hdr)
    print(sep)

    def row(label: str, m: dict) -> str:
        return (
            f"  {label:<18} {m['micro_f1']:>9.4f}"
            f"  {m['g1_f1']:>7.4f}  {m['g2_f1']:>7.4f}  {m['g3_f1']:>7.4f}"
            f"  {m['g1_recall']:>7.4f}  {m['g2_recall']:>7.4f}  {m['g3_recall']:>7.4f}"
        )

    # Historical: load CV mean only (no OOF saved for runs 001-004)
    for hist_id in ["run_001", "run_002", "run_003", "run_004"]:
        cv = load_cv_mean(hist_id)
        if cv is not None:
            print(f"  {hist_id:<18} {cv:>9.4f}  {'(CV score, no OOF breakdown)':>52}")

    print(sep)

    # Per base model (OOF)
    for name in model_names:
        m = compute_metrics(y, oofs[name])
        cv_str = ""
        if name in saved_cv:
            cv_str = f"  [CV={saved_cv[name]['mean']:.4f}±{saved_cv[name]['std']:.4f}]"
        print(row(f"  oof_{name}", m) + cv_str)

    print(sep)

    # Blend
    blend_m = compute_metrics(y, blend_proba)
    print(row("  BLEND (uniform)", blend_m))

    # ── Delta vs run_004 ─────────────────────────────────────────────────────
    r4 = load_cv_mean("run_004")
    if r4 is not None:
        delta = blend_m["micro_f1"] - r4
        print(f"\n  Blend OOF micro_F1 vs run_004 CV: {blend_m['micro_f1']:.4f} vs {r4:.4f}"
              f"  →  delta = {delta:+.4f}")
        r3 = load_cv_mean("run_003")
        if r3 is not None:
            delta3 = blend_m["micro_f1"] - r3
            print(f"  Blend OOF micro_F1 vs run_003 CV: {blend_m['micro_f1']:.4f} vs {r3:.4f}"
                  f"  →  delta = {delta3:+.4f}")

    # ── Grade-3 highlight ────────────────────────────────────────────────────
    print(f"\n  Grade-3 detail (blend):")
    print(f"    F1={blend_m['g3_f1']:.4f}  recall={blend_m['g3_recall']:.4f}"
          f"  precision={blend_m['g3_prec']:.4f}")

    # Best individual model by g3 recall
    best_g3_model = max(model_names, key=lambda n: compute_metrics(y, oofs[n])["g3_recall"])
    best_g3 = compute_metrics(y, oofs[best_g3_model])
    print(f"    Best single model g3_recall: {best_g3_model} = {best_g3['g3_recall']:.4f}")
    print()


if __name__ == "__main__":
    main()
