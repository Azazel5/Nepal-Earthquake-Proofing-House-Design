#!/usr/bin/env python3
"""
Submission generation from saved ensemble artifacts.

Loads test probability arrays from an ensemble base run and a trained
meta-model (from a stack run) to produce a fresh submission without
retraining. Useful for experimenting with different blend weights or
switching between stacking and blending after stack.py has been run.

Usage
-----
    # Stacked (default) — uses meta_model.pkl from the stack run
    python src/submit_ensemble.py

    # Uniform blend from base learner test probas
    python src/submit_ensemble.py --method blend

    # Specify runs explicitly
    python src/submit_ensemble.py --stack-run-id run_006 --base-run-id run_005

    # Custom blend weights (proportional; missing models get weight 1)
    python src/submit_ensemble.py --method blend --weights lgbm=2 catboost=1.5

    # Save under a different file name
    python src/submit_ensemble.py --method blend --out my_blend.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import GRADES, load_run003_features
from inference import grades_to_submission_df, print_grade_distribution
from run_manager import ROOT, RunManager
from stack import (
    blend_proba,
    build_stacking_matrix,
    load_base_artifacts,
    proba_to_grades,
)
from train_ensemble import align_proba


def _find_stack_run(rm: RunManager, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    for rid in reversed(rm.list_runs()):
        try:
            meta = rm.load_metadata(rid)
        except Exception:
            continue
        if str(meta.get("model_type", "")).startswith("ensemble_stack"):
            return rid
    return None


def _find_base_run(rm: RunManager, explicit: str | None) -> str:
    if explicit:
        return explicit
    for rid in reversed(rm.list_runs()):
        try:
            meta = rm.load_metadata(rid)
        except Exception:
            continue
        if meta.get("model_type") == "ensemble_base":
            return rid
    raise FileNotFoundError(
        "No ensemble_base run found. Pass --base-run-id or run src/train_ensemble.py."
    )


def parse_weights(raw: list[str] | None) -> dict[str, float] | None:
    """Parse 'model=weight' strings into a dict."""
    if not raw:
        return None
    out: dict[str, float] = {}
    for item in raw:
        if "=" not in item:
            print(f"WARNING: skipping malformed weight '{item}' (expected 'model=value')",
                  file=sys.stderr)
            continue
        model, val = item.split("=", 1)
        try:
            out[model.strip()] = float(val.strip())
        except ValueError:
            print(f"WARNING: non-numeric weight '{val}' for '{model}'", file=sys.stderr)
    return out or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ensemble submission from saved artifacts")
    parser.add_argument("--method", choices=["stack", "blend"], default="stack",
                        help="stack: use trained meta-model; blend: weighted average (default: stack)")
    parser.add_argument("--stack-run-id", type=str, default=None,
                        help="Run containing meta_model.pkl (default: most recent stack run)")
    parser.add_argument("--base-run-id", type=str, default=None,
                        help="Run containing oof/test_proba_*.npy (default: auto-detect)")
    parser.add_argument("--weights", nargs="+", default=None,
                        metavar="MODEL=WEIGHT",
                        help="Custom blend weights e.g. --weights lgbm=2 xgb=1")
    parser.add_argument("--out", type=str, default=None,
                        help="Output CSV filename (default: submission_{method}.csv in stack run)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rm = RunManager()

    base_run_id = _find_base_run(rm, args.base_run_id)
    base_run_dir = rm.run_path(base_run_id)
    print(f"Base run: {base_run_id}")

    oof_probas, test_probas, model_names = load_base_artifacts(base_run_dir)
    print(f"Models: {model_names}")

    _, _, _, building_ids = load_run003_features()

    # ── Generate grades ───────────────────────────────────────────────────────
    if args.method == "blend":
        weights = parse_weights(args.weights)
        blend_p = blend_proba(test_probas, model_names, weights=weights)
        grades = proba_to_grades(blend_p)
        label = "blend"
        if weights:
            label += "(" + ",".join(f"{k}={v:.1f}" for k, v in sorted(weights.items())) + ")"

    else:  # stack
        stack_run_id = _find_stack_run(rm, args.stack_run_id)
        if stack_run_id is None:
            print("ERROR: No stack run found. Run src/stack.py first, or use --method blend.",
                  file=sys.stderr)
            sys.exit(1)
        print(f"Stack run: {stack_run_id}")
        meta_model_path = rm.run_path(stack_run_id) / "meta_model.pkl"
        if not meta_model_path.exists():
            print(f"ERROR: Missing {meta_model_path}", file=sys.stderr)
            sys.exit(1)

        meta_model = joblib.load(meta_model_path)
        S_test = build_stacking_matrix(test_probas, model_names)

        if hasattr(meta_model, "predict_proba"):
            raw_proba = meta_model.predict_proba(
                pd.DataFrame(S_test) if hasattr(meta_model, "booster_") else S_test
            )
            if hasattr(meta_model, "classes_"):
                stacked_proba = align_proba(raw_proba, meta_model.classes_)
            else:
                stacked_proba = raw_proba
            grades = proba_to_grades(stacked_proba)
        else:
            grades = meta_model.predict(S_test).astype(int)

        label = "stacked"

    # ── Save ──────────────────────────────────────────────────────────────────
    submission = grades_to_submission_df(building_ids, grades)
    print_grade_distribution(grades, f"\n{label} grade distribution")

    if args.out:
        out_path = Path(args.out)
    elif args.method == "stack" and stack_run_id:
        out_path = rm.run_path(stack_run_id) / f"submission_{label}.csv"
    else:
        out_path = ROOT / "outputs" / f"submission_{label}.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    print(f"Saved: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
