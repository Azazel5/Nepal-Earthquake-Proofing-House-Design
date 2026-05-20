# pip install lightgbm category_encoders if missing
"""
Generate submission from the best CV run (outputs/best_run.json) or a specified run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

from inference import (
    build_test_matrix,
    grades_to_submission_df,
    predict_multiclass,
    print_grade_distribution,
)
from run_manager import ROOT, RunManager


def resolve_run_id(rm: RunManager, run_id: str | None) -> str:
    """Use CLI run_id or best_cv_run from outputs/best_run.json."""
    if run_id:
        return run_id
    best_path = rm.best_run_path
    if not best_path.exists():
        print("ERROR: No outputs/best_run.json — run retrain.py or update_best_run().", file=sys.stderr)
        sys.exit(1)
    with best_path.open(encoding="utf-8") as f:
        best = json.load(f)
    rid = best.get("best_cv_run")
    if not rid:
        print("ERROR: best_run.json has no best_cv_run.", file=sys.stderr)
        sys.exit(1)
    return rid


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate submission from a run model.")
    parser.add_argument("--run-id", default=None, help="e.g. run_002 (default: best_cv_run)")
    args = parser.parse_args()

    rm = RunManager()
    run_id = resolve_run_id(rm, args.run_id)
    model_path = rm.run_path(run_id) / "model.pkl"

    if not model_path.exists():
        print(f"ERROR: Missing {model_path.relative_to(ROOT)}", file=sys.stderr)
        sys.exit(1)

    model = joblib.load(model_path)
    building_ids, X_test = build_test_matrix()
    print(f"Test set shape after preprocessing: {X_test.shape}")

    grades = predict_multiclass(model, X_test)
    submission = grades_to_submission_df(building_ids, grades)
    out_path = rm.save_submission(run_id, submission)

    meta = rm.load_metadata(run_id)
    print(f"\nRun: {run_id} — {meta.get('description', '')}")
    print_grade_distribution(grades, "Prediction grade distribution")
    print(f"Ready to submit: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
