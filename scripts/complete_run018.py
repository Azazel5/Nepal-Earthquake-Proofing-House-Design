#!/usr/bin/env python3
"""Re-run run_018 blend diagnostics from saved OOF (no retraining)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_manager import PROCESSED_DIR, RunManager
from run_trees_260k import (
    _oof_f1,
    build_blend_submission,
    pairwise_diagnostic,
    threeway_optimize,
)

LGBM_CV_REF = 0.7588

LGBM_OOF = ROOT / "runs" / "run_012" / "oof_proba.npy"
LGBM_TEST = ROOT / "runs" / "run_012" / "test_proba.npy"
XGB_OOF = ROOT / "runs" / "run_015" / "oof_proba.npy"
XGB_TEST = ROOT / "runs" / "run_015" / "test_proba.npy"
RUN018_OOF = ROOT / "runs" / "run_018" / "oof_proba.npy"
RUN018_TEST = ROOT / "runs" / "run_018" / "test_proba.npy"

NOISE = 0.0016
THRESHOLD = LGBM_CV_REF + NOISE


def main() -> None:
    y = pd.read_csv(PROCESSED_DIR / "y_train_full.csv")["damage_grade"].to_numpy()
    oof_018 = np.load(RUN018_OOF).astype(np.float64)
    test_018 = np.load(RUN018_TEST).astype(np.float64)
    lgbm_oof = np.load(LGBM_OOF).astype(np.float64)
    lgbm_test = np.load(LGBM_TEST).astype(np.float64)
    xgb_oof = np.load(XGB_OOF).astype(np.float64)
    xgb_test = np.load(XGB_TEST).astype(np.float64)

    solo_f1 = _oof_f1(oof_018, y)
    print(f"run_018 solo OOF F1: {solo_f1:.4f}")

    print("\n" + "═" * 60)
    print("PAIRWISE BLEND DIAGNOSTICS (reference LGBM run_012 OOF = 0.7588)")
    print("═" * 60)
    r = pairwise_diagnostic("lgbm_012", lgbm_oof, "run_018", oof_018, y)

    print("\n" + "═" * 60)
    print("3-WAY BLEND: run_012 + run_015 XGB + run_018")
    print("═" * 60)
    f3, w_lg, w_xg, w_018 = threeway_optimize(lgbm_oof, xgb_oof, oof_018, y)
    print(f"  Best 3-way OOF F1: {f3:.4f}")
    print(f"  Weights — LGBM: {w_lg:.3f}  XGB: {w_xg:.3f}  run_018: {w_018:.3f}")

    best_2way = r["best_score"]
    best_3way = f3
    best_overall = max(best_2way, best_3way)

    print("\n" + "═" * 60)
    print("SUMMARY")
    print("═" * 60)
    print(f"  LGBM run_012 solo: {LGBM_CV_REF:.4f}  (reference)")
    print(f"  run_018 solo:      {solo_f1:.4f}")
    print(f"  2-way lgbm+run_018: {best_2way:.4f}  gain={best_2way - LGBM_CV_REF:+.4f}")
    print(f"  3-way blend:        {best_3way:.4f}  gain={best_3way - LGBM_CV_REF:+.4f}")
    print(f"  Best overall:       {best_overall:.4f}  "
          f"{'>>> CLEARS THRESHOLD <<<' if best_overall > THRESHOLD else f'noise (threshold={THRESHOLD:.4f})'}")

    if best_overall > THRESHOLD:
        if best_3way >= best_2way:
            build_blend_submission(
                {"lgbm": lgbm_test, "xgb": xgb_test, "run_018": test_018},
                {"lgbm": w_lg, "xgb": w_xg, "run_018": w_018},
                space="proba",
                tag=f"run018_3way_lg{w_lg:.2f}_xg{w_xg:.2f}_018_{w_018:.2f}",
            )
        else:
            build_blend_submission(
                {"lgbm": lgbm_test, "run_018": test_018},
                {"lgbm": r["best_alpha"], "run_018": 1.0 - r["best_alpha"]},
                space=r["best_space"],
                tag=f"run018_2way_a{r['best_alpha']:.3f}",
            )
    else:
        print("\n  No blend clears the noise threshold.")

    sub_path = ROOT / "runs" / "run_018" / "submission.csv"
    print(f"\nSolo submission ready: {sub_path}")
    print("Upload to DrivenData, then: python -c \"from run_manager import RunManager; "
          "RunManager().update_public_score('run_018', SCORE)\"")


if __name__ == "__main__":
    main()
