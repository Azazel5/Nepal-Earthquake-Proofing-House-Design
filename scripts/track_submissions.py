#!/usr/bin/env python3
"""Record DrivenData public scores and print CV vs public gap table."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_manager import RunManager


def main() -> None:
    p = argparse.ArgumentParser(description="Track submission scores for experiment runs")
    p.add_argument("--run-id", action="append", dest="run_ids", default=[],
                   help="Run id to update, e.g. --run-id run_018 --score 0.7505")
    p.add_argument("--score", type=float, action="append", default=[],
                   help="Public score matching each --run-id (same order)")
    p.add_argument("--list", action="store_true", help="Print CV vs public table for all runs")
    args = p.parse_args()

    rm = RunManager()

    if args.run_ids:
        if len(args.score) != len(args.run_ids):
            sys.exit("Provide one --score per --run-id")
        for rid, score in zip(args.run_ids, args.score):
            rm.update_public_score(rid, score)
            print(f"Updated {rid}: public={score:.4f}")

    print("\n── CV vs Public Leaderboard ──")
    print(f"  {'Run':<10} {'CV':>8}  {'Public':>8}  {'Gap':>8}  Submitted")
    print("  " + "-" * 52)
    for rid in rm.list_runs():
        meta = rm.load_metadata(rid)
        cv = meta.get("cv_mean")
        pub = meta.get("public_leaderboard_score")
        if cv is None and pub is None:
            continue
        cv_s = f"{cv:.4f}" if cv is not None else "   —  "
        pub_s = f"{pub:.4f}" if pub is not None else "   —  "
        gap_s = f"{pub - cv:+.4f}" if cv is not None and pub is not None else "   —  "
        sub = "yes" if meta.get("submitted") else "no"
        print(f"  {rid:<10} {cv_s:>8}  {pub_s:>8}  {gap_s:>8}  {sub}")

    rm.update_best_run()
    print(f"\nUpdated {ROOT / 'outputs' / 'best_run.json'}")


if __name__ == "__main__":
    main()
