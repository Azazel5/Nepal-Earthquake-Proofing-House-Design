"""
Run-based artifact management for Nepal earthquake damage prediction experiments.

Each training run lives under runs/run_NNN/ with metadata, model, submission, and CV scores.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"
PREPROCESSING_DIR = ROOT / "models" / "preprocessing"
OPTUNA_DIR = ROOT / "outputs" / "optuna"
PROCESSED_DIR = ROOT / "data" / "processed"
BEST_RUN_PATH = ROOT / "outputs" / "best_run.json"

RUN_ID_PATTERN = re.compile(r"^run_(\d+)$")


class RunManager:
    """Create, update, and compare experiment runs under runs/."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or ROOT
        self.runs_dir = self.root / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.best_run_path = self.root / "outputs" / "best_run.json"

    def get_next_run_id(self) -> str:
        """Read runs/ and return the next run_NNN id (e.g. run_003)."""
        max_num = 0
        if self.runs_dir.exists():
            for path in self.runs_dir.iterdir():
                match = RUN_ID_PATTERN.match(path.name)
                if match:
                    max_num = max(max_num, int(match.group(1)))
        return f"run_{max_num + 1:03d}"

    def run_path(self, run_id: str) -> Path:
        """Absolute path to a run folder."""
        return self.runs_dir / run_id

    def create_run(
        self,
        description: str,
        model_type: str,
        feature_set: str,
        params: dict[str, Any],
        *,
        run_id: str | None = None,
        objective: str = "multiclass",
        n_features: int | None = None,
        cv_folds: int | None = None,
        cv_metric: str | None = None,
        notes: str = "",
    ) -> tuple[str, Path]:
        """Create runs/run_NNN/, write metadata.json and params.json."""
        rid = run_id or self.get_next_run_id()
        path = self.run_path(rid)
        path.mkdir(parents=True, exist_ok=False)

        metadata: dict[str, Any] = {
            "run_id": rid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "description": description,
            "model_type": model_type,
            "objective": objective,
            "feature_set": feature_set,
            "n_features": n_features,
            "params": params,
            "cv_folds": cv_folds,
            "cv_metric": cv_metric,
            "cv_mean": None,
            "cv_std": None,
            "cv_scores_per_fold": None,
            "public_leaderboard_score": None,
            "submitted": False,
            "submission_date": None,
            "notes": notes,
        }
        self._write_json(path / "metadata.json", metadata)
        self._write_json(path / "params.json", params)
        return rid, path

    def save_model(self, run_id: str, model: Any) -> Path:
        """Persist trained model to runs/run_NNN/model.pkl."""
        out = self.run_path(run_id) / "model.pkl"
        joblib.dump(model, out)
        return out

    def save_submission(self, run_id: str, df: pd.DataFrame) -> Path:
        """Save building_id + damage_grade submission CSV."""
        out = self.run_path(run_id) / "submission.csv"
        df.to_csv(out, index=False)
        return out

    def save_feature_importance(self, run_id: str, df: pd.DataFrame) -> Path:
        """Save feature importance table for the run."""
        out = self.run_path(run_id) / "feature_importance.csv"
        df.to_csv(out, index=False)
        return out

    def save_cv_scores(
        self,
        run_id: str,
        scores_per_fold: list[float],
        mean: float,
        std: float,
    ) -> Path:
        """Write per-fold CV scores and update metadata.json."""
        payload = {
            "scores_per_fold": scores_per_fold,
            "mean": mean,
            "std": std,
        }
        path = self.run_path(run_id)
        self._write_json(path / "cv_scores.json", payload)

        meta_path = path / "metadata.json"
        metadata = self._read_json(meta_path)
        metadata["cv_mean"] = mean
        metadata["cv_std"] = std
        metadata["cv_scores_per_fold"] = scores_per_fold
        self._write_json(meta_path, metadata)
        return path / "cv_scores.json"

    def update_public_score(
        self,
        run_id: str,
        score: float,
        *,
        submitted: bool = True,
        submission_date: str | None = None,
    ) -> None:
        """Record DrivenData public leaderboard score on a run."""
        meta_path = self.run_path(run_id) / "metadata.json"
        metadata = self._read_json(meta_path)
        metadata["public_leaderboard_score"] = score
        metadata["submitted"] = submitted
        metadata["submission_date"] = submission_date or datetime.now(timezone.utc).date().isoformat()
        self._write_json(meta_path, metadata)

    def load_metadata(self, run_id: str) -> dict[str, Any]:
        """Load metadata.json for a run."""
        return self._read_json(self.run_path(run_id) / "metadata.json")

    def list_runs(self) -> list[str]:
        """Return sorted run ids (run_001, run_002, ...)."""
        if not self.runs_dir.exists():
            return []
        ids = [p.name for p in self.runs_dir.iterdir() if RUN_ID_PATTERN.match(p.name)]
        return sorted(ids)

    def update_best_run(self, recommended_run: str | None = None) -> dict[str, Any]:
        """Compare all runs and write outputs/best_run.json."""
        runs = self.list_runs()
        best_cv_run: str | None = None
        best_cv_score = -1.0
        best_public_run: str | None = None
        best_public_score = -1.0

        for rid in runs:
            meta = self.load_metadata(rid)
            cv = meta.get("cv_mean")
            if cv is not None and cv > best_cv_score:
                best_cv_score = cv
                best_cv_run = rid
            pub = meta.get("public_leaderboard_score")
            if pub is not None and pub > best_public_score:
                best_public_score = pub
                best_public_run = rid

        if recommended_run:
            recommendation = f"submit {recommended_run}"
        elif best_cv_run is None:
            recommendation = "no runs with CV scores yet"
        elif best_public_run is None:
            recommendation = f"submit {best_cv_run}"
        elif best_cv_run == best_public_run:
            recommendation = f"submit {best_cv_run}"
        else:
            recommendation = (
                f"submit {best_cv_run} (best CV); public leader is {best_public_run}"
            )

        payload = {
            "best_cv_run": best_cv_run,
            "best_cv_score": best_cv_score if best_cv_run else None,
            "best_public_run": best_public_run,
            "best_public_score": best_public_score if best_public_run else None,
            "recommended_run": recommended_run,
            "recommendation": recommendation,
        }
        self.best_run_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_json(self.best_run_path, payload)
        return payload

    def print_run_summary(self, run_id: str) -> None:
        """Print all metadata fields for one run."""
        meta = self.load_metadata(run_id)
        print(f"\n{'=' * 60}")
        print(f"Run: {run_id}")
        print(f"{'=' * 60}")
        for key, value in meta.items():
            if key == "params":
                print(f"  {key}: <see params.json>")
            else:
                print(f"  {key}: {value}")

    def print_all_runs(self) -> None:
        """Print comparison table of all runs sorted by CV mean."""
        rows: list[dict[str, Any]] = []
        for rid in self.list_runs():
            meta = self.load_metadata(rid)
            rows.append(
                {
                    "run_id": rid,
                    "description": meta.get("description", "")[:32],
                    "cv_mean": meta.get("cv_mean"),
                    "public_score": meta.get("public_leaderboard_score"),
                    "submitted": meta.get("submitted", False),
                }
            )
        rows.sort(
            key=lambda r: (r["cv_mean"] is None, -(r["cv_mean"] or 0)),
        )

        print("\n┌─────────┬──────────────────────────────────┬──────────┬──────────────┬───────────┐")
        print("│ Run     │ Description                      │ CV F1    │ Public Score │ Submitted │")
        print("├─────────┼──────────────────────────────────┼──────────┼──────────────┼───────────┤")
        for r in rows:
            cv = f"{r['cv_mean']:.4f}" if r["cv_mean"] is not None else "  —   "
            pub = f"{r['public_score']:.4f}" if r["public_score"] is not None else "pending     "
            sub = "Yes" if r["submitted"] else "No"
            desc = (r["description"][:32] + "..") if len(r["description"]) > 32 else r["description"].ljust(32)
            print(f"│ {r['run_id']:<7} │ {desc:<32} │ {cv:>8} │ {pub:>12} │ {sub:<9} │")
        print("└─────────┴──────────────────────────────────┴──────────┴──────────────┴───────────┘")

        best = self._read_json(self.best_run_path) if self.best_run_path.exists() else self.update_best_run()
        if best.get("best_cv_run"):
            print(f"\nBest CV run:     {best['best_cv_run']} ({best['best_cv_score']:.4f})")
        if best.get("best_public_run"):
            print(f"Best public run: {best['best_public_run']} ({best['best_public_score']:.4f})")
        rec_run = best.get("recommended_run") or best.get("best_cv_run")
        if rec_run:
            print(f"Ready to submit: runs/{rec_run}/submission.csv")

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
