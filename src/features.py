"""
Feature loader for run_003-style embedded feature matrices.

These files are produced by src/embed.py. Run embed.py first if they are missing:
    python src/embed.py --skip-train

Functions
---------
load_run003_features()
    Returns (X_train, y_train, X_test, building_ids) ready for ensemble training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from preprocess import DATA_DIR
from run_manager import PROCESSED_DIR

GRADES = [1, 2, 3]

_REQUIRED_FILES = [
    PROCESSED_DIR / "X_train_embedded.csv",
    PROCESSED_DIR / "X_test_embedded.csv",
    PROCESSED_DIR / "y_train_multiclass.csv",
]


def load_run003_features() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Load the run_003-style embedded feature matrices.

    Returns
    -------
    X_train      : pd.DataFrame   shape (n_train, n_features)
    y_train      : np.ndarray     damage_grade in {1,2,3}, shape (n_train,)
    X_test       : pd.DataFrame   shape (n_test, n_features)
    building_ids : np.ndarray     test building IDs, shape (n_test,)

    Raises
    ------
    FileNotFoundError
        If embed.py or preprocess.py have not been run yet.
    """
    missing = [p for p in _REQUIRED_FILES if not p.exists()]
    if missing:
        names = [str(p.relative_to(PROCESSED_DIR.parent.parent)) for p in missing]
        raise FileNotFoundError(
            f"Missing feature files: {', '.join(names)}\n"
            "  1. Run `python src/preprocess.py` to create y_train_multiclass.csv\n"
            "  2. Run `python src/embed.py --skip-train` to generate embedded matrices"
        )

    test_ids_path = DATA_DIR / "test_values.csv"
    if not test_ids_path.exists():
        raise FileNotFoundError(
            f"Missing {test_ids_path}. "
            "Place the driven_data CSVs under data/driven_data/."
        )

    X_train = pd.read_csv(PROCESSED_DIR / "X_train_embedded.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test_embedded.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train_multiclass.csv")["damage_grade"].to_numpy()
    building_ids = pd.read_csv(test_ids_path)["building_id"].to_numpy()

    return X_train, y_train, X_test, building_ids
