#!/usr/bin/env python3
"""
Build run_003-style feature matrix on ALL 260k training rows.

embed.py trains on 80% (208k) to avoid leaking test labels during embedding
training.  The embedding weights are frozen lookup tables — we can apply them
to the remaining 20% without retraining.  This script:

  1. Loads the frozen embedding weights from models/embedding/
  2. Applies them to all 260,601 training rows
  3. Recomputes Laplace-smoothed CV geo rates with 5-fold CV on the full set
  4. Recomputes interaction features
  5. Saves X_train_full.csv  (260,601 × 191)  and y_train_full.csv
     X_test_embedded.csv is unchanged (already built from full train weights)

Run from project root:
    python src/features_full.py [--skip-retrain]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embed import (
    EMBED_COLS,
    EMBED_PREFIX,
    GEO_RATE_COLS,
    GEO_RATE_ALPHA,
    GEO_RATE_CV_FOLDS,
    GRADES,
    NUMERIC_COLS,
    RANDOM_STATE,
    apply_age_sentinel,
    binary_columns,
    extract_embedding_features,
    global_grade_rates,
    load_raw_data,
    smoothed_grade_rates_by_geo,
    assign_geo_rates,
    add_interaction_features,
    assemble_features,
    NepalEmbeddingNet,
)
from run_manager import PROCESSED_DIR, PREPROCESSING_DIR

print = partial(print, flush=True)

EMBED_DIR = ROOT / "models" / "embedding"


def load_full_train() -> tuple[pd.DataFrame, pd.DataFrame]:
    """All 260,601 training rows + test, with age sentinel handled."""
    train_full, test_raw = load_raw_data()
    train_full, test_raw = apply_age_sentinel(train_full, test_raw)
    return train_full.reset_index(drop=True), test_raw.reset_index(drop=True)


def encode_full_train(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[dict, dict, dict[str, int]]:
    """Apply saved label encoders to all rows (index 0 = unknown)."""
    encoders = joblib.load(EMBED_DIR / "label_encoders.pkl")
    from sklearn.preprocessing import LabelEncoder

    enc_train: dict = {}
    enc_test:  dict = {}
    n_emb:     dict = {}

    for col in EMBED_COLS:
        le: LabelEncoder = encoders[col]
        tr_str = train[col].astype(str).values
        te_str = test[col].astype(str).values

        # Unknown categories (in holdout rows not seen during embed training) → index 0
        tr = np.zeros(len(train), dtype=np.int64)
        tr_known = np.isin(tr_str, le.classes_)
        if tr_known.any():
            tr[tr_known] = le.transform(tr_str[tr_known]) + 1

        te = np.zeros(len(test), dtype=np.int64)
        te_known = np.isin(te_str, le.classes_)
        if te_known.any():
            te[te_known] = le.transform(te_str[te_known]) + 1

        enc_train[col] = tr
        enc_test[col]  = te
        n_emb[col]     = len(le.classes_) + 1

    return enc_train, enc_test, n_emb


def scale_full_numerics(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the saved scaler (fit on 208k train split) to all rows."""
    scaler = joblib.load(PREPROCESSING_DIR / "scaler.pkl")
    return (
        scaler.transform(train[NUMERIC_COLS]),
        scaler.transform(test[NUMERIC_COLS]),
    )


def compute_full_cv_geo_rates(
    train: pd.DataFrame, global_rates: dict[int, float]
) -> pd.DataFrame:
    """5-fold CV Laplace geo rates on all 260k rows — no leakage."""
    geo_cols = [
        f"{EMBED_PREFIX[g]}_p_grade{grade}"
        for g in GEO_RATE_COLS
        for grade in GRADES
    ]
    out = pd.DataFrame(index=train.index, columns=geo_cols, dtype=float)
    for col in geo_cols:
        g = int(col.rsplit("grade", maxsplit=1)[-1])
        out[col] = global_rates[g]

    skf = StratifiedKFold(
        n_splits=GEO_RATE_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    for _, (tr_idx, val_idx) in enumerate(
        skf.split(train, train["damage_grade"])
    ):
        tr_fold = train.iloc[tr_idx]
        val_df  = train.iloc[val_idx]
        for geo_col in GEO_RATE_COLS:
            maps   = smoothed_grade_rates_by_geo(tr_fold, geo_col, global_rates)
            prefix = EMBED_PREFIX[geo_col]
            for g in GRADES:
                col = f"{prefix}_p_grade{g}"
                out.loc[train.index[val_idx], col] = (
                    val_df[geo_col].map(maps[g]).fillna(global_rates[g]).values
                )

    print("  CV geo rates on 260k — no leakage.")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-retrain", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0   = time.time()

    print("── Loading full training data (260k rows) ──")
    train, test = load_full_train()
    print(f"  train: {len(train):,}  test: {len(test):,}")

    print("\n── Encoding categoricals with saved label encoders ──")
    enc_train, enc_test, n_emb = encode_full_train(train, test)

    print("\n── Scaling numerics with saved scaler ──")
    train_num, test_num = scale_full_numerics(train, test)

    bin_cols  = binary_columns(train)
    train_bin = train[bin_cols].values.astype(np.float32)
    test_bin  = test[bin_cols].values.astype(np.float32)

    print("\n── Loading frozen embedding weights ──")
    import torch
    device = torch.device("cpu")
    model  = NepalEmbeddingNet(n_emb, EMBED_COLS, train_num.shape[1], train_bin.shape[1])
    ckpt   = EMBED_DIR / "embedding_network.pt"
    try:
        state = torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"  Loaded {ckpt.name}")

    print("\n── Extracting embedding features ──")
    emb_train, emb_test = extract_embedding_features(model, enc_train, enc_test)
    print(f"  emb_train: {emb_train.shape}")

    print("\n── Computing CV geo rates on full 260k ──")
    global_rates = global_grade_rates(train)
    geo_train    = compute_full_cv_geo_rates(train, global_rates)

    # Test rates: smoothed from full train (same as embed.py)
    rate_maps = {
        geo_col: smoothed_grade_rates_by_geo(train, geo_col, global_rates)
        for geo_col in GEO_RATE_COLS
    }
    geo_test = assign_geo_rates(test, rate_maps, global_rates)

    print("\n── Computing interaction features ──")
    interact_train, interact_test = add_interaction_features(
        train, test, geo_train, geo_test
    )

    print("\n── Assembling final matrices ──")
    X_train_full, X_test_full = assemble_features(
        emb_train, emb_test,
        geo_train, geo_test,
        train, test,
        train_num, test_num,
        interact_train, interact_test,
    )
    print(f"  X_train_full: {X_train_full.shape}")
    print(f"  X_test_full:  {X_test_full.shape}")

    assert not X_train_full.isna().any().any(), "NaN in X_train_full!"
    assert not X_test_full.isna().any().any(),  "NaN in X_test_full!"

    out_train = PROCESSED_DIR / "X_train_full.csv"
    out_test  = PROCESSED_DIR / "X_test_full.csv"
    out_y     = PROCESSED_DIR / "y_train_full.csv"

    X_train_full.to_csv(out_train, index=False)
    # X_test_full is identical to X_test_embedded.csv (test rows unchanged)
    # but save it under a clear name for retrain.py
    X_test_full.to_csv(out_test, index=False)
    train["damage_grade"].to_frame().to_csv(out_y, index=False)

    print(f"\n  Saved {out_train.name}, {out_test.name}, {out_y.name}")
    print(f"  Feature engineering: {time.time() - t0:.1f}s")

    if args.skip_retrain:
        print("\n--skip-retrain set; done.")
        return

    from run_manager import RunManager
    run_id = RunManager().get_next_run_id()
    print(f"\n── Triggering retrain.py ({run_id}) ──")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "src" / "retrain.py"),
            "--run-id",     run_id,
            "--x-train",    str(out_train),
            "--x-test",     str(out_test),
            "--y-train",    str(out_y),
            "--description",
            "Full 260k training rows — same run_003 features, no holdout",
        ],
        check=True,
        cwd=str(ROOT),
    )
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
