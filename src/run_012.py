#!/usr/bin/env python3
"""
run_012: unweighted embedding network → full-260k features → LGBM.

The only change vs run_011: the entity embedding network is retrained without
inverse-frequency class weights.  Micro F1 == accuracy, so weighting the loss
penalises grade-2 (57% of rows) predictions with no leaderboard benefit.
Unweighted training lets the network optimise grade-2 separability directly,
producing better embedding representations for LGBM.

New embedding weights saved to models/embedding_unweighted/ — original
models/embedding/ weights are untouched.

Run from project root:
    python src/run_012.py [--skip-retrain]
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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embed import (
    BATCH_SIZE,
    EMBED_COLS,
    EMBED_DIR,
    EPOCHS,
    GEO_RATE_ALPHA,
    GEO_RATE_CV_FOLDS,
    GEO_RATE_COLS,
    EMBED_PREFIX,
    GRADES,
    NUMERIC_COLS,
    PATIENCE,
    RANDOM_STATE,
    NepalEmbeddingNet,
    add_interaction_features,
    apply_age_sentinel,
    assemble_features,
    assign_geo_rates,
    binary_columns,
    extract_embedding_features,
    fit_label_encoders,
    get_device,
    global_grade_rates,
    lightgbm_train_split,
    load_raw_data,
    scale_numerics,
    set_seed,
    smoothed_grade_rates_by_geo,
    train_embedding_network,
)
from features_full import compute_full_cv_geo_rates, encode_full_train
from run_manager import PREPROCESSING_DIR, PROCESSED_DIR, ROOT as RM_ROOT

print = partial(print, flush=True)

EMBED_UW_DIR = ROOT / "models" / "embedding_unweighted"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-retrain", action="store_true")
    args = p.parse_args()
    t0 = time.time()

    set_seed()
    device = get_device()
    EMBED_UW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load raw data ──────────────────────────────────────────────────────
    print("── Loading raw data ──")
    train_full, test_raw = load_raw_data()
    train_full, test_raw = apply_age_sentinel(train_full, test_raw)

    # Embedding network trained on the same 208k split as before (clean holdout)
    train_df = lightgbm_train_split(train_full).reset_index(drop=True)
    print(f"  Embedding train split: {len(train_df):,} rows")

    # ── 2. Encode & scale for embedding training ──────────────────────────────
    print("\n── Encoding categoricals ──")
    enc_train_208, enc_test, encoders, n_emb = fit_label_encoders(train_df, test_raw)
    joblib.dump(encoders, EMBED_UW_DIR / "label_encoders.pkl")

    bin_cols   = binary_columns(train_df)
    train_bin  = train_df[bin_cols].values.astype(np.float32)
    test_bin   = test_raw[bin_cols].values.astype(np.float32)
    train_num_208, test_num, _ = scale_numerics(train_df, test_raw)

    # ── 3. Train embedding network WITHOUT class weights ──────────────────────
    ckpt = EMBED_UW_DIR / "embedding_network.pt"
    model = NepalEmbeddingNet(n_emb, EMBED_COLS, train_num_208.shape[1], train_bin.shape[1]).to(device)

    if ckpt.exists():
        print(f"\n── Loading existing unweighted checkpoint {ckpt} ──")
        try:
            state = torch.load(ckpt, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        train_info = {"best_epoch": "saved", "best_val_micro_f1": "saved"}
    else:
        print("\n── Training embedding network (unweighted) ──")
        y_208 = train_df["damage_grade"].values
        model, train_info = train_embedding_network(
            enc_train_208, train_num_208, train_bin, y_208, n_emb, device,
            use_class_weights=False,
            save_dir=EMBED_UW_DIR,   # saves best epoch here, never touches models/embedding/
        )
        # best checkpoint already saved inside train_embedding_network via save_dir
        print(f"  Saved unweighted weights → {ckpt}")

    model.eval()

    # ── 4. Build full-260k feature matrix ─────────────────────────────────────
    print("\n── Building features on full 260k rows ──")
    train_full_r = train_full.reset_index(drop=True)
    test_r       = test_raw.reset_index(drop=True)

    # Encode full 260k (unknown → index 0)
    enc_train_full, enc_test_full, _ = encode_full_train(train_full_r, test_r)
    # Note: encode_full_train uses EMBED_DIR encoders; we saved identical encoders
    # to EMBED_UW_DIR above (same 208k split, same label encoders).

    # Scale full numerics
    scaler = joblib.load(PREPROCESSING_DIR / "scaler.pkl")
    train_num_full = scaler.transform(train_full_r[NUMERIC_COLS])
    test_num_full  = scaler.transform(test_r[NUMERIC_COLS])

    train_bin_full = train_full_r[bin_cols].values.astype(np.float32)
    test_bin_full  = test_r[bin_cols].values.astype(np.float32)

    print("  Extracting embeddings...")
    emb_train, emb_test = extract_embedding_features(model, enc_train_full, enc_test_full)

    print("  Computing CV geo rates on 260k...")
    global_rates = global_grade_rates(train_full_r)
    geo_train    = compute_full_cv_geo_rates(train_full_r, global_rates)
    rate_maps    = {
        gc: smoothed_grade_rates_by_geo(train_full_r, gc, global_rates)
        for gc in GEO_RATE_COLS
    }
    geo_test = assign_geo_rates(test_r, rate_maps, global_rates)

    print("  Computing interactions...")
    interact_train, interact_test = add_interaction_features(
        train_full_r, test_r, geo_train, geo_test
    )

    print("  Assembling matrices...")
    X_train_full, X_test_full = assemble_features(
        emb_train, emb_test,
        geo_train, geo_test,
        train_full_r, test_r,
        train_num_full, test_num_full,
        interact_train, interact_test,
    )
    print(f"  X_train: {X_train_full.shape}  X_test: {X_test_full.shape}")

    assert not X_train_full.isna().any().any(), "NaN in X_train!"
    assert not X_test_full.isna().any().any(),  "NaN in X_test!"

    out_train = PROCESSED_DIR / "X_train_run012.csv"
    out_test  = PROCESSED_DIR / "X_test_run012.csv"
    out_y     = PROCESSED_DIR / "y_train_full.csv"   # reuse existing — same labels

    X_train_full.to_csv(out_train, index=False)
    X_test_full.to_csv(out_test,   index=False)
    if not out_y.exists():
        train_full_r["damage_grade"].to_frame().to_csv(out_y, index=False)

    print(f"\n  Saved {out_train.name}, {out_test.name}")
    print(f"  Feature build: {time.time() - t0:.1f}s")

    if args.skip_retrain:
        print("\n--skip-retrain set; done.")
        return

    # ── 5. Retrain LGBM → run_012 ─────────────────────────────────────────────
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
            "Unweighted embedding network + full 260k — remove class-weight micro-F1 tax",
        ],
        check=True,
        cwd=str(ROOT),
    )
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
