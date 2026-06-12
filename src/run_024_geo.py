#!/usr/bin/env python3
"""Train Shoumik geo encoders and export latent features (torch-only subprocess)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_024_features import (
    ENCODER_PATH,
    MODEL_DIR,
    fit_geo_label_encoders,
    load_frames,
    load_geo_encoders,
    train_geo_models,
    transform_geo,
    _encode_batches,
)

DR_TRAIN = MODEL_DIR / "geo_dr_train.npy"
DR_TEST = MODEL_DIR / "geo_dr_test.npy"
RU_TRAIN = MODEL_DIR / "geo_rollup_train.npy"
RU_TEST = MODEL_DIR / "geo_rollup_test.npy"
GEO_TRAIN = MODEL_DIR / "geo_idx_train.npy"
GEO_TEST = MODEL_DIR / "geo_idx_test.npy"


def export_latents(
    train,
    test,
    device: torch.device,
    *,
    dr_epochs: int,
    rollup_epochs: int,
    force: bool,
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if (
        not force
        and DR_TRAIN.exists()
        and DR_TEST.exists()
        and RU_TRAIN.exists()
        and RU_TEST.exists()
        and GEO_TRAIN.exists()
        and GEO_TEST.exists()
        and ENCODER_PATH.exists()
    ):
        print("Geo latents already exported — skipping")
        return

    encoders = fit_geo_label_encoders(train, test)
    joblib.dump(encoders, ENCODER_PATH)
    geo_train = transform_geo(encoders, train)
    geo_test = transform_geo(encoders, test)
    geo_all = np.vstack([geo_train, geo_test])

    train_geo_models(geo_all, device, dr_epochs=dr_epochs, rollup_epochs=rollup_epochs, force=force)
    dr_enc, rollup_enc = load_geo_encoders(geo_all, device)

    dr_train = _encode_batches(dr_enc, geo_train, device)
    dr_test = _encode_batches(dr_enc, geo_test, device)
    ru_train = _encode_batches(rollup_enc, geo_train[:, 2:3], device)
    ru_test = _encode_batches(rollup_enc, geo_test[:, 2:3], device)

    np.save(GEO_TRAIN, geo_train)
    np.save(GEO_TEST, geo_test)
    np.save(DR_TRAIN, dr_train)
    np.save(DR_TEST, dr_test)
    np.save(RU_TRAIN, ru_train)
    np.save(RU_TEST, ru_test)
    print(f"Saved geo latents to {MODEL_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, test = load_frames()
    dr_epochs = 2 if args.quick else 10
    rollup_epochs = 2 if args.quick else 10
    export_latents(train, test, device, dr_epochs=dr_epochs, rollup_epochs=rollup_epochs, force=args.force)


if __name__ == "__main__":
    main()
