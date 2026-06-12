"""Feature extraction for run_024 — Shoumik replication."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset

from embed import NUMERIC_COLS, apply_age_sentinel, binary_columns, load_raw_data
from run_024_model import (
    GEO_LATENT_DR,
    GEO_LATENT_ROLLUP,
    RollUpGeo3AutoEncoder,
    RollUpGeo3Encoder,
    ShoumikDRAutoEncoder,
    ShoumikDREncoder,
)

GEO_COLS = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]
CAT_COLS = [
    "foundation_type", "ground_floor_type", "land_surface_condition",
    "legal_ownership_status", "other_floor_type", "plan_configuration",
    "position", "roof_type",
]

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "shoumik_run024"
DR_CKPT = MODEL_DIR / "geo_dr_encoder.pt"
ROLLUP_CKPT = MODEL_DIR / "geo3_rollup_encoder.pt"
ENCODER_PATH = MODEL_DIR / "geo_label_encoders.pkl"
DR_TRAIN = MODEL_DIR / "geo_dr_train.npy"
DR_TEST = MODEL_DIR / "geo_dr_test.npy"
RU_TRAIN = MODEL_DIR / "geo_rollup_train.npy"
RU_TEST = MODEL_DIR / "geo_rollup_test.npy"
GEO_TRAIN = MODEL_DIR / "geo_idx_train.npy"
GEO_TEST = MODEL_DIR / "geo_idx_test.npy"


class ThresholdReplacer:
    """Rare geo categories (count <= threshold) → unk_value."""

    def __init__(self, threshold: int = 3, unk_value: int = -1):
        self.threshold = threshold
        self.unk_value = unk_value
        self.keep_: dict[str, set] = {}

    def fit(self, df: pd.DataFrame, cols: list[str]) -> "ThresholdReplacer":
        self.keep_ = {}
        for c in cols:
            vc = df[c].value_counts()
            self.keep_[c] = set(vc[vc > self.threshold].index.tolist())
        return self

    def transform(self, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
        out = df[cols].copy()
        for c in cols:
            keep = self.keep_.get(c, set())
            out[c] = out[c].where(out[c].isin(keep), self.unk_value)
        return out[cols].to_numpy(dtype=np.float32)


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = load_raw_data()
    train, test = apply_age_sentinel(train, test)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def fit_geo_label_encoders(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, LabelEncoder]:
    encoders: dict[str, LabelEncoder] = {}
    combined = pd.concat([train, test], axis=0)
    for col in GEO_COLS:
        le = LabelEncoder()
        le.fit(combined[col].astype(str))
        encoders[col] = le
    return encoders


def transform_geo(encoders: dict[str, LabelEncoder], df: pd.DataFrame) -> np.ndarray:
    parts = []
    for col in GEO_COLS:
        le = encoders[col]
        s = df[col].astype(str)
        out = np.full(len(df), -1, dtype=np.int64)
        known = s.isin(le.classes_)
        if known.any():
            out[known] = le.transform(s[known])
        parts.append(out)
    return np.column_stack(parts)


def _inv_freq_weights(indices: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(indices, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = 1.0 / counts
    return w / w.sum() * n_classes


@torch.no_grad()
def _encode_batches(model: torch.nn.Module, x: np.ndarray, device: torch.device, batch: int = 4096) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    for i in range(0, len(x), batch):
        t = torch.from_numpy(x[i:i + batch]).long().to(device)
        parts.append(model(t).cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float32)


def train_geo_models(
    geo_idx: np.ndarray,
    device: torch.device,
    *,
    dr_epochs: int = 10,
    rollup_epochs: int = 10,
    batch_size: int = 128,
    force: bool = False,
) -> tuple[ShoumikDREncoder, RollUpGeo3Encoder]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    n1 = int(geo_idx[:, 0].max()) + 2
    n2 = int(geo_idx[:, 1].max()) + 2
    n3 = int(geo_idx[:, 2].max()) + 2

    dr_enc = ShoumikDRAutoEncoder(n1, n2, n3, GEO_LATENT_DR).to(device)
    rollup_enc = RollUpGeo3AutoEncoder(n1, n2, n3, GEO_LATENT_ROLLUP).to(device)

    if DR_CKPT.exists() and ROLLUP_CKPT.exists() and not force:
        dr_enc.encoder.load_state_dict(torch.load(DR_CKPT, map_location=device, weights_only=True))
        rollup_enc.encoder.load_state_dict(torch.load(ROLLUP_CKPT, map_location=device, weights_only=True))
        return dr_enc.encoder, rollup_enc.encoder

    # ── Geo DR AE ───────────────────────────────────────────────────────────
    ds = TensorDataset(torch.from_numpy(geo_idx).long(), torch.from_numpy(geo_idx).long())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    w1 = torch.tensor(_inv_freq_weights(geo_idx[:, 0], n1), dtype=torch.float32, device=device)
    w2 = torch.tensor(_inv_freq_weights(geo_idx[:, 1], n2), dtype=torch.float32, device=device)
    w3 = torch.tensor(_inv_freq_weights(geo_idx[:, 2], n3), dtype=torch.float32, device=device)
    c1 = torch.nn.CrossEntropyLoss(weight=w1)
    c2 = torch.nn.CrossEntropyLoss(weight=w2)
    c3 = torch.nn.CrossEntropyLoss(weight=w3)
    opt = torch.optim.Adam(dr_enc.parameters(), lr=1e-3)

    for epoch in range(1, dr_epochs + 1):
        dr_enc.train()
        total = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            _, p1, p2, p3 = dr_enc(xb)
            loss = (c1(p1, yb[:, 0]) + c2(p2, yb[:, 1]) + c3(p3, yb[:, 2])) / 3.0
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        if epoch % 2 == 0 or epoch == 1:
            print(f"    DR AE epoch {epoch}: loss={total/len(dl):.4f}")

    torch.save(dr_enc.encoder.state_dict(), DR_CKPT)

    # ── Geo3 rollup ─────────────────────────────────────────────────────────
    geo3 = geo_idx[:, 2:3]
    targets = geo_idx[:, :2]
    ds_r = TensorDataset(torch.from_numpy(geo3).long(), torch.from_numpy(targets).long())
    dl_r = DataLoader(ds_r, batch_size=batch_size, shuffle=True)
    c1r = torch.nn.CrossEntropyLoss(weight=w1)
    c2r = torch.nn.CrossEntropyLoss(weight=w2)
    opt_r = torch.optim.Adam(rollup_enc.parameters(), lr=1e-3)

    for epoch in range(1, rollup_epochs + 1):
        rollup_enc.train()
        total = 0.0
        for xb, yb in dl_r:
            xb, yb = xb.to(device), yb.to(device)
            _, p1, p2 = rollup_enc(xb)
            loss = (c1r(p1, yb[:, 0]) + c2r(p2, yb[:, 1])) / 2.0
            opt_r.zero_grad()
            loss.backward()
            opt_r.step()
            total += loss.item()
        if epoch % 2 == 0 or epoch == 1:
            print(f"    Rollup epoch {epoch}: loss={total/len(dl_r):.4f}")

    torch.save(rollup_enc.encoder.state_dict(), ROLLUP_CKPT)
    return dr_enc.encoder, rollup_enc.encoder


class ShoumikFeatureBuilder:
    """Build XGB feature matrix from precomputed geo latents (no torch at inference)."""

    def __init__(self):
        self.ohe: OneHotEncoder | None = None
        self.bin_cols: list[str] = []

    def fit(self, train: pd.DataFrame) -> "ShoumikFeatureBuilder":
        self.bin_cols = binary_columns(train)
        self.ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
        self.ohe.fit(train[CAT_COLS].astype(str))
        return self

    def transform(
        self,
        df: pd.DataFrame,
        geo_idx: np.ndarray,
        geo_dr: np.ndarray,
        geo_ru: np.ndarray,
    ) -> np.ndarray:
        assert self.ohe is not None
        cat_ohe = self.ohe.transform(df[CAT_COLS].astype(str))
        geo_raw = geo_idx.astype(np.float32)
        numeric = df[NUMERIC_COLS].to_numpy(dtype=np.float32)
        binary = df[self.bin_cols].to_numpy(dtype=np.float32)
        return np.hstack([cat_ohe, geo_raw, geo_dr, geo_ru, numeric, binary]).astype(np.float32)


def load_geo_latents() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.load(GEO_TRAIN),
        np.load(GEO_TEST),
        np.load(DR_TRAIN),
        np.load(DR_TEST),
        np.load(RU_TRAIN),
        np.load(RU_TEST),
    )


def geo_latents_ready() -> bool:
    return all(p.exists() for p in (DR_TRAIN, DR_TEST, RU_TRAIN, RU_TEST, GEO_TRAIN, GEO_TEST))


class SolutionFeatureBuilder:
    """Solution.ipynb layout: OHE(cats+geo1), DR16, threshold geo, passthrough."""

    def __init__(self):
        self.ohe: OneHotEncoder | None = None
        self.threshold = ThresholdReplacer(threshold=3, unk_value=-1)
        self.bin_cols: list[str] = []
        self.dr_enc: torch.nn.Module | None = None
        self.device = torch.device("cpu")

    def set_geo_encoder(self, dr_enc: torch.nn.Module, device: torch.device):
        self.dr_enc = dr_enc
        self.device = device

    def fit(self, train: pd.DataFrame) -> "SolutionFeatureBuilder":
        self.bin_cols = binary_columns(train)
        cat_plus_geo1 = CAT_COLS + ["geo_level_1_id"]
        self.ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
        self.ohe.fit(train[cat_plus_geo1].astype(str))
        self.threshold.fit(train, GEO_COLS)
        return self

    def transform(self, df: pd.DataFrame, geo_idx: np.ndarray) -> np.ndarray:
        assert self.ohe is not None and self.dr_enc is not None
        cat_ohe = self.ohe.transform(df[CAT_COLS + ["geo_level_1_id"]].astype(str))
        geo_dr = _encode_batches(self.dr_enc, geo_idx, self.device)
        geo_thr = self.threshold.transform(df, GEO_COLS)
        numeric = df[NUMERIC_COLS].to_numpy(dtype=np.float32)
        binary = df[self.bin_cols].to_numpy(dtype=np.float32)
        return np.hstack([cat_ohe, geo_dr, geo_thr, numeric, binary]).astype(np.float32)


def load_geo_encoders(
    geo_idx: np.ndarray,
    device: torch.device,
) -> tuple[ShoumikDREncoder, RollUpGeo3Encoder]:
    n1 = int(geo_idx[:, 0].max()) + 2
    n2 = int(geo_idx[:, 1].max()) + 2
    n3 = int(geo_idx[:, 2].max()) + 2
    dr_enc = ShoumikDREncoder(n1, n2, n3, GEO_LATENT_DR).to(device)
    rollup_enc = RollUpGeo3Encoder(n_geo3=n3, latent_dim=GEO_LATENT_ROLLUP).to(device)
    dr_enc.load_state_dict(torch.load(DR_CKPT, map_location=device, weights_only=True))
    rollup_enc.load_state_dict(torch.load(ROLLUP_CKPT, map_location=device, weights_only=True))
    dr_enc.eval()
    rollup_enc.eval()
    return dr_enc, rollup_enc


def prepare_geo(
    train: pd.DataFrame,
    test: pd.DataFrame,
    device: torch.device,
    *,
    dr_epochs: int,
    rollup_epochs: int,
    force: bool,
) -> tuple[dict[str, LabelEncoder], np.ndarray, np.ndarray, ShoumikDREncoder, RollUpGeo3Encoder]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    encoders = fit_geo_label_encoders(train, test)
    joblib.dump(encoders, ENCODER_PATH)
    geo_train = transform_geo(encoders, train)
    geo_test = transform_geo(encoders, test)
    geo_all = np.vstack([geo_train, geo_test])
    train_geo_models(geo_all, device, dr_epochs=dr_epochs, rollup_epochs=rollup_epochs, force=force)
    dr_enc, rollup_enc = load_geo_encoders(geo_all, device)
    return encoders, geo_train, geo_test, dr_enc, rollup_enc
