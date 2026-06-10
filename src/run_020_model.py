"""
Tabular autoencoder for run_020 — unsupervised representation learning.

Encodes categoricals via embeddings + numerics/binaries as floats,
projects to a latent vector for downstream LGBM or fine-tuning in run_021.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

GEO_COLS = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]
LOW_CARD_COLS = [
    "foundation_type", "roof_type", "land_surface_condition", "ground_floor_type",
    "other_floor_type", "position", "plan_configuration", "legal_ownership_status",
]
CAT_COLS = GEO_COLS + LOW_CARD_COLS

LATENT_DIM = 48
HIDDEN_DIMS = (256, 128)
NUMERIC_DIM = 6  # 5 numerics + age_unknown flag as float
CAT_EMBED_DIMS = {
    "geo_level_1_id": 8,
    "geo_level_2_id": 16,
    "geo_level_3_id": 16,
    "foundation_type": 4,
    "roof_type": 3,
    "land_surface_condition": 3,
    "ground_floor_type": 4,
    "other_floor_type": 3,
    "position": 3,
    "plan_configuration": 5,
    "legal_ownership_status": 3,
}


class TabularAE(nn.Module):
    def __init__(
        self,
        n_cat: dict[str, int],
        n_binary: int,
        latent_dim: int = LATENT_DIM,
        noise_std: float = 0.0,
    ):
        super().__init__()
        self.noise_std = noise_std
        self.cat_cols = CAT_COLS
        self.embeds = nn.ModuleDict({
            c: nn.Embedding(n_cat[c], CAT_EMBED_DIMS[c]) for c in CAT_COLS
        })
        embed_out = sum(CAT_EMBED_DIMS[c] for c in CAT_COLS)
        enc_in = embed_out + NUMERIC_DIM + n_binary
        layers: list[nn.Module] = []
        prev = enc_in
        for h in HIDDEN_DIMS:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.LayerNorm(h)]
            prev = h
        self.encoder_body = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.dec_in = nn.Linear(latent_dim, HIDDEN_DIMS[-1])
        self.decoder_body = nn.Sequential(
            nn.GELU(),
            nn.LayerNorm(HIDDEN_DIMS[-1]),
            nn.Linear(HIDDEN_DIMS[-1], HIDDEN_DIMS[0]),
            nn.GELU(),
            nn.LayerNorm(HIDDEN_DIMS[0]),
            nn.Linear(HIDDEN_DIMS[0], enc_in),
        )
        self.n_binary = n_binary
        self.n_cat = n_cat
        self._recon_slices: list[tuple[int, int]] = []
        off = 0
        for c in CAT_COLS:
            d = CAT_EMBED_DIMS[c]
            self._recon_slices.append((off, off + d))
            off += d
        self._num_slice = (off, off + NUMERIC_DIM)
        off += NUMERIC_DIM
        self._bin_slice = (off, off + n_binary)

    def encode(self, cat_idx: Tensor, numeric: Tensor, binary: Tensor) -> Tensor:
        parts = [self.embeds[c](cat_idx[:, i]) for i, c in enumerate(self.cat_cols)]
        x = torch.cat(parts + [numeric, binary], dim=1)
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        h = self.encoder_body(x)
        return self.fc_mu(h)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder_body(self.dec_in(z))

    def forward(self, cat_idx: Tensor, numeric: Tensor, binary: Tensor) -> tuple[Tensor, Tensor]:
        z = self.encode(cat_idx, numeric, binary)
        recon = self.decode(z)
        return z, recon

    def reconstruction_loss(
        self,
        cat_idx: Tensor,
        numeric: Tensor,
        binary: Tensor,
        recon: Tensor,
    ) -> Tensor:
        target_parts = [self.embeds[c](cat_idx[:, i]).detach() for i, c in enumerate(self.cat_cols)]
        target = torch.cat(target_parts + [numeric, binary], dim=1)
        return F.mse_loss(recon, target)


class TabularAEClassifier(nn.Module):
    """run_021: encoder + 3-class head."""

    def __init__(self, ae: TabularAE, n_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.ae = ae
        self.head = nn.Sequential(
            nn.LayerNorm(ae.fc_mu.out_features),
            nn.Dropout(dropout),
            nn.Linear(ae.fc_mu.out_features, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, cat_idx: Tensor, numeric: Tensor, binary: Tensor) -> Tensor:
        z = self.ae.encode(cat_idx, numeric, binary)
        return self.head(z)

    def freeze_encoder(self) -> None:
        for p in self.ae.parameters():
            p.requires_grad = False

    def unfreeze_encoder_top(self) -> None:
        for p in self.ae.encoder_body[-1].parameters():
            p.requires_grad = True
        for p in self.ae.fc_mu.parameters():
            p.requires_grad = True
