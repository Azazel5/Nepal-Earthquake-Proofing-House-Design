"""
Categorical autoencoder for run_022 — encodes 11 categorical columns only.

Unsupervised objective: reconstruct per-field embedding vectors.
Optional Shoumik-style geo rollup auxiliary: predict geo1/geo2 from geo3 latent.
"""

from __future__ import annotations

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

CAT_LATENT_DIM = 32
HIDDEN_DIMS = (128, 64)
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

GEO1_IDX = 0
GEO2_IDX = 1
GEO3_IDX = 2


class CategoricalAE(nn.Module):
    def __init__(self, n_cat: dict[str, int], latent_dim: int = CAT_LATENT_DIM):
        super().__init__()
        self.cat_cols = CAT_COLS
        self.latent_dim = latent_dim
        self.embeds = nn.ModuleDict({
            c: nn.Embedding(n_cat[c], CAT_EMBED_DIMS[c]) for c in CAT_COLS
        })
        embed_out = sum(CAT_EMBED_DIMS[c] for c in CAT_COLS)
        layers: list[nn.Module] = []
        prev = embed_out
        for h in HIDDEN_DIMS:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.LayerNorm(h)]
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.fc_z = nn.Linear(prev, latent_dim)
        self.dec_in = nn.Linear(latent_dim, HIDDEN_DIMS[-1])
        self.decoder = nn.Sequential(
            nn.GELU(),
            nn.LayerNorm(HIDDEN_DIMS[-1]),
            nn.Linear(HIDDEN_DIMS[-1], HIDDEN_DIMS[0]),
            nn.GELU(),
            nn.Linear(HIDDEN_DIMS[0], embed_out),
        )
        self.n_geo1 = n_cat["geo_level_1_id"]
        self.n_geo2 = n_cat["geo_level_2_id"]
        self.geo1_head = nn.Linear(latent_dim, self.n_geo1)
        self.geo2_head = nn.Linear(latent_dim, self.n_geo2)

    def _embed_all(self, cat_idx: Tensor) -> Tensor:
        return torch.cat([self.embeds[c](cat_idx[:, i]) for i, c in enumerate(self.cat_cols)], dim=1)

    def encode(self, cat_idx: Tensor) -> Tensor:
        return self.fc_z(self.encoder(self._embed_all(cat_idx)))

    def forward(self, cat_idx: Tensor) -> tuple[Tensor, Tensor]:
        emb = self._embed_all(cat_idx)
        z = self.fc_z(self.encoder(emb))
        recon = self.decoder(self.dec_in(z))
        return z, recon

    def reconstruction_loss(self, cat_idx: Tensor, recon: Tensor) -> Tensor:
        target = self._embed_all(cat_idx).detach()
        return F.mse_loss(recon, target)

    def rollup_loss(self, cat_idx: Tensor, z: Tensor) -> Tensor:
        """Auxiliary: predict geo1 and geo2 indices from latent (geo3 in input)."""
        g1 = cat_idx[:, GEO1_IDX]
        g2 = cat_idx[:, GEO2_IDX]
        return F.cross_entropy(self.geo1_head(z), g1) + F.cross_entropy(self.geo2_head(z), g2)
