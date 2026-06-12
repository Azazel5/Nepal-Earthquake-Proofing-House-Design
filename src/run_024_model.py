"""Shoumik-style geo representation models (run_024 replication)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

GEO_LATENT_DR = 32
GEO_LATENT_ROLLUP = 16
GEO1_EMBED = 16
GEO2_EMBED = 128
GEO3_EMBED = 128


class ShoumikDREncoder(nn.Module):
    def __init__(self, n_geo1: int, n_geo2: int, n_geo3: int, latent_dim: int = GEO_LATENT_DR):
        super().__init__()
        self.emb1 = nn.Embedding(n_geo1, GEO1_EMBED)
        self.emb2 = nn.Embedding(n_geo2, GEO2_EMBED)
        self.emb3 = nn.Embedding(n_geo3, GEO3_EMBED)
        self.compressor = nn.Linear(GEO1_EMBED + GEO2_EMBED + GEO3_EMBED, latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        e1 = self.emb1(x[:, 0])
        e2 = self.emb2(x[:, 1])
        e3 = self.emb3(x[:, 2])
        h = torch.cat([e1, e2, e3], dim=1)
        return self.compressor(torch.relu(h))


class ShoumikDRDecoder(nn.Module):
    def __init__(self, n_geo1: int, n_geo2: int, n_geo3: int, latent_dim: int = GEO_LATENT_DR):
        super().__init__()
        self.p1 = nn.Linear(latent_dim, n_geo1)
        self.p2 = nn.Linear(latent_dim, n_geo2)
        self.p3 = nn.Linear(latent_dim, n_geo3)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.p1(z), self.p2(z), self.p3(z)


class ShoumikDRAutoEncoder(nn.Module):
    def __init__(self, n_geo1: int, n_geo2: int, n_geo3: int, latent_dim: int = GEO_LATENT_DR):
        super().__init__()
        self.encoder = ShoumikDREncoder(n_geo1, n_geo2, n_geo3, latent_dim)
        self.decoder = ShoumikDRDecoder(n_geo1, n_geo2, n_geo3, latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        z = torch.relu(self.encoder(x))
        p1, p2, p3 = self.decoder(z)
        return z, p1, p2, p3


class RollUpGeo3Encoder(nn.Module):
    def __init__(self, n_geo3: int, latent_dim: int = GEO_LATENT_ROLLUP, embed_dim: int = 128):
        super().__init__()
        self.emb3 = nn.Embedding(n_geo3, embed_dim)
        self.compressor = nn.Linear(embed_dim, latent_dim)

    def forward(self, geo3: Tensor) -> Tensor:
        x = self.emb3(geo3.squeeze(-1) if geo3.dim() > 1 else geo3)
        return self.compressor(torch.relu(x))


class RollUpGeo3AutoEncoder(nn.Module):
    def __init__(self, n_geo1: int, n_geo2: int, n_geo3: int, latent_dim: int = GEO_LATENT_ROLLUP):
        super().__init__()
        self.encoder = RollUpGeo3Encoder(n_geo3, latent_dim)
        self.p1 = nn.Linear(latent_dim, n_geo1)
        self.p2 = nn.Linear(latent_dim, n_geo2)

    def forward(self, geo3: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        z = torch.relu(self.encoder(geo3))
        return z, self.p1(z), self.p2(z)
