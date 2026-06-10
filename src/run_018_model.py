"""
Architecture for run_018 — "MLP rebuild as a quality play on the decorrelated
architecture" (see project memory: run_013 is the most decorrelated model so
far at 0.887 loss-corr / 9.3% disagreement vs LGBM run_012, but too weak
solo at 0.7427).

Components:
  - HierGeoEmbedding: hierarchical residual geo embeddings
        emb(geo2) = proj(emb(geo1)) + Delta2(geo2)
        emb(geo3) = emb(geo2)       + Delta3(geo3)
    Delta tables zero-initialized (rare geo collapses to parent).
    Delta2/Delta3 are dropped per-sample during training (embedding dropout)
    to simulate rare/unseen geo at test time.
  - PiecewiseLinearEncoder (numpy): quantile-bin piecewise-linear encoding for
    numeric + geo-rate features (fit on train, applied to train/test).
  - PLEEmbedding (nn.Module): per-feature Linear+ReLU on the PLE encoding.
  - NepalRun018MLP: geo embeddings + low-card categorical embeddings +
    PLE features + raw binary flags -> input projection -> N x ResBlock -> head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mlp_model import ResBlock

GEO_DIMS = (16, 32, 32)  # (geo1 table dim, Delta2/proj-out dim, Delta3 dim)
DELTA2_DROPOUT = 0.05
DELTA3_DROPOUT = 0.15
PLE_BINS = 48
PLE_EMBED_DIM = 8
HIDDEN = 448
N_BLOCKS = 3
DROPOUT = 0.3


# ── Hierarchical residual geo embedding ──────────────────────────────────────

class HierGeoEmbedding(nn.Module):
    def __init__(
        self,
        n_geo1: int,
        n_geo2: int,
        n_geo3: int,
        dim1: int = GEO_DIMS[0],
        dim2: int = GEO_DIMS[1],
        dim3: int = GEO_DIMS[2],
        delta2_dropout: float = DELTA2_DROPOUT,
        delta3_dropout: float = DELTA3_DROPOUT,
    ):
        super().__init__()
        assert dim3 == dim2, "Delta3 must match Delta2/proj output dim for residual add"
        self.geo1_emb = nn.Embedding(n_geo1, dim1)
        self.proj_1to2 = nn.Linear(dim1, dim2)
        self.delta2 = nn.Embedding(n_geo2, dim2)
        self.delta3 = nn.Embedding(n_geo3, dim3)
        self.delta2_dropout = delta2_dropout
        self.delta3_dropout = delta3_dropout
        self.out_dim = dim1 + dim2 + dim3

        nn.init.normal_(self.geo1_emb.weight, std=0.01)
        nn.init.xavier_uniform_(self.proj_1to2.weight)
        nn.init.zeros_(self.proj_1to2.bias)
        nn.init.zeros_(self.delta2.weight)
        nn.init.zeros_(self.delta3.weight)

    def forward(self, geo1_idx: Tensor, geo2_idx: Tensor, geo3_idx: Tensor) -> Tensor:
        e1 = self.geo1_emb(geo1_idx)
        proj1 = self.proj_1to2(e1)

        d2 = self.delta2(geo2_idx)
        if self.training and self.delta2_dropout > 0:
            mask = (torch.rand(d2.size(0), 1, device=d2.device) >= self.delta2_dropout).float()
            d2 = d2 * mask
        e2 = proj1 + d2

        d3 = self.delta3(geo3_idx)
        if self.training and self.delta3_dropout > 0:
            mask = (torch.rand(d3.size(0), 1, device=d3.device) >= self.delta3_dropout).float()
            d3 = d3 * mask
        e3 = e2 + d3

        return torch.cat([e1, e2, e3], dim=1)


# ── Piecewise-linear encoding for numeric / geo-rate features ───────────────

class PiecewiseLinearEncoder:
    """Fits per-feature quantile bin edges on train; transforms to (N, F, n_bins)."""

    def __init__(self, n_bins: int = PLE_BINS):
        self.n_bins = n_bins
        self.edges: np.ndarray | None = None  # (F, n_bins+1)

    def fit(self, X: np.ndarray) -> "PiecewiseLinearEncoder":
        n_features = X.shape[1]
        edges = np.zeros((n_features, self.n_bins + 1), dtype=np.float64)
        qs = np.linspace(0, 100, self.n_bins + 1)
        for j in range(n_features):
            e = np.percentile(X[:, j], qs)
            for i in range(1, len(e)):
                if e[i] <= e[i - 1]:
                    e[i] = e[i - 1] + 1e-6
            edges[j] = e
        self.edges = edges
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.edges is not None
        N, n_features = X.shape
        out = np.zeros((N, n_features, self.n_bins), dtype=np.float32)
        arange = np.arange(self.n_bins)
        for j in range(n_features):
            e = self.edges[j]
            x = X[:, j]
            k = np.searchsorted(e, x, side="right") - 1
            k = np.clip(k, 0, self.n_bins - 1)
            denom = e[k + 1] - e[k]
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = (x - e[k]) / denom
            # rare FP edge cases at the max-value bin boundary can yield nan/inf;
            # those rows sit at the top edge of their bin, so frac should be 1.0
            frac = np.nan_to_num(frac, nan=1.0, posinf=1.0, neginf=0.0)
            frac = np.clip(frac, 0.0, 1.0)
            less = (arange[None, :] < k[:, None]).astype(np.float32)
            equal = (arange[None, :] == k[:, None]).astype(np.float32) * frac[:, None].astype(np.float32)
            out[:, j, :] = less + equal
        return out


class PLEEmbedding(nn.Module):
    """Per-feature Linear(n_bins -> embed_dim) + ReLU on piecewise-linear encodings."""

    def __init__(self, n_features: int, n_bins: int = PLE_BINS, embed_dim: int = PLE_EMBED_DIM):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, n_bins, embed_dim))
        self.bias = nn.Parameter(torch.zeros(n_features, embed_dim))
        nn.init.normal_(self.weight, std=0.01)
        self.out_dim = n_features * embed_dim

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, F, n_bins) -> (B, F, embed_dim)
        out = torch.einsum("bfk,fkd->bfd", x, self.weight) + self.bias
        return F.relu(out).flatten(1)


# ── Full model ────────────────────────────────────────────────────────────────

class NepalRun018MLP(nn.Module):
    def __init__(
        self,
        n_geo1: int,
        n_geo2: int,
        n_geo3: int,
        n_cat: dict[str, int],
        cat_dims: dict[str, int],
        n_ple_features: int,
        n_binary: int,
        ple_bins: int = PLE_BINS,
        ple_embed_dim: int = PLE_EMBED_DIM,
        hidden: int = HIDDEN,
        n_blocks: int = N_BLOCKS,
        dropout: float = DROPOUT,
        geo_dims: tuple[int, int, int] = GEO_DIMS,
        delta2_dropout: float = DELTA2_DROPOUT,
        delta3_dropout: float = DELTA3_DROPOUT,
    ):
        super().__init__()
        self.geo_embed = HierGeoEmbedding(
            n_geo1, n_geo2, n_geo3, *geo_dims,
            delta2_dropout=delta2_dropout, delta3_dropout=delta3_dropout,
        )
        self.cat_cols = list(cat_dims.keys())
        self.cat_embeds = nn.ModuleDict(
            {c: nn.Embedding(n_cat[c], cat_dims[c]) for c in self.cat_cols}
        )
        self.ple = PLEEmbedding(n_ple_features, ple_bins, ple_embed_dim)

        total_in = self.geo_embed.out_dim + sum(cat_dims.values()) + self.ple.out_dim + n_binary
        self.input_proj = nn.Linear(total_in, hidden)
        self.input_norm = nn.LayerNorm(hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, 3)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        for emb in self.cat_embeds.values():
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, geo_idx: Tensor, cat_idx: Tensor, ple: Tensor, binary: Tensor) -> Tensor:
        geo_feat = self.geo_embed(geo_idx[:, 0], geo_idx[:, 1], geo_idx[:, 2])
        cat_feats = [self.cat_embeds[c](cat_idx[:, i]) for i, c in enumerate(self.cat_cols)]
        ple_feat = self.ple(ple)
        x = torch.cat([geo_feat, *cat_feats, ple_feat, binary], dim=1)
        x = F.gelu(self.input_norm(self.input_proj(x)))
        for block in self.blocks:
            x = block(x)
        return self.head(x)
