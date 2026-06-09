"""
ResNet-style tabular MLP for Nepal earthquake damage prediction.

Architecture:
  - Entity embeddings for 11 categorical cols (warm-startable from run_012 checkpoint)
  - Piecewise-bin embeddings for 5 numeric features (32 bins × 8 dims = 40)
  - Direct tabular inputs (geo rates + binary flags + interactions, 51 features)
  - Input projection → hidden_dim=256 → 3 × ResBlock → 3-class head
  - geo3 embedding dropout (10%) for regularisation
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from embed import EMBED_COLS, NUMERIC_COLS

CAT_COLS: list[str] = list(EMBED_COLS.keys())
GEO3_COL = "geo_level_3_id"
GEO3_IDX = CAT_COLS.index(GEO3_COL)

NUM_BINS = 32
BIN_EMBED_DIM = 8
HIDDEN = 256
N_BLOCKS = 3
DROPOUT = 0.25
GEO3_EMBED_DROPOUT = 0.10


class NumericBins:
    """Fits quantile bin edges on training data; transforms raw values → bin indices."""

    def __init__(self, n_bins: int = NUM_BINS):
        self.n_bins = n_bins
        self.edges: list[np.ndarray] = []

    def fit(self, X_num: np.ndarray) -> "NumericBins":
        """X_num: (N, n_features) float array."""
        self.edges = []
        for j in range(X_num.shape[1]):
            col = X_num[:, j]
            finite = col[np.isfinite(col)]
            quantiles = np.linspace(0, 100, self.n_bins + 1)
            edges = np.percentile(finite, quantiles)
            # Drop duplicates at boundaries
            self.edges.append(np.unique(edges))
        return self

    def transform(self, X_num: np.ndarray) -> np.ndarray:
        """Returns (N, n_features) int64 array of bin indices in [0, n_bins-1]."""
        out = np.zeros(X_num.shape, dtype=np.int64)
        for j, edges in enumerate(self.edges):
            idx = np.searchsorted(edges, X_num[:, j], side="right") - 1
            out[:, j] = np.clip(idx, 0, self.n_bins - 1)
        return out


class ResBlock(nn.Module):
    """Pre-activation residual block: LN → Linear → GELU → Dropout → Linear → Dropout → +x."""

    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class NepalResNetMLP(nn.Module):
    """
    End-to-end ResNet MLP combining entity embeddings, numeric bin embeddings,
    and direct tabular features for 3-class damage grade prediction.
    """

    def __init__(
        self,
        n_emb: dict[str, int],
        tabular_dim: int,
        hidden: int = HIDDEN,
        n_blocks: int = N_BLOCKS,
        dropout: float = DROPOUT,
        geo3_embed_dropout: float = GEO3_EMBED_DROPOUT,
    ):
        super().__init__()
        self.geo3_embed_dropout = geo3_embed_dropout

        # Entity embeddings — same vocab sizes as NepalEmbeddingNet (warm-startable)
        self.embeddings = nn.ModuleDict(
            {col: nn.Embedding(n_emb[col], dim) for col, dim in EMBED_COLS.items()}
        )
        embed_out_dim = sum(EMBED_COLS.values())  # 140

        # Numeric bin embeddings
        self.bin_embeds = nn.ModuleList(
            [nn.Embedding(NUM_BINS, BIN_EMBED_DIM) for _ in NUMERIC_COLS]
        )
        bin_out_dim = len(NUMERIC_COLS) * BIN_EMBED_DIM  # 40

        total_in = embed_out_dim + bin_out_dim + tabular_dim

        # MLP body
        self.input_proj = nn.Linear(total_in, hidden)
        self.input_norm = nn.LayerNorm(hidden)
        self.blocks = nn.ModuleList(
            [ResBlock(hidden, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(hidden, 3)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        for emb in self.embeddings.values():
            nn.init.normal_(emb.weight, std=0.01)
        for be in self.bin_embeds:
            nn.init.normal_(be.weight, std=0.01)

    def forward(self, cat_matrix: Tensor, num_bins: Tensor, tabular: Tensor) -> Tensor:
        """
        cat_matrix : (B, 11) int64  — categorical indices per EMBED_COLS order
        num_bins   : (B, 5)  int64  — quantile bin indices per NUMERIC_COLS order
        tabular    : (B, D)  float  — direct tabular features (geo rates, flags, etc.)
        Returns    : (B, 3)  float  — raw logits
        """
        embeds: list[Tensor] = []
        for i, col in enumerate(CAT_COLS):
            e = self.embeddings[col](cat_matrix[:, i])
            if col == GEO3_COL and self.training and self.geo3_embed_dropout > 0:
                # Drop entire geo3 embedding vector for random rows (embedding dropout)
                mask = torch.bernoulli(
                    torch.full(
                        (e.size(0), 1), 1.0 - self.geo3_embed_dropout, device=e.device
                    )
                )
                e = e * mask
            embeds.append(e)

        bin_embs: list[Tensor] = [
            self.bin_embeds[i](num_bins[:, i]) for i in range(len(NUMERIC_COLS))
        ]

        x = torch.cat(embeds + bin_embs + [tabular], dim=1)
        x = F.gelu(self.input_norm(self.input_proj(x)))
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def warm_start_embeddings(
    model: NepalResNetMLP,
    ckpt_path: str | "Path",
    device: torch.device,
) -> NepalResNetMLP:
    """Load entity embedding weights from a NepalEmbeddingNet checkpoint."""
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    embed_state = {k: v for k, v in state.items() if k.startswith("embeddings.")}
    missing, unexpected = model.load_state_dict(embed_state, strict=False)
    loaded = len(embed_state)
    print(f"  Warm-start: loaded {loaded} embedding tensors "
          f"({len(missing)} missing, {len(unexpected)} unexpected)")
    return model
