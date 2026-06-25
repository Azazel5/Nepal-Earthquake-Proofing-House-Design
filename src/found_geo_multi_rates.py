"""CV-safe hierarchical foundation×geo rate features at geo1 + geo2 (dense).

Rebuild of run_035's foundation×geo3 idea at granularities that GENERALIZE.

Why this exists: run_035 keyed P(grade | geo3, foundation_type) on
foundation×geo_level_3_id (~18k cells, median 6 rows/cell). 30.7% of TEST rows
fall in cells with <20 training samples, so the CV-smoothed rates were dominated
by spatial-autocorrelation leakage — CV rose to 0.7564 but public fell to 0.7523.

This module keeps the real signal (foundation modulates geo damage risk; weak
foundations drive most grade-3→grade-2 errors) but at densities that transfer:
  - foundation × geo_level_1_id : 147 cells, 0.1% test rows <20  (Laplace → global)
  - foundation × geo_level_2_id : 3.5k cells, 4.2% test rows <20  (shrinks → geo1 parent)

The geo2 cell rate uses hierarchical empirical-Bayes shrinkage toward its
foundation×geo1 PARENT rate, so a sparse geo2 cell falls back to a still-
informative parent rather than the global prior (or geo3 noise).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from embed import GRADES, RANDOM_STATE
from g23_features import FOUNDATION_WEAK

ALPHA_G1 = 5    # geo1×found cell shrinks toward global prior
ALPHA_G2 = 20   # geo2×found cell shrinks toward its geo1×found parent (cells sparser)
CV_FOLDS = 5
PREFIX1 = "foundg1"
PREFIX2 = "foundg2"

FOUND_GEO_FEATURE_COLS = [
    f"{PREFIX1}_p_grade1", f"{PREFIX1}_p_grade2", f"{PREFIX1}_p_grade3",
    f"{PREFIX2}_p_grade1", f"{PREFIX2}_p_grade2", f"{PREFIX2}_p_grade3",
    f"weak_x_{PREFIX2}_p3",
]


def _global_rates(y: np.ndarray) -> dict[int, float]:
    n = len(y)
    return {g: float((y == g).sum() / n) for g in GRADES}


def _fit_maps(ref: pd.DataFrame, y_ref: np.ndarray):
    """Build geo1×found and geo2×found rate maps from a reference (training) split.

    geo1 cells shrink toward global; geo2 cells shrink toward their geo1 parent.
    """
    gr = _global_rates(y_ref)
    found = ref["foundation_type"].astype(str).to_numpy()
    g1 = ref["geo_level_1_id"].to_numpy()
    g2 = ref["geo_level_2_id"].to_numpy()
    key1 = pd.Series(list(zip(g1, found)), index=ref.index)
    key2 = pd.Series(list(zip(g2, found)), index=ref.index)
    tot1 = key1.value_counts()
    tot2 = key2.value_counts()

    # each geo2×found cell nests in exactly one geo1×found parent
    parent_of: dict = {}
    keydf = pd.DataFrame({"k2": key2.to_numpy(), "g1": g1, "found": found})
    for k2, sub in keydf.groupby("k2", sort=False):
        parent_of[k2] = (sub["g1"].iloc[0], sub["found"].iloc[0])

    g1_maps: dict[int, dict] = {}
    g2_maps: dict[int, dict] = {}
    for g in GRADES:
        cnt1 = key1[y_ref == g].value_counts()
        g1_maps[g] = {
            k: (cnt1.get(k, 0) + ALPHA_G1 * gr[g]) / (n + ALPHA_G1)
            for k, n in tot1.items()
        }
        cnt2 = key2[y_ref == g].value_counts()
        m2 = {}
        for k2, n in tot2.items():
            parent_rate = g1_maps[g].get(parent_of[k2], gr[g])
            m2[k2] = (cnt2.get(k2, 0) + ALPHA_G2 * parent_rate) / (n + ALPHA_G2)
        g2_maps[g] = m2
    return g1_maps, g2_maps, gr


def _apply(df: pd.DataFrame, g1_maps, g2_maps, gr) -> pd.DataFrame:
    found = df["foundation_type"].astype(str).to_numpy()
    key1 = list(zip(df["geo_level_1_id"].to_numpy(), found))
    key2 = list(zip(df["geo_level_2_id"].to_numpy(), found))
    out: dict[str, list] = {}
    for g in GRADES:
        out[f"{PREFIX1}_p_grade{g}"] = [g1_maps[g].get(k, gr[g]) for k in key1]
    for g in GRADES:
        col = []
        for k2, k1 in zip(key2, key1):
            if k2 in g2_maps[g]:
                col.append(g2_maps[g][k2])
            elif k1 in g1_maps[g]:          # unseen geo2 cell → geo1 parent
                col.append(g1_maps[g][k1])
            else:                            # unseen geo1 cell → global
                col.append(gr[g])
        out[f"{PREFIX2}_p_grade{g}"] = col
    res = pd.DataFrame(out, index=df.index)
    weak = df["foundation_type"].astype(str).isin(FOUNDATION_WEAK).astype(np.float32).to_numpy()
    res[f"weak_x_{PREFIX2}_p3"] = (weak * res[f"{PREFIX2}_p_grade3"].to_numpy()).astype(np.float32)
    return res[FOUND_GEO_FEATURE_COLS].astype(np.float32)


def cv_found_geo_rates(train: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """5-fold CV: each row's rate uses only other folds (no label leakage)."""
    out = pd.DataFrame(
        np.zeros((len(train), len(FOUND_GEO_FEATURE_COLS)), dtype=np.float32),
        columns=FOUND_GEO_FEATURE_COLS, index=train.index,
    )
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for tr_idx, val_idx in skf.split(train, y):
        g1m, g2m, gr = _fit_maps(train.iloc[tr_idx], y[tr_idx])
        assigned = _apply(train.iloc[val_idx], g1m, g2m, gr)
        out.iloc[val_idx, :] = assigned.values
    return out


def fulltrain_found_geo_rates(train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Test rates: smoothed from the full training set."""
    g1m, g2m, gr = _fit_maps(train, y)
    return _apply(test, g1m, g2m, gr)


def rates_to_array(rates: pd.DataFrame) -> np.ndarray:
    return rates[FOUND_GEO_FEATURE_COLS].to_numpy(dtype=np.float32)
