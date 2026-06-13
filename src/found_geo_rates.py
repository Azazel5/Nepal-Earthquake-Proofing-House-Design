"""CV-safe Laplace-smoothed P(grade | geo3, foundation_type) rate features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from embed import GRADES, RANDOM_STATE
from g23_features import FOUNDATION_WEAK

ALPHA = 5  # sparser than pure-geo cells — stronger shrinkage toward global prior
CV_FOLDS = 5
GEO_COL = "geo_level_3_id"
PREFIX = "foundg3"

FOUND_GEO_FEATURE_COLS = [
    f"{PREFIX}_p_grade1",
    f"{PREFIX}_p_grade2",
    f"{PREFIX}_p_grade3",
    "weak_x_foundg3_p3",
]


def _global_rates(df: pd.DataFrame, y: np.ndarray) -> dict[int, float]:
    n = len(y)
    return {g: float((y == g).sum() / n) for g in GRADES}


def _cross_rate_maps(
    ref: pd.DataFrame,
    y_ref: np.ndarray,
    cat: pd.Series,
    global_rates: dict[int, float],
) -> dict[int, dict]:
    key = pd.Series(list(zip(ref[GEO_COL].values, cat.values)), index=ref.index)
    totals = key.value_counts()
    maps: dict[int, dict] = {}
    for g in GRADES:
        cnt = key[y_ref == g].value_counts()
        maps[g] = {
            k: (cnt.get(k, 0) + ALPHA * global_rates[g]) / (n + ALPHA)
            for k, n in totals.items()
        }
    return maps


def _apply_cross_rates(
    df: pd.DataFrame,
    cat: pd.Series,
    maps: dict[int, dict],
    global_rates: dict[int, float],
) -> pd.DataFrame:
    key = list(zip(df[GEO_COL].values, cat.values))
    out = {
        f"{PREFIX}_p_grade{g}": [maps[g].get(k, global_rates[g]) for k in key]
        for g in GRADES
    }
    return pd.DataFrame(out, index=df.index)


def _weak_interaction(df: pd.DataFrame, rates: pd.DataFrame) -> pd.ndarray:
    weak = df["foundation_type"].astype(str).isin(FOUNDATION_WEAK).astype(np.float32).to_numpy()
    return (weak * rates[f"{PREFIX}_p_grade3"].to_numpy()).astype(np.float32)


def cv_found_geo_rates(train: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """5-fold CV: each row's rate uses only other folds (no label leakage)."""
    global_rates = _global_rates(train, y)
    cat = train["foundation_type"].astype(str)
    cols = [f"{PREFIX}_p_grade{g}" for g in GRADES]
    out = pd.DataFrame(
        {c: global_rates[int(c.rsplit("grade", 1)[-1])] for c in cols},
        index=train.index,
        dtype=np.float32,
    )
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for tr_idx, val_idx in skf.split(train, y):
        maps = _cross_rate_maps(
            train.iloc[tr_idx], y[tr_idx], cat.iloc[tr_idx], global_rates,
        )
        assigned = _apply_cross_rates(train.iloc[val_idx], cat.iloc[val_idx], maps, global_rates)
        for c in cols:
            out.iloc[val_idx, out.columns.get_loc(c)] = assigned[c].values
    out["weak_x_foundg3_p3"] = _weak_interaction(train, out)
    return out[FOUND_GEO_FEATURE_COLS]


def fulltrain_found_geo_rates(train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Test rates: smoothed from full training set."""
    global_rates = _global_rates(train, y)
    cat_tr = train["foundation_type"].astype(str)
    cat_te = test["foundation_type"].astype(str)
    maps = _cross_rate_maps(train, y, cat_tr, global_rates)
    rates = _apply_cross_rates(test, cat_te, maps, global_rates)
    rates["weak_x_foundg3_p3"] = _weak_interaction(test, rates)
    return rates[FOUND_GEO_FEATURE_COLS].astype(np.float32)


def rates_to_array(rates: pd.DataFrame) -> np.ndarray:
    return rates[FOUND_GEO_FEATURE_COLS].to_numpy(dtype=np.float32)
