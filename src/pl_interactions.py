"""Foundation × geo / age interaction features for run_034 PL experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from g23_features import FOUNDATION_WEAK

PL_INTERACTION_NAMES = [
    "pl_weak_found",
    "pl_found_w",
    "pl_found_i",
    "pl_found_u",
    "pl_found_r",
    "pl_geo1_enc",
    "pl_geo3_enc",
    "pl_weak_x_geo1",
    "pl_weak_x_geo3",
    "pl_found_w_x_geo1",
    "pl_age_x_weak",
    "pl_floors_x_weak",
    "pl_height_x_weak",
    "pl_age_x_found_w",
]


def compute_pl_interactions(df: pd.DataFrame) -> np.ndarray:
    """Numeric interaction block aligned with run_033 FN segments."""
    ft = df["foundation_type"].astype(str)
    weak = ft.isin(FOUNDATION_WEAK).astype(np.float32).to_numpy()
    g1 = df["geo_level_1_id"].astype(np.float32).to_numpy()
    g3 = df["geo_level_3_id"].astype(np.float32).to_numpy()
    age = df["age"].astype(np.float32).to_numpy()
    floors = df["count_floors_pre_eq"].astype(np.float32).to_numpy()
    height = df["height_percentage"].astype(np.float32).to_numpy()

    fw = (ft == "w").astype(np.float32).to_numpy()
    fi = (ft == "i").astype(np.float32).to_numpy()
    fu = (ft == "u").astype(np.float32).to_numpy()
    fr = (ft == "r").astype(np.float32).to_numpy()

    return np.column_stack([
        weak,
        fw, fi, fu, fr,
        g1, g3,
        weak * g1,
        weak * g3,
        fw * g1,
        age * weak,
        floors * weak,
        height * weak,
        age * fw,
    ]).astype(np.float32)


def append_pl_interactions_df(df: pd.DataFrame) -> pd.DataFrame:
    arr = compute_pl_interactions(df)
    out = pd.DataFrame(arr, columns=PL_INTERACTION_NAMES, index=df.index)
    return pd.concat([df, out], axis=1)
