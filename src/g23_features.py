"""G2 vs G3 specialist features — foundation, geo, superstructure signal."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from embed import NUMERIC_COLS, apply_age_sentinel, load_raw_data
from preprocess import binary_columns

FOUNDATION_WEAK = frozenset({"w", "i", "u"})
FOUNDATION_STRONG = frozenset({"r", "h"})
WEAK_SUPER_COLS = [
    "has_superstructure_adobe_mud",
    "has_superstructure_mud_mortar_stone",
    "has_superstructure_bamboo",
]
STRONG_SUPER_COLS = [
    "has_superstructure_rc_engineered",
    "has_superstructure_cement_mortar_brick",
    "has_superstructure_cement_mortar_stone",
]
RC_COLS = [
    "has_superstructure_rc_engineered",
    "has_superstructure_rc_non_engineered",
]
CAT_COLS = [
    "foundation_type",
    "roof_type",
    "land_surface_condition",
    "ground_floor_type",
    "plan_configuration",
    "position",
]
GEO_COLS = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]
RATE_COLS = ["geo_level_1_id", "geo_level_3_id", "foundation_type"]
LAPLACE = 5.0


def load_g23_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = load_raw_data()
    train, test = apply_age_sentinel(train, test)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def _laplace_g3_rate(keys: pd.Series, is_g3: np.ndarray, alpha: float = LAPLACE) -> dict:
    """P(G3 | group) among G2/G3 rows with Laplace smoothing."""
    global_mean = float(is_g3.mean()) if len(is_g3) else 0.5
    out: dict = {}
    for k in keys.unique():
        m = keys == k
        n = int(m.sum())
        s = int(is_g3[m].sum())
        out[k] = (s + alpha * global_mean) / (n + alpha)
    return out


class G23FeatureBuilder:
    """OOF-safe feature matrix for G2 vs G3 binary specialist."""

    def __init__(self):
        self.ohe: OneHotEncoder | None = None
        self.bin_cols: list[str] = []
        self.rate_maps: dict[str, dict] = {}
        self.geo1_weak_map: dict = {}

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "G23FeatureBuilder":
        g23 = np.isin(y, [2, 3])
        sub = df.loc[g23]
        sub_y = (y[g23] == 3).astype(np.int8)

        self.bin_cols = binary_columns(df)
        self.ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
        self.ohe.fit(sub[CAT_COLS].astype(str))

        self.rate_maps = {}
        for col in RATE_COLS:
            self.rate_maps[col] = _laplace_g3_rate(sub[col].astype(str), sub_y)

        # geo1 × foundation_weak collapse rate
        weak_flag = sub["foundation_type"].astype(str).isin(FOUNDATION_WEAK).astype(int)
        combo = sub["geo_level_1_id"].astype(str) + "_" + weak_flag.astype(str)
        self.geo1_weak_map = _laplace_g3_rate(combo, sub_y)
        return self

    def _engineered(self, df: pd.DataFrame) -> np.ndarray:
        ft = df["foundation_type"].astype(str)
        weak_f = ft.isin(FOUNDATION_WEAK).astype(np.float32).to_numpy()
        strong_f = ft.isin(FOUNDATION_STRONG).astype(np.float32).to_numpy()

        weak_mat = df[WEAK_SUPER_COLS].max(axis=1).astype(np.float32).to_numpy() if all(
            c in df.columns for c in WEAK_SUPER_COLS
        ) else np.zeros(len(df), dtype=np.float32)
        strong_mat = df[STRONG_SUPER_COLS].max(axis=1).astype(np.float32).to_numpy() if all(
            c in df.columns for c in STRONG_SUPER_COLS
        ) else np.zeros(len(df), dtype=np.float32)
        has_rc = df[RC_COLS].max(axis=1).astype(np.float32).to_numpy() if all(
            c in df.columns for c in RC_COLS
        ) else np.zeros(len(df), dtype=np.float32)

        super_cols = [c for c in df.columns if c.startswith("has_superstructure_")]
        n_super = df[super_cols].sum(axis=1).astype(np.float32).to_numpy() if super_cols else np.zeros(len(df))

        age = df["age"].astype(np.float32).to_numpy()
        height = df["height_percentage"].astype(np.float32).to_numpy()
        floors = df["count_floors_pre_eq"].astype(np.float32).to_numpy()
        area = df["area_percentage"].astype(np.float32).to_numpy()

        global_g3 = 0.33
        g1_rate = df["geo_level_1_id"].astype(str).map(self.rate_maps.get("geo_level_1_id", {})).fillna(global_g3).to_numpy(dtype=np.float32)
        g3_rate = df["geo_level_3_id"].astype(str).map(self.rate_maps.get("geo_level_3_id", {})).fillna(global_g3).to_numpy(dtype=np.float32)
        f_rate = ft.map(self.rate_maps.get("foundation_type", {})).fillna(global_g3).to_numpy(dtype=np.float32)
        geo1 = df["geo_level_1_id"].astype(str).to_numpy()
        combo = geo1 + "_" + weak_f.astype(int).astype(str)
        geo_weak_rate = pd.Series(combo).map(self.geo1_weak_map).fillna(global_g3).to_numpy(dtype=np.float32)

        eng = np.column_stack([
            weak_f,
            strong_f,
            weak_mat,
            strong_mat,
            has_rc,
            strong_mat - weak_mat,
            n_super,
            age * height,
            floors * height,
            area * floors,
            weak_f * g1_rate,
            weak_mat * g3_rate,
            g1_rate,
            g3_rate,
            f_rate,
            geo_weak_rate,
        ]).astype(np.float32)
        return eng

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        assert self.ohe is not None
        cat_ohe = self.ohe.transform(df[CAT_COLS].astype(str))
        numeric = df[NUMERIC_COLS].to_numpy(dtype=np.float32)
        binary = df[self.bin_cols].to_numpy(dtype=np.float32)
        geo_raw = df[GEO_COLS].to_numpy(dtype=np.float32)
        eng = self._engineered(df)
        return np.hstack([cat_ohe, geo_raw, numeric, binary, eng]).astype(np.float32)
